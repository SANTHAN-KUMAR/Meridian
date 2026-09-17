// ggml_matmul_bench.cpp — per-op throughput of ONE matmul shape on ONE ggml device, plus the cost of
// getting its weights into that device's buffer.
//
// WHY THIS EXISTS. End-to-end decode rate cannot answer "would this device compute this matmul faster",
// because the ops we can move are a small share of a token (results/2026-09-17/compute_trace.json,
// claim trace_mul_mat_id_share and the node table) and the effect lands inside the run-to-run spread of
// the end-to-end rate: the A/B is underpowered by construction (CLAUDE.md 4.1). The device question is
// answered directly instead -- one shape, one device, many iterations -- and the answer then predicts
// what any placement decision can buy before a patch is written for it.
//
// TWO NUMBERS PER SHAPE, and the second is the one that decides expert placement:
//   compute   ms and GB/s for the matmul itself, weights already resident in the device's buffer.
//   upload    ms and GB/s for ggml_backend_tensor_set of those weights. For a streaming engine this is
//             not a setup cost: a cache miss pays it per token. ggml-hexagon REPACKS Q4_0/Q4_1/Q8_0/
//             IQ4_NL/MXFP4 weights on set_tensor (ggml_hexagon_is_repack_type -> repack_tensor_tiled),
//             so the upload is not a memcpy and this is where an expert-streaming port would pay.
//
// The matmul is measured with GGML_OP_MUL_MAT (one row of activations, i.e. decode) and, with
// --ids N, with GGML_OP_MUL_MAT_ID over N of the experts in a 3-D expert tensor, which is the op that
// actually dominates decode.
//
// Correctness before speed: every configuration also runs on the CPU backend and the two results are
// compared elementwise. A device whose output does not match the CPU's within tolerance is reported as
// WRONG and its timings are not printed -- a fast wrong kernel is not a result.
//
// Usage (in-process via the app shim, because the DSP session only opens in an app):
//   matmul_bench --device HTP0 --k 2048 --n 4096 --type q4_0 --iters 50
//   matmul_bench --device HTP0 --k 2048 --n 768 --experts 128 --ids 8 --type q4_0 --iters 20
// Output is one `RESULT ` line of key=value pairs per configuration, for a parser to read.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

struct Cfg {
    std::string device = "CPU";
    std::string type   = "q4_0";
    int64_t k = 2048, n = 4096, experts = 0, ids = 0;
    int iters = 50, warmup = 5, threads = 4;
    // Independent copies of the weight, rotated one per iteration. With ONE copy a timing loop re-reads
    // the same bytes and the caches serve them, which reports a compute-bound rate for what is in
    // reality a DRAM-bound op: a decode touches each weight once per token. Enough copies to exceed the
    // last-level cache makes the measurement DRAM-bound, like the real thing.
    int copies = 1;
};

static ggml_type type_of(const std::string & s) {
    if (s == "q4_0")   return GGML_TYPE_Q4_0;
    if (s == "q4_1")   return GGML_TYPE_Q4_1;
    if (s == "q8_0")   return GGML_TYPE_Q8_0;
    if (s == "f16")    return GGML_TYPE_F16;
    if (s == "mxfp4")  return GGML_TYPE_MXFP4;
    if (s == "iq4_nl") return GGML_TYPE_IQ4_NL;
    fprintf(stderr, "unknown type %s\n", s.c_str());
    exit(2);
}

// Quantised bytes for a weight, from ggml's own block sizes: never from an assumed bits-per-weight.
static size_t weight_bytes(ggml_type t, int64_t k, int64_t n, int64_t e) {
    return ggml_row_size(t, k) * (size_t) n * (size_t) (e ? e : 1);
}

struct Run {
    bool ok = false;
    double compute_ms = 0, upload_ms = 0;
    std::vector<float> out;
};

// One backend, one shape: build the graph, upload the weights (timed), compute it `iters` times (timed).
static Run run_one(ggml_backend_t be, const Cfg & c, ggml_type wt,
                   const std::vector<uint8_t> & wdata, const std::vector<float> & act,
                   const std::vector<int32_t> & idsv) {
    Run r;
    const int64_t rows_out = c.ids ? c.ids : 1;
    const int copies = c.copies > 0 ? c.copies : 1;

    ggml_init_params ip = { /*mem_size*/ ggml_tensor_overhead() * (size_t) (8 * copies + 8)
                                         + ggml_graph_overhead() * (size_t) copies,
                            /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return r; }

    // Decode shape, exactly as llama.cpp builds it: ONE activation row broadcast to the selected
    // experts (b = [k,1,1], ids = [n_used,1]), which ggml asserts with ids->ne[0] % b->ne[1] == 0.
    ggml_tensor * a = c.ids ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.k, 1, 1)
                            : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.k, 1);
    ggml_set_name(a, "act");
    ggml_tensor * ids = nullptr;
    if (c.ids) {
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, rows_out, 1);
        ggml_set_name(ids, "ids");
    }
    std::vector<ggml_tensor *> ws;
    std::vector<ggml_cgraph *> gfs;
    std::vector<ggml_tensor *> dsts;
    for (int j = 0; j < copies; j++) {
        ggml_tensor * w = c.experts ? ggml_new_tensor_3d(ctx, wt, c.k, c.n, c.experts)
                                    : ggml_new_tensor_2d(ctx, wt, c.k, c.n);
        ggml_set_name(w, ("weight" + std::to_string(j)).c_str());
        ggml_tensor * dst = c.ids ? ggml_mul_mat_id(ctx, w, a, ids) : ggml_mul_mat(ctx, w, a);
        ggml_set_name(dst, ("dst" + std::to_string(j)).c_str());
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, dst);
        ws.push_back(w);
        gfs.push_back(gf);
        dsts.push_back(dst);
    }

    // The weights go in a buffer marked WEIGHTS, because that is the usage under which ggml-hexagon
    // decides to repack; measuring an upload into a non-weight buffer would measure the wrong thing.
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!wbuf) { fprintf(stderr, "buffer alloc failed (%s)\n", ggml_backend_name(be)); ggml_free(ctx); return r; }
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Upload cost is per copy and averaged: it is what a streaming engine pays on every cache miss,
    // and on ggml-hexagon it includes the tiled repack, not just a memcpy.
    auto t0 = clk::now();
    for (int j = 0; j < copies; j++) ggml_backend_tensor_set(ws[j], wdata.data(), 0, ggml_nbytes(ws[j]));
    ggml_backend_synchronize(be);
    r.upload_ms = ms_since(t0) / copies;

    ggml_backend_tensor_set(a, act.data(), 0, ggml_nbytes(a));
    if (ids) ggml_backend_tensor_set(ids, idsv.data(), 0, ggml_nbytes(ids));

    for (int i = 0; i < c.warmup; i++) {
        if (ggml_backend_graph_compute(be, gfs[i % copies]) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "compute failed (%s)\n", ggml_backend_name(be));
            ggml_backend_buffer_free(wbuf); ggml_free(ctx); return r;
        }
    }
    ggml_backend_synchronize(be);
    t0 = clk::now();
    for (int i = 0; i < c.iters; i++) ggml_backend_graph_compute(be, gfs[i % copies]);
    ggml_backend_synchronize(be);
    r.compute_ms = ms_since(t0) / c.iters;

    // Read back the LAST computed copy, and note that every copy holds identical bytes, so all copies
    // must produce the same output: that is asserted here rather than assumed.
    r.out.resize(ggml_nelements(dsts[0]));
    ggml_backend_tensor_get(dsts[(c.iters - 1) % copies], r.out.data(), 0, ggml_nbytes(dsts[0]));
    std::vector<float> other(r.out.size());
    for (int j = 0; j < copies; j++) {
        ggml_backend_graph_compute(be, gfs[j]);
        ggml_backend_synchronize(be);
        ggml_backend_tensor_get(dsts[j], other.data(), 0, ggml_nbytes(dsts[j]));
        for (size_t i = 0; i < other.size(); i++) {
            if (other[i] != r.out[i]) {
                fprintf(stderr, "copy %d disagrees with copy %d at element %zu (%g vs %g): the copies do "
                                "not hold identical weights, so the rotation is not a fair repeat\n",
                        j, (int) ((c.iters - 1) % copies), i, other[i], r.out[i]);
                ggml_backend_buffer_free(wbuf); ggml_free(ctx); return r;
            }
        }
    }
    r.ok = true;
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx);
    return r;
}

extern "C" int matmul_bench_main(int argc, char ** argv) {
    Cfg c;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", what); exit(2); }
            return argv[++i];
        };
        if      (s == "--device")  c.device  = next("--device");
        else if (s == "--type")    c.type    = next("--type");
        else if (s == "--k")       c.k       = atoll(next("--k"));
        else if (s == "--n")       c.n       = atoll(next("--n"));
        else if (s == "--experts") c.experts = atoll(next("--experts"));
        else if (s == "--ids")     c.ids     = atoll(next("--ids"));
        else if (s == "--iters")   c.iters   = atoi(next("--iters"));
        else if (s == "--warmup")  c.warmup  = atoi(next("--warmup"));
        else if (s == "--threads") c.threads = atoi(next("--threads"));
        else if (s == "--copies")  c.copies  = atoi(next("--copies"));
        else { fprintf(stderr, "unknown arg %s\n", s.c_str()); return 2; }
    }
    if (c.ids && !c.experts) { fprintf(stderr, "--ids needs --experts\n"); return 2; }

    const ggml_type wt = type_of(c.type);
    const size_t wbytes = weight_bytes(wt, c.k, c.n, c.experts);
    // Only the selected experts' bytes are read by a MUL_MAT_ID, and that is the quantity a decode
    // pays; the full tensor's bytes are what an upload pays. Both are reported.
    const size_t read_bytes = c.ids ? weight_bytes(wt, c.k, c.n, c.ids) : wbytes;

    // The weight bytes must be a VALID quantisation, not random bytes: a random 16-bit block scale is
    // frequently a NaN or an Inf, every product becomes NaN, and both backends then "agree" on garbage.
    // (The copy-agreement check above is what caught this.) So: random floats, quantised by ggml itself.
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> uf(-1.f, 1.f);
    const int64_t wrows = c.n * (c.experts ? c.experts : 1);
    std::vector<float> wf((size_t) c.k * (size_t) wrows);
    for (auto & x : wf) x = uf(rng);
    std::vector<uint8_t> wdata(wbytes);
    const size_t q = ggml_quantize_chunk(wt, wf.data(), wdata.data(), 0, wrows, c.k, nullptr);
    if (q != wbytes) {
        fprintf(stderr, "ggml_quantize_chunk wrote %zu bytes, expected %zu\n", q, wbytes);
        return 1;
    }
    std::vector<float> act((size_t) c.k);
    for (auto & x : act) x = uf(rng);
    std::vector<int32_t> idsv;
    for (int64_t i = 0; i < c.ids; i++) idsv.push_back((int32_t) ((i * 7 + 3) % c.experts));

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, c.threads);
    Run ref = run_one(cpu, c, wt, wdata, act, idsv);
    if (!ref.ok) { fprintf(stderr, "CPU reference failed\n"); return 1; }

    ggml_backend_dev_t dev = ggml_backend_dev_by_name(c.device.c_str());
    if (!dev) {
        fprintf(stderr, "no ggml device named %s; devices:", c.device.c_str());
        for (size_t i = 0; i < ggml_backend_dev_count(); i++)
            fprintf(stderr, " %s", ggml_backend_dev_name(ggml_backend_dev_get(i)));
        fprintf(stderr, "\n");
        return 1;
    }
    Run got = ref;
    bool same_as_cpu = true;
    if (c.device != "CPU") {
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (!be) { fprintf(stderr, "%s init failed (no session)\n", c.device.c_str()); return 1; }
        got = run_one(be, c, wt, wdata, act, idsv);
        ggml_backend_free(be);
        if (!got.ok) return 1;
        same_as_cpu = false;
    }

    // Elementwise agreement with the CPU. The tolerance is on the relative error of a dot product of
    // length k over random inputs; anything beyond it is a different kernel, not rounding.
    double max_rel = 0.0;
    if (!same_as_cpu) {
        double scale = 0.0;
        for (float v : ref.out) scale = std::max(scale, (double) std::fabs(v));
        if (scale == 0.0) scale = 1.0;
        for (size_t i = 0; i < ref.out.size() && i < got.out.size(); i++)
            max_rel = std::max(max_rel, std::fabs((double) got.out[i] - ref.out[i]) / scale);
    }
    const bool correct = same_as_cpu || max_rel < 5e-2;

    printf("RESULT device=%s type=%s k=%lld n=%lld experts=%lld ids=%lld iters=%d threads=%d copies=%d "
           "weight_MB=%.3f read_MB=%.3f compute_ms=%.4f compute_GBps=%.3f upload_ms=%.4f upload_GBps=%.3f "
           "cpu_compute_ms=%.4f cpu_compute_GBps=%.3f max_rel_err=%.3e correct=%d\n",
           c.device.c_str(), c.type.c_str(), (long long) c.k, (long long) c.n, (long long) c.experts,
           (long long) c.ids, c.iters, c.threads, c.copies,
           wbytes / 1e6, read_bytes / 1e6,
           got.compute_ms, read_bytes / 1e9 / (got.compute_ms / 1e3),
           got.upload_ms, wbytes / 1e9 / (got.upload_ms / 1e3),
           ref.compute_ms, read_bytes / 1e9 / (ref.compute_ms / 1e3),
           max_rel, correct ? 1 : 0);
    if (!correct) {
        fprintf(stderr, "WRONG: %s disagrees with the CPU backend (max relative error %.3e) — "
                        "its timings above are not usable\n", c.device.c_str(), max_rel);
        return 3;
    }
    ggml_backend_free(cpu);
    return 0;
}

#ifndef MATMUL_BENCH_NO_MAIN
int main(int argc, char ** argv) { return matmul_bench_main(argc, argv); }
#endif
