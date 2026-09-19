// repack_bench.cpp — is ggml-cpu's repacked (interleaved i8mm/dotprod) MUL_MAT_ID kernel faster than the generic Q4_0
// kernel on Qwen3-30B-A3B's expert shapes, on the phone, with the engine's own compute placement?
//
// Why (design doc R4): the only in-engine measurement of --repack-experts (2026-09-17, n=2, mismatched budgets) found it
// slower, but it repacked on the I/O path and skipped the fork's hooks, so it cannot separate kernel speed from repack
// cost. This bench times the kernel alone: the SAME Q4_0 bytes, (A) generic, (B) bound with ggml_cpu_repack_bind_tensor and
// repacked once per expert with ggml_cpu_repack_slice (the fork's API, the one the engine uses), outside the timed loop.
// One MUL_MAT_ID per iteration: 8 experts selected of 8 stored, one token, 4 threads pinned to cpu4-7 (strict).
// Shapes: gate/up (k=2048 -> 768 rows), down (k=768 -> 2048 rows), Q4_0 (the type of 42 of 48 down layers, all gate/up).
// PRE-REGISTERED (2026-09-19 08:50, before the first run):
//   ratio = median(B) / median(A) per shape, 300 iterations each, A and B interleaved in blocks of 25.
//   The repacked kernel is worth building into the engine iff the ratio is <= 0.8 on BOTH shapes. It is dead iff the
//   ratio is >= 0.95 on both. Outputs: the max |A-B| per shape is reported. The repacked kernels use a different
//   accumulation order, so the engine's text-identity gate decides losslessness separately.
//   repack_bench [iters=300]
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

struct Case {
    ggml_context * ctx = nullptr;
    ggml_tensor * w = nullptr, * x = nullptr, * ids = nullptr, * out = nullptr;
    ggml_cgraph * gf = nullptr;
};

static Case make_case(int k, int rows, int n_as, const std::vector<char> & wbytes, const std::vector<float> & xin) {
    Case c;
    const size_t wsz = wbytes.size();
    ggml_init_params ip = { wsz + (size_t) 64 * 1024 * 1024, nullptr, false };
    c.ctx = ggml_init(ip);
    c.w = ggml_new_tensor_3d(c.ctx, GGML_TYPE_Q4_0, k, rows, n_as);
    std::memcpy(c.w->data, wbytes.data(), wsz);
    c.x = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, k, n_as, 1);
    for (int e = 0; e < n_as; ++e) std::memcpy((float *) c.x->data + (size_t) e * k, xin.data(), (size_t) k * sizeof(float));
    c.ids = ggml_new_tensor_2d(c.ctx, GGML_TYPE_I32, n_as, 1);
    for (int e = 0; e < n_as; ++e) ((int32_t *) c.ids->data)[e] = e;
    c.out = ggml_mul_mat_id(c.ctx, c.w, c.x, c.ids);
    c.gf = ggml_new_graph(c.ctx);
    ggml_build_forward_expand(c.gf, c.out);
    return c;
}

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 300;
    ggml_threadpool_params tpp = ggml_threadpool_params_default(4);
    for (int i = 0; i < GGML_MAX_N_THREADS; ++i) tpp.cpumask[i] = (i >= 4 && i < 8);
    tpp.strict_cpu = true;
    ggml_threadpool * tp = ggml_threadpool_new(&tpp);
    if (!tp) { std::printf("FATAL threadpool\n"); return 1; }

    const int n_as = 8;
    struct Shape { const char * name; int k, rows; } shapes[] = {{"gate_up", 2048, 768}, {"down", 768, 2048}};
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 0.02f), nx(0.f, 1.f);
    bool all_ok = true;
    double ratios[2] = {0, 0};
    for (int si = 0; si < 2; ++si) {
        const Shape & s = shapes[si];
        std::vector<float> wf((size_t) s.k * s.rows * n_as);
        for (float & v : wf) v = nd(rng);
        const size_t row_sz = ggml_row_size(GGML_TYPE_Q4_0, s.k);
        std::vector<char> wq(row_sz * (size_t) s.rows * n_as);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, wf.data(), wq.data(), 0, (int64_t) s.rows * n_as, s.k, nullptr);
        std::vector<float> xin(s.k);
        for (float & v : xin) v = nx(rng);

        Case A = make_case(s.k, s.rows, n_as, wq, xin);
        Case B = make_case(s.k, s.rows, n_as, wq, xin);
        if (!ggml_cpu_repack_bind_tensor(B.w)) { std::printf("RESULT shape=%s status=no_repacked_form\n", s.name); all_ok = false; continue; }
        const size_t slice = row_sz * (size_t) s.rows;
        for (int e = 0; e < n_as; ++e)
            if (!ggml_cpu_repack_slice(B.w, (char *) B.w->data + (size_t) e * slice, slice)) {
                std::printf("RESULT shape=%s status=repack_slice_failed expert=%d\n", s.name, e); all_ok = false;
            }
        ggml_cplan pa = ggml_graph_plan(A.gf, 4, tp), pb = ggml_graph_plan(B.gf, 4, tp);
        std::vector<uint8_t> wa(pa.work_size + 1), wb(pb.work_size + 1);
        pa.work_data = wa.data(); pb.work_data = wb.data();
        // warm-up, then the output comparison
        for (int i = 0; i < 20; ++i) { ggml_graph_compute(A.gf, &pa); ggml_graph_compute(B.gf, &pb); }
        double maxd = 0, maxa = 0;
        const float * oa = (const float *) A.out->data, * ob = (const float *) B.out->data;
        for (int64_t i = 0; i < ggml_nelements(A.out); ++i) { maxd = std::max(maxd, (double) std::fabs(oa[i] - ob[i])); maxa = std::max(maxa, (double) std::fabs(oa[i])); }
        std::vector<double> ta, tb;
        for (int blk = 0; blk < iters / 25; ++blk) {
            for (int i = 0; i < 25; ++i) { const double t0 = now_s(); ggml_graph_compute(A.gf, &pa); ta.push_back(now_s() - t0); }
            for (int i = 0; i < 25; ++i) { const double t0 = now_s(); ggml_graph_compute(B.gf, &pb); tb.push_back(now_s() - t0); }
        }
        const double ma = median(ta) * 1e3, mb = median(tb) * 1e3;
        const double gb = (double) wq.size() / 1e9;
        ratios[si] = mb / ma;
        std::printf("RESULT shape=%s k=%d rows=%d experts=%d generic_ms=%.4f repacked_ms=%.4f ratio=%.3f generic_GBps=%.1f repacked_GBps=%.1f max_abs_diff=%.3g max_abs_out=%.3g\n",
                    s.name, s.k, s.rows, n_as, ma, mb, mb / ma, gb / (ma / 1e3), gb / (mb / 1e3), maxd, maxa);
        ggml_free(A.ctx); ggml_free(B.ctx);
    }
    const bool build = all_ok && ratios[0] <= 0.8 && ratios[1] <= 0.8;
    const bool dead = all_ok && ratios[0] >= 0.95 && ratios[1] >= 0.95;
    std::printf("VERDICT %s (ratios gate_up %.3f down %.3f; build iff both <= 0.8, dead iff both >= 0.95)\n",
                !all_ok ? "INCOMPLETE" : build ? "BUILD" : dead ? "DEAD" : "UNDECIDED", ratios[0], ratios[1]);
    ggml_threadpool_free(tp);
    return 0;
}
