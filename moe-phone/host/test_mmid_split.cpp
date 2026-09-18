// test_mmid_split.cpp — milestone M1 of the GPU expert path (research/2026-09-18_gpu_expert_path_design.md §7).
//
// Proves the ggml-cpu MUL_MAT_ID split hooks (ggml_cpu_set_mmid_split_hooks) are plumbed correctly, with an
// expected value fixed by construction: a CPU STAND-IN takes the "external device" role and computes the
// skipped experts with ggml-cpu's own routines (the type's from_float into vec_dot_type, then vec_dot per
// output row, exactly what mul_mat_id's one_chunk does on a 1-row-per-call target). So with the hooks on, dst
// must be BIT-IDENTICAL to the unhooked op for every skip set: any difference is a plumbing bug (wrong row
// mapping, a skipped expert computed anyway, a row left unfilled, a race).
//
// Shapes are Qwen3-30B-A3B's: gate/up Q4_0 [2048 -> 768] with one activation row broadcast to the 8 selected
// experts, and down Q4_1 [768 -> 2048] with one activation row per slot. 128 experts. Random weights.
//
// Caveat recorded, not hidden: on an ARM build with i8mm, q4_0's vec_dot takes TWO rows per call (nrows = 2),
// so the stand-in must mirror that there; this x86 test covers the 1-row path. The real GPU kernel is NOT
// expected to be bit-identical (design §6); this test is about the plumbing only.
//
// Build (host):   g++ -O2 -std=c++17 test_mmid_split.cpp -I$E/third_party/llama.cpp/ggml/include \
//                     -L$E/build-host/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$E/build-host/bin
// Run:            ./test_mmid_split [trials] [threads]
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// MMID_BREAK negative-control mode, read in main (see on_end)
static int g_break = 0;
// Work check: ggml-cpu calls the expert READY hook once per thread for every expert it computes, so a skipped
// expert must produce zero calls. Output alone cannot see a skip that is ignored (the CPU computes the same
// bits the stand-in then writes); this counter can.
#include <atomic>
static std::atomic<long long> g_ready_calls{0};
static void on_ready(const ggml_tensor *, int, void *) { g_ready_calls.fetch_add(1, std::memory_order_relaxed); }

struct Standin {
    int k = 0;                          // experts to hand to the "device" per op
    std::mt19937 * rng = nullptr;
    std::vector<unsigned char> taken;   // which experts were skipped this op
    std::vector<std::vector<float>> rows; // computed rows for (slot) -> ne01 floats
    std::vector<int> row_slot;          // slot index for each computed row
    long long ops = 0, skipped_total = 0, rows_filled = 0;
};

// Compute one expert's output rows for every slot routed to it, with ggml-cpu's own routines.
static void standin_compute(Standin & S, const ggml_tensor * dst, int expert) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    const auto * tr0 = ggml_get_type_traits_cpu(src0->type);
    const ggml_type vt = tr0->vec_dot_type;
    const auto * trv = ggml_get_type_traits_cpu(vt);
    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1];
    const int64_t n_ids = ids->ne[0];
    std::vector<char> q(ggml_row_size(vt, ne00));
    for (int64_t id = 0; id < n_ids; ++id) {
        const int32_t e = *(const int32_t *) ((const char *) ids->data + id * ids->nb[0]);
        if (e != expert) continue;
        // src1 row for this slot: broadcast row 0 when src1 has one row (gate/up), else row `id` (down)
        const int64_t i11 = (src1->ne[1] == 1) ? 0 : id;
        const float * x = (const float *) ((const char *) src1->data + i11 * src1->nb[1]);
        trv->from_float(x, q.data(), ne00);
        std::vector<float> out(ne01);
        const char * w = (const char *) src0->data + (size_t) expert * src0->nb[2];
        for (int64_t r = 0; r < ne01; ++r) tr0->vec_dot((int) ne00, &out[r], 0, w + r * src0->nb[1], 0, q.data(), 0, 1);
        S.rows.push_back(std::move(out));
        S.row_slot.push_back((int) id);
    }
}

static bool on_begin(const ggml_tensor * dst, unsigned char * skip, int n_as, void * ud) {
    Standin & S = *(Standin *) ud;
    ++S.ops;
    S.rows.clear(); S.row_slot.clear();
    const ggml_tensor * ids = dst->src[2];
    const int n_ids = (int) ids->ne[0];
    std::vector<int> sel;
    for (int i = 0; i < n_ids; ++i) sel.push_back(*(const int32_t *) ((const char *) ids->data + i * ids->nb[0]));
    std::shuffle(sel.begin(), sel.end(), *S.rng);
    int n = 0;
    for (int e : sel) {
        if (n >= S.k) break;
        if (e < 0 || e >= n_as || skip[e]) continue;
        if (g_break != 3) skip[e] = 1;
        ++n;
        standin_compute(S, dst, e);
    }
    // make the skipped slots' dst rows safe to read until end fills them (design §4 step 3)
    for (int slot : S.row_slot) std::memset((char *) dst->data + slot * dst->nb[1], 0, dst->ne[0] * sizeof(float));
    S.skipped_total += n;
    return n > 0;
}

// MMID_BREAK (negative controls: the test must FAIL under each, or it proves nothing):
//   1 = drop the last filled row, 2 = write each row into the NEXT slot, 3 = compute but do not mark skipped
static void on_end(const ggml_tensor * dst, void * ud) {
    Standin & S = *(Standin *) ud;
    const int n_used = (int) dst->ne[1];
    for (size_t i = 0; i < S.rows.size(); ++i) {
        if (g_break == 1 && i + 1 == S.rows.size()) continue;
        const int slot = g_break == 2 ? (S.row_slot[i] + 1) % n_used : S.row_slot[i];
        std::memcpy((char *) dst->data + slot * dst->nb[1], S.rows[i].data(), dst->ne[0] * sizeof(float));
        ++S.rows_filled;
    }
}

static void fill_quant(ggml_tensor * w, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.f, 0.02f);
    const int64_t per = w->ne[0] * w->ne[1];
    std::vector<float> f(per);
    for (int64_t e = 0; e < w->ne[2]; ++e) {
        for (auto & v : f) v = nd(rng);
        ggml_quantize_chunk(w->type, f.data(), (char *) w->data + e * w->nb[2], 0, w->ne[1], w->ne[0], nullptr);
    }
}

int main(int argc, char ** argv) {
    const int trials = argc > 1 ? atoi(argv[1]) : 1000;
    const int nthreads = argc > 2 ? atoi(argv[2]) : 4;
    if (const char * b = std::getenv("MMID_BREAK")) g_break = std::atoi(b);
    const int n_as = 128, n_used = 8, n_embd = 2048, n_ff = 768;
    ggml_init_params ip = { (size_t) 1024 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    std::mt19937 rng(12345);
    ggml_tensor * wg = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, n_embd, n_ff, n_as);   // gate/up shape
    ggml_tensor * wd = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_1, n_ff, n_embd, n_as);   // down shape
    fill_quant(wg, rng); fill_quant(wd, rng);
    ggml_tensor * x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, 1);
    ggml_tensor * xd = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff, n_used, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);
    ggml_tensor * og = ggml_mul_mat_id(ctx, wg, x, ids);
    ggml_tensor * od = ggml_mul_mat_id(ctx, wd, xd, ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, og);
    ggml_build_forward_expand(gf, od);

    Standin S; S.rng = &rng;
    std::normal_distribution<float> nx(0.f, 1.f);
    std::vector<float> ref_g(ggml_nelements(og)), ref_d(ggml_nelements(od));
    int failures = 0;
    long long compared = 0;
    for (int t = 0; t < trials; ++t) {
        // random inputs and a random distinct expert selection
        for (int i = 0; i < n_embd; ++i) ((float *) x->data)[i] = nx(rng);
        for (int i = 0; i < n_ff * n_used; ++i) ((float *) xd->data)[i] = nx(rng);
        std::vector<int> perm(n_as); for (int i = 0; i < n_as; ++i) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int i = 0; i < n_used; ++i) ((int32_t *) ids->data)[i] = perm[i];
        // reference: no hooks
        ggml_cpu_set_mmid_split_hooks(nullptr, nullptr, nullptr);
        ggml_graph_compute_with_ctx(ctx, gf, nthreads);
        std::memcpy(ref_g.data(), og->data, ref_g.size() * sizeof(float));
        std::memcpy(ref_d.data(), od->data, ref_d.size() * sizeof(float));
        // poison dst so an unfilled row cannot pass by coincidence
        std::memset(og->data, 0x7f, ggml_nbytes(og)); std::memset(od->data, 0x7f, ggml_nbytes(od));
        // hooked: k cycles through 0..8
        S.k = t % (n_used + 1);
        const long long skipped_before = S.skipped_total;
        g_ready_calls.store(0);
        ggml_cpu_set_expert_ready_hook(on_ready, nullptr);
        ggml_cpu_set_mmid_split_hooks(on_begin, on_end, &S);
        ggml_graph_compute_with_ctx(ctx, gf, nthreads);
        ggml_cpu_set_mmid_split_hooks(nullptr, nullptr, nullptr);
        ggml_cpu_set_expert_ready_hook(nullptr, nullptr);
        // two ops (gate, down), each over n_used distinct experts, minus what each op skipped
        const long long expect_ready = (long long) nthreads * (2LL * n_used - (S.skipped_total - skipped_before));
        const bool okw = g_ready_calls.load() == expect_ready;
        const bool okg = std::memcmp(ref_g.data(), og->data, ref_g.size() * sizeof(float)) == 0;
        const bool okd = std::memcmp(ref_d.data(), od->data, ref_d.size() * sizeof(float)) == 0;
        compared += 2;
        if (!okg || !okd || !okw) {
            if (failures < 5) std::fprintf(stderr, "trial %d k=%d: gate %s down %s work %s (ready %lld expected %lld)\n", t, S.k,
                                           okg ? "ok" : "DIFF", okd ? "ok" : "DIFF", okw ? "ok" : "WRONG", g_ready_calls.load(), expect_ready);
            ++failures;
        }
    }
    std::printf("{\"trials\": %d, \"threads\": %d, \"ops_compared\": %lld, \"failures\": %d, \"hook_ops\": %lld, "
                "\"experts_skipped\": %lld, \"rows_filled\": %lld}\n",
                trials, nthreads, compared, failures, S.ops, S.skipped_total, S.rows_filled);
    ggml_free(ctx);
    return failures == 0 ? 0 : 1;
}
