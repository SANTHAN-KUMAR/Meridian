// ggml_ref.cpp — the CPU reference for gx: runs one case through ggml-cpu's own MoE FFN graph, the ops
// llama.cpp's build_moe_ffn emits for Qwen3-MoE (gate/up MUL_MAT_ID, ggml_swiglu_split, down MUL_MAT_ID;
// the expert weighting and sum come after and are not part of the FFN gx replaces).
// Built twice by build.sh: aarch64 static (the phone's ggml, armv8.6-a+dotprod+i8mm+fp16, run under
// qemu-aarch64-static on the laptop: the bit-exact reference) and x86 (the laptop's ggml: a different
// accumulation order, reported for scale only).
//   ggml_ref <case.bin> <out.bin> [threads]
//   ggml_ref --quant <blocks.f32> <out_q8_0.bin> <out_q8_1.bin>   ggml-cpu's own from_float for Q8_0 / Q8_1
//   ggml_ref --swiglu <pairs.f32> <out.f32>   ggml_swiglu_split over N gates then N ups (N a multiple of 768)
#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

static bool rd(FILE * f, void * p, size_t n) { return fread(p, 1, n, f) == n; }

static int quant_mode(int argc, char ** argv) {
    if (argc != 5) { fprintf(stderr, "usage: ggml_ref --quant blocks.f32 out_q8_0.bin out_q8_1.bin\n"); return 2; }
    FILE * f = fopen(argv[2], "rb");
    if (!f) { perror(argv[2]); return 1; }
    std::vector<float> v;
    float b[4096];
    size_t n;
    while ((n = fread(b, 4, 4096, f)) > 0) v.insert(v.end(), b, b + n);
    fclose(f);
    if (v.empty() || v.size() % 32) { fprintf(stderr, "need a multiple of 32 floats\n"); return 1; }
    ggml_cpu_init();
    for (int t = 0; t < 2; t++) {
        const ggml_type ty = t ? GGML_TYPE_Q8_1 : GGML_TYPE_Q8_0;
        std::vector<uint8_t> q(ggml_row_size(ty, (int64_t) v.size()));
        ggml_get_type_traits_cpu(ty)->from_float(v.data(), q.data(), (int64_t) v.size());   // the arch (NEON) quantizer
        FILE * o = fopen(argv[3 + t], "wb");
        if (!o) { perror(argv[3 + t]); return 1; }
        fwrite(q.data(), 1, q.size(), o);
        fclose(o);
    }
    return 0;
}

static int swiglu_mode(int argc, char ** argv) {
    if (argc != 4) { fprintf(stderr, "usage: ggml_ref --swiglu pairs.f32 out.f32\n"); return 2; }
    FILE * f = fopen(argv[2], "rb");
    if (!f) { perror(argv[2]); return 1; }
    std::vector<float> v;
    float b[4096];
    size_t n;
    while ((n = fread(b, 4, 4096, f)) > 0) v.insert(v.end(), b, b + n);
    fclose(f);
    const int64_t N = (int64_t) v.size() / 2, row = 768;   // rows of 768 like ffn_moe_gate: the vector path covers all
    if (N == 0 || N % row) { fprintf(stderr, "need 2*N floats, N a multiple of 768\n"); return 1; }
    ggml_init_params ip = {(size_t) N * 4 * 3 + 16u * 1024 * 1024, nullptr, false};
    ggml_context * c = ggml_init(ip);
    ggml_tensor * g = ggml_new_tensor_2d(c, GGML_TYPE_F32, row, N / row);
    ggml_tensor * u = ggml_new_tensor_2d(c, GGML_TYPE_F32, row, N / row);
    memcpy(g->data, v.data(), N * 4);
    memcpy(u->data, v.data() + N, N * 4);
    ggml_tensor * h = ggml_swiglu_split(c, g, u);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, h);
    if (ggml_graph_compute_with_ctx(c, gf, 2) != GGML_STATUS_SUCCESS) return 1;
    FILE * o = fopen(argv[3], "wb");
    if (!o) { perror(argv[3]); return 1; }
    fwrite(h->data, 4, N, o);
    fclose(o);
    ggml_free(c);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1 && !strcmp(argv[1], "--quant")) return quant_mode(argc, argv);
    if (argc > 1 && !strcmp(argv[1], "--swiglu")) return swiglu_mode(argc, argv);
    if (argc < 3) { fprintf(stderr, "usage: ggml_ref case.bin out.bin [threads]\n"); return 2; }
    const int nth = argc > 3 ? atoi(argv[3]) : 2;
    FILE * f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    char magic[4];
    int32_t hdr[4];
    if (!rd(f, magic, 4) || memcmp(magic, "GXC2", 4) || !rd(f, hdr, sizeof hdr)) { fprintf(stderr, "bad case\n"); return 1; }
    const int ne = hdr[0], nf = hdr[1], k = hdr[2], dq41 = hdr[3];   // hdr[3]: down type, 0 = Q4_0, 1 = Q4_1
    std::vector<int32_t> ids(k);
    std::vector<float> x(ne);
    const size_t gu = (size_t) nf * (ne / 32) * 18, dn = (size_t) ne * (nf / 32) * (dq41 ? 20 : 18);
    std::vector<uint8_t> wg(gu * k), wu(gu * k), wd(dn * k);
    bool ok = rd(f, ids.data(), 4 * k) && rd(f, x.data(), 4 * ne);
    for (int e = 0; e < k && ok; e++)
        ok = rd(f, wg.data() + e * gu, gu) && rd(f, wu.data() + e * gu, gu) && rd(f, wd.data() + e * dn, dn);
    fclose(f);
    if (!ok) { fprintf(stderr, "short case file\n"); return 1; }

    ggml_init_params ip = {(size_t) 3 * k * (gu + dn) + 64u * 1024 * 1024, nullptr, false};
    ggml_context * c = ggml_init(ip);
    ggml_tensor * tg = ggml_new_tensor_3d(c, GGML_TYPE_Q4_0, ne, nf, k);
    ggml_tensor * tu = ggml_new_tensor_3d(c, GGML_TYPE_Q4_0, ne, nf, k);
    ggml_tensor * td = ggml_new_tensor_3d(c, dq41 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q4_0, nf, ne, k);
    ggml_set_name(tg, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(tu, "blk.0.ffn_up_exps.weight");
    ggml_set_name(td, "blk.0.ffn_down_exps.weight");
    if (ggml_nbytes(tg) != wg.size() || ggml_nbytes(td) != wd.size() || ggml_nbytes(tu) != wu.size()) { fprintf(stderr, "layout mismatch\n"); return 1; }
    memcpy(tg->data, wg.data(), wg.size());
    memcpy(tu->data, wu.data(), wu.size());
    memcpy(td->data, wd.data(), wd.size());
    ggml_tensor * tx = ggml_new_tensor_3d(c, GGML_TYPE_F32, ne, 1, 1);   // [n_embd, 1, n_tokens], as build_moe_ffn
    memcpy(tx->data, x.data(), 4 * ne);
    ggml_tensor * tid = ggml_new_tensor_2d(c, GGML_TYPE_I32, k, 1);
    memcpy(tid->data, ids.data(), 4 * k);

    ggml_tensor * gate = ggml_mul_mat_id(c, tg, tx, tid);   // [n_ff, k, 1]
    ggml_tensor * up = ggml_mul_mat_id(c, tu, tx, tid);
    ggml_tensor * h = ggml_swiglu_split(c, gate, up);       // silu(gate) * up
    ggml_tensor * down = ggml_mul_mat_id(c, td, h, tid);    // [n_embd, k, 1]
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, gate);
    ggml_build_forward_expand(gf, up);
    ggml_build_forward_expand(gf, down);
    if (ggml_graph_compute_with_ctx(c, gf, nth) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }

    FILE * o = fopen(argv[2], "wb");
    if (!o) { perror(argv[2]); return 1; }
    const int32_t oh[3] = {k, nf, ne};
    fwrite("GXR1", 1, 4, o);
    fwrite(oh, 4, 3, o);
    fwrite(gate->data, 4, (size_t) k * nf, o);
    fwrite(up->data, 4, (size_t) k * nf, o);
    fwrite(h->data, 4, (size_t) k * nf, o);
    fwrite(down->data, 4, (size_t) k * ne, o);
    fclose(o);
    ggml_free(c);
    return 0;
}
