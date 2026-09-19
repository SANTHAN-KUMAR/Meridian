// gx_test.cpp — acceptance test for gx (M3). For each case produced by make_cases.py:
//   - runs gx on the first OpenCL GPU, weights in one CL_MEM_ALLOC_HOST_PTR buffer, slots addressed by offset
//     (slot s -> expert ids[s], so the id mapping is tested too);
//   - compares the down output and the SwiGLU output bit for bit with ggml-cpu's ARM result (ggml_ref built
//     for aarch64, run under qemu: <case>.arm.out), and reports the error against ggml-cpu x86
//     (<case>.x86.out) and against an fp64 evaluation of the same dequantized weights and unquantized x.
// Also checks the kernels' correctly rounded division against IEEE division on 2^24 random pairs plus the
// operand ranges the kernels use.
//   gx_test <case_dir> [out.json]
#include "gx.h"

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

static const char * k_src =
#ifndef GX_KERNELS_INC
#define GX_KERNELS_INC "gx_kernels.inc"   /* negative_controls.sh builds against mutated copies */
#endif
#include GX_KERNELS_INC
    ;

static const char * k_test_src = R"CL(
kernel void t_q8(global const float * in, global uchar * out, int with_s) {   // local size 1
    local float v[32];
    for (int i = 0; i < 32; i++) v[i] = in[get_group_id(0) * 32 + i];
    q8_block(v, out + get_group_id(0) * 36, with_s);
}
kernel void t_swiglu(global const float * g, global const float * u, global float * h, global int * risk) {
    const size_t i = get_global_id(0);
    int r = 0;
    h[i] = gx_swiglu_r(g[i], u[i], &r);
    risk[i] = r;
}
kernel void t_div(global const float * a, global const float * b, global float * q) {
    const size_t i = get_global_id(0);
    q[i] = cr_div(a[i], b[i]);
}
)CL";

struct Case {
    std::string name;
    int ne = 0, nf = 0, k = 0, dt = 0;
    std::vector<int32_t> ids;
    std::vector<float> x;
    std::vector<uint8_t> w;   // expert e at e * (2*gu + dn): gate, up, down
};

static bool load_case(const std::string & path, Case & c) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char m[4];
    int32_t h[4];
    bool ok = fread(m, 1, 4, f) == 4 && !memcmp(m, "GXC2", 4) && fread(h, 4, 4, f) == 4;
    if (ok) {
        c.ne = h[0]; c.nf = h[1]; c.k = h[2]; c.dt = h[3];
        c.ids.resize(c.k);
        c.x.resize(c.ne);
        const size_t per = 2 * (size_t) c.nf * (c.ne / 32) * 18 + (size_t) c.ne * (c.nf / 32) * (c.dt ? 20 : 18);
        c.w.resize(per * c.k);
        ok = fread(c.ids.data(), 4, c.k, f) == (size_t) c.k && fread(c.x.data(), 4, c.ne, f) == (size_t) c.ne &&
             fread(c.w.data(), 1, c.w.size(), f) == c.w.size();
    }
    fclose(f);
    return ok;
}

struct Ref { bool ok = false; std::vector<float> gate, up, h, down; };
static Ref load_ref(const std::string & path, int k, int nf, int ne) {
    Ref r;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return r;
    char m[4];
    int32_t h[3];
    if (fread(m, 1, 4, f) == 4 && !memcmp(m, "GXR1", 4) && fread(h, 4, 3, f) == 3 && h[0] == k && h[1] == nf && h[2] == ne) {
        r.gate.resize((size_t) k * nf); r.up.resize((size_t) k * nf); r.h.resize((size_t) k * nf); r.down.resize((size_t) k * ne);
        r.ok = fread(r.gate.data(), 4, r.gate.size(), f) == r.gate.size() && fread(r.up.data(), 4, r.up.size(), f) == r.up.size() &&
               fread(r.h.data(), 4, r.h.size(), f) == r.h.size() && fread(r.down.data(), 4, r.down.size(), f) == r.down.size();
    }
    fclose(f);
    return r;
}

static float h2f(const uint8_t * p) {
    uint16_t v;
    memcpy(&v, p, 2);
    const uint32_t s = (v >> 15) & 1, e = (v >> 10) & 31, m = v & 1023;
    double r = e == 0 ? ldexp((double) m, -24) : e == 31 ? (m ? NAN : INFINITY) : ldexp((double) (m | 1024), (int) e - 25);
    return (float) (s ? -r : r);
}

// fp64 evaluation of the FFN on the dequantized weights and the unquantized x (the "ideal" the quantized
// paths both approximate). Output: k * ne.
static std::vector<double> fp64_ffn(const Case & c) {
    const int ne = c.ne, nf = c.nf;
    const size_t qb = c.dt ? 20 : 18, gu = (size_t) nf * (ne / 32) * 18, per = 2 * gu + (size_t) ne * (nf / 32) * qb;
    std::vector<double> out((size_t) c.k * ne), h(nf);
    for (int s = 0; s < c.k; s++) {
        const uint8_t * E = c.w.data() + per * c.ids[s];
        for (int r = 0; r < nf; r++) {
            double g = 0, u = 0;
            for (int b = 0; b < ne / 32; b++) {
                const uint8_t * G = E + ((size_t) r * (ne / 32) + b) * 18, * U = E + gu + ((size_t) r * (ne / 32) + b) * 18;
                const double dg = h2f(G), du = h2f(U);
                for (int i = 0; i < 16; i++) {
                    g += dg * (((G[2 + i] & 15) - 8) * (double) c.x[b * 32 + i] + ((G[2 + i] >> 4) - 8) * (double) c.x[b * 32 + 16 + i]);
                    u += du * (((U[2 + i] & 15) - 8) * (double) c.x[b * 32 + i] + ((U[2 + i] >> 4) - 8) * (double) c.x[b * 32 + 16 + i]);
                }
            }
            h[r] = g / (1 + exp(-g)) * u;
        }
        for (int r = 0; r < ne; r++) {
            double a = 0;
            for (int b = 0; b < nf / 32; b++) {
                const uint8_t * D = E + 2 * gu + ((size_t) r * (nf / 32) + b) * qb;
                const double d = h2f(D), m = c.dt ? h2f(D + 2) : 0.0;
                const uint8_t * Q = D + (c.dt ? 4 : 2);
                const int off = c.dt ? 0 : 8;
                for (int i = 0; i < 16; i++)
                    a += (d * ((Q[i] & 15) - off) + m) * h[b * 32 + i] + (d * ((Q[i] >> 4) - off) + m) * h[b * 32 + 16 + i];
            }
            out[(size_t) s * ne + r] = a;
        }
    }
    return out;
}

static size_t diff_count(const float * a, const float * b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; i++) d += memcmp(a + i, b + i, 4) != 0;
    return d;
}

static int64_t ulp_dist(float a, float b) {
    int32_t ia, ib;
    memcpy(&ia, &a, 4); memcpy(&ib, &b, 4);
    if (ia < 0) ia = INT32_MIN - ia;
    if (ib < 0) ib = INT32_MIN - ib;
    return llabs((int64_t) ia - ib);
}

struct Cmp { size_t n = 0, diff_bits = 0; int64_t max_ulp = 0; double max_rel = 0; };
// per-slot max|a-b| / max|b|, worst slot; bitwise count over all
template <class T> static Cmp compare(const std::vector<float> & a, const std::vector<T> & b, int k, int n) {
    Cmp r;
    r.n = (size_t) k * n;
    for (int s = 0; s < k; s++) {
        double num = 0, den = 0;
        for (int i = 0; i < n; i++) {
            const size_t j = (size_t) s * n + i;
            num = std::max(num, fabs((double) a[j] - (double) b[j]));
            den = std::max(den, fabs((double) b[j]));
            if (std::is_same<T, float>::value) {
                float bf = (float) b[j];
                if (memcmp(&a[j], &bf, 4)) { r.diff_bits++; r.max_ulp = std::max(r.max_ulp, ulp_dist(a[j], bf)); }
            }
        }
        r.max_rel = std::max(r.max_rel, den > 0 ? num / den : num);
    }
    return r;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: gx_test <case_dir> [out.json]\n"); return 2; }
    const std::string dir = argv[1];
    cl_platform_id plats[8];
    cl_uint np = 0;
    clGetPlatformIDs(8, plats, &np);
    cl_device_id dev = nullptr;
    for (cl_uint i = 0; i < np && !dev; i++)
        if (clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 1, &dev, nullptr) != CL_SUCCESS) dev = nullptr;
    if (!dev) { fprintf(stderr, "no OpenCL GPU\n"); return 1; }
    char name[256] = {0}, ver[256] = {0};
    cl_device_fp_config fpc = 0;
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof name, name, nullptr);
    clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof ver, ver, nullptr);
    clGetDeviceInfo(dev, CL_DEVICE_SINGLE_FP_CONFIG, sizeof fpc, &fpc, nullptr);
    cl_int e;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &e);
    printf("DEVICE name=\"%s\" driver=\"%s\" fp32_denorm=%d fp32_fma=%d fp32_cr_divide=%d\n", name, ver,
           !!(fpc & CL_FP_DENORM), !!(fpc & CL_FP_FMA), !!(fpc & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT));

    // ---- correctly rounded division vs IEEE (the host's '/' is IEEE fp32 division)
    size_t div_n = 0, div_bad = 0;
    {
        const size_t N = 1u << 24;
        std::vector<float> a(N), b(N), r(N);
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> lg(-30.f, 30.f), un(0.f, 1.f);
        for (size_t i = 0; i < N; i++) {
            switch (i % 4) {
                case 0: a[i] = ldexpf(1 + un(rng), (int) lg(rng)) * (rng() & 1 ? -1 : 1); b[i] = ldexpf(1 + un(rng), (int) lg(rng)); break;
                case 1: a[i] = ldexpf(un(rng), -(int) (un(rng) * 20)); b[i] = 127.0f; break;                       // amax / 127
                case 2: a[i] = 1.0f; b[i] = ldexpf(1 + un(rng), -(int) (un(rng) * 30)) / 127.0f; break;           // 1 / d
                default: a[i] = (un(rng) - 0.5f) * 40; b[i] = 1.0f + expf(-a[i]); break;                           // silu
            }
        }
        const std::string src = std::string("#pragma OPENCL FP_CONTRACT OFF\n") + k_src + k_test_src;
        const char * s = src.c_str();
        cl_program p = clCreateProgramWithSource(ctx, 1, &s, nullptr, &e);
        e = clBuildProgram(p, 1, &dev, "-cl-std=CL1.2 -DN_EMBD=2048 -DN_FF=768", nullptr, nullptr);
        if (e != CL_SUCCESS) {
            static char log[1 << 16];
            clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, sizeof log - 1, log, nullptr);
            fprintf(stderr, "test program build failed %d:\n%s\n", e, log);
            return 1;
        }
        cl_kernel kd = clCreateKernel(p, "t_div", &e);
        cl_mem ba = clCreateBuffer(ctx, CL_MEM_COPY_HOST_PTR, N * 4, a.data(), &e);
        cl_mem bb = clCreateBuffer(ctx, CL_MEM_COPY_HOST_PTR, N * 4, b.data(), &e);
        cl_mem bq = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, N * 4, nullptr, &e);
        clSetKernelArg(kd, 0, sizeof ba, &ba); clSetKernelArg(kd, 1, sizeof bb, &bb); clSetKernelArg(kd, 2, sizeof bq, &bq);
        clEnqueueNDRangeKernel(q, kd, 1, nullptr, &N, nullptr, 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, bq, CL_TRUE, 0, N * 4, r.data(), 0, nullptr, nullptr);
        for (size_t i = 0; i < N; i++) {
            volatile float ref = a[i] / b[i];
            const float rf = ref;
            div_n++;
            if (memcmp(&rf, &r[i], 4)) div_bad++;
        }
        printf("DIV n=%zu mismatches=%zu\n", div_n, div_bad);
        clReleaseMemObject(ba); clReleaseMemObject(bb); clReleaseMemObject(bq); clReleaseKernel(kd); clReleaseProgram(p);
    }

    // ---- the quantizers on crafted near-tie blocks, vs ggml-cpu ARM's own from_float (ggml_ref --quant)
    long q8_blocks = -1, q8_0_bad = 0, q8_1_bad = 0, host_q8_0_bad = 0;
    {
        FILE * fb = fopen((dir + "/quant_blocks.f32").c_str(), "rb");
        FILE * f0 = fopen((dir + "/quant_blocks.f32.q8_0.arm").c_str(), "rb");
        FILE * f1 = fopen((dir + "/quant_blocks.f32.q8_1.arm").c_str(), "rb");
        if (fb && f0 && f1) {
            std::vector<float> v;
            float b[4096];
            size_t n;
            while ((n = fread(b, 4, 4096, fb)) > 0) v.insert(v.end(), b, b + n);
            const size_t nb = v.size() / 32;
            std::vector<uint8_t> r0(nb * 34), r1(nb * 36), g0(nb * 36), g1(nb * 36), h0(nb * 34);
            const bool rd_ok = fread(r0.data(), 1, r0.size(), f0) == r0.size() && fread(r1.data(), 1, r1.size(), f1) == r1.size();
            const std::string src = k_src + std::string(k_test_src);
            const char * sp = src.c_str();
            cl_program p = clCreateProgramWithSource(ctx, 1, &sp, nullptr, &e);
            e = clBuildProgram(p, 1, &dev, "-cl-std=CL1.2 -DN_EMBD=2048 -DN_FF=768", nullptr, nullptr);
            cl_kernel kq = clCreateKernel(p, "t_q8", &e);
            cl_mem bi = clCreateBuffer(ctx, CL_MEM_COPY_HOST_PTR, v.size() * 4, v.data(), &e);
            cl_mem bo = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nb * 36, nullptr, &e);
            for (int ws = 0; ws < 2; ws++) {
                clSetKernelArg(kq, 0, sizeof bi, &bi); clSetKernelArg(kq, 1, sizeof bo, &bo); clSetKernelArg(kq, 2, sizeof(cl_int), &ws);
                const size_t one = 1;
                clEnqueueNDRangeKernel(q, kq, 1, nullptr, &nb, &one, 0, nullptr, nullptr);
                clEnqueueReadBuffer(q, bo, CL_TRUE, 0, nb * 36, ws ? g1.data() : g0.data(), 0, nullptr, nullptr);
            }
            gx_quantize_q8_0(v.data(), h0.data(), (int) v.size());
            for (size_t i = 0; rd_ok && i < nb; i++) {
                // GPU Q8_0: d at 0, qs at 4 (s unused); ggml Q8_0: d at 0, qs at 2
                if (memcmp(g0.data() + i * 36, r0.data() + i * 34, 2) || memcmp(g0.data() + i * 36 + 4, r0.data() + i * 34 + 2, 32)) q8_0_bad++;
                if (memcmp(g1.data() + i * 36, r1.data() + i * 36, 36)) q8_1_bad++;
                if (memcmp(h0.data() + i * 34, r0.data() + i * 34, 34)) host_q8_0_bad++;
            }
            q8_blocks = rd_ok ? (long) nb : -2;
            clReleaseMemObject(bi); clReleaseMemObject(bo); clReleaseKernel(kq); clReleaseProgram(p);
        }
        for (FILE * f : {fb, f0, f1}) if (f) fclose(f);
        printf("QUANT blocks=%ld gpu_q8_0_mismatch=%ld gpu_q8_1_mismatch=%ld host_q8_0_mismatch=%ld\n", q8_blocks, q8_0_bad, q8_1_bad, host_q8_0_bad);
    }

    // ---- SwiGLU (gx_expf + cr_div) on crafted operands over the whole float range, vs ggml ARM's swiglu_split
    long sw_n = -1, sw_bad = 0, sw_bad_unflagged = 0, sw_flagged = 0, sw_bad_dinput = 0;
    {
        FILE * fp = fopen((dir + "/swiglu_pairs.f32").c_str(), "rb");
        FILE * fr = fopen((dir + "/swiglu_pairs.f32.arm").c_str(), "rb");
        if (fp && fr) {
            std::vector<float> v, ref;
            float b[4096];
            size_t n;
            while ((n = fread(b, 4, 4096, fp)) > 0) v.insert(v.end(), b, b + n);
            while ((n = fread(b, 4, 4096, fr)) > 0) ref.insert(ref.end(), b, b + n);
            const size_t N = v.size() / 2;
            if (ref.size() == N) {
                const std::string src = k_src + std::string(k_test_src);
                const char * sp = src.c_str();
                cl_program p = clCreateProgramWithSource(ctx, 1, &sp, nullptr, &e);
                e = clBuildProgram(p, 1, &dev, "-cl-std=CL1.2 -DN_EMBD=2048 -DN_FF=768", nullptr, nullptr);
                cl_kernel ks = clCreateKernel(p, "t_swiglu", &e);
                cl_mem bg = clCreateBuffer(ctx, CL_MEM_COPY_HOST_PTR, N * 4, v.data(), &e);
                cl_mem bu = clCreateBuffer(ctx, CL_MEM_COPY_HOST_PTR, N * 4, v.data() + N, &e);
                cl_mem bh = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, N * 4, nullptr, &e);
                cl_mem br = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, N * 4, nullptr, &e);
                clSetKernelArg(ks, 0, sizeof bg, &bg); clSetKernelArg(ks, 1, sizeof bu, &bu); clSetKernelArg(ks, 2, sizeof bh, &bh);
                clSetKernelArg(ks, 3, sizeof br, &br);
                clEnqueueNDRangeKernel(q, ks, 1, nullptr, &N, nullptr, 0, nullptr, nullptr);
                std::vector<float> h(N);
                std::vector<int> rk(N);
                clEnqueueReadBuffer(q, bh, CL_TRUE, 0, N * 4, h.data(), 0, nullptr, nullptr);
                clEnqueueReadBuffer(q, br, CL_TRUE, 0, N * 4, rk.data(), 0, nullptr, nullptr);
                sw_n = (long) N;
                FILE * dump = fopen((dir + "/swiglu_mismatch.csv").c_str(), "w");   // every mismatch, for classification
                if (dump) fprintf(dump, "index,gate,up,gpu,ref,flagged\n");
                for (size_t i = 0; i < N; i++) {
                    sw_flagged += rk[i] != 0;
                    if (memcmp(&h[i], &ref[i], 4) && !(h[i] != h[i] && ref[i] != ref[i])) {   // NaN == NaN here
                        sw_bad++;
                        uint32_t bg, bu;
                        memcpy(&bg, &v[i], 4); memcpy(&bu, &v[N + i], 4);
                        bg &= 0x7fffffffu; bu &= 0x7fffffffu;
                        // a denormal gate/up input cannot occur in a gx dispatch (dot-product outputs are 0 or
                        // >= ~2^-71; checked below on every case) and a DAZ device hides it from the detector
                        if ((bg && bg < 0x00800000u) || (bu && bu < 0x00800000u)) sw_bad_dinput++;
                        else if (!rk[i]) sw_bad_unflagged++;
                        if (dump) fprintf(dump, "%zu,%a,%a,%a,%a,%d\n", i, v[i], v[N + i], h[i], ref[i], rk[i]);
                    }
                }
                if (dump) fclose(dump);
                clReleaseMemObject(bg); clReleaseMemObject(bu); clReleaseMemObject(bh); clReleaseMemObject(br);
                clReleaseKernel(ks); clReleaseProgram(p);
            } else sw_n = -2;
        }
        for (FILE * f : {fp, fr}) if (f) fclose(f);
        printf("SWIGLU n=%ld mismatches=%ld of_which_denormal_input=%ld unflagged_mismatches(normal inputs)=%ld flagged=%ld\n", sw_n,
               sw_bad, sw_bad_dinput, sw_bad_unflagged, sw_flagged);
    }

    // ---- cases
    std::vector<std::string> files;
    if (DIR * d = opendir(dir.c_str())) {
        while (dirent * de = readdir(d)) {
            std::string n = de->d_name;
            if (n.size() > 4 && n.substr(n.size() - 4) == ".bin") files.push_back(n);
        }
        closedir(d);
    }
    std::sort(files.begin(), files.end());
    FILE * js = fopen(argc > 2 ? argv[2] : "/dev/null", "w");
    fprintf(js, "{\"device\": \"%s\", \"driver\": \"%s\", \"fp32_denorm\": %d, \"div_n\": %zu, \"div_mismatches\": %zu, "
                "\"quant_blocks\": %ld, \"gpu_q8_0_mismatch\": %ld, \"gpu_q8_1_mismatch\": %ld, \"host_q8_0_mismatch\": %ld, \"swiglu_n\": %ld, \"swiglu_mismatches\": %ld, \"swiglu_unflagged_mismatches\": %ld, \"swiglu_flagged\": %ld, \"swiglu_denormal_input_mismatches\": %ld, \"cases\": [\n",
            name, ver, !!(fpc & CL_FP_DENORM), div_n, div_bad, q8_blocks, q8_0_bad, q8_1_bad, host_q8_0_bad, sw_n, sw_bad, sw_bad_unflagged, sw_flagged, sw_bad_dinput);
    gx_ctx * gv[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};   // variants 0, 1, 2 (2: repacked slots)
    // totals per variant (0 = lane-mapped, 1 = row per work-item); cases are counted once
    size_t tot_down[5] = {0}, tot_down_diff[5] = {0}, tot_h[5] = {0}, tot_h_diff[5] = {0};
    size_t missing_arm = 0, n_cases = 0, gx_errors = 0;
    size_t flagged_slots[5] = {0}, flagged_real[5] = {0}, unflagged_diff[5] = {0};
    float min_gu = INFINITY;
    size_t roundtrip_n = 0, roundtrip_bad = 0, wrongtype_n = 0, wrongtype_undetected = 0;   // variant 2: gx_unpack_expert(gx_repack_expert(x)) == x   // smallest nonzero |gate|, |up| in the ggml reference: the no-denormal-input premise
    for (size_t fi = 0; fi < files.size(); fi++) {
        Case c;
        const std::string base = dir + "/" + files[fi];
        if (!load_case(base, c)) { fprintf(stderr, "bad case %s\n", base.c_str()); return 1; }
        for (int v = 0; v < 5; v++)
            if (!gv[v]) {
                char err[4096];
                gv[v] = gx_init(ctx, dev, gx_params{c.ne, c.nf, 1, v, 0, v & 1 /* spin_wait: v1 polls, v0 and v2 block */}, err, sizeof err);
                if (!gv[v]) { fprintf(stderr, "gx_init variant %d: %s\n", v, err); return 1; }
            }
      for (int v = 0; v < 5; v++) {
        gx_ctx * g = gv[v];
        const size_t gu = (size_t) c.nf * (c.ne / 32) * 18, per = 2 * gu + (size_t) c.ne * (c.nf / 32) * (c.dt ? 20 : 18);
        // weights in a host-visible buffer, filled through a map (the pool's protocol)
        cl_mem blk = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, c.w.size(), nullptr, &e);
        void * mp = clEnqueueMapBuffer(q, blk, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, c.w.size(), 0, nullptr, nullptr, &e);
        std::vector<gx_slot> slots(c.k);
        for (int s = 0; s < c.k; s++) slots[s] = {blk, per * c.ids[s], per * c.ids[s] + gu, per * c.ids[s] + 2 * gu};
        if (gx_variant_uses_repack(v)) {   // variants 2, 3: every expert written through gx_repack_expert
            for (int ex = 0; ex < c.k; ex++) {
                const gx_slot se = {blk, per * ex, per * ex + gu, per * ex + 2 * gu};
                const uint8_t * src = c.w.data() + per * ex;
                if (gx_repack_expert(g, &se, (uint8_t *) mp + per * ex, src, src + gu, src + 2 * gu, c.dt)) gx_errors++;
                // round trip: unpack must give back the GGUF bytes exactly
                std::vector<uint8_t> back(per);
                if (gx_unpack_expert(g, &se, (uint8_t *) mp + per * ex, back.data(), back.data() + gu, back.data() + 2 * gu, c.dt) ||
                    memcmp(back.data(), src, per)) { roundtrip_bad++; }
                roundtrip_n++;
                // negative check: unpacking with the wrong down type must be refused (-2), never misread
                {
                    std::vector<uint8_t> wrong((size_t) c.ne * (c.nf / 32) * 20);
                    if (gx_unpack_expert(g, &se, (uint8_t *) mp + per * ex, nullptr, nullptr, wrong.data(),
                                         c.dt ? GX_DOWN_Q4_0 : GX_DOWN_Q4_1) != -2) wrongtype_undetected++;
                    wrongtype_n++;
                }
            }
        } else {
            memcpy(mp, c.w.data(), c.w.size());
        }
        clEnqueueUnmapMemObject(q, blk, mp, 0, nullptr, nullptr);
        clFinish(q);
        std::vector<float> out((size_t) c.k * c.ne), h((size_t) c.k * c.nf);
        int rc = gx_dispatch(g, 0, c.dt, c.k, slots.data(), c.x.data(), out.data());
        if (!rc) rc = gx_wait(g);
        if (!rc) rc = gx_debug_h(g, c.k, h.data());
        const uint32_t rmask = gx_last_risk_mask(g);
        if (rc) { gx_errors++; fprintf(stderr, "gx error %d on %s\n", rc, files[fi].c_str()); }
        clReleaseMemObject(blk);
        const Ref arm = load_ref(base + ".arm.out", c.k, c.nf, c.ne), x86 = load_ref(base + ".x86.out", c.k, c.nf, c.ne);
        const std::vector<double> ex = fp64_ffn(c);
        Cmp da, ha, dx, de_gx, de_arm;
        if (arm.ok) {
            da = compare(out, arm.down, c.k, c.ne);
            ha = compare(h, arm.h, c.k, c.nf);
            de_arm = compare(arm.down, ex, c.k, c.ne);
            tot_down[v] += da.n; tot_down_diff[v] += da.diff_bits; tot_h[v] += ha.n; tot_h_diff[v] += ha.diff_bits;
            for (int sl = 0; sl < c.k; sl++) {   // differences in slots the detector did NOT flag
                if (rmask & (1u << sl)) { flagged_slots[v]++; if (files[fi].rfind("qwen3_", 0) == 0) flagged_real[v]++; continue; }
                unflagged_diff[v] += diff_count(out.data() + (size_t) sl * c.ne, arm.down.data() + (size_t) sl * c.ne, c.ne) +
                                     diff_count(h.data() + (size_t) sl * c.nf, arm.h.data() + (size_t) sl * c.nf, c.nf);
            }
        } else if (v == 0) missing_arm++;
        if (x86.ok) dx = compare(out, x86.down, c.k, c.ne);
        if (arm.ok && v == 0)
            for (size_t i = 0; i < arm.gate.size(); i++)
                for (float z : {arm.gate[i], arm.up[i]})
                    if (z != 0.0f && fabsf(z) < min_gu) min_gu = fabsf(z);
        de_gx = compare(out, ex, c.k, c.ne);
        if (v == 0) n_cases++;
        printf("CASE %-16s v%d down=%s k=%d arm=%d down_diff_bits=%zu/%zu max_ulp=%lld h_diff_bits=%zu/%zu rel_vs_arm=%.3g "
               "rel_vs_x86=%.3g rel_vs_fp64=%.3g arm_rel_vs_fp64=%.3g\n",
               files[fi].c_str(), v, c.dt ? "Q4_1" : "Q4_0", c.k, arm.ok, da.diff_bits, da.n, (long long) da.max_ulp, ha.diff_bits, ha.n, da.max_rel,
               x86.ok ? dx.max_rel : NAN, de_gx.max_rel, arm.ok ? de_arm.max_rel : NAN);
        fprintf(js, "%s {\"case\": \"%s\", \"variant\": %d, \"down\": \"%s\", \"k\": %d, \"arm_ref\": %d, \"down_n\": %zu, \"down_diff_bits\": %zu, \"down_max_ulp\": %lld, "
                    "\"h_n\": %zu, \"h_diff_bits\": %zu, \"h_max_ulp\": %lld, \"rel_vs_arm\": %.6g, \"rel_vs_x86\": %.6g, "
                    "\"rel_vs_fp64\": %.6g, \"arm_rel_vs_fp64\": %.6g, \"gx_error\": %d}\n",
                (fi || v) ? "," : "", files[fi].c_str(), v, c.dt ? "Q4_1" : "Q4_0", c.k, arm.ok, da.n, da.diff_bits, (long long) da.max_ulp, ha.n, ha.diff_bits,
                (long long) ha.max_ulp, arm.ok ? da.max_rel : -1.0, x86.ok ? dx.max_rel : -1.0, de_gx.max_rel,
                arm.ok ? de_arm.max_rel : -1.0, rc);
      }
    }
    unsigned long long st_err = 0;
    for (gx_ctx * g : gv) if (g) st_err += gx_get_stats(g).errors;
    // variant 0 keeps the unsuffixed names (claims gx_down_*); variants 1, 2, 3 are *_row, *_soa, *_tiled
    fprintf(js, "], \"cases_n\": %zu, \"missing_arm_ref\": %zu, \"down_values\": %zu, \"down_diff_bits\": %zu, "
                "\"h_values\": %zu, \"h_diff_bits\": %zu, \"down_values_row\": %zu, \"down_diff_bits_row\": %zu, "
                "\"h_values_row\": %zu, \"h_diff_bits_row\": %zu, \"down_values_soa\": %zu, \"down_diff_bits_soa\": %zu, "
                "\"h_values_soa\": %zu, \"h_diff_bits_soa\": %zu, \"down_values_tiled\": %zu, \"down_diff_bits_tiled\": %zu, "
                "\"h_values_tiled\": %zu, \"h_diff_bits_tiled\": %zu, \"gx_errors\": %zu, \"gx_stats_errors\": %llu, "
                "\"flagged_slots\": [%zu, %zu, %zu, %zu], \"flagged_real_slots\": [%zu, %zu, %zu, %zu], "
                "\"unflagged_diff\": [%zu, %zu, %zu, %zu], \"roundtrip_experts\": %zu, \"roundtrip_mismatched\": %zu, "
                "\"min_nonzero_gate_up\": %.9g}\n",
            n_cases, missing_arm, tot_down[0], tot_down_diff[0], tot_h[0], tot_h_diff[0], tot_down[1], tot_down_diff[1], tot_h[1],
            tot_h_diff[1], tot_down[2], tot_down_diff[2], tot_h[2], tot_h_diff[2], tot_down[3], tot_down_diff[3], tot_h[3],
            tot_h_diff[3], gx_errors, st_err, flagged_slots[0], flagged_slots[1], flagged_slots[2], flagged_slots[3],
            flagged_real[0], flagged_real[1], flagged_real[2], flagged_real[3], unflagged_diff[0], unflagged_diff[1],
            unflagged_diff[2], unflagged_diff[3], roundtrip_n, roundtrip_bad, min_gu);
    fclose(js);
    bool all_raw = true, all_unflagged = true;
    for (int v = 0; v < 5; v++) {
        all_raw = all_raw && tot_down_diff[v] == 0 && tot_h_diff[v] == 0 && tot_down[v] == tot_down[0];
        all_unflagged = all_unflagged && unflagged_diff[v] == 0 && flagged_real[v] == 0 && tot_down[v] == tot_down[0];
    }
    const bool common = missing_arm == 0 && n_cases > 0 && roundtrip_bad == 0 && roundtrip_n > 0 && wrongtype_undetected == 0 &&
                        div_bad == 0 && gx_errors == 0 && q8_blocks > 0 && q8_0_bad == 0 && q8_1_bad == 0 && host_q8_0_bad == 0 && sw_n > 0;
    const bool exact = common && all_raw && sw_bad == 0;
    // Rule for devices that flush fp32 denormals (pre-registered 2026-09-18, after M6 run gx_m6_224521):
    // PASS_WITH_DETECTOR = zero differences in every slot / sweep value the detector did not flag, zero
    // flags on the real Qwen3 cases, quantizer and division exact. BIT-EXACT additionally needs zero raw diffs.
    const bool detector_pass = common && all_unflagged && sw_bad_unflagged == 0 && min_gu >= 0x1p-126f;
    printf("ROUNDTRIP repack/unpack experts=%zu mismatched=%zu wrong_down_type_checks=%zu undetected=%zu\n", roundtrip_n,
           roundtrip_bad, wrongtype_n, wrongtype_undetected);
    printf("PREMISE min_nonzero_gate_up=%a (%s 2^-126)\n", min_gu, min_gu >= 0x1p-126f ? ">=" : "< !!");
    printf("DETECTOR flagged_slots(v0/v1/v2/v3)=%zu/%zu/%zu/%zu flagged_real_slots=%zu/%zu/%zu/%zu unflagged_diff=%zu/%zu/%zu/%zu "
           "swiglu_flagged=%ld swiglu_unflagged_mismatches=%ld -> %s\n", flagged_slots[0], flagged_slots[1], flagged_slots[2],
           flagged_slots[3], flagged_real[0], flagged_real[1], flagged_real[2], flagged_real[3], unflagged_diff[0], unflagged_diff[1],
           unflagged_diff[2], unflagged_diff[3], sw_flagged, sw_bad_unflagged, detector_pass ? "PASS_WITH_DETECTOR" : "DETECTOR_FAIL");
    printf("SUMMARY cases=%zu missing_arm_ref=%zu down_diff_bits(v0/v1/v2/v3)=%zu/%zu/%zu/%zu of %zu h_diff_bits(v0/v1/v2/v3)=%zu/%zu/%zu/%zu of %zu "
           "div_mismatches=%zu quant_blocks=%ld q8_mismatch(gpu0/gpu1/host0)=%ld/%ld/%ld swiglu=%ld/%ld gx_errors=%zu -> %s\n",
           n_cases, missing_arm, tot_down_diff[0], tot_down_diff[1], tot_down_diff[2], tot_down_diff[3], tot_down[0], tot_h_diff[0],
           tot_h_diff[1], tot_h_diff[2], tot_h_diff[3], tot_h[0], div_bad, q8_blocks, q8_0_bad, q8_1_bad, host_q8_0_bad, sw_bad, sw_n,
           gx_errors, exact ? "BIT-EXACT vs ggml-cpu ARM" : "NOT bit-exact");
    printf("VARIANT4 down_diff_bits=%zu/%zu h_diff_bits=%zu/%zu flagged_slots=%zu flagged_real_slots=%zu unflagged_diff=%zu\n",
           tot_down_diff[4], tot_down[4], tot_h_diff[4], tot_h[4], flagged_slots[4], flagged_real[4], unflagged_diff[4]);
    for (gx_ctx * g : gv) if (g) gx_free(g);
    return exact ? 0 : (detector_pass ? 4 : 3);
}
