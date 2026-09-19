#include "gx.h"

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

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


// gx_tol_test — acceptance for gx variant 5 (E4: declared tolerance instead of bit-exactness).
// Per case: max abs error and max relative error (per slot max|a-b| / max|b|, worst slot; and the largest
// elementwise |a-b|) of variant 5 against (1) an fp64 FFN on the same dequantized weights with the UNQUANTIZED x
// (the kernel's own arithmetic error) and (2) ggml-cpu ARM's output (<case>.arm.out; dominated by ggml's Q8
// activation error, reported for the record). GATE, fixed before the first run: (1) <= 1e-5 for every slot of
// every case. Exit 1 on a gate failure or any gx error. Also prints ggml's own error vs fp64 for scale.
//   gx_tol_test <case_dir> [variant=5]
int main(int argc, char ** argv) {
    const std::string dir = argv[1];
    const int variant = argc > 2 ? atoi(argv[2]) : 5;
    const double GATE = 1e-5;
    std::vector<std::string> files;
    if (DIR * d = opendir(dir.c_str())) {
        while (dirent * de = readdir(d)) {
            const std::string n = de->d_name;
            if (n.size() > 4 && n.substr(n.size() - 4) == ".bin" && (n.rfind("rand_", 0) == 0 || n.rfind("qwen3_", 0) == 0 || n.rfind("olmoe_", 0) == 0)) files.push_back(n);
        }
        closedir(d);
    }
    std::sort(files.begin(), files.end());
    cl_platform_id plat; cl_device_id dev; cl_int e;
    if (clGetPlatformIDs(1, &plat, nullptr) || clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr)) { fprintf(stderr, "no GPU\n"); return 1; }
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &e);
    gx_ctx * g = nullptr;
    int fails = 0, errors = 0, n_cases = 0;
    double worst_rel = 0, worst_abs = 0, worst_arm_rel = 0;
    for (const std::string & f : files) {
        Case c;
        if (!load_case(dir + "/" + f, c)) { fprintf(stderr, "bad case %s\n", f.c_str()); return 1; }
        if (!g) {
            char err[4096];
            g = gx_init(ctx, dev, gx_params{c.ne, c.nf, 1, variant, 0, 0}, err, sizeof err);
            if (!g) { fprintf(stderr, "gx_init: %s\n", err); return 1; }
        }
        const size_t gu = (size_t) c.nf * (c.ne / 32) * 18, per = 2 * gu + (size_t) c.ne * (c.nf / 32) * (c.dt ? 20 : 18);
        cl_mem blk = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, c.w.size(), nullptr, &e);
        void * mp = clEnqueueMapBuffer(q, blk, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, c.w.size(), 0, nullptr, nullptr, &e);
        memcpy(mp, c.w.data(), c.w.size());
        clEnqueueUnmapMemObject(q, blk, mp, 0, nullptr, nullptr);
        clFinish(q);
        std::vector<gx_slot> slots(c.k);
        for (int s = 0; s < c.k; s++) slots[s] = {blk, per * c.ids[s], per * c.ids[s] + gu, per * c.ids[s] + 2 * gu};
        std::vector<float> out((size_t) c.k * c.ne);
        int rc = gx_dispatch(g, 0, c.dt, c.k, slots.data(), c.x.data(), out.data());
        if (!rc) rc = gx_wait(g);
        clReleaseMemObject(blk);
        if (rc) { errors++; fprintf(stderr, "gx error %d on %s\n", rc, f.c_str()); continue; }
        const std::vector<double> ex = fp64_ffn(c);
        double rel = 0, mabs = 0, ymax = 0;
        bool finite = true;
        for (int s = 0; s < c.k; s++) {
            double num = 0, den = 0;
            for (int i = 0; i < c.ne; i++) {
                const size_t j = (size_t) s * c.ne + i;
                if (!std::isfinite(out[j])) finite = false;
                num = std::max(num, fabs((double) out[j] - ex[j]));
                den = std::max(den, fabs(ex[j]));
            }
            mabs = std::max(mabs, num); ymax = std::max(ymax, den);
            rel = std::max(rel, den > 0 ? num / den : num);
        }
        double arm_rel = NAN, arm_vs64 = NAN;
        const Ref arm = load_ref(dir + "/" + f + ".arm.out", c.k, c.nf, c.ne);
        if (arm.ok) {
            arm_rel = 0; arm_vs64 = 0;
            for (int s = 0; s < c.k; s++) {
                double n1 = 0, n2 = 0, den = 0;
                for (int i = 0; i < c.ne; i++) {
                    const size_t j = (size_t) s * c.ne + i;
                    n1 = std::max(n1, fabs((double) out[j] - arm.down[j]));
                    n2 = std::max(n2, fabs((double) arm.down[j] - ex[j]));
                    den = std::max(den, fabs((double) arm.down[j]));
                }
                arm_rel = std::max(arm_rel, den > 0 ? n1 / den : n1);
                arm_vs64 = std::max(arm_vs64, den > 0 ? n2 / den : n2);
            }
        }
        const bool ok = finite && rel <= GATE;
        if (!ok) fails++;
        n_cases++;
        worst_rel = std::max(worst_rel, rel); worst_abs = std::max(worst_abs, mabs);
        if (arm.ok) worst_arm_rel = std::max(worst_arm_rel, arm_rel);
        printf("TOL %-16s v%d down=%s k=%d max_abs_vs_fp64=%.3g max_out=%.3g rel_vs_fp64=%.3g rel_vs_arm=%.3g arm_rel_vs_fp64=%.3g %s\n", f.c_str(), variant,
               c.dt ? "Q4_1" : "Q4_0", c.k, mabs, ymax, rel, arm_rel, arm_vs64, ok ? "ok" : "FAIL");
    }
    const gx_stats st = gx_get_stats(g);
    printf("TOLSUMMARY variant=%d cases=%d gate=%.0e fails=%d gx_errors=%d stats_errors=%llu worst_rel_vs_fp64=%.3g worst_abs_vs_fp64=%.3g worst_rel_vs_arm=%.3g\n", variant,
           n_cases, GATE, fails, errors, (unsigned long long) st.errors, worst_rel, worst_abs, worst_arm_rel);
    return fails || errors || st.errors || n_cases == 0;
}
