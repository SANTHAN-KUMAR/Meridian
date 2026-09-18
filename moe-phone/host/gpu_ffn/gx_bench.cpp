// gx_bench.cpp — dispatch latency and throughput of gx, for M6 on the phone. Correctness is gx_test's job;
// this only times. On the laptop it is a functional smoke test and its numbers mean nothing (NVIDIA copies).
//   gx_bench [--iters N] [--variants 0,1] [--ks 1,2,4,8] [--slots N]
// Pool: one expert per block (the recommended shape), N slots per down type (default 32), filled with
// random but valid blocks through map/unmap. Each timed dispatch uses k slots rotating through the pool so
// consecutive dispatches read different weights (as consecutive layers do), dispatch -> gx_wait wall time.
// Output: BENCH lines (variant, down type, k, n, median/p10/p90 ms, GB/s of weights at the median) and one
// SLOTWRITE line (map + 2.6 MB memcpy + unmap per slot).
#include "gx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

static const size_t GU = 768 * (2048 / 32) * 18, DN[2] = {2048 * (768 / 32) * 18, 2048 * (768 / 32) * 20};
static double now_ms() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

static std::vector<int> ints(const char * s) {
    std::vector<int> v;
    for (const char * p = s; *p;) { v.push_back(atoi(p)); while (*p && *p != ',') p++; if (*p) p++; }
    return v;
}

static void fill_q4(std::mt19937 & rng, uint8_t * p, size_t bytes, int blk) {   // blk = 18 (Q4_0) or 20 (Q4_1)
    for (size_t o = 0; o + blk <= bytes; o += blk) {
        const uint16_t d = 0x1000 + (rng() & 0x3ff), m = 0x9000 + (rng() & 0x3ff);   // small positive d, negative m
        memcpy(p + o, &d, 2);
        if (blk == 20) memcpy(p + o + 2, &m, 2);
        for (int i = blk - 16; i < blk; i++) p[o + i] = (uint8_t) rng();
    }
}

int main(int argc, char ** argv) {
    int iters = 300, nslots = 32;
    std::vector<int> variants = {0, 1}, ks = {1, 2, 4, 8};
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--iters")) iters = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--variants")) variants = ints(argv[i + 1]);
        else if (!strcmp(argv[i], "--ks")) ks = ints(argv[i + 1]);
        else if (!strcmp(argv[i], "--slots")) nslots = atoi(argv[i + 1]);
    }
    cl_platform_id plat;
    cl_device_id dev;
    char name[256] = {0};
    if (clGetPlatformIDs(1, &plat, nullptr) || clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr)) { fprintf(stderr, "no GPU\n"); return 1; }
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof name, name, nullptr);
    cl_int e;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e);
    printf("DEVICE name=\"%s\" iters=%d slots_per_type=%d\n", name, iters, nslots);
    std::mt19937 rng(1);
    std::vector<float> x(2048);
    for (auto & v : x) v = (float) ((int) (rng() % 2001) - 1000) / 250.0f;
    for (int variant : variants) {
        char err[1024];
        gx_ctx * g = gx_init(ctx, dev, gx_params{2048, 768, 0, variant}, err, sizeof err);
        if (!g) { printf("BENCH variant=%d status=init_failed why=\"%s\"\n", variant, err); continue; }
        const size_t span = 2 * GU + DN[1];   // Q4_1-sized block holds either type (all sizes are 4 KB multiples)
        const int got = gx_pool_create(g, span, 2 * nslots, err, sizeof err);
        printf("POOL variant=%d blocks=%d/%d :: %s\n", variant, got, 2 * nslots, err);
        if (got != 2 * nslots) { gx_free(g); continue; }
        std::vector<gx_slot> sl[2];
        std::vector<double> wr;
        std::vector<uint8_t> src(span);
        for (int dt = 0; dt < 2; dt++)
            for (int i = 0; i < nslots; i++) {
                gx_slot s;
                if (!gx_slot_alloc(g, GU, GU, DN[dt], &s)) { printf("BENCH status=alloc_failed\n"); return 1; }
                fill_q4(rng, src.data(), GU, 18);
                fill_q4(rng, src.data() + GU, GU, 18);
                fill_q4(rng, src.data() + 2 * GU, DN[dt], dt ? 20 : 18);
                const double t0 = now_ms();
                uint8_t * p = (uint8_t *) gx_slot_map_write(g, &s);
                memcpy(p, src.data(), GU);
                memcpy(p + (s.off_up - s.off_gate), src.data() + GU, GU);
                memcpy(p + (s.off_down - s.off_gate), src.data() + 2 * GU, DN[dt]);
                gx_slot_unmap(g, &s, p);
                wr.push_back(now_ms() - t0);
                sl[dt].push_back(s);
            }
        std::sort(wr.begin(), wr.end());
        printf("SLOTWRITE variant=%d n=%zu median_ms=%.3f p90_ms=%.3f\n", variant, wr.size(), wr[wr.size() / 2], wr[wr.size() * 9 / 10]);
        std::vector<float> out(8 * 2048);
        for (int dt = 0; dt < 2; dt++)
            for (int k : ks) {
                std::vector<double> t;
                gx_slot use[GX_MAX_K];
                for (int it = -20; it < iters; it++) {
                    for (int j = 0; j < k; j++) use[j] = sl[dt][((it + 20) * k + j) % nslots];
                    const double t0 = now_ms();
                    const int rc = gx_dispatch(g, 0, dt, k, use, x.data(), out.data());
                    const int rw = rc ? rc : gx_wait(g);
                    const double dtm = now_ms() - t0;
                    if (rc || rw) { printf("BENCH status=dispatch_error rc=%d\n", rc ? rc : rw); return 1; }
                    if (it >= 0) t.push_back(dtm);
                }
                std::sort(t.begin(), t.end());
                const double med = t[t.size() / 2];
                const double bytes = (double) k * (2 * GU + DN[dt]);
                printf("BENCH variant=%d down=%s k=%d n=%zu median_ms=%.4f p10_ms=%.4f p90_ms=%.4f GBps_at_median=%.2f\n", variant,
                       dt ? "Q4_1" : "Q4_0", k, t.size(), med, t[t.size() / 10], t[t.size() * 9 / 10], bytes / (med / 1e3) / 1e9);
                fflush(stdout);
            }
        const gx_stats st = gx_get_stats(g);
        printf("STATS variant=%d dispatches=%llu errors=%llu map_errors=%llu\n", variant, (unsigned long long) st.dispatches,
               (unsigned long long) st.errors, (unsigned long long) st.map_errors);
        for (auto & v : sl) for (auto & s : v) gx_slot_free(g, &s);
        gx_free(g);
    }
    return 0;
}
