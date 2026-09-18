// gx_bench.cpp — dispatch latency and throughput of gx, for M6 on the phone. Correctness is gx_test's job;
// this only times. On the laptop it is a functional smoke test and its numbers mean nothing (NVIDIA copies).
//   gx_bench [--iters N] [--variants 0,1,2] [--ks 1,..,8] [--slots N] [--spin 0,1]
// A 1.5 s k=8 warm-up precedes timing, and the (down type, k) cells are timed in interleaved rounds of 10.
// Pool: one expert per block (the recommended shape), N slots per down type (default 32), filled with
// random but valid blocks through map/unmap. Each timed dispatch uses k slots rotating through the pool so
// consecutive dispatches read different weights (as consecutive layers do), dispatch -> gx_wait wall time.
// Output: BENCH lines (variant, down type, k, n, host median/p10/p90 ms dispatch->wait, GB/s of weights at the
// median, device (profiling-event) median/p90 ms; profiling is on, as in the engine) and one
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
    std::vector<int> variants = {0, 1, 2}, ks = {1, 2, 3, 4, 5, 6, 7, 8}, spins = {0, 1};
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--iters")) iters = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--variants")) variants = ints(argv[i + 1]);
        else if (!strcmp(argv[i], "--ks")) ks = ints(argv[i + 1]);
        else if (!strcmp(argv[i], "--slots")) nslots = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--spin")) spins = ints(argv[i + 1]);
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
    for (int spin : spins)
    for (int variant : variants) {
        char err[1024];
        gx_ctx * g = gx_init(ctx, dev, gx_params{2048, 768, 0, variant, 1, spin}, err, sizeof err);
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
                if (variant == 2) {   // variant 2 slots hold the repacked layout (the engine's promotion path)
                    gx_repack_expert(g, &s, p, src.data(), src.data() + GU, src.data() + 2 * GU, dt);
                } else {
                    memcpy(p, src.data(), GU);
                    memcpy(p + (s.off_up - s.off_gate), src.data() + GU, GU);
                    memcpy(p + (s.off_down - s.off_gate), src.data() + 2 * GU, DN[dt]);
                }
                gx_slot_unmap(g, &s, p);
                wr.push_back(now_ms() - t0);
                sl[dt].push_back(s);
            }
        std::sort(wr.begin(), wr.end());
        printf("SLOTWRITE variant=%d n=%zu median_ms=%.3f p90_ms=%.3f\n", variant, wr.size(), wr[wr.size() / 2], wr[wr.size() * 9 / 10]);
        std::vector<float> out(8 * 2048);
        // GPU clock ramp (M6 run 2: the first rows ran slow): 1.5 s of k=8 dispatches before any timing, then
        // every (down type, k) cell is visited in rounds of 10 dispatches, in a rotated order each round, so a
        // clock drift spreads over all cells instead of biasing the first ones.
        {
            gx_slot use[GX_MAX_K];
            const double tw = now_ms();
            for (int it = 0; now_ms() - tw < 1500.0; it++) {
                for (int j = 0; j < 8; j++) use[j] = sl[it & 1][(it * 8 + j) % nslots];
                if (gx_dispatch(g, 0, it & 1, 8, use, x.data(), out.data()) || gx_wait(g)) { printf("BENCH status=warmup_error\n"); return 1; }
            }
        }
        struct Cell { int dt, k; std::vector<double> t, td; long it = 0; };
        std::vector<Cell> cells;
        for (int dt = 0; dt < 2; dt++)
            for (int k : ks) cells.push_back(Cell{dt, k, {}, {}});
        const int per_round = 10, rounds = (iters + per_round - 1) / per_round;
        for (int r = 0; r < rounds; r++)
            for (size_t ci = 0; ci < cells.size(); ci++) {
                Cell & c = cells[(ci + (size_t) r * 3) % cells.size()];
                gx_slot use[GX_MAX_K];
                for (int rep = 0; rep < per_round; rep++, c.it++) {
                    for (int j = 0; j < c.k; j++) use[j] = sl[c.dt][(c.it * c.k + j) % nslots];
                    const double t0 = now_ms();
                    const int rc = gx_dispatch(g, 0, c.dt, c.k, use, x.data(), out.data());
                    const int rw = rc ? rc : gx_wait(g);
                    const double dtm = now_ms() - t0;
                    if (rc || rw) { printf("BENCH status=dispatch_error rc=%d\n", rc ? rc : rw); return 1; }
                    uint64_t dn = 0, hn = 0;
                    gx_last_timing(g, &dn, &hn);
                    c.t.push_back(dtm);
                    c.td.push_back(dn / 1e6);
                }
            }
        for (Cell & c : cells) {
            std::sort(c.t.begin(), c.t.end());
            std::sort(c.td.begin(), c.td.end());
            const double med = c.t[c.t.size() / 2];
            const double bytes = (double) c.k * (2 * GU + DN[c.dt]);
            printf("BENCH spin=%d variant=%d down=%s k=%d n=%zu median_ms=%.4f p10_ms=%.4f p90_ms=%.4f GBps_at_median=%.2f "
                   "device_median_ms=%.4f device_p90_ms=%.4f\n", spin, variant, c.dt ? "Q4_1" : "Q4_0", c.k, c.t.size(), med,
                   c.t[c.t.size() / 10], c.t[c.t.size() * 9 / 10], bytes / (med / 1e3) / 1e9, c.td[c.td.size() / 2],
                   c.td[c.td.size() * 9 / 10]);
        }
        fflush(stdout);
        const gx_stats st = gx_get_stats(g);
        printf("STATS variant=%d dispatches=%llu errors=%llu map_errors=%llu\n", variant, (unsigned long long) st.dispatches,
               (unsigned long long) st.errors, (unsigned long long) st.map_errors);
        for (auto & v : sl) for (auto & s : v) gx_slot_free(g, &s);
        gx_free(g);
    }
    return 0;
}
