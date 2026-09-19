// gx_bench.cpp — dispatch latency and throughput of gx, for M6 on the phone. Correctness is gx_test's job;
// this only times. On the laptop it is a functional smoke test and its numbers mean nothing (NVIDIA copies).
//   gx_bench [--iters N] [--variants 0,1,2,3] [--ks 1,..,8] [--slots N] [--spin 0,1] [--downs 0,1] [--memprobe-only 1]
// A memory-only probe (MEMPROBE lines: aos / soa / tiled read orders, no arithmetic) runs once first.
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

// Memory-only probe (M6 run 3 follow-up): read the gate matrix of a slot (768 rows x 64 blocks) with no arithmetic,
// in three orders, one work-item per row, 64 per group, rotating over the pool's slots:
//   mode 0 "aos"   v0/v1: row-major 18-byte blocks, work-item streams its own row (vload16 at 2 + ib*18)
//   mode 1 "soa"   v2:    16-byte chunks [ib][row]: coalesced per block, a new page per block
//   mode 2 "tiled" v3:    [row/64][ib][row%64]: coalesced and one contiguous region per work-group
// Reported: device GB/s of the 16 qs bytes per block (the d bytes are not read), from profiling events.
static const char * k_mem_src = R"CL(
kernel void mem_rd(global const uchar * w, int mode, int R, int NB, global uint * sink) {
    const int row = get_global_id(0);
    uint4 acc = (uint4)(0);
    for (int ib = 0; ib < NB; ib++) {
        uint4 v;
        if (mode == 0) v = as_uint4(vload16(0, w + ((size_t) row * NB + ib) * 18 + 2));
        else if (mode == 1) v = ((global const uint4 *) w)[(size_t) ib * R + row];
        else v = ((global const uint4 *) w)[(((size_t) row >> 6) * NB + ib) * 64 + (row & 63)];
        acc ^= v;
    }
    sink[row] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}
)CL";

static void mem_probe(cl_context ctx, cl_device_id dev, const std::vector<gx_slot> & slots, int iters) {
    cl_int e;
    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &e);
    cl_program p = clCreateProgramWithSource(ctx, 1, &k_mem_src, nullptr, &e);
    if (e == CL_SUCCESS) e = clBuildProgram(p, 1, &dev, "-cl-std=CL1.2", nullptr, nullptr);
    if (e != CL_SUCCESS) { printf("MEMPROBE status=build_failed %d\n", e); return; }
    cl_kernel k = clCreateKernel(p, "mem_rd", &e);
    cl_mem sink = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, 768 * 4, nullptr, &e);
    const int R = 768, NB = 64;
    static const char * names[3] = {"aos", "soa", "tiled"};
    for (int round = 0; round < 2; round++)          // two passes, modes interleaved, to expose drift
        for (int mode = 0; mode < 3; mode++) {
            std::vector<double> t;
            for (int it = 0; it < iters; it++) {
                const gx_slot & s = slots[(size_t) it % slots.size()];
                clSetKernelArg(k, 0, sizeof(cl_mem), &s.block);
                clSetKernelArg(k, 1, sizeof(int), &mode);
                clSetKernelArg(k, 2, sizeof(int), &R);
                clSetKernelArg(k, 3, sizeof(int), &NB);
                clSetKernelArg(k, 4, sizeof(cl_mem), &sink);
                const size_t g = R, l = 64;
                cl_event ev = nullptr;
                // the slot's gate slice starts at off_gate; the kernel indexes from the buffer start, so use slots at 0
                if (s.off_gate != 0) continue;
                clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, &l, 0, nullptr, &ev);
                clWaitForEvents(1, &ev);
                cl_ulong a = 0, b = 0;
                clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof a, &a, nullptr);
                clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof b, &b, nullptr);
                clReleaseEvent(ev);
                t.push_back((b - a) / 1e6);
            }
            if (t.empty()) { printf("MEMPROBE status=no_slot_at_offset_0\n"); return; }
            std::sort(t.begin(), t.end());
            const double med = t[t.size() / 2], bytes = (double) R * NB * 16;
            printf("MEMPROBE pass=%d order=%s n=%zu device_median_ms=%.4f device_p90_ms=%.4f GBps=%.2f\n", round, names[mode], t.size(),
                   med, t[t.size() * 9 / 10], bytes / (med / 1e3) / 1e9);
            fflush(stdout);
        }
    clReleaseMemObject(sink);
    clReleaseKernel(k);
    clReleaseProgram(p);
    clReleaseCommandQueue(q);
}

int main(int argc, char ** argv) {
    int iters = 300, nslots = 32;
    int memprobe_only = 0;
    std::vector<int> downs = {0, 1};
    std::vector<int> variants = {0, 1, 2, 3, 4}, ks = {1, 2, 3, 4, 5, 6, 7, 8}, spins = {0, 1};
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--iters")) iters = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--variants")) variants = ints(argv[i + 1]);
        else if (!strcmp(argv[i], "--ks")) ks = ints(argv[i + 1]);
        else if (!strcmp(argv[i], "--slots")) nslots = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--spin")) spins = ints(argv[i + 1]);
        else if (!strcmp(argv[i], "--memprobe-only")) memprobe_only = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--downs")) downs = ints(argv[i + 1]);
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
                if (gx_variant_uses_repack(variant)) {   // variants 2, 3: slots hold the repacked layout (the engine's promotion path)
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
        static bool probed = false;
        if (!probed) {   // memory-only probe once, on this context's slots (one expert per block: off_gate = 0)
            probed = true;
            std::vector<gx_slot> all(sl[0]);
            all.insert(all.end(), sl[1].begin(), sl[1].end());
            mem_probe(ctx, dev, all, iters);
        }
        if (memprobe_only) { for (auto & v : sl) for (auto & s2 : v) gx_slot_free(g, &s2); gx_free(g); return 0; }
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
        struct Cell { int dt, k; std::vector<double> t, td, t1, tg, t2; long it = 0; };
        std::vector<Cell> cells;
        for (int dt : downs)
            for (int k : ks) cells.push_back(Cell{dt, k, {}, {}, {}, {}, {}});
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
                    uint64_t k1 = 0, gp = 0, k2 = 0;
                    gx_last_timing_split(g, &k1, &gp, &k2);
                    c.t1.push_back(k1 / 1e6); c.tg.push_back(gp / 1e6); c.t2.push_back(k2 / 1e6);
                }
            }
        for (Cell & c : cells) {
            std::sort(c.t.begin(), c.t.end());
            std::sort(c.td.begin(), c.td.end());
            for (auto * v : {&c.t1, &c.tg, &c.t2}) std::sort(v->begin(), v->end());
            const double med = c.t[c.t.size() / 2];
            const double bytes = (double) c.k * (2 * GU + DN[c.dt]);
            printf("BENCH spin=%d variant=%d down=%s k=%d n=%zu median_ms=%.4f p10_ms=%.4f p90_ms=%.4f GBps_at_median=%.2f "
                   "device_median_ms=%.4f device_p90_ms=%.4f k1_median_ms=%.4f gap_median_ms=%.4f k2_median_ms=%.4f\n", spin, variant,
                   c.dt ? "Q4_1" : "Q4_0", c.k, c.t.size(), med, c.t[c.t.size() / 10], c.t[c.t.size() * 9 / 10],
                   bytes / (med / 1e3) / 1e9, c.td[c.td.size() / 2], c.td[c.td.size() * 9 / 10], c.t1[c.t1.size() / 2],
                   c.tg[c.tg.size() / 2], c.t2[c.t2.size() / 2]);
        }
        fflush(stdout);
        const gx_stats st = gx_get_stats(g);
        printf("STATS variant=%d dispatches=%llu errors=%llu map_errors=%llu risk_dispatches=%llu risk_slots=%llu experts=%llu\n", variant,
               (unsigned long long) st.dispatches, (unsigned long long) st.errors, (unsigned long long) st.map_errors,
               (unsigned long long) st.risk_dispatches, (unsigned long long) st.risk_slots, (unsigned long long) st.experts);
        for (auto & v : sl) for (auto & s : v) gx_slot_free(g, &s);
        gx_free(g);
    }
    return 0;
}
