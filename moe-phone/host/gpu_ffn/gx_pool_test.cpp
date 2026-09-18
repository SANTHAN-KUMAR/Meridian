// gx_pool_test.cpp — the gx pool API on a real OpenCL device (laptop: NVIDIA, a copying driver).
//   1. allocator: 20000 random alloc/free ops over two slot sizes (Qwen3's Q4_0-down and Q4_1-down experts);
//      every live slot 4096-aligned per slice, inside its block, disjoint from every other live slot; after
//      freeing everything the pool holds as many slots as when it was new (coalescing works)
//   2. correctness: each real case's experts written into pool slots through map/unmap, dispatched from the
//      pool, compared bit for bit with ggml-cpu ARM (<case>.arm.out)
//   3. concurrency: an I/O thread maps/writes/unmaps other slots in the SAME blocks, nonstop, while the main
//      thread dispatches from resident slots; every dispatch must stay bit-exact
//   4. negative control: overwrite a resident slot with another expert's bytes and dispatch BEFORE unmapping;
//      report whether the kernel saw the old or the new bytes; then unmap and require the new bytes
//   5. READMAP: while a dispatch on the resident slots is in flight (enqueued, not waited), read-map a
//      different slot C, bit-check its bytes, unmap; then wait and bit-check the dispatch (100 rounds)
//   6. x lifetime: x overwritten right after gx_dispatch returns; the output must stay bit-exact
//   7. timing: with profile = 1, every dispatch reports 0 < device_ns <= host_ns
// Two pool shapes: one slot per block, and four slots per block.
//   gx_pool_test <case_dir>
#include "gx.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <random>
#include <string>
#include <thread>
#include <vector>

static const size_t GU = 768 * (2048 / 32) * 18, DN0 = 2048 * (768 / 32) * 18, DN1 = 2048 * (768 / 32) * 20;
static size_t a4k(size_t n) { return (n + 4095) & ~(size_t) 4095; }
static double now_ms() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

struct Case { std::string name; int k = 0, dt = 0; std::vector<int32_t> ids; std::vector<float> x; std::vector<uint8_t> w; std::vector<float> ref; };

static bool load(const std::string & path, Case & c) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char m[4];
    int32_t h[4];
    bool ok = fread(m, 1, 4, f) == 4 && !memcmp(m, "GXC2", 4) && fread(h, 4, 4, f) == 4 && h[0] == 2048 && h[1] == 768;
    if (ok) {
        c.k = h[2]; c.dt = h[3];
        c.ids.resize(c.k); c.x.resize(2048);
        c.w.resize((2 * GU + (c.dt ? DN1 : DN0)) * c.k);
        ok = fread(c.ids.data(), 4, c.k, f) == (size_t) c.k && fread(c.x.data(), 4, 2048, f) == 2048 &&
             fread(c.w.data(), 1, c.w.size(), f) == c.w.size();
    }
    fclose(f);
    if (!ok) return false;
    FILE * r = fopen((path + ".arm.out").c_str(), "rb");
    if (!r) return false;
    int32_t rh[3];
    ok = fread(m, 1, 4, r) == 4 && fread(rh, 4, 3, r) == 3;
    std::vector<float> all((size_t) c.k * (3 * 768 + 2048));
    ok = ok && fread(all.data(), 4, all.size(), r) == all.size();
    fclose(r);
    c.ref.assign(all.begin() + (size_t) c.k * 3 * 768, all.end());   // down rows
    return ok;
}

static const uint8_t * expert_bytes(const Case & c, int e, int part) {   // part 0 gate, 1 up, 2 down
    const size_t per = 2 * GU + (c.dt ? DN1 : DN0);
    return c.w.data() + per * e + (part == 0 ? 0 : part == 1 ? GU : 2 * GU);
}

static bool write_slot(gx_ctx * g, const gx_slot & s, const Case & c, int e) {
    uint8_t * p = (uint8_t *) gx_slot_map_write(g, &s);
    if (!p) return false;
    memcpy(p, expert_bytes(c, e, 0), GU);
    memcpy(p + (s.off_up - s.off_gate), expert_bytes(c, e, 1), GU);
    memcpy(p + (s.off_down - s.off_gate), expert_bytes(c, e, 2), c.dt ? DN1 : DN0);
    return gx_slot_unmap(g, &s, p) == 0;
}

static size_t diff_bits(const float * a, const float * b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; i++) d += memcmp(a + i, b + i, 4) != 0;
    return d;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: gx_pool_test <case_dir>\n"); return 2; }
    const std::string dir = argv[1];
    std::vector<Case> cases;
    if (DIR * d = opendir(dir.c_str())) {
        std::vector<std::string> names;
        while (dirent * de = readdir(d)) {
            std::string n = de->d_name;
            if (n.rfind("qwen3_", 0) == 0 && n.size() > 4 && n.substr(n.size() - 4) == ".bin") names.push_back(n);
        }
        closedir(d);
        std::sort(names.begin(), names.end());
        for (auto & n : names) { Case c; c.name = n; if (load(dir + "/" + n, c)) cases.push_back(std::move(c)); }
    }
    if (cases.empty()) { fprintf(stderr, "no qwen3_*.bin cases with .arm.out in %s\n", dir.c_str()); return 1; }

    cl_platform_id plat;
    cl_device_id dev;
    clGetPlatformIDs(1, &plat, nullptr);
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
    cl_int e;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e);
    int fails = 0;
    const size_t span1 = 2 * a4k(GU) + a4k(DN1);   // the larger (Q4_1-down) slot

    for (int per_block : {1, 4}) {
        char err[1024];
        gx_ctx * g = gx_init(ctx, dev, gx_params{2048, 768, 0, 0, 1}, err, sizeof err);
        if (!g) { fprintf(stderr, "gx_init: %s\n", err); return 1; }
        const int nblk = 48 / per_block;
        const int got = gx_pool_create(g, span1 * per_block, nblk, err, sizeof err);
        printf("POOL per_block=%d blocks=%d/%d block_bytes=%zu :: %s\n", per_block, got, nblk, span1 * per_block, err);
        if (got != nblk) { fails++; gx_free(g); continue; }

        // 1. allocator
        {
            std::mt19937 rng(per_block);
            std::vector<std::pair<gx_slot, size_t>> live;   // slot, its down bytes
            size_t bad = 0, ops = 0;
            auto cap = [&]() {   // how many large slots fit right now
                std::vector<gx_slot> t;
                gx_slot s;
                while (gx_slot_alloc(g, GU, GU, DN1, &s)) t.push_back(s);
                for (auto & x : t) gx_slot_free(g, &x);
                return t.size();
            };
            const size_t cap0 = cap();
            for (int i = 0; i < 20000; i++, ops++) {
                if (live.empty() || (rng() % 3 != 0)) {
                    gx_slot s;
                    const size_t dn = (rng() & 1) ? DN1 : DN0;
                    if (gx_slot_alloc(g, GU, GU, dn, &s)) live.push_back({s, dn});
                } else {
                    const size_t j = rng() % live.size();
                    gx_slot_free(g, &live[j].first);
                    live.erase(live.begin() + j);
                }
                if (i % 997 == 0 || i == 19999) {   // invariants over every live slot
                    std::vector<std::pair<std::pair<cl_mem, size_t>, size_t>> iv;   // (block, start) -> end
                    for (auto & [s, dn] : live) {
                        if (s.off_gate % 4096 || s.off_up % 4096 || s.off_down % 4096) bad++;
                        if (s.off_up < s.off_gate + GU || s.off_down < s.off_up + GU) bad++;
                        const size_t end = s.off_down + a4k(dn);
                        if (end > span1 * per_block) bad++;
                        iv.push_back({{s.block, s.off_gate}, end});
                    }
                    std::sort(iv.begin(), iv.end());
                    for (size_t a = 1; a < iv.size(); a++)
                        if (iv[a].first.first == iv[a - 1].first.first && iv[a].first.second < iv[a - 1].second) bad++;
                }
            }
            for (auto & sd : live) gx_slot_free(g, &sd.first);
            const size_t cap1 = cap();
            printf("ALLOC per_block=%d ops=%zu invariant_violations=%zu capacity_new=%zu capacity_after=%zu\n", per_block, ops,
                   bad, cap0, cap1);
            if (bad || cap0 != cap1 || cap0 != (size_t) nblk * per_block) fails++;
        }

        // 2. correctness from pool slots (+ 6. x lifetime, 7. timing)
        size_t tot = 0, dif = 0, timing_bad = 0;
        for (auto & c : cases) {
            std::vector<gx_slot> sl(c.k);
            for (int s = 0; s < c.k; s++) {
                if (!gx_slot_alloc(g, GU, GU, c.dt ? DN1 : DN0, &sl[s]) || !write_slot(g, sl[s], c, c.ids[s])) { fails++; }
            }
            std::vector<float> out((size_t) c.k * 2048), xs = c.x;
            if (gx_dispatch(g, 0, c.dt, c.k, sl.data(), xs.data(), out.data())) fails++;
            for (auto & v : xs) v = 1e30f;   // 6. x lifetime: gx must not read x after gx_dispatch returns
            if (gx_wait(g)) fails++;
            uint64_t dns = 0, hns = 0;
            if (gx_last_timing(g, &dns, &hns) || dns == 0 || dns > hns) { timing_bad++; }
            tot += out.size();
            dif += diff_bits(out.data(), c.ref.data(), out.size());
            for (auto & s : sl) gx_slot_free(g, &s);
        }
        printf("POOLCASES per_block=%d cases=%zu down_values=%zu diff_bits=%zu (x overwritten after dispatch) timing_bad=%zu\n",
               per_block, cases.size(), tot, dif, timing_bad);
        if (dif || !tot || timing_bad) fails++;

        // 3. concurrency: resident slots for the largest case; an I/O thread churns other slots in the same blocks
        const Case & c = *std::max_element(cases.begin(), cases.end(), [](const Case & a, const Case & b) { return a.k < b.k; });
        std::vector<gx_slot> sl(c.k);
        for (int s = 0; s < c.k; s++)
            if (!gx_slot_alloc(g, GU, GU, c.dt ? DN1 : DN0, &sl[s]) || !write_slot(g, sl[s], c, c.ids[s])) fails++;
        std::atomic<bool> stop{false};
        std::atomic<long> io_ops{0}, io_err{0};
        std::thread io([&]() {
            std::mt19937 rng(7);
            std::vector<uint8_t> junk(DN1);
            for (auto & b : junk) b = (uint8_t) rng();
            while (!stop) {
                gx_slot s;
                if (!gx_slot_alloc(g, GU, GU, DN1, &s)) { std::this_thread::yield(); continue; }
                uint8_t * p = (uint8_t *) gx_slot_map_write(g, &s);
                if (!p) { io_err++; gx_slot_free(g, &s); continue; }
                memcpy(p, junk.data(), GU);
                memcpy(p + (s.off_down - s.off_gate), junk.data(), DN1);
                if (gx_slot_unmap(g, &s, p)) io_err++;
                gx_slot_free(g, &s);
                io_ops++;
            }
        });
        size_t cdif = 0;
        int iters = 0;
        std::vector<float> out((size_t) c.k * 2048);
        for (; iters < 200; iters++) {
            if (gx_dispatch(g, 0, c.dt, c.k, sl.data(), c.x.data(), out.data()) || gx_wait(g)) fails++;
            cdif += diff_bits(out.data(), c.ref.data(), out.size());
        }
        stop = true;
        io.join();
        printf("CONCURRENT per_block=%d case=%s dispatches=%d io_map_write_unmap=%ld io_errors=%ld diff_bits=%zu\n", per_block,
               c.name.c_str(), iters, io_ops.load(), io_err.load(), cdif);
        if (cdif || io_err || io_ops == 0) fails++;

        // 4. negative control: new bytes (the expert of slot 1) into slot 0, dispatch BEFORE unmap
        if (c.k >= 2) {
            uint8_t * p = (uint8_t *) gx_slot_map_write(g, &sl[0]);
            memcpy(p, expert_bytes(c, c.ids[1], 0), GU);
            memcpy(p + (sl[0].off_up - sl[0].off_gate), expert_bytes(c, c.ids[1], 1), GU);
            memcpy(p + (sl[0].off_down - sl[0].off_gate), expert_bytes(c, c.ids[1], 2), c.dt ? DN1 : DN0);
            gx_dispatch(g, 0, c.dt, c.k, sl.data(), c.x.data(), out.data());
            gx_wait(g);
            const bool saw_old = diff_bits(out.data(), c.ref.data(), 2048) == 0;
            const bool saw_new = diff_bits(out.data(), c.ref.data() + 2048, 2048) == 0;
            gx_slot_unmap(g, &sl[0], p);
            gx_dispatch(g, 0, c.dt, c.k, sl.data(), c.x.data(), out.data());
            gx_wait(g);
            const bool after_new = diff_bits(out.data(), c.ref.data() + 2048, 2048) == 0;
            printf("NEGCTRL per_block=%d dispatch_while_mapped: saw_old=%d saw_new=%d (neither=%d) | after_unmap_saw_new=%d\n",
                   per_block, saw_old, saw_new, !saw_old && !saw_new, after_new);
            if (!after_new) fails++;
        }
        // 5. READMAP: slot C (another copy of expert ids[1]) read-mapped while a dispatch on sl is in flight
        {
            gx_slot C;
            size_t rbad = 0, dbad = 0, rounds = 0;
            double map_ms = 0, disp_ms = 0;
            if (!gx_slot_alloc(g, GU, GU, c.dt ? DN1 : DN0, &C) || !write_slot(g, C, c, c.ids[1])) fails++;
            else {
                // restore slot 0 (the negative control left expert ids[1] in it)
                if (!write_slot(g, sl[0], c, c.ids[0])) fails++;
                for (; rounds < 100; rounds++) {
                    const double t0 = now_ms();
                    if (gx_dispatch(g, 0, c.dt, c.k, sl.data(), c.x.data(), out.data())) fails++;
                    const double t1 = now_ms();
                    const uint8_t * p = (const uint8_t *) gx_slot_map_read(g, &C);
                    const double t2 = now_ms();
                    if (!p) { rbad++; continue; }
                    if (memcmp(p, expert_bytes(c, c.ids[1], 0), GU) || memcmp(p + (C.off_up - C.off_gate), expert_bytes(c, c.ids[1], 1), GU) ||
                        memcmp(p + (C.off_down - C.off_gate), expert_bytes(c, c.ids[1], 2), c.dt ? DN1 : DN0)) rbad++;
                    if (gx_slot_unmap_read(g, &C, p)) rbad++;
                    if (gx_wait(g)) fails++;
                    const double t3 = now_ms();
                    dbad += diff_bits(out.data(), c.ref.data(), out.size());
                    map_ms += t2 - t1;
                    disp_ms += t3 - t0;
                }
                gx_slot_free(g, &C);
            }
            printf("READMAP per_block=%d rounds=%zu read_bytes_bad=%zu dispatch_diff_bits=%zu mean_read_map_ms=%.4f mean_dispatch_to_wait_ms=%.4f\n",
                   per_block, rounds, rbad, dbad, rounds ? map_ms / rounds : 0.0, rounds ? disp_ms / rounds : 0.0);
            if (rbad || dbad || rounds != 100) fails++;
        }
        for (auto & s : sl) gx_slot_free(g, &s);
        const gx_stats st = gx_get_stats(g);
        printf("STATS per_block=%d dispatches=%llu errors=%llu maps=%llu unmaps=%llu map_errors=%llu bytes_mapped=%llu "
               "map_ms_mean=%.4f unmap_ms_mean=%.4f read_maps=%llu slots_live=%llu alloc_fail=%llu timed=%llu device_ms_mean=%.4f host_ms_mean=%.4f\n",
               per_block, (unsigned long long) st.dispatches, (unsigned long long) st.errors, (unsigned long long) st.maps,
               (unsigned long long) st.unmaps, (unsigned long long) st.map_errors, (unsigned long long) st.bytes_mapped,
               st.maps ? st.map_ns / 1e6 / st.maps : 0.0, st.unmaps ? st.unmap_ns / 1e6 / st.unmaps : 0.0,
               (unsigned long long) st.read_maps, (unsigned long long) st.slots_live, (unsigned long long) st.slot_alloc_fail,
               (unsigned long long) st.timed_dispatches, st.timed_dispatches ? st.device_ns / 1e6 / st.timed_dispatches : 0.0,
               st.timed_dispatches ? st.host_ns / 1e6 / st.timed_dispatches : 0.0);
        if (st.errors || st.map_errors || st.slots_live) fails++;
        gx_free(g);
    }
    // layout guard: a variant-2/3 context must refuse to dispatch a slot written without gx_repack_expert
    for (int v : {2, 3}) {
        char err[1024];
        gx_ctx * g = gx_init(ctx, dev, gx_params{2048, 768, 0, v, 0, 0}, err, sizeof err);
        const size_t span = 2 * a4k(GU) + a4k(DN1);
        if (!g || gx_pool_create(g, span, 2, err, sizeof err) != 2) { fails++; if (g) gx_free(g); continue; }
        const Case & c = cases[0];
        gx_slot a1, b1;
        gx_slot_alloc(g, GU, GU, c.dt ? DN1 : DN0, &a1);
        gx_slot_alloc(g, GU, GU, c.dt ? DN1 : DN0, &b1);
        write_slot(g, a1, c, c.ids[0]);   // plain memcpy: NOT repacked
        uint8_t * p = (uint8_t *) gx_slot_map_write(g, &b1);
        gx_repack_expert(g, &b1, p, expert_bytes(c, c.ids[0], 0), expert_bytes(c, c.ids[0], 1), expert_bytes(c, c.ids[0], 2), c.dt);
        gx_slot_unmap(g, &b1, p);
        std::vector<float> out(2048);
        const int r_plain = gx_dispatch(g, 0, c.dt, 1, &a1, c.x.data(), out.data());
        if (!r_plain) gx_wait(g);
        const int r_wrongdt = gx_dispatch(g, 0, c.dt ? GX_DOWN_Q4_0 : GX_DOWN_Q4_1, 1, &b1, c.x.data(), out.data());
        if (!r_wrongdt) gx_wait(g);
        const int r_ok = gx_dispatch(g, 0, c.dt, 1, &b1, c.x.data(), out.data());
        const int w_ok = r_ok ? r_ok : gx_wait(g);
        const bool ok_bits = !w_ok && diff_bits(out.data(), c.ref.data(), 2048) == 0;
        printf("LAYOUTGUARD variant=%d plain_slot_refused=%d wrong_down_type_refused=%d repacked_slot_ok=%d refused=%llu\n", v,
               r_plain != 0, r_wrongdt != 0, ok_bits, (unsigned long long) gx_get_stats(g).layout_refused);
        if (!r_plain || !r_wrongdt || !ok_bits) fails++;
        gx_slot_free(g, &a1); gx_slot_free(g, &b1);
        gx_free(g);
    }
    printf("POOL_TEST %s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 3 : 0;
}
