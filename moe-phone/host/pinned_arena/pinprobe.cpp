// pinprobe.cpp — can the expert cache live in memory that zram cannot take away, with no CPU cost?
//
// Background (design doc R1, claims arena2_*): the slot arena cut cache management by ~18 ms/token on Qwen3 but raised
// compute by ~14 ms, and the leading explanation is that its anonymous pages are compressed into zram and faulted back
// (22-90 major faults/token measured). mlock is not available to the shell user (RLIMIT_MEMLOCK = 64 KiB). OpenCL
// CL_MEM_ALLOC_HOST_PTR buffers are allocated by the GPU driver (kgsl), not from the process's anonymous LRU, so they
// may be immune to reclaim. This probe measures, on the same phone, for a region of N MiB of each kind:
//   (1) CPU read bandwidth with the engine's compute placement (4 threads on cpu4-7), kgsl-mapped vs malloc
//   (2) swappability: madvise(MADV_PAGEOUT) on the region (the kernel reclaims now: anon pages go to zram), then a
//       re-read; major faults during the re-read and the process's VmSwap tell whether the pages left RAM
//   (3) the kernel's accounting before/after (VmRSS, RssAnon, RssShmem, VmSwap; the smaps entry of the mapping)
// PRE-REGISTERED (2026-09-19 ~00:50, before the first run). The kgsl arena is worth building iff
//   (a) kgsl read GB/s >= 0.95 x malloc read GB/s (median of 5), AND
//   (b) after PAGEOUT the kgsl re-read takes < 1% of the malloc re-read's major faults (and malloc's re-read must show
//       > 1000 major faults, else the control failed and the probe says nothing).
// Written to host/pinned_arena/; built with the NDK and gx's cl_shim.cpp (dlopens the vendor libOpenCL).
//   pinprobe [MiB=768]
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 200
#endif
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifndef MADV_PAGEOUT
#define MADV_PAGEOUT 21
#endif

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static long majflt() { rusage r{}; getrusage(RUSAGE_SELF, &r); return r.ru_majflt; }

static std::string status_line(const char * key) {
    FILE * f = std::fopen("/proc/self/status", "r");
    char line[256];
    std::string out;
    while (f && std::fgets(line, sizeof line, f))
        if (!std::strncmp(line, key, std::strlen(key))) { out = line; out.erase(out.find_last_not_of("\n") + 1); }
    if (f) std::fclose(f);
    return out;
}
static void accounting(const char * tag) {
    std::printf("ACCT %s | %s | %s | %s | %s\n", tag, status_line("VmRSS").c_str(), status_line("RssAnon").c_str(),
                status_line("RssShmem").c_str(), status_line("VmSwap").c_str());
}
// the smaps block of the mapping containing p: header line, Rss, Swap, VmFlags
static void smaps_of(const char * tag, const void * p) {
    FILE * f = std::fopen("/proc/self/smaps", "r");
    char line[512];
    bool in = false;
    const uintptr_t a = (uintptr_t) p;
    while (f && std::fgets(line, sizeof line, f)) {
        unsigned long lo, hi;
        if (std::sscanf(line, "%lx-%lx ", &lo, &hi) == 2 && std::strchr(line, '-') == line + std::strcspn(line, "-")) {
            in = (a >= lo && a < hi);
            if (in) std::printf("SMAPS %s %s", tag, line);
            continue;
        }
        if (in && (!std::strncmp(line, "Rss:", 4) || !std::strncmp(line, "Swap:", 5) || !std::strncmp(line, "VmFlags:", 8) ||
                   !std::strncmp(line, "Anonymous:", 10)))
            std::printf("SMAPS %s %s", tag, line);
    }
    if (f) std::fclose(f);
}

// 4 threads pinned to cpu4-7 sum the regions' 64-bit words; returns GB/s
static double read_bw(const std::vector<std::pair<char *, size_t>> & regs, uint64_t * sink) {
    size_t total = 0;
    for (auto & r : regs) total += r.second;
    std::atomic<uint64_t> acc{0};
    const double t0 = now_s();
    std::vector<std::thread> th;
    for (int t = 0; t < 4; ++t)
        th.emplace_back([&, t] {
            cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(4 + t, &cs); sched_setaffinity(0, sizeof cs, &cs);
            uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (auto & r : regs) {
                const size_t n = r.second / 8, lo = n * t / 4, hi = n * (t + 1) / 4;
                const uint64_t * w = (const uint64_t *) r.first;
                size_t i = lo;
                for (; i + 4 <= hi; i += 4) { s0 += w[i]; s1 += w[i + 1]; s2 += w[i + 2]; s3 += w[i + 3]; }
                for (; i < hi; ++i) s0 += w[i];
            }
            acc += s0 + s1 + s2 + s3;
        });
    for (auto & x : th) x.join();
    const double dt = now_s() - t0;
    *sink += acc.load();
    return (double) total / dt / 1e9;
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char ** argv) {
    const size_t mib = argc > 1 ? (size_t) std::atol(argv[1]) : 768;
    const size_t blk = 64ull << 20, nblk = (mib << 20) / blk;
    uint64_t sink = 0;
    accounting("start");

    cl_platform_id plat[4]; cl_uint np = 0;
    if (clGetPlatformIDs(4, plat, &np) != CL_SUCCESS || !np) { std::printf("FATAL no platform\n"); return 1; }
    cl_device_id dev = nullptr; cl_uint nd = 0;
    for (cl_uint i = 0; i < np && !dev; ++i) if (clGetDeviceIDs(plat[i], CL_DEVICE_TYPE_GPU, 1, &dev, &nd) != CL_SUCCESS) dev = nullptr;
    if (!dev) { std::printf("FATAL no GPU\n"); return 1; }
    cl_int e = 0;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &e);
    if (!ctx || !q) { std::printf("FATAL context/queue %d\n", e); return 1; }

    // kgsl region: nblk buffers, each mapped once for the process lifetime
    std::vector<std::pair<char *, size_t>> kg, ml;
    for (size_t b = 0; b < nblk; ++b) {
        cl_mem m = clCreateBuffer(ctx, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, blk, nullptr, &e);
        if (!m || e != CL_SUCCESS) { std::printf("FATAL clCreateBuffer block %zu: %d\n", b, e); return 1; }
        char * p = (char *) clEnqueueMapBuffer(q, m, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, blk, 0, nullptr, nullptr, &e);
        if (!p || e != CL_SUCCESS) { std::printf("FATAL map block %zu: %d\n", b, e); return 1; }
        kg.push_back({p, blk});
    }
    for (size_t b = 0; b < nblk; ++b) {
        char * p = (char *) aligned_alloc(4096, blk);
        if (!p) { std::printf("FATAL malloc\n"); return 1; }
        ml.push_back({p, blk});
    }
    for (size_t b = 0; b < nblk; ++b) {
        for (size_t i = 0; i < blk; i += 8) { *(uint64_t *) (kg[b].first + i) = i ^ b; *(uint64_t *) (ml[b].first + i) = i ^ b; }
    }
    accounting("after_fill");
    smaps_of("kgsl", kg[0].first);
    smaps_of("malloc", ml[0].first);

    // (1) bandwidth, alternating kgsl/malloc, 5 each, after one warm pass each
    read_bw(kg, &sink); read_bw(ml, &sink);
    std::vector<double> bk, bm;
    for (int r = 0; r < 5; ++r) { bk.push_back(read_bw(kg, &sink)); bm.push_back(read_bw(ml, &sink)); }
    std::printf("BW kgsl_GBps=%.2f malloc_GBps=%.2f ratio=%.3f (medians of 5; runs kgsl:", median(bk), median(bm), median(bk) / median(bm));
    for (double x : bk) std::printf(" %.2f", x);
    std::printf(" malloc:");
    for (double x : bm) std::printf(" %.2f", x);
    std::printf(")\n");

    // (2) swappability: page out each region, then re-read and count major faults
    int rk = 0, rm = 0;
    for (auto & r : kg) rk |= madvise(r.first, r.second, MADV_PAGEOUT);
    for (auto & r : ml) rm |= madvise(r.first, r.second, MADV_PAGEOUT);
    std::printf("PAGEOUT kgsl_rc=%d malloc_rc=%d (errno after last: %d)\n", rk, rm, errno);
    accounting("after_pageout");
    smaps_of("kgsl", kg[0].first);
    smaps_of("malloc", ml[0].first);
    long f0 = majflt(); const double t0 = now_s(); read_bw(kg, &sink); const double tk = now_s() - t0; long fk = majflt() - f0;
    f0 = majflt(); const double t1 = now_s(); read_bw(ml, &sink); const double tm = now_s() - t1; long fm = majflt() - f0;
    std::printf("REREAD kgsl_majflt=%ld kgsl_s=%.3f malloc_majflt=%ld malloc_s=%.3f\n", fk, tk, fm, tm);
    accounting("after_reread");

    const double ratio = median(bk) / median(bm);
    const bool control = fm > 1000;
    const bool a = ratio >= 0.95, b = control && (double) fk < 0.01 * (double) fm;
    std::printf("VERDICT %s (a: bw ratio %.3f >= 0.95 -> %d; b: control malloc majflt %ld > 1000 -> %d, kgsl %ld < 1%% -> %d) sink=%llu\n",
                !control ? "CONTROL_FAILED" : (a && b) ? "BUILD" : "DONT_BUILD", ratio, a, fm, control, fk, b,
                (unsigned long long) (sink & 0xff));
    return 0;
}
