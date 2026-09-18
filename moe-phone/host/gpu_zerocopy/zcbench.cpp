// zcbench — can the Adreno GPU compute streamed Qwen3-30B-A3B experts WITHOUT an upload copy, how fast,
// and does CPU + GPU computing experts at the same time add throughput?
//
// Written 2026-09-18 for moe-phone (see NOTE.md beside this file for the question, the decision rule
// fixed before any phone run, and how to read the output). Standalone: it does not link the engine.
//
//   (a) zero-copy: an expert slice read from flash (O_DIRECT pread) into host memory is handed to OpenCL
//       by each candidate mechanism, and three facts are measured per mechanism: (1) does a map of the
//       cl_mem return the very pointer the bytes were read into; (2) after new bytes are pread into that
//       memory, does the kernel see them with NO enqueue-write (and with only a map/unmap, if needed);
//       (3) the cost of that sync step for one expert vs the whole region — a copy scales with bytes,
//       a zero-copy handoff does not. Mechanisms: copy (device buffer + clEnqueueWriteBuffer, the
//       baseline), use_host_ptr (posix_memalign'd host memory), alloc_host_ptr (driver memory, mapped,
//       O_DIRECT into the mapping), ion_dmabuf (a /dev/dma_heap buffer imported with
//       cl_qcom_ion_host_ptr). Each reports whether it works at all; nothing is assumed.
//   (b) GPU expert GEMV throughput on Qwen3 shapes, n = 1, 8 experts per layer: gate/up q4_0 k=2048
//       n=768 and down q4_1 k=768 n=2048 (the file's own down type), weights in the file's NATIVE
//       block layout (what zero-copy of file bytes gives: 18-byte q4_0 / 20-byte q4_1 blocks). Reported
//       as weight GB/s, per layer both "sync per layer" (clFinish each layer, as an engine must) and
//       "batched" (48 layers enqueued, then one clFinish).
//   (c) CPU (ggml, the engine's own kernels, 4 threads strictly pinned) and GPU computing DISJOINT expert
//       sets at the same time. Every iteration of each worker is time-stamped inside the worker; the
//       overlap fraction is computed from those intervals, and the aggregate GB/s is taken only over
//       the window in which both ran.
//   (d) correctness: every GPU and CPU result is compared with a double-precision scalar reference over
//       the same bytes; a device whose result is wrong has its timings flagged unusable.
//
// Output: `RESULT key=value ...` lines, one per measurement, parsed by analyze.py.
//
// Build: build.sh (NDK for the phone, host clang for a laptop self-test). OpenCL is dlopen'ed so one
// binary works with /vendor/lib64/libOpenCL.so on Android and libOpenCL.so.1 on Linux.

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_ext.h>

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <sched.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>

// ------------------------------------------------------------------ shapes (Qwen3-30B-A3B, from its GGUF)
static const int K_GU = 2048, N_GU = 768; // ffn_gate_exps / ffn_up_exps: q4_0
static const int K_D = 768, N_D = 2048;   // ffn_down_exps: q4_1
static const int TOPK = 8;
static const size_t Q40_BLK = 18, Q41_BLK = 20;
static const size_t GU_BYTES = (size_t) N_GU * (K_GU / 32) * Q40_BLK; // 884736 per expert
static const size_t D_BYTES = (size_t) N_D * (K_D / 32) * Q41_BLK;    // 983040 per expert

static double now_s() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static void pin_self(unsigned mask) {
    cpu_set_t s;
    CPU_ZERO(&s);
    for (int i = 0; i < 32; ++i)
        if (mask >> i & 1) CPU_SET(i, &s);
    sched_setaffinity(0, sizeof(s), &s);
}
static std::string read_first_line(const char * path) {
    FILE * f = fopen(path, "r");
    if (!f) return "NA";
    char b[128] = {0};
    if (!fgets(b, sizeof b, f)) b[0] = 0;
    fclose(f);
    std::string s(b);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s.empty() ? "NA" : s;
}
static void log_clocks(const char * tag) {
    std::string s;
    for (int p : {0, 2, 5, 6, 7}) {
        char a[128], b[128];
        snprintf(a, sizeof a, "/sys/devices/system/cpu/cpufreq/policy%d/scaling_max_freq", p);
        snprintf(b, sizeof b, "/sys/devices/system/cpu/cpufreq/policy%d/scaling_cur_freq", p);
        std::string mx = read_first_line(a);
        if (mx == "NA") continue;
        s += " cap" + std::to_string(p) + "=" + mx + " cur" + std::to_string(p) + "=" + read_first_line(b);
    }
    s += " gpu_cur=" + read_first_line("/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq");
    s += " gpu_max=" + read_first_line("/sys/class/kgsl/kgsl-3d0/devfreq/max_freq");
    s += " gpu_busy=" + read_first_line("/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage");
    printf("CLOCKS tag=%s%s\n", tag, s.c_str());
    fflush(stdout);
}

// Clock sampler: every second, for the whole run, the CPU caps and current frequencies, the GPU's, and
// the phase in progress. Every timing can then be read against the clock it actually ran at, rather
// than against a snapshot before or after it. analyze.py summarises the samples per phase.
static std::atomic<const char *> g_phase{"init"};
static std::atomic<bool> g_sampler_stop{false};
static void clock_sampler() {
    while (!g_sampler_stop.load()) {
        std::string s;
        for (int p : {0, 6}) {
            char a[128], b[128];
            snprintf(a, sizeof a, "/sys/devices/system/cpu/cpufreq/policy%d/scaling_max_freq", p);
            snprintf(b, sizeof b, "/sys/devices/system/cpu/cpufreq/policy%d/scaling_cur_freq", p);
            s += " cap" + std::to_string(p) + "=" + read_first_line(a) + " cur" + std::to_string(p) + "=" + read_first_line(b);
        }
        s += " gpu_cur=" + read_first_line("/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq");
        printf("CLKS t=%.3f phase=%s%s\n", now_s(), g_phase.load(), s.c_str());
        fflush(stdout);
        for (int i = 0; i < 10 && !g_sampler_stop.load(); ++i) usleep(100000);
    }
}

// ------------------------------------------------------------------ fp16
static uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000, mant = x & 0x7fffff;
    int exp = (int) ((x >> 23) & 0xff) - 127 + 15;
    if (exp <= 0) return (uint16_t) sign;
    if (exp >= 31) return (uint16_t) (sign | 0x7c00);
    return (uint16_t) (sign | (exp << 10) | ((mant + 0x1000) >> 13));
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1f, mant = h & 0x3ff, x;
    if (exp == 0)
        x = sign; // flush subnormals: the generator never makes them
    else if (exp == 31)
        x = sign | 0x7f800000 | (mant << 13);
    else
        x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float f;
    memcpy(&f, &x, 4);
    return f;
}

// ------------------------------------------------------------------ synthetic but valid expert weights
// One expert = gate (q4_0) + up (q4_0) + down (q4_1), laid out as three per-projection arrays over all
// experts, exactly as ffn_{gate,up,down}_exps sit in the GGUF. Random nibbles, fp16 scales in a range
// that keeps outputs well inside fp32 without overflow, so a wrong kernel cannot hide behind NaN.
static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static uint32_t rnd() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t) rng_state;
}
static float rndf(float lo, float hi) { return lo + (hi - lo) * (rnd() / 4294967296.0f); }
static void gen_q4_0(uint8_t * p, size_t nblk) {
    for (size_t b = 0; b < nblk; ++b, p += Q40_BLK) {
        uint16_t d = f32_to_f16(rndf(0.002f, 0.02f));
        memcpy(p, &d, 2);
        for (int j = 0; j < 16; ++j) p[2 + j] = (uint8_t) rnd();
    }
}
static void gen_q4_1(uint8_t * p, size_t nblk) {
    for (size_t b = 0; b < nblk; ++b, p += Q41_BLK) {
        uint16_t d = f32_to_f16(rndf(0.002f, 0.02f)), m = f32_to_f16(rndf(-0.12f, -0.01f));
        memcpy(p, &d, 2);
        memcpy(p + 2, &m, 2);
        for (int j = 0; j < 16; ++j) p[4 + j] = (uint8_t) rnd();
    }
}

// scalar reference, double accumulation: y[r] = sum_b sum_j w(r,b,j) * x[b*32+j]
static void ref_q4_0(const uint8_t * W, const float * x, int k, int n, float * y) {
    const int nb = k / 32;
    for (int r = 0; r < n; ++r) {
        double acc = 0;
        for (int b = 0; b < nb; ++b) {
            const uint8_t * p = W + ((size_t) r * nb + b) * Q40_BLK;
            uint16_t dh;
            memcpy(&dh, p, 2);
            const double d = f16_to_f32(dh);
            double s = 0;
            for (int j = 0; j < 16; ++j) {
                s += ((p[2 + j] & 15) - 8) * (double) x[b * 32 + j];
                s += ((p[2 + j] >> 4) - 8) * (double) x[b * 32 + j + 16];
            }
            acc += d * s;
        }
        y[r] = (float) acc;
    }
}
static void ref_q4_1(const uint8_t * W, const float * x, int k, int n, float * y) {
    const int nb = k / 32;
    for (int r = 0; r < n; ++r) {
        double acc = 0;
        for (int b = 0; b < nb; ++b) {
            const uint8_t * p = W + ((size_t) r * nb + b) * Q41_BLK;
            uint16_t dh, mh;
            memcpy(&dh, p, 2);
            memcpy(&mh, p + 2, 2);
            const double d = f16_to_f32(dh), m = f16_to_f32(mh);
            for (int j = 0; j < 16; ++j) {
                acc += (d * (p[4 + j] & 15) + m) * x[b * 32 + j];
                acc += (d * (p[4 + j] >> 4) + m) * x[b * 32 + j + 16];
            }
        }
        y[r] = (float) acc;
    }
}
// max |a - r| / max |r|: the error relative to the output's own scale, so a few near-zero outputs
// cannot blow the ratio up and a wrong kernel cannot hide behind a small absolute error.
static double rel_err(const float * a, const float * r, size_t n) {
    double emax = 0, rmax = 0;
    for (size_t i = 0; i < n; ++i) {
        emax = std::max(emax, (double) fabsf(a[i] - r[i]));
        rmax = std::max(rmax, (double) fabsf(r[i]));
    }
    return rmax > 0 ? emax / rmax : (emax > 0 ? INFINITY : 0);
}

// ------------------------------------------------------------------ OpenCL, dlopen'ed
#define CL_FNS(X)                                                                                            \
    X(clGetPlatformIDs) X(clGetDeviceIDs) X(clGetDeviceInfo) X(clCreateContext) X(clCreateCommandQueue)     \
    X(clCreateProgramWithSource) X(clBuildProgram) X(clGetProgramBuildInfo) X(clCreateKernel)               \
    X(clCreateBuffer) X(clSetKernelArg) X(clEnqueueNDRangeKernel) X(clEnqueueWriteBuffer)                   \
    X(clEnqueueReadBuffer) X(clEnqueueMapBuffer) X(clEnqueueUnmapMemObject) X(clFinish)                     \
    X(clReleaseMemObject) X(clReleaseKernel) X(clReleaseProgram) X(clReleaseCommandQueue)                   \
    X(clReleaseContext) X(clGetPlatformInfo)
#define DECL(f) static decltype(&::f) p_##f = nullptr;
CL_FNS(DECL)
static bool cl_load() {
    const char * names[] = {"libOpenCL.so", "/vendor/lib64/libOpenCL.so", "libOpenCL.so.1", nullptr};
    void * h = nullptr;
    for (int i = 0; names[i] && !h; ++i) h = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        printf("RESULT phase=opencl status=no_library err=%s\n", dlerror());
        return false;
    }
#define LOAD(f)                                                                                              \
    p_##f = (decltype(p_##f)) dlsym(h, #f);                                                                  \
    if (!p_##f) {                                                                                            \
        printf("RESULT phase=opencl status=missing_symbol sym=%s\n", #f);                                    \
        return false;                                                                                        \
    }
    CL_FNS(LOAD)
    return true;
}

static const char * KSRC = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifndef LANES
#define LANES 32
#endif
// One work-group per output row, LANES work-items striding the row's blocks, reduced in local memory.
// get_global_id(1) is the routing slot (0..7); ids[slot] picks the expert. x_stride = 0 broadcasts one
// activation to every slot (gate/up), K for per-slot activations (down), as MUL_MAT_ID does.
#define RED(acc)                                                                        \
    local float red[LANES];                                                             \
    red[lane] = acc;                                                                    \
    barrier(CLK_LOCAL_MEM_FENCE);                                                       \
    for (int s = LANES / 2; s > 0; s >>= 1) {                                           \
        if (lane < s) red[lane] += red[lane + s];                                       \
        barrier(CLK_LOCAL_MEM_FENCE);                                                   \
    }

kernel void gemv_q4_0(global const uchar * W, ulong base, global const int * ids, global const float * x,
                      int x_stride, global float * y, int k, int n) {
    const int lane = get_local_id(0), row = get_group_id(0), slot = get_global_id(1);
    const int nb = k / 32, e = ids[slot];
    global const uchar * rowp = W + base + ((size_t) e * n + row) * (size_t) nb * 18;
    global const float * xs = x + slot * x_stride;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += LANES) {
        global const uchar * p = rowp + b * 18;
        const float d = vload_half(0, (global const half *) p);
        const uchar16 q = vload16(0, p + 2);
        global const float * xb = xs + b * 32;
        const float4 m8 = (float4)(8.0f);
        float s = dot(convert_float4(q.s0123 & (uchar4)(15)) - m8, vload4(0, xb))
                + dot(convert_float4(q.s4567 & (uchar4)(15)) - m8, vload4(1, xb))
                + dot(convert_float4(q.s89ab & (uchar4)(15)) - m8, vload4(2, xb))
                + dot(convert_float4(q.scdef & (uchar4)(15)) - m8, vload4(3, xb))
                + dot(convert_float4(q.s0123 >> (uchar4)(4)) - m8, vload4(4, xb))
                + dot(convert_float4(q.s4567 >> (uchar4)(4)) - m8, vload4(5, xb))
                + dot(convert_float4(q.s89ab >> (uchar4)(4)) - m8, vload4(6, xb))
                + dot(convert_float4(q.scdef >> (uchar4)(4)) - m8, vload4(7, xb));
        acc += d * s;
    }
    RED(acc)
    if (lane == 0) y[slot * n + row] = red[0];
}

kernel void gemv_q4_1(global const uchar * W, ulong base, global const int * ids, global const float * x,
                      int x_stride, global float * y, int k, int n) {
    const int lane = get_local_id(0), row = get_group_id(0), slot = get_global_id(1);
    const int nb = k / 32, e = ids[slot];
    global const uchar * rowp = W + base + ((size_t) e * n + row) * (size_t) nb * 20;
    global const float * xs = x + slot * x_stride;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += LANES) {
        global const uchar * p = rowp + b * 20;
        const float d = vload_half(0, (global const half *) p);
        const float m = vload_half(1, (global const half *) p);
        const uchar16 q = vload16(0, p + 4);
        global const float * xb = xs + b * 32;
        const float4 x0 = vload4(0, xb), x1 = vload4(1, xb), x2 = vload4(2, xb), x3 = vload4(3, xb);
        const float4 x4 = vload4(4, xb), x5 = vload4(5, xb), x6 = vload4(6, xb), x7 = vload4(7, xb);
        float sq = dot(convert_float4(q.s0123 & (uchar4)(15)), x0) + dot(convert_float4(q.s4567 & (uchar4)(15)), x1)
                 + dot(convert_float4(q.s89ab & (uchar4)(15)), x2) + dot(convert_float4(q.scdef & (uchar4)(15)), x3)
                 + dot(convert_float4(q.s0123 >> (uchar4)(4)), x4) + dot(convert_float4(q.s4567 >> (uchar4)(4)), x5)
                 + dot(convert_float4(q.s89ab >> (uchar4)(4)), x6) + dot(convert_float4(q.scdef >> (uchar4)(4)), x7);
        const float4 one = (float4)(1.0f);
        float sx = dot(x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7, one);
        acc += d * sq + m * sx;
    }
    RED(acc)
    if (lane == 0) y[slot * n + row] = red[0];
}
)CLC";

struct Cl {
    cl_platform_id plat = nullptr;
    cl_device_id dev = nullptr;
    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
    std::string exts, name;
    size_t page = 4096, pad = 0;
    bool has_ion = false, has_ext_host = false, has_iocoh = false;
};
static bool cl_init(Cl & c) {
    cl_uint np = 0;
    if (p_clGetPlatformIDs(1, &c.plat, &np) != CL_SUCCESS || np == 0) return false;
    if (p_clGetDeviceIDs(c.plat, CL_DEVICE_TYPE_GPU, 1, &c.dev, nullptr) != CL_SUCCESS) return false;
    char buf[8192] = {0};
    p_clGetDeviceInfo(c.dev, CL_DEVICE_NAME, sizeof buf, buf, nullptr);
    c.name = buf;
    memset(buf, 0, sizeof buf);
    p_clGetDeviceInfo(c.dev, CL_DEVICE_EXTENSIONS, sizeof buf - 1, buf, nullptr);
    c.exts = buf;
    c.has_ion = c.exts.find("cl_qcom_ion_host_ptr") != std::string::npos;
    c.has_ext_host = c.exts.find("cl_qcom_ext_host_ptr") != std::string::npos;
    c.has_iocoh = c.exts.find("cl_qcom_ext_host_ptr_iocoherent") != std::string::npos;
    if (c.has_ext_host) {
        cl_uint v = 0;
        size_t sz = 0;
        if (p_clGetDeviceInfo(c.dev, CL_DEVICE_PAGE_SIZE_QCOM, sizeof v, &v, nullptr) == CL_SUCCESS && v) c.page = v;
        if (p_clGetDeviceInfo(c.dev, CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM, sizeof sz, &sz, nullptr) == CL_SUCCESS) c.pad = sz;
    }
    cl_int err;
    c.ctx = p_clCreateContext(nullptr, 1, &c.dev, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return false;
    c.q = p_clCreateCommandQueue(c.ctx, c.dev, 0, &err);
    if (err != CL_SUCCESS) return false;
    printf("RESULT phase=opencl status=ok device=\"%s\" page=%zu ext_pad=%zu ion_host_ptr=%d ext_host_ptr=%d "
           "iocoherent=%d dmabuf_ext=%d ahb_ext=%d\n",
           c.name.c_str(), c.page, c.pad, c.has_ion, c.has_ext_host, c.has_iocoh,
           c.exts.find("dmabuf") != std::string::npos, c.exts.find("ahardwarebuffer") != std::string::npos);
    return true;
}
static cl_program build_prog(Cl & c, int lanes) {
    cl_int err;
    cl_program p = p_clCreateProgramWithSource(c.ctx, 1, &KSRC, nullptr, &err);
    char opts[64];
    snprintf(opts, sizeof opts, "-DLANES=%d -cl-fast-relaxed-math", lanes);
    if (p_clBuildProgram(p, 1, &c.dev, opts, nullptr, nullptr) != CL_SUCCESS) {
        static char log[16384];
        p_clGetProgramBuildInfo(p, c.dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, nullptr);
        printf("RESULT phase=build lanes=%d status=failed log=\"%.400s\"\n", lanes, log);
        return nullptr;
    }
    return p;
}

// ------------------------------------------------------------------ expert region in host memory
// Three per-projection arrays over E experts: gate[E], up[E], down[E], each expert slice 4 KB aligned so
// an O_DIRECT read can land on it. The file on flash has the same layout.
struct Region {
    int E = 0;
    size_t gu_stride = 0, d_stride = 0; // per-expert stride, 4 KB aligned
    size_t off_gate = 0, off_up = 0, off_down = 0, bytes = 0;
    uint8_t * base = nullptr; // host view (malloc, driver mapping or dmabuf mmap)
};
static size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }
static void region_layout(Region & r, int E) {
    r.E = E;
    r.gu_stride = align_up(GU_BYTES, 4096);
    r.d_stride = align_up(D_BYTES, 4096);
    r.off_gate = 0;
    r.off_up = r.gu_stride * E;
    r.off_down = 2 * r.gu_stride * E;
    r.bytes = r.off_down + r.d_stride * E;
}
// Note: the kernels index experts as e * n * nb * block, i.e. a TIGHT stride. With 4 KB-aligned strides the
// kernel is handed a stride argument instead. To keep the kernel identical to what a zero-copy engine would
// run, the benchmark uses tight strides when the expert bytes are already 4 KB multiples (884736 and 983040
// both are: 216 and 240 pages), which is checked here rather than assumed.
static bool strides_are_tight() { return GU_BYTES % 4096 == 0 && D_BYTES % 4096 == 0; }

static bool make_file(const char * path, int E) {
    Region r;
    region_layout(r, E);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return false;
    std::vector<uint8_t> buf(std::max(r.gu_stride, r.d_stride));
    auto put = [&](size_t n) { return write(fd, buf.data(), n) == (ssize_t) n; };
    for (int e = 0; e < E; ++e) {
        std::fill(buf.begin(), buf.end(), 0);
        gen_q4_0(buf.data(), GU_BYTES / Q40_BLK);
        if (!put(r.gu_stride)) return false;
    }
    for (int e = 0; e < E; ++e) {
        std::fill(buf.begin(), buf.end(), 0);
        gen_q4_0(buf.data(), GU_BYTES / Q40_BLK);
        if (!put(r.gu_stride)) return false;
    }
    for (int e = 0; e < E; ++e) {
        std::fill(buf.begin(), buf.end(), 0);
        gen_q4_1(buf.data(), D_BYTES / Q41_BLK);
        if (!put(r.d_stride)) return false;
    }
    fsync(fd);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
    return true;
}
// Read `len` bytes at file offset `off` into dst: O_DIRECT first; on failure (EINVAL/EFAULT on memory
// the kernel cannot pin) fall back to buffered and SAY so. Returns seconds, or -1 on failure.
struct ReadStat {
    long long direct_ok = 0, direct_fail = 0, buffered = 0;
    int last_errno = 0;
};
static double read_into(const char * path, uint8_t * dst, size_t off, size_t len, ReadStat & st) {
    const double t0 = now_s();
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd >= 0) {
        ssize_t got = pread(fd, dst, len, (off_t) off);
        int e = errno;
        close(fd);
        if (got == (ssize_t) len) {
            ++st.direct_ok;
            return now_s() - t0;
        }
        ++st.direct_fail;
        st.last_errno = e;
    }
    fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    ssize_t got = pread(fd, dst, len, (off_t) off);
    close(fd);
    if (got != (ssize_t) len) return -1;
    ++st.buffered;
    return now_s() - t0;
}

// ------------------------------------------------------------------ GPU side
enum Mech { M_COPY, M_UHP, M_AHP, M_ION, M_N };
static const char * mech_name[M_N] = {"copy", "use_host_ptr", "alloc_host_ptr", "ion_dmabuf"};

struct GpuRegion {
    Mech m;
    Region r;
    cl_mem buf = nullptr;
    uint8_t * host = nullptr; // where bytes are read to (for copy: a staging malloc)
    int dmabuf_fd = -1;
    bool ok = false;
    std::string why;
};
// dma-buf heap allocation (Android). Names tried in order; the first that opens wins.
struct dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};
#define DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0x0, struct dma_heap_allocation_data)
static int dmabuf_alloc(size_t len, std::string & which) {
    const char * heaps[] = {"/dev/dma_heap/qcom,system", "/dev/dma_heap/system", nullptr};
    for (int i = 0; heaps[i]; ++i) {
        int h = open(heaps[i], O_RDONLY | O_CLOEXEC);
        if (h < 0) {
            which += std::string(heaps[i]) + ":open_errno=" + std::to_string(errno) + ";";
            continue;
        }
        dma_heap_allocation_data a{};
        a.len = len;
        a.fd_flags = O_RDWR | O_CLOEXEC;
        int rc = ioctl(h, DMA_HEAP_IOCTL_ALLOC, &a);
        int e = errno;
        close(h);
        if (rc == 0) {
            which += std::string(heaps[i]);
            return (int) a.fd;
        }
        which += std::string(heaps[i]) + ":ioctl_errno=" + std::to_string(e) + ";";
    }
    return -1;
}
static bool gpu_region_create(Cl & c, GpuRegion & g, int E) {
    region_layout(g.r, E);
    const size_t alloc = align_up(g.r.bytes + c.pad, c.page);
    cl_int err = CL_SUCCESS;
    switch (g.m) {
    case M_COPY:
        g.host = (uint8_t *) aligned_alloc(4096, align_up(g.r.bytes, 4096));
        g.buf = p_clCreateBuffer(c.ctx, CL_MEM_READ_ONLY, g.r.bytes, nullptr, &err);
        break;
    case M_UHP:
        g.host = (uint8_t *) aligned_alloc(c.page, alloc);
        g.buf = p_clCreateBuffer(c.ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, g.r.bytes, g.host, &err);
        break;
    case M_AHP: {
        g.buf = p_clCreateBuffer(c.ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, g.r.bytes, nullptr, &err);
        if (err == CL_SUCCESS)
            g.host = (uint8_t *) p_clEnqueueMapBuffer(c.q, g.buf, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, g.r.bytes, 0, nullptr,
                                                      nullptr, &err);
        break;
    }
    case M_ION: {
        if (!c.has_ion || !c.has_ext_host) {
            g.why = "no cl_qcom_ion_host_ptr / cl_qcom_ext_host_ptr";
            return false;
        }
        std::string which;
        g.dmabuf_fd = dmabuf_alloc(alloc, which);
        if (g.dmabuf_fd < 0) {
            g.why = "dma_heap: " + which;
            return false;
        }
        g.host = (uint8_t *) mmap(nullptr, alloc, PROT_READ | PROT_WRITE, MAP_SHARED, g.dmabuf_fd, 0);
        if (g.host == MAP_FAILED) {
            g.host = nullptr;
            g.why = "mmap(dmabuf) errno=" + std::to_string(errno);
            return false;
        }
        cl_mem_ion_host_ptr ion{};
        ion.ext_host_ptr.allocation_type = CL_MEM_ION_HOST_PTR_QCOM;
        ion.ext_host_ptr.host_cache_policy = c.has_iocoh ? CL_MEM_HOST_IOCOHERENT_QCOM : CL_MEM_HOST_WRITEBACK_QCOM;
        ion.ion_filedesc = g.dmabuf_fd;
        ion.ion_hostptr = g.host;
        g.buf = p_clCreateBuffer(c.ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM, g.r.bytes,
                                 &ion, &err);
        g.why = "heap=" + which;
        break;
    }
    default: return false;
    }
    if (err != CL_SUCCESS || !g.buf || !g.host) {
        g.why += " clCreateBuffer/map err=" + std::to_string(err);
        return false;
    }
    g.r.base = g.host;
    g.ok = true;
    return true;
}

static std::vector<float> g_xg, g_xd; // shared activations, filled by make_acts()
struct Kern {
    cl_program prog = nullptr;
    cl_kernel k40 = nullptr, k41 = nullptr;
    int lanes = 32;
    cl_mem ids = nullptr, xg = nullptr, xd = nullptr, yg = nullptr, yu = nullptr, yd = nullptr;
};
static bool kern_init(Cl & c, Kern & K, int lanes) {
    K.lanes = lanes;
    K.prog = build_prog(c, lanes);
    if (!K.prog) return false;
    cl_int err;
    K.k40 = p_clCreateKernel(K.prog, "gemv_q4_0", &err);
    K.k41 = p_clCreateKernel(K.prog, "gemv_q4_1", &err);
    K.ids = p_clCreateBuffer(c.ctx, CL_MEM_READ_ONLY, TOPK * sizeof(int), nullptr, &err);
    K.xg = p_clCreateBuffer(c.ctx, CL_MEM_READ_ONLY, K_GU * sizeof(float), nullptr, &err);
    K.xd = p_clCreateBuffer(c.ctx, CL_MEM_READ_ONLY, TOPK * K_D * sizeof(float), nullptr, &err);
    K.yg = p_clCreateBuffer(c.ctx, CL_MEM_WRITE_ONLY, TOPK * N_GU * sizeof(float), nullptr, &err);
    K.yu = p_clCreateBuffer(c.ctx, CL_MEM_WRITE_ONLY, TOPK * N_GU * sizeof(float), nullptr, &err);
    K.yd = p_clCreateBuffer(c.ctx, CL_MEM_WRITE_ONLY, TOPK * N_D * sizeof(float), nullptr, &err);
    // the activations every check and timing uses (make_acts() runs before any kernel is built)
    if (g_xg.size() != (size_t) K_GU || g_xd.size() != (size_t) TOPK * K_D) return false;
    p_clEnqueueWriteBuffer(c.q, K.xg, CL_TRUE, 0, K_GU * sizeof(float), g_xg.data(), 0, nullptr, nullptr);
    p_clEnqueueWriteBuffer(c.q, K.xd, CL_TRUE, 0, (size_t) TOPK * K_D * sizeof(float), g_xd.data(), 0, nullptr, nullptr);
    return K.k40 && K.k41;
}
// Enqueue one layer: gate and up (q4_0, one broadcast activation) and down (q4_1, one activation per
// routed slot) for the 8 routed experts, over ONE cl_mem holding all three projection arrays; `base` is
// each projection array's byte offset in it (the layout is tight: checked by strides_are_tight()).
static const size_t LAYER_BYTES = (size_t) TOPK * (2 * GU_BYTES + D_BYTES); // weight bytes one layer reads
static void set_args(cl_kernel k, cl_mem W, cl_ulong base, cl_mem ids, cl_mem x, int xs, cl_mem y, int kk, int nn) {
    p_clSetKernelArg(k, 0, sizeof(cl_mem), &W);
    p_clSetKernelArg(k, 1, sizeof(cl_ulong), &base);
    p_clSetKernelArg(k, 2, sizeof(cl_mem), &ids);
    p_clSetKernelArg(k, 3, sizeof(cl_mem), &x);
    p_clSetKernelArg(k, 4, sizeof(int), &xs);
    p_clSetKernelArg(k, 5, sizeof(cl_mem), &y);
    p_clSetKernelArg(k, 6, sizeof(int), &kk);
    p_clSetKernelArg(k, 7, sizeof(int), &nn);
}
static cl_int enqueue_layer(Cl & c, Kern & K, GpuRegion & g, const int * ids) {
    cl_int e = p_clEnqueueWriteBuffer(c.q, K.ids, CL_TRUE, 0, TOPK * sizeof(int), ids, 0, nullptr, nullptr);
    if (g.m == M_COPY) {
        // the copy engine's per-miss cost: upload each routed expert's three slices before computing
        for (int i = 0; i < TOPK; ++i) {
            const size_t og = g.r.off_gate + ids[i] * GU_BYTES, ou = g.r.off_up + ids[i] * GU_BYTES,
                         od = g.r.off_down + ids[i] * D_BYTES;
            p_clEnqueueWriteBuffer(c.q, g.buf, CL_FALSE, og, GU_BYTES, g.host + og, 0, nullptr, nullptr);
            p_clEnqueueWriteBuffer(c.q, g.buf, CL_FALSE, ou, GU_BYTES, g.host + ou, 0, nullptr, nullptr);
            p_clEnqueueWriteBuffer(c.q, g.buf, CL_FALSE, od, D_BYTES, g.host + od, 0, nullptr, nullptr);
        }
    }
    struct L { cl_kernel k; size_t base; cl_mem x; int xs; cl_mem y; int kk, nn; } ls[3] = {
        {K.k40, g.r.off_gate, K.xg, 0, K.yg, K_GU, N_GU},
        {K.k40, g.r.off_up, K.xg, 0, K.yu, K_GU, N_GU},
        {K.k41, g.r.off_down, K.xd, K_D, K.yd, K_D, N_D}};
    for (auto & l : ls) {
        set_args(l.k, g.buf, (cl_ulong) l.base, K.ids, l.x, l.xs, l.y, l.kk, l.nn);
        size_t gws[2] = {(size_t) l.nn * K.lanes, TOPK}, lws[2] = {(size_t) K.lanes, 1};
        cl_int r = p_clEnqueueNDRangeKernel(c.q, l.k, 2, nullptr, gws, lws, 0, nullptr, nullptr);
        if (r != CL_SUCCESS) e = r;
    }
    return e;
}

// Shared activations: fixed, deterministic, in [-1, 1] (declared above kern_init, which uploads them).
static void make_acts() {
    g_xg.resize(K_GU);
    g_xd.resize((size_t) TOPK * K_D);
    for (auto & v : g_xg) v = rndf(-1, 1);
    for (auto & v : g_xd) v = rndf(-1, 1);
}

// Reference outputs for one routing over a host region view.
struct Ref { std::vector<float> g, u, d; };
static Ref reference(const Region & r, const uint8_t * base, const int * ids) {
    Ref R;
    R.g.resize((size_t) TOPK * N_GU);
    R.u.resize((size_t) TOPK * N_GU);
    R.d.resize((size_t) TOPK * N_D);
    for (int i = 0; i < TOPK; ++i) {
        ref_q4_0(base + r.off_gate + ids[i] * GU_BYTES, g_xg.data(), K_GU, N_GU, &R.g[(size_t) i * N_GU]);
        ref_q4_0(base + r.off_up + ids[i] * GU_BYTES, g_xg.data(), K_GU, N_GU, &R.u[(size_t) i * N_GU]);
        ref_q4_1(base + r.off_down + ids[i] * D_BYTES, &g_xd[(size_t) i * K_D], K_D, N_D, &R.d[(size_t) i * N_D]);
    }
    return R;
}
static double gpu_check(Cl & c, Kern & K, GpuRegion & g, const int * ids, const Ref & R) {
    if (enqueue_layer(c, K, g, ids) != CL_SUCCESS) return INFINITY;
    Ref G;
    G.g.resize(R.g.size());
    G.u.resize(R.u.size());
    G.d.resize(R.d.size());
    p_clEnqueueReadBuffer(c.q, K.yg, CL_FALSE, 0, G.g.size() * 4, G.g.data(), 0, nullptr, nullptr);
    p_clEnqueueReadBuffer(c.q, K.yu, CL_FALSE, 0, G.u.size() * 4, G.u.data(), 0, nullptr, nullptr);
    p_clEnqueueReadBuffer(c.q, K.yd, CL_TRUE, 0, G.d.size() * 4, G.d.data(), 0, nullptr, nullptr);
    return std::max({rel_err(G.g.data(), R.g.data(), R.g.size()), rel_err(G.u.data(), R.u.data(), R.u.size()),
                     rel_err(G.d.data(), R.d.data(), R.d.size())});
}

// Fill a region's host view from the file (experts [first, first+E) of each projection array).
static bool fill_from_file(const char * path, int file_E, int first, Region & r, uint8_t * host, ReadStat & st,
                           double & secs) {
    Region f;
    region_layout(f, file_E);
    secs = 0;
    for (int p = 0; p < 3; ++p) {
        const size_t stride = p < 2 ? GU_BYTES : D_BYTES;
        const size_t foff = (p == 0 ? f.off_gate : p == 1 ? f.off_up : f.off_down) + (size_t) first * stride;
        const size_t roff = p == 0 ? r.off_gate : p == 1 ? r.off_up : r.off_down;
        double t = read_into(path, host + roff, foff, stride * r.E, st);
        if (t < 0) return false;
        secs += t;
    }
    return true;
}

// ------------------------------------------------------------------ (a) zero-copy tests
// For one mechanism: fill from flash, check correctness, overwrite 8 experts in place with DIFFERENT
// experts' bytes read from flash, and check whether the kernel sees the new bytes (i) with no sync at all
// and (ii) after a map+unmap of the touched range; time the sync step for one expert slice and for the
// whole region. A copy scales with bytes; a zero-copy handoff does not.
static bool g_zc_pass[M_N] = {false, false, false, false}; // zero_copy && correct, per mechanism
static void zero_copy_test(Cl & c, Kern & K, GpuRegion & g, const char * path, int file_E) {
    const char * nm = mech_name[g.m];
    ReadStat st;
    double rsecs = 0;
    cl_int err = CL_SUCCESS;
    if (!fill_from_file(path, file_E, 0, g.r, g.host, st, rsecs)) {
        printf("RESULT phase=zc mech=%s status=read_failed errno=%d\n", nm, st.last_errno);
        return;
    }
    double t0 = now_s();
    if (g.m == M_COPY) p_clEnqueueWriteBuffer(c.q, g.buf, CL_TRUE, 0, g.r.bytes, g.host, 0, nullptr, nullptr);
    if (g.m == M_AHP) p_clEnqueueUnmapMemObject(c.q, g.buf, g.host, 0, nullptr, nullptr), p_clFinish(c.q);
    const double first_sync = now_s() - t0;
    int ids[TOPK];
    for (int i = 0; i < TOPK; ++i) ids[i] = (i * 5 + 3) % g.r.E;
    // host view for the reference: for AHP the mapping is gone after unmap, so read the reference bytes
    // from a private copy of the file region instead
    std::vector<uint8_t> shadow(g.r.bytes);
    ReadStat st2;
    double s2;
    fill_from_file(path, file_E, 0, g.r, shadow.data(), st2, s2);
    Ref R0 = reference(g.r, shadow.data(), ids);
    const double e0 = gpu_check(c, K, g, ids, R0);

    // pointer identity: map the gate array for READ and compare with the pointer the bytes were read into
    void * mp = p_clEnqueueMapBuffer(c.q, g.buf, CL_TRUE, CL_MAP_READ, 0, 4096, 0, nullptr, nullptr, &err);
    const bool same_ptr = err == CL_SUCCESS && mp == (void *) g.host && g.m != M_COPY && g.m != M_AHP;
    const bool ahp_same = g.m == M_AHP && err == CL_SUCCESS; // AHP: the mapping IS the host view by construction
    if (err == CL_SUCCESS) p_clEnqueueUnmapMemObject(c.q, g.buf, mp, 0, nullptr, nullptr), p_clFinish(c.q);

    // update in place: routed experts' bytes replaced by other experts' bytes from flash
    auto update = [&](uint8_t * view) {
        ReadStat s;
        Region f;
        region_layout(f, file_E);
        for (int i = 0; i < TOPK; ++i) {
            const int src = (ids[i] + g.r.E / 2) % g.r.E;
            read_into(path, view + g.r.off_gate + ids[i] * GU_BYTES, f.off_gate + src * GU_BYTES, GU_BYTES, s);
            read_into(path, view + g.r.off_up + ids[i] * GU_BYTES, f.off_up + src * GU_BYTES, GU_BYTES, s);
            read_into(path, view + g.r.off_down + ids[i] * D_BYTES, f.off_down + src * D_BYTES, D_BYTES, s);
        }
        return s;
    };
    for (int i = 0; i < TOPK; ++i) { // the shadow gets the same update, for the reference
        const int src = (ids[i] + g.r.E / 2) % g.r.E;
        memcpy(shadow.data() + g.r.off_gate + ids[i] * GU_BYTES, shadow.data() + g.r.off_gate + src * GU_BYTES, GU_BYTES);
        memcpy(shadow.data() + g.r.off_up + ids[i] * GU_BYTES, shadow.data() + g.r.off_up + src * GU_BYTES, GU_BYTES);
        memcpy(shadow.data() + g.r.off_down + ids[i] * D_BYTES, shadow.data() + g.r.off_down + src * D_BYTES, D_BYTES);
    }
    Ref R1 = reference(g.r, shadow.data(), ids);

    double e_nosync = NAN, e_mapsync = NAN, e_copy = NAN;
    ReadStat su;
    if (g.m == M_UHP || g.m == M_ION) {
        su = update(g.host); // (i) no sync at all
        e_nosync = gpu_check(c, K, g, ids, R1);
        // (ii) map for write + unmap around the same update (spec-correct protocol)
        void * w = p_clEnqueueMapBuffer(c.q, g.buf, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, g.r.bytes, 0, nullptr, nullptr, &err);
        if (err == CL_SUCCESS) {
            update((uint8_t *) w);
            p_clEnqueueUnmapMemObject(c.q, g.buf, w, 0, nullptr, nullptr);
            p_clFinish(c.q);
            e_mapsync = gpu_check(c, K, g, ids, R1);
        }
    } else if (g.m == M_AHP) {
        void * w = p_clEnqueueMapBuffer(c.q, g.buf, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, g.r.bytes, 0, nullptr, nullptr, &err);
        if (err == CL_SUCCESS) {
            su = update((uint8_t *) w);
            p_clEnqueueUnmapMemObject(c.q, g.buf, w, 0, nullptr, nullptr);
            p_clFinish(c.q);
            e_mapsync = gpu_check(c, K, g, ids, R1);
        }
    } else { // copy: update the staging copy, upload, check
        su = update(g.host);
        p_clEnqueueWriteBuffer(c.q, g.buf, CL_TRUE, 0, g.r.bytes, g.host, 0, nullptr, nullptr);
        e_copy = gpu_check(c, K, g, ids, R1);
    }

    // sync-step cost: one expert slice (gate array, 864 KiB) and the whole region, 5 repeats, min
    auto sync_cost = [&](size_t off, size_t len) -> double {
        double best = INFINITY;
        for (int rep = 0; rep < 5; ++rep) {
            const double a = now_s();
            if (g.m == M_COPY) {
                p_clEnqueueWriteBuffer(c.q, g.buf, CL_TRUE, off, len, g.host + off, 0, nullptr, nullptr);
            } else {
                void * w = p_clEnqueueMapBuffer(c.q, g.buf, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, off, len, 0, nullptr, nullptr, &err);
                if (err != CL_SUCCESS) return NAN;
                p_clEnqueueUnmapMemObject(c.q, g.buf, w, 0, nullptr, nullptr);
                p_clFinish(c.q);
            }
            best = std::min(best, now_s() - a);
        }
        return best;
    };
    const double c1 = sync_cost(g.r.off_gate, GU_BYTES), call = sync_cost(0, g.r.bytes);
    // memcpy reference: what copying the same bytes costs on this CPU
    std::vector<uint8_t> dst(g.r.bytes);
    double mc = INFINITY;
    for (int rep = 0; rep < 3; ++rep) {
        const double a = now_s();
        memcpy(dst.data(), shadow.data(), g.r.bytes);
        mc = std::min(mc, now_s() - a);
    }
    const bool correct = e0 < 1e-3;
    // zero_copy = the bytes read from flash reach the kernel through the same memory (same pointer, the
    // update visible with no sync or with only a map/unmap) AND that sync does not behave like a copy:
    // the whole-region sync must run at >= 10x the memcpy rate of the same bytes (NOTE.md condition 1).
    // A copying driver (e.g. a discrete GPU) passes the first part and fails the second.
    const bool handoff = (g.m == M_UHP || g.m == M_ION) ? (same_ptr && (e_nosync < 1e-3 || e_mapsync < 1e-3))
                                                        : (g.m == M_AHP ? (ahp_same && e_mapsync < 1e-3) : false);
    const bool zero_copy = handoff && std::isfinite(call) && (g.r.bytes / call) >= 10.0 * (g.r.bytes / mc);
    // correct for the verdict = right after the in-place update with the mechanism's own sync (map/unmap
    // for use_host_ptr / dmabuf / alloc_host_ptr); err_initial is also reported but, for use_host_ptr, it
    // was taken without any sync and a copying driver fails it by design.
    const double e_used = (g.m == M_COPY) ? e_copy : e_mapsync;
    g_zc_pass[g.m] = zero_copy && e_used < 1e-3;
    printf("RESULT phase=zc mech=%s status=ok %s correct=%d err_initial=%.3g same_ptr=%d err_update_nosync=%.3g "
           "err_update_mapsync=%.3g err_update_copy=%.3g handoff=%d zero_copy=%d first_sync_ms=%.3f sync_1slice_ms=%.4f "
           "sync_region_ms=%.3f region_MB=%.1f sync_region_GBps=%.2f memcpy_region_GBps=%.2f "
           "odirect_ok=%lld odirect_fail=%lld buffered=%lld last_errno=%d read_GBps=%.2f\n",
           nm, g.why.c_str(), correct, e0, same_ptr || ahp_same, e_nosync, e_mapsync, e_copy, handoff, zero_copy,
           first_sync * 1e3, c1 * 1e3, call * 1e3, g.r.bytes / 1e6, g.r.bytes / call / 1e9, g.r.bytes / mc / 1e9,
           st.direct_ok + su.direct_ok, st.direct_fail + su.direct_fail, st.buffered + su.buffered,
           st.last_errno ? st.last_errno : su.last_errno, g.r.bytes / rsecs / 1e9);
    fflush(stdout);
}

// After the host wrote into a use_host_ptr / dmabuf region directly: map+unmap the whole region, the
// spec-correct way to tell the driver the host changed it (cheap if zero-copy; measured in zero_copy_test).
static void sync_host_write(Cl & c, GpuRegion & g) {
    cl_int err;
    void * w = p_clEnqueueMapBuffer(c.q, g.buf, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, g.r.bytes, 0, nullptr, nullptr, &err);
    if (err == CL_SUCCESS) p_clEnqueueUnmapMemObject(c.q, g.buf, w, 0, nullptr, nullptr);
    p_clFinish(c.q);
}

// ------------------------------------------------------------------ iteration traces
struct Iter { double t0, t1; size_t bytes; };
static void next_ids(int * ids, int E, unsigned & ctr) {
    // 8 distinct experts per layer, rotating through the region so consecutive layers touch different bytes
    for (int i = 0; i < TOPK; ++i) ids[i] = (int) ((ctr * TOPK + i * 7) % E);
    std::sort(ids, ids + TOPK);
    for (int i = 1; i < TOPK; ++i)
        if (ids[i] <= ids[i - 1]) ids[i] = (ids[i - 1] + 1) % E; // keep distinct (E >= 16)
    ++ctr;
}

// GPU loop: sync per layer (clFinish each layer) or batched (48 layers, one clFinish)
static std::vector<Iter> gpu_loop(Cl & c, Kern & K, GpuRegion & g, double secs, int batch, std::atomic<bool> * go,
                                  std::atomic<bool> * stop) {
    std::vector<Iter> it;
    unsigned ctr = 0;
    int ids[TOPK];
    if (go)
        while (!go->load()) {}
    const double end = now_s() + secs;
    while (stop ? !stop->load() : now_s() < end) {
        const double a = now_s();
        for (int l = 0; l < batch; ++l) {
            next_ids(ids, g.r.E, ctr);
            enqueue_layer(c, K, g, ids);
        }
        p_clFinish(c.q);
        it.push_back({a, now_s(), LAYER_BYTES * batch});
        if (!stop && now_s() >= end) break;
    }
    return it;
}

// ------------------------------------------------------------------ CPU side (ggml, the engine's kernels)
struct CpuSide {
    Region r;
    uint8_t * host = nullptr;
    ggml_context *cw = nullptr, *cc = nullptr;
    ggml_tensor *wg, *wu, *wd, *x, *xd, *ids, *g, *u, *d;
    ggml_cgraph * gf = nullptr;
    ggml_threadpool * tp = nullptr;
    ggml_cplan plan{};
    std::vector<uint8_t> work;
};
static bool cpu_init(CpuSide & s, int E, const char * path, int file_E, int first, int n_threads, unsigned mask) {
    region_layout(s.r, E);
    s.host = (uint8_t *) aligned_alloc(4096, align_up(s.r.bytes, 4096));
    ReadStat st;
    double secs;
    if (!fill_from_file(path, file_E, first, s.r, s.host, st, secs)) return false;
    ggml_init_params pw = {ggml_tensor_overhead() * 8, nullptr, true};
    s.cw = ggml_init(pw);
    ggml_init_params pc = {64 * 1024 * 1024, nullptr, false};
    s.cc = ggml_init(pc);
    s.wg = ggml_new_tensor_3d(s.cw, GGML_TYPE_Q4_0, K_GU, N_GU, E);
    s.wu = ggml_new_tensor_3d(s.cw, GGML_TYPE_Q4_0, K_GU, N_GU, E);
    s.wd = ggml_new_tensor_3d(s.cw, GGML_TYPE_Q4_1, K_D, N_D, E);
    if (s.wg->nb[2] != GU_BYTES || s.wd->nb[2] != D_BYTES) {
        printf("RESULT phase=cpu status=layout_mismatch nb2_gu=%zu nb2_d=%zu\n", s.wg->nb[2], s.wd->nb[2]);
        return false;
    }
    s.wg->data = s.host + s.r.off_gate;
    s.wu->data = s.host + s.r.off_up;
    s.wd->data = s.host + s.r.off_down;
    ggml_set_name(s.wg, "ffn_gate_exps");
    ggml_set_name(s.wu, "ffn_up_exps");
    ggml_set_name(s.wd, "ffn_down_exps");
    s.x = ggml_new_tensor_3d(s.cc, GGML_TYPE_F32, K_GU, 1, 1);
    s.xd = ggml_new_tensor_3d(s.cc, GGML_TYPE_F32, K_D, TOPK, 1);
    s.ids = ggml_new_tensor_2d(s.cc, GGML_TYPE_I32, TOPK, 1);
    memcpy(s.x->data, g_xg.data(), K_GU * 4);
    memcpy(s.xd->data, g_xd.data(), (size_t) TOPK * K_D * 4);
    s.g = ggml_mul_mat_id(s.cc, s.wg, s.x, s.ids);
    s.u = ggml_mul_mat_id(s.cc, s.wu, s.x, s.ids);
    s.d = ggml_mul_mat_id(s.cc, s.wd, s.xd, s.ids);
    s.gf = ggml_new_graph(s.cc);
    ggml_build_forward_expand(s.gf, s.g);
    ggml_build_forward_expand(s.gf, s.u);
    ggml_build_forward_expand(s.gf, s.d);
    ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    for (int i = 0; i < 32 && i < GGML_MAX_N_THREADS; ++i) tpp.cpumask[i] = (mask >> i) & 1;
    tpp.strict_cpu = true;
    s.tp = ggml_threadpool_new(&tpp);
    s.plan = ggml_graph_plan(s.gf, n_threads, s.tp);
    s.work.resize(s.plan.work_size + 64);
    s.plan.work_data = s.work.data();
    return true;
}
static double cpu_check(CpuSide & s, const int * ids) {
    memcpy(s.ids->data, ids, TOPK * sizeof(int));
    ggml_graph_compute(s.gf, &s.plan);
    Ref R = reference(s.r, s.host, ids);
    return std::max({rel_err((float *) s.g->data, R.g.data(), R.g.size()),
                     rel_err((float *) s.u->data, R.u.data(), R.u.size()),
                     rel_err((float *) s.d->data, R.d.data(), R.d.size())});
}
static std::vector<Iter> cpu_loop(CpuSide & s, double secs, std::atomic<bool> * go, std::atomic<bool> * stop) {
    std::vector<Iter> it;
    unsigned ctr = 0;
    int ids[TOPK];
    if (go)
        while (!go->load()) {}
    const double end = now_s() + secs;
    while (stop ? !stop->load() : now_s() < end) {
        next_ids(ids, s.r.E, ctr);
        memcpy(s.ids->data, ids, TOPK * sizeof(int));
        const double a = now_s();
        ggml_graph_compute(s.gf, &s.plan);
        it.push_back({a, now_s(), LAYER_BYTES});
        if (!stop && now_s() >= end) break;
    }
    return it;
}

// ------------------------------------------------------------------ throughput summaries
static double gbps(const std::vector<Iter> & it) {
    if (it.empty()) return 0;
    size_t b = 0;
    for (auto & x : it) b += x.bytes;
    return b / (it.back().t1 - it.front().t0) / 1e9;
}
static double busy_union(const std::vector<Iter> & it, double lo, double hi) {
    double s = 0;
    for (auto & x : it) s += std::max(0.0, std::min(x.t1, hi) - std::max(x.t0, lo));
    return s;
}
// time during which BOTH workers had an iteration in flight, inside [lo, hi]
static double both_busy(const std::vector<Iter> & a, const std::vector<Iter> & b, double lo, double hi) {
    double s = 0;
    size_t j = 0;
    for (auto & x : a) {
        const double x0 = std::max(x.t0, lo), x1 = std::min(x.t1, hi);
        if (x1 <= x0) continue;
        while (j < b.size() && b[j].t1 <= x0) ++j;
        for (size_t k = j; k < b.size() && b[k].t0 < x1; ++k)
            s += std::max(0.0, std::min(x1, b[k].t1) - std::max(x0, b[k].t0));
    }
    return s;
}
static size_t bytes_in(const std::vector<Iter> & it, double lo, double hi) {
    size_t b = 0;
    for (auto & x : it)
        if (x.t0 >= lo && x.t1 <= hi) b += x.bytes;
    return b;
}

// ------------------------------------------------------------------ main
int main(int argc, char ** argv) {
    const char * path = "zc_experts.bin";
    int E = 48, threads = 4;
    double T = 8.0;
    unsigned cpu_mask = 0xf0, gpu_host_mask = 0x08;
    bool make = false, skip_cpu = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--file") path = nx();
        else if (a == "--experts") E = atoi(nx());
        else if (a == "--secs") T = atof(nx());
        else if (a == "--threads") threads = atoi(nx());
        else if (a == "--cpu-mask") cpu_mask = (unsigned) strtoul(nx(), nullptr, 16);
        else if (a == "--gpu-host-mask") gpu_host_mask = (unsigned) strtoul(nx(), nullptr, 16);
        else if (a == "--make-file") make = true;
        else if (a == "--skip-cpu") skip_cpu = true;
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (E < 16) { fprintf(stderr, "--experts must be >= 16\n"); return 2; }
    if (!strides_are_tight()) { printf("RESULT phase=layout status=not_tight\n"); return 1; }
    const int file_E = 2 * E; // CPU region: experts [0,E), GPU regions: experts [E,2E)
    printf("RESULT phase=config file=%s experts_per_region=%d secs=%.1f threads=%d cpu_mask=0x%x gpu_host_mask=0x%x "
           "layer_MB=%.3f\n", path, E, T, threads, cpu_mask, gpu_host_mask, LAYER_BYTES / 1e6);
    if (make) {
        const double a = now_s();
        if (!make_file(path, file_E)) { printf("RESULT phase=make_file status=failed errno=%d\n", errno); return 1; }
        printf("RESULT phase=make_file status=ok secs=%.1f\n", now_s() - a);
    }
    make_acts();
    log_clocks("start");
    std::thread sampler(clock_sampler);
    struct SamplerJoin { std::thread & t; ~SamplerJoin() { g_sampler_stop = true; if (t.joinable()) t.join(); } } sj{sampler};
    if (!cl_load()) return 1;
    Cl c;
    if (!cl_init(c)) { printf("RESULT phase=opencl status=init_failed\n"); return 1; }

    // ---- kernel variants (LANES), picked on a use_host_ptr region, correctness required
    GpuRegion probe{M_UHP};
    Region fr;
    region_layout(fr, file_E);
    if (!gpu_region_create(c, probe, E)) {
        probe = GpuRegion{M_COPY};
        gpu_region_create(c, probe, E);
    }
    {
        ReadStat st;
        double s;
        fill_from_file(path, file_E, E, probe.r, probe.host, st, s);
        if (probe.m == M_COPY) p_clEnqueueWriteBuffer(c.q, probe.buf, CL_TRUE, 0, probe.r.bytes, probe.host, 0, nullptr, nullptr);
        else sync_host_write(c, probe);
    }
    int best_lanes = 0;
    double best_g = 0;
    for (int lanes : {16, 32, 64}) {
        Kern K;
        if (!kern_init(c, K, lanes)) continue;
        int ids[TOPK];
        for (int i = 0; i < TOPK; ++i) ids[i] = i * 2 + 1;
        const Ref R = reference(probe.r, probe.host, ids);
        const double e = gpu_check(c, K, probe, ids, R);
        g_phase = lanes == 16 ? "lanes16" : lanes == 32 ? "lanes32" : "lanes64";
        auto it = gpu_loop(c, K, probe, 2.0, 1, nullptr, nullptr);
        g_phase = "setup";
        const double gb = gbps(it);
        printf("RESULT phase=lanes mech=%s lanes=%d max_rel_err=%.3g correct=%d GBps_sync=%.2f\n", mech_name[probe.m],
               lanes, e, e < 1e-3, gb);
        if (e < 1e-3 && gb > best_g) best_g = gb, best_lanes = lanes;
    }
    if (!best_lanes) { printf("RESULT phase=lanes status=no_correct_variant\n"); return 1; }
    Kern K;
    kern_init(c, K, best_lanes);

    // ---- (a) zero-copy per mechanism, then (b) throughput per working mechanism
    std::vector<GpuRegion> regs;
    for (int m = 0; m < M_N; ++m) {
        GpuRegion g{(Mech) m};
        if (!gpu_region_create(c, g, E)) {
            printf("RESULT phase=zc mech=%s status=unavailable why=\"%s\"\n", mech_name[m], g.why.c_str());
            continue;
        }
        g_phase = "zero_copy_test";
        zero_copy_test(c, K, g, path, file_E);
        g_phase = "setup";
        // (re)load the region for throughput: map/read/unmap for AHP, read for the rest (+upload for copy)
        ReadStat st;
        double s;
        if (g.m == M_AHP) {
            cl_int err;
            void * w = p_clEnqueueMapBuffer(c.q, g.buf, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, g.r.bytes, 0, nullptr, nullptr, &err);
            fill_from_file(path, file_E, E, g.r, (uint8_t *) w, st, s);
            p_clEnqueueUnmapMemObject(c.q, g.buf, w, 0, nullptr, nullptr);
            p_clFinish(c.q);
        } else {
            fill_from_file(path, file_E, E, g.r, g.host, st, s);
            if (g.m == M_COPY) p_clEnqueueWriteBuffer(c.q, g.buf, CL_TRUE, 0, g.r.bytes, g.host, 0, nullptr, nullptr);
            else sync_host_write(c, g);
        }
        for (int batch : {1, 48}) {
            log_clocks((std::string(mech_name[m]) + "_b" + std::to_string(batch)).c_str());
            static std::string ph[M_N][2];
            ph[m][batch == 1 ? 0 : 1] = std::string("gpu_alone_") + mech_name[m] + (batch == 1 ? "_sync" : "_batch48");
            g_phase = ph[m][batch == 1 ? 0 : 1].c_str();
            auto it = gpu_loop(c, K, g, T / 2, batch, nullptr, nullptr);
            g_phase = "setup";
            printf("RESULT phase=gpu_alone mech=%s lanes=%d batch=%d iters=%zu GBps=%.2f ms_per_layer=%.3f%s\n",
                   mech_name[m], best_lanes, batch, it.size(), gbps(it),
                   it.empty() ? 0 : (it.back().t1 - it.front().t0) * 1e3 / (it.size() * batch),
                   m == M_COPY ? " note=includes_per_layer_upload_of_8_experts" : "");
        }
        regs.push_back(g);
    }
    if (skip_cpu) return 0;

    // ---- (c) CPU alone, GPU alone, concurrent, CPU alone again (drift bracket)
    CpuSide cs;
    if (!cpu_init(cs, E, path, file_E, 0, threads, cpu_mask)) { printf("RESULT phase=cpu status=init_failed\n"); return 1; }
    {
        int ids[TOPK];
        for (int i = 0; i < TOPK; ++i) ids[i] = i * 3 % E;
        const double e = cpu_check(cs, ids);
        printf("RESULT phase=cpu_check max_rel_err=%.3g note=ggml_quantizes_activations_to_q8\n", e);
    }
    // the GPU arm for (c): a mechanism that PASSED the zero-copy test (zero_copy=1 and correct), in the
    // order alloc_host_ptr, ion_dmabuf, use_host_ptr; else the copy baseline, flagged in the output.
    // (Fixed 2026-09-18 after the first phone run: the previous loop took the first zero-copy-FAMILY
    // mechanism present, use_host_ptr, which had just failed the test on Adreno.)
    GpuRegion * gz = nullptr;
    for (Mech want : {M_AHP, M_ION, M_UHP})
        for (auto & g : regs)
            if (!gz && g.m == want && g_zc_pass[g.m]) gz = &g;
    if (!gz)
        for (auto & g : regs)
            if (!gz && g.m == M_COPY) gz = &g;
    if (!gz && !regs.empty()) gz = &regs[0];
    printf("RESULT phase=concurrent_arm mech=%s passed_zero_copy=%d\n", mech_name[gz->m], (int) g_zc_pass[gz->m]);
    pin_self(cpu_mask);
    log_clocks("cpu_alone_1");
    g_phase = "cpu_alone_1";
    auto ca1 = cpu_loop(cs, T, nullptr, nullptr);
    g_phase = "gpu_alone_for_c";
    printf("RESULT phase=cpu_alone rep=1 iters=%zu GBps=%.2f ms_per_layer=%.3f\n", ca1.size(), gbps(ca1),
           (ca1.back().t1 - ca1.front().t0) * 1e3 / ca1.size());
    std::vector<Iter> ga;
    {
        std::thread t([&] { pin_self(gpu_host_mask); ga = gpu_loop(c, K, *gz, T, 1, nullptr, nullptr); });
        t.join();
    }
    printf("RESULT phase=gpu_alone_for_c mech=%s iters=%zu GBps=%.2f\n", mech_name[gz->m], ga.size(), gbps(ga));
    log_clocks("concurrent");
    g_phase = "concurrent";
    std::atomic<bool> go{false}, stop{false};
    std::vector<Iter> cc, gc;
    std::thread tc([&] { pin_self(cpu_mask); cc = cpu_loop(cs, 0, &go, &stop); });
    std::thread tg([&] { pin_self(gpu_host_mask); gc = gpu_loop(c, K, *gz, 0, 1, &go, &stop); });
    const double t0 = now_s();
    go = true;
    while (now_s() - t0 < T) usleep(10000);
    stop = true;
    tc.join();
    tg.join();
    const double lo = std::max(cc.front().t0, gc.front().t0), hi = std::min(cc.back().t1, gc.back().t1);
    const double win = hi - lo;
    const size_t bc = bytes_in(cc, lo, hi), bg = bytes_in(gc, lo, hi);
    const double cbusy = busy_union(cc, lo, hi), gbusy = busy_union(gc, lo, hi), both = both_busy(cc, gc, lo, hi);
    log_clocks("cpu_alone_2");
    g_phase = "cpu_alone_2";
    auto ca2 = cpu_loop(cs, T, nullptr, nullptr);
    g_phase = "done";
    const double cpu_alone = 0.5 * (gbps(ca1) + gbps(ca2));
    const double agg = (bc + bg) / win / 1e9;
    printf("RESULT phase=concurrent mech=%s window_s=%.2f cpu_GBps=%.2f gpu_GBps=%.2f aggregate_GBps=%.2f "
           "cpu_alone_GBps=%.2f cpu_alone_rep1=%.2f cpu_alone_rep2=%.2f gpu_alone_GBps=%.2f ratio_vs_cpu_alone=%.3f "
           "overlap_frac_of_cpu_busy=%.3f overlap_frac_of_gpu_busy=%.3f cpu_busy_frac=%.3f gpu_busy_frac=%.3f\n",
           mech_name[gz->m], win, bc / win / 1e9, bg / win / 1e9, agg, cpu_alone, gbps(ca1), gbps(ca2), gbps(ga),
           agg / cpu_alone, cbusy > 0 ? both / cbusy : 0, gbusy > 0 ? both / gbusy : 0, cbusy / win, gbusy / win);
    // raw traces for the analysis script
    FILE * f = fopen("zc_concurrent_trace.csv", "w");
    if (f) {
        fprintf(f, "worker,t0,t1,bytes\n");
        for (auto & x : cc) fprintf(f, "cpu,%.6f,%.6f,%zu\n", x.t0, x.t1, x.bytes);
        for (auto & x : gc) fprintf(f, "gpu,%.6f,%.6f,%zu\n", x.t0, x.t1, x.bytes);
        fclose(f);
    }
    log_clocks("end");
    return 0;
}
