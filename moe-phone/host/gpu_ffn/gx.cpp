// gx.cpp — host side of the fused MoE expert FFN (gx.h). OpenCL is resolved with dlopen, so the file links
// into any build with no OpenCL import library; the caller's cl_context and cl_device_id come from the same
// driver. Kernels: gx_kernels.cl, embedded through gx_kernels.inc (build.sh regenerates it).
#include "gx.h"

#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <time.h>

#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace {

const char * k_src =
#ifndef GX_KERNELS_INC
#define GX_KERNELS_INC "gx_kernels.inc"   /* negative_controls.sh builds against mutated copies */
#endif
#include GX_KERNELS_INC
    ;

#define GX_CL_FUNCS(X)                                                                                        \
    X(clGetDeviceInfo) X(clCreateCommandQueue) X(clCreateProgramWithSource) X(clBuildProgram)                 \
    X(clGetProgramBuildInfo) X(clCreateKernel) X(clGetKernelWorkGroupInfo) X(clCreateBuffer) X(clSetKernelArg) \
    X(clEnqueueNDRangeKernel) X(clEnqueueWriteBuffer) X(clEnqueueReadBuffer) X(clFlush) X(clFinish)          \
    X(clReleaseMemObject) X(clReleaseKernel) X(clReleaseProgram) X(clReleaseCommandQueue)                    \
    X(clEnqueueMapBuffer) X(clEnqueueUnmapMemObject) X(clWaitForEvents) X(clReleaseEvent) X(clGetEventProfilingInfo)  \
    X(clGetEventInfo)

#define X(f) decltype(&::f) p_##f = nullptr;
GX_CL_FUNCS(X)
#undef X

bool load_cl(std::string & why) {
    if (p_clFinish) return true;
    const char * libs[] = {"libOpenCL.so", "/vendor/lib64/libOpenCL.so", "/system/vendor/lib64/libOpenCL.so",
                           "libOpenCL.so.1"};
    void * h = nullptr;
    for (const char * l : libs)
        if ((h = dlopen(l, RTLD_NOW | RTLD_LOCAL))) break;
    if (!h) { why = "dlopen libOpenCL failed"; return false; }
#define X(f)                                                                   \
    p_##f = (decltype(p_##f)) dlsym(h, #f);                                    \
    if (!p_##f) { why = std::string("missing symbol ") + #f; return false; }
    GX_CL_FUNCS(X)
#undef X
    return true;
}

// fp32 -> fp16, round to nearest even, with subnormals: what the ARM fcvt does under the default FPCR
// (ggml's GGML_CPU_FP32_TO_FP16 on aarch64). Portable, so the host result is the same on x86 and ARM.
uint16_t f32_to_f16_rne(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t ax = x & 0x7fffffffu;
    if (ax >= 0x7f800000u) return (uint16_t) (sign | 0x7c00u | (ax > 0x7f800000u ? 0x200u : 0u));
    if (ax >= 0x477ff000u) return (uint16_t) (sign | 0x7c00u);   // rounds to >= 65520: inf
    if (ax < 0x38800000u) {                                      // below fp16 normal range
        if (ax < 0x33000000u) return (uint16_t) sign;            // < 2^-25: rounds to 0 (2^-25 exactly ties to 0)
        const uint32_t e = ax >> 23, m = (ax & 0x7fffffu) | 0x800000u;
        const uint32_t s2 = 126 - e;   // value / 2^-24 = m * 2^(e-150) / 2^-24 = m >> (126 - e); 14..24
        uint32_t q = m >> s2;
        const uint32_t rem = m & ((1u << s2) - 1), half = 1u << (s2 - 1);
        if (rem > half || (rem == half && (q & 1u))) q++;
        return (uint16_t) (sign | q);
    }
    uint32_t r = ax - 0x38000000u;   // rebias exponent 127 -> 15, mantissa still 23 bits
    const uint32_t rem = r & 0x1fffu;
    r >>= 13;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) r++;
    return (uint16_t) (sign | r);
}

// Repack one Q4 matrix (rows x nb blocks, GGUF block layout) into variant 2's planes: qs [ib][row] 16 B,
// d [ib][row], and for Q4_1 m [ib][row]. A pure byte permutation (same size).
// plane index of (ib, row): variant 2 [ib][row]; variant 3 tiled [row/64][ib][row%64] (gx_kernels.cl TIL_I)
static inline size_t plane_index(size_t ib, size_t r, size_t rows, size_t nb, int tiled) {
    return tiled ? ((r >> 6) * nb + ib) * 64 + (r & 63) : ib * rows + r;
}

static void repack_matrix(const uint8_t * src, uint8_t * dst, size_t rows, size_t nb, int q41, int tiled) {
    const size_t bs = q41 ? 20 : 18, qo = q41 ? 4 : 2;
    uint8_t * qs = dst, * dp = dst + nb * rows * 16, * mp = dst + nb * rows * 18;
    for (size_t r = 0; r < rows; r++)
        for (size_t ib = 0; ib < nb; ib++) {
            const uint8_t * b = src + (r * nb + ib) * bs;
            const size_t j = plane_index(ib, r, rows, nb, tiled);
            memcpy(qs + j * 16, b + qo, 16);
            memcpy(dp + j * 2, b, 2);
            if (q41) memcpy(mp + j * 2, b + 2, 2);
        }
}

extern "C" void gx_quantize_q8_0(const float * x, void * vy, int n) {
    // quantize_row_q8_0, ggml-cpu/arch/arm/quants.c (the __ARM_NEON branch): amax over |x|; d = amax/127
    // and id = 1/d as IEEE fp32 divisions; y.d = fp16(d); q = round-half-even(x * id).
    uint8_t * y = (uint8_t *) vy;
    for (int b = 0; b < n / 32; b++) {
        const float * xb = x + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(xb[i]));
        volatile float d = amax / 127.0f;   // volatile: keep it an fp32 division on every compiler
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        const uint16_t dh = f32_to_f16_rne(d);
        memcpy(y + b * 34, &dh, 2);
        for (int i = 0; i < 32; i++) {
            volatile float v = xb[i] * id;
            y[b * 34 + 2 + i] = (uint8_t) (int8_t) (int) nearbyintf(v);   // default mode: ties to even
        }
    }
}

uint64_t now_ns() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t) t.tv_sec * 1000000000ull + (uint64_t) t.tv_nsec;
}

struct PoolBlock {
    cl_mem m = nullptr;
    std::map<size_t, size_t> free_ext;   // offset -> bytes, coalesced
};

}  // namespace

struct gx_ctx {
    cl_context ctx = nullptr;
    cl_device_id dev = nullptr;
    cl_command_queue q = nullptr;
    cl_program prog = nullptr;
    cl_kernel k1 = nullptr, k2 = nullptr;
    // packed I/O, layout mirrored from gx_kernels.cl (IN_OFFS_OFF, RISK_WG, OUT_OFF): one write in, one read out
    cl_mem inb = nullptr, outb = nullptr, hq = nullptr, hdbg = nullptr;
    std::vector<uint8_t> in_host, out_host;
    size_t in_offs_off = 0, risk_wg = 0, out_off = 0;
    float * user_out = nullptr;
    cl_event ev_read = nullptr;
    uint32_t last_risk = 0;
    gx_params p{};
    int last_k = 0;
    bool pending = false;
    gx_stats st{};
    // pool (gx_pool_create); guarded by mu, as are the pool/map counters in st
    std::mutex mu;
    cl_command_queue q_io = nullptr;
    size_t block_bytes = 0;
    std::vector<PoolBlock> blocks;
    std::map<std::pair<cl_mem, size_t>, size_t> slot_span;   // (block, off_gate) -> bytes
    std::map<std::pair<cl_mem, size_t>, int> slot_dtype;      // variant 2: down type each slot was repacked with
    // timing of the pending / last dispatch
    cl_event ev_first = nullptr, ev_last = nullptr;
    uint64_t t_dispatch = 0, last_device_ns = 0, last_host_ns = 0, last_k1_ns = 0, last_gap_ns = 0, last_k2_ns = 0;
    bool have_last = false;
};

static void seterr(char * err, size_t n, const std::string & s) {
    if (err && n) snprintf(err, n, "%s", s.c_str());
}

extern "C" void gx_free(gx_ctx * g) {
    if (!g) return;
    if (g->q) p_clFinish(g->q);
    if (g->ev_read) p_clReleaseEvent(g->ev_read);
    for (cl_mem m : {g->inb, g->outb, g->hq, g->hdbg})
        if (m) p_clReleaseMemObject(m);
    for (PoolBlock & b : g->blocks) p_clReleaseMemObject(b.m);
    if (g->q_io) p_clReleaseCommandQueue(g->q_io);
    if (g->k1) p_clReleaseKernel(g->k1);
    if (g->k2) p_clReleaseKernel(g->k2);
    if (g->prog) p_clReleaseProgram(g->prog);
    if (g->q) p_clReleaseCommandQueue(g->q);
    delete g;
}

extern "C" gx_ctx * gx_init(cl_context ctx, cl_device_id dev, gx_params p, char * err, size_t errlen) {
    std::string why;
    if (!load_cl(why)) { seterr(err, errlen, why); return nullptr; }
    if (p.n_embd <= 0 || p.n_ff <= 0 || p.n_embd % 64 || p.n_ff % 64 || p.n_embd > 8192) {
        seterr(err, errlen, "n_embd and n_ff must be positive multiples of 64 (n_embd <= 8192)");
        return nullptr;
    }
    gx_ctx * g = new gx_ctx;
    g->ctx = ctx; g->dev = dev; g->p = p;
    cl_int e;
    g->q = p_clCreateCommandQueue(ctx, dev, p.profile ? CL_QUEUE_PROFILING_ENABLE : 0, &e);
    if (e != CL_SUCCESS) { seterr(err, errlen, "clCreateCommandQueue " + std::to_string(e)); gx_free(g); return nullptr; }
    g->prog = p_clCreateProgramWithSource(ctx, 1, &k_src, nullptr, &e);
    const std::string opts = "-cl-std=CL1.2 -DN_EMBD=" + std::to_string(p.n_embd) + " -DN_FF=" + std::to_string(p.n_ff);
    if (e == CL_SUCCESS) e = p_clBuildProgram(g->prog, 1, &dev, opts.c_str(), nullptr, nullptr);
    if (e != CL_SUCCESS) {
        std::string log(1 << 16, '\0');
        size_t n = 0;
        if (g->prog) p_clGetProgramBuildInfo(g->prog, dev, CL_PROGRAM_BUILD_LOG, log.size() - 1, &log[0], &n);
        log.resize(n ? n - 1 : 0);
        seterr(err, errlen, "build failed " + std::to_string(e) + ": " + log);
        gx_free(g);
        return nullptr;
    }
    if (p.variant < 0 || p.variant > 5) { seterr(err, errlen, "variant must be 0..5"); gx_free(g); return nullptr; }
    if (p.variant == 3 && (p.n_ff % 64 || p.n_embd % 64 || p.n_ff == p.n_embd)) {
        seterr(err, errlen, "variant 3 needs n_ff, n_embd multiples of 64 and n_ff != n_embd"); gx_free(g); return nullptr;
    }
    static const size_t wg_of[6] = {256, 64, 64, 64, 128, 128};
    const size_t wg_need = wg_of[p.variant];
    static const char * kn1[6] = {"gx_gate_up", "gx_gate_up_row", "gx_gate_up_soa", "gx_gate_up_tiled", "gx_gate_up_pair", "gx_gate_up_fast"},
                      * kn2[6] = {"gx_down", "gx_down_row", "gx_down_soa", "gx_down_tiled", "gx_down_pair", "gx_down_fast"};
    g->k1 = p_clCreateKernel(g->prog, kn1[p.variant], &e);
    if (e == CL_SUCCESS) g->k2 = p_clCreateKernel(g->prog, kn2[p.variant], &e);
    for (cl_kernel k : {g->k1, g->k2}) {
        size_t wg = 0;
        if (e == CL_SUCCESS) e = p_clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof wg, &wg, nullptr);
        if (e == CL_SUCCESS && wg < wg_need) {
            seterr(err, errlen, "kernel work-group limit " + std::to_string(wg) + " < " + std::to_string(wg_need));
            gx_free(g);
            return nullptr;
        }
    }
    const cl_mem_flags hv = CL_MEM_ALLOC_HOST_PTR;
    auto mk = [&](cl_mem_flags f, size_t bytes) {
        cl_mem m = nullptr;
        if (e == CL_SUCCESS) m = p_clCreateBuffer(ctx, f | hv, bytes, nullptr, &e);
        return m;
    };
    g->in_offs_off = p.variant == 5 ? (size_t) p.n_embd * 4 : ((size_t) p.n_embd / 32 * 34 + 7) & ~(size_t) 7;   // variant 5: x as fp32
    g->risk_wg = (size_t) p.n_ff / 32;
    g->out_off = GX_MAX_K * g->risk_wg * sizeof(cl_int);
    g->in_host.assign(g->in_offs_off + 3 * GX_MAX_K * sizeof(cl_ulong), 0);
    g->out_host.assign(g->out_off + (size_t) GX_MAX_K * p.n_embd * sizeof(float), 0);
    g->inb = mk(CL_MEM_READ_ONLY, g->in_host.size());
    g->hq = mk(CL_MEM_READ_WRITE, p.variant == 5 ? (size_t) GX_MAX_K * p.n_ff * 4 : (size_t) GX_MAX_K * p.n_ff / 32 * 36);
    g->outb = mk(CL_MEM_READ_WRITE, g->out_host.size());
    g->hdbg = mk(CL_MEM_WRITE_ONLY, (size_t) GX_MAX_K * p.n_ff * sizeof(float));
    // risk flags are written per work-group; variant 1 writes half the entries, so start from zeros once
    if (e == CL_SUCCESS) e = p_clEnqueueWriteBuffer(g->q, g->outb, CL_TRUE, 0, g->out_host.size(), g->out_host.data(), 0, nullptr, nullptr);
    // arguments that never change are set once
    const cl_int dbg = p.debug_h ? 1 : 0;
    if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k1, 8, sizeof(cl_mem), &g->inb);
    if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k1, 9, sizeof(cl_mem), &g->hq);
    if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k1, 10, sizeof(cl_mem), &g->hdbg);
    if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k1, 11, sizeof(cl_int), &dbg);
    if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k1, 13, sizeof(cl_mem), &g->outb);
    if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k2, 8, sizeof(cl_mem), &g->inb);
    if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k2, 9, sizeof(cl_mem), &g->hq);
    if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k2, 10, sizeof(cl_mem), &g->outb);
    if (e != CL_SUCCESS) { seterr(err, errlen, "setup " + std::to_string(e)); gx_free(g); return nullptr; }
    return g;
}

extern "C" int gx_dispatch(gx_ctx * g, int layer, int down_type, int k, const gx_slot * slots, const float * x, float * out) {
    (void) layer;
    if (k < 1 || k > GX_MAX_K || (down_type != GX_DOWN_Q4_0 && down_type != GX_DOWN_Q4_1)) return CL_INVALID_VALUE;
    if (g->pending) {   // the host staging arrays are reused: finish the previous dispatch first
        const int e = gx_wait(g);
        if (e) return e;
    }
    if (gx_variant_uses_repack(g->p.variant)) {   // repacked layouts: refuse a slot not written through gx_repack_expert (or with another down type)
        std::lock_guard<std::mutex> lk(g->mu);
        for (int s = 0; s < k; s++) {
            auto it = g->slot_dtype.find({slots[s].block, slots[s].off_gate});
            if (it == g->slot_dtype.end() || it->second != down_type) { g->st.errors++; g->st.layout_refused++; return CL_INVALID_MEM_OBJECT; }
        }
    }
    const int ne = g->p.n_embd, nf = g->p.n_ff;
    g->t_dispatch = now_ns();
    if (g->p.variant == 5) memcpy(g->in_host.data(), x, (size_t) ne * sizeof(float));   // variant 5: x stays fp32
    else gx_quantize_q8_0(x, g->in_host.data(), ne);   // x is not read after this
    cl_ulong * offs = (cl_ulong *) (g->in_host.data() + g->in_offs_off);
    for (int s = 0; s < k; s++) {
        offs[3 * s + 0] = slots[s].off_gate;
        offs[3 * s + 1] = slots[s].off_up;
        offs[3 * s + 2] = slots[s].off_down;
    }
    const cl_int dt = down_type;
    cl_int e = CL_SUCCESS;
    for (int a = 0; a < 8 && e == CL_SUCCESS; a++) {
        const cl_mem b = slots[a < k ? a : 0].block;
        e = p_clSetKernelArg(g->k1, a, sizeof(cl_mem), &b);
        if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k2, a, sizeof(cl_mem), &b);
    }
    if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k1, 12, sizeof(cl_int), &dt);
    if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k2, 11, sizeof(cl_int), &dt);
    // 1 write (x as Q8_0 + slot offsets), 2 kernels, 1 read (risk flags + out rows): 4 commands per dispatch
    if (e == CL_SUCCESS)
        e = p_clEnqueueWriteBuffer(g->q, g->inb, CL_FALSE, 0, g->in_offs_off + 3 * (size_t) k * sizeof(cl_ulong), g->in_host.data(),
                                   0, nullptr, nullptr);
    // work-items per row: variant 0 = 8 (256-wide groups of 32 rows), variants 1-3 = 1 (64 rows per group),
    // variant 4 = 2 (128-wide groups of 64 rows)
    const int v = g->p.variant;
    const size_t per_row = v == 0 ? 8 : v >= 4 ? 2 : 1;
    const size_t l[2] = {(size_t) (v == 0 ? 256 : v >= 4 ? 128 : 64), 1};
    const size_t g1[2] = {per_row * (size_t) nf, (size_t) k};
    const size_t g2[2] = {per_row * (size_t) ne, (size_t) k};
    cl_event * e1 = g->p.profile ? &g->ev_first : nullptr, * e2 = g->p.profile ? &g->ev_last : nullptr;
    if (e == CL_SUCCESS) e = p_clEnqueueNDRangeKernel(g->q, g->k1, 2, nullptr, g1, l, 0, nullptr, e1);
    if (e == CL_SUCCESS) e = p_clEnqueueNDRangeKernel(g->q, g->k2, 2, nullptr, g2, l, 0, nullptr, e2);
    if (e == CL_SUCCESS)
        e = p_clEnqueueReadBuffer(g->q, g->outb, CL_FALSE, 0, g->out_off + sizeof(float) * (size_t) k * ne, g->out_host.data(), 0,
                                  nullptr, &g->ev_read);
    g->user_out = out;
    if (e == CL_SUCCESS) e = p_clFlush(g->q);
    {
        std::lock_guard<std::mutex> lk(g->mu);
        g->st.dispatches++;
        if (e != CL_SUCCESS) {
            g->st.errors++;
            for (cl_event * ev : {&g->ev_first, &g->ev_last, &g->ev_read})
                if (*ev) { p_clReleaseEvent(*ev); *ev = nullptr; }
            return e;
        }
        g->st.experts += k;
    }
    g->last_k = k;
    g->pending = true;
    return 0;
}

extern "C" int gx_wait(gx_ctx * g) {
    if (!g->pending) return 0;
    cl_int e;
    if (g->p.spin_wait && g->ev_read) {   // poll the last command instead of sleeping in clFinish
        cl_int st = CL_QUEUED;
        do {
            e = p_clGetEventInfo(g->ev_read, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof st, &st, nullptr);
        } while (e == CL_SUCCESS && st > CL_COMPLETE);
        if (e == CL_SUCCESS && st < 0) e = st;
    } else {
        e = p_clFinish(g->q);
    }
    if (e == CL_SUCCESS)
        memcpy(g->user_out, g->out_host.data() + g->out_off, sizeof(float) * (size_t) g->last_k * g->p.n_embd);
    const uint64_t t1 = now_ns();
    g->pending = false;
    uint64_t dev_ns = 0;
    uint64_t k1_ns = 0, gap_ns = 0, k2_ns = 0;
    if (g->ev_first && g->ev_last && e == CL_SUCCESS) {
        cl_ulong a = 0, a1 = 0, b0 = 0, b = 0;   // kernel 1 start/end, kernel 2 start/end
        if (p_clGetEventProfilingInfo(g->ev_first, CL_PROFILING_COMMAND_START, sizeof a, &a, nullptr) == CL_SUCCESS &&
            p_clGetEventProfilingInfo(g->ev_first, CL_PROFILING_COMMAND_END, sizeof a1, &a1, nullptr) == CL_SUCCESS &&
            p_clGetEventProfilingInfo(g->ev_last, CL_PROFILING_COMMAND_START, sizeof b0, &b0, nullptr) == CL_SUCCESS &&
            p_clGetEventProfilingInfo(g->ev_last, CL_PROFILING_COMMAND_END, sizeof b, &b, nullptr) == CL_SUCCESS && b >= a) {
            dev_ns = b - a;
            k1_ns = a1 >= a ? a1 - a : 0;
            gap_ns = b0 >= a1 ? b0 - a1 : 0;
            k2_ns = b >= b0 ? b - b0 : 0;
        }
    }
    for (cl_event * ev : {&g->ev_first, &g->ev_last, &g->ev_read})
        if (*ev) { p_clReleaseEvent(*ev); *ev = nullptr; }
    std::lock_guard<std::mutex> lk(g->mu);
    if (e != CL_SUCCESS) { g->st.errors++; return e; }
    uint32_t mask = 0;
    const cl_int * rk = (const cl_int *) g->out_host.data();
    for (int i = 0; i < g->last_k; i++)
        for (size_t w = 0; w < g->risk_wg; w++)
            if (rk[(size_t) i * g->risk_wg + w]) { mask |= 1u << i; break; }
    g->last_risk = mask;
    if (mask) { g->st.risk_dispatches++; g->st.risk_slots += (uint64_t) __builtin_popcount(mask); }
    g->last_device_ns = dev_ns;
    g->last_k1_ns = k1_ns; g->last_gap_ns = gap_ns; g->last_k2_ns = k2_ns;
    g->last_host_ns = t1 - g->t_dispatch;
    g->have_last = true;
    g->st.timed_dispatches++;
    g->st.device_ns += dev_ns;
    g->st.host_ns += g->last_host_ns;
    return e;
}

extern "C" uint32_t gx_last_risk_mask(const gx_ctx * g) {
    std::lock_guard<std::mutex> lk(const_cast<gx_ctx *>(g)->mu);
    return g->last_risk;
}

extern "C" int gx_last_timing(const gx_ctx * g, uint64_t * device_ns, uint64_t * host_ns) {
    std::lock_guard<std::mutex> lk(const_cast<gx_ctx *>(g)->mu);
    if (!g->have_last) return -1;
    if (device_ns) *device_ns = g->last_device_ns;
    if (host_ns) *host_ns = g->last_host_ns;
    return 0;
}

extern "C" gx_stats gx_get_stats(const gx_ctx * g) {
    std::lock_guard<std::mutex> lk(const_cast<gx_ctx *>(g)->mu);
    return g->st;
}

static size_t align4k(size_t n) { return (n + 4095) & ~(size_t) 4095; }

extern "C" int gx_pool_create(gx_ctx * g, size_t block_bytes, int n_blocks, char * err, size_t errlen) {
    std::lock_guard<std::mutex> lk(g->mu);
    if (!g->blocks.empty() || g->q_io) { seterr(err, errlen, "pool already created"); return -1; }
    cl_ulong max_alloc = 0, global_mem = 0;
    p_clGetDeviceInfo(g->dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof max_alloc, &max_alloc, nullptr);
    p_clGetDeviceInfo(g->dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof global_mem, &global_mem, nullptr);
    const std::string lim = "max_mem_alloc=" + std::to_string(max_alloc) + " global_mem=" + std::to_string(global_mem);
    if (block_bytes == 0 || block_bytes % 4096 || n_blocks < 1 || block_bytes > max_alloc) {
        seterr(err, errlen, "block_bytes must be a nonzero multiple of 4096 and <= max_mem_alloc; " + lim);
        return -1;
    }
    cl_int e;
    g->q_io = p_clCreateCommandQueue(g->ctx, g->dev, 0, &e);
    if (e != CL_SUCCESS) { g->q_io = nullptr; seterr(err, errlen, "io queue " + std::to_string(e)); return -1; }
    g->block_bytes = block_bytes;
    std::string why;
    for (int i = 0; i < n_blocks; i++) {
        cl_mem m = p_clCreateBuffer(g->ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, block_bytes, nullptr, &e);
        if (e != CL_SUCCESS) { why = "clCreateBuffer " + std::to_string(e) + " at block " + std::to_string(i); break; }
        // touch it once through a map so allocation failures surface here, not at the first expert
        void * p = p_clEnqueueMapBuffer(g->q_io, m, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, block_bytes, 0, nullptr, nullptr, &e);
        if (e == CL_SUCCESS) {
            cl_event ev = nullptr;
            e = p_clEnqueueUnmapMemObject(g->q_io, m, p, 0, nullptr, &ev);
            if (e == CL_SUCCESS) e = p_clWaitForEvents(1, &ev);
            if (ev) p_clReleaseEvent(ev);
        }
        if (e != CL_SUCCESS) { p_clReleaseMemObject(m); why = "map/unmap " + std::to_string(e) + " at block " + std::to_string(i); break; }
        PoolBlock b;
        b.m = m;
        b.free_ext[0] = block_bytes;
        g->blocks.push_back(b);
    }
    g->st.pool_blocks = g->blocks.size();
    g->st.pool_bytes = g->blocks.size() * block_bytes;
    seterr(err, errlen, lim + (why.empty() ? "" : "; stopped: " + why));
    return (int) g->blocks.size();
}

extern "C" int gx_slot_alloc(gx_ctx * g, size_t bg, size_t bu, size_t bd, gx_slot * out) {
    const size_t ag = align4k(bg), au = align4k(bu), span = ag + au + align4k(bd);
    std::lock_guard<std::mutex> lk(g->mu);
    for (PoolBlock & b : g->blocks) {
        for (auto it = b.free_ext.begin(); it != b.free_ext.end(); ++it) {
            if (it->second < span) continue;
            const size_t off = it->first, rest = it->second - span;
            b.free_ext.erase(it);
            if (rest) b.free_ext[off + span] = rest;
            *out = gx_slot{b.m, off, off + ag, off + ag + au};
            g->slot_span[{b.m, off}] = span;
            g->st.slots_live++;
            return 1;
        }
    }
    g->st.slot_alloc_fail++;
    return 0;
}

extern "C" void gx_slot_free(gx_ctx * g, const gx_slot * s) {
    std::lock_guard<std::mutex> lk(g->mu);
    auto sp = g->slot_span.find({s->block, s->off_gate});
    if (sp == g->slot_span.end()) { g->st.errors++; return; }
    const size_t span = sp->second;
    g->slot_span.erase(sp);
    g->slot_dtype.erase({s->block, s->off_gate});
    g->st.slots_live--;
    for (PoolBlock & b : g->blocks) {
        if (b.m != s->block) continue;
        size_t off = s->off_gate, len = span;
        auto nx = b.free_ext.lower_bound(off);
        if (nx != b.free_ext.end() && nx->first == off + len) { len += nx->second; nx = b.free_ext.erase(nx); }
        if (nx != b.free_ext.begin()) {
            auto pv = std::prev(nx);
            if (pv->first + pv->second == off) { off = pv->first; len += pv->second; b.free_ext.erase(pv); }
        }
        b.free_ext[off] = len;
        return;
    }
}

extern "C" void * gx_slot_map_write(gx_ctx * g, const gx_slot * s) {
    size_t span;
    {
        std::lock_guard<std::mutex> lk(g->mu);
        auto sp = g->slot_span.find({s->block, s->off_gate});
        if (sp == g->slot_span.end()) { g->st.map_errors++; return nullptr; }
        span = sp->second;
    }
    const uint64_t t0 = now_ns();
    cl_int e;
    void * p = p_clEnqueueMapBuffer(g->q_io, s->block, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, s->off_gate, span, 0, nullptr,
                                    nullptr, &e);
    const uint64_t t1 = now_ns();
    std::lock_guard<std::mutex> lk(g->mu);
    if (e != CL_SUCCESS) { g->st.map_errors++; return nullptr; }
    g->st.maps++;
    g->st.bytes_mapped += span;
    g->st.map_ns += t1 - t0;
    return p;
}

extern "C" const void * gx_slot_map_read(gx_ctx * g, const gx_slot * s) {
    size_t span;
    {
        std::lock_guard<std::mutex> lk(g->mu);
        auto sp = g->slot_span.find({s->block, s->off_gate});
        if (sp == g->slot_span.end()) { g->st.map_errors++; return nullptr; }
        span = sp->second;
    }
    const uint64_t t0 = now_ns();
    cl_int e;
    void * p = p_clEnqueueMapBuffer(g->q_io, s->block, CL_TRUE, CL_MAP_READ, s->off_gate, span, 0, nullptr, nullptr, &e);
    const uint64_t t1 = now_ns();
    std::lock_guard<std::mutex> lk(g->mu);
    if (e != CL_SUCCESS) { g->st.map_errors++; return nullptr; }
    g->st.read_maps++;
    g->st.read_map_ns += t1 - t0;
    return p;
}

extern "C" int gx_slot_unmap_read(gx_ctx * g, const gx_slot * s, const void * p) {
    const uint64_t t0 = now_ns();
    cl_event ev = nullptr;
    cl_int e = p_clEnqueueUnmapMemObject(g->q_io, s->block, const_cast<void *>(p), 0, nullptr, &ev);
    if (e == CL_SUCCESS) e = p_clWaitForEvents(1, &ev);
    if (ev) p_clReleaseEvent(ev);
    const uint64_t t1 = now_ns();
    std::lock_guard<std::mutex> lk(g->mu);
    if (e != CL_SUCCESS) { g->st.map_errors++; return e; }
    g->st.read_map_ns += t1 - t0;
    return 0;
}

extern "C" int gx_slot_unmap(gx_ctx * g, const gx_slot * s, void * p) {
    const uint64_t t0 = now_ns();
    cl_event ev = nullptr;
    cl_int e = p_clEnqueueUnmapMemObject(g->q_io, s->block, p, 0, nullptr, &ev);
    if (e == CL_SUCCESS) e = p_clWaitForEvents(1, &ev);   // completed: later dispatches see the bytes
    if (ev) p_clReleaseEvent(ev);
    const uint64_t t1 = now_ns();
    std::lock_guard<std::mutex> lk(g->mu);
    if (e != CL_SUCCESS) { g->st.map_errors++; return e; }
    g->st.unmaps++;
    g->st.unmap_ns += t1 - t0;
    return 0;
}

extern "C" int gx_debug_h(gx_ctx * g, int k, float * h) {
    if (!g->p.debug_h || k < 1 || k > g->last_k) return CL_INVALID_OPERATION;
    return p_clEnqueueReadBuffer(g->q, g->hdbg, CL_TRUE, 0, sizeof(float) * k * g->p.n_ff, h, 0, nullptr, nullptr);
}

extern "C" int gx_repack_expert(const gx_ctx * g, const gx_slot * s, void * mapped, const void * src_gate, const void * src_up,
                                const void * src_down, int down_type) {
    if (!g || !s || !mapped || (down_type != GX_DOWN_Q4_0 && down_type != GX_DOWN_Q4_1)) return -1;
    if (!gx_variant_uses_repack(g->p.variant)) return -4;   // this context's slots use the native layout
    const size_t ne = (size_t) g->p.n_embd, nf = (size_t) g->p.n_ff;
    uint8_t * base = (uint8_t *) mapped;
    const int tl = g->p.variant == 3;
    repack_matrix((const uint8_t *) src_gate, base, nf, ne / 32, 0, tl);
    repack_matrix((const uint8_t *) src_up, base + (s->off_up - s->off_gate), nf, ne / 32, 0, tl);
    repack_matrix((const uint8_t *) src_down, base + (s->off_down - s->off_gate), ne, nf / 32, down_type == GX_DOWN_Q4_1, tl);
    gx_ctx * gm = const_cast<gx_ctx *>(g);
    std::lock_guard<std::mutex> lk(gm->mu);
    gm->slot_dtype[{s->block, s->off_gate}] = down_type;
    return 0;
}


// Inverse of repack_matrix: planes back to GGUF blocks.
static void unpack_matrix(const uint8_t * src, uint8_t * dst, size_t rows, size_t nb, int q41, int tiled) {
    const size_t bs = q41 ? 20 : 18, qo = q41 ? 4 : 2;
    const uint8_t * qs = src, * dp = src + nb * rows * 16, * mp = src + nb * rows * 18;
    for (size_t r = 0; r < rows; r++)
        for (size_t ib = 0; ib < nb; ib++) {
            uint8_t * b = dst + (r * nb + ib) * bs;
            const size_t j = plane_index(ib, r, rows, nb, tiled);
            memcpy(b + qo, qs + j * 16, 16);
            memcpy(b, dp + j * 2, 2);
            if (q41) memcpy(b + 2, mp + j * 2, 2);
        }
}

extern "C" int gx_unpack_expert(const gx_ctx * g, const gx_slot * s, const void * mapped, void * dst_gate, void * dst_up,
                                void * dst_down, int down_type) {
    if (!g || !s || !mapped || (down_type != GX_DOWN_Q4_0 && down_type != GX_DOWN_Q4_1)) return -1;
    if (!gx_variant_uses_repack(g->p.variant)) return -4;
    {   // refuse a slot that was never repacked here, or was repacked with the other down type (would misread)
        gx_ctx * gm = const_cast<gx_ctx *>(g);
        std::lock_guard<std::mutex> lk(gm->mu);
        auto it = gm->slot_dtype.find({s->block, s->off_gate});
        if (it == gm->slot_dtype.end()) return -3;
        if (it->second != down_type) return -2;
    }
    const size_t ne = (size_t) g->p.n_embd, nf = (size_t) g->p.n_ff;
    const uint8_t * base = (const uint8_t *) mapped;
    const int tl = g->p.variant == 3;
    if (dst_gate) unpack_matrix(base, (uint8_t *) dst_gate, nf, ne / 32, 0, tl);
    if (dst_up) unpack_matrix(base + (s->off_up - s->off_gate), (uint8_t *) dst_up, nf, ne / 32, 0, tl);
    if (dst_down) unpack_matrix(base + (s->off_down - s->off_gate), (uint8_t *) dst_down, ne, nf / 32, down_type == GX_DOWN_Q4_1, tl);
    return 0;
}

extern "C" int gx_last_timing_split(const gx_ctx * g, uint64_t * k1_ns, uint64_t * gap_ns, uint64_t * k2_ns) {
    std::lock_guard<std::mutex> lk(const_cast<gx_ctx *>(g)->mu);
    if (!g->have_last) return -1;
    if (k1_ns) *k1_ns = g->last_k1_ns;
    if (gap_ns) *gap_ns = g->last_gap_ns;
    if (k2_ns) *k2_ns = g->last_k2_ns;
    return 0;
}

extern "C" int gx_variant_uses_repack(int variant) { return variant == 2 || variant == 3; }
