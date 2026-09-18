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
    X(clEnqueueMapBuffer) X(clEnqueueUnmapMemObject) X(clWaitForEvents) X(clReleaseEvent) X(clGetEventProfilingInfo)

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
    cl_mem xq = nullptr, offs = nullptr, hq = nullptr, out = nullptr, hdbg = nullptr;
    gx_params p{};
    uint8_t xq_host[(8192 / 32) * 34];
    cl_ulong offs_host[3 * GX_MAX_K];
    int last_k = 0;
    bool pending = false;
    gx_stats st{};
    // pool (gx_pool_create); guarded by mu, as are the pool/map counters in st
    std::mutex mu;
    cl_command_queue q_io = nullptr;
    size_t block_bytes = 0;
    std::vector<PoolBlock> blocks;
    std::map<std::pair<cl_mem, size_t>, size_t> slot_span;   // (block, off_gate) -> bytes
    // timing of the pending / last dispatch
    cl_event ev_first = nullptr, ev_last = nullptr;
    uint64_t t_dispatch = 0, last_device_ns = 0, last_host_ns = 0;
    bool have_last = false;
};

static void seterr(char * err, size_t n, const std::string & s) {
    if (err && n) snprintf(err, n, "%s", s.c_str());
}

extern "C" void gx_free(gx_ctx * g) {
    if (!g) return;
    if (g->q) p_clFinish(g->q);
    for (cl_mem m : {g->xq, g->offs, g->hq, g->out, g->hdbg})
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
    if (p.variant != 0 && p.variant != 1) { seterr(err, errlen, "variant must be 0 or 1"); gx_free(g); return nullptr; }
    const size_t wg_need = p.variant ? 64 : 256;
    g->k1 = p_clCreateKernel(g->prog, p.variant ? "gx_gate_up_row" : "gx_gate_up", &e);
    if (e == CL_SUCCESS) g->k2 = p_clCreateKernel(g->prog, p.variant ? "gx_down_row" : "gx_down", &e);
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
    g->xq = mk(CL_MEM_READ_ONLY, (size_t) p.n_embd / 32 * 34);
    g->offs = mk(CL_MEM_READ_ONLY, sizeof g->offs_host);
    g->hq = mk(CL_MEM_READ_WRITE, (size_t) GX_MAX_K * p.n_ff / 32 * 36);
    g->out = mk(CL_MEM_WRITE_ONLY, (size_t) GX_MAX_K * p.n_embd * sizeof(float));
    g->hdbg = mk(CL_MEM_WRITE_ONLY, (size_t) GX_MAX_K * p.n_ff * sizeof(float));
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
    const int ne = g->p.n_embd, nf = g->p.n_ff;
    g->t_dispatch = now_ns();
    gx_quantize_q8_0(x, g->xq_host, ne);   // x is not read after this
    for (int s = 0; s < k; s++) {
        g->offs_host[3 * s + 0] = slots[s].off_gate;
        g->offs_host[3 * s + 1] = slots[s].off_up;
        g->offs_host[3 * s + 2] = slots[s].off_down;
    }
    cl_int e = p_clEnqueueWriteBuffer(g->q, g->xq, CL_FALSE, 0, (size_t) ne / 32 * 34, g->xq_host, 0, nullptr, nullptr);
    if (e == CL_SUCCESS)
        e = p_clEnqueueWriteBuffer(g->q, g->offs, CL_FALSE, 0, sizeof(cl_ulong) * 3 * k, g->offs_host, 0, nullptr, nullptr);
    const cl_int dbg = g->p.debug_h ? 1 : 0, dt = down_type;
    for (int a = 0; a < 8 && e == CL_SUCCESS; a++) {
        const cl_mem b = slots[a < k ? a : 0].block;
        e = p_clSetKernelArg(g->k1, a, sizeof(cl_mem), &b);
        if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k2, a, sizeof(cl_mem), &b);
    }
    if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k1, 8, sizeof(cl_mem), &g->offs);
    if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k1, 9, sizeof(cl_mem), &g->xq);
    if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k1, 10, sizeof(cl_mem), &g->hq);
    if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k1, 11, sizeof(cl_mem), &g->hdbg);
    if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k1, 12, sizeof(cl_int), &dbg);
    if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k1, 13, sizeof(cl_int), &dt);
    if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k2, 8, sizeof(cl_mem), &g->offs);
    if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k2, 9, sizeof(cl_mem), &g->hq);
    if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k2, 10, sizeof(cl_mem), &g->out);
    if (e == CL_SUCCESS) e = p_clSetKernelArg(g->k2, 11, sizeof(cl_int), &dt);
    // variant 0: 256 work-items = 32 rows x 8 lanes per group; variant 1: 64 work-items = 64 rows per group
    const size_t l[2] = {(size_t) (g->p.variant ? 64 : 256), 1};
    const size_t g1[2] = {g->p.variant ? (size_t) nf : (size_t) 256 * (nf / 32), (size_t) k};
    const size_t g2[2] = {g->p.variant ? (size_t) ne : (size_t) 256 * (ne / 32), (size_t) k};
    cl_event * e1 = g->p.profile ? &g->ev_first : nullptr, * e2 = g->p.profile ? &g->ev_last : nullptr;
    if (e == CL_SUCCESS) e = p_clEnqueueNDRangeKernel(g->q, g->k1, 2, nullptr, g1, l, 0, nullptr, e1);
    if (e == CL_SUCCESS) e = p_clEnqueueNDRangeKernel(g->q, g->k2, 2, nullptr, g2, l, 0, nullptr, e2);
    if (e == CL_SUCCESS)
        e = p_clEnqueueReadBuffer(g->q, g->out, CL_FALSE, 0, sizeof(float) * k * ne, out, 0, nullptr, nullptr);
    if (e == CL_SUCCESS) e = p_clFlush(g->q);
    {
        std::lock_guard<std::mutex> lk(g->mu);
        g->st.dispatches++;
        if (e != CL_SUCCESS) {
            g->st.errors++;
            for (cl_event * ev : {&g->ev_first, &g->ev_last})
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
    const cl_int e = p_clFinish(g->q);
    const uint64_t t1 = now_ns();
    g->pending = false;
    uint64_t dev_ns = 0;
    if (g->ev_first && g->ev_last && e == CL_SUCCESS) {
        cl_ulong a = 0, b = 0;
        if (p_clGetEventProfilingInfo(g->ev_first, CL_PROFILING_COMMAND_START, sizeof a, &a, nullptr) == CL_SUCCESS &&
            p_clGetEventProfilingInfo(g->ev_last, CL_PROFILING_COMMAND_END, sizeof b, &b, nullptr) == CL_SUCCESS && b >= a)
            dev_ns = b - a;
    }
    for (cl_event * ev : {&g->ev_first, &g->ev_last})
        if (*ev) { p_clReleaseEvent(*ev); *ev = nullptr; }
    std::lock_guard<std::mutex> lk(g->mu);
    if (e != CL_SUCCESS) { g->st.errors++; return e; }
    g->last_device_ns = dev_ns;
    g->last_host_ns = t1 - g->t_dispatch;
    g->have_last = true;
    g->st.timed_dispatches++;
    g->st.device_ns += dev_ns;
    g->st.host_ns += g->last_host_ns;
    return e;
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
