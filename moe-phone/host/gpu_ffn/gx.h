/*
 * gx.h — fused MoE expert FFN on an OpenCL GPU, bit-matched to ggml-cpu's ARM path.
 *
 * One call computes, for k experts of one layer and one token (decode, n_tokens = 1):
 *     out[s] = down_s( silu(gate_s(x)) * up_s(x) )          s = 0..k-1
 * with gate/up in Q4_0 [n_ff rows x n_embd] and down in Q4_0 or Q4_1 [n_embd rows x n_ff] (in
 * Qwen3-30B-A3B-Q4_0.gguf down is Q4_1 in layers 0-5 and Q4_0 in layers 6-47), each expert's
 * three projections read in place from the on-flash (GGUF) block layout inside cl_mem blocks the
 * caller owns (e.g. CL_MEM_ALLOC_HOST_PTR slots filled by O_DIRECT reads). No ggml dependency.
 *
 * Numerics (see NOTE.md): the operation sequence of ggml-cpu built for armv8.x+dotprod with NEON
 * (quantize_row_q8_0 -> vec_dot_q4_0_q8_0, ggml_vec_swiglu_f32, then quantize_row_q8_1 ->
 * vec_dot_q4_1_q8_1 or quantize_row_q8_0 -> vec_dot_q4_0_q8_0, nrc = 1) is reproduced operation by operation, including its fused
 * multiply-adds, lane accumulation order and correctly rounded divisions. Whether the result is
 * bit-identical on a given device is measured by gx_test, never assumed.
 *
 * Threading: one gx context is used by one thread at a time. gx_dispatch returns after enqueueing;
 * x may be reused on return; out must stay valid until gx_wait returns.
 */
#ifndef GX_H
#define GX_H

#include <stddef.h>
#include <stdint.h>

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 200
#endif
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS   /* clCreateCommandQueue: the 1.2 entry point Adreno and every 2.x/3.x driver export */
#endif
#include <CL/cl.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GX_MAX_K 8

enum gx_down_type { GX_DOWN_Q4_0 = 0, GX_DOWN_Q4_1 = 1 };   /* the layer's ffn_down_exps tensor type */

typedef struct gx_params {
    int n_embd;     /* 2048 for Qwen3-30B-A3B; multiple of 64 */
    int n_ff;       /* 768; multiple of 64 */
    int debug_h;    /* 1: keep the fp32 SwiGLU output for gx_debug_h (tests only) */
    int variant;    /* work mapping, identical arithmetic: 0 = 8 work-items per row (lane-mapped),
                       1 = one work-item per row (whole-block loads). Both are held to the same bit-exact test. */
} gx_params;

typedef struct gx_slot {
    cl_mem block;       /* buffer holding this expert's projections (may be shared by slots) */
    size_t off_gate;    /* byte offset of the Q4_0 gate slice, n_ff * n_embd/32 * 18 bytes */
    size_t off_up;      /* byte offset of the Q4_0 up slice, same size */
    size_t off_down;    /* byte offset of the down slice, n_embd * n_ff/32 * (18 for Q4_0, 20 for Q4_1) bytes */
} gx_slot;

typedef struct gx_ctx gx_ctx;

/* Returns NULL on failure and writes a reason to err (if non-NULL, errlen bytes). */
gx_ctx * gx_init(cl_context ctx, cl_device_id dev, gx_params p, char * err, size_t errlen);
void     gx_free(gx_ctx * g);

/* Enqueue the FFN of k (1..GX_MAX_K) experts of one layer. down_type: that layer's down tensor type.
 * x: n_embd fp32 (the MoE input, i.e. the normed hidden state ggml passes as src1). out: k * n_embd
 * fp32, row s for slots[s]. Returns 0 or a CL error. */
int gx_dispatch(gx_ctx * g, int layer, int down_type, int k, const gx_slot * slots, const float * x, float * out);

/* Block until the last dispatch has written out. Returns 0 or a CL error. */
int gx_wait(gx_ctx * g);

/* ---- GPU expert pool: GPU-visible memory owned by libgx, so the engine never calls OpenCL itself.
 * Blocks are CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR buffers. A slot is one expert: its gate, up and
 * down slices, contiguous in one block, each slice offset 4096-aligned (O_DIRECT). The CPU writes a slot
 * only between gx_slot_map_write and gx_slot_unmap, and never computes from pool memory.
 *
 * Protocol (the only one verified on the phone, host/gpu_zerocopy/NOTE.md): map a slot's exact byte range
 * with CL_MAP_WRITE_INVALIDATE_REGION, write it (pread/memcpy), unmap. gx_slot_unmap returns only after the
 * unmap has completed, so every gx_dispatch enqueued after it returns sees the new bytes. A slot must not be
 * passed to gx_dispatch while it is mapped, and must not be freed or re-mapped while a dispatch that reads
 * it is pending (gx_wait first). Map/unmap may be called from other threads than gx_dispatch; they use
 * their own command queue and blocking completion.
 *
 * OpenCL leaves open whether a kernel may read one region of a buffer while a different region of the
 * same buffer is mapped. block_bytes == one slot avoids the question entirely (one expert per buffer);
 * gx_pool_create accepts larger blocks, and M6 must test them before the engine relies on them. */

/* Allocate n_blocks blocks of block_bytes. Returns the number actually allocated (0..n_blocks), or -1 on a
 * hard error; writes the driver limits and the reason for any shortfall to err. May be called once. */
int  gx_pool_create(gx_ctx * g, size_t block_bytes, int n_blocks, char * err, size_t errlen);
/* One expert's slot. Returns 1 and fills *out, or 0 if no block has room. Thread-safe. */
int  gx_slot_alloc(gx_ctx * g, size_t bytes_gate, size_t bytes_up, size_t bytes_down, gx_slot * out);
void gx_slot_free(gx_ctx * g, const gx_slot * s);
/* Map the slot's whole byte range for writing; returns the pointer to its gate slice (up and down follow at
 * off_up - off_gate and off_down - off_gate), or NULL on error. Thread-safe. */
void * gx_slot_map_write(gx_ctx * g, const gx_slot * s);
/* Unmap and wait for completion. Returns 0 or a CL error. Thread-safe. */
int  gx_slot_unmap(gx_ctx * g, const gx_slot * s, void * p);

/* Counters for telemetry (CLAUDE.md §6.3). */
typedef struct gx_stats {
    uint64_t dispatches, experts, errors;
    uint64_t pool_blocks, pool_bytes, slots_live, slot_alloc_fail;
    uint64_t maps, unmaps, map_errors, bytes_mapped, map_ns, unmap_ns;
} gx_stats;
gx_stats gx_get_stats(const gx_ctx * g);

/* Tests only (debug_h = 1): after gx_wait, copy the last dispatch's fp32 SwiGLU output, k * n_ff. */
int gx_debug_h(gx_ctx * g, int k, float * h);

/* Host-side activation quantization, identical to ggml-cpu's quantize_row_q8_0 (ARM NEON) output:
 * n/32 blocks of {fp16 d; int8 qs[32]} (34 bytes). Exposed for tests. */
void gx_quantize_q8_0(const float * x, void * y, int n);

#ifdef __cplusplus
}
#endif
#endif
