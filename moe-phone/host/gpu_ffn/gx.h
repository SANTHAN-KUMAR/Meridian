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

/* Counters for telemetry (CLAUDE.md §6.3): dispatches, errors. */
typedef struct gx_stats { uint64_t dispatches, experts, errors; } gx_stats;
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
