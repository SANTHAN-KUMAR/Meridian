R"GXCL(// gx_kernels.cl — fused MoE expert FFN, bit-matched to ggml-cpu's ARM NEON path (see NOTE.md §Numerics).
// Built by gx.cpp with -DN_EMBD=<n> -DN_FF=<n>. Never build with -cl-mad-enable, -cl-fast-relaxed-math or
// -cl-unsafe-math-optimizations: every multiply-add below is either an explicit fma() (where ggml's ARM
// build emits a fused fmla/fmadd) or a separate multiply and add (where it does not).
#pragma OPENCL FP_CONTRACT OFF

#define NBX (N_EMBD / 32)   // Q4_0 / Q8_0 blocks per gate/up row (64 for Qwen3-30B-A3B)
#define NBH (N_FF / 32)     // Q4_1 / Q8_1 blocks per down row (24)
#define QB0 18              // block_q4_0: fp16 d, 16 bytes of nibbles
#define QB1 20              // block_q4_1: fp16 d, fp16 m, 16 bytes
#define YB0 34              // block_q8_0: fp16 d, int8 qs[32]
#define YB1 36              // block_q8_1: fp16 d, fp16 s, int8 qs[32]
// h (the SwiGLU output) is stored between the kernels at a 36-byte stride for both down types:
// fp16 d at 0, fp16 s at 2 (Q8_1 only), int8 qs[32] at 4. Only the values must match ggml, not the layout.
#define DT_Q4_0 0
#define DT_Q4_1 1

// Correctly rounded a/b. ggml-cpu's divisions are IEEE fdiv; OpenCL's '/' may be off by 2.5 ulp. One
// refinement, then pick the nearest of q and its two neighbours by the exact residual a - q*b (exact
// in fma when q is within an ulp of a/b). A binary quotient is never exactly half-way, so no tie rule
// is needed; the even rule below only makes the code total.
static float cr_div(float a, float b) {
    if (a == 0.0f || b == 0.0f || isinf(a) || isinf(b) || isnan(a) || isnan(b)) return a / b;
    float q = a / b;
    q = fma(fma(-q, b, a), 1.0f / b, q);
    float best = q, rb = fabs(fma(-q, b, a));
    const float qm = nextafter(q, -INFINITY), qp = nextafter(q, INFINITY);
    const float rm = fabs(fma(-qm, b, a)), rp = fabs(fma(-qp, b, a));
    if (rm < rb || (rm == rb && (as_uint(qm) & 1u) == 0u)) { best = qm; rb = rm; }
    if (rp < rb || (rp == rb && (as_uint(qp) & 1u) == 0u)) { best = qp; rb = rp; }
    return best;
}

// ggml_v_expf, ggml-cpu/vec.h (the __ARM_NEON branch), one lane. vfmaq_f32(a,b,c) = fma(b,c,a);
// vfmsq_f32(a,b,c) = fma(-b,c,a). The vector version branches on "any lane |n| > 126"; lanes with
// |n| <= 126 get fma(j,k,k) either way (vfmaq_f32(k,j,k) and vfmaq_f32(k,k,j) are the same value).
static float gx_expf(float x) {
    const float r = 0x1.8p23f;
    const float z = fma(x, 0x1.715476p+0f, r);
    const float n = z - r;
    const float b = fma(-n, 0x1.7f7d1cp-20f, fma(-n, 0x1.62e4p-1f, x));
    const uint  e = as_uint(z) << 23;
    const float k = as_float(e + as_uint(1.0f));
    const float u = b * b;
    const float j = fma(fma(fma(0x1.0e4020p-7f, b, 0x1.573e2ep-5f), u, fma(0x1.555e66p-3f, b, 0x1.fffdb6p-2f)),
                        u, 0x1.ffffecp-1f * b);
    if (!(fabs(n) > 126.0f)) return fma(j, k, k);
    const uint  d  = (n <= 0.0f) ? 0x82000000u : 0u;
    const float s1 = as_float(d + 0x7f000000u);
    const float s2 = as_float(e - d);
    if (fabs(n) > 192.0f) return s1 * s1;
    return fma(s2, j, s2) * s1;
}

// ggml_v_silu: x / (1 + expf(0 - x)), then ggml_vec_swiglu_f32 multiplies by the up value.
static float gx_swiglu(float g, float u) {
    const float e = gx_expf(0.0f - g);
    return cr_div(g, 1.0f + e) * u;
}

// quantize_row_q8_0 / quantize_row_q8_1, ggml-cpu/arch/arm/quants.c (__ARM_NEON branch), one 32-value
// block, written as fp16 d at 0, fp16 s at 2 (with_s only), int8 qs at 4:
//   amax = max|v| (order-free); d = amax / 127 and id = 1 / d, both IEEE fp32 divisions (not 127 / amax);
//   y.d = fp16_rne(d); q = vcvtnq_s32_f32(v * id), i.e. round to nearest, ties to even;
//   y.s = fp16_rne(d * (float) sum(q)), with the fp32 d, not the stored fp16 one.
static void q8_block(local const float * v, global uchar * yo, int with_s) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) amax = fmax(amax, fabs(v[i]));
    const float d = cr_div(amax, 127.0f);
    const float id = d != 0.0f ? cr_div(1.0f, d) : 0.0f;
    vstore_half_rte(d, 0, (global half *) yo);
    int sum = 0;
    for (int i = 0; i < 32; i++) {
        const int q = convert_int_rte(v[i] * id);
        yo[4 + i] = (uchar) (char) q;
        sum += q;
    }
    if (with_s) vstore_half_rte(d * (float) sum, 1, (global half *) yo);
}

static global const uchar * pick(int s, global const uchar * w0, global const uchar * w1, global const uchar * w2,
                                 global const uchar * w3, global const uchar * w4, global const uchar * w5,
                                 global const uchar * w6, global const uchar * w7) {
    switch (s) {
        case 0: return w0; case 1: return w1; case 2: return w2; case 3: return w3;
        case 4: return w4; case 5: return w5; case 6: return w6; default: return w7;
    }
}

// Integer part of one NEON sdot lane pair: lane j of the low-nibble dot plus lane j of the high-nibble
// dot (ggml_vdotq_s32 lane j sums bytes 4j..4j+3). Exact in int.
static int lane_dot(uchar4 q, int off, char4 yl, char4 yh) {
    const int4 lo = convert_int4(q & (uchar4)(15)) - off;
    const int4 hi = convert_int4(q >> (uchar4)(4)) - off;
    const int4 a = convert_int4(yl), c = convert_int4(yh);
    return lo.x*a.x + lo.y*a.y + lo.z*a.z + lo.w*a.w + hi.x*c.x + hi.y*c.y + hi.z*c.z + hi.w*c.w;
}

// Horizontal sum in the order of vaddvq_f32(sumv0) + vaddvq_f32(sumv1): faddp, faddp, fadd.
static float hsum8(local const float * l) {
    return ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]));
}

// Kernel 1: gate and up rows (Q4_0 x Q8_0), SwiGLU, and the quantization ggml applies to h before the
// down MUL_MAT_ID: quantize_row_q8_1 for a Q4_1 down (vec_dot_type Q8_1), quantize_row_q8_0 for a Q4_0
// down (Q8_0). The two differ only in the s field.
// Work-group = 256 = 32 rows (one Q8_1 block of h) x 8 work-items. Work-item w of a row plays NEON
// accumulator (w >> 2) (0: even blocks, 1: odd) lane (w & 3), accumulating in block order exactly as
// sumv0/sumv1 do. Grid: (256 * NBH, k).
kernel void gx_gate_up(global const uchar * w0, global const uchar * w1, global const uchar * w2, global const uchar * w3,
                       global const uchar * w4, global const uchar * w5, global const uchar * w6, global const uchar * w7,
                       global const ulong * offs, global const uchar * xq, global uchar * hq,
                       global float * hdbg, int dbg, int dt) {
    local float lg[256], lu[256], lh[32];
    const int t = get_local_id(0), blk = get_group_id(0), s = get_group_id(1);
    const int rr = t >> 3, w = t & 7, p = w >> 2, j = w & 3;
    const int row = blk * 32 + rr;
    global const uchar * W = pick(s, w0, w1, w2, w3, w4, w5, w6, w7);
    global const uchar * G = W + offs[s * 3 + 0] + (size_t) row * NBX * QB0;
    global const uchar * U = W + offs[s * 3 + 1] + (size_t) row * NBX * QB0;
    float ag = 0.0f, au = 0.0f;
    for (int ib = p; ib < NBX; ib += 2) {
        global const uchar * yb = xq + ib * YB0;
        const float dy = vload_half(0, (global const half *) yb);
        const char4 yl = vload4(0, (global const char *) (yb + 2 + 4 * j));
        const char4 yh = vload4(0, (global const char *) (yb + 2 + 16 + 4 * j));
        global const uchar * gb = G + ib * QB0;
        global const uchar * ub = U + ib * QB0;
        const int ig = lane_dot(vload4(0, gb + 2 + 4 * j), 8, yl, yh);
        const int iu = lane_dot(vload4(0, ub + 2 + 4 * j), 8, yl, yh);
        const float sg = vload_half(0, (global const half *) gb) * dy;   // fmul, then fused fmla
        const float su = vload_half(0, (global const half *) ub) * dy;
        ag = fma((float) ig, sg, ag);
        au = fma((float) iu, su, au);
    }
    lg[t] = ag; lu[t] = au;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (w == 0) {
        const float h = gx_swiglu(hsum8(lg + t), hsum8(lu + t));
        lh[rr] = h;
        if (dbg) hdbg[s * N_FF + row] = h;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (t == 0) q8_block(lh, hq + ((size_t) s * NBH + blk) * YB1, dt == DT_Q4_1);
}

// Kernel 2: down rows. Q4_1 x Q8_1 (dt = DT_Q4_1): same lane mapping; summs as the ARM build computes it,
// summs += fma(m0, s0, m1 * s1) per block pair; result = summs + (hsum(v0) + hsum(v1)).
// Q4_0 x Q8_0 (dt = DT_Q4_0): as the gate/up dot; result = hsum(v0) + hsum(v1).
// Grid: (256 * N_EMBD / 32, k).
kernel void gx_down(global const uchar * w0, global const uchar * w1, global const uchar * w2, global const uchar * w3,
                    global const uchar * w4, global const uchar * w5, global const uchar * w6, global const uchar * w7,
                    global const ulong * offs, global const uchar * hq, global float * out, int dt) {
    local float la[256];
    const int t = get_local_id(0), blk = get_group_id(0), s = get_group_id(1);
    const int rr = t >> 3, w = t & 7, p = w >> 2, j = w & 3;
    const int row = blk * 32 + rr;
    global const uchar * W = pick(s, w0, w1, w2, w3, w4, w5, w6, w7);
    const int qb = dt == DT_Q4_1 ? QB1 : QB0, qo = dt == DT_Q4_1 ? 4 : 2, bias = dt == DT_Q4_1 ? 0 : 8;
    global const uchar * D = W + offs[s * 3 + 2] + (size_t) row * NBH * qb;
    global const uchar * Y = hq + (size_t) s * NBH * YB1;
    float acc = 0.0f;
    for (int ib = p; ib < NBH; ib += 2) {
        global const uchar * yb = Y + ib * YB1;
        global const uchar * xb = D + ib * qb;
        const char4 yl = vload4(0, (global const char *) (yb + 4 + 4 * j));
        const char4 yh = vload4(0, (global const char *) (yb + 4 + 16 + 4 * j));
        const int is = lane_dot(vload4(0, xb + qo + 4 * j), bias, yl, yh);
        const float sc = vload_half(0, (global const half *) xb) * vload_half(0, (global const half *) yb);
        acc = fma((float) is, sc, acc);
    }
    la[t] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (w == 0) {
        float summs = 0.0f;
        for (int ib = 0; dt == DT_Q4_1 && ib + 1 < NBH; ib += 2) {
            global const uchar * x0 = D + ib * QB1;
            global const uchar * y0 = Y + ib * YB1;
            const float m0 = vload_half(1, (global const half *) x0), s0 = vload_half(1, (global const half *) y0);
            const float m1 = vload_half(1, (global const half *) (x0 + QB1)), s1 = vload_half(1, (global const half *) (y0 + YB1));
            summs = summs + fma(m0, s0, m1 * s1);
        }
        const float v = hsum8(la + t);
        out[(size_t) s * N_EMBD + row] = dt == DT_Q4_1 ? summs + v : v;
    }
}
)GXCL"
