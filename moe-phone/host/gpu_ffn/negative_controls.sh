#!/bin/bash
# negative_controls.sh — does gx_test detect a kernel that is numerically wrong in the ways that matter?
# Each control applies ONE mutation to a copy of gx_kernels.cl, rebuilds gx_test against it, and runs the
# same cases. A control PASSES (is detected) only if gx_test reports a nonzero bit difference or a nonzero
# division mismatch. If a control is not detected, the bit-exact result says nothing about that mechanism.
#   host/gpu_ffn/negative_controls.sh <case_dir>
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
WORK=${WORK:-$(cd "$REPO/.." && pwd)/moe-work}
CL_HDR=${CL_HDR:-$WORK/opencl/OpenCL-Headers}
CASES=${1:?case dir}
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

# name | python expression applied to the kernel source s (must change it)
controls=(
  "unfused_gate_up_fmla|s.replace('ag = fma((float) ig, sg, ag);', 'ag = (float) ig * sg + ag;')"
  "unfused_down_fmla|s.replace('acc = fma((float) is, sc, acc);', 'acc = (float) is * sc + acc;')"
  "q41_summs_two_roundings|s.replace('summs = summs + fma(m0, s0, m1 * s1);', 'summs = (summs + m0 * s0) + m1 * s1;')"
  "sequential_hsum|s.replace('return ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]));', 'return ((((((l[0] + l[1]) + l[2]) + l[3]) + l[4]) + l[5]) + l[6]) + l[7];')"
  "native_exp_in_silu|s.replace('    const float e = gx_expf(0.0f - g);\n    const float sil', '    const float e = exp(0.0f - g);\n    const float sil')"
  "plain_div_in_silu|s.replace('const float sil = cr_div(g, 1.0f + e);', 'const float sil = native_divide(g, 1.0f + e);')"
  "q8_round_toward_zero|s.replace('convert_int_rte(v[i] * id)', 'convert_int_rtz(v[i] * id)')"
  "q8_scale_127_over_amax|s.replace('const float id = d != 0.0f ? cr_div(1.0f, d) : 0.0f;', 'const float id = d != 0.0f ? cr_div(127.0f, amax) : 0.0f;')"
  "q8_s_from_fp16_d|s.replace('vstore_half_rte(d * (float) sum, 1', 'vstore_half_rte(vload_half(0, (global const half *) yo) * (float) sum, 1')"
  "expf_unfused_poly|s.replace('    const float z = fma(x, 0x1.715476p+0f, r);', '    const float z = x * 0x1.715476p+0f + r;')"
  "expf_special_branch_dropped|s.replace('    if (!(fabs(n) > 126.0f)) return fma(j, k, k);', '    return fma(j, k, k);')"
  "row_unfused_fma|s.replace('    *a4 = fma(ls, (float4)(sc), *a4);\n}\n\nstatic float hsum2', '    *a4 = ls * (float4)(sc) + *a4;\n}\n\nstatic float hsum2')"
  "row_sequential_hsum|s.replace('    return ((a.x + a.y) + (a.z + a.w)) + ((b.x + b.y) + (b.z + b.w));', '    return ((((((a.x + a.y) + a.z) + a.w) + b.x) + b.y) + b.z) + b.w;')"
  "row_wrong_parity|s.replace('row_block(&ag1, gb + QB0 + 2', 'row_block(&ag0, gb + QB0 + 2')"
  "row_summs_two_roundings|s.replace('            const float m0 = h2f(((global const ushort *) xb)[1]), m1 = h2f(((global const ushort *) (xb + QB1))[1]);\n            summs = summs + fma(m0, ls[ib], m1 * ls[ib + 1]);', '            const float m0 = h2f(((global const ushort *) xb)[1]), m1 = h2f(((global const ushort *) (xb + QB1))[1]);\n            summs = (summs + m0 * ls[ib]) + m1 * ls[ib + 1];')"
  "soa_tiled_unfused_fma|s.replace('    *a4 = fma(ls, (float4)(sc), *a4);\n}\n\n#define SOA_Q', '    *a4 = ls * (float4)(sc) + *a4;\n}\n\n#define SOA_Q')"
  "soa_wrong_parity|s.replace('row_block_v(&ag1, SOA_Q(G, ib + 1', 'row_block_v(&ag0, SOA_Q(G, ib + 1')"
  "soa_wrong_plane_stride|s.replace('#define SOA_Q(base, ib, row, R) as_uchar16(((global const uint4 *) (base))[(size_t) (ib) * (R) + (row)])', '#define SOA_Q(base, ib, row, R) as_uchar16(((global const uint4 *) (base))[(size_t) (row) * (NBX) + (ib) % (NBX)])')"
  "soa_summs_two_roundings|s.replace('            const float m0 = SOA_H(D, NBH, N_EMBD, 1, ib, row), m1 = SOA_H(D, NBH, N_EMBD, 1, ib + 1, row);\n            summs = summs + fma(m0, ls[ib], m1 * ls[ib + 1]);', '            const float m0 = SOA_H(D, NBH, N_EMBD, 1, ib, row), m1 = SOA_H(D, NBH, N_EMBD, 1, ib + 1, row);\n            summs = (summs + m0 * ls[ib]) + m1 * ls[ib + 1];')"
  "tiled_untiled_index|s.replace('#define TIL_I(ib, row, NB) ((((size_t) (row) >> 6) * (size_t) (NB) + (size_t) (ib)) * 64 + ((size_t) (row) & 63))', '#define TIL_I(ib, row, NB) ((((size_t) (row) >> 6) * (size_t) (NB) + (size_t) (ib)) * 64 + ((size_t) ((row) + 1) & 63))')"
  "tiled_wrong_parity|s.replace('row_block_v(&ag1, TIL_Q(G, ib + 1', 'row_block_v(&ag0, TIL_Q(G, ib + 1')"
  "pair_same_parity|s.replace('    for (int ib = p; ib < NBX; ib += 2) {\n        const float dy = ldx[ib];', '    for (int ib = 0; ib < NBX; ib += 2) {\n        const float dy = ldx[ib];')"
  "pair_wrong_row_partner|s.replace('const float h = gx_swiglu_r(hsum2(xg[t], xg[t + 1]), hsum2(xu[t], xu[t + 1]), &r);', 'const float h = gx_swiglu_r(hsum2(xg[t], xg[(t + 3) & 127]), hsum2(xu[t], xu[t + 1]), &r);')"
  "one_accumulator_per_lane|s.replace('for (int ib = p; ib < NBX; ib += 2) {', 'for (int ib = p ? NBX : 0; ib < NBX; ib += 1) {')"
)
fail=0
for c in "${controls[@]}"; do
  name=${c%%|*}; expr=${c#*|}
  python3 - "$HERE/gx_kernels.cl" "$T/k.cl" "$expr" <<'EOF' || { echo "CONTROL $name: mutation did not apply"; fail=1; continue; }
import sys
s = open(sys.argv[1]).read()
m = eval(sys.argv[3])
if m == s: sys.exit(1)
open(sys.argv[2], "w").write(m)
EOF
  { printf 'R"GXCL('; cat "$T/k.cl"; printf ')GXCL"\n'; } > "$T/k.inc"
  c++ -std=c++17 -O2 -ffp-contract=off -DGX_KERNELS_INC="\"$T/k.inc\"" -I"$HERE" -I"$CL_HDR" "$HERE/gx.cpp" "$HERE/gx_test.cpp" \
    -l:libOpenCL.so.1 -ldl -o "$T/gx_test" || { echo "CONTROL $name: build failed"; fail=1; continue; }
  out=$("$T/gx_test" "$CASES" 2>&1 | grep -E "^SUMMARY|^DIV")
  sum=$(echo "$out" | grep SUMMARY)
  if echo "$sum" | grep -q "BIT-EXACT"; then det=0; else det=1; fi
  echo "CONTROL $name detected=$det :: $(echo "$sum" | sed 's/^SUMMARY //')"
  [ $det = 1 ] || fail=1
done
echo "NEGATIVE_CONTROLS $([ $fail = 0 ] && echo ALL_DETECTED || echo SOME_UNDETECTED)"
exit $fail
