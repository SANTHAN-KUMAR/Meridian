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
  "native_exp_in_silu|s.replace('const float e = gx_expf(0.0f - g);', 'const float e = exp(0.0f - g);')"
  "plain_div_in_silu|s.replace('return cr_div(g, 1.0f + e) * u;', 'return native_divide(g, 1.0f + e) * u;')"
  "q8_round_toward_zero|s.replace('convert_int_rte(v[i] * id)', 'convert_int_rtz(v[i] * id)')"
  "q8_scale_127_over_amax|s.replace('const float id = d != 0.0f ? cr_div(1.0f, d) : 0.0f;', 'const float id = d != 0.0f ? cr_div(127.0f, amax) : 0.0f;')"
  "q8_s_from_fp16_d|s.replace('vstore_half_rte(d * (float) sum, 1', 'vstore_half_rte(vload_half(0, (global const half *) yo) * (float) sum, 1')"
  "row_unfused_fma|s.replace('acc[p * 4 + 0] = fma((float) (pr.s0 + pr.s1 + pr.s2 + pr.s3), sc, acc[p * 4 + 0]);', 'acc[p * 4 + 0] = (float) (pr.s0 + pr.s1 + pr.s2 + pr.s3) * sc + acc[p * 4 + 0];')"
  "row_sequential_hsum|s.replace('static float hsum8p(const float * l) {\n    return ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]));', 'static float hsum8p(const float * l) {\n    return ((((((l[0] + l[1]) + l[2]) + l[3]) + l[4]) + l[5]) + l[6]) + l[7];')"
  "row_wrong_parity|s.replace('row_block(ag, ib & 1,', 'row_block(ag, (ib >> 1) & 1,')"
  "row_summs_two_roundings|s.replace('summs = summs + fma(m0, ls[ib - 1], m1 * ls[ib]);', 'summs = (summs + m0 * ls[ib - 1]) + m1 * ls[ib];')"
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
