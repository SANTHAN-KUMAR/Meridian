#!/bin/bash
# run_phone_m6.sh — gx on the phone (design M6, component C). PREPARED, NOT RUN. The phone is resting: do not
# run this until the user releases it, and only in a slot agreed with the engine session (one phone job at a
# time; check for orphan benchmark processes first).
#
# Order is fixed; step 2 only runs if step 1 is bit-exact (a speed row from a wrong kernel is void):
#   0. device facts: the full CL_DEVICE_EXTENSIONS string and fp32 config (gx_test prints the config)
#   1. correctness on Adreno: ggml_ref_arm runs NATIVELY on the phone over the pushed cases (closing the
#      "qemu equals hardware" assumption), then gx_test (both variants) and gx_pool_test against it
#   2. gx_bench: dispatch latency / GB/s per variant, down type, k; slot write cost
# Needs: build.sh arm-ref && build.sh android; a case dir from make_cases.py (about 250 MB are pushed).
#   host/gpu_ffn/run_phone_m6.sh <case_dir>
set -eu
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
HERE=$(cd "$(dirname "$0")" && pwd)
MP=$(cd "$HERE/../.." && pwd)
CASES=${1:?case dir}
D=/data/local/tmp/gx
R="$MP/results/$(date +%F)/gx_m6_$(date +%H%M%S)"
[ -e "$R" ] && { echo "exists: $R"; exit 1; }
busy=$(adb shell "ps -A -o NAME 2>/dev/null | grep -E 'bmoe-cli|llama-|zcbench|gx_' | head -3" | tr -d '\r')
if [ -n "$busy" ]; then echo "phone busy: $busy"; exit 1; fi
mkdir -p "$R"
adb shell "mkdir -p $D/cases"
adb push "$HERE/out/android/gx_test" "$HERE/out/android/gx_pool_test" "$HERE/out/android/gx_bench" "$HERE/out/arm/ggml_ref_arm" "$D/" >/dev/null
for f in "$CASES"/*.bin "$CASES"/quant_blocks.f32; do adb push "$f" "$D/cases/" >/dev/null; done
TG=/data/local/tmp/moe-stream/thermal_gate.sh
adb shell "if [ -f $TG ]; then . $TG; quiesce; mem_ready 2000 60; thermal_state; else echo NO_THERMAL_GATE; fi" > "$R/state_before.txt" 2>&1 || true
E="cd $D && LD_LIBRARY_PATH=/vendor/lib64"
# 1. references natively, then correctness
adb shell "$E; for c in cases/*.bin; do ./ggml_ref_arm \$c \$c.arm.out 4 || echo REF_FAIL \$c; done; \
  ./ggml_ref_arm --quant cases/quant_blocks.f32 cases/quant_blocks.f32.q8_0.arm cases/quant_blocks.f32.q8_1.arm" > "$R/ref.out" 2>&1
adb shell "$E; ./gx_test cases gx_test.json" | tee "$R/gx_test.out"
adb pull "$D/gx_test.json" "$R/" >/dev/null
adb shell "$E; ./gx_pool_test cases" | tee "$R/pool_test.out"
if ! grep -q "BIT-EXACT" "$R/gx_test.out"; then
  echo "NOT bit-exact on the phone: no speed rows (find the mechanism first)" | tee "$R/VERDICT"
  exit 3
fi
# 2. timing (capped clocks logged by the gate before and after)
adb shell "$E; ./gx_bench --iters 300" | tee "$R/bench.out"
adb shell "if [ -f $TG ]; then . $TG; thermal_state; fi" > "$R/state_after.txt" 2>&1 || true
echo "results in $R"
