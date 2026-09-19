#!/bin/bash
# run_phone_e4.sh — E4 of research/2026-09-19_RESEARCH_SPEC.md on the phone: gx variant 5 (declared tolerance).
#   1. gx_tol_test on Adreno (gate: rel vs fp64 <= 1e-5 per slot). Speed rows only if it passes.
#   2. gx_bench --variants 4,5 (v4 = the bit-exact same-session comparison), k = 1..4, both down types, spin 0 and 1.
# Criterion (spec E4, judged on host-visible median_ms of the v5 rows): k=2 <= 0.25 ms AND k=1 <= 0.15 ms positive;
# k=2 >= 0.40 ms kill; between: rerun once more (E4_BENCH_ARGS) and treat as kill if still between.
# Shared phone lock, battery guard >= 25%, ~30 min slot. ~350 MB of cases are pushed (only the .bin files).
#   host/gpu_ffn/run_phone_e4.sh <case_dir>
set -eu
export ANDROID_SERIAL=${ANDROID_SERIAL:-3C15CK0028J00000}
HERE=$(cd "$(dirname "$0")" && pwd)
MP=$(cd "$HERE/../.." && pwd)
CASES=${1:?case dir from make_cases.py}
D=/data/local/tmp/gx
AD=$HERE/out/android_e4
R="$MP/results/$(date +%F)/gx_e4_phone_$(date +%H%M%S)"
[ -e "$R" ] && { echo "exists: $R"; exit 1; }
[ -x "$AD/gx_bench" ] && [ -x "$AD/gx_tol_test" ] || { echo "no Android build in $AD (build.sh android with ANDROID_OUT=$AD)"; exit 1; }
busy=$(adb shell "ps -A -o NAME 2>/dev/null | grep -E 'bmoe-cli|llama-|zcbench|gx_' | head -3" | tr -d '\r')
[ -n "$busy" ] && { echo "phone busy: $busy"; exit 1; }
LOCK=/data/local/tmp/moe-stream/.phone_busy
holder=$(adb shell "cat $LOCK 2>/dev/null" | tr -d '\r')
[ -n "$holder" ] && { echo "phone locked by: $holder"; exit 1; }
bat=$(adb shell "dumpsys battery | grep level" | tr -dc '0-9')
[ "${bat:-0}" -lt 25 ] && { echo "battery ${bat}% < 25%: not running"; exit 1; }
adb shell "echo 'gx run_phone_e4 $(date +%H:%M:%S)' > $LOCK"
trap 'adb shell "rm -f $LOCK" >/dev/null 2>&1' EXIT
mkdir -p "$R"
{ echo "android build: $AD"; md5sum "$AD/gx_tol_test" "$AD/gx_bench"; echo "kernel_md5=$(md5sum < "$HERE/gx_kernels.cl" | cut -c1-16) git=$(git -C "$MP" rev-parse --short HEAD)"; } > "$R/BUILD"
adb shell "mkdir -p $D/cases"
adb push "$AD/gx_tol_test" "$AD/gx_bench" "$D/" >/dev/null
for f in "$CASES"/rand_*.bin "$CASES"/qwen3_*.bin; do adb push "$f" "$D/cases/" >/dev/null; done
TG=/data/local/tmp/moe-stream/thermal_gate.sh
adb shell "if [ -f $TG ]; then . $TG; quiesce; mem_ready 2000 60; thermal_state; else echo NO_THERMAL_GATE; fi" > "$R/state_before.txt" 2>&1 || true
E="cd $D && LD_LIBRARY_PATH=/vendor/lib64"
set +e
adb shell "$E; ./gx_tol_test cases 5; echo exit=\$?" | tee "$R/tol_test.out"
set -e
if ! grep -q "^exit=0" "$R/tol_test.out"; then echo "FAIL: tolerance gate: no speed rows" | tee "$R/VERDICT"; exit 3; fi
BA=${E4_BENCH_ARGS:---variants 4,5 --ks 1,2,3,4 --iters 300}
echo "bench args: $BA" >> "$R/BUILD"
adb shell "$E; ./gx_bench $BA" | tee "$R/bench.out"
adb shell "if [ -f $TG ]; then . $TG; thermal_state; fi" > "$R/state_after.txt" 2>&1 || true
echo "results in $R"
