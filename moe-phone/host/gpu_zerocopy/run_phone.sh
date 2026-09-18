#!/bin/bash
# run_phone.sh — push zcbench to the 15R, run it once, pull the output. Coordinate with any running phone
# queue first: this uses the CPU and GPU for ~2-3 minutes and must not overlap another benchmark.
#   host/gpu_zerocopy/run_phone.sh [extra zcbench args]
set -eu
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
D=/data/local/tmp/zcbench
R="$REPO/moe-phone/results/$(date +%F)/zcbench"
mkdir -p "$R"
# refuse to start if another benchmark is running on the phone
busy=$(adb shell "ps -A -o NAME 2>/dev/null | grep -E 'bmoe-cli|llama-|zcbench' | head -3" | tr -d '\r')
if [ -n "$busy" ]; then echo "phone busy: $busy"; exit 1; fi
adb shell "mkdir -p $D"
adb push "$HERE/out/android/." "$D/" >/dev/null
mk=""
adb shell "test -f $D/zc_experts.bin" || mk="--make-file"
TG=/data/local/tmp/moe-stream/thermal_gate.sh
adb shell "if [ -f $TG ]; then . $TG; quiesce; thermal_state; fi" > "$R/state_before.txt" 2>&1 || true
adb shell "cd $D && LD_LIBRARY_PATH=/vendor/lib64:$D ./zcbench --file $D/zc_experts.bin $mk $*" | tee "$R/zcbench.out"
adb shell "if [ -f $TG ]; then . $TG; thermal_state; fi" > "$R/state_after.txt" 2>&1 || true
adb pull "$D/zc_concurrent_trace.csv" "$R/" >/dev/null 2>&1 || true
echo "results in $R"
