#!/bin/bash
# agg_overlap.sh — do two devices ADD decode throughput? Version 2, designed so overlap is PROVEN, not assumed.
#
# Why v2: in agg_bandwidth.sh each member ran -n 64 -r 1 (about 1.5 s of generation) after a load that
# differs by device (the app members upload 3.7 GB into device buffers first), and the only timing was
# host-side start/end stamps around both members. Those stamps cannot show that the two GENERATION phases
# overlapped, which the pre-registration (PREREG_zram_and_aggregate.md, Q2) required before a positive
# result is reported. So agg_bandwidth's "pairs add" reading is not established.
#
# Design: per pair, the CPU member runs LONG in the shell (-n 128 -r 40, jsonl, ~2 min) and the device
# member starts 30 s later in the app (-n 128 -r 10, jsonl). llama-bench's jsonl gives each test's start
# time and every repetition's duration (samples_ns), so each repetition's wall interval is reconstructed
# and the analysis compares:
#   CPU reps entirely INSIDE the device member's generation window   vs   CPU reps entirely OUTSIDE it
#   device member's concurrent reps                                  vs   the device's own solo run
# Aggregate = concurrent CPU rate + concurrent device rate, against the best solo rate.
#   bash agg_overlap.sh REPS
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
PKG=com.moephone.npu2
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/$(date +%F)/agg_overlap"
H=/data/local/tmp/moe-stream
SHELL_MODEL=$H/olmoe-1b-7b-0924-q4_0.gguf
APP_MODEL=/data/data/$PKG/files/olmoe.gguf
REPS=${1:-2}
mkdir -p "$R"
exec 9>"/tmp/claude-1000/agg_overlap.lock"; flock -n 9 || { echo "another agg_overlap holds the lock" >&2; exit 3; }
log() { echo "$(date -Iseconds) $*" >> "$R/driver.log"; }
A() { adb shell "$@" </dev/null; }
cpu_long() {  # tag
  A "cd $H/ocl && LD_LIBRARY_PATH=. ./llama-bench -m $SHELL_MODEL -p 0 -n 128 -r 40 -t 4 -ngl 0 -o jsonl; echo exit=\$?" > "$R/$1.jsonl" 2> "$R/$1.stderr"
}
app_bench() {  # tag device reps
  local tag=$1 dev=$2 reps=$3 i=0
  A "am force-stop $PKG" >/dev/null 2>&1; sleep 2
  A "run-as $PKG sh -c 'rm -f files/out.txt files/bench.txt'" >/dev/null 2>&1
  A "input keyevent KEYCODE_WAKEUP; am start -n $PKG/com.moephone.npu.Run --es bench '-m~~$APP_MODEL~~-p~~0~~-n~~128~~-r~~$reps~~-t~~4~~-dev~~$dev~~-ngl~~99~~-o~~jsonl'" >/dev/null 2>&1
  while [ $i -lt 120 ]; do
    sleep 5
    A "run-as $PKG sh -c 'grep -c EXIT= files/out.txt 2>/dev/null'" 2>/dev/null | tr -d '\r' | grep -qv '^0$' && break
    i=$((i + 1))
  done
  A "run-as $PKG sh -c 'cat files/bench.txt'" > "$R/$tag.jsonl" 2>/dev/null
  A "run-as $PKG sh -c 'cat files/out.txt'" > "$R/$tag.runner.txt" 2>/dev/null
  A "am force-stop $PKG" >/dev/null 2>&1
}
log "START reps=$REPS"
A "run-as $PKG sh -c 'cat files/olmoe.gguf > /dev/null 2>&1'; cat $SHELL_MODEL > /dev/null" >/dev/null 2>&1
for rep in $(seq 1 "$REPS"); do
  for dev in GPUOpenCL HTP0; do
    d=$(echo $dev | tr 'A-Z' 'a-z' | cut -c1-3)
    app_bench "solo_${d}_rep$rep" "$dev" 10; log "solo_${d}_rep$rep done"
    sleep 20
    cpu_long "pair_${d}_cpu_rep$rep" & cp=$!
    sleep 30
    app_bench "pair_${d}_dev_rep$rep" "$dev" 10
    wait $cp; log "pair_${d}_rep$rep done"
    sleep 20
  done
done
log "DONE"; touch "$R/DONE"
