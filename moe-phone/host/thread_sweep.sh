#!/bin/bash
# thread_sweep.sh — how many CPU threads maximise weight-byte throughput when NOTHING is faulting?
#
# WHY RE-ASK A SETTLED QUESTION. Every streaming configuration in this project pins 4 threads to the big
# cluster, and the reason on record is a measured thread cliff: at 4+ threads the phone took 520k-879k
# MAJOR faults in under a minute (results/2026-09-17/decode_15r_pin.json) and got slower. But that
# campaign ran a model that did not fit in RAM, so the cliff may be a property of the FAULT STORM rather
# than of the cores: a fault storm gets worse with more threads competing for the same page cache, and
# that says nothing about how many threads a resident matmul wants. Tonight's in-app CPU row on a fully
# resident model already came out well above that campaign's best (claim inapp_olmoe_cpu_tok_s vs the
# 31.64 tok/s in decode_15r_pin.json), which is exactly what a fault-storm explanation predicts.
#
# If throughput keeps rising past 4 threads on a resident model, then the CPU's ceiling for the target
# model (claim device_ceiling_qwen3_cpu) is understated, and the engine's own 4-thread pin is leaving
# bandwidth unused whenever its cache is hitting. That is worth knowing before any more device work.
#
# Arms: -t 2, 4, 6, 8 on OLMoE (resident, no flash in the loop), in the app process, thread count order
# rotated per repeat. llama-bench's own tg rate; the analyser converts it to GB/s with the model's active
# bytes. This SoC has no little cores: cpu0-5 top out at 3.3216 GHz and cpu6-7 at 3.8016 GHz (read from
# /sys/.../cpuinfo_max_freq), so 8 threads is 8 fast cores and the usual big/little explanation for a
# thread cliff does not apply here. Each row records the THROTTLED caps as well (thermal_gate.sh
# thermal_state), because a row taken at a 1.65 GHz cap measures the cap, not the thread count.
set -u
export ANDROID_SERIAL=192.168.0.65:5555
PKG=${PKG:-com.moephone.npu2}          # already holds olmoe.gguf
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/$(date +%F)/thread_sweep"
M=/data/data/$PKG/files/olmoe.gguf
N=${N:-128}
REPS=${1:-3}
mkdir -p "$R"
log() { echo "$(date -Iseconds) $*" >> "$R/driver.log"; }

run_one() {  # tag threads
  local tag=$1 t=$2 i=0
  adb shell "run-as $PKG sh -c 'rm -f files/out.txt files/bench.txt'" >/dev/null 2>&1
  adb shell "input keyevent KEYCODE_WAKEUP; am start -n $PKG/com.moephone.npu.Run --es bench '-m~~$M~~-p~~0~~-n~~$N~~-r~~2~~-t~~$t~~-ngl~~0'" >/dev/null 2>&1
  while [ $i -lt 120 ]; do
    sleep 5
    if adb shell "run-as $PKG sh -c 'grep -c EXIT= files/out.txt 2>/dev/null'" 2>/dev/null | tr -d '\r' | grep -qv '^0$'; then break; fi
    i=$((i + 1))
  done
  adb shell "run-as $PKG sh -c 'cat files/bench.txt'" > "$R/$tag.txt" 2>/dev/null
  adb shell "run-as $PKG sh -c 'cat files/out.txt'"   > "$R/$tag.runner.txt" 2>/dev/null
  adb shell '. /data/local/tmp/moe-stream/thermal_gate.sh; thermal_state' > "$R/$tag.state.txt" 2>/dev/null
  log "$tag :: $(grep -oE 'tg[0-9]+ *\| *[0-9.]+' "$R/$tag.txt" | tail -1)"
}

log "START reps=$REPS n_gen=$N pkg=$PKG"
# Warm the model into the page cache first: the first run after an install or a reboot otherwise
# spends minutes faulting a multi-GB file in and can exceed the driver\'s wait, which is exactly how
# the 09:44 sweep lost its htp_default and gpu rows (both were still running when it gave up).
adb shell "run-as $PKG sh -c \'cat files/olmoe.gguf > /dev/null 2>&1\'" >/dev/null 2>&1 </dev/null
log "model warmed into page cache"
for rep in $(seq 1 "$REPS"); do
  case $((rep % 4)) in
    1) order="2 4 6 8" ;;
    2) order="4 6 8 2" ;;
    3) order="6 8 2 4" ;;
    0) order="8 2 4 6" ;;
  esac
  for t in $order; do
    run_one "cpu_t${t}_rep${rep}" "$t"
    sleep 15
  done
done
log "DONE"
touch "$R/DONE"
