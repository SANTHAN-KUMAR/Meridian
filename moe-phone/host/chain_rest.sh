#!/bin/bash
# chain_rest.sh — everything after chain_arena.sh, in ONE sequential chain (replaces chain_after.sh,
# chain_attn.sh and chain_threads2.sh, which were killed while still waiting, before running anything).
# Order, most consequential first:
#   1. bmoe_perfmode.sh   OEM high-performance mode off/on (clock caps at 43-68% of hardware max in every row)
#   2. bmoe_slru.sh       SLRU decode, clean re-run (pre-registered in host/chain_after.sh's header)
#   3. agg_bandwidth.sh   fixed (CPU arm on the phone)
#   4. bmoe_attn_gpu.sh   attention on the GPU (pre-registered in its header)
#   5. thread_sweep.sh    fixed (fresh app process per row)
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
HOSTDIR="$(cd "$(dirname "$0")" && pwd)"
DEV="$(cd "$HOSTDIR/../device" && pwd)"
LOG="$R/chain_rest.log"
log() { echo "$(date -Iseconds) $*" >> "$LOG"; }
A() { adb shell "$@" </dev/null; }
status() { A 'dumpsys thermalservice 2>/dev/null' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'; }
cool_gate() { local t=0 s; s=$(status); while [ "${s:-9}" -gt 1 ] && [ $t -lt 1200 ]; do sleep 30; t=$((t+30)); s=$(status); done; log "cool_gate waited ${t}s, status ${s:-?}"; }
awake() { A 'svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable' >/dev/null 2>&1; }
device_campaign() {  # script reps glob results_subdir
  local script=$1 reps=$2 glob=$3 sub=$4 d
  log "$script start"
  A "cd /data/local/tmp/moe-stream && (setsid nohup sh $script $reps > ${script%.sh}_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
  sleep 60
  while :; do
    d=$(A "ls -d /data/local/tmp/moe-stream/$glob | tail -1" | tr -d '\r')
    [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
    [ -z "$(A 'ps -A -o ARGS' | grep "sh $script")" ] && { log "$script exited without DONE ($d)"; break; }
    sleep 90
  done
  mkdir -p "$R/$sub"; adb pull "$d" "$R/$sub" >/dev/null 2>&1 </dev/null; log "$script pulled $d"
}
log "waiting for chain_arena"
while [ ! -f "$R/CHAIN_ARENA_DONE" ]; do sleep 60; done
awake
adb push "$DEV/thermal_gate.sh" "$DEV/bmoe_perfmode.sh" "$DEV/bmoe_slru.sh" /data/local/tmp/moe-stream/ >/dev/null 2>&1 </dev/null
for b in bmoe-i8mm-order2 bmoe-i8mm-slru; do
  [ "$(A "ls /data/local/tmp/moe-stream/$b/bmoe-cli 2>/dev/null | wc -l" | tr -d '\r')" = 1 ] || { log "FATAL: $b missing"; exit 1; }
done
cool_gate; device_campaign bmoe_perfmode.sh 3 'bmoe_perfmode_*/' bmoe_perfmode
A 'settings put system high_performance_mode_on 0' >/dev/null 2>&1
cool_gate; awake; device_campaign bmoe_slru.sh 3 'bmoe_slru_2*/' bmoe_slru_decode
cool_gate; awake; A 'am force-stop com.moephone.bmoe3' >/dev/null 2>&1
bash "$HOSTDIR/agg_bandwidth.sh" 3 >> "$LOG" 2>&1; log "agg_bandwidth exit $?"
A 'am force-stop com.moephone.npu2' >/dev/null 2>&1
cool_gate; awake; bash "$HOSTDIR/bmoe_attn_gpu.sh" 3 >> "$LOG" 2>&1; log "bmoe_attn_gpu exit $?"
cool_gate; awake; bash "$HOSTDIR/thread_sweep.sh" 3 >> "$LOG" 2>&1; log "thread_sweep exit $?"
A 'am force-stop com.moephone.bmoe3; am force-stop com.moephone.npu2; settings put system high_performance_mode_on 0; svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
log "phone restored"; touch "$R/CHAIN_REST_DONE"
