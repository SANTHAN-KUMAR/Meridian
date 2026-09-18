#!/bin/bash
# chain_final3.sh — trimmed queue (user: too many phone checks). Replaces chain_rest2 (killed during
# agg_bandwidth, whose own driver is left to finish -- nothing else starts until its DONE exists),
# chain_diag and chain_gate (killed while waiting). Order by value to the 10 tok/s goal:
#   1. bmoe_specgate.sh  patch 0013 confidence gates (pre-registered)       ~40 min
#   2. bmoe_attn_gpu.sh  attention on the GPU (pre-registered)              ~30 min
#   3. bmoe_arena_perf.sh arena counters (why +14 ms)                        ~10 min
# Dropped: thread_sweep re-run (OLMoE resident; lowest value to the goal now).
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
HOSTDIR="$(cd "$(dirname "$0")" && pwd)"; DEV="$(cd "$HOSTDIR/../device" && pwd)"
BIN="/tmp/claude-1000/-run-media-santhankumar-New-Volume-identifying-variation/2d806460-912e-4de7-a437-dc77597a5bb6/scratchpad/bmoe-i8mm-0013"
LOG="$R/chain_final3.log"; log() { echo "$(date -Iseconds) $*" >> "$LOG"; }
A() { adb shell "$@" </dev/null; }
status() { A 'dumpsys thermalservice 2>/dev/null' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'; }
cool_gate() { local t=0 s; s=$(status); while [ "${s:-9}" -gt 1 ] && [ $t -lt 900 ]; do sleep 30; t=$((t+30)); s=$(status); done; log "cool_gate waited ${t}s, status ${s:-?}"; }
awake() { A 'svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable' >/dev/null 2>&1; }
device_campaign() {  # script arg glob subdir
  local script=$1 arg=$2 glob=$3 sub=$4 d
  log "$script start"
  A "cd /data/local/tmp/moe-stream && (setsid nohup sh $script $arg > ${script%.sh}_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
  sleep 60
  while :; do
    d=$(A "ls -d /data/local/tmp/moe-stream/$glob | tail -1" | tr -d '\r')
    [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
    [ -z "$(A 'ps -A -o ARGS' | grep "sh $script")" ] && { log "$script exited without DONE ($d)"; break; }
    sleep 60
  done
  mkdir -p "$R/$sub"; adb pull "$d" "$R/$sub" >/dev/null 2>&1 </dev/null; log "$script pulled $d"
}
log "waiting for agg_bandwidth DONE"
while [ ! -f "$R/agg_bandwidth/DONE" ]; do sleep 20; done
log "agg_bandwidth done"; A 'am force-stop com.moephone.npu2' >/dev/null 2>&1
awake
A 'mkdir -p /data/local/tmp/moe-stream/bmoe-i8mm-0013' >/dev/null 2>&1
adb push "$BIN"/* /data/local/tmp/moe-stream/bmoe-i8mm-0013/ >/dev/null 2>&1 </dev/null
adb push "$DEV/bmoe_specgate.sh" "$DEV/bmoe_arena_perf.sh" "$DEV/thermal_gate.sh" /data/local/tmp/moe-stream/ >/dev/null 2>&1 </dev/null
A 'chmod 755 /data/local/tmp/moe-stream/bmoe-i8mm-0013/bmoe-cli'
cool_gate; device_campaign bmoe_specgate.sh 3 'bmoe_specgate_*/' bmoe_specgate
cool_gate; awake; bash "$HOSTDIR/bmoe_attn_gpu.sh" 3 >> "$LOG" 2>&1; log "bmoe_attn_gpu exit $?"
A 'am force-stop com.moephone.bmoe3' >/dev/null 2>&1
cool_gate; awake; device_campaign bmoe_arena_perf.sh "" 'bmoe_arena_perf_*/' bmoe_arena_perf
A 'am force-stop com.moephone.bmoe3; am force-stop com.moephone.npu2; svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
log "phone restored"; touch "$R/CHAIN_FINAL3_DONE"
