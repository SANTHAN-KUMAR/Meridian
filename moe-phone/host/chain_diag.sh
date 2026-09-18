#!/bin/bash
# chain_diag.sh — after chain_rest2.sh: the arena counter diagnostic (device/bmoe_arena_perf.sh).
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
DEV="$(cd "$(dirname "$0")/../device" && pwd)"
LOG="$R/chain_diag.log"; log() { echo "$(date -Iseconds) $*" >> "$LOG"; }
A() { adb shell "$@" </dev/null; }
log "waiting for chain_rest2"
while [ ! -f "$R/CHAIN_REST_DONE" ]; do sleep 60; done
A 'svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable' >/dev/null 2>&1
sed 's#CHAIN_NEXT_DONE#CHAIN_DIAG_DONE#' /tmp/claude-1000/waker.sh > /tmp/claude-1000/waker_diag.sh
setsid nohup bash /tmp/claude-1000/waker_diag.sh >/dev/null 2>&1 </dev/null &
adb push "$DEV/bmoe_arena_perf.sh" "$DEV/thermal_gate.sh" /data/local/tmp/moe-stream/ >/dev/null 2>&1 </dev/null
log "bmoe_arena_perf.sh start"
A "cd /data/local/tmp/moe-stream && (setsid nohup sh bmoe_arena_perf.sh > bmoe_arena_perf_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
sleep 60
while :; do
  d=$(A "ls -d /data/local/tmp/moe-stream/bmoe_arena_perf_*/ | tail -1" | tr -d '\r')
  [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
  [ -z "$(A 'ps -A -o ARGS' | grep 'sh bmoe_arena_perf')" ] && { log "exited without DONE ($d)"; break; }
  sleep 60
done
mkdir -p "$R/bmoe_arena_perf"; adb pull "$d" "$R/bmoe_arena_perf" >/dev/null 2>&1 </dev/null; log "pulled $d"
A 'svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
log "phone restored"; touch "$R/CHAIN_DIAG_DONE"
