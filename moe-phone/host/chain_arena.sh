#!/bin/bash
# chain_arena.sh — run the slot-arena campaign after chain_next.sh finishes. It is separate because the
# arena script had to be fixed mid-queue (a sampler-kill bug under mksh ended every attempt after its first
# row) and chain_next.sh is a running bash script, which must not be edited in place.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
DEV="$(cd "$(dirname "$0")/../device" && pwd)"
LOG="$R/chain_arena.log"
log() { echo "$(date -Iseconds) $*" >> "$LOG"; }
A() { adb shell "$@" </dev/null; }
log "waiting for chain_next"
while [ ! -f "$R/CHAIN_NEXT_DONE" ]; do sleep 60; done
adb connect "$ANDROID_SERIAL" >/dev/null 2>&1
A 'svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable' >/dev/null 2>&1
adb push "$DEV/thermal_gate.sh" "$DEV/bmoe_arena2.sh" /data/local/tmp/moe-stream/ >/dev/null 2>&1 </dev/null
s=$(A 'dumpsys thermalservice' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'); t=0
while [ "${s:-9}" -gt 1 ] && [ $t -lt 1200 ]; do sleep 30; t=$((t+30)); s=$(A 'dumpsys thermalservice' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'); done
log "cool_gate waited ${t}s, status ${s:-?}"
log "bmoe_arena2.sh start"
A "cd /data/local/tmp/moe-stream && (setsid nohup sh bmoe_arena2.sh 3 > bmoe_arena2_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
sleep 60
while :; do
  d=$(A "ls -d /data/local/tmp/moe-stream/bmoe_arena2_*/ | tail -1" | tr -d '\r')
  [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
  [ -z "$(A 'ps -A -o ARGS' | grep 'sh bmoe_arena2')" ] && { log "arena2 exited without DONE ($d)"; break; }
  sleep 90
done
mkdir -p "$R/bmoe_arena2"; adb pull "$d" "$R/bmoe_arena2" >/dev/null 2>&1 </dev/null; log "pulled $d"
A 'am force-stop com.moephone.bmoe3; svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
log "phone restored"; touch "$R/CHAIN_ARENA_DONE"
