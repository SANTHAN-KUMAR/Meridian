#!/bin/bash
# chain_overlap.sh — after chain_final3.sh: two-device throughput with proven overlap (host/agg_overlap.sh).
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
HOSTDIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$R/chain_overlap.log"; log() { echo "$(date -Iseconds) $*" >> "$LOG"; }
A() { adb shell "$@" </dev/null; }
log "waiting for chain_final3"
while [ ! -f "$R/CHAIN_FINAL3_DONE" ]; do sleep 60; done
A 'svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable' >/dev/null 2>&1
sed 's#CHAIN_NEXT_DONE#CHAIN_OVERLAP_DONE#' /tmp/claude-1000/waker.sh > /tmp/claude-1000/waker_overlap.sh
setsid nohup bash /tmp/claude-1000/waker_overlap.sh >/dev/null 2>&1 </dev/null &
s=$(A 'dumpsys thermalservice' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'); t=0
while [ "${s:-9}" -gt 1 ] && [ $t -lt 900 ]; do sleep 30; t=$((t+30)); s=$(A 'dumpsys thermalservice' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'); done
log "cool_gate waited ${t}s, status ${s:-?}"
bash "$HOSTDIR/agg_overlap.sh" 2 >> "$LOG" 2>&1; log "agg_overlap exit $?"
A 'am force-stop com.moephone.npu2; svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
log "phone restored"; touch "$R/CHAIN_OVERLAP_DONE"
