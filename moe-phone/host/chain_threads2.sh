#!/bin/bash
# chain_threads2.sh — re-run the thread sweep (fixed: fresh app process per row) after chain_attn.sh.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
HOSTDIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$R/chain_threads2.log"
log() { echo "$(date -Iseconds) $*" >> "$LOG"; }
A() { adb shell "$@" </dev/null; }
log "waiting for chain_attn"
while [ ! -f "$R/CHAIN_ATTN_DONE" ]; do sleep 60; done
A 'svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable' >/dev/null 2>&1
sed 's#CHAIN_NEXT_DONE#CHAIN_THREADS2_DONE#' /tmp/claude-1000/waker.sh > /tmp/claude-1000/waker4.sh
setsid nohup bash /tmp/claude-1000/waker4.sh >/dev/null 2>&1 </dev/null &
s=$(A 'dumpsys thermalservice' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'); t=0
while [ "${s:-9}" -gt 1 ] && [ $t -lt 1200 ]; do sleep 30; t=$((t+30)); s=$(A 'dumpsys thermalservice' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'); done
log "cool_gate waited ${t}s, status ${s:-?}"
bash "$HOSTDIR/thread_sweep.sh" 3 >> "$LOG" 2>&1; log "thread_sweep exit $?"
A 'am force-stop com.moephone.npu2; svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
log "phone restored"; touch "$R/CHAIN_THREADS2_DONE"
