#!/bin/bash
# chain_attn.sh — after chain_after.sh: GPU-attention A/B (host/bmoe_attn_gpu.sh), with its own host-side
# waker (the earlier wakers exit on their chains' DONE markers), then the phone is restored.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
HOSTDIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$R/chain_attn.log"
log() { echo "$(date -Iseconds) $*" >> "$LOG"; }
A() { adb shell "$@" </dev/null; }
log "waiting for chain_after"
while [ ! -f "$R/CHAIN_AFTER_DONE" ]; do sleep 60; done
A 'svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable' >/dev/null 2>&1
sed 's#CHAIN_NEXT_DONE#CHAIN_ATTN_DONE#' /tmp/claude-1000/waker.sh > /tmp/claude-1000/waker3.sh
setsid nohup bash /tmp/claude-1000/waker3.sh >/dev/null 2>&1 </dev/null &
s=$(A 'dumpsys thermalservice' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'); t=0
while [ "${s:-9}" -gt 1 ] && [ $t -lt 1200 ]; do sleep 30; t=$((t+30)); s=$(A 'dumpsys thermalservice' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'); done
log "cool_gate waited ${t}s, status ${s:-?}"
bash "$HOSTDIR/bmoe_attn_gpu.sh" 3 >> "$LOG" 2>&1; log "bmoe_attn_gpu exit $?"
A 'am force-stop com.moephone.bmoe3; svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
log "phone restored"; touch "$R/CHAIN_ATTN_DONE"
