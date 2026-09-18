#!/bin/bash
# chain_stack2.sh — the stacked comparison again, on the fixed engine (patch 0015), after chain_stack.sh has
# pulled the stopped first run.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
DEV="$(cd "$(dirname "$0")/../device" && pwd)"
BIN="/tmp/claude-1000/-run-media-santhankumar-New-Volume-identifying-variation/2d806460-912e-4de7-a437-dc77597a5bb6/scratchpad/bmoe-i8mm-0015"
LOG="$R/chain_stack2.log"; log() { echo "$(date -Iseconds) $*" >> "$LOG"; }
A() { adb shell "$@" </dev/null; }
log "waiting for chain_stack to pull the stopped run"
while [ ! -f "$R/CHAIN_STACK_DONE" ]; do sleep 15; done
A 'svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable' >/dev/null 2>&1
sed 's#CHAIN_NEXT_DONE#CHAIN_STACK2_DONE#' /tmp/claude-1000/waker.sh > /tmp/claude-1000/waker_stack2.sh
setsid nohup bash /tmp/claude-1000/waker_stack2.sh >/dev/null 2>&1 </dev/null &
A 'mkdir -p /data/local/tmp/moe-stream/bmoe-i8mm-0015' >/dev/null 2>&1
adb push "$BIN"/* /data/local/tmp/moe-stream/bmoe-i8mm-0015/ >/dev/null 2>&1 </dev/null
adb push "$DEV/bmoe_stack.sh" "$DEV/thermal_gate.sh" /data/local/tmp/moe-stream/ >/dev/null 2>&1 </dev/null
A 'chmod 755 /data/local/tmp/moe-stream/bmoe-i8mm-0015/bmoe-cli'
log "0015 pushed: $(A 'sha256sum /data/local/tmp/moe-stream/bmoe-i8mm-0015/bmoe-cli' | cut -c1-16)"
s=$(A 'dumpsys thermalservice' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'); t=0
while [ "${s:-9}" -gt 1 ] && [ $t -lt 900 ]; do sleep 30; t=$((t+30)); s=$(A 'dumpsys thermalservice' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'); done
log "cool_gate waited ${t}s, status ${s:-?}; bmoe_stack.sh (0015) start"
A "cd /data/local/tmp/moe-stream && (setsid nohup sh bmoe_stack.sh 6 '' bmoe-i8mm-0015 > bmoe_stack_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
sleep 60
while :; do
  d=$(A "ls -d /data/local/tmp/moe-stream/bmoe_stack_*/ | tail -1" | tr -d '\r')
  [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
  [ -z "$(A 'ps -A -o ARGS' | grep 'sh bmoe_stack')" ] && { log "exited without DONE ($d)"; break; }
  sleep 60
done
mkdir -p "$R/bmoe_stack"; adb pull "$d" "$R/bmoe_stack" >/dev/null 2>&1 </dev/null; log "pulled $d"
A 'svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
log "phone restored"; touch "$R/CHAIN_STACK2_DONE"
