#!/bin/bash
# chain_gate.sh — after chain_diag.sh: patch 0013 confidence-gate sweep (device/bmoe_specgate.sh).
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
DEV="$(cd "$(dirname "$0")/../device" && pwd)"
BIN="/tmp/claude-1000/-run-media-santhankumar-New-Volume-identifying-variation/2d806460-912e-4de7-a437-dc77597a5bb6/scratchpad/bmoe-i8mm-0013"
LOG="$R/chain_gate.log"; log() { echo "$(date -Iseconds) $*" >> "$LOG"; }
A() { adb shell "$@" </dev/null; }
log "waiting for chain_diag"
while [ ! -f "$R/CHAIN_DIAG_DONE" ]; do sleep 60; done
A 'svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable' >/dev/null 2>&1
sed 's#CHAIN_NEXT_DONE#CHAIN_GATE_DONE#' /tmp/claude-1000/waker.sh > /tmp/claude-1000/waker_gate.sh
setsid nohup bash /tmp/claude-1000/waker_gate.sh >/dev/null 2>&1 </dev/null &
A 'mkdir -p /data/local/tmp/moe-stream/bmoe-i8mm-0013' >/dev/null 2>&1
adb push "$BIN"/* /data/local/tmp/moe-stream/bmoe-i8mm-0013/ >/dev/null 2>&1 </dev/null
adb push "$DEV/bmoe_specgate.sh" "$DEV/thermal_gate.sh" /data/local/tmp/moe-stream/ >/dev/null 2>&1 </dev/null
A 'chmod 755 /data/local/tmp/moe-stream/bmoe-i8mm-0013/bmoe-cli'
log "0013 pushed; bmoe_specgate.sh start"
A "cd /data/local/tmp/moe-stream && (setsid nohup sh bmoe_specgate.sh 3 > bmoe_specgate_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
sleep 60
while :; do
  d=$(A "ls -d /data/local/tmp/moe-stream/bmoe_specgate_*/ | tail -1" | tr -d '\r')
  [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
  [ -z "$(A 'ps -A -o ARGS' | grep 'sh bmoe_specgate')" ] && { log "exited without DONE ($d)"; break; }
  sleep 60
done
mkdir -p "$R/bmoe_specgate"; adb pull "$d" "$R/bmoe_specgate" >/dev/null 2>&1 </dev/null; log "pulled $d"
A 'svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
log "phone restored"; touch "$R/CHAIN_GATE_DONE"
