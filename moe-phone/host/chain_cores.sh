#!/bin/bash
# chain_cores.sh — core-placement campaign (device/bmoe_cores.sh) on the patch-0015 engine.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
DEV="$(cd "$(dirname "$0")/../device" && pwd)"
LOG="$R/chain_cores.log"; log() { echo "$(date -Iseconds) $*" >> "$LOG"; }
A() { adb shell "$@" </dev/null; }
A 'svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable' >/dev/null 2>&1
sed 's#CHAIN_NEXT_DONE#CHAIN_CORES_DONE#' /tmp/claude-1000/waker.sh > /tmp/claude-1000/waker_cores.sh
setsid nohup bash /tmp/claude-1000/waker_cores.sh >/dev/null 2>&1 </dev/null &
adb push "$DEV/bmoe_cores.sh" "$DEV/thermal_gate.sh" /data/local/tmp/moe-stream/ >/dev/null 2>&1 </dev/null
[ "$(A 'ls /data/local/tmp/moe-stream/bmoe-i8mm-0015/bmoe-cli 2>/dev/null | wc -l' | tr -d '\r')" = 1 ] || { log "FATAL 0015 binary missing"; exit 1; }
s=$(A 'dumpsys thermalservice' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'); t=0
while [ "${s:-9}" -gt 1 ] && [ $t -lt 900 ]; do sleep 30; t=$((t+30)); s=$(A 'dumpsys thermalservice' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'); done
log "cool_gate waited ${t}s, status ${s:-?}; bmoe_cores.sh start"
A "cd /data/local/tmp/moe-stream && (setsid nohup sh bmoe_cores.sh 4 > bmoe_cores_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
sleep 60
while :; do
  d=$(A "ls -d /data/local/tmp/moe-stream/bmoe_cores_*/ | tail -1" | tr -d '\r')
  [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
  [ -z "$(A 'ps -A -o ARGS' | grep 'sh bmoe_cores')" ] && { log "exited without DONE ($d)"; break; }
  sleep 60
done
mkdir -p "$R/bmoe_cores"; adb pull "$d" "$R/bmoe_cores" >/dev/null 2>&1 </dev/null; log "pulled $d"
A 'svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
log "phone restored"; touch "$R/CHAIN_CORES_DONE"
