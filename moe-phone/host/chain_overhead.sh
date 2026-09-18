#!/bin/bash
# chain_overhead.sh — after chain_cores.sh: smoke-test the 0016 engine on OLMoE (plain and stream, -n 16),
# then the pre-registered overhead campaign (device/bmoe_overhead.sh).
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
DEV="$(cd "$(dirname "$0")/../device" && pwd)"
BIN="/tmp/claude-1000/-run-media-santhankumar-New-Volume-identifying-variation/2d806460-912e-4de7-a437-dc77597a5bb6/scratchpad/bmoe-i8mm-0016"
LOG="$R/chain_overhead.log"; log() { echo "$(date -Iseconds) $*" >> "$LOG"; }
A() { adb shell "$@" </dev/null; }
log "waiting for chain_cores"
while [ ! -f "$R/CHAIN_CORES_DONE" ]; do sleep 20; done
A 'svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable' >/dev/null 2>&1
sed 's#CHAIN_NEXT_DONE#CHAIN_OVERHEAD_DONE#' /tmp/claude-1000/waker.sh > /tmp/claude-1000/waker_overhead.sh
setsid nohup bash /tmp/claude-1000/waker_overhead.sh >/dev/null 2>&1 </dev/null &
A 'mkdir -p /data/local/tmp/moe-stream/bmoe-i8mm-0016' >/dev/null 2>&1
adb push "$BIN"/* /data/local/tmp/moe-stream/bmoe-i8mm-0016/ >/dev/null 2>&1 </dev/null
adb push "$DEV/bmoe_overhead.sh" "$DEV/thermal_gate.sh" /data/local/tmp/moe-stream/ >/dev/null 2>&1 </dev/null
A 'chmod 755 /data/local/tmp/moe-stream/bmoe-i8mm-0016/bmoe-cli'
M=/data/local/tmp/moe-stream/olmoe-1b-7b-0924-q4_0.gguf
for mode in plain stream; do
  x=""; [ $mode = stream ] && x="--moe-stream --cache-mb 4500 --force-cache --overlap --dense-weights anon --io-threads 4 --io-cpu-mask 0f"
  A "cd /data/local/tmp/moe-stream/bmoe-i8mm-0016 && LD_LIBRARY_PATH=. ./bmoe-cli -m $M --chatml -n 16 -t 4 --cpu-mask f0 $x -p hello 2>&1 | grep -E 'generation:|moe-cache:|error|fail' | head -3" > "$R/overhead_smoke_$mode.txt" 2>&1
  log "smoke $mode: $(tr '\n' ' ' < "$R/overhead_smoke_$mode.txt")"
  grep -q "generation:" "$R/overhead_smoke_$mode.txt" || { log "SMOKE FAILED ($mode); campaign not started"; touch "$R/CHAIN_OVERHEAD_DONE"; exit 1; }
done
s=$(A 'dumpsys thermalservice' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'); t=0
while [ "${s:-9}" -gt 1 ] && [ $t -lt 900 ]; do sleep 30; t=$((t+30)); s=$(A 'dumpsys thermalservice' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'); done
log "cool_gate waited ${t}s, status ${s:-?}; bmoe_overhead.sh start"
A "cd /data/local/tmp/moe-stream && (setsid nohup sh bmoe_overhead.sh 3 > bmoe_overhead_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
sleep 60
while :; do
  d=$(A "ls -d /data/local/tmp/moe-stream/bmoe_overhead_*/ | tail -1" | tr -d '\r')
  [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
  [ -z "$(A 'ps -A -o ARGS' | grep 'sh bmoe_overhead')" ] && { log "exited without DONE ($d)"; break; }
  sleep 60
done
mkdir -p "$R/bmoe_overhead"; adb pull "$d" "$R/bmoe_overhead" >/dev/null 2>&1 </dev/null; log "pulled $d"
A 'svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
log "phone restored"; touch "$R/CHAIN_OVERHEAD_DONE"
