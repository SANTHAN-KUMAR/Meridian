#!/bin/bash
# chain_swapguard.sh — push the 0001-0019 engine and run the pre-registered swap-guard A/B (device/bmoe_swapguard.sh).
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/$(date +%F)"
DEV="$(cd "$(dirname "$0")/../device" && pwd)"
BIN="/tmp/claude-1000/-run-media-santhankumar-New-Volume-identifying-variation/2d806460-912e-4de7-a437-dc77597a5bb6/scratchpad/bmoe-i8mm-0019"
LOG="$R/chain_swapguard.log"; mkdir -p "$R"; log() { echo "$(date -Iseconds) $*" >> "$LOG"; }
A() { adb shell "$@" </dev/null; }
[ "$(strings "$BIN/bmoe-cli" | grep -c 'swap-guard')" -ge 1 ] || { log "FATAL: $BIN/bmoe-cli has no --swap-guard (stale build)"; exit 1; }
busy=$(A 'ps -A -o ARGS' | grep -E 'bmoe-cli|llama-|zcbench|gx_|sh bmoe_' | grep -v grep)
[ -n "$busy" ] && { log "FATAL: phone busy: $busy"; exit 1; }
A 'svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable' >/dev/null 2>&1
sed "s#CHAIN_NEXT_DONE#CHAIN_SWAPGUARD_DONE#; s#results/2026-09-18#results/$(date +%F)#" /tmp/claude-1000/waker.sh > /tmp/claude-1000/waker_sg.sh
setsid nohup bash /tmp/claude-1000/waker_sg.sh >/dev/null 2>&1 </dev/null &
A 'mkdir -p /data/local/tmp/moe-stream/bmoe-i8mm-0019' >/dev/null 2>&1
adb push "$BIN"/* /data/local/tmp/moe-stream/bmoe-i8mm-0019/ >/dev/null 2>&1 </dev/null
adb push "$DEV/bmoe_swapguard.sh" "$DEV/thermal_gate.sh" /data/local/tmp/moe-stream/ >/dev/null 2>&1 </dev/null
A 'chmod 755 /data/local/tmp/moe-stream/bmoe-i8mm-0019/bmoe-cli'
log "0019 pushed: $(A 'sha256sum /data/local/tmp/moe-stream/bmoe-i8mm-0019/bmoe-cli' | cut -c1-16)"
# smoke: one short guard-on run must print the guard line and exit 0
A "cd /data/local/tmp/moe-stream/bmoe-i8mm-0019 && LD_LIBRARY_PATH=. ./bmoe-cli -m ../Qwen3-30B-A3B-Q4_0.gguf --chatml -n 8 --moe-stream --cache-mb auto --cache-ceil-mb 5000 --overlap --swap-guard 1 -p hi 2>&1 | grep -E 'generation:|swap-guard|FATAL|error' | head -4" > "$R/swapguard_smoke.txt" 2>&1
log "smoke: $(tr '\n' ' ' < "$R/swapguard_smoke.txt")"
grep -q "generation:" "$R/swapguard_smoke.txt" || { log "SMOKE FAILED"; touch "$R/CHAIN_SWAPGUARD_DONE"; exit 1; }
s=$(A 'dumpsys thermalservice' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'); t=0
while [ "${s:-9}" -gt 1 ] && [ $t -lt 900 ]; do sleep 30; t=$((t+30)); s=$(A 'dumpsys thermalservice' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'); done
log "cool_gate waited ${t}s, status ${s:-?}; bmoe_swapguard.sh start"
A "cd /data/local/tmp/moe-stream && (setsid nohup sh bmoe_swapguard.sh 6 > bmoe_swapguard_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
sleep 60
while :; do
  d=$(A "ls -d /data/local/tmp/moe-stream/bmoe_swapguard_*/ | tail -1" | tr -d '\r')
  [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
  [ -z "$(A 'ps -A -o ARGS' | grep 'sh bmoe_swapguard')" ] && { log "exited without DONE ($d)"; break; }
  sleep 60
done
mkdir -p "$R/bmoe_swapguard"; adb pull "$d" "$R/bmoe_swapguard" >/dev/null 2>&1 </dev/null; log "pulled $d"
A 'svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
log "phone restored"; touch "$R/CHAIN_SWAPGUARD_DONE"
