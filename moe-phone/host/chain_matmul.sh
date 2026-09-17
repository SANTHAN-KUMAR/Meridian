#!/bin/bash
# chain_matmul.sh — last phase of the 2026-09-18 night: install the app build that carries the per-op
# matmul benchmark, sweep every Qwen3-30B-A3B matmul shape on CPU / HTP0 / GPUOpenCL, then hand the
# phone back to the memory campaign. Runs after chain_after_ab.sh so the install's flash writes never
# land inside a streaming measurement.
set -u
export ANDROID_SERIAL=192.168.0.65:5555
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
HOSTDIR="$(cd "$(dirname "$0")" && pwd)"
APK=${APK:-$HOME/moework/npu-apk3/bmoe3.apk}
LOG="$R/chain_matmul.log"
log() { echo "$(date -Iseconds) $*" >> "$LOG"; }

log "waiting for chain_after_ab"
while [ ! -f "$R/CHAIN_AFTER_AB_DONE" ]; do sleep 60; done

for pat in 'sh bmoe_mem\.sh' 'bmoe-cli'; do
  pids=$(adb shell "ps -A -o PID,ARGS" | awk -v p="$pat" '$0 ~ p && $0 !~ /awk/ {print $1}')
  [ -n "$pids" ] && { log "stopping [$pat]: $pids"; adb shell "kill $pids" >/dev/null 2>&1; sleep 10; }
done

log "installing $APK"
adb install -r "$APK" >> "$LOG" 2>&1 || { log "INSTALL FAILED"; exit 1; }
log "sweeping"
sh "$HOSTDIR/matmul_sweep.sh" 2 >> "$LOG" 2>&1
log "sweep done"

adb shell 'cd /data/local/tmp/moe-stream && echo "chain_matmul: bmoe_mem $(date)" >> phone_queue.log && (setsid nohup sh bmoe_mem.sh 3 > bmoe_mem_nohup.log 2>&1 < /dev/null &)' >/dev/null 2>&1
log "bmoe_mem launched"
touch "$R/CHAIN_MATMUL_DONE"
