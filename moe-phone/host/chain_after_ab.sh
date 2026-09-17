#!/bin/bash
# chain_after_ab.sh — the rest of the 2026-09-18 night, in the order the arithmetic makes interesting:
#   1. bmoe_zram.sh      does an oversized (ZRAM-backed) expert cache beat one sized to MemAvailable?
#   2. agg_bandwidth.sh  do two compute devices ADD weight-byte throughput, or share one bottleneck?
#   3. bmoe_mem.sh       the row-stream / small-context memory campaign the earlier driver queued
# Each waits for the one before it, and bmoe_mem is stopped whenever it has been restarted by an earlier
# driver, so only one campaign ever touches the flash.
set -u
export ANDROID_SERIAL=192.168.0.65:5555
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
HOSTDIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$R/chain_after_ab.log"
log() { echo "$(date -Iseconds) $*" >> "$LOG"; }

stop_campaigns() {
  for pat in 'sh bmoe_mem\.sh' 'bmoe-cli'; do
    pids=$(adb shell "ps -A -o PID,ARGS" | awk -v p="$pat" '$0 ~ p && $0 !~ /awk/ {print $1}')
    [ -n "$pids" ] && { log "stopping [$pat]: $pids"; adb shell "kill $pids" >/dev/null 2>&1; sleep 10; }
  done
}

log "waiting for the attention A/B chain"
while [ ! -f "$R/bmoe_attn_ab/CHAIN_DONE" ]; do sleep 60; done
log "A/B done"
stop_campaigns

log "bmoe_zram start"
adb shell 'cd /data/local/tmp/moe-stream && echo "chain_after_ab: bmoe_zram $(date)" >> phone_queue.log && (setsid nohup sh bmoe_zram.sh 3 > bmoe_zram_nohup.log 2>&1 < /dev/null &)' >/dev/null 2>&1
sleep 60
while :; do
  d=$(adb shell 'ls -d /data/local/tmp/moe-stream/bmoe_zram_*/ 2>/dev/null | tail -1' | tr -d '\r')
  [ -n "$d" ] && adb shell "[ -f ${d}DONE ]" && break
  running=$(adb shell "ps -A -o PID,ARGS" | awk '$0 ~ /sh bmoe_zram\.sh/ && $0 !~ /awk/ {print $1}')
  [ -z "$running" ] && { log "bmoe_zram exited without DONE (dir $d)"; break; }
  sleep 120
done
d=$(adb shell 'ls -d /data/local/tmp/moe-stream/bmoe_zram_*/ 2>/dev/null | tail -1' | tr -d '\r')
mkdir -p "$R/bmoe_zram"
adb pull "$d" "$R/bmoe_zram" >/dev/null 2>&1
log "bmoe_zram pulled from $d"

log "agg_bandwidth start"
sh "$HOSTDIR/agg_bandwidth.sh" 3 >> "$LOG" 2>&1
log "agg_bandwidth done"

adb shell 'cd /data/local/tmp/moe-stream && echo "chain_after_ab: bmoe_mem $(date)" >> phone_queue.log && (setsid nohup sh bmoe_mem.sh 3 > bmoe_mem_nohup.log 2>&1 < /dev/null &)' >/dev/null 2>&1
log "bmoe_mem launched"
touch "$R/CHAIN_AFTER_AB_DONE"
