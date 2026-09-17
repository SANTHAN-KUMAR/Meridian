#!/bin/bash
# chain_final.sh — the last leg of the 2026-09-18 night. Replaces chain_threads.sh so that the BALANCED
# expert-order A/B (device/bmoe_order3.sh, ABBA within each repeat) runs before the two CPU campaigns,
# because the first two expert-order campaigns disagreed and both were position-confounded.
#
#   1. bmoe_order3.sh   12 runs, 6 per arm, ABBA order, instrumented binary (probe counters)
#   2. thread_sweep.sh  2/4/6/8 threads on a resident model, in-app -- is the 4-thread pin still right
#                       once nothing is faulting?
#   3. app_vs_shell.sh  what the app process actually costs, measured after a 10-minute idle with the
#                       arms alternating and every row's frequency caps recorded
#   4. bmoe_mem.sh      the row-stream / small-context campaign queued much earlier
set -u
export ANDROID_SERIAL=192.168.0.65:5555
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
HOSTDIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$R/chain_final.log"
log() { echo "$(date -Iseconds) $*" >> "$LOG"; }

stop_campaigns() {
  for pat in 'sh bmoe_mem\.sh' 'sh bmoe_zram\.sh' 'sh bmoe_order' 'bmoe-cli'; do
    pids=$(adb shell "ps -A -o PID,ARGS" | awk -v p="$pat" '$0 ~ p && $0 !~ /awk/ {print $1}')
    [ -n "$pids" ] && { log "stopping [$pat]: $pids"; adb shell "kill $pids" >/dev/null 2>&1; sleep 10; }
  done
}

device_campaign() {  # script reps results_subdir dir_glob
  local script=$1 reps=$2 sub=$3 glob=$4
  stop_campaigns
  log "$script start"
  adb shell "cd /data/local/tmp/moe-stream && echo 'chain_final: $script \$(date)' >> phone_queue.log && (setsid nohup sh $script $reps > ${script%.sh}_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
  sleep 60
  while :; do
    d=$(adb shell "ls -d /data/local/tmp/moe-stream/$glob 2>/dev/null | tail -1" | tr -d '\r')
    if [ -n "$d" ] && adb shell "[ -f ${d}DONE ]"; then break; fi
    running=$(adb shell "ps -A -o PID,ARGS" | awk -v p="sh $script" '$0 ~ p && $0 !~ /awk/ {print $1}')
    if [ -z "$running" ]; then log "$script exited without DONE (dir '$d')"; break; fi
    sleep 90
  done
  d=$(adb shell "ls -d /data/local/tmp/moe-stream/$glob 2>/dev/null | tail -1" | tr -d '\r')
  mkdir -p "$R/$sub"
  adb pull "$d" "$R/$sub" >/dev/null 2>&1
  log "$script pulled from $d"
}

log "waiting for chain_matmul"
while [ ! -f "$R/CHAIN_MATMUL_DONE" ]; do sleep 60; done
log "matmul sweep done"

device_campaign bmoe_order3.sh 3 bmoe_order3 'bmoe_order3_*/'

log "thread_sweep start"
sh "$HOSTDIR/thread_sweep.sh" 3 >> "$LOG" 2>&1
log "thread_sweep done"

log "app_vs_shell start"
sh "$HOSTDIR/app_vs_shell.sh" 3 >> "$LOG" 2>&1
log "app_vs_shell done"

stop_campaigns
adb shell 'cd /data/local/tmp/moe-stream && echo "chain_final: bmoe_mem $(date)" >> phone_queue.log && (setsid nohup sh bmoe_mem.sh 3 > bmoe_mem_nohup.log 2>&1 < /dev/null &)' >/dev/null 2>&1
log "bmoe_mem launched"
touch "$R/CHAIN_FINAL_DONE"
