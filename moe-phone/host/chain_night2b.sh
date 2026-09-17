#!/bin/bash
# chain_night2b.sh — the tail of the night, launched after the ZRAM campaign so that the instrumented
# expert-order A/B (bmoe_order2.sh) can run before the host-driven campaigns take the phone.
#
# Order: instrumented expert-order A/B -> ZRAM cache -> aggregate bandwidth -> release chain_matmul.
# (This replaces chain_night2.sh, which was stopped after its expert-order phase so that the
# instrumented re-run could go first; its ZRAM phase is carried over here rather than dropped.)
# Why order2 first: the plain campaign showed neither a gain nor any change in the stall, which is the
# signature of a reordering that never fires. Until the probe counters say whether it fired, every other
# expert-scheduling idea would be built on an unmeasured assumption.
set -u
export ANDROID_SERIAL=192.168.0.65:5555
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
HOSTDIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$R/chain_night2b.log"
log() { echo "$(date -Iseconds) $*" >> "$LOG"; }

stop_campaigns() {
  for pat in 'sh bmoe_mem\.sh' 'sh bmoe_zram\.sh' 'sh bmoe_order\.sh' 'sh bmoe_order2\.sh' 'bmoe-cli'; do
    pids=$(adb shell "ps -A -o PID,ARGS" | awk -v p="$pat" '$0 ~ p && $0 !~ /awk/ {print $1}')
    [ -n "$pids" ] && { log "stopping [$pat]: $pids"; adb shell "kill $pids" >/dev/null 2>&1; sleep 10; }
  done
}

device_campaign() {  # script reps results_subdir dir_glob
  local script=$1 reps=$2 sub=$3 glob=$4
  stop_campaigns
  log "$script start"
  adb shell "cd /data/local/tmp/moe-stream && echo 'chain_night2b: $script \$(date)' >> phone_queue.log && (setsid nohup sh $script $reps > ${script%.sh}_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
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

log "START"
device_campaign bmoe_order2.sh 3 bmoe_order2 'bmoe_order2_*/'
device_campaign bmoe_zram.sh   3 bmoe_zram   'bmoe_zram_*/'
log "agg_bandwidth start"
sh "$HOSTDIR/agg_bandwidth.sh" 3 >> "$LOG" 2>&1
log "agg_bandwidth done"
touch "$R/CHAIN_AFTER_AB_DONE"   # releases chain_matmul.sh (installs the fixed app, sweeps per op)
log "released chain_matmul"
touch "$R/CHAIN_NIGHT2_DONE"
