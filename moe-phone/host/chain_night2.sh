#!/bin/bash
# chain_night2.sh — the queue as re-ordered at 02:10 on 2026-09-18, after the attention A/B was stopped.
#
# WHY IT WAS RE-ORDERED. Two things happened:
#   * the A/B's GPU arm crashed at model load (SIGSEGV in ggml_backend_opencl_buffer_type_get_alignment:
#     our --attn-device path used a device's buffer type without initialising its backend first) and each
#     crash costs the driver 25 minutes of polling, so the campaign was stopped after its first repeat.
#     Its HTP row is kept: 3.82 tok/s against the same-campaign CPU row's 4.59, a 17% LOSS, with the
#     engine's own split showing compute 0.102 -> 0.177 s/token, i.e. the cost of splitting the graph
#     across a backend boundary 192 times per token rather than any slowness in the DSP's arithmetic.
#   * decomposing the best measured cell against the hardware floor (gates/decode_budget.py) showed the
#     largest single addressable term is the 48.3 ms/token stall, and that it is almost exactly the
#     UNHIDDEN read time -- because ggml consumes a layer's experts in ascending expert id, so a layer
#     blocks on its lowest-id miss while its other seven resident experts wait their turn.
# Patch 0010 (--expert-resident-first) reorders that consumption. It is pure scheduling, verified
# lossless on the laptop (identical text), and it is therefore the first thing that runs here.
#
# Order: expert order -> ZRAM cache -> aggregate bandwidth. The matmul sweep and the thread/app-vs-shell
# chains are already waiting on the markers this touches.
set -u
export ANDROID_SERIAL=192.168.0.65:5555
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
HOSTDIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$R/chain_night2.log"
log() { echo "$(date -Iseconds) $*" >> "$LOG"; }

stop_campaigns() {
  for pat in 'sh bmoe_mem\.sh' 'sh bmoe_zram\.sh' 'sh bmoe_order\.sh' 'bmoe-cli'; do
    pids=$(adb shell "ps -A -o PID,ARGS" | awk -v p="$pat" '$0 ~ p && $0 !~ /awk/ {print $1}')
    [ -n "$pids" ] && { log "stopping [$pat]: $pids"; adb shell "kill $pids" >/dev/null 2>&1; sleep 10; }
  done
}

# run a device-side campaign, wait for its DONE, pull it
device_campaign() {  # script  reps  results_subdir  dir_glob
  local script=$1 reps=$2 sub=$3 glob=$4
  stop_campaigns
  log "$script start"
  adb shell "cd /data/local/tmp/moe-stream && echo 'chain_night2: $script \$(date)' >> phone_queue.log && (setsid nohup sh $script $reps > ${script%.sh}_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
  sleep 60
  while :; do
    d=$(adb shell "ls -d /data/local/tmp/moe-stream/$glob 2>/dev/null | tail -1" | tr -d '\r')
    if [ -n "$d" ] && adb shell "[ -f ${d}DONE ]"; then break; fi
    running=$(adb shell "ps -A -o PID,ARGS" | awk -v p="sh $script" '$0 ~ p && $0 !~ /awk/ {print $1}')
    if [ -z "$running" ]; then log "$script exited without DONE (dir '$d')"; break; fi
    sleep 120
  done
  d=$(adb shell "ls -d /data/local/tmp/moe-stream/$glob 2>/dev/null | tail -1" | tr -d '\r')
  mkdir -p "$R/$sub"
  adb pull "$d" "$R/$sub" >/dev/null 2>&1
  log "$script pulled from $d"
}

log "START"
device_campaign bmoe_order.sh 3 bmoe_order 'bmoe_order_*/'
device_campaign bmoe_zram.sh  3 bmoe_zram  'bmoe_zram_*/'

log "agg_bandwidth start"
sh "$HOSTDIR/agg_bandwidth.sh" 3 >> "$LOG" 2>&1
log "agg_bandwidth done"

touch "$R/CHAIN_AFTER_AB_DONE"   # releases chain_matmul.sh, which installs the fixed app and sweeps
log "released chain_matmul"
touch "$R/CHAIN_NIGHT2_DONE"
