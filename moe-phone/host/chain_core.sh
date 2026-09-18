#!/bin/bash
# chain_core.sh — the remaining CORE queue, reordered by what each campaign can do for the goal
# (10 tok/s on Qwen3-30B-A3B). It takes over from chain_resume.sh as soon as the ZRAM campaign finishes,
# because chain_resume's tail order was written before two things were known: that the slot arena is the
# largest single saving ever measured here, and that its own arena run was killed after one row.
#
# ORDER, AND THE REASON FOR EACH PLACE:
#
#   1. bmoe_arena2   The only lever that has ever removed a large block of time: cache management fell
#                    34.0 -> 2.0 ms/token in the original campaign (results/2026-09-17/bmoe_arena, read
#                    through gates/decode_budget.py). It lost overall on +14 ms compute and +24 ms stall
#                    at a 4000 MiB budget and a 78% hit rate; today's operating point has a third fewer
#                    misses for that penalty to attach to. It samples MemAvailable and SwapFree every 2 s
#                    to test whether the penalty is the arena's never-released slots overrunning the
#                    budget the engine thinks it is honouring. Re-run because the first attempt was
#                    SIGTERMed after one row.
#   2. agg_bandwidth The strategic one: do two compute devices ADD weight-byte throughput or share one
#                    bottleneck? 10 tok/s needs 18.40 GB/s and the best single device delivers 34.85, so
#                    the goal does not depend on this -- but every rate above ~19 tok/s does, and so does
#                    knowing whether a split-FFN patch is worth writing at all.
#   3. matmul_sweep  Per-op truth for every shape a token computes, on each device, with the upload cost
#                    that decides whether streamed experts could ever compute somewhere other than the
#                    CPU. This is also the instrument that replaces the OLMoE-derived ceilings with
#                    Qwen3-shaped ones, which the README flags as the optimistic part of every ceiling
#                    claim in the project.
#   4. thread_sweep  Cheap, and it re-asks a settled question whose evidence was taken during a
#                    page-fault storm: is 4 threads still right when nothing is faulting?
#   5. app_vs_shell  Explains a confound rather than moving the number, so it goes last.
#
# The memory campaign (bmoe_mem.sh) is deliberately NOT here: row-stream and small-context cells were
# measured as losses already, and phone time is better spent on the five above.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
HOSTDIR="$(cd "$(dirname "$0")" && pwd)"
DEV="$(cd "$HOSTDIR/../device" && pwd)"
LOG="$R/chain_core.log"
log() { echo "$(date -Iseconds) $*" >> "$LOG"; }

# 0. wait for the ZRAM campaign that chain_resume is running, then take the phone over from it
log "waiting for the ZRAM campaign to finish"
while :; do
  d=$(adb shell "ls -d /data/local/tmp/moe-stream/bmoe_zram_*/ 2>/dev/null | tail -1" </dev/null | tr -d '\r')
  [ -n "$d" ] && adb shell "[ -f ${d}DONE ]" </dev/null && break
  running=$(adb shell "ps -A -o PID,ARGS" </dev/null | awk '/sh bmoe_zram\.sh/ && !/awk/ {print $1}')
  [ -z "$running" ] && { log "zram is not running and has no DONE; taking over anyway"; break; }
  sleep 60
done
d=$(adb shell "ls -d /data/local/tmp/moe-stream/bmoe_zram_*/ 2>/dev/null | tail -1" </dev/null | tr -d '\r')
[ -n "$d" ] && { mkdir -p "$R/bmoe_zram"; adb pull "$d" "$R/bmoe_zram" >/dev/null 2>&1 </dev/null; log "zram pulled from $d"; }

# stop chain_resume by exact argv (matching a literal "bash chain_resume.sh" missed "bash host/..." once
# and left three drivers running at the same time)
for p in $(ps -eo pid,args | awk '$NF ~ /chain_resume\.sh$/ {print $1}'); do
  kill "$p" 2>/dev/null && log "stopped chain_resume pid $p"
done
sleep 3

stop_campaigns() {
  for pat in 'sh bmoe_mem\.sh' 'sh bmoe_zram\.sh' 'sh bmoe_order' 'sh bmoe_arena' 'sh bmoe_prefill' 'bmoe-cli'; do
    pids=$(adb shell "ps -A -o PID,ARGS" </dev/null | awk -v p="$pat" '$0 ~ p && $0 !~ /awk/ {print $1}')
    [ -n "$pids" ] && { log "stopping [$pat]: $pids"; adb shell "kill $pids" >/dev/null 2>&1 </dev/null; sleep 10; }
  done
}

device_campaign() {  # script reps results_subdir dir_glob
  local script=$1 reps=$2 sub=$3 glob=$4
  stop_campaigns
  adb push "$DEV/$script" /data/local/tmp/moe-stream/ >/dev/null 2>&1 </dev/null
  log "$script start"
  adb shell "cd /data/local/tmp/moe-stream && echo 'chain_core: $script \$(date)' >> phone_queue.log && (setsid nohup sh $script $reps > ${script%.sh}_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1 </dev/null
  sleep 60
  local lost=0
  while :; do
    if ! adb shell 'echo up' >/dev/null 2>&1 </dev/null; then
      adb connect "$ANDROID_SERIAL" >/dev/null 2>&1; lost=$((lost + 1))
      [ $lost -gt 40 ] && { log "$script: phone gone for 10 min, giving up on this campaign"; break; }
      sleep 15; continue
    fi
    d=$(adb shell "ls -d /data/local/tmp/moe-stream/$glob 2>/dev/null | tail -1" </dev/null | tr -d '\r')
    [ -n "$d" ] && adb shell "[ -f ${d}DONE ]" </dev/null && break
    running=$(adb shell "ps -A -o PID,ARGS" </dev/null | awk -v p="sh $script" '$0 ~ p && $0 !~ /awk/ {print $1}')
    [ -z "$running" ] && { log "$script exited without DONE (dir '$d')"; break; }
    sleep 90
  done
  d=$(adb shell "ls -d /data/local/tmp/moe-stream/$glob 2>/dev/null | tail -1" </dev/null | tr -d '\r')
  [ -n "$d" ] && { mkdir -p "$R/$sub"; adb pull "$d" "$R/$sub" >/dev/null 2>&1 </dev/null; log "$script pulled from $d"; }
}

log "START core queue"
device_campaign bmoe_arena2.sh 3 bmoe_arena2 'bmoe_arena2_*/'
sh "$HOSTDIR/agg_bandwidth.sh" 3 >> "$LOG" 2>&1; log "agg_bandwidth done"
sh "$HOSTDIR/matmul_sweep.sh" 2 >> "$LOG" 2>&1; log "matmul_sweep done"
sh "$HOSTDIR/thread_sweep.sh" 3 >> "$LOG" 2>&1; log "thread_sweep done"
sh "$HOSTDIR/app_vs_shell.sh" 3 >> "$LOG" 2>&1; log "app_vs_shell done"
log "CORE QUEUE DONE"
touch "$R/CHAIN_CORE_DONE"
