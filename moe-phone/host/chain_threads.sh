#!/bin/bash
# chain_threads.sh — the last two phases of the 2026-09-18 night, both prompted by what the earlier
# phases turned up:
#   1. thread_sweep.sh   the 4-thread pin came from a thread cliff measured under a page-fault storm;
#                        on a resident model in-app the CPU already beat that campaign's best, so the
#                        cliff may not be about cores at all. 2/4/6/8 threads, rotated.
#   2. app_vs_shell.sh   the first in-app row of the attention A/B was 26% under the shell baseline, and
#                        its own log blames a smaller granted cache budget and a 1.65 GHz thermal cap
#                        rather than the app domain. This campaign idles first and alternates the arms,
#                        so the domain cost can be separated from the temperature.
# Then the phone goes back to the memory campaign.
set -u
export ANDROID_SERIAL=192.168.0.65:5555
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
HOSTDIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$R/chain_threads.log"
log() { echo "$(date -Iseconds) $*" >> "$LOG"; }

log "waiting for chain_matmul"
while [ ! -f "$R/CHAIN_MATMUL_DONE" ]; do sleep 60; done

for pat in 'sh bmoe_mem\.sh' 'bmoe-cli'; do
  pids=$(adb shell "ps -A -o PID,ARGS" | awk -v p="$pat" '$0 ~ p && $0 !~ /awk/ {print $1}')
  [ -n "$pids" ] && { log "stopping [$pat]: $pids"; adb shell "kill $pids" >/dev/null 2>&1; sleep 10; }
done

log "thread_sweep start"
sh "$HOSTDIR/thread_sweep.sh" 3 >> "$LOG" 2>&1
log "thread_sweep done"

log "app_vs_shell start"
sh "$HOSTDIR/app_vs_shell.sh" 3 >> "$LOG" 2>&1
log "app_vs_shell done"

adb shell 'cd /data/local/tmp/moe-stream && echo "chain_threads: bmoe_mem $(date)" >> phone_queue.log && (setsid nohup sh bmoe_mem.sh 3 > bmoe_mem_nohup.log 2>&1 < /dev/null &)' >/dev/null 2>&1
log "bmoe_mem launched"
touch "$R/CHAIN_THREADS_DONE"
