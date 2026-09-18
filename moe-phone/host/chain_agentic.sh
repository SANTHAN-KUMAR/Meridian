#!/bin/bash
# chain_agentic.sh — run the agentic-workload measurements after the 2026-09-18 chain drains.
#
# WHY IT IS SEPARATE FROM chain_resume.sh. That chain is already running; editing a live bash script
# is not safe (bash re-reads the file from its current byte offset), so this waits for its sentinel
# instead of being appended to it. It also keeps the side track's phone time visibly separate from the
# main campaign's, which is the point of running it as a side track at all.
#
# WHAT IT RUNS, AND WHY THIS IS THE DECISIVE ROW. Every prefill figure this project holds was measured
# at a ~26-token prompt, so it is a cold-start artifact, not an ingestion rate. A phone-automation agent
# re-sends a UI snapshot every turn and generates a short structured tool call, so its turn cost is
# dominated by prefill over ~1-4k tokens, not by decode. bmoe_prefill.sh walks 128 / 512 / 2048 / 8192
# tokens at the best known configuration, which brackets that range on both sides.
#
# NOT ARMED. This was started once on 2026-09-18 and stopped before it ran anything: the side track
# is not to take phone time, and the main campaign has priority. It refuses to run unless the owner
# says so explicitly, so that neither a future session nor a stray shell can spend the device on it:
#
#   AGENTIC_PHONE_TIME_APPROVED=1 ANDROID_SERIAL=192.168.0.65:5555 bash chain_agentic.sh &
set -u
if [ "${AGENTIC_PHONE_TIME_APPROVED:-0}" != "1" ]; then
  echo "chain_agentic.sh: refusing to run. This side track has no phone-time allocation." >&2
  echo "Set AGENTIC_PHONE_TIME_APPROVED=1 only when the owner has said the device is free." >&2
  exit 3
fi
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
HOSTDIR="$(cd "$(dirname "$0")" && pwd)"
DEV="$(cd "$HOSTDIR/../device" && pwd)"
LOG="$R/chain_agentic.log"
mkdir -p "$R"
log() { echo "$(date -Iseconds) $*" >> "$LOG"; }

adb_up() { adb connect "$ANDROID_SERIAL" >/dev/null 2>&1; adb shell 'echo up' >/dev/null 2>&1; }

# 1. wait for chain_resume.sh's sentinel
log "waiting for CHAIN_RESUME_DONE"
while [ ! -f "$R/CHAIN_RESUME_DONE" ]; do sleep 120; done
log "CHAIN_RESUME_DONE seen"

# 2. chain_resume launches bmoe_mem.sh in the background and touches the sentinel immediately, so the
#    phone is still busy at that point. Wait for that campaign's own DONE file before asking for the SoC.
log "waiting for bmoe_mem to finish"
lost=0
while :; do
  if ! adb_up; then
    lost=$((lost + 1))
    [ $lost -gt 240 ] && { log "phone unreachable for 1 h while waiting for bmoe_mem; giving up"; exit 1; }
    sleep 15; continue
  fi
  lost=0
  d=$(adb shell "ls -d /data/local/tmp/moe-stream/bmoe_mem_*/ 2>/dev/null | tail -1" | tr -d '\r')
  if [ -n "$d" ] && adb shell "[ -f ${d}DONE ]"; then log "bmoe_mem done ($d)"; break; fi
  running=$(adb shell "ps -A -o PID,ARGS" | awk '$0 ~ /sh bmoe_mem\.sh/ && $0 !~ /awk/ {print $1}')
  if [ -z "$running" ]; then log "bmoe_mem not running and no DONE (dir '$d'); proceeding anyway"; break; fi
  sleep 120
done

# 3. the prefill ladder
adb push "$DEV/bmoe_prefill.sh" /data/local/tmp/moe-stream/ >/dev/null 2>&1
adb push "$DEV/thermal_gate.sh" /data/local/tmp/moe-stream/ >/dev/null 2>&1
log "bmoe_prefill.sh start"
adb shell "cd /data/local/tmp/moe-stream && echo 'chain_agentic: bmoe_prefill \$(date)' >> phone_queue.log && (setsid nohup sh bmoe_prefill.sh 2 > bmoe_prefill_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
sleep 60
lost=0
while :; do
  if ! adb_up; then
    lost=$((lost + 1))
    [ $lost -gt 40 ] && { log "lost the phone for 10 min during bmoe_prefill"; break; }
    sleep 15; continue
  fi
  lost=0
  d=$(adb shell "ls -d /data/local/tmp/moe-stream/bmoe_prefill_*/ 2>/dev/null | tail -1" | tr -d '\r')
  if [ -n "$d" ] && adb shell "[ -f ${d}DONE ]"; then break; fi
  running=$(adb shell "ps -A -o PID,ARGS" | awk '$0 ~ /sh bmoe_prefill\.sh/ && $0 !~ /awk/ {print $1}')
  if [ -z "$running" ]; then log "bmoe_prefill exited without DONE (dir '$d')"; break; fi
  sleep 90
done
d=$(adb shell "ls -d /data/local/tmp/moe-stream/bmoe_prefill_*/ 2>/dev/null | tail -1" | tr -d '\r')
[ -n "$d" ] && { mkdir -p "$R/bmoe_prefill"; adb pull "$d" "$R/bmoe_prefill" >/dev/null 2>&1; log "bmoe_prefill pulled from $d"; }
touch "$R/CHAIN_AGENTIC_DONE"
log "done"
