#!/bin/bash
# chain_next.sh — the core queue after the 2026-09-18 midday reboot, in the order the laptop work made
# worth running, with the phone's heat treated as a first-class constraint instead of an afterthought.
#
# Starts on USB or wireless. On USB it re-enables tcpip 5555 first, so the cable can come out.
#
#   PHASE A — clock-INDEPENDENT (safe to run on a warm phone). The engine's cache counters (hit rate,
#   MiB read per token) are fixed by the routing, the budget and the policy; across twelve rows that
#   spanned thermal status 0-3 and 5.39-6.99 tok/s they were identical to the decimal.
#     1. bmoe_hitrate.sh  cache budget 5000 / 7000 / 8500 MiB. The ceiling has been 5000 since a day
#                         with ~6 GB free; the phone now reports ~7.6 GB. Simulation projects 58.3 vs
#                         119.8 MiB/token at 7000 (gates/cache_projection.py) -- this checks it.
#     2. bmoe_slru.sh     segmented LRU (patch 0011) vs LRU, ABBA-balanced; host run: hit 40.6 -> 46.4%.
#
#   PHASE B — clock-SENSITIVE. Every row is a rate, so each campaign first waits for the phone to cool
#   (thermal status <= 1, capped at 20 minutes, the wait and the state it ended in are logged). The
#   thermal status comes from the framework, not from our own result, so the gate cannot be tuned
#   against the metric it protects.
#     3. bmoe_arena2.sh   slot arena re-run (cache mgmt 34 -> 2 ms/token originally; killed after one row)
#     4. agg_bandwidth.sh do two devices ADD throughput?
#     5. matmul_sweep.sh  per-op truth per device, and Qwen3-shaped ceilings
#     6. thread_sweep.sh  is 4 threads right now that nothing faults?
#
# The memory guard that the midday hang exposed is here too: no device campaign starts while
# MemAvailable is below what the engine needs, and the phone is left in its normal state at the end
# (screen timeout, stay-awake off, doze on) so it is usable the moment the queue finishes.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
HOSTDIR="$(cd "$(dirname "$0")" && pwd)"
DEV="$(cd "$HOSTDIR/../device" && pwd)"
LOG="$R/chain_next.log"
mkdir -p "$R"
log() { echo "$(date -Iseconds) $*" >> "$LOG"; }

# --- 0. get a connection: prefer wireless, but if a USB device is present, switch it to tcpip first
log "waiting for the phone (USB or wireless)"
while :; do
  usb=$(adb devices | awk 'NR>1 && $2=="device" && $1 !~ /:/ {print $1; exit}')
  if [ -n "$usb" ]; then
    log "USB device $usb present: enabling tcpip 5555"
    ANDROID_SERIAL=$usb adb tcpip 5555 >/dev/null 2>&1; sleep 6
  fi
  adb connect "$ANDROID_SERIAL" >/dev/null 2>&1
  adb shell 'echo up' >/dev/null 2>&1 </dev/null && break
  sleep 15
done
log "connected; uptime $(adb shell 'cut -d" " -f1 /proc/uptime' </dev/null | tr -d '\r')s"

A() { adb shell "$@" </dev/null; }
status() { A 'dumpsys thermalservice 2>/dev/null' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'; }
memavail_mib() { A "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo" | tr -d '\r'; }

# the phone state a reboot clears
A 'svc power stayon true' >/dev/null 2>&1
A 'settings put system screen_off_timeout 1800000' >/dev/null 2>&1
A 'dumpsys deviceidle disable' >/dev/null 2>&1
for f in thermal_gate.sh bmoe_hitrate.sh bmoe_slru.sh bmoe_arena2.sh; do
  adb push "$DEV/$f" /data/local/tmp/moe-stream/ >/dev/null 2>&1 </dev/null
done
log "scripts pushed"

quiesce() {
  A 'for p in $(pm list packages -3 | sed "s/^package://"); do am force-stop $p; done; am kill-all' >/dev/null 2>&1
}

cool_gate() {  # max_seconds
  local max=$1 t=0 s
  s=$(status)
  while [ "${s:-9}" -gt 1 ] && [ $t -lt "$max" ]; do sleep 30; t=$((t + 30)); s=$(status); done
  log "cool_gate: waited ${t}s, thermal status now ${s:-?}"
}

mem_gate() {  # min_mib
  local need=$1 have
  quiesce; sleep 5
  have=$(memavail_mib)
  log "mem_gate: MemAvailable ${have:-?} MiB (need >= $need)"
  [ "${have:-0}" -ge "$need" ]
}

stop_campaigns() {
  for pat in 'sh bmoe_' 'bmoe-cli' 'keep_awake'; do
    pids=$(A "ps -A -o PID,ARGS" | awk -v p="$pat" '$0 ~ p && $0 !~ /awk/ {print $1}')
    [ -n "$pids" ] && { A "kill -9 $pids" >/dev/null 2>&1; log "stopped [$pat]: $pids"; }
  done
}

device_campaign() {  # script reps results_subdir dir_glob
  local script=$1 reps=$2 sub=$3 glob=$4
  stop_campaigns
  log "$script start"
  A "cd /data/local/tmp/moe-stream && (setsid nohup sh $script $reps > ${script%.sh}_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
  sleep 60
  local lost=0 d running
  while :; do
    if ! adb shell 'echo up' >/dev/null 2>&1 </dev/null; then
      adb connect "$ANDROID_SERIAL" >/dev/null 2>&1; lost=$((lost + 1))
      [ $lost -gt 40 ] && { log "$script: phone gone 10 min, giving up on it"; break; }
      sleep 15; continue
    fi
    d=$(A "ls -d /data/local/tmp/moe-stream/$glob 2>/dev/null | tail -1" | tr -d '\r')
    [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
    running=$(A "ps -A -o PID,ARGS" | awk -v p="sh $script" '$0 ~ p && $0 !~ /awk/ {print $1}')
    [ -z "$running" ] && { log "$script exited without DONE ('$d')"; break; }
    sleep 90
  done
  d=$(A "ls -d /data/local/tmp/moe-stream/$glob 2>/dev/null | tail -1" | tr -d '\r')
  [ -n "$d" ] && { mkdir -p "$R/$sub"; adb pull "$d" "$R/$sub" >/dev/null 2>&1 </dev/null; log "$script pulled from $d"; }
}

restore_phone() {
  stop_campaigns
  A 'am force-stop com.moephone.bmoe3; am force-stop com.moephone.npu2' >/dev/null 2>&1
  A 'svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
  log "phone restored to normal settings"
}
trap restore_phone EXIT

# --- PHASE A: clock-independent
if mem_gate 4000; then device_campaign bmoe_hitrate.sh 2 bmoe_hitrate 'bmoe_hitrate_*/'
else log "SKIPPED bmoe_hitrate: not enough free memory for a 7000 MiB cell"; fi
if mem_gate 5000; then device_campaign bmoe_slru.sh 3 bmoe_slru 'bmoe_slru_*/'
else log "SKIPPED bmoe_slru: not enough free memory"; fi

# --- PHASE B: clock-sensitive, each behind a cooldown
cool_gate 1200; mem_gate 5000 && device_campaign bmoe_arena2.sh 3 bmoe_arena2 'bmoe_arena2_*/'
cool_gate 1200; sh "$HOSTDIR/agg_bandwidth.sh" 3 >> "$LOG" 2>&1; log "agg_bandwidth done"
cool_gate 1200; sh "$HOSTDIR/matmul_sweep.sh" 2 >> "$LOG" 2>&1; log "matmul_sweep done"
cool_gate 1200; sh "$HOSTDIR/thread_sweep.sh" 3 >> "$LOG" 2>&1; log "thread_sweep done"

log "QUEUE DONE"
touch "$R/CHAIN_NEXT_DONE"
