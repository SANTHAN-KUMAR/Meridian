#!/bin/bash
# chain_after.sh — the two re-runs the 14:45 audit made necessary, after chain_arena.sh finishes.
#
# 1. SLRU DECODE, CLEAN. The first SLRU campaign (bmoe_slru_20260918_1408) ran while an orphaned
#    agg_bandwidth.sh driver (START 13:40, DONE 14:35) was launching llama-bench in com.moephone.npu2 on
#    the same phone. Its cache counters are unaffected (clock-independent: lru_rep1b ran at 0.205 s/token
#    and still read exactly 119.8 MiB/token), but its decode rates are not interpretable, and the
#    "+4 to +7%" decode range quoted in commit acef356 is retracted.
#    PRE-REGISTERED (written 2026-09-18 ~14:50, before any row of the re-run exists):
#      outcome   decode tok/s per row, as bmoe-cli prints it ("generation: ... s/token")
#      keep row  iff exit=0 AND the engine's granted budget is 5000 MiB AND foreign=[] in its log line
#      estimate  median(slru kept rows) / median(lru kept rows) - 1, reported with every kept row
#      verdict   decode claim made only if every kept SLRU row's pair-mate in the same ABBA repeat is
#                slower (sign holds within all repeats); otherwise "no decode effect resolved at n=3"
# 2. AGG_BANDWIDTH, FIXED. The CPU arm ran on the laptop (missing `adb shell`) and measured nothing, so
#    every "pair" was a single device. The fixed driver runs it on the phone and holds a lock.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
HOSTDIR="$(cd "$(dirname "$0")" && pwd)"
DEV="$(cd "$HOSTDIR/../device" && pwd)"
LOG="$R/chain_after.log"
log() { echo "$(date -Iseconds) $*" >> "$LOG"; }
A() { adb shell "$@" </dev/null; }
status() { A 'dumpsys thermalservice 2>/dev/null' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'; }
cool_gate() { local t=0 s; s=$(status); while [ "${s:-9}" -gt 1 ] && [ $t -lt 1200 ]; do sleep 30; t=$((t+30)); s=$(status); done; log "cool_gate waited ${t}s, status ${s:-?}"; }

log "waiting for chain_arena"
while [ ! -f "$R/CHAIN_ARENA_DONE" ]; do sleep 60; done
adb connect "$ANDROID_SERIAL" >/dev/null 2>&1
A 'svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable' >/dev/null 2>&1
adb push "$DEV/thermal_gate.sh" "$DEV/bmoe_slru.sh" /data/local/tmp/moe-stream/ >/dev/null 2>&1 </dev/null
[ "$(A 'ls /data/local/tmp/moe-stream/bmoe-i8mm-slru/bmoe-cli 2>/dev/null | wc -l' | tr -d '\r')" = 1 ] || { log "FATAL: slru binary missing"; exit 1; }
# nothing else of ours may be running on the phone or the laptop
others=$(ps -eo args | grep -E 'agg_bandwidth|matmul_sweep|thread_sweep|npu_tuning|chain_(next|arena|final|core|resume)' | grep -v grep)
[ -n "$others" ] && log "WARNING other drivers alive on the laptop: $(echo "$others" | tr '\n' ';')"

cool_gate
log "bmoe_slru.sh (decode re-run) start"
A "cd /data/local/tmp/moe-stream && (setsid nohup sh bmoe_slru.sh 3 > bmoe_slru_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
sleep 60
while :; do
  d=$(A "ls -d /data/local/tmp/moe-stream/bmoe_slru_*/ | tail -1" | tr -d '\r')
  [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
  [ -z "$(A 'ps -A -o ARGS' | grep 'sh bmoe_slru')" ] && { log "slru exited without DONE ($d)"; break; }
  sleep 90
done
mkdir -p "$R/bmoe_slru_decode"; adb pull "$d" "$R/bmoe_slru_decode" >/dev/null 2>&1 </dev/null; log "pulled $d"

cool_gate
A 'am force-stop com.moephone.bmoe3' >/dev/null 2>&1
bash "$HOSTDIR/agg_bandwidth.sh" 3 >> "$LOG" 2>&1; log "agg_bandwidth done (exit $?)"

A 'am force-stop com.moephone.bmoe3; am force-stop com.moephone.npu2; svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
log "phone restored"; touch "$R/CHAIN_AFTER_DONE"
