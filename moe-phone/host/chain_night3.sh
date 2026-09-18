#!/bin/bash
# chain_night3.sh — takes over from chain_night2 during the kgsl A/B (which runs on the phone under setsid, independent
# of this host script):
#   1. wait for bmoe_kgsl_ab_*/DONE on the phone, pull it
#   2. the gx session's window (unrolled-accumulator kernels): wait up to 15 min for its lock, then until it clears
#   3. GPU tier smoke + A/B. Variant: $R/GTIER_VARIANT (written after the gx window from its bench; default 0), spin 1
#   4. restore the phone
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
MP="/run/media/santhankumar/New Volume/identifying-variation/moe-phone"
R="$MP/results/2026-09-19"
H=/data/local/tmp/moe-stream
LOG="$R/chain_night2.log"; log() { echo "$(date -Iseconds) [night3] $*" | tee -a "$LOG"; }
A() { adb shell "$@" </dev/null; }
reconnect() { adb shell 'echo up' </dev/null 2>/dev/null | grep -q up || adb connect "$ANDROID_SERIAL" >/dev/null 2>&1; }
lock() { A "cat $H/.phone_busy 2>/dev/null" | tr -d '\r'; }
restore() { A 'svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1; log "phone restored"; }
finish() { restore; touch "$R/CHAIN_NIGHT3_DONE"; exit "${1:-0}"; }
campaign() {  # script mode
  local d
  A "cd $H && (${GTENV:-} setsid nohup sh $1 $2 > ${1%.sh}_$2_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
  sleep 45
  while :; do
    reconnect
    d=$(A "ls -d $H/${1%.sh}_${2}_*/ | tail -1" | tr -d '\r')
    [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
    [ -n "$d" ] && A "[ -f ${d}REFUSED ]" && { log "REFUSED: $(A "cat ${d}REFUSED")"; break; }
    [ -z "$(A 'ps -A -o ARGS' | grep "sh $1" | grep -v grep)" ] && { log "$1 $2 exited without DONE ($d)"; break; }
    sleep 45
  done
  mkdir -p "$R/${1%.sh}"; adb pull "$d" "$R/${1%.sh}" >/dev/null 2>&1 </dev/null; log "pulled $d"
  LAST="$R/${1%.sh}/$(basename "$d")"
}
log "start (0022 engine): waiting for the kgsl A/B"
while :; do
  reconnect
  d=$(A "ls -d $H/bmoe_kgsl_ab_*/ 2>/dev/null | tail -1" | tr -d '\r')
  [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
  [ -n "$d" ] && [ -z "$(A 'ps -A -o ARGS' | grep 'sh bmoe_kgsl' | grep -v grep)" ] && [ -z "$(lock)" ] && { log "kgsl A/B ended without DONE ($d)"; break; }
  sleep 60
done
mkdir -p "$R/bmoe_kgsl"; adb pull "$d" "$R/bmoe_kgsl" >/dev/null 2>&1 </dev/null; log "kgsl A/B pulled: $R/bmoe_kgsl/$(basename "$d")"
w=0; while [ $w -lt 15 ] && [ -z "$(lock)" ]; do sleep 60; w=$((w+1)); done
if [ -n "$(lock)" ]; then
  log "gx window: $(lock)"; w=0
  while [ $w -lt 60 ] && [ -n "$(lock)" ]; do sleep 60; w=$((w+1)); reconnect; done
  [ -n "$(lock)" ] && { log "lock still held after 60 min; stopping"; finish 1; }
  log "gx window closed after ${w} min"
else
  log "gx window: no gx run within 15 min"
fi
# the variant decision is written by the agent from the gx bench; wait up to 10 min for it, else v0
w=0; while [ $w -lt 10 ] && [ ! -f "$R/GTIER_VARIANT" ]; do sleep 60; w=$((w+1)); done
V=$(cat "$R/GTIER_VARIANT" 2>/dev/null | tr -dc 0-9); V=${V:-0}
# engine bmoe-i8mm-0022: libgx 58b13dd (unrolled float4 accumulators, bit-exact); pushed and md5-checked here
SP=/tmp/claude-1000/-run-media-santhankumar-New-Volume-identifying-variation/2d806460-912e-4de7-a437-dc77597a5bb6/scratchpad
A "mkdir -p $H/bmoe-i8mm-0022" >/dev/null
for f in "$SP/bmoe-i8mm-0022"/*; do adb push "$f" "$H/bmoe-i8mm-0022/" >/dev/null 2>&1 </dev/null; done
adb push "$MP/device/bmoe_gtier.sh" "$H/" >/dev/null 2>&1 </dev/null
want=$(grep bmoe-cli "$SP/bmoe-i8mm-0022/MD5" | cut -d' ' -f1); got=$(A "md5sum $H/bmoe-i8mm-0022/bmoe-cli" | cut -d' ' -f1)
[ "$want" = "$got" ] || { log "FATAL: pushed 0022 md5 $got != $want"; finish 1; }
A "chmod 755 $H/bmoe-i8mm-0022/bmoe-cli" >/dev/null
export GTENV="GT_BIN=bmoe-i8mm-0022 GT_VARIANT=$V GT_SPIN=1"
log "gtier variant $V (GTIER_VARIANT file: $( [ -f "$R/GTIER_VARIANT" ] && echo present || echo absent ))"
A "svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable" >/dev/null 2>&1
campaign bmoe_gtier.sh smoke
S="$LAST"; ok=1
[ "$(grep -c 'text_match .* OK' "$S/log.txt")" = 2 ] || ok=0
grep -q FATAL "$S"/*.err && ok=0
grep -q "dispatches [1-9]" "$S/stack_reps1.err" || ok=0
grep -q "warm start filled [1-9]" "$S/stack_reps1.err" || ok=0
dev=$(grep -ho "experts on device [0-9]*" "$S/stack_reps1.err" | grep -o "[0-9]*$"); rk=$(grep -ho "recomputed on the CPU [0-9]*" "$S/stack_reps1.err" | grep -o "[0-9]*$")
[ -n "$dev" ] && [ -n "$rk" ] && [ $((rk * 100)) -le "$dev" ] || ok=0
log "gtier smoke ok=$ok dev=$dev risk=$rk :: $(grep -h 'generation:' "$S"/*.out | tr '\n' ' ')"
if [ $ok = 1 ]; then log "gtier A/B start"; campaign bmoe_gtier.sh ab; log "gtier A/B pulled: $LAST"; fi
finish 0
