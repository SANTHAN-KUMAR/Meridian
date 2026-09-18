#!/bin/bash
# chain_tier.sh — tier smoke + A/B only (after the 02:23 smoke's flag bug); derived from chain_night3, which takes over from chain_night2 during the kgsl A/B (which runs on the phone under setsid, independent
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
log "[tier] start (script fixed: --gpu-spin-wait 1)"
reconnect
[ -n "$(lock)" ] && { log "[tier] phone busy: $(lock)"; exit 1; }
orph=$(A 'ps -A -o ARGS' | grep -E 'bmoe-cli|gx_|sh bmoe_|pinprobe' | grep -v grep)
[ -n "$orph" ] && { log "[tier] FATAL foreign: $orph"; exit 1; }
SP=/tmp/claude-1000/-run-media-santhankumar-New-Volume-identifying-variation/2d806460-912e-4de7-a437-dc77597a5bb6/scratchpad
want=$(grep bmoe-cli "$SP/bmoe-i8mm-0022/MD5" | cut -d' ' -f1); got=$(A "md5sum $H/bmoe-i8mm-0022/bmoe-cli" | cut -d' ' -f1)
[ "$want" = "$got" ] || { log "[tier] FATAL 0022 md5 $got != $want"; exit 1; }
adb push "$MP/device/bmoe_gtier.sh" "$H/" >/dev/null 2>&1 </dev/null
A "grep -q 'gpu-spin-wait 1' $H/bmoe_gtier.sh" || { log "[tier] FATAL pushed script lacks the fix"; exit 1; }
V=$(cat "$R/GTIER_VARIANT" | tr -dc 0-9)
export GTENV="GT_BIN=bmoe-i8mm-0022 GT_VARIANT=$V GT_SPIN=1"
A "svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable" >/dev/null 2>&1
campaign bmoe_gtier.sh smoke
S="$LAST"; ok=1
[ "$(grep -c 'text_match .* OK' "$S/log.txt")" = 2 ] || ok=0
grep -q FATAL "$S"/*.err && ok=0
grep -q "dispatches [1-9]" "$S/stack_reps1.err" || ok=0
grep -q "warm start filled [1-9]" "$S/stack_reps1.err" || ok=0
dev=$(grep -ho "experts on device [0-9]*" "$S/stack_reps1.err" | grep -o "[0-9]*$"); rk=$(grep -ho "recomputed on the CPU [0-9]*" "$S/stack_reps1.err" | grep -o "[0-9]*$")
[ -n "$dev" ] && [ -n "$rk" ] && [ $((rk * 100)) -le "$dev" ] || ok=0
log "[tier] smoke ok=$ok variant=$V dev=$dev risk=$rk :: $(grep -h 'generation:' "$S"/*.out | tr '\n' ' ')"
if [ $ok = 1 ]; then log "[tier] A/B start"; campaign bmoe_gtier.sh ab; log "[tier] A/B pulled: $LAST"; fi
restore
touch "$R/CHAIN_TIER_DONE"
