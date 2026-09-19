#!/bin/bash
# chain_h2h.sh — head-to-head vs BigMoeOnEdge (device/bmoe_h2h.sh) after chain_prerepack; from chain_prerepack.sh — pre-repacked experts + dense repack (device/bmoe_prerepack.sh) after the arena3 A/B; from chain_arena3.sh — anonymous slot arena A/B, awake-held (device/bmoe_arena3.sh), after chain_v4; from chain_v4.sh — tier A/B with the variant chosen from the gx v4 bench (GTIER_VARIANT2), engine 0024 (gx 5e00544), phone on USB; from chain_awake.sh — cap-8 A/B with the phone held Awake (bmoe_gtier.sh REVISION 04:55); from chain_cap8.sh — cap-8 follow-up (pre-registered in bmoe_gtier.sh 03:05) after chain_tier; derived from chain_tier.sh — tier smoke + A/B only (after the 02:23 smoke's flag bug); derived from chain_night3, which takes over from chain_night2 during the kgsl A/B (which runs on the phone under setsid, independent
# of this host script):
#   1. wait for bmoe_kgsl_ab_*/DONE on the phone, pull it
#   2. the gx session's window (unrolled-accumulator kernels): wait up to 15 min for its lock, then until it clears
#   3. GPU tier smoke + A/B. Variant: $R/GTIER_VARIANT (written after the gx window from its bench; default 0), spin 1
#   4. restore the phone
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-3C15CK0028J00000}
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
log "[h2h] waiting for CHAIN_PREREPACK_DONE"
while [ ! -f "$R/CHAIN_PREREPACK_DONE" ]; do sleep 60; done
OURS=plain
PA=$(ls -d "$R"/bmoe_prerepack/bmoe_prerepack_ab_* 2>/dev/null | tail -1)
if [ -n "$PA" ] && python3 "$MP/gates/stack_summary.py" "$PA" --any-budget-prereg --require-awake --out "$R/prerepack_ab_summary.json" > /dev/null 2>&1; then
  OURS=$(python3 -c "
import json;d=json.load(open('$R/prerepack_ab_summary.json'))['results']['decode_tok_s']
print('repack' if d.get('decisive') and d.get('mean_diff',0)>0 else 'plain')")
fi
log "[h2h] ours=$OURS (from the prerepack A/B: ${PA:-none})"
reconnect
[ -n "$(lock)" ] && { log "[h2h] phone busy: $(lock)"; exit 1; }
SP=/tmp/claude-1000/-run-media-santhankumar-New-Volume-identifying-variation/2d806460-912e-4de7-a437-dc77597a5bb6/scratchpad
adb push "$MP/device/bmoe_h2h.sh" "$H/" >/dev/null 2>&1 </dev/null
A "[ -x $H/bmoe-ref/bmoe-cli ]" || { log "[h2h] FATAL no reference build on the phone"; exit 1; }
export GTENV="GT_BIN=bmoe-i8mm-0025 H2H_OURS=$OURS GT_PIN=$(cat "$SP/phone_pin")"
A "svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable" >/dev/null 2>&1
campaign bmoe_h2h.sh smoke
S="$LAST"; ok=1
[ "$(grep -c '^exit=0' "$S/log.txt")" = 2 ] || ok=0
grep -q FATAL "$S"/*.err && ok=0
[ "$(grep -c 'wake_start=Awake' "$S/log.txt")" = 2 ] || ok=0
[ "$(grep -c 'AFTER wake=Awake' "$S/log.txt")" = 2 ] || ok=0
log "[h2h] smoke ok=$ok :: $(grep -h 'generation:' "$S"/*.out | tr '\n' ' ') :: $(grep -o 'text_match.*' "$S/log.txt" | tr '\n' ' ')"
b=$(A "dumpsys battery | grep -m1 ' level' | tr -dc 0-9")
if [ $ok = 1 ] && [ "${b:-0}" -ge 30 ]; then log "[h2h] A/B start (battery $b%)"; campaign bmoe_h2h.sh ab; log "[h2h] A/B pulled: $LAST"; else log "[h2h] A/B not started (smoke ok=$ok, battery ${b}%)"; fi
restore
touch "$R/CHAIN_H2H_DONE"
