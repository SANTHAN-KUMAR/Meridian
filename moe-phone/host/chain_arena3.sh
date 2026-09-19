#!/bin/bash
# chain_arena3.sh — anonymous slot arena A/B, awake-held (device/bmoe_arena3.sh), after chain_v4; from chain_v4.sh — tier A/B with the variant chosen from the gx v4 bench (GTIER_VARIANT2), engine 0024 (gx 5e00544), phone on USB; from chain_awake.sh — cap-8 A/B with the phone held Awake (bmoe_gtier.sh REVISION 04:55); from chain_cap8.sh — cap-8 follow-up (pre-registered in bmoe_gtier.sh 03:05) after chain_tier; derived from chain_tier.sh — tier smoke + A/B only (after the 02:23 smoke's flag bug); derived from chain_night3, which takes over from chain_night2 during the kgsl A/B (which runs on the phone under setsid, independent
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
log "[arena3] waiting for CHAIN_V4_DONE"
while [ ! -f "$R/CHAIN_V4_DONE" ]; do sleep 30; done
reconnect
[ -n "$(lock)" ] && { log "[arena3] phone busy: $(lock)"; exit 1; }
orph=$(A 'ps -A -o ARGS' | grep -E 'bmoe-cli|gx_|sh bmoe_|pinprobe' | grep -v grep)
[ -n "$orph" ] && { log "[arena3] FATAL foreign: $orph"; exit 1; }
SP=/tmp/claude-1000/-run-media-santhankumar-New-Volume-identifying-variation/2d806460-912e-4de7-a437-dc77597a5bb6/scratchpad
# repack_bench (pre-registered in host/repack_bench/repack_bench.cpp), ~1-2 min, under the lock, screen held awake
A "echo 'repack_bench $(date +%H:%M:%S)' > $H/.phone_busy; mkdir -p /data/local/tmp/repack_bench" >/dev/null
adb push "$MP/host/repack_bench/out/repack_bench" /data/local/tmp/repack_bench/ >/dev/null 2>&1 </dev/null || { log "[arena3] FATAL repack_bench push"; A "rm -f $H/.phone_busy"; exit 1; }
A "settings put system screen_off_timeout 1800000; input keyevent KEYCODE_WAKEUP" >/dev/null 2>&1
mkdir -p "$R/repack_bench"
A "cd /data/local/tmp/repack_bench && chmod 755 repack_bench && dumpsys power | grep -m1 mWakefulness= && LD_LIBRARY_PATH=$H/bmoe-i8mm-0024 ./repack_bench 300 2>/dev/null && dumpsys power | grep -m1 mWakefulness=" > "$R/repack_bench/repack_bench_phone.out" 2>&1
A "rm -f $H/.phone_busy" >/dev/null
log "[arena3] repack_bench: $(grep -E '^(RESULT|VERDICT)|mWakefulness' "$R/repack_bench/repack_bench_phone.out" | tr '\n' ' ')"
# repack engine A/B first, only on the phone microbench's pre-registered VERDICT BUILD
if grep -q "^VERDICT BUILD" "$R/repack_bench/repack_bench_phone.out"; then
  adb push "$MP/device/bmoe_repack.sh" "$H/" >/dev/null 2>&1 </dev/null
  export GTENV="GT_BIN=bmoe-i8mm-0024 GT_PIN=$(cat "$SP/phone_pin")"
  A "svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable" >/dev/null 2>&1
  campaign bmoe_repack.sh smoke
  S="$LAST"; ok=1
  [ "$(grep -c 'text_match .* OK' "$S/log.txt")" = 2 ] || ok=0
  grep -q FATAL "$S"/*.err && ok=0
  grep -q "repack-experts: 48 expert tensors on repacked" "$S/stack_reps1.err" || ok=0
  [ "$(grep -c 'wake_start=Awake' "$S/log.txt")" = 2 ] || ok=0
  [ "$(grep -c 'AFTER wake=Awake' "$S/log.txt")" = 2 ] || ok=0
  log "[repack] smoke ok=$ok :: $(grep -h 'generation:' "$S"/*.out | tr '\n' ' ')"
  b=$(A "dumpsys battery | grep -m1 ' level' | tr -dc 0-9")
  if [ $ok = 1 ] && [ "${b:-0}" -ge 30 ]; then log "[repack] A/B start (battery $b%)"; campaign bmoe_repack.sh ab; log "[repack] A/B pulled: $LAST"; else log "[repack] A/B not started (smoke ok=$ok, battery ${b}%)"; fi
else
  log "[repack] phone microbench not BUILD: engine A/B not run"
fi
adb push "$MP/device/bmoe_arena3.sh" "$H/" >/dev/null 2>&1 </dev/null
A "grep -q 'slot-arena --cache-ceil-mb 4600' $H/bmoe_arena3.sh" || { log "[arena3] FATAL pushed script wrong"; exit 1; }
export GTENV="GT_BIN=bmoe-i8mm-0024 GT_PIN=$(cat "$SP/phone_pin")"
A "svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable" >/dev/null 2>&1
campaign bmoe_arena3.sh smoke
S="$LAST"; ok=1
[ "$(grep -c 'text_match .* OK' "$S/log.txt")" = 2 ] || ok=0
grep -q FATAL "$S"/*.err && ok=0
grep -q "slot-arena:" "$S/stack_reps1.err" || ok=0
[ "$(grep -c 'wake_start=Awake' "$S/log.txt")" = 2 ] || ok=0
[ "$(grep -c 'AFTER wake=Awake' "$S/log.txt")" = 2 ] || ok=0
log "[arena3] smoke ok=$ok :: $(grep -h 'generation:' "$S"/*.out | tr '\n' ' ') :: $(grep -h 'slot-arena:' "$S/stack_reps1.err")"
b=$(A "dumpsys battery | grep -m1 ' level' | tr -dc 0-9")
if [ $ok = 1 ] && [ "${b:-0}" -ge 30 ]; then log "[arena3] A/B start (battery $b%)"; campaign bmoe_arena3.sh ab; log "[arena3] A/B pulled: $LAST"; else log "[arena3] A/B not started (smoke ok=$ok, battery ${b}%)"; fi
restore
touch "$R/CHAIN_ARENA3_DONE"
