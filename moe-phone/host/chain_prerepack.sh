#!/bin/bash
# chain_prerepack.sh — pre-repacked experts + dense repack (device/bmoe_prerepack.sh) after the arena3 A/B; from chain_arena3.sh — anonymous slot arena A/B, awake-held (device/bmoe_arena3.sh), after chain_v4; from chain_v4.sh — tier A/B with the variant chosen from the gx v4 bench (GTIER_VARIANT2), engine 0024 (gx 5e00544), phone on USB; from chain_awake.sh — cap-8 A/B with the phone held Awake (bmoe_gtier.sh REVISION 04:55); from chain_cap8.sh — cap-8 follow-up (pre-registered in bmoe_gtier.sh 03:05) after chain_tier; derived from chain_tier.sh — tier smoke + A/B only (after the 02:23 smoke's flag bug); derived from chain_night3, which takes over from chain_night2 during the kgsl A/B (which runs on the phone under setsid, independent
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
log "[prerepack] waiting for the arena3 A/B on the phone to finish"
while :; do
  reconnect
  d=$(A "ls -d $H/bmoe_arena3_ab_*/ 2>/dev/null | tail -1" | tr -d '\r')
  [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
  [ -n "$d" ] && [ -z "$(A 'ps -A -o ARGS' | grep 'sh bmoe_arena3' | grep -v grep)" ] && [ -z "$(lock)" ] && { log "[prerepack] arena3 ended without DONE ($d)"; break; }
  sleep 60
done
mkdir -p "$R/bmoe_arena3"; adb pull "$d" "$R/bmoe_arena3" >/dev/null 2>&1 </dev/null; log "[prerepack] arena3 A/B pulled: $R/bmoe_arena3/$(basename "$d")"
[ -n "$(lock)" ] && { log "[prerepack] phone busy: $(lock)"; exit 1; }
SP=/tmp/claude-1000/-run-media-santhankumar-New-Volume-identifying-variation/2d806460-912e-4de7-a437-dc77597a5bb6/scratchpad
A "echo 'prerepack-convert $(date +%H:%M:%S)' > $H/.phone_busy; mkdir -p $H/bmoe-i8mm-0025" >/dev/null
for f in "$SP/bmoe-i8mm-0025"/*; do adb push "$f" "$H/bmoe-i8mm-0025/" >/dev/null 2>&1 </dev/null; done
want=$(grep bmoe-cli "$SP/bmoe-i8mm-0025/MD5" | cut -d' ' -f1); got=$(A "md5sum $H/bmoe-i8mm-0025/bmoe-cli" | cut -d' ' -f1)
[ "$want" = "$got" ] || { log "[prerepack] FATAL 0025 md5 $got != $want"; A "rm -f $H/.phone_busy"; exit 1; }
A "chmod 755 $H/bmoe-i8mm-0025/bmoe-cli" >/dev/null
adb push "$MP/host/repack_bench/out/repack_gguf" "$H/bmoe-i8mm-0025/" >/dev/null 2>&1 </dev/null
adb push "$MP/device/bmoe_prerepack.sh" "$H/" >/dev/null 2>&1 </dev/null
if ! A "[ -f $H/Qwen3-30B-A3B-Q4_0.repacked.gguf.repacked ]"; then
  log "[prerepack] converting on the phone (cp 17 GB + repack in place)"
  A "cd $H && rm -f Qwen3-30B-A3B-Q4_0.repacked.gguf Qwen3-30B-A3B-Q4_0.repacked.gguf.repacked && cp Qwen3-30B-A3B-Q4_0.gguf Qwen3-30B-A3B-Q4_0.repacked.gguf && cd bmoe-i8mm-0025 && chmod 755 repack_gguf && LD_LIBRARY_PATH=. ./repack_gguf $H/Qwen3-30B-A3B-Q4_0.repacked.gguf 2>/dev/null" > "$R/bmoe_prerepack_convert.out" 2>&1
  log "[prerepack] convert: $(grep RESULT "$R/bmoe_prerepack_convert.out")"
fi
A "cat $H/Qwen3-30B-A3B-Q4_0.repacked.gguf.repacked" > "$R/bmoe_prerepack_marker.txt"
A "rm -f $H/.phone_busy" >/dev/null
grep -q "^repacked_tensors 138$" "$R/bmoe_prerepack_marker.txt" || { log "[prerepack] FATAL marker: $(head -4 "$R/bmoe_prerepack_marker.txt" | tr '\n' ' ')"; exit 1; }
export GTENV="GT_BIN=bmoe-i8mm-0025 GT_PIN=$(cat "$SP/phone_pin")"
A "svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable" >/dev/null 2>&1
campaign bmoe_prerepack.sh fidelity
F="$LAST"
pb=$(grep -o "ppl_base exit=0 ppl: [0-9.]*" "$F/log.txt" | awk '{print $4}'); ps=$(grep -o "ppl_stack exit=0 ppl: [0-9.]*" "$F/log.txt" | awk '{print $4}')
hb=$(grep -o "ppl_base.*hits: [0-9]*/[0-9]*" "$F/log.txt" | grep -o "[0-9]*/[0-9]*$"); hs=$(grep -o "ppl_stack.*hits: [0-9]*/[0-9]*" "$F/log.txt" | grep -o "[0-9]*/[0-9]*$")
fid=$(python3 -c "
pb,ps='$pb','$ps'; hb,hs='$hb','$hs'
try:
  r=abs(float(ps)-float(pb))/float(pb); nb,t=map(int,hb.split('/')); ns,_=map(int,hs.split('/')); f=abs(ns-nb)/t
  print('ok' if (r<=0.002 and f<=0.01) else 'FAIL', 'dppl=%.4f%% dhits=%d/%d'%(100*r,abs(ns-nb),t))
except Exception as e: print('FAIL parse', e)")
log "[prerepack] fidelity: base ppl $pb hits $hb, stack ppl $ps hits $hs -> $fid"
case "$fid" in ok*) ;; *) log "[prerepack] fidelity gate failed: no smoke, no A/B"; restore; touch "$R/CHAIN_PREREPACK_DONE"; exit 0;; esac
campaign bmoe_prerepack.sh smoke
S="$LAST"; ok=1
[ "$(grep -c '^exit=0' "$S/log.txt")" = 2 ] || ok=0
grep -q FATAL "$S"/*.err && ok=0
grep -q "marker ok" "$S/stack_reps1.err" || ok=0
grep -q "repack-dense: [1-9]" "$S/stack_reps1.err" || ok=0
[ "$(grep -c 'wake_start=Awake' "$S/log.txt")" = 2 ] || ok=0
[ "$(grep -c 'AFTER wake=Awake' "$S/log.txt")" = 2 ] || ok=0
log "[prerepack] smoke ok=$ok :: $(grep -h 'generation:' "$S"/*.out | tr '\n' ' ') :: $(grep -hE 'marker ok|repack-dense:' "$S/stack_reps1.err" | tr '\n' ' ') :: $(grep -o 'text_match.*' "$S/log.txt" | tr '\n' ' ')"
b=$(A "dumpsys battery | grep -m1 ' level' | tr -dc 0-9")
if [ $ok = 1 ] && [ "${b:-0}" -ge 30 ]; then log "[prerepack] A/B start (battery $b%)"; campaign bmoe_prerepack.sh ab; log "[prerepack] A/B pulled: $LAST"; else log "[prerepack] A/B not started (smoke ok=$ok, battery ${b}%)"; fi
restore
touch "$R/CHAIN_PREREPACK_DONE"
