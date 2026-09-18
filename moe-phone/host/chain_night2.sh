#!/bin/bash
# chain_night2.sh — overnight 2026-09-19, phone order (battery-limited; the phone is unplugged):
#   1. wait for the shared lock (the gx session's M6 run 3)
#   2. pinprobe (~1 min; pre-registered in host/pinned_arena/pinprobe.cpp)
#   3. the gx session's window: wait up to 10 min for its lock to appear (memprobe + variant-3 bench), then until it
#      clears (at most 90 min)
#   4. only on pinprobe VERDICT BUILD: device/bmoe_kgsl.sh smoke; gate (identical text, "(kgsl)", no FATAL); then ab
#   5. restore the phone
# The GPU-tier A/B (bmoe_gtier.sh) is launched separately, once the variant-3 bench says which kernel to use.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
MP="/run/media/santhankumar/New Volume/identifying-variation/moe-phone"
R="$MP/results/2026-09-19"; mkdir -p "$R/bmoe_kgsl" "$R/pinprobe"
SP=/tmp/claude-1000/-run-media-santhankumar-New-Volume-identifying-variation/2d806460-912e-4de7-a437-dc77597a5bb6/scratchpad
H=/data/local/tmp/moe-stream
LOG="$R/chain_night2.log"; log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }
A() { adb shell "$@" </dev/null; }
reconnect() { adb shell 'echo up' </dev/null >/dev/null 2>&1 || adb connect "$ANDROID_SERIAL" >/dev/null 2>&1; }
lock() { A "cat $H/.phone_busy 2>/dev/null" | tr -d '\r'; }
restore() { A 'svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1; log "phone restored"; }
finish() { restore; touch "$R/CHAIN_NIGHT2_DONE"; exit "${1:-0}"; }
campaign() {  # script mode
  local d
  A "cd $H && (setsid nohup sh $1 $2 > ${1%.sh}_$2_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
  sleep 45
  while :; do
    reconnect
    d=$(A "ls -d $H/${1%.sh}_${2}_*/ | tail -1" | tr -d '\r')
    [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
    [ -n "$d" ] && A "[ -f ${d}REFUSED ]" && { log "REFUSED: $(A "cat ${d}REFUSED")"; break; }
    [ -z "$(A 'ps -A -o ARGS' | grep "sh $1" | grep -v grep)" ] && { log "$1 $2 exited without DONE ($d)"; break; }
    sleep 45
  done
  adb pull "$d" "$R/${1%.sh}" >/dev/null 2>&1 </dev/null; log "pulled $d"
  LAST="$R/${1%.sh}/$(basename "$d")"
}
log "start; waiting for the phone lock"
while :; do reconnect; [ -z "$(lock)" ] && break; sleep 60; done
orph=$(A 'ps -A -o ARGS' | grep -E 'bmoe-cli|llama-|zcbench|gx_|sh bmoe_|run_phone|pinprobe' | grep -v grep)
[ -n "$orph" ] && { log "FATAL: foreign phone processes: $orph"; finish 1; }
# 2. pinprobe
A "echo 'pinprobe $(date +%H:%M:%S)' > $H/.phone_busy; mkdir -p /data/local/tmp/pinprobe" >/dev/null
adb push "$MP/host/pinned_arena/out/pinprobe" /data/local/tmp/pinprobe/ >/dev/null 2>&1 </dev/null
A "cd /data/local/tmp/pinprobe && chmod 755 pinprobe && ./pinprobe 768" > "$R/pinprobe/pinprobe.out" 2>&1
A "rm -f $H/.phone_busy" >/dev/null
log "pinprobe: $(grep -E '^(BW|REREAD|VERDICT)' "$R/pinprobe/pinprobe.out" | tr '\n' ' ')"
# 3. the gx session's window
w=0; while [ $w -lt 10 ] && [ -z "$(lock)" ]; do sleep 60; w=$((w+1)); done
if [ -n "$(lock)" ]; then
  log "gx window: $(lock)"; w=0
  while [ $w -lt 90 ] && [ -n "$(lock)" ]; do sleep 60; w=$((w+1)); reconnect; done
  log "gx window closed after ${w} min (lock now: '$(lock)')"
  [ -n "$(lock)" ] && { log "lock still held after 90 min; stopping"; finish 1; }
else
  log "gx window: no gx run started within 10 min"
fi
# 4. kgsl arena
if ! grep -q "^VERDICT BUILD" "$R/pinprobe/pinprobe.out"; then log "pinprobe verdict not BUILD: kgsl campaign not run"; finish 0; fi
A "mkdir -p $H/bmoe-i8mm-0020kgsl" >/dev/null
for f in "$SP/bmoe-i8mm-0020kgsl"/*; do adb push "$f" "$H/bmoe-i8mm-0020kgsl/" >/dev/null 2>&1 </dev/null; done
adb push "$MP/device/bmoe_kgsl.sh" "$H/" >/dev/null 2>&1 </dev/null
want=$(grep bmoe-cli "$SP/bmoe-i8mm-0020kgsl/MD5" | cut -d' ' -f1); got=$(A "md5sum $H/bmoe-i8mm-0020kgsl/bmoe-cli" | cut -d' ' -f1)
[ "$want" = "$got" ] || { log "FATAL: pushed md5 $got != $want"; finish 1; }
A "chmod 755 $H/bmoe-i8mm-0020kgsl/bmoe-cli; svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable" >/dev/null 2>&1
campaign bmoe_kgsl.sh smoke
S="$LAST"; ok=1
[ "$(grep -c 'text_match .* OK' "$S/log.txt")" = 2 ] || ok=0
grep -q FATAL "$S"/*.err && ok=0
grep -q "(kgsl)" "$S/stack_reps1.err" || ok=0
log "kgsl smoke ok=$ok :: $(grep -hE 'generation:|slot-arena:' "$S"/*.out "$S"/*.err | tr '\n' ' ')"
[ $ok = 1 ] || finish 1
log "kgsl A/B start"; campaign bmoe_kgsl.sh ab; log "kgsl A/B pulled: $LAST"
finish 0
