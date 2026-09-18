#!/bin/bash
# chain_gtier_v0.sh — overnight 2026-09-19: engine A/B of the GPU tier with gx variant 0 (bmoe_gtier.sh, see its
# DEVIATION note). Steps:
#   1. wait for the shared phone lock to clear
#   2. push bmoe-i8mm-0019v2 (engine with repack/unpack; v0 and v1 use the plain copy path)
#   3. smoke (GT_VARIANT=0). Gate: text_match OK on both rows, dispatches > 0, warm start > 0, no FATAL
#   4. only if the smoke passes: ab
#   5. restore the phone (stayon false, timeout 60000, deviceidle enable)
# Unattended: no OOM (the phone job is the only one; the laptop runs nothing heavy), no reboot (battery guard in the
# script; nothing here touches adb tcpip or power).
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
MP="/run/media/santhankumar/New Volume/identifying-variation/moe-phone"
R="$MP/results/2026-09-19"; mkdir -p "$R/bmoe_gtier"
BIN="/tmp/claude-1000/-run-media-santhankumar-New-Volume-identifying-variation/2d806460-912e-4de7-a437-dc77597a5bb6/scratchpad/bmoe-i8mm-0019v2"
H=/data/local/tmp/moe-stream
LOG="$R/chain_gtier_v0.log"; log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }
A() { adb shell "$@" </dev/null; }
reconnect() { adb shell 'echo up' </dev/null >/dev/null 2>&1 || adb connect "$ANDROID_SERIAL" >/dev/null 2>&1; }
restore() { A 'svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1; log "phone restored"; }
campaign() {  # mode
  local d
  A "cd $H && (GT_BIN=bmoe-i8mm-0019v2 GT_VARIANT=0 setsid nohup sh bmoe_gtier.sh $1 > bmoe_gtier_$1_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
  sleep 45
  while :; do
    reconnect
    d=$(A "ls -d $H/bmoe_gtier_${1}_*/ | tail -1" | tr -d '\r')
    [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
    [ -n "$d" ] && A "[ -f ${d}REFUSED ]" && { log "REFUSED: $(A "cat ${d}REFUSED")"; break; }
    [ -z "$(A 'ps -A -o ARGS' | grep 'sh bmoe_gtier' | grep -v grep)" ] && { log "bmoe_gtier $1 exited without DONE ($d)"; break; }
    sleep 45
  done
  adb pull "$d" "$R/bmoe_gtier" >/dev/null 2>&1 </dev/null; log "pulled $d"
  LAST="$R/bmoe_gtier/$(basename "$d")"
}
log "start; waiting for the phone lock"
while :; do reconnect; b=$(A "cat $H/.phone_busy 2>/dev/null" | tr -d '\r'); [ -z "$b" ] && break; sleep 60; done
orph=$(A 'ps -A -o ARGS' | grep -E 'bmoe-cli|llama-|zcbench|gx_|sh bmoe_|run_phone' | grep -v grep)
[ -n "$orph" ] && { log "FATAL: foreign phone processes: $orph"; touch "$R/CHAIN_GTIER_V0_DONE"; exit 1; }
A "mkdir -p $H/bmoe-i8mm-0019v2" >/dev/null
for f in "$BIN"/*; do adb push "$f" "$H/bmoe-i8mm-0019v2/" >/dev/null 2>&1 </dev/null; done
adb push "$MP/device/bmoe_gtier.sh" "$H/" >/dev/null 2>&1 </dev/null
want=$(grep bmoe-cli "$BIN/MD5" | cut -d' ' -f1); got=$(A "md5sum $H/bmoe-i8mm-0019v2/bmoe-cli" | cut -d' ' -f1)
[ "$want" = "$got" ] || { log "FATAL: pushed bmoe-cli md5 $got != $want"; touch "$R/CHAIN_GTIER_V0_DONE"; exit 1; }
# pinprobe (host/pinned_arena, pre-registered in its header): ~1 min, 2 x 768 MiB, under the lock
A "echo 'pinprobe $(date +%H:%M:%S)' > $H/.phone_busy; mkdir -p /data/local/tmp/pinprobe" >/dev/null
adb push "$MP/host/pinned_arena/out/pinprobe" /data/local/tmp/pinprobe/ >/dev/null 2>&1 </dev/null
mkdir -p "$R/pinprobe"; A "cd /data/local/tmp/pinprobe && chmod 755 pinprobe && ./pinprobe 768" > "$R/pinprobe/pinprobe.out" 2>&1
A "rm -f $H/.phone_busy" >/dev/null
log "pinprobe: $(grep -E '^(BW|REREAD|VERDICT)' "$R/pinprobe/pinprobe.out" | tr '\n' ' ')"
A 'chmod 755 /data/local/tmp/moe-stream/bmoe-i8mm-0019v2/bmoe-cli; svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable' >/dev/null 2>&1
log "pushed (md5 ok); smoke v0"
campaign smoke
S="$LAST"
ok=1
[ "$(grep -c 'text_match .* OK' "$S/log.txt")" = 2 ] || ok=0
grep -q FATAL "$S"/*.err && ok=0
grep -q "dispatches [1-9]" "$S/stack_reps1.err" || ok=0
grep -q "warm start filled [1-9]" "$S/stack_reps1.err" || ok=0
log "smoke ok=$ok :: $(grep -h 'generation:' "$S"/*.out | tr '\n' ' ')"
if [ $ok = 1 ]; then log "A/B start"; campaign ab; log "A/B pulled: $LAST"; else log "A/B not run (smoke failed)"; fi
restore
touch "$R/CHAIN_GTIER_V0_DONE"
