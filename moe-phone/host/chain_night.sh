#!/bin/bash
# chain_night.sh — unattended overnight queue (user asleep: no OOM, no phone reboot, nothing needing input).
#   1. wait for chain_swapguard.sh (CHAIN_SWAPGUARD_DONE)
#   2. M6 run 2 (host/gpu_ffn/run_phone_m6.sh, the teammate's gx kernel re-test)
#   3. push the gx engine (bmoe-i8mm-0019gx) + Qwen3 prior; bmoe_gtier.sh smoke; gate: identical text, dispatches > 0,
#      warm start > 0, no FATAL
#   4. only if the smoke passes: bmoe_gtier.sh ab (pre-registered in its header)
#   5. restore the phone
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
MP="/run/media/santhankumar/New Volume/identifying-variation/moe-phone"
R="$MP/results/2026-09-18"
DEV="$MP/device"
BIN="/tmp/claude-1000/-run-media-santhankumar-New-Volume-identifying-variation/2d806460-912e-4de7-a437-dc77597a5bb6/scratchpad/bmoe-i8mm-0019gx"
CASES="/tmp/claude-1000/-run-media-santhankumar-New-Volume-identifying-variation/776493a2-6bd6-44c3-95ca-f7e463b046d4/scratchpad/gxcases"
LOG="$R/chain_night.log"; log() { echo "$(date -Iseconds) $*" >> "$LOG"; }
A() { adb shell "$@" </dev/null; }
reconnect() { adb shell 'echo up' </dev/null >/dev/null 2>&1 || adb connect "$ANDROID_SERIAL" >/dev/null 2>&1; }
wait_phone_idle() { local i=0; while [ $i -lt 60 ]; do reconnect; [ -z "$(A 'ps -A -o ARGS' | grep -E 'bmoe-cli|llama-|zcbench|gx_|sh bmoe_' | grep -v grep)" ] && return 0; sleep 10; i=$((i+1)); done; return 1; }
campaign() {  # mode
  local d
  A "cd /data/local/tmp/moe-stream && (setsid nohup sh bmoe_gtier.sh $1 > bmoe_gtier_$1_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
  sleep 45
  while :; do
    reconnect
    d=$(A "ls -d /data/local/tmp/moe-stream/bmoe_gtier_${1}_*/ | tail -1" | tr -d '\r')
    [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
    [ -z "$(A 'ps -A -o ARGS' | grep "sh bmoe_gtier")" ] && { log "bmoe_gtier $1 exited without DONE ($d)"; break; }
    sleep 45
  done
  mkdir -p "$R/bmoe_gtier"; adb pull "$d" "$R/bmoe_gtier" >/dev/null 2>&1 </dev/null; log "pulled $d"
  echo "$R/bmoe_gtier/$(basename "$d")"
}
log "waiting for chain_swapguard"
while [ ! -f "$R/CHAIN_SWAPGUARD_DONE" ]; do sleep 30; done
sed 's#CHAIN_NEXT_DONE#CHAIN_NIGHT_DONE#' /tmp/claude-1000/waker.sh > /tmp/claude-1000/waker_night.sh
setsid nohup bash /tmp/claude-1000/waker_night.sh >/dev/null 2>&1 </dev/null &
A 'svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable' >/dev/null 2>&1
wait_phone_idle || { log "FATAL: phone never became idle"; touch "$R/CHAIN_NIGHT_DONE"; exit 1; }
# 2. M6 run 2
log "M6 run 2 start"
bash "$MP/host/gpu_ffn/run_phone_m6.sh" "$CASES" > "$R/m6_run2.log" 2>&1
log "M6 run 2 exit $? :: $(tail -2 "$R/m6_run2.log" | tr '\n' ' ')"
wait_phone_idle || { log "FATAL: phone busy after M6"; touch "$R/CHAIN_NIGHT_DONE"; exit 1; }
# 3. push + smoke
[ "$(strings "$BIN/bmoe-cli" | grep -c 'gpu-tier-prior')" -ge 1 ] || { log "FATAL: stale gx engine"; touch "$R/CHAIN_NIGHT_DONE"; exit 1; }
A 'mkdir -p /data/local/tmp/moe-stream/bmoe-i8mm-0019gx' >/dev/null 2>&1
adb push "$BIN"/* /data/local/tmp/moe-stream/bmoe-i8mm-0019gx/ >/dev/null 2>&1 </dev/null
adb push "$R/qwen3_gtier_prior.txt" "$DEV/bmoe_gtier.sh" "$DEV/thermal_gate.sh" /data/local/tmp/moe-stream/ >/dev/null 2>&1 </dev/null
A 'chmod 755 /data/local/tmp/moe-stream/bmoe-i8mm-0019gx/bmoe-cli'
log "gx engine pushed: $(A 'sha256sum /data/local/tmp/moe-stream/bmoe-i8mm-0019gx/bmoe-cli' | cut -c1-16)"
SD=$(campaign smoke | tail -1)
ok=1
grep -q "text_match.*DIFFERS" "$SD/log.txt" && ok=0
grep -q "FATAL" "$SD"/*.err 2>/dev/null && ok=0
grep -qE "dispatches [1-9]" "$SD/stack_reps1.err" || ok=0
grep -qE "warm start filled [1-9]" "$SD/stack_reps1.err" || ok=0
grep -q "^exit=0" "$SD/log.txt" || ok=0
log "smoke gate ok=$ok :: $(grep -hE 'warm start|gpu-tier: backend gx slots' "$SD/stack_reps1.err" | cut -c1-300 | tr '\n' ' ') :: $(grep text_match "$SD/log.txt" | tr '\n' ' ')"
if [ "$ok" = 1 ]; then
  wait_phone_idle
  s=$(A 'dumpsys thermalservice' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'); t=0
  while [ "${s:-9}" -gt 1 ] && [ $t -lt 900 ]; do sleep 30; t=$((t+30)); s=$(A 'dumpsys thermalservice' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'); done
  log "cool_gate ${t}s status ${s:-?}; A/B start"
  AD=$(campaign ab | tail -1)
  log "A/B done: $AD"
else
  log "SMOKE FAILED: A/B not run"
fi
A 'am force-stop com.moephone.bmoe3; svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
log "phone restored"; touch "$R/CHAIN_NIGHT_DONE"
