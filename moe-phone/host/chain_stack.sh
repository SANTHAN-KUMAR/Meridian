#!/bin/bash
# chain_stack.sh — finish the 0013 sweep (already running on the phone), then the single stacked comparison.
# The stack's gate flags are written to $R/STACK_GATE after the sweep is scored by its pre-registered rule
# (empty file = no gate). The waker started for chain_final3 keeps the phone awake until CHAIN_FINAL3_DONE,
# which this chain touches at the end.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
DEV="$(cd "$(dirname "$0")/../device" && pwd)"
LOG="$R/chain_stack.log"; log() { echo "$(date -Iseconds) $*" >> "$LOG"; }
A() { adb shell "$@" </dev/null; }
wait_pull() {  # script glob subdir
  local d
  while :; do
    d=$(A "ls -d /data/local/tmp/moe-stream/$2 | tail -1" | tr -d '\r')
    [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
    [ -z "$(A 'ps -A -o ARGS' | grep "sh $1")" ] && { log "$1 exited without DONE ($d)"; break; }
    sleep 60
  done
  mkdir -p "$R/$3"; adb pull "$d" "$R/$3" >/dev/null 2>&1 </dev/null; log "$1 pulled $d"
}
log "waiting for bmoe_specgate on the phone"
wait_pull bmoe_specgate.sh 'bmoe_specgate_*/' bmoe_specgate
touch "$R/SPECGATE_PULLED"
log "waiting for STACK_GATE (written after scoring the sweep)"
while [ ! -f "$R/STACK_GATE" ]; do sleep 20; done
G=$(cat "$R/STACK_GATE"); log "gate flags: [$G]"
A 'svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable' >/dev/null 2>&1
adb push "$DEV/bmoe_stack.sh" "$DEV/thermal_gate.sh" /data/local/tmp/moe-stream/ >/dev/null 2>&1 </dev/null
s=$(A 'dumpsys thermalservice' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'); t=0
while [ "${s:-9}" -gt 1 ] && [ $t -lt 900 ]; do sleep 30; t=$((t+30)); s=$(A 'dumpsys thermalservice' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'); done
log "cool_gate waited ${t}s, status ${s:-?}; bmoe_stack.sh start"
A "cd /data/local/tmp/moe-stream && (setsid nohup sh bmoe_stack.sh 6 '$G' > bmoe_stack_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
sleep 60
wait_pull bmoe_stack.sh 'bmoe_stack_*/' bmoe_stack
A 'am force-stop com.moephone.npu2; am force-stop com.moephone.bmoe3; svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
log "phone restored"; touch "$R/CHAIN_STACK_DONE" "$R/CHAIN_FINAL3_DONE"
