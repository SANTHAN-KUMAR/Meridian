#!/bin/bash
# chain_fix1.sh — (1) where the engine's extra instructions go (profile, ~5 min), then (2) the dense-weight fix
# tested on Qwen3 (bmoe_densemap.sh, ABBA x4, ~35 min).
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
DEV="$(cd "$(dirname "$0")/../device" && pwd)"
LOG="$R/chain_fix1.log"; log() { echo "$(date -Iseconds) $*" >> "$LOG"; }
A() { adb shell "$@" </dev/null; }
A 'svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable' >/dev/null 2>&1
sed 's#CHAIN_NEXT_DONE#CHAIN_FIX1_DONE#' /tmp/claude-1000/waker.sh > /tmp/claude-1000/waker_fix1.sh
setsid nohup bash /tmp/claude-1000/waker_fix1.sh >/dev/null 2>&1 </dev/null &
adb push "$DEV/bmoe_profile.sh" "$DEV/bmoe_densemap.sh" "$DEV/thermal_gate.sh" /data/local/tmp/moe-stream/ >/dev/null 2>&1 </dev/null
campaign() {  # script arg glob subdir
  local d
  log "$1 start"
  A "cd /data/local/tmp/moe-stream && (setsid nohup sh $1 $2 > ${1%.sh}_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
  sleep 30
  while :; do
    d=$(A "ls -d /data/local/tmp/moe-stream/$3 | tail -1" | tr -d '\r')
    [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
    [ -z "$(A 'ps -A -o ARGS' | grep "sh $1")" ] && { log "$1 exited without DONE ($d)"; break; }
    sleep 30
  done
  mkdir -p "$R/$4"; adb pull "$d" "$R/$4" >/dev/null 2>&1 </dev/null; log "$1 pulled $d"
}
campaign bmoe_profile.sh "" 'bmoe_profile_*/' bmoe_profile
campaign bmoe_densemap.sh 4 'bmoe_densemap_*/' bmoe_densemap
A 'svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
log "phone restored"; touch "$R/CHAIN_FIX1_DONE"
