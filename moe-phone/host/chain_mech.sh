#!/bin/bash
# chain_mech.sh — the overhead-mechanism diagnostic, started only after the teammate's zcbench run hands the
# phone back ($R/PHONE_BACK_FROM_ZC is touched by hand when that session reports done).
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
DEV="$(cd "$(dirname "$0")/../device" && pwd)"
LOG="$R/chain_mech.log"; log() { echo "$(date -Iseconds) $*" >> "$LOG"; }
A() { adb shell "$@" </dev/null; }
log "waiting for the phone to come back from zcbench"
while [ ! -f "$R/PHONE_BACK_FROM_ZC" ]; do sleep 10; done
A 'svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable' >/dev/null 2>&1
sed 's#CHAIN_NEXT_DONE#CHAIN_MECH_DONE#' /tmp/claude-1000/waker.sh > /tmp/claude-1000/waker_mech.sh
setsid nohup bash /tmp/claude-1000/waker_mech.sh >/dev/null 2>&1 </dev/null &
adb push "$DEV/bmoe_ovh_mech.sh" "$DEV/thermal_gate.sh" /data/local/tmp/moe-stream/ >/dev/null 2>&1 </dev/null
log "bmoe_ovh_mech.sh start"
A "cd /data/local/tmp/moe-stream && (setsid nohup sh bmoe_ovh_mech.sh 3 > bmoe_ovh_mech_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
sleep 30
while :; do
  d=$(A "ls -d /data/local/tmp/moe-stream/bmoe_ovh_mech_*/ | tail -1" | tr -d '\r')
  [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
  [ -z "$(A 'ps -A -o ARGS' | grep 'sh bmoe_ovh_mech')" ] && { log "exited without DONE ($d)"; break; }
  sleep 30
done
mkdir -p "$R/bmoe_ovh_mech"; adb pull "$d" "$R/bmoe_ovh_mech" >/dev/null 2>&1 </dev/null; log "pulled $d"
A 'svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
log "phone restored"; touch "$R/CHAIN_MECH_DONE"
