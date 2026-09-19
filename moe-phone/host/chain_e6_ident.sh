#!/bin/bash
# chain_e6_ident.sh — after chain_e6b (CHAIN_E6_DONE): the phone-side determinism control (bmoe_e6.sh ident). Expected KL 0, 0 flips.
set -u
MP="/run/media/santhankumar/New Volume/identifying-variation/moe-phone"; R="$MP/results/2026-09-19"
export ANDROID_SERIAL=${ANDROID_SERIAL:-3C15CK0028J00000}
SP=/tmp/claude-1000/-run-media-santhankumar-New-Volume-identifying-variation/2d806460-912e-4de7-a437-dc77597a5bb6/scratchpad
H=/data/local/tmp/moe-stream; LOG="$R/chain_e6.log"; log() { echo "$(date -Iseconds) [ident] $*" | tee -a "$LOG"; }
A() { adb shell "$@" </dev/null; }
while [ ! -f "$R/CHAIN_E6_DONE" ]; do sleep 60; done
adb push "$MP/device/bmoe_e6.sh" "$H/" >/dev/null 2>&1 </dev/null
A "grep -q 'MODE\" = ident' $H/bmoe_e6.sh" || { log "FATAL pushed script lacks ident"; exit 1; }
A "cd $H && (E6_NOWAKE=1 setsid nohup sh bmoe_e6.sh ident > bmoe_e6_ident_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
log "started"; sleep 60
while :; do
  d=$(A "ls -d $H/bmoe_e6_ident_*/ | tail -1" | tr -d '\r')
  [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
  [ -z "$(A 'ps -A -o ARGS' | grep 'sh bmoe_e6' | grep -v grep)" ] && { log "exited without DONE ($d)"; break; }
  sleep 60
done
adb pull "$d" "$R/bmoe_e6" >/dev/null 2>&1 </dev/null
log "result: $(grep -h '^ppl-kl' "$R/bmoe_e6/$(basename "$d")/FLOOR_self.out" 2>/dev/null)"
A 'svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
touch "$R/CHAIN_E6_IDENT_DONE"
