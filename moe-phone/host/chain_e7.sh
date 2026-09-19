#!/bin/bash
# chain_e7.sh — after E1 (CHAIN_E1_DONE): push llama-bench (build-android-ocl) and the L4/L8 specimens, run device/bmoe_e7.sh, pull.
set -u
MP="/run/media/santhankumar/New Volume/identifying-variation/moe-phone"; R="$MP/results/2026-09-19"; mkdir -p "$R/bmoe_e7"
MW="/run/media/santhankumar/New Volume/moe-work"
H=/data/local/tmp/moe-stream; E=/data/local/tmp/e7; LOG="$R/chain_e7.log"; log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }
A() { adb shell "$@" </dev/null; }
log "waiting for CHAIN_E1_DONE"
while [ ! -f "$R/CHAIN_E1_DONE" ]; do sleep 60; done
export ANDROID_SERIAL=$(cat "$R/E1_GO" | tr -d ' \n')
[ -n "$(A "cat $H/.phone_busy 2>/dev/null" | tr -d '\r')" ] && { log "phone busy"; exit 1; }
A "mkdir -p $E" >/dev/null
adb push "$MW/llama.cpp/build-android-ocl/bin/llama-bench" "$E/" >/dev/null 2>&1 </dev/null
adb push "$MW/models/Qwen3-30B-A3B-Q4_0.L4.gguf" "$E/L4.gguf" >/dev/null 2>&1 </dev/null
adb push "$MW/models/Qwen3-30B-A3B-Q4_0.L8.gguf" "$E/L8.gguf" >/dev/null 2>&1 </dev/null
for f in L4 L8; do
  want=$(stat -c %s "$MW/models/Qwen3-30B-A3B-Q4_0.$f.gguf"); got=$(A "stat -c %s $E/$f.gguf" | tr -d '\r')
  [ "$want" = "$got" ] || { log "FATAL $f size $got != $want"; exit 1; }
done
A "chmod 755 $E/llama-bench" >/dev/null
adb push "$MP/device/bmoe_e7.sh" "$H/" >/dev/null 2>&1 </dev/null
A "svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable" >/dev/null 2>&1
A "cd $H && (GT_PIN=$(cat "$HOME/.config/moe-phone/phone_pin") setsid nohup sh bmoe_e7.sh > bmoe_e7_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
log "E7 started"; sleep 60
while :; do
  d=$(A "ls -d $H/bmoe_e7_2*/ | tail -1" | tr -d '\r')
  [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
  [ -n "$d" ] && A "[ -f ${d}REFUSED ]" && { log "REFUSED"; break; }
  [ -z "$(A 'ps -A -o ARGS' | grep 'sh bmoe_e7' | grep -v grep)" ] && { log "E7 exited without DONE ($d)"; break; }
  sleep 60
done
adb pull "$d" "$R/bmoe_e7" >/dev/null 2>&1 </dev/null; log "E7 pulled: $R/bmoe_e7/$(basename "$d")"
A 'svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
touch "$R/CHAIN_E7_DONE"
