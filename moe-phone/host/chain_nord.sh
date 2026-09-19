#!/bin/bash
# chain_nord.sh — wait for the model push to the OnePlus Nord, verify its size, keep it awake on USB power, run the 10-min sustained test, pull.
set -u
export ANDROID_SERIAL=c808b54b
MP="/run/media/santhankumar/New Volume/identifying-variation/moe-phone"; R="$MP/results/2026-09-19/nord"; mkdir -p "$R"
H=/data/local/tmp/moe-stream; LOG="$R/chain_nord.log"; log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }
A() { adb shell "$@" </dev/null; }
log "waiting for the model push"
while pgrep -f "adb push .*Qwen3-30B-A3B-Q4_0.gguf /data/local/tmp/moe-stream/Qwen3-30B-A3B-Q4_0.gguf" >/dev/null; do sleep 30; done
sz=$(A "stat -c %s $H/Qwen3-30B-A3B-Q4_0.gguf" | tr -d '\r')
[ "$sz" = 17379988032 ] || { log "FATAL model size $sz != 17379988032 (push incomplete)"; exit 1; }
log "model on the Nord verified by size"
A "svc power stayon true; settings put system screen_off_timeout 1800000; input keyevent KEYCODE_WAKEUP" >/dev/null 2>&1
A "cd $H && (BIN=bmoe-v82 CMASK=c0 NT=2 IOMASK=0f SUSTAIN_S=600 setsid nohup sh bmoe_sustain.sh > bmoe_sustain_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
log "sustained test started (compute on cpu6-7, -t 2; I/O on cpu0-3)"
sleep 60
while :; do
  d=$(A "ls -d $H/bmoe_sustain_2*/ | tail -1" | tr -d '\r')
  [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
  [ -n "$d" ] && [ -z "$(A 'ps -A -o ARGS' | grep 'sh bmoe_sustain' | grep -v grep)" ] && { log "exited without DONE ($d)"; break; }
  sleep 60
done
adb pull "$d" "$R/" >/dev/null 2>&1 </dev/null; log "pulled $d"
A 'svc power stayon false; settings put system screen_off_timeout 60000' >/dev/null 2>&1
touch "$R/CHAIN_NORD_DONE"
