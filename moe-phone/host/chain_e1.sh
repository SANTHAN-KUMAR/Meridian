#!/bin/bash
# chain_e1.sh — after the E6 determinism control (CHAIN_E6_IDENT_DONE): free the Q8_0 reference, convert the pre-repacked Q4_0 on
# the phone with engine 0027's own libggml-cpu (marker must say 138 tensors), push engine 0027 + bmoe_e1.sh, run E1, pull.
set -u
MP="/run/media/santhankumar/New Volume/identifying-variation/moe-phone"; R="$MP/results/2026-09-19"; mkdir -p "$R/bmoe_e1"
export ANDROID_SERIAL=${ANDROID_SERIAL:-$(adb devices | awk 'NR>1 && $2=="device"{print $1; exit}')}
SP=/tmp/claude-1000/-run-media-santhankumar-New-Volume-identifying-variation/2d806460-912e-4de7-a437-dc77597a5bb6/scratchpad
H=/data/local/tmp/moe-stream; LOG="$R/chain_e1.log"; log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }
A() { adb shell "$@" </dev/null; }
log "waiting for CHAIN_E6_IDENT_DONE"
while [ ! -f "$R/CHAIN_E6_IDENT_DONE" ]; do sleep 60; done
# the user unplugs for E4/E1 (deployment condition): wait for the agent to switch adb to wireless and write E1_GO (serial inside)
log "E6 done; waiting for E1_GO (wireless adb + phone unplugged)"
while [ ! -f "$R/E1_GO" ]; do sleep 20; done
export ANDROID_SERIAL=$(cat "$R/E1_GO" | tr -d ' \n')
log "E1_GO: serial $ANDROID_SERIAL, power: $(A 'dumpsys battery | grep -E "AC powered|USB powered"' | tr -s ' ' | tr '\n' ' ')"
# E4 (gx v5, the peer's pre-registered run, ~15 min) before E1, while the phone is otherwise idle
log "E4: running host/gpu_ffn/run_phone_e4.sh"
bash "$MP/host/gpu_ffn/run_phone_e4.sh" /tmp/claude-1000/gxcases > "$R/e4_run.log" 2>&1
log "E4 exit=$?: $(grep -E 'BENCH spin=1 variant=5 down=Q4_0 k=[12] |TOL|VERDICT|PASS|FAIL|results in' "$R/e4_run.log" | tr '\n' ' ' | cut -c1-600)"
[ -n "$(A "cat $H/.phone_busy 2>/dev/null" | tr -d '\r')" ] && { log "phone busy"; exit 1; }
orph=$(A 'ps -A -o ARGS' | grep -E 'bmoe-cli|gx_|sh bmoe_' | grep -v grep); [ -n "$orph" ] && { log "FATAL foreign: $orph"; exit 1; }
A "echo 'e1-prep $(date +%H:%M:%S)' > $H/.phone_busy; rm -f $H/Qwen3-30B-A3B-Q8_0.gguf; mkdir -p $H/bmoe-i8mm-0027" >/dev/null
log "Q8_0 reference deleted from the phone (E6 done); free: $(A 'df -h /data | tail -1' | tr -s ' ' | cut -d' ' -f4)"
for f in "$SP/bmoe-i8mm-0027"/*; do adb push "$f" "$H/bmoe-i8mm-0027/" >/dev/null 2>&1 </dev/null; done
want=$(grep bmoe-cli "$SP/bmoe-i8mm-0027/MD5" | cut -d' ' -f1); got=$(A "md5sum $H/bmoe-i8mm-0027/bmoe-cli" | cut -d' ' -f1)
[ "$want" = "$got" ] || { log "FATAL 0027 md5 $got != $want"; A "rm -f $H/.phone_busy"; exit 1; }
A "chmod 755 $H/bmoe-i8mm-0027/bmoe-cli" >/dev/null
adb push "$MP/host/repack_bench/out/repack_gguf" "$H/bmoe-i8mm-0027/" >/dev/null 2>&1 </dev/null
if ! A "grep -q '^repacked_tensors 138$' $H/Qwen3-30B-A3B-Q4_0.repacked.gguf.repacked 2>/dev/null"; then
  log "converting the pre-repacked Q4_0 on the phone (cp 17 GB + repack)"
  A "cd $H && rm -f Qwen3-30B-A3B-Q4_0.repacked.gguf Qwen3-30B-A3B-Q4_0.repacked.gguf.repacked && cp Qwen3-30B-A3B-Q4_0.gguf Qwen3-30B-A3B-Q4_0.repacked.gguf && cd bmoe-i8mm-0027 && chmod 755 repack_gguf && LD_LIBRARY_PATH=. ./repack_gguf $H/Qwen3-30B-A3B-Q4_0.repacked.gguf 2>/dev/null" > "$R/bmoe_e1/convert.out" 2>&1
  log "convert: $(grep RESULT "$R/bmoe_e1/convert.out")"
fi
A "cat $H/Qwen3-30B-A3B-Q4_0.repacked.gguf.repacked" > "$R/bmoe_e1/marker.txt"
A "rm -f $H/.phone_busy" >/dev/null
grep -q "^repacked_tensors 138$" "$R/bmoe_e1/marker.txt" || { log "FATAL marker: $(head -4 "$R/bmoe_e1/marker.txt" | tr '\n' ' ')"; exit 1; }
adb push "$MP/device/bmoe_e1.sh" "$H/" >/dev/null 2>&1 </dev/null
A "svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable" >/dev/null 2>&1
A "cd $H && (GT_PIN=$(cat "$SP/phone_pin") setsid nohup sh bmoe_e1.sh > bmoe_e1_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
log "E1 started"; sleep 60
while :; do
  d=$(A "ls -d $H/bmoe_e1_2*/ | tail -1" | tr -d '\r')
  [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
  [ -n "$d" ] && A "[ -f ${d}REFUSED ]" && { log "REFUSED"; break; }
  [ -z "$(A 'ps -A -o ARGS' | grep 'sh bmoe_e1' | grep -v grep)" ] && { log "E1 exited without DONE ($d)"; break; }
  sleep 60
done
adb pull "$d" "$R/bmoe_e1" >/dev/null 2>&1 </dev/null; log "E1 pulled: $R/bmoe_e1/$(basename "$d")"
A 'svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
touch "$R/CHAIN_E1_DONE"
