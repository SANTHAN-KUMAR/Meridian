#!/bin/bash
# chain_attn.sh — 2026-09-18. Runs after overnight_npu.sh's in-app sweep so nothing contends for flash:
#   1. wait for the app-engine driver to finish
#   2. park the bmoe_mem campaign it launches (it would fight our A/B for RAM and flash)
#   3. copy Qwen3 into com.moephone.bmoe3's storage (untrusted_app cannot read /data/local/tmp)
#   4. run our engine's attention-device A/B in-app (bmoe_attn_ab.sh)
#   5. hand the phone back to bmoe_mem
set -u
export ANDROID_SERIAL=192.168.0.65:5555
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
W="/run/media/santhankumar/New Volume/moe-work"
PKG=com.moephone.bmoe3
LOG="$R/bmoe_attn_ab/chain.log"
mkdir -p "$R/bmoe_attn_ab"
log() { echo "$(date -Iseconds) $*" >> "$LOG"; }

log "waiting for app-engine driver"
while [ ! -f "$R/app_engine/DRIVER_DONE" ]; do sleep 60; done
log "driver done"

# bmoe_mem.sh is launched by the driver's last line; stop it by its exact argv, never with pkill -f
# (which matches this script's own command line).
for i in 1 2 3; do
  pids=$(adb shell "ps -A -o PID,ARGS" | awk '$0 ~ /sh bmoe_mem\.sh/ && $0 !~ /awk/ {print $1}')
  [ -z "$pids" ] && break
  log "stopping bmoe_mem pids: $pids"
  adb shell "kill $pids" >/dev/null 2>&1
  sleep 10
done
# and any engine process it had already started
epids=$(adb shell "ps -A -o PID,ARGS" | awk '$0 ~ /bmoe-cli/ && $0 !~ /awk/ {print $1}')
[ -n "$epids" ] && { log "stopping bmoe-cli pids: $epids"; adb shell "kill $epids" >/dev/null 2>&1; sleep 10; }

have=$(adb shell "run-as $PKG sh -c 'stat -c %s files/qwen3.gguf 2>/dev/null || echo 0'" | tr -d '\r')
want=$(adb shell "stat -c %s /data/local/tmp/moe-stream/Qwen3-30B-A3B-Q4_0.gguf" | tr -d '\r')
log "model in app storage: $have of $want bytes"
if [ "$have" != "$want" ]; then
  log "copying model"
  t0=$(date +%s)
  adb shell "run-as $PKG sh -c 'dd if=/data/local/tmp/moe-stream/Qwen3-30B-A3B-Q4_0.gguf of=files/qwen3.gguf bs=8M'" >> "$LOG" 2>&1
  have=$(adb shell "run-as $PKG sh -c 'stat -c %s files/qwen3.gguf 2>/dev/null || echo 0'" | tr -d '\r')
  log "copy finished in $(( $(date +%s) - t0 ))s, size $have (want $want)"
  if [ "$have" != "$want" ]; then log "COPY FAILED — aborting A/B"; exit 1; fi
fi

log "starting attention-device A/B"
sh "$W/bmoe_attn_ab.sh" 3 >> "$LOG" 2>&1
log "A/B done"

# hand the phone back to the memory campaign
adb shell 'cd /data/local/tmp/moe-stream && echo "chain_attn: bmoe_mem restart $(date)" >> phone_queue.log && (setsid nohup sh bmoe_mem.sh 3 > bmoe_mem_nohup.log 2>&1 < /dev/null &)' >/dev/null 2>&1
log "bmoe_mem relaunched"
touch "$R/bmoe_attn_ab/CHAIN_DONE"
