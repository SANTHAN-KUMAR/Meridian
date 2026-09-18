#!/bin/bash
# chain_resume.sh — restart the 2026-09-18 night's queue the moment the phone is reachable again.
#
# WHY IT EXISTS. The ZRAM campaign's 10000 MiB cell (on an 11366 MiB device) took the phone out: adbd
# stopped listening and never came back, the tcp port was gone, and the latency spike followed by a
# clean recovery is what a reboot looks like from outside (a reboot also loses the non-persistent
# service.adb.tcp.port, which is exactly the symptom). The device-side campaigns survive a reboot --
# they run under setsid and /data/local/tmp is persistent -- but every driver here needs adb, so the
# queue stops. This script waits for access, restores the phone state a reboot would have cleared, and
# then runs what is left, in the order the night's findings make interesting.
#
# It is safe to start while the phone is still unreachable: it blocks until adb answers.
#
# PIN is read from the environment (BMOE_PIN) and never written to disk by this script.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
HOSTDIR="$(cd "$(dirname "$0")" && pwd)"
DEV="$(cd "$HOSTDIR/../device" && pwd)"
LOG="$R/chain_resume.log"
mkdir -p "$R"
log() { echo "$(date -Iseconds) $*" >> "$LOG"; }

log "waiting for adb"
i=0
while :; do
  adb connect "$ANDROID_SERIAL" >/dev/null 2>&1
  adb shell 'echo up' >/dev/null 2>&1 && break
  i=$((i + 1))
  if [ $((i % 40)) -eq 0 ]; then log "still unreachable after $((i * 15))s"; fi
  sleep 15
done
log "adb back; uptime: $(adb shell 'cat /proc/uptime' | tr -d '\r')"
log "meminfo: $(adb shell 'awk "/MemTotal|MemAvailable|SwapFree/{printf \"%s=%s \", \$1, \$2}" /proc/meminfo' | tr -d '\r')"

# A reboot clears these; re-applying them is harmless if it did not happen.
adb shell 'svc power stayon true' >/dev/null 2>&1
adb shell 'settings put system screen_off_timeout 1800000' >/dev/null 2>&1
adb shell 'dumpsys deviceidle disable' >/dev/null 2>&1
adb shell "input keyevent KEYCODE_WAKEUP" >/dev/null 2>&1
if [ -n "${BMOE_PIN:-}" ]; then
  adb shell "rm -f /data/local/tmp/moe-stream/keep_awake.run" >/dev/null 2>&1
  adb shell "cd /data/local/tmp/moe-stream && (setsid nohup sh keep_awake.sh '$BMOE_PIN' >/dev/null 2>&1 < /dev/null &)" >/dev/null 2>&1
  log "keep_awake started"
fi

# re-push the campaign scripts, in case the tree changed while the phone was away
for f in thermal_gate.sh keep_awake.sh bmoe_zram.sh bmoe_order3.sh bmoe_arena2.sh bmoe_mem.sh; do
  [ -f "$DEV/$f" ] && adb push "$DEV/$f" /data/local/tmp/moe-stream/ >/dev/null 2>&1
done
log "scripts pushed"

stop_campaigns() {
  for pat in 'sh bmoe_mem\.sh' 'sh bmoe_zram\.sh' 'sh bmoe_order' 'sh bmoe_arena' 'bmoe-cli'; do
    pids=$(adb shell "ps -A -o PID,ARGS" | awk -v p="$pat" '$0 ~ p && $0 !~ /awk/ {print $1}')
    [ -n "$pids" ] && { log "stopping [$pat]: $pids"; adb shell "kill $pids" >/dev/null 2>&1; sleep 10; }
  done
}

device_campaign() {  # script reps results_subdir dir_glob
  local script=$1 reps=$2 sub=$3 glob=$4
  stop_campaigns
  log "$script start"
  adb shell "cd /data/local/tmp/moe-stream && echo 'chain_resume: $script \$(date)' >> phone_queue.log && (setsid nohup sh $script $reps > ${script%.sh}_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
  sleep 60
  local lost=0
  while :; do
    if ! adb shell 'echo up' >/dev/null 2>&1; then
      adb connect "$ANDROID_SERIAL" >/dev/null 2>&1
      lost=$((lost + 1))
      if [ $lost -gt 40 ]; then log "$script: lost the phone for 10 min, giving up on this campaign"; break; fi
      sleep 15
      continue
    fi
    d=$(adb shell "ls -d /data/local/tmp/moe-stream/$glob 2>/dev/null | tail -1" | tr -d '\r')
    if [ -n "$d" ] && adb shell "[ -f ${d}DONE ]"; then break; fi
    running=$(adb shell "ps -A -o PID,ARGS" | awk -v p="sh $script" '$0 ~ p && $0 !~ /awk/ {print $1}')
    if [ -z "$running" ]; then log "$script exited without DONE (dir '$d')"; break; fi
    sleep 90
  done
  d=$(adb shell "ls -d /data/local/tmp/moe-stream/$glob 2>/dev/null | tail -1" | tr -d '\r')
  [ -n "$d" ] && { mkdir -p "$R/$sub"; adb pull "$d" "$R/$sub" >/dev/null 2>&1; log "$script pulled from $d"; }
}

# Both app packages get the build that can set environment variables in-process, which is what the NPU
# tuning sweep needs; npu2 holds the OLMoE copy, bmoe3 holds Qwen3 and the per-op benchmark.
log "installing apps"
adb install -r "$HOME/moework/npu-apk2/npu2_env.apk" >> "$LOG" 2>&1
adb install -r "$HOME/moework/npu-apk3/bmoe3.apk"    >> "$LOG" 2>&1

# 1. FIRST, because it is the open question about a conclusion already drawn: the NPU came last at
#    decode in the committed rows, but those ran the backend's defaults, and GGML_HEXAGON_OPPOLL=0
#    means the host waits on an interrupt for every DSP batch. Decode is dozens of tiny round-trips per
#    token. The same rows put the NPU FIRST at prefill by 2.1x, so the silicon is not the issue.
log "npu_tuning start"
sh "$HOSTDIR/npu_tuning.sh" 2 >> "$LOG" 2>&1
log "npu_tuning done"

# 2. the balanced expert-order A/B: two earlier campaigns disagreed and both were position-confounded
device_campaign bmoe_order3.sh 3 bmoe_order3 'bmoe_order3_*/'
# 3. the slot arena at the current operating point: it cut cache management 34 ms -> 2 ms and still lost
device_campaign bmoe_arena2.sh 3 bmoe_arena2 'bmoe_arena2_*/'
# 4. the ZRAM cache, now capped below what took the phone down
device_campaign bmoe_zram.sh 3 bmoe_zram 'bmoe_zram_*/'

# 5. host-driven: per-op device truth, then whether two devices add bandwidth, then the CPU questions
sh "$HOSTDIR/matmul_sweep.sh" 2 >> "$LOG" 2>&1
log "matmul sweep done"
sh "$HOSTDIR/agg_bandwidth.sh" 3 >> "$LOG" 2>&1
log "agg_bandwidth done"
sh "$HOSTDIR/thread_sweep.sh" 3 >> "$LOG" 2>&1
log "thread_sweep done"
sh "$HOSTDIR/app_vs_shell.sh" 3 >> "$LOG" 2>&1
log "app_vs_shell done"

# NOT QUEUED: host/coresidency.sh and device/bmoe_prefill.sh were written for an on-device task-agent
# question that is a SEPARATE track, worked by someone else. They stay in the tree because they are
# finished and documented, but they do not take phone time from the core goal, which is unchanged:
# 10 tok/s on Qwen3-30B-A3B, and 5 tok/s on models larger than it.
# (prefill sweep: side-track, not queued -- see chain note)     
# (coresidency: side-track, not queued)      
# (side-track removed)

stop_campaigns
adb shell 'cd /data/local/tmp/moe-stream && echo "chain_resume: bmoe_mem $(date)" >> phone_queue.log && (setsid nohup sh bmoe_mem.sh 3 > bmoe_mem_nohup.log 2>&1 < /dev/null &)' >/dev/null 2>&1
log "bmoe_mem launched"
touch "$R/CHAIN_RESUME_DONE"
