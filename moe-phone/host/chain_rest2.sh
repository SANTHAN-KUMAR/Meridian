#!/bin/bash
# chain_rest2.sh — replaces chain_rest.sh mid-flight (killed while it was only POLLING the SLRU campaign,
# which runs on the phone under setsid and was not touched). Moves patch 0012 up, ahead of the
# lower-value campaigns:
#   1. wait for bmoe_slru (decode re-run) to finish on the phone, pull it
#   2. bmoe_specadopt.sh  patch 0012 selective adoption (pre-registered in its header)
#   3. agg_bandwidth.sh, 4. bmoe_attn_gpu.sh, 5. thread_sweep.sh  (as in chain_rest.sh)
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
HOSTDIR="$(cd "$(dirname "$0")" && pwd)"
DEV="$(cd "$HOSTDIR/../device" && pwd)"
BIN0012="/tmp/claude-1000/-run-media-santhankumar-New-Volume-identifying-variation/2d806460-912e-4de7-a437-dc77597a5bb6/scratchpad/bmoe-i8mm-0012"
LOG="$R/chain_rest.log"
log() { echo "$(date -Iseconds) [rest2] $*" >> "$LOG"; }
A() { adb shell "$@" </dev/null; }
status() { A 'dumpsys thermalservice 2>/dev/null' | awk -F': ' '/^Thermal Status/{print $2; exit}' | tr -d '\r'; }
cool_gate() { local t=0 s; s=$(status); while [ "${s:-9}" -gt 1 ] && [ $t -lt 1200 ]; do sleep 30; t=$((t+30)); s=$(status); done; log "cool_gate waited ${t}s, status ${s:-?}"; }
awake() { A 'svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable' >/dev/null 2>&1; }
wait_pull() {  # script glob subdir
  local script=$1 glob=$2 sub=$3 d
  while :; do
    d=$(A "ls -d /data/local/tmp/moe-stream/$glob | tail -1" | tr -d '\r')
    [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
    [ -z "$(A 'ps -A -o ARGS' | grep "sh $script")" ] && { log "$script exited without DONE ($d)"; break; }
    sleep 60
  done
  mkdir -p "$R/$sub"; adb pull "$d" "$R/$sub" >/dev/null 2>&1 </dev/null; log "$script pulled $d"
}
log "taking over from chain_rest.sh: waiting for bmoe_slru on the phone"
wait_pull bmoe_slru.sh 'bmoe_slru_2*/' bmoe_slru_decode
awake
A 'mkdir -p /data/local/tmp/moe-stream/bmoe-i8mm-0012' >/dev/null 2>&1
adb push "$BIN0012"/* /data/local/tmp/moe-stream/bmoe-i8mm-0012/ >/dev/null 2>&1 </dev/null
adb push "$DEV/bmoe_specadopt.sh" "$DEV/thermal_gate.sh" /data/local/tmp/moe-stream/ >/dev/null 2>&1 </dev/null
A 'chmod 755 /data/local/tmp/moe-stream/bmoe-i8mm-0012/bmoe-cli'
log "0012 pushed: $(A 'sha256sum /data/local/tmp/moe-stream/bmoe-i8mm-0012/bmoe-cli' | cut -c1-16)"
cool_gate
log "bmoe_specadopt.sh start"
A "cd /data/local/tmp/moe-stream && (setsid nohup sh bmoe_specadopt.sh 3 > bmoe_specadopt_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
sleep 60
wait_pull bmoe_specadopt.sh 'bmoe_specadopt_*/' bmoe_specadopt
cool_gate; awake; A 'am force-stop com.moephone.bmoe3' >/dev/null 2>&1
bash "$HOSTDIR/agg_bandwidth.sh" 3 >> "$LOG" 2>&1; log "agg_bandwidth exit $?"
A 'am force-stop com.moephone.npu2' >/dev/null 2>&1
cool_gate; awake; bash "$HOSTDIR/bmoe_attn_gpu.sh" 3 >> "$LOG" 2>&1; log "bmoe_attn_gpu exit $?"
cool_gate; awake; bash "$HOSTDIR/thread_sweep.sh" 3 >> "$LOG" 2>&1; log "thread_sweep exit $?"
A 'am force-stop com.moephone.bmoe3; am force-stop com.moephone.npu2; settings put system high_performance_mode_on 0; svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1
log "phone restored"; touch "$R/CHAIN_REST_DONE"
