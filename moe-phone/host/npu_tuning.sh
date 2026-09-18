#!/bin/bash
# npu_tuning.sh — is the Hexagon NPU actually slower at decode, or was it just untuned?
#
# WHY THIS EXISTS. The in-app device comparison (results/2026-09-18/app_engine) put the NPU last at
# token generation -- 40.73 tok/s against the Adreno's 49.98 and four CPU threads' 44.47 -- and that
# number went into a claim. But the SAME rows put the NPU first at prefill by a wide margin: pp64 287.05
# against the GPU's 134.51 and the CPU's 19.19, i.e. 2.1x the GPU on compute-bound work. A device that
# is twice as fast when the work is big and slightly slower when the work is tiny is not bandwidth
# limited, it is LATENCY limited -- and every one of those rows ran the ggml-hexagon backend in its
# DEFAULT configuration, which is the part of the comparison that was not fair.
#
# The defaults that matter for batch-1 decode, read from ggml-hexagon.cpp:
#   opt_oppoll  = 0      the host waits on an INTERRUPT for each DSP batch to complete instead of
#                        polling. Decode is dozens of tiny round-trips per token; a wakeup costs
#                        hundreds of microseconds and there is no large computation to hide it behind.
#                        This is the single most likely explanation for the gap.
#   opt_hostbuf = false  activations do not use host (shared, uncached-mapped) buffers, so each op's
#                        inputs and outputs may be copied rather than shared.
#   opt_ndev    = 1      one DSP session.
#   opt_opbatch = 1024 / opt_opqueue = 64   how many ops go in a batch and how many batches may be in
#                        flight; a graph that does not fill a batch pays the round trip anyway.
#   opt_nhvx    = 0 (all HVX threads), opt_nhmx = 1 (HMX on) -- already the permissive defaults.
#
# ARMS (OLMoE, fully resident, in-app llama-bench, tg and pp both recorded so a change that helps decode
# and hurts prefill is visible):
#   default     nothing set -- reproduces the committed rows
#   oppoll      GGML_HEXAGON_OPPOLL=1
#   hostbuf     GGML_HEXAGON_HOSTBUF=1
#   both        OPPOLL=1 + HOSTBUF=1
#   ndev2       GGML_HEXAGON_DEVICES=2 (two sessions)
#   nobatch     GGML_HEXAGON_OPBATCH=1 -- the opposite direction, to check that batching is helping and
#               not hurting; a null here would mean the batch path is not on the critical path at all
# The CPU and GPU arms are re-run in the same campaign so the comparison is against rows taken under the
# same thermal conditions, not against last night's.
#
# Arm order is rotated per repeat. Every row carries its thermal state.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
PKG=${PKG:-com.moephone.npu2}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/$(date +%F)/npu_tuning"
M=/data/data/$PKG/files/olmoe.gguf
N=${N:-128}
REPS=${1:-2}
mkdir -p "$R"
log() { echo "$(date -Iseconds) $*" >> "$R/driver.log"; }

run_one() {  # tag  device  env
  local tag=$1 dev=$2 env=$3 i=0
  local devargs="-ngl~~99~~-dev~~$dev"
  [ "$dev" = "CPU" ] && devargs="-ngl~~0"
  adb shell "run-as $PKG sh -c 'rm -f files/out.txt files/bench.txt'" >/dev/null 2>&1 </dev/null
  adb shell "input keyevent KEYCODE_WAKEUP; am start -n $PKG/com.moephone.npu.Run --es env '$env' --es bench '-m~~$M~~-p~~64~~-n~~$N~~-r~~2~~-t~~4~~$devargs'" >/dev/null 2>&1 </dev/null
  while [ $i -lt 60 ]; do
    sleep 5
    if adb shell "run-as $PKG sh -c 'grep -c EXIT= files/out.txt 2>/dev/null'" 2>/dev/null </dev/null | tr -d '\r' | grep -qv '^0$'; then break; fi
    i=$((i + 1))
  done
  adb shell "run-as $PKG sh -c 'cat files/bench.txt'" > "$R/$tag.txt" 2>/dev/null </dev/null
  adb shell "run-as $PKG sh -c 'cat files/out.txt'"   > "$R/$tag.runner.txt" 2>/dev/null </dev/null
  adb shell '. /data/local/tmp/moe-stream/thermal_gate.sh; thermal_state' > "$R/$tag.state.txt" 2>/dev/null </dev/null
  log "$tag [$dev] env='$env' :: $(grep -oE '(pp|tg)[0-9]+ *\| *[0-9.]+' "$R/$tag.txt" | tr '\n' ' ')"
}

ARMS="
htp_default|HTP0|
htp_oppoll|HTP0|GGML_HEXAGON_OPPOLL=1
htp_hostbuf|HTP0|GGML_HEXAGON_HOSTBUF=1
htp_both|HTP0|GGML_HEXAGON_OPPOLL=1;GGML_HEXAGON_HOSTBUF=1
htp_ndev2|HTP0|GGML_HEXAGON_DEVICES=2
htp_nobatch|HTP0|GGML_HEXAGON_OPBATCH=1
gpu|GPUOpenCL|
cpu|CPU|
"

# NOTE: the arm list is walked with a for-loop over an array, NOT a `... | while read` pipeline. Inside
# such a pipeline every `adb shell` inherits the loop's stdin and swallows the remaining lines, so the
# campaign silently runs one arm per repeat and then reports DONE. That happened on the first attempt
# (results/2026-09-18/npu_tuning/driver.log, 09:41): two arms, then DONE.
log "START reps=$REPS n_gen=$N pkg=$PKG"
mapfile -t ARM_LIST < <(printf '%s\n' "$ARMS" | grep '|')
n=${#ARM_LIST[@]}
log "arms: $n"
for rep in $(seq 1 "$REPS"); do
  # rotate which arm leads, so a thermal ramp does not always land on the same one
  shift_by=$(( (rep - 1) % n ))
  for idx in $(seq 0 $((n - 1))); do
    line=${ARM_LIST[$(( (idx + shift_by) % n ))]}
    name=${line%%|*}; rest=${line#*|}; dev=${rest%%|*}; env=${rest#*|}
    [ -z "$name" ] && continue
    run_one "${name}_rep${rep}" "$dev" "$env"
    sleep 20
  done
done
log "DONE"
touch "$R/DONE"
