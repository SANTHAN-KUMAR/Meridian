#!/bin/bash
# agg_bandwidth.sh — does this phone deliver MORE weight-byte throughput when two compute devices run at
# once than it does from the best single device?
#
# WHY THIS IS THE QUESTION. A decoded token reads a fixed set of weights (gates/gguf_active.py reads the
# set from the GGUF), so the goal rate is equivalent to a required weight-byte throughput. The single-
# device throughputs this SoC actually delivers, and the target rate each implies, are computed by
# gates/device_bandwidth.py -- every one of them is below the SoC's DRAM peak, so whether the goal is
# reachable at all depends on whether two devices ADD or merely SHARE one bottleneck. If they add,
# splitting the expert FFN across devices is the route to the goal and worth a patch; if they share, it
# is wasted work and no amount of engine engineering changes it. No number is repeated here on purpose:
# read them from that artifact.
#
# HOW. The same resident model (OLMoE) decodes on two devices at once, then on each alone, all through
# llama-bench so the number is llama.cpp's own tg rate:
#   solo_cpu   CPU only,  in the adb shell domain
#   solo_gpu   Adreno (OpenCL) only, in the app process
#   solo_htp   Hexagon only, in the app process
#   pair_*     both members started together; each rate is that process's own tg
# Aggregate throughput = sum over concurrent runs of (active_bytes_per_token * tg). A pair whose sum
# exceeds the best solo run is evidence that the devices add; a sum at or below the best solo is
# evidence of one shared bottleneck (and then per-device rates should each fall by roughly half).
#
# Both members of a pair use the same n_gen so their measurement windows overlap; the windows are not
# identical (load and prefill differ per device), so a pair's sum is a LOWER BOUND on the aggregate the
# hardware can deliver, which is the safe direction for a "they do not add" conclusion and the unsafe
# one for "they do" -- hence 3 repeats and the per-run start/end timestamps written next to each row.
set -u
export ANDROID_SERIAL=192.168.0.65:5555
PKG=${PKG:-com.moephone.npu2}          # holds olmoe.gguf in its files/ already
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/$(date +%F)/agg_bandwidth"
H=/data/local/tmp/moe-stream
SHELL_MODEL=$H/olmoe-1b-7b-0924-q4_0.gguf
APP_MODEL=/data/data/$PKG/files/olmoe.gguf
N=${N:-64}
REPS=${1:-3}
mkdir -p "$R"
log() { echo "$(date -Iseconds) $*" >> "$R/driver.log"; }

# llama-bench in the shell domain (CPU only; the DSP is refused here and OpenCL is not what we test here)
shell_cpu() {  # tag
  local tag=$1
  ( cd $H/ocl && LD_LIBRARY_PATH=. ./llama-bench -m $SHELL_MODEL -p 0 -n $N -r 1 -t 4 -ngl 0 ) \
    > "$R/$tag.txt" 2>&1
  echo "exit=$?" >> "$R/$tag.txt"
}

# llama-bench in the app process (the only domain where HTP0 opens; OpenCL works in both but is kept
# in-app so the pair members differ only in device, not in domain)
app_dev() {  # tag device
  local tag=$1 dev=$2 i=0
  adb shell "run-as $PKG sh -c 'rm -f files/out.txt files/bench.txt'" >/dev/null 2>&1
  adb shell "am start -n $PKG/com.moephone.npu.Run --es bench '-m~~$APP_MODEL~~-p~~0~~-n~~$N~~-r~~1~~-t~~4~~-dev~~$dev~~-ngl~~99'" >/dev/null 2>&1
  while [ $i -lt 120 ]; do
    sleep 5
    if adb shell "run-as $PKG sh -c 'grep -c EXIT= files/out.txt 2>/dev/null'" 2>/dev/null | tr -d '\r' | grep -qv '^0$'; then break; fi
    i=$((i + 1))
  done
  adb shell "run-as $PKG sh -c 'cat files/bench.txt'" > "$R/$tag.txt" 2>/dev/null
}

stamp() { echo "$1 $(date -Iseconds)" >> "$R/$2.stamps"; }

run_solo() {  # tag  kind  [device]
  local tag=$1 kind=$2 dev=${3:-}
  adb shell "input keyevent KEYCODE_WAKEUP" >/dev/null 2>&1
  sleep 20
  stamp start "$tag"
  case $kind in
    cpu) shell_cpu "$tag" ;;
    app) app_dev "$tag" "$dev" ;;
  esac
  stamp end "$tag"
  log "solo $tag: $(grep -oE 'tg[0-9]+ *\| *[0-9.]+' "$R/$tag.txt" | tail -1)"
}

run_pair() {  # tagA kindA devA  tagB kindB devB
  local ta=$1 ka=$2 da=$3 tb=$4 kb=$5 db=$6
  adb shell "input keyevent KEYCODE_WAKEUP" >/dev/null 2>&1
  sleep 20
  stamp start "$ta"; stamp start "$tb"
  case $ka in cpu) shell_cpu "$ta" & ;; app) app_dev "$ta" "$da" & ;; esac
  pa=$!
  case $kb in cpu) shell_cpu "$tb" & ;; app) app_dev "$tb" "$db" & ;; esac
  pb=$!
  wait $pa; stamp end "$ta"
  wait $pb; stamp end "$tb"
  log "pair $ta+$tb: $(grep -oE 'tg[0-9]+ *\| *[0-9.]+' "$R/$ta.txt" | tail -1) | $(grep -oE 'tg[0-9]+ *\| *[0-9.]+' "$R/$tb.txt" | tail -1)"
}

log "START reps=$REPS n_gen=$N pkg=$PKG"
# Warm the model into the page cache first: the first run after an install or a reboot otherwise
# spends minutes faulting a multi-GB file in and can exceed the driver\'s wait, which is exactly how
# the 09:44 sweep lost its htp_default and gpu rows (both were still running when it gave up).
adb shell "run-as $PKG sh -c \'cat files/olmoe.gguf > /dev/null 2>&1\'" >/dev/null 2>&1 </dev/null
log "model warmed into page cache"
for rep in $(seq 1 "$REPS"); do
  run_solo "solo_cpu_rep$rep" cpu
  run_solo "solo_gpu_rep$rep" app GPUOpenCL
  run_solo "solo_htp_rep$rep" app HTP0
  run_pair "pair_cpu_with_gpu_cpu_rep$rep" cpu "" "pair_cpu_with_gpu_gpu_rep$rep" app GPUOpenCL
  run_pair "pair_cpu_with_htp_cpu_rep$rep" cpu "" "pair_cpu_with_htp_htp_rep$rep" app HTP0
done
log "DONE"
touch "$R/DONE"
