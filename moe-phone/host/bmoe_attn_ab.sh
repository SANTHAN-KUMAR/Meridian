#!/bin/bash
# bmoe_attn_ab.sh — 2026-09-18. OUR engine (BigMoeOnEdge, patch 0009) run INSIDE an app process, which is
# the only domain where the Hexagon DSP session opens, A/B'd on where the attention matmuls compute:
#   attn=cpu   everything on the CPU (the 6.20 tok/s configuration, but in-app)
#   attn=htp   attn_q/k/v/output on HTP0, routed experts still streamed + computed on the CPU
#   attn=gpu   the same four tensors on the Adreno via OpenCL
# The output head cannot move: it is Q6_K here, and ggml-hexagon refuses ne[1] > 32768 regardless.
# Cells are rotated (Latin-square) across repeats so a thermal drift cannot line up with one arm.
set -u
export ANDROID_SERIAL=192.168.0.65:5555
PKG=com.moephone.bmoe3
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18/bmoe_attn_ab"
M=/data/data/$PKG/files/qwen3.gguf
REPS=${1:-3}
mkdir -p "$R"
log() { echo "$(date -Iseconds) $*" >> "$R/driver.log"; }

# Byte-for-byte the flag string of the best measured cell so far -- pinned compute/IO + auto cache capped at
# 5000 MiB, median 6.20 tok/s (results/2026-09-17/bmoe_cache, device/bmoe_cache.sh) -- so the only things
# that differ between these rows and that baseline are the app process and the attention device.
PROMPT="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
COMMON="-m~~$M~~-p~~$PROMPT~~--chatml~~-n~~256~~--ubatch~~512~~--moe-stream~~--cache-mb~~auto~~--cache-floor-mb~~1024~~--cache-ceil-mb~~5000~~--overlap~~--dense-weights~~anon~~-t~~4~~--cpu-mask~~f0~~--io-threads~~4~~--io-cpu-mask~~0f"

run_cell() {  # name  extra-args(~~ separated, may be empty)
  local name=$1 extra=$2 i=0
  adb shell "run-as $PKG sh -c 'rm -f files/out.txt files/bench.txt'" >/dev/null 2>&1
  # quiesce other third-party apps, never our own package
  adb shell "for p in \$(pm list packages -3 | sed 's/^package://'); do [ \"\$p\" = $PKG ] || am force-stop \$p; done; am kill-all" >/dev/null 2>&1
  adb shell "input keyevent KEYCODE_WAKEUP" >/dev/null 2>&1
  sleep 30   # equal settle before every cell (the relaxed thermal gate: equal pause, not a temperature veto)
  adb shell '. /data/local/tmp/moe-stream/thermal_gate.sh; thermal_state' > "$R/$name.state_before.txt" 2>/dev/null
  adb shell "am start -n $PKG/com.moephone.npu.Run --es bench 'bmoe~~$COMMON~~--csv~~/data/data/$PKG/files/run.csv$extra'" >/dev/null 2>&1
  while [ $i -lt 100 ]; do
    sleep 15
    if adb shell "run-as $PKG sh -c 'grep -c EXIT= files/out.txt 2>/dev/null'" 2>/dev/null | tr -d '\r' | grep -qv '^0$'; then break; fi
    i=$((i + 1))
  done
  adb shell "run-as $PKG sh -c 'cat files/bench.txt'" > "$R/$name.txt" 2>/dev/null
  adb shell "run-as $PKG sh -c 'cat files/out.txt'"   > "$R/$name.runner.txt" 2>/dev/null
  adb shell "run-as $PKG sh -c 'cat files/run.csv'"   > "$R/$name.csv" 2>/dev/null
  adb shell '. /data/local/tmp/moe-stream/thermal_gate.sh; thermal_state' > "$R/$name.state_after.txt" 2>/dev/null
  log "done $name :: $(grep -oE 'generation: .*tok/s\)' "$R/$name.txt" | tail -1)"
}

log "START reps=$REPS pkg=$PKG"
for rep in $(seq 1 "$REPS"); do
  case $((rep % 3)) in
    1) order="cpu htp gpu" ;;
    2) order="htp gpu cpu" ;;
    0) order="gpu cpu htp" ;;
  esac
  for arm in $order; do
    case $arm in
      cpu) extra="" ;;
      htp) extra="~~--attn-device~~HTP0" ;;
      gpu) extra="~~--attn-device~~GPUOpenCL" ;;
    esac
    run_cell "attn_${arm}_rep${rep}" "$extra"
  done
done
log "DONE"
touch "$R/DONE"
