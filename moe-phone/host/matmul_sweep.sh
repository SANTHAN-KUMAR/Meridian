#!/bin/bash
# matmul_sweep.sh — every matmul shape a Qwen3-30B-A3B token actually computes, on every device this
# phone exposes, measured per op instead of end to end.
#
# WHY. The end-to-end A/B for --attn-device cannot resolve its own effect: the four attention matmuls are
# a small share of decode node time (results/2026-09-17/compute_trace.json node table) and the predicted
# change lands inside the run-to-run spread of the best cell (results/2026-09-17/bmoe_cache.json). Rather
# than chase that with more repeats, the device question is asked directly of the ops
# (host/app/ggml_matmul_bench.cpp), and the answer is then used to predict what any placement can buy.
#
# SHAPES, read from the model's own tensor table (gguf-py), not assumed:
#   attn_q       k=2048 n=4096   Q4_0     per layer, per token
#   attn_k/v     k=2048 n=512    Q4_0     per layer, per token (GQA: 4 kv heads x 128)
#   attn_output  k=4096 n=2048   Q4_0     per layer, per token
#   ffn_gate/up  k=2048 n=768    Q4_0     MUL_MAT_ID over 8 of 128 experts, per layer, per token
#   ffn_down     k=768  n=2048   Q4_1     MUL_MAT_ID over 8 of 128 experts, per layer, per token -- Q4_1 is the down type
#                                          of layers 0-5 ONLY; layers 6-47 (42 of 48) are Q4_0 (GGUF tensor table, 2026-09-18)
#   output head  k=2048 n=151936 Q6_K     once per token -- NOT swept: ggml-hexagon refuses ne[1] > 32768
#                                         and does not implement Q6_K, so it cannot move at all
# The expert rows are the ones that decide anything: MUL_MAT_ID is the majority of decode node time
# (claim trace_mul_mat_id_share).
#
# Every row reports compute ms/GB/s AND upload ms/GB/s. Upload is not a setup cost for a streaming
# engine -- a cache miss pays it per token -- and on ggml-hexagon it includes a tiled repack of the
# weights, which is exactly the cost that decides whether streamed experts can compute on the DSP.
#
# --copies rotates over independent weight tensors so the timing loop cannot be served by cache; the
# benchmark also verifies every device's output against the CPU backend and refuses to report timings
# for a device that disagrees.
set -u
export ANDROID_SERIAL=192.168.0.65:5555
PKG=${PKG:-com.moephone.bmoe3}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/$(date +%F)/matmul_sweep"
ITERS=${ITERS:-60}
REPS=${1:-2}
mkdir -p "$R"
log() { echo "$(date -Iseconds) $*" >> "$R/driver.log"; }

# name              args
# 2026-09-18 19:40: expert shapes carry engine-style weight names (--wname), because ggml-opencl only runs its
# Adreno MoE kernels for weights named *ffn*exps*; the 14:42 sweep (weight<j>) measured the GENERIC GPU fallback.
SHAPES="
attn_q|--k 2048 --n 4096 --type q4_0 --copies 24
attn_kv|--k 2048 --n 512 --type q4_0 --copies 96
attn_output|--k 4096 --n 2048 --type q4_0 --copies 24
ffn_gate_up|--k 2048 --n 768 --experts 128 --ids 8 --type q4_0 --copies 2 --wname ffn_up_exps
ffn_down|--k 768 --n 2048 --experts 128 --ids 8 --type q4_1 --copies 2 --wname ffn_down_exps
"

run_one() {  # tag  args
  local tag=$1 args=$2 i=0
  local jargs=$(echo "mmb $args --iters $ITERS" | sed 's/ /~~/g')
  adb shell "run-as $PKG sh -c 'rm -f files/out.txt files/bench.txt'" >/dev/null 2>&1 </dev/null
  adb shell "input keyevent KEYCODE_WAKEUP; am start -n $PKG/com.moephone.npu.Run --es bench '$jargs'" >/dev/null 2>&1 </dev/null
  while [ $i -lt 60 ]; do
    sleep 5
    if adb shell "run-as $PKG sh -c 'grep -c EXIT= files/out.txt 2>/dev/null'" 2>/dev/null </dev/null | tr -d '\r' | grep -qv '^0$'; then break; fi
    i=$((i + 1))
  done
  adb shell "run-as $PKG sh -c 'cat files/bench.txt'" > "$R/$tag.txt" 2>/dev/null </dev/null
  adb shell "run-as $PKG sh -c 'cat files/out.txt'"   > "$R/$tag.runner.txt" 2>/dev/null </dev/null
  log "$tag :: $(grep -o 'RESULT .*' "$R/$tag.txt" | tail -1)"
}

log "START reps=$REPS iters=$ITERS pkg=$PKG"
# for-loop over an array, not a pipeline: see the note in npu_tuning.sh about adb eating a loop's stdin.
mapfile -t SHAPE_LIST < <(printf '%s\n' "$SHAPES" | grep '|')
for rep in $(seq 1 "$REPS"); do
  for line in "${SHAPE_LIST[@]}"; do
    name=${line%%|*}; args=${line#*|}
    [ -z "$name" ] && continue
    # device order rotated per repeat so a thermal drift cannot line up with one device
    case $((rep % 3)) in
      1) order="CPU HTP0 GPUOpenCL" ;;
      2) order="HTP0 GPUOpenCL CPU" ;;
      0) order="GPUOpenCL CPU HTP0" ;;
    esac
    for dev in $order; do
      run_one "${name}_${dev}_rep${rep}" "--device $dev $args --threads 4"
      sleep 10
    done
  done
done
log "DONE"
touch "$R/DONE"
