#!/system/bin/sh
# gpu_stream_check2.sh — rerun after the OpenCL router-reorder fixes (streamed ids: reorder trigger, expert count, row stride). the MoE expert cache ON THE ADRENO GPU while experts stream from flash
# (llama.cpp PR #25294 branch + moe-phone patch 0005: partial Q4_0 slot uploads into the OpenCL SoA layout).
#
# Step 1, correctness (OLMoE Q4_0 fits in RAM, so a non-streamed GPU run is the reference): perplexity of
#   gpu_full   -ngl 99, every expert uploaded whole (upstream conversion path)
#   gpu_stream -ngl 99 --moe-stream-cache 32s (half the experts per layer, slots filled by partial uploads)
#   cpu        -ngl 0 (context only)
# gpu_stream must match gpu_full: same kernels on the same bytes, only which slot holds an expert differs.
# Step 2, speed: Qwen3-30B-A3B Q4_0 (17 GB, beyond RAM) decode with dense layers and a slot cache on the GPU.
# Thermal state logged before/after each run (thermal_gate.sh).
#   sh gpu_stream_check.sh
set -u
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
B=$H/ocl-stream2
O=$H/gpu_stream2_$(date +%Y%m%d_%H%M); mkdir -p "$O"
OL=$H/olmoe-1b-7b-0924-q4_0.gguf
Q=$H/Qwen3-30B-A3B-Q4_0.gguf
T=$H/verify_cost_text.txt
ppl() { # name flags
  g=$(thermal_wait 30)
  echo "=== $1 $(date +%H:%M:%S) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo) BEFORE $g" | tee -a "$O/log.txt"
  LD_LIBRARY_PATH=/vendor/lib64 $B/llama-perplexity -m $OL -f $T -c 128 -b 128 -t 4 $2 > "$O/$1.out" 2> "$O/$1.err"
  echo "exit=$? AFTER $(thermal_state) $(grep -hoE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' "$O/$1.out" "$O/$1.err") $(grep -hE 'expert cache size|disabling op offload' "$O/$1.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
gen() { # name flags
  g=$(thermal_wait 30)
  echo "=== $1 $(date +%H:%M:%S) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo) BEFORE $g" | tee -a "$O/log.txt"
  LD_LIBRARY_PATH=/vendor/lib64 $B/llama-completion -m $Q -p "Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field" -n 128 -no-cnv --temp 0 $2 < /dev/null > "$O/$1.out" 2> "$O/$1.err"
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'eval time|expert cache size|disabling op offload' "$O/$1.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
ppl gpu_full   "-ngl 99"
ppl gpu_stream "-ngl 99 --moe-stream-cache 32s"
# cpu reference omitted in v2 (context only, not part of the correctness pair)
gen qwen_gpu_stream24 "-ngl 99 -t 4 --moe-stream-cache 24s --moe-stream-direct --moe-stream-io-threads 6"
echo "done $(date)" | tee "$O/DONE"
