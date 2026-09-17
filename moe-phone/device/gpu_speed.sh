#!/system/bin/sh
# gpu_speed.sh — decode rate of Qwen3-30B-A3B Q4_0 streamed from flash with the expert slot cache on the
# Adreno GPU (llama.cpp PR #25294 + patches 0001/0005 v4), against the same binary with everything on
# the CPU. Interleaved by repeat; 128 generated tokens; llama-completion's own eval time is the rate.
# Correctness of the GPU cache path: OLMoE perplexity identical to whole-model GPU (gpu_stream3).
#   sh gpu_speed.sh REPS
set -u
REPS=${1:-2}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
B=$H/ocl-stream4
O=$H/gpu_speed_$(date +%Y%m%d_%H%M); mkdir -p "$O"
Q=$H/Qwen3-30B-A3B-Q4_0.gguf
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
run() {
  tag=$1_rep$3
  g=$(thermal_wait 30)
  echo "=== $tag $(date +%H:%M:%S) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo) BEFORE $g" | tee -a "$O/log.txt"
  LD_LIBRARY_PATH=/vendor/lib64 $B/llama-completion -m $Q -p "$P" -n 128 -no-cnv --temp 0 -c 1024 -b 32 -ub 32 $2 < /dev/null > "$O/$tag.out" 2> "$O/$tag.err"
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'eval time' "$O/$tag.err" | tr -s ' ' | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  run gpu_slots24 "-ngl 99 -t 4 --moe-stream-cache 24s --moe-stream-direct --moe-stream-io-threads 6" "$r"
  run cpu_slots32 "-ngl 0 -t 4 --cpu-strict 1 -C 0xf0 --moe-stream-cache 32s --moe-stream-direct --moe-stream-io-threads 4" "$r"
  run gpu_slots32 "-ngl 99 -t 4 --moe-stream-cache 32s --moe-stream-direct --moe-stream-io-threads 6" "$r"
done
echo "done $(date)" | tee "$O/DONE"
