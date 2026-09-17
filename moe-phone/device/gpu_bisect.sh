#!/system/bin/sh
# gpu_bisect.sh — locate the error in the Adreno GPU expert cache (patch 0005 v2): OLMoE Q4_0 perplexity was
# 14.6022 with --moe-stream-cache 32s on the GPU vs 11.5482 whole-model GPU and 11.5128 CPU
# (results/2026-09-17/gpu_stream2). Each cell changes one factor:
#   cpu_stream32   streaming on the CPU        -> does the streaming engine itself change the result here?
#   gpu_stream64   all 64 experts resident      -> no eviction, one wave; upload + remap + GEMM kernel only
#   gpu_full_ub1   whole model, 1 token/ubatch  -> GEMV (single-token) reference
#   gpu_stream32_ub1 cache, 1 token/ubatch      -> GEMV path with slot uploads
set -u
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
B=$H/ocl-stream2
O=$H/gpu_bisect_$(date +%Y%m%d_%H%M); mkdir -p "$O"
OL=$H/olmoe-1b-7b-0924-q4_0.gguf
T=$H/verify_cost_text.txt
ppl() {
  g=$(thermal_wait 30)
  echo "=== $1 $(date +%H:%M:%S) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo) BEFORE $g" | tee -a "$O/log.txt"
  LD_LIBRARY_PATH=/vendor/lib64 $B/llama-perplexity -m $OL -f $T -c 128 -b 128 -t 4 $2 > "$O/$1.out" 2> "$O/$1.err"
  echo "exit=$? AFTER $(thermal_state) $(grep -hoE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' "$O/$1.out" "$O/$1.err")" | tee -a "$O/log.txt"
}
ppl cpu_stream32     "-ngl 0 --moe-stream-cache 32s"
ppl gpu_stream64     "-ngl 99 --moe-stream-cache 64s"
ppl gpu_full_ub1     "-ngl 99 -ub 1"
ppl gpu_stream32_ub1 "-ngl 99 --moe-stream-cache 32s -ub 1"
echo "done $(date)" | tee "$O/DONE"
