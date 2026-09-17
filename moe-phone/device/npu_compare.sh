#!/system/bin/sh
# npu_compare.sh — where should decode compute run on the 15R: pinned CPU, Adreno GPU (OpenCL) or Hexagon
# NPU (HTP v81, llama.cpp ggml-hexagon, unprivileged)? OLMoE-1B-7B Q4_0 fits in RAM, so this isolates the
# compute device from flash streaming. Same build (upstream llama.cpp 9f31776, OpenCL + Hexagon), llama-bench
# pp64/tg64, interleaved by repeat, quiesce + thermal/wake state per row. Device list logged first.
#   sh npu_compare.sh REPS
set -u
REPS=${1:-3}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
X=$H/hex
O=$H/npu_compare_$(date +%Y%m%d_%H%M); mkdir -p "$O"
M=$H/olmoe-1b-7b-0924-q4_0.gguf
export LD_LIBRARY_PATH=$X/lib:/vendor/lib64 ADSP_LIBRARY_PATH=$X/lib
$X/bin/llama-bench --list-devices > "$O/devices.txt" 2>&1
cell() {
  tag=$1_rep$3
  g=$(thermal_wait 30)
  echo "=== $tag $(date +%H:%M:%S) BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  $X/bin/llama-bench -m $M -p 64 -n 64 -r 1 -o csv $2 > "$O/$tag.csv" 2> "$O/$tag.err"
  echo "exit=$? AFTER $(thermal_state) $(awk -F, 'NR==1{for(i=1;i<=NF;i++){g=$i;gsub(/"/,"",g); if(g=="avg_ts")c=i; if(g=="n_gen")ng=i; if(g=="n_prompt")np=i}} NR>1{v=$c;gsub(/"/,"",v); a=$ng;gsub(/"/,"",a); b=$np;gsub(/"/,"",b); printf "pp%s_tg%s=%s ", b, a, v}' "$O/$tag.csv")" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  cell cpu_pinned "-dev none -ngl 0 -t 4 -C 0xf0 --cpu-strict 1" "$r"
  cell gpu_opencl "-dev GPUOpenCL -ngl 99 -t 4" "$r"
  cell npu_htp0   "-dev HTP0 -ngl 99 -t 4" "$r"
done
echo "done $(date)" | tee "$O/DONE"
