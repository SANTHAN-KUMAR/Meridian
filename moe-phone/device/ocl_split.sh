#!/system/bin/sh
# ocl_split.sh — where should decode compute run on the 15R: CPU cores, the Adreno GPU (OpenCL), or split
# (attention + shared weights on the GPU, routed experts on the CPU, the split a streamed >RAM MoE needs)?
# Same llama.cpp build (OpenCL + i8mm CPU), same model, interleaved by repeat, battery temperature and
# MemAvailable logged per run so heat and memory state are part of the row.
#   sh ocl_split.sh MODEL REPS NGEN
set -u
M=$1; REPS=${2:-3}; N=${3:-64}
B=/data/local/tmp/moe-stream/ocl/llama-bench
O=/data/local/tmp/moe-stream/ocl_split_$(date +%Y%m%d_%H%M); mkdir -p "$O"
echo "cell,rep,tg_tok_s,pp_tok_s,temp_dC_before,temp_dC_after,memavail_kb,exit" > "$O/runs.csv"
temp() { dumpsys battery | awk '/^  temperature:/{print $2}'; }
cell() { # name flags rep
  t0=$(temp); ma=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  LD_LIBRARY_PATH=/vendor/lib64 $B -m "$M" -p 64 -n "$N" -r 1 -o csv $2 > "$O/$1_r$3.csv" 2> "$O/$1_r$3.err"; ex=$?
  tg=$(awk -F, 'NR==1{for(i=1;i<=NF;i++){g=$i;gsub(/"/,"",g); if(g=="avg_ts")c=i; if(g=="n_gen")ng=i}} NR>1{v=$c;gsub(/"/,"",v); n=$ng;gsub(/"/,"",n); if(n>0)print v}' "$O/$1_r$3.csv")
  pp=$(awk -F, 'NR==1{for(i=1;i<=NF;i++){g=$i;gsub(/"/,"",g); if(g=="avg_ts")c=i; if(g=="n_prompt")np=i}} NR>1{v=$c;gsub(/"/,"",v); n=$np;gsub(/"/,"",n); if(n>0)print v}' "$O/$1_r$3.csv")
  echo "$1,$3,$tg,$pp,$t0,$(temp),$ma,$ex" | tee -a "$O/runs.csv"
}
for r in $(seq 1 "$REPS"); do
  cell cpu_t4_pinned "-ngl 0 -t 4 --cpu-strict 1 --cpu-mask 0xf0" "$r"
  cell gpu_all       "-ngl 99 -t 4" "$r"
  cell gpu_dense_cpu_experts "-ngl 99 -ncmoe 99 -t 4 --cpu-strict 1 --cpu-mask 0xf0" "$r"
done
echo "done $(date)" > "$O/DONE"
