#!/system/bin/sh
# bmoe_pin.sh — compute/IO thread placement for streamed Qwen3-30B-A3B, on a NON-throttled SoC.
#
# Why: BigMoeOnEdge's compute moves ~1.8 GB of weights per token at ~12 GB/s effective, while pinned
# llama.cpp moved OLMoE's at ~23 GB/s; its ggml workers are unpinned, and the earlier "more threads is
# slower" sweep (bmoe_levers) ADDED compute threads on top of 4 I/O lanes (oversubscribed 8 cores) and
# ran on a thermally capped CPU. Here the total stays at 8 threads, compute and I/O are pinned to
# disjoint cores (moe-phone patch: --cpu-mask strict threadpool, --io-cpu-mask lanes), and every run
# starts only after thermal_gate reports status 0 with uncapped cpufreq. The gate result and the thermal
# state after the run are part of each row. The gate waits at most 300 s: on 2026-09-17 the cpufreq cap
# stayed below hardware maximum even at thermal status 0 (2.02/2.48 GHz), so the caps are logged per run
# and the cells are interleaved rather than waiting indefinitely. One binary (patches 0002-0004) for every
# cell; --recycle-pages (patch 0004) is its own cell on top of the best placement candidate.
#   sh bmoe_pin.sh REPS
set -u
REPS=${1:-2}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_pin_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-ceil-mb 4000 --overlap --dense-weights anon"
run() { # name extra rep
  tag=$1_rep$3
  g=$(thermal_wait 300)
  echo "=== $tag $(date +%H:%M:%S) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo) BEFORE $g" | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-recycle && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  rc=$?
  echo "exit=$rc AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:|threadpool pinned|sched_setaffinity|recycle-pages:' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  run unpinned_t4_io4      "-t 4 --io-threads 4" "$r"
  run pin_t4c47_io4c03     "-t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f" "$r"
  run pin_t6c27_io2c01     "-t 6 --cpu-mask fc --io-threads 2 --io-cpu-mask 03" "$r"
  run pin_t2c67_io4c03     "-t 2 --cpu-mask c0 --io-threads 4 --io-cpu-mask 0f" "$r"
  run unpinned_recycle     "-t 4 --io-threads 4 --recycle-pages" "$r"
  run pin_t4_recycle       "-t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f --recycle-pages" "$r"
done
echo "done $(date)" | tee "$O/DONE"
