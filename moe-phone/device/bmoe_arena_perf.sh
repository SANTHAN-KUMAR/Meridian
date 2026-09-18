#!/system/bin/sh
# bmoe_arena_perf.sh — WHY does the slot arena's compute residual rise 14 ms/token (claim
# arena2_compute_delta_ms)? Hardware/software counters for base vs arena, via the system simpleperf
# (works on shell-owned processes without root). Transparent huge pages are "never" on this kernel
# (/sys/kernel/mm/transparent_hugepage/enabled, 2026-09-18), so a MADV_HUGEPAGE variant cannot be tried;
# this decides between the remaining explanations before any variant is written:
#   faults     minor-faults per token higher under the arena -> pre-touch/contiguous pool is the fix
#   TLB        dTLB-load-misses per token higher             -> slot layout / locality is the cause
#   cache      L1/LLC misses per token higher                -> the matmul reads scattered slots
#   none       counters equal, cycles higher                 -> look at the expert-data hook path
# Counters are totals per run (prefill + decode); both arms run the identical prompt and -n, so the
# difference is attributable to the arm. ABAB x2, -n 128.
set -u
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_arena_perf_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 128 --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f"
EV="minor-faults,major-faults,page-faults,dTLB-loads,dTLB-load-misses,L1-dcache-load-misses,cache-misses,cpu-cycles,instructions,task-clock"
run() {
  tag=$1_rep$3
  g=$(thermal_wait 30); mr=$(mem_ready 6500 120)
  echo "=== $tag $(date +%H:%M:%S) $mr BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-order2 && LD_LIBRARY_PATH=. simpleperf stat -e $EV -o "$O/$tag.perf" ./bmoe-cli -m $M $BASE $2 -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:|slot-arena' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
run base "" 1; run arena "--slot-arena" 1; run arena "--slot-arena" 2; run base "" 2
echo "done $(date)" | tee "$O/DONE"
