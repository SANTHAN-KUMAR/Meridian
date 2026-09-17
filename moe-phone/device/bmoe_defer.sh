#!/system/bin/sh
# bmoe_defer.sh — deferred, batched eviction (--defer-evict): the MADV_DONTNEED of evicted expert slices runs
# once after each decode instead of mid-graph, so TLB-shootdown IPIs hit idle compute cores. Motivation:
# --recycle-pages cut cache-management time 0.034 -> 0.013 s/token but raised compute by more (bmoe_pin2).
# Step 1, correctness: teacher-forced --ppl-step on a fixed text, with and without the flag; nll must be
#   identical (same kernels, same bytes; only page-release timing differs).
# Step 2, speed: pinned t4 (the confirmed best) vs pinned t4 + --defer-evict, 3 repeats, alternating order,
#   quiesce + thermal/wake state per row. One binary (patches 0002-0007).
set -u
REPS=${1:-3}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_defer_$(date +%Y%m%d_%H%M); mkdir -p "$O"
PIN="-t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f"
BASE="--moe-stream --cache-mb auto --cache-ceil-mb 4000 --overlap --dense-weights anon $PIN"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
for f in "" "--defer-evict"; do
  tag=ppl${f:+_defer}
  g=$(thermal_wait 30)
  ( cd $H/bmoe-i8mm-defer && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE --ppl $H/defer_ppl_text.txt --ppl-step $f > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "=== $tag exit=$? $(grep -h '^ppl:' "$O/$tag.out") $(grep -h 'defer-evict:' "$O/$tag.err" | head -1) BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
done
run() {
  tag=$1_rep$3
  g=$(thermal_wait 30)
  echo "=== $tag $(date +%H:%M:%S) BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-defer && LD_LIBRARY_PATH=. ./bmoe-cli -m $M --chatml -n 256 --ubatch 512 $BASE $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:|defer-evict:' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  if [ $((r % 2)) = 1 ]; then run pinned "" "$r"; run pinned_defer "--defer-evict" "$r"
  else run pinned_defer "--defer-evict" "$r"; run pinned "" "$r"; fi
done
echo "done $(date)" | tee "$O/DONE"
