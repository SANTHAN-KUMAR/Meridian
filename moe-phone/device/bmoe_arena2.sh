#!/system/bin/sh
# bmoe_arena2.sh — re-measure --slot-arena (patch 0008) at the CURRENT best operating point, and sample
# memory during every run, because the first campaign's verdict hid the most interesting number in this
# project.
#
# WHAT THE FIRST CAMPAIGN ACTUALLY SHOWED (results/2026-09-17/bmoe_arena, re-read with
# gates/decode_budget.py's decomposition):
#     cell            decode     compute   mgmt   stall
#     pinned          179.3 ms    91.0     34.0    54.3
#     pinned_arena    185.4 ms   105.0      2.0    78.4
# The arena did exactly what it was designed to do -- cache management fell from 34 ms to 2 ms, the
# largest single saving found anywhere in this project -- and it still lost, because compute rose 14 ms
# and the stall rose 24 ms. "Net loss" was the right verdict and the wrong conclusion to stop at.
#
# TWO THINGS DIFFER NOW, which is why it is worth re-running rather than re-reading:
#   1. That campaign ran at a 4000 MiB budget and a 78.0% hit rate. The current best configuration runs
#      at ~5000 MiB and ~85%, so there are roughly a third fewer misses for the arena's penalty to
#      attach to, while its 32 ms saving does not depend on the hit rate at all.
#   2. The hypothesis for the penalty is memory: arena slots are reserved and never released, so the
#      process's anon footprint exceeds the cache budget the engine thinks it is honouring. If that is
#      the mechanism, MemAvailable during an arena run is lower and SwapFree falls further -- and this
#      campaign samples both every 2 s, which the first one did not record at all.
# The arena is also known lossless: the first campaign's perplexity cells agree exactly (8.1273 both).
#
# DESIGN: ABBA within each repeat (A,B,B,A, leading arm swapped on alternate repeats) so a monotone
# drift cancels inside the repeat -- the unbalanced rotation is what manufactured a 6% phantom effect in
# bmoe_order2 (see results/2026-09-18/bmoe_order/FINDING.md).
#   sh bmoe_arena2.sh REPS
set -u
REPS=${1:-3}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_arena2_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f"
run() {
  tag=$1_rep$3
  g=$(thermal_wait 30)
  echo "=== $tag $(date +%H:%M:%S) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo) BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  # sample memory while the run happens; the engine's own budget figure does not include arena slots
  ( while :; do awk '/MemAvailable|SwapFree/{printf "%s ", $2}' /proc/meminfo; echo; sleep 2; done > "$O/$tag.mem" ) &
  sampler=$!
  ( cd $H/bmoe-i8mm-order2 && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  e=$?
  kill $sampler 2>/dev/null
  mem_min=$(awk '{print $1}' "$O/$tag.mem" | sort -n | head -1)
  swap_min=$(awk '{print $2}' "$O/$tag.mem" | sort -n | head -1)
  echo "exit=$e AFTER $(thermal_state) memavail_min=$mem_min swapfree_min=$swap_min $(grep -hE 'generation:|moe-stream:|moe-cache:|slot-arena' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  if [ $((r % 2)) -eq 1 ]; then
    run base "" "${r}a"; run arena "--slot-arena" "${r}a"
    run arena "--slot-arena" "${r}b"; run base "" "${r}b"
  else
    run arena "--slot-arena" "${r}a"; run base "" "${r}a"
    run base "" "${r}b"; run arena "--slot-arena" "${r}b"
  fi
done
# losslessness: only where the bytes live changed, so every row's text must match the first row's
first=""
for f in "$O"/*.out; do
  t=$(sed -n '1,/^generation:/p' "$f" | sed '$d')
  if [ -z "$first" ]; then first="$t"; echo "text_reference=$(basename "$f")" >> "$O/log.txt"; fi
  if [ "$t" = "$first" ]; then echo "text_match $(basename "$f") OK" >> "$O/log.txt"; else echo "text_match $(basename "$f") DIFFERS" >> "$O/log.txt"; fi
done
echo "done $(date)" | tee "$O/DONE"
