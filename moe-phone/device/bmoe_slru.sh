#!/system/bin/sh
# bmoe_slru.sh — segmented-LRU eviction (patch 0011) against plain LRU, on the phone.
#
# Simulation says +1.42 points of hit rate and 10.8% less miss traffic at this cache fraction
# (gates/evict_sim.py); the host run says 40.6% -> 46.4% hit and 188.4 -> 164.5 MiB/token re-read at a
# 1600 MiB cache. This measures it at the phone's own budget with the ABBA design the expert-order
# campaigns turned out to need: two cells and an alternating rotation leave one arm in the early slot of
# every repeat, which manufactured a 6% phantom effect once already.
#
# PRIMARY OUTCOME IS THE HIT RATE AND THE BYTES READ, not the decode rate. Those two counters do not
# depend on the clock -- they were identical to the decimal across twelve expert-order rows spanning
# thermal status 0 to 3 and 5.39 to 6.99 tok/s -- so this campaign stays interpretable on a hot phone,
# and the rate column is read only if the thermal state allows it.
set -u
REPS=${1:-3}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_slru_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f"
run() {
  tag=$1_rep$3
  g=$(thermal_wait 30)
  echo "=== $tag $(date +%H:%M:%S) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo) BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-slru && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:|expert order|expert-order probes' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
# ABBA within each repeat: positions 1 and 4 for one arm, 2 and 3 for the other, and the pair swaps
# which arm leads on alternate repeats so neither arm owns the outer slots.
for r in $(seq 1 "$REPS"); do
  if [ $((r % 2)) -eq 1 ]; then
    run lru "" "${r}a"; run slru "--expert-slru" "${r}a"
    run slru "--expert-slru" "${r}b"; run lru "" "${r}b"
  else
    run slru "--expert-slru" "${r}a"; run lru "" "${r}a"
    run lru "" "${r}b"; run slru "--expert-slru" "${r}b"
  fi
done
# Losslessness: every row's generated text must match the first row's, since only the order of a sum
# changed. A mismatch is a defect in the reordering, not a rounding difference.
first=""
for f in "$O"/*.out; do
  t=$(sed -n '1,/^generation:/p' "$f" | sed '$d')
  if [ -z "$first" ]; then first="$t"; echo "text_reference=$(basename "$f")" >> "$O/log.txt"; fi
  if [ "$t" = "$first" ]; then echo "text_match $(basename "$f") OK" >> "$O/log.txt"; else echo "text_match $(basename "$f") DIFFERS" >> "$O/log.txt"; fi
done
echo "done $(date)" | tee "$O/DONE"
