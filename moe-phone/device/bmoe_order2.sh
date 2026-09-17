#!/system/bin/sh
# bmoe_order2.sh — the SAME A/B as bmoe_order.sh, run with the instrumented binary, because the first
# repeat of that campaign showed no benefit AND no change in the stall (0.044 -> 0.047 s/token), which is
# what one sees when the reordering never fires. A probe that always answers "resident" and a probe that
# always answers "in flight" both make the two-pass loop behave exactly like the ascending one, and from
# the outside that is indistinguishable from "the lever is not there". bmoe-i8mm-order2 counts the
# answers and prints them ("bmoe: expert-order probes: R resident, F in flight"), so the next
# conclusion is about the mechanism rather than about a rate.
#
# Original question (patch 0010, --expert-resident-first): does computing a layer's RESIDENT experts
# first hide the reads of the ones still in flight?
#
# THE ARITHMETIC THAT MOTIVATED IT, all from committed artifacts. At the best measured configuration
# (results/2026-09-17/bmoe_cache.json cell ceil5000) a token costs 161.3 ms and splits into
# 93.0 ms compute residual + 20.0 ms cache management + 48.3 ms stall
# (gates/decode_budget.py). The stall is flash time the overlap did NOT hide, and it is almost exactly
# what one would predict if none of it were hidden: the cell reads 119.8 MiB/token of misses, which at
# the measured aggregate flash bandwidth (results/*/g1_storage.json, ~2.8 GB/s over 4 lanes) is ~43 ms.
#
# WHY IT WAS NOT HIDDEN. ggml's MUL_MAT_ID consumes a layer's selected experts in ASCENDING EXPERT ID
# (ggml-cpu.c, `for (int cur_a = 0; cur_a < n_as; ++cur_a)`), and the expert-ready hook blocks on each
# in turn. So a layer whose lowest-id selected expert is still being read stalls there, even when the
# other seven are already in memory and could have been computed meanwhile. With an 85.3% hit rate only
# about one expert per layer is a miss, so the compute of the resident seven -- roughly 2.4 ms of the
# layer's 2.8 ms -- is exactly the cover that miss needs.
#
# THE CHANGE IS PURE SCHEDULING: the same experts, the same sum, a different order, with the ready hook
# still blocking before any expert's bytes are read. It is therefore lossless, and that is checked
# rather than asserted -- the two arms decode the same prompt at temp 0, and their generated text must
# be identical (verified on the laptop before this campaign; checked again per row here).
#
# Cells differ ONLY in the flag; both use the same binary (bmoe-i8mm-order2), so the comparison cannot
# pick up a build difference. Order rotated per repeat, thermal and wake state per row.
#   sh bmoe_order.sh REPS
set -u
REPS=${1:-3}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_order2_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f"
run() {
  tag=$1_rep$3
  g=$(thermal_wait 30)
  echo "=== $tag $(date +%H:%M:%S) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo) BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-order2 && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:|expert order|expert-order probes' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  case $((r % 2)) in
    1) run idorder "" "$r"; run residentfirst "--expert-resident-first" "$r" ;;
    0) run residentfirst "--expert-resident-first" "$r"; run idorder "" "$r" ;;
  esac
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
