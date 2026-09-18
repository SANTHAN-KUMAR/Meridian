#!/system/bin/sh
# bmoe_stack.sh — ONE decisive comparison: every lever that helped, stacked, against the plain baseline.
# Individually each lever is 3-9% (below what n=3 resolves); stacked, the effect should be large enough
# to resolve in a single balanced campaign, which ends the one-lever-at-a-time loop.
#   base   the current default (LRU, no prefetch)
#   stack  --expert-slru + --predict-prefetch --spec-adopt-selective [+ gate from STACK_GATE, chosen by the
#          0013 sweep's pre-registered rule before this runs]
# Binary bmoe-i8mm-0013 (patches 0001-0013) for both arms. ABBA x4 = 16 rows.
# PRE-REGISTERED (2026-09-18 ~16:50, before any row):
#   keep row  exit=0, granted budget 5000 MiB, foreign=[]
#   estimate  median(stack)/median(base) - 1 over kept rows, every row listed
#   verdict   "stack faster" iff stack's within-repeat mean beats base's in all 4 repeats; else not resolved
#   also      text identical in every row; hit rate, MiB/token, stall, compute, mgmt per arm
#   sh bmoe_stack.sh REPS "GATE_FLAGS"
set -u
REPS=${1:-4}
GATE=${2:-}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_stack_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f"
STACK="--expert-slru --predict-prefetch --spec-adopt-selective $GATE"
echo "stack_flags=[$STACK]" >> "$O/log.txt"
run() {
  tag=$1_rep$3
  g=$(thermal_wait 30); mr=$(mem_ready 6500 120)
  foreign=$(ps -A -o ARGS | grep -E "llama-bench|bmoe-cli|com\.moephone" | grep -v grep | tr " " "_" | tr "\n" "," )
  echo "=== $tag $(date +%H:%M:%S) $mr foreign=[${foreign}] BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-0013 && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:|moe-prefetch|moe-overlap|spec-adopt-selective:|predict-gate:' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  if [ $((r % 2)) -eq 1 ]; then run base "" "${r}a"; run stack "$STACK" "${r}a"; run stack "$STACK" "${r}b"; run base "" "${r}b"
  else run stack "$STACK" "${r}a"; run base "" "${r}a"; run base "" "${r}b"; run stack "$STACK" "${r}b"; fi
done
first=""
for f in "$O"/*.out; do
  t=$(sed -n '1,/^generation:/p' "$f" | sed '$d')
  if [ -z "$first" ]; then first="$t"; echo "text_reference=$(basename "$f")" >> "$O/log.txt"; fi
  if [ "$t" = "$first" ]; then echo "text_match $(basename "$f") OK" >> "$O/log.txt"; else echo "text_match $(basename "$f") DIFFERS" >> "$O/log.txt"; fi
done
echo "done $(date)" | tee "$O/DONE"
