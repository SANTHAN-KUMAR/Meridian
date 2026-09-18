#!/system/bin/sh
# bmoe_stack.sh — ONE decisive comparison: every lever that helped, stacked, against the plain baseline.
# Individually each lever is 3-9% (below what n=3 resolves); stacked, the effect should be large enough
# to resolve in a single balanced campaign, which ends the one-lever-at-a-time loop.
#   base   the current default (LRU, no prefetch)
#   stack  --expert-slru + --predict-prefetch --spec-adopt-selective [+ gate from STACK_GATE, chosen by the
#          0013 sweep's pre-registered rule before this runs]
# Binary bmoe-i8mm-0013 (patches 0001-0013) for both arms. ABBA x6 = 24 rows.
# PRE-REGISTERED (2026-09-18 ~17:00, before any row; revised from the ~16:50 draft BEFORE any row existed,
# because its power was too low -- noise measured on 24 existing rows, see gates/stack_summary.py header):
#   keep row  exit=0, granted budget 5000 MiB, foreign=[]
#   PRIMARY   stall+mgmt ms/token (the terms the levers act on; row sd ~3.6 ms vs an expected ~10 ms effect)
#   GUARD     compute ms/token: if it rises by more than the stall+mgmt saving, the stack loses on net
#   SECONDARY decode tok/s (row cv ~5%; with 12 rows/arm SE/|effect| ~0.45 at the expected +4.5%)
#   COUNTERS  MiB read/token and hit rate (clock-independent)
#   estimate  mean of within-repeat paired differences (stack - base), SE over the 6 repeats, SE/|diff|
#   verdict   a difference is DECISIVE only if SE/|diff| < 0.5 AND its sign holds in >= 5 of 6 repeats;
#             otherwise "not resolved", with the n that would resolve it. Text identical in every row.
# RE-RUN NOTE (2026-09-18 17:30): the first run (bmoe_stack_20260918_1712, binary 0013) was stopped after
# repeat 1 because the stack read 2.0-2.1x the base's flash MiB/token -- a list-corruption bug in patch 0011
# (retain() re-linked PROTECTED entries into the probation list; only reachable with the predictor on). Fixed
# by patch 0015; this re-run uses binary bmoe-i8mm-0015 with the pre-registration above unchanged.
#   sh bmoe_stack.sh REPS "GATE_FLAGS" [BINARY_DIR]
set -u
REPS=${1:-6}
GATE=${2:-}
BIN=${3:-bmoe-i8mm-0013}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_stack_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f"
STACK="--expert-slru --predict-prefetch --spec-adopt-selective $GATE"
echo "stack_flags=[$STACK] binary=$BIN" >> "$O/log.txt"
run() {
  tag=$1_rep$3
  g=$(thermal_wait 30); mr=$(mem_ready 6500 120)
  foreign=$(ps -A -o ARGS | grep -E "llama-bench|bmoe-cli|com\.moephone" | grep -v grep | tr " " "_" | tr "\n" "," )
  echo "=== $tag $(date +%H:%M:%S) $mr foreign=[${foreign}] BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/$BIN && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
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
