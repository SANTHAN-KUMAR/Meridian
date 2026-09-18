#!/system/bin/sh
# bmoe_specadopt.sh — patch 0012 (--spec-adopt-selective) on the phone: does the predictor's speculation
# stop destroying itself, and does that reach the stall?
#
# Arms (same binary, bmoe-i8mm-0012 = 0001-0012; 0012 is default OFF so arm `base` is the 0011 engine):
#   base     no prefetch
#   predpf   --predict-prefetch                         (the current, self-cancelling behaviour)
#   predsel  --predict-prefetch --spec-adopt-selective  (confirmed guesses adopted, refuted unstarted ones cancelled)
# Latin rotation over 3 repeats so each arm takes each position once.
#
# PRE-REGISTERED (2026-09-18 ~16:00, before any row):
#   manipulation check  predsel prints "spec-adopt-selective: quiesces Q adopted A ..." with A > 0.
#   primary (counters)  experts integrated from speculation per token and speculative MiB per token
#                       (moe-prefetch line); MiB read per token and hit rate (moe-stream/moe-cache lines).
#                       Expectation from the teammate's analysis: predsel integrates many more speculated
#                       experts per token than predpf's 3.1, and discards far less speculative traffic.
#   secondary (time)    the engine's stall term and decode tok/s. Rate verdict only if predsel beats both
#                       other arms in every repeat; otherwise "no rate effect resolved at n=3".
#   lossless            every row's text must equal the first row's (0012 changes only when bytes arrive).
#   sh bmoe_specadopt.sh REPS
set -u
REPS=${1:-3}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_specadopt_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f"
run() {  # arm extra rep
  tag=$1_rep$3
  g=$(thermal_wait 30)
  mr=$(mem_ready 6500 120)
  foreign=$(ps -A -o ARGS | grep -E "llama-bench|bmoe-cli|com\.moephone" | grep -v grep | tr " " "_" | tr "\n" "," )
  echo "=== $tag $(date +%H:%M:%S) $mr foreign=[${foreign}] BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-0012 && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:|moe-prefetch|spec-adopt-selective:' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
PF="--predict-prefetch"
for r in $(seq 1 "$REPS"); do
  case $((r % 3)) in
    1) run base "" $r; run predpf "$PF" $r; run predsel "$PF --spec-adopt-selective" $r ;;
    2) run predpf "$PF" $r; run predsel "$PF --spec-adopt-selective" $r; run base "" $r ;;
    0) run predsel "$PF --spec-adopt-selective" $r; run base "" $r; run predpf "$PF" $r ;;
  esac
done
first=""
for f in "$O"/*.out; do
  t=$(sed -n '1,/^generation:/p' "$f" | sed '$d')
  if [ -z "$first" ]; then first="$t"; echo "text_reference=$(basename "$f")" >> "$O/log.txt"; fi
  if [ "$t" = "$first" ]; then echo "text_match $(basename "$f") OK" >> "$O/log.txt"; else echo "text_match $(basename "$f") DIFFERS" >> "$O/log.txt"; fi
done
echo "done $(date)" | tee "$O/DONE"
