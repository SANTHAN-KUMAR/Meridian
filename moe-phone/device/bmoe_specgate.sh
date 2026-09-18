#!/system/bin/sh
# bmoe_specgate.sh — patch 0013 confidence gates on predictive speculation; the sweep pre-registered in
# tools/patches/0013-NOTE.md (arms, grid, decision rule), run with binary bmoe-i8mm-0013 (0001-0013).
# Common: --predict-prefetch --predict-spec-max 3 --spec-adopt-selective, except `base` (no prefetch).
#   base | sel | r1 (--predict-spec-rank 1) | r2 (rank 2) | m05 (--predict-spec-margin 0.5) | m10 (margin 1.0)
# Cyclic Latin order over 3 repeats (shift 2 per repeat), text identity per row.
# CLARIFICATION, written 2026-09-18 ~16:35 BEFORE any row: the NOTE's absolute thresholds (read <= 125.3
# MiB/tok, stall <= 37.8 ms) were derived from the 0012 run at --predict-spec-max 2, while every arm here
# runs at 3. The PRIMARY decision therefore applies the NOTE's rule in its relative form against THIS
# sweep's own sel and base medians (removes >= half of sel's read excess over base AND keeps >= 80% of
# sel's stall cut vs base); the absolute thresholds are reported alongside.
#   sh bmoe_specgate.sh REPS
set -u
REPS=${1:-3}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_specgate_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f"
S="--predict-prefetch --predict-spec-max 3 --spec-adopt-selective"
args() {
  case $1 in
    base) echo "" ;; sel) echo "$S" ;;
    r1) echo "$S --predict-spec-rank 1" ;; r2) echo "$S --predict-spec-rank 2" ;;
    m05) echo "$S --predict-spec-margin 0.5" ;; m10) echo "$S --predict-spec-margin 1.0" ;;
  esac
}
run() {  # arm rep
  tag=$1_rep$2
  g=$(thermal_wait 30); mr=$(mem_ready 6500 120)
  foreign=$(ps -A -o ARGS | grep -E "llama-bench|bmoe-cli|com\.moephone" | grep -v grep | tr " " "_" | tr "\n" "," )
  echo "=== $tag $(date +%H:%M:%S) $mr foreign=[${foreign}] BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-0013 && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $(args $1) --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:|moe-prefetch|moe-overlap|spec-adopt-selective:|predict-gate:' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
ARMS="base sel r1 r2 m05 m10"
for r in $(seq 1 "$REPS"); do
  sh_=$(( (r - 1) * 2 ))
  set -- $ARMS; n=$#; i=0
  while [ $i -lt $n ]; do
    k=$(( (i + sh_) % n + 1 )); eval a=\${$k}; run $a $r; i=$((i + 1))
  done
done
first=""
for f in "$O"/*.out; do
  t=$(sed -n '1,/^generation:/p' "$f" | sed '$d')
  if [ -z "$first" ]; then first="$t"; echo "text_reference=$(basename "$f")" >> "$O/log.txt"; fi
  if [ "$t" = "$first" ]; then echo "text_match $(basename "$f") OK" >> "$O/log.txt"; else echo "text_match $(basename "$f") DIFFERS" >> "$O/log.txt"; fi
done
echo "done $(date)" | tee "$O/DONE"
