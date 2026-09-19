#!/system/bin/sh
# bmoe_prerepack.sh — all weights on ggml-cpu's repacked (interleaved i8mm) kernels, as stock llama.cpp runs this CPU:
#   base   the stack on the original Qwen3-30B-A3B-Q4_0.gguf (generic Q4_0 kernels)
#   stack  the stack on Qwen3-30B-A3B-Q4_0.repacked.gguf (expert slices pre-repacked on this phone by repack_gguf, marker
#          checked) + --experts-prerepacked + --repack-dense (dense matmul weights repacked once at load)
# Why: repack_bench on the phone (VERDICT BUILD): repacked MUL_MAT_ID 0.596x (gate/up) and 0.786x (down) of generic. The run-time
# repack (--repack-experts) moved the saving into I/O-lane stall (smoke 0844: compute -15, stall +17 ms); pre-repacked
# bytes remove that cost. One lever: llama.cpp's own kernels for every weight.
# PRE-REGISTERED (2026-09-19 09:05, before any Qwen3 row of it):
#   FIDELITY (Tier E with a declared reordering tolerance): teacher-forced --ppl on defer_ppl_text.txt, base vs stack:
#            |dPPL|/PPL <= 0.2% and top-1 next-token-hit changes <= 1% of scored tokens. (Laptop OLMoE, pure reordering:
#            0.046% and 2/561; the tolerance was set after seeing that value.) Fail -> stop, no A/B.
#   SMOKE  both rows exit 0, Awake at start and end, "marker ok" in the stack row, no FATAL; text differences are reported
#          (divergence point), not gating, since a reordering can flip a near-tie.
#   A/B    ABBA x6 (24 rows, -n 256). Keep rule: exit 0, foreign=[], Awake at start and end (stack_summary.py
#          --any-budget-prereg --require-awake). PRIMARY decode tok/s; MECHANISM compute_ms (expected lower), stall_ms.
#   verdict stack_summary.py rule (SE/|mean| < 0.5 and sign in >= 5 of 6 repeats).
# Engine bmoe-i8mm-0025 (GT_BIN).
set -u
MODE=${1:-smoke}
GT_BIN=${GT_BIN:-bmoe-i8mm-0019gx}; GT_VARIANT=${GT_VARIANT:-1}; GT_SPIN=${GT_SPIN:-0}; GT_CAP=${GT_CAP:-3}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_prerepack_${MODE}_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f --expert-slru --predict-prefetch --spec-adopt-selective"
TIER="--experts-prerepacked --repack-dense"
MR=$H/Qwen3-30B-A3B-Q4_0.repacked.gguf
if [ -e $H/.phone_busy ]; then echo "phone busy: $(cat $H/.phone_busy)" | tee "$O/REFUSED"; exit 3; fi
echo "prerepack $MODE $(date +%H:%M:%S)" > $H/.phone_busy
trap 'rm -f $H/.phone_busy' EXIT
echo "stack_flags=[$TIER] binary=$GT_BIN md5=$(md5sum $H/$GT_BIN/bmoe-cli | cut -d' ' -f1)" >> "$O/log.txt"
batt() { dumpsys battery | grep -m1 ' level:' | tr -dc 0-9; }
PIN=${GT_PIN:-}
awake() {
  settings put system screen_off_timeout 1800000
  dumpsys power | grep -q 'mWakefulness=Awake' || { input keyevent KEYCODE_WAKEUP; sleep 1; }
  if [ -n "$PIN" ] && dumpsys window | grep -q 'isKeyguardShowing=true'; then
    input swipe 540 1900 540 700 200; sleep 1; input text "$PIN"; input keyevent 66; sleep 2
  fi
}
trap 'rm -f $H/.phone_busy; input keyevent KEYCODE_SLEEP' EXIT
run() {  # arm extra tag n
  tag=$1_rep$3
  b=$(batt); if [ "${b:-0}" -lt 25 ]; then echo "STOP battery ${b}% before $tag" | tee -a "$O/log.txt"; return 1; fi
  g=$(thermal_wait 30); mr=$(mem_ready 6500 120); awake
  foreign=$(ps -A -o ARGS | grep -E "llama-bench|bmoe-cli|zcbench|gx_|com\.moephone" | grep -v grep | tr " " "_" | tr "\n" "," )
  ws=$(dumpsys power | grep -m1 'mWakefulness=' | cut -d= -f2)
  echo "=== $tag $(date +%H:%M:%S) batt=${b}% wake_start=${ws} $mr foreign=[${foreign}] BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  MM=$M; [ "$1" = stack ] && MM=$MR
  ( cd $H/$GT_BIN && LD_LIBRARY_PATH=. ./bmoe-cli -m $MM $BASE -n $4 $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:|moe-overlap|gpu-tier:|FATAL' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
if [ "$MODE" = fidelity ]; then
  for arm in base stack; do
    awake; MM=$M; X=""; [ $arm = stack ] && { MM=$MR; X="$TIER"; }
    ( cd $H/$GT_BIN && LD_LIBRARY_PATH=. ./bmoe-cli -m $MM $BASE $X --ppl $H/defer_ppl_text.txt > "$O/ppl_$arm.out" 2> "$O/ppl_$arm.err" )
    echo "ppl_$arm exit=$? $(grep -h '^ppl:' "$O/ppl_$arm.out" "$O/ppl_$arm.err")" | tee -a "$O/log.txt"
  done
elif [ "$MODE" = smoke ]; then
  run base "" s1 64; run stack "$TIER" s1 64
else
  for r in $(seq 1 6); do
    if [ $((r % 2)) -eq 1 ]; then run base "" "${r}a" 256; run stack "$TIER" "${r}a" 256; run stack "$TIER" "${r}b" 256; run base "" "${r}b" 256
    else run stack "$TIER" "${r}a" 256; run base "" "${r}a" 256; run base "" "${r}b" 256; run stack "$TIER" "${r}b" 256; fi
  done
fi
first=""
for f in "$O"/*.out; do
  t=$(sed -n '1,/^generation:/p' "$f" | sed '$d')
  if [ -z "$first" ]; then first="$t"; echo "text_reference=$(basename "$f")" >> "$O/log.txt"; fi
  if [ "$t" = "$first" ]; then echo "text_match $(basename "$f") OK" >> "$O/log.txt"; else echo "text_match $(basename "$f") DIFFERS" >> "$O/log.txt"; fi
done
echo "done $(date)" | tee "$O/DONE"
