#!/system/bin/sh
# bmoe_h2h.sh — head-to-head in ONE session: BigMoeOnEdge's published engine and command vs our engine, the phone held Awake.
#   base   BigMoeOnEdge reference build (bmoe-ref, its shipped flags armv8.2-a+dotprod+fp16) with its documented command
#          (docs/community-benchmarks.md; as in device/bmoe_campaign.sh, which reproduced 3.98 tok/s = claim
#          bmoe_reproduced_decode): --chatml -n 256 -t 4 --ubatch 512 --moe-stream --cache-mb auto --cache-ceil-mb 4000
#          --io-threads 4 --overlap --dense-weights anon, on the ORIGINAL Qwen3-30B-A3B-Q4_0.gguf
#   stack  our engine (bmoe-i8mm-0025) with our stack. H2H_OURS=repack adds the pre-repacked file + --experts-prerepacked
#          --repack-dense; H2H_OURS=plain does not. Which one runs is fixed by the bmoe_prerepack A/B verdict BEFORE the
#          first h2h row (repack only if that verdict is decisive positive on decode) and logged in log.txt.
# Same model bytes and quantisation (Q4_0) for both, same prompt, same phone, same session, ABBA.
# PRE-REGISTERED (2026-09-19 09:15, before any row):
#   A/B    ABBA x6 (24 rows, -n 256). Keep rule: exit 0, foreign=[], Awake at start and end (stack_summary.py
#          --any-budget-prereg --require-awake).
#   PRIMARY decode tok/s ratio stack/base with the stack_summary.py rule (SE/|mean| < 0.5, sign in >= 5 of 6 repeats).
#   FIDELITY  the text of every row is reported. Both engines run Q4_0 greedy; a divergence is reported, never hidden.
set -u
MODE=${1:-smoke}
GT_BIN=${GT_BIN:-bmoe-i8mm-0019gx}; GT_VARIANT=${GT_VARIANT:-1}; GT_SPIN=${GT_SPIN:-0}; GT_CAP=${GT_CAP:-3}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_h2h_${MODE}_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f --expert-slru --predict-prefetch --spec-adopt-selective"
H2H_OURS=${H2H_OURS:-plain}
TIER=""; MR=$M
[ "$H2H_OURS" = repack ] && { TIER="--experts-prerepacked --repack-dense"; MR=$H/Qwen3-30B-A3B-Q4_0.repacked.gguf; }
REFCMD="--chatml -t 4 --ubatch 512 --moe-stream --cache-mb auto --cache-ceil-mb 4000 --io-threads 4 --overlap --dense-weights anon"
if [ -e $H/.phone_busy ]; then echo "phone busy: $(cat $H/.phone_busy)" | tee "$O/REFUSED"; exit 3; fi
echo "h2h $MODE $(date +%H:%M:%S)" > $H/.phone_busy
trap 'rm -f $H/.phone_busy' EXIT
echo "h2h_ours=$H2H_OURS base=bmoe-ref md5=$(md5sum $H/bmoe-ref/bmoe-cli | cut -d' ' -f1) refcmd=[$REFCMD]" >> "$O/log.txt"
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
  if [ "$1" = base ]; then
    ( cd $H/bmoe-ref && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $REFCMD -n $4 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  else
    ( cd $H/$GT_BIN && LD_LIBRARY_PATH=. ./bmoe-cli -m $MR $BASE -n $4 $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  fi
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:|moe-overlap|gpu-tier:|FATAL' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
if [ "$MODE" = smoke ]; then
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
