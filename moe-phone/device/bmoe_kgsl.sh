#!/system/bin/sh
# bmoe_kgsl.sh — the slot arena in GPU-driver memory (--arena-kgsl, engine bmoe-i8mm-0020kgsl) against the stack.
#   base   the stack (SLRU + predict-prefetch + selective adoption; I/O cpu0-3; compute cpu4-7), ceiling 4000 MiB
#   stack  the same + --arena-kgsl
# Why: R1 (arena2_summary.json): the anonymous slot arena cut cache mgmt 20 -> 2 ms/token but raised compute +14 ms,
# attributed to zram swapping its anonymous pages. kgsl memory is outside the anonymous LRU; pinprobe
# (results/2026-09-19/pinprobe) is the gate: this campaign runs only on its pre-registered verdict BUILD.
# Ceiling 4000 (both arms) instead of 5000: pinned memory cannot be swapped, and the arena overshoots its budget ~10%
# (arena_kgsl_laptop/README.md), so the phone keeps >= ~1.5 GB more headroom than the anonymous arena's 767 MiB minimum.
# PRE-REGISTERED (2026-09-19 ~00:55, before any row):
#   SMOKE  base and stack at -n 64: identical text, "(kgsl)" in the slot-arena line, no FATAL; else stop, no A/B.
#   A/B    ABBA x6 (24 rows, -n 256). Keep rule: exit 0 and foreign=[] (gates/stack_summary.py --any-budget-prereg;
#          both arms have the same ceiling, so granted budgets should match, and they are reported).
#   PRIMARY  decode tok/s (the lever acts on two terms with opposite prior signs, so their sum is the question)
#   MECHANISM compute_ms (expected: no rise, unlike the anonymous arena's +14) and mgmt_ms (expected ~-18)
#   verdict  stack_summary.py rule (SE/|mean| < 0.5 and sign in >= 5 of 6 repeats); text identical in every row.
# Guards: shared phone lock ($H/.phone_busy); no row below 25% battery; mem_ready 6500 before each row.
#   sh bmoe_kgsl.sh MODE   (MODE = smoke | ab)
set -u
MODE=${1:-smoke}
BIN=bmoe-i8mm-0021
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_kgsl_${MODE}_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 4000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f --expert-slru --predict-prefetch --spec-adopt-selective"
ARM="--arena-kgsl"
if [ -e $H/.phone_busy ]; then echo "phone busy: $(cat $H/.phone_busy)" | tee "$O/REFUSED"; exit 3; fi
echo "kgsl $MODE $(date +%H:%M:%S)" > $H/.phone_busy
trap 'rm -f $H/.phone_busy' EXIT
echo "stack_flags=[$ARM] binary=$BIN md5=$(md5sum $H/$BIN/bmoe-cli | cut -d' ' -f1)" >> "$O/log.txt"
batt() { dumpsys battery | grep -m1 ' level:' | tr -dc 0-9; }
run() {  # arm extra tag n
  tag=$1_rep$3
  b=$(batt); if [ "${b:-0}" -lt 25 ]; then echo "STOP battery ${b}% before $tag" | tee -a "$O/log.txt"; return 1; fi
  g=$(thermal_wait 30); mr=$(mem_ready 6500 120)
  foreign=$(ps -A -o ARGS | grep -E "llama-bench|bmoe-cli|zcbench|gx_|pinprobe|com\.moephone" | grep -v grep | tr " " "_" | tr "\n" "," )
  echo "=== $tag $(date +%H:%M:%S) batt=${b}% $mr foreign=[${foreign}] BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/$BIN && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE -n $4 $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) memavail_after=$(grep MemAvailable /proc/meminfo | tr -s ' ' | cut -d' ' -f2) $(grep -hE 'generation:|moe-stream:|moe-cache:|moe-overlap|slot-arena:|FATAL' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
if [ "$MODE" = smoke ]; then
  run base "" s1 64; run stack "$ARM" s1 64
else
  for r in $(seq 1 6); do
    if [ $((r % 2)) -eq 1 ]; then run base "" "${r}a" 256; run stack "$ARM" "${r}a" 256; run stack "$ARM" "${r}b" 256; run base "" "${r}b" 256
    else run stack "$ARM" "${r}a" 256; run base "" "${r}a" 256; run base "" "${r}b" 256; run stack "$ARM" "${r}b" 256; fi
  done
fi
first=""
for f in "$O"/*.out; do
  t=$(sed -n '1,/^generation:/p' "$f" | sed '$d')
  if [ -z "$first" ]; then first="$t"; echo "text_reference=$(basename "$f")" >> "$O/log.txt"; fi
  if [ "$t" = "$first" ]; then echo "text_match $(basename "$f") OK" >> "$O/log.txt"; else echo "text_match $(basename "$f") DIFFERS" >> "$O/log.txt"; fi
done
echo "done $(date)" | tee "$O/DONE"
