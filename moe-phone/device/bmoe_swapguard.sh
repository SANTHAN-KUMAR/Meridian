#!/system/bin/sh
# bmoe_swapguard.sh — the swap guard (patch 0018) on Qwen3 on the phone, against the stack configuration.
#   base   the stack (SLRU + predict-prefetch + selective adoption, I/O lanes on cpu0-3), --swap-guard 0
#   stack  the same with --swap-guard 1 (sampled check: first and last page of each hit's slices)
# DEVIATION, recorded before any row completed (2026-09-18 22:54): the first launch used --swap-guard 2. Its smoke run
# on the phone measured 72.1 ms for 2347 checks = ~31 us per check (mincore over every page of three slices), i.e.
# ~12 ms per decode token at 384 checks/token -- more than the ~22 faults/token it could remove are worth. Running the
# pre-registered mode-2 arm would test a configuration already known to lose, so it was stopped (0 rows done) and the
# arm changed to mode 1 (two pages per slice). A rate-limited full check is engineering follow-up (design §11).
# (arm names reuse gates/stack_summary.py unchanged; "stack" means "guard on".)
# Why: claims r1_base_majflt -- the stack configuration runs with ~370 MiB of the engine in zram and ~22 major faults
# per decode token, each paid synchronously on a compute thread. On the laptop under forced swap the guard removed
# 96.6% of faults and 36.9% of compute, traded for flash re-reads (claim swapguard_faults); the phone's much
# milder regime decides whether that trade pays here.
# Binary: bmoe-i8mm-0019 (0001-0019; the GPU tier is OFF in both arms).
# PRE-REGISTERED (2026-09-19, before any row):
#   MANIPULATION  guard arm's "swapped-hits" > 0 AND major faults/token (CSV majflt, tokens >= 33) lower than base
#                 in every repeat; if not, the guard did not engage and no timing verdict is drawn
#   PRIMARY       compute_ms/token (faults are paid there); expected a few ms; row sd ~5.5 ms, 12 rows/arm
#   GUARD         stall_ms (re-reads land there) and stall+mgmt
#   SECONDARY     decode tok/s; counters: text identical in every row
#   verdict       gates/stack_summary.py rule: decisive iff SE/|diff| < 0.5 and the sign holds in >= 5 of 6 repeats
#   sh bmoe_swapguard.sh REPS
set -u
REPS=${1:-6}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_swapguard_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f --expert-slru --predict-prefetch --spec-adopt-selective"
echo "stack_flags=[--swap-guard 1 vs 0] binary=bmoe-i8mm-0019" >> "$O/log.txt"
run() {
  tag=$1_rep$3
  g=$(thermal_wait 30); mr=$(mem_ready 6500 120)
  foreign=$(ps -A -o ARGS | grep -E "llama-bench|bmoe-cli|zcbench|gx_|com\.moephone" | grep -v grep | tr " " "_" | tr "\n" "," )
  echo "=== $tag $(date +%H:%M:%S) $mr foreign=[${foreign}] BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-0019 && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:|moe-overlap|swap-guard:' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  if [ $((r % 2)) -eq 1 ]; then run base "--swap-guard 0" "${r}a"; run stack "--swap-guard 1" "${r}a"; run stack "--swap-guard 1" "${r}b"; run base "--swap-guard 0" "${r}b"
  else run stack "--swap-guard 1" "${r}a"; run base "--swap-guard 0" "${r}a"; run base "--swap-guard 0" "${r}b"; run stack "--swap-guard 1" "${r}b"; fi
done
first=""
for f in "$O"/*.out; do
  t=$(sed -n '1,/^generation:/p' "$f" | sed '$d')
  if [ -z "$first" ]; then first="$t"; echo "text_reference=$(basename "$f")" >> "$O/log.txt"; fi
  if [ "$t" = "$first" ]; then echo "text_match $(basename "$f") OK" >> "$O/log.txt"; else echo "text_match $(basename "$f") DIFFERS" >> "$O/log.txt"; fi
done
echo "done $(date)" | tee "$O/DONE"
