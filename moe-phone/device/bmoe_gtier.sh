#!/system/bin/sh
# bmoe_gtier.sh — the GPU expert tier (patches 0017-0019 + libgx, component C) on Qwen3, phone, against the stack.
#   base   the stack configuration (SLRU + predict-prefetch + selective adoption; I/O lanes cpu0-3; compute cpu4-7)
#   stack  the same + --gpu-tier-mb 1000 --gpu-backend gx --gpu-variant 1 --gpu-tier-prior <qwen3 prior>
#          --gpu-max-per-layer 3 --gpu-promotions-per-token 1
# (arm names reuse gates/stack_summary.py; "stack" means "GPU tier on").
# The tier's 1000 MiB come out of the CPU cache budget (the total stays at the auto-sized budget). Tier size from
# gates/gtier_size.py (1000 MiB = 380 slots covers 32% of selections out of sample; 2.19 device experts/layer at
# cap 3); warm start from gates/gtier_prior.py (llama.cpp's own prompts, not this prompt); promotions limited to 1 per
# token because each costs ~2.5 ms on this phone (map 0.6 + unmap 1.8 ms, gx_m6_224521 pool test).
# PRE-REGISTERED (2026-09-18 ~23:15, before any row):
#   SMOKE (must pass before any A/B row): base and stack at -n 64 produce IDENTICAL text; the stack row reports
#          dispatches > 0, warm start filled > 0, no FATAL. Any failure: stop, no A/B.
#   A/B    ABBA x6 (24 rows, -n 256). Keep rule: exit 0, foreign=[] (the budget differs by design: the stack arm's
#          CPU budget is 1000 MiB smaller; both arms' granted budgets are reported).
#   PRIMARY  compute_ms/token (the device takes expert work off the CPU)
#   GUARD    stall_ms and mgmt_ms (the smaller CPU cache raises CPU-tier misses; promotions cost mgmt)
#   SECONDARY decode tok/s. COUNTERS: device experts/dispatch, risk recomputes, wait ms, device ms.
#   LOSSLESS text identical in EVERY row (both paths are ARM-exact); any DIFFERS voids the run's correctness claim.
#   verdict  gates/stack_summary.py rule with a keep rule of "exit 0 and foreign=[]" (budget differs by design).
# FIX 2026-09-19 00:0x: the first smoke (bmoe_gtier_smoke_20260918_2344) set LD_LIBRARY_PATH=.:/vendor/lib64, and the
# linker then resolved a dependency from /vendor and failed to map android.hardware.power-V6-ndk.so ("CANNOT LINK
# EXECUTABLE", both rows exit=1, no model ran). gx's cl_shim dlopens /vendor/lib64/libOpenCL.so by absolute path, so
# the vendor directory is not needed on the search path; removed.
# REVISION 2026-09-19 00:25 (before any v2 row): the binary directory, gx variant and spin_wait are environment
# parameters, GT_BIN (default bmoe-i8mm-0019gx), GT_VARIANT (default 1) and GT_SPIN (default 0), all logged. Variant 2
# (gx 44a0cbf/7fc8367: SoA-repacked slots, bit-exact) needs bmoe-i8mm-0019v2, which writes slots through
# gx_repack_expert and reads CPU copies through gx_unpack_expert. The smoke and A/B rules above are unchanged. Other
# additions: the shared phone lock ($H/.phone_busy) and a battery guard (no row below 25%, so the phone never dies
# mid-campaign).
# DEVIATION 2026-09-19 00:40 (recorded before any v0 engine row): the latency gate (k=3 host <= ~0.6 ms) FAILS again in
# M6 run 3: best is v0, host 1.39 / device 0.72 ms; v2 regressed to 9.7 / 8.5 ms. Nevertheless the A/B is run with
# GT_VARIANT=0. Reason: the gate was a proxy that assumed full-clock CPU time per expert (0.125 ms) and no overlap of the
# dispatch's host time with the CPU's own experts. The engine smoke (v1) measured an average CPU-side wait of only
# 0.24 ms per dispatch (422 ms / 1764). So the proxy cannot settle the question, and the A/B's own pre-registered rule
# above decides it, unchanged. Perfect-device ceiling for this tier (2.19 device experts/layer of 8; expert arithmetic
# ~half of compute): about -14 ms/token.
# CHOICE 2026-09-19 00:45 (before any v0 row): GT_VARIANT=0 with GT_SPIN=1, per M6 run 3 (gx_m6_001107): v0+spin k=3
# 1.06 host / 0.72 device ms, k=1 0.58 / 0.27 ms; spin removes ~0.3 ms of host time per dispatch. Binary bmoe-i8mm-0021
# (engine fix: repacked layouts for every variant >= 2; libgx HEAD refuses unrepacked slots). Smoke gate adds:
# risk-recomputed experts <= 1% of device experts (checked by host/chain_night2.sh).
# CHOICE REVISED 2026-09-19 02:12 (before any tier A/B row): GT_VARIANT=1, GT_SPIN=1, GT_BIN=bmoe-i8mm-0022 (libgx 58b13dd,
# unrolled float4 accumulators, bit-exact). M6 run 5 (gx_m6_020545), k=3 spin, down Q4_0, host / device ms: v0 1.07/0.72,
# v1 0.92/0.70, v3 1.00/0.64. v1 has the lowest host time (what the CPU waits on) and keeps the native layout (no repack on
# promotion, no unpack on overflow). The A/B rule above is unchanged.
#   GT_BIN=... GT_VARIANT=... sh bmoe_gtier.sh MODE   (MODE = smoke | ab)
set -u
MODE=${1:-smoke}
GT_BIN=${GT_BIN:-bmoe-i8mm-0019gx}; GT_VARIANT=${GT_VARIANT:-1}; GT_SPIN=${GT_SPIN:-0}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_gtier_${MODE}_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f --expert-slru --predict-prefetch --spec-adopt-selective"
SPINF=""; [ "$GT_SPIN" = 1 ] && SPINF="--gpu-spin-wait"
TIER="--gpu-tier-mb 1000 --gpu-backend gx --gpu-variant $GT_VARIANT $SPINF --gpu-tier-prior $H/qwen3_gtier_prior.txt --gpu-max-per-layer 3 --gpu-promotions-per-token 1"
if [ -e $H/.phone_busy ]; then echo "phone busy: $(cat $H/.phone_busy)" | tee "$O/REFUSED"; exit 3; fi
echo "gtier $MODE $(date +%H:%M:%S)" > $H/.phone_busy
trap 'rm -f $H/.phone_busy' EXIT
echo "stack_flags=[$TIER] binary=$GT_BIN md5=$(md5sum $H/$GT_BIN/bmoe-cli | cut -d' ' -f1)" >> "$O/log.txt"
batt() { dumpsys battery | grep -m1 ' level:' | tr -dc 0-9; }
run() {  # arm extra tag n
  tag=$1_rep$3
  b=$(batt); if [ "${b:-0}" -lt 25 ]; then echo "STOP battery ${b}% before $tag" | tee -a "$O/log.txt"; return 1; fi
  g=$(thermal_wait 30); mr=$(mem_ready 6500 120)
  foreign=$(ps -A -o ARGS | grep -E "llama-bench|bmoe-cli|zcbench|gx_|com\.moephone" | grep -v grep | tr " " "_" | tr "\n" "," )
  echo "=== $tag $(date +%H:%M:%S) batt=${b}% $mr foreign=[${foreign}] BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/$GT_BIN && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE -n $4 $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
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
