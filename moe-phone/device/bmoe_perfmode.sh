#!/system/bin/sh
# bmoe_perfmode.sh — OnePlus "high performance mode" off vs on, our engine, ABBA-balanced.
#
# WHY. Every Qwen3 row in this project ran with the CPU clocks CAPPED by the OEM power policy, at
# framework thermal status 0: during a run policy0 (cpu0-5) sits at scaling_max_freq 2265600 of 3321600,
# and policy6 (cpu6-7) at 1651200 of 3801600. That is 68% and 43% of the hardware maximum (logged as
# cap0/cap6 in every bmoe_slru and bmoe_arena2 row). The engine's compute threads are pinned to cpu4-7
# (--cpu-mask f0), so the 93 ms/token compute term in decode_budget.json was measured at about half clock.
# `settings get system high_performance_mode_on` is 0.
#
# PRE-REGISTERED (2026-09-18 ~15:20, before any row):
#   manipulation check  the median cap0/cap6 sampled during each row. If the "on" arm's caps do not
#                       rise, the setting did not engage and the campaign says only that -- no rate verdict.
#   outcome             decode tok/s per row; also the compute residual and the stall (the lever should
#                       move compute, not the counters -- hit rate / MiB per token must be identical).
#   verdict             "perf mode helps" only if on beats its same-repeat off pair-mates in every repeat.
#   heat                the on arm may heat the phone for the rows after it; ABBA cancels a monotone drift,
#                       the per-row thermal state is logged, and each row waits for status <= 1 first.
#   sh bmoe_perfmode.sh REPS
set -u
REPS=${1:-3}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_perfmode_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f"
C0=/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq
C6=/sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq
run() {  # arm(off|on) rep
  tag=$1_rep$2
  if [ "$1" = on ]; then settings put system high_performance_mode_on 1; else settings put system high_performance_mode_on 0; fi
  sleep 10
  g=$(thermal_wait 30)
  mr=$(mem_ready 6500 120)
  foreign=$(ps -A -o ARGS | grep -E "llama-bench|bmoe-cli|com\.moephone" | grep -v grep | tr " " "_" | tr "\n" "," )
  echo "=== $tag $(date +%H:%M:%S) setting=$(settings get system high_performance_mode_on) $mr foreign=[${foreign}] BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-order2 && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" ) &
  eng=$!
  : > "$O/$tag.caps"
  while kill -0 $eng 2>/dev/null; do echo "$(cat $C0) $(cat $C6)" >> "$O/$tag.caps"; sleep 2; done
  wait $eng; e=$?
  c0=$(awk '{print $1}' "$O/$tag.caps" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  c6=$(awk '{print $2}' "$O/$tag.caps" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  echo "exit=$e AFTER $(thermal_state) cap0_median_run=$c0 cap6_median_run=$c6 $(grep -hE 'generation:|moe-stream:|moe-cache:' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  if [ $((r % 2)) -eq 1 ]; then run off "${r}a"; run on "${r}a"; run on "${r}b"; run off "${r}b"
  else                          run on "${r}a"; run off "${r}a"; run off "${r}b"; run on "${r}b"; fi
done
settings put system high_performance_mode_on 0
first=""
for f in "$O"/*.out; do
  t=$(sed -n '1,/^generation:/p' "$f" | sed '$d')
  if [ -z "$first" ]; then first="$t"; echo "text_reference=$(basename "$f")" >> "$O/log.txt"; fi
  if [ "$t" = "$first" ]; then echo "text_match $(basename "$f") OK" >> "$O/log.txt"; else echo "text_match $(basename "$f") DIFFERS" >> "$O/log.txt"; fi
done
echo "done $(date)" | tee "$O/DONE"
