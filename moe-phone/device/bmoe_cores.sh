#!/system/bin/sh
# bmoe_cores.sh — WHICH CORES should compute? The engine pins 4 compute threads to cpu4-7 (--cpu-mask f0),
# which spans two clock domains: cpu4-5 are policy0, cpu6-7 are policy6. Under the OEM's skin-temperature
# caps, policy6 is capped HARDER than policy0 in every capped sample logged (e.g. 1651200 vs 2265600 kHz,
# 1248000 vs 1401600), and ggml's threads meet at a barrier after every op, so the two slowest cores set the
# pace for all four. Arms (the decisive stack config otherwise, patch-0015 engine):
#   f0    -t 4 --cpu-mask f0 --io-cpu-mask 0f   current: compute cpu4-7 (mixed domains), I/O cpu0-3
#   3c    -t 4 --cpu-mask 3c --io-cpu-mask c3   compute cpu2-5 (all policy0), I/O cpu0,1,6,7
#   3f    -t 6 --cpu-mask 3f --io-cpu-mask c0   compute cpu0-5 (all policy0, 6 threads), I/O cpu6-7
# Latin rotation over 4 repeats (12 rows). Caps sampled every 2 s during each row (medians logged).
# PRE-REGISTERED (2026-09-18 ~18:55, before any row):
#   PRIMARY   compute_ms/token (engine term). Row sd ~5.5 ms vs an expected ~20 ms effect.
#   GUARD     stall_ms/token: I/O threads move cores; a stall rise larger than the compute saving voids a win
#   SECONDARY decode tok/s. COUNTERS hit rate / MiB per token must not change (placement is not policy).
#   estimate  per repeat, arm mean minus f0 mean; mean over 4 repeats, SE, SE/|diff|
#   verdict   DECISIVE iff SE/|diff| < 0.5 and the sign holds in >= 3 of 4 repeats; else "not resolved"
#   lossless  text identical in every row
#   sh bmoe_cores.sh REPS
set -u
REPS=${1:-4}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_cores_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon --io-threads 4 --expert-slru --predict-prefetch --spec-adopt-selective"
args() { case $1 in f0) echo "-t 4 --cpu-mask f0 --io-cpu-mask 0f" ;; 3c) echo "-t 4 --cpu-mask 3c --io-cpu-mask c3" ;; 3f) echo "-t 6 --cpu-mask 3f --io-cpu-mask c0" ;; esac; }
C0=/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq
C6=/sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq
run() {  # arm rep
  tag=$1_rep$2
  g=$(thermal_wait 30); mr=$(mem_ready 6500 120)
  foreign=$(ps -A -o ARGS | grep -E "llama-bench|bmoe-cli|com\.moephone" | grep -v grep | tr " " "_" | tr "\n" "," )
  echo "=== $tag $(date +%H:%M:%S) $mr foreign=[${foreign}] BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-0015 && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $(args $1) --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" ) &
  eng=$!
  : > "$O/$tag.caps"
  while kill -0 $eng 2>/dev/null; do echo "$(cat $C0) $(cat $C6)" >> "$O/$tag.caps"; sleep 2; done
  wait $eng; e=$?
  c0=$(awk '{print $1}' "$O/$tag.caps" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  c6=$(awk '{print $2}' "$O/$tag.caps" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  echo "exit=$e AFTER $(thermal_state) cap0_median_run=$c0 cap6_median_run=$c6 $(grep -hE 'generation:|moe-stream:|moe-cache:|moe-overlap' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  case $((r % 3)) in
    1) run f0 $r; run 3c $r; run 3f $r ;;
    2) run 3c $r; run 3f $r; run f0 $r ;;
    0) run 3f $r; run f0 $r; run 3c $r ;;
  esac
done
first=""
for f in "$O"/*.out; do
  t=$(sed -n '1,/^generation:/p' "$f" | sed '$d')
  if [ -z "$first" ]; then first="$t"; echo "text_reference=$(basename "$f")" >> "$O/log.txt"; fi
  if [ "$t" = "$first" ]; then echo "text_match $(basename "$f") OK" >> "$O/log.txt"; else echo "text_match $(basename "$f") DIFFERS" >> "$O/log.txt"; fi
done
echo "done $(date)" | tee "$O/DONE"
