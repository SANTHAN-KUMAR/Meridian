#!/system/bin/sh
# bmoe_overhead.sh — is the engine's compute gap vs llama.cpp OUR overhead, or Qwen3's shape?
# At identical clock caps, stock llama.cpp processes ~31 GB/s of weights on resident OLMoE (claim
# inapp_olmoe_cpu_tok_s) while our engine's Qwen3 compute term runs at ~19 GB/s. This separates the two causes
# on ONE model, OLMoE-1B-7B q4_0, which fits in RAM, with the SAME libraries (binary bmoe-i8mm-0016 =
# 0001-0015 + a measurement-only OLMoE recipe row):
#   plain   bmoe-cli WITHOUT --moe-stream: llama.cpp's own decode, no routing hook, mmap weights
#   stream  bmoe-cli --moe-stream with a cache larger than all of OLMoE's experts: after warm-up every expert
#           is a cache hit, so any extra time per token is the engine's own overhead (hook, sync, bookkeeping)
#   lb      llama-bench from the OpenCL build (a DIFFERENT build; reference only, never in the primary contrast)
# Same threads/cores for all: -t 4 on cpu4-7 (bmoe --cpu-mask f0; llama-bench under taskset f0).
# PRE-REGISTERED (2026-09-18 ~19:20, before any row):
#   PRIMARY   steady-state ms/token = median per-token wall_ms over decode tokens 33..128 (CSV), stream vs plain,
#             paired within each of 3 ABBA repeats; overhead = mean paired difference, SE, SE/|diff|
#   verdict   DECISIVE iff SE/|diff| < 0.5 and the sign holds in >= 2 of 3 repeats
#   reading   stream >> plain  -> the engine adds compute overhead, which is a fixable lever for Qwen3
#             stream ~= plain  -> the engine is not the gap; Qwen3's 19 vs 31 GB/s comes from its shape
#   lossless  plain and stream must generate identical text
#   sh bmoe_overhead.sh REPS
set -u
REPS=${1:-3}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/olmoe-1b-7b-0924-q4_0.gguf
O=$H/bmoe_overhead_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
COMMON="--chatml -n 128 --ubatch 512 -t 4 --cpu-mask f0"
STREAM="--moe-stream --cache-mb 4500 --force-cache --overlap --dense-weights anon --io-threads 4 --io-cpu-mask 0f"
C0=/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq
C6=/sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq
cat $M > /dev/null
run() {  # arm rep
  tag=$1_rep$2
  g=$(thermal_wait 30); mr=$(mem_ready 6000 120)
  foreign=$(ps -A -o ARGS | grep -E "llama-bench|bmoe-cli|com\.moephone" | grep -v grep | tr " " "_" | tr "\n" "," )
  echo "=== $tag $(date +%H:%M:%S) $mr foreign=[${foreign}] BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  case $1 in
    plain)  ( cd $H/bmoe-i8mm-0016 && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $COMMON --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" ) & ;;
    stream) ( cd $H/bmoe-i8mm-0016 && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $COMMON $STREAM --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" ) & ;;
    lb)     ( cd $H/ocl && LD_LIBRARY_PATH=. taskset f0 ./llama-bench -m $M -p 0 -n 128 -r 3 -t 4 -ngl 0 > "$O/$tag.out" 2> "$O/$tag.err" ) & ;;
  esac
  eng=$!
  : > "$O/$tag.caps"
  while kill -0 $eng 2>/dev/null; do echo "$(cat $C0) $(cat $C6)" >> "$O/$tag.caps"; sleep 2; done
  wait $eng; e=$?
  c0=$(awk '{print $1}' "$O/$tag.caps" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  c6=$(awk '{print $2}' "$O/$tag.caps" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
  echo "exit=$e AFTER $(thermal_state) cap0_median_run=$c0 cap6_median_run=$c6 $(grep -hE 'generation:|moe-stream:|moe-cache:|tg128' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  if [ $((r % 2)) -eq 1 ]; then run plain $r; run stream $r; run stream "${r}b"; run plain "${r}b"
  else run stream $r; run plain $r; run plain "${r}b"; run stream "${r}b"; fi
  run lb $r
done
first=""
for f in "$O"/plain_*.out "$O"/stream_*.out; do
  t=$(sed -n '1,/^generation:/p' "$f" | sed '$d')
  if [ -z "$first" ]; then first="$t"; echo "text_reference=$(basename "$f")" >> "$O/log.txt"; fi
  if [ "$t" = "$first" ]; then echo "text_match $(basename "$f") OK" >> "$O/log.txt"; else echo "text_match $(basename "$f") DIFFERS" >> "$O/log.txt"; fi
done
echo "done $(date)" | tee "$O/DONE"
