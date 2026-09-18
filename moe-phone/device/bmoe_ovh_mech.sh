#!/system/bin/sh
# bmoe_ovh_mech.sh — WHY is the engine 1.90x slower than plain llama.cpp on the same fully-cached model
# (claims overhead_stream_minus_plain_ms, overhead_ratio)? Same OLMoE, same binary (bmoe-i8mm-0016), every
# row under the system simpleperf (per-process hardware/software counters, no root needed).
#   plain       no --moe-stream: llama.cpp's own decode, weights mmap'd from the file
#   stream      --moe-stream, all experts cached, dense weights copied to anon memory (the measured arm)
#   streammap   as stream but --dense-weights mmap: isolates the dense-weight copy
# PRE-REGISTERED (2026-09-18 ~20:05, before any row), per token over the whole run (prefill identical across arms):
#   instructions/token  stream >> plain  -> the engine EXECUTES more (hook callbacks, graph splits, bookkeeping)
#   dTLB-load-misses    stream >> plain at similar instructions -> memory layout (4 KiB anon slices vs file pages)
#   cache-misses        likewise, locality
#   streammap vs stream -> how much of it is the dense-weight copy alone
# Latin order over 3 repeats (9 rows).
set -u
REPS=${1:-3}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/olmoe-1b-7b-0924-q4_0.gguf
O=$H/bmoe_ovh_mech_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
COMMON="--chatml -n 128 --ubatch 512 -t 4 --cpu-mask f0"
STREAM="--moe-stream --cache-mb 4500 --force-cache --overlap --io-threads 4 --io-cpu-mask 0f"
EV="instructions,cpu-cycles,dTLB-loads,dTLB-load-misses,cache-references,cache-misses,page-faults,minor-faults,major-faults,context-switches,task-clock"
cat $M > /dev/null
run() {
  tag=$1_rep$2
  g=$(thermal_wait 30); mr=$(mem_ready 6000 120)
  foreign=$(ps -A -o ARGS | grep -E "llama-bench|bmoe-cli|zcbench|com\.moephone" | grep -v grep | tr " " "_" | tr "\n" "," )
  echo "=== $tag $(date +%H:%M:%S) $mr foreign=[${foreign}] BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  case $1 in
    plain)     X="" ;;
    stream)    X="$STREAM --dense-weights anon" ;;
    streammap) X="$STREAM --dense-weights mmap" ;;
  esac
  ( cd $H/bmoe-i8mm-0016 && LD_LIBRARY_PATH=. simpleperf stat -e $EV -o "$O/$tag.perf" ./bmoe-cli -m $M $COMMON $X --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-cache:' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  case $((r % 3)) in
    1) run plain $r; run stream $r; run streammap $r ;;
    2) run stream $r; run streammap $r; run plain $r ;;
    0) run streammap $r; run plain $r; run stream $r ;;
  esac
done
echo "done $(date)" | tee "$O/DONE"
