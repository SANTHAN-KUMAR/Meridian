#!/system/bin/sh
# bmoe_ovh_mech.sh — WHY is the engine 1.90x slower than plain llama.cpp on the same fully-cached model
# (claims overhead_stream_minus_plain_ms, overhead_ratio)? Same OLMoE, same binary (bmoe-i8mm-0016), every
# row under the system simpleperf (per-process hardware/software counters, no root needed).
#   plain       no --moe-stream: llama.cpp's own decode, weights mmap'd from the file
#   stream      --moe-stream, all experts cached, dense weights copied to anon memory (the measured arm)
#   streammap   as stream but --dense-weights mmap: isolates the dense-weight copy
# PRE-REGISTERED (revised 2026-09-18 ~20:25 before any VALID row): per STEADY decode token, isolated by differencing:
#   counts(-n 160) - counts(-n 32) = 128 decode tokens after warm-up (load, prefill and the cache fill cancel).
#   instructions/token  stream >> plain  -> the engine EXECUTES more (hook callbacks, graph splits, bookkeeping)
#   dTLB-load-misses    stream >> plain at similar instructions -> memory layout (4 KiB anon slices vs file pages)
#   cache-misses        likewise, locality
#   streammap vs stream -> how much of it is the dense-weight copy alone
# Latin order over 3 repeats (9 rows).
set -u
REPS=${1:-2}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/olmoe-1b-7b-0924-q4_0.gguf
O=$H/bmoe_ovh_mech_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
COMMON="--chatml --ubatch 512 -t 4 --cpu-mask f0"
STREAM="--moe-stream --cache-mb 4500 --force-cache --overlap --io-threads 4 --io-cpu-mask 0f"
# User-space hardware counters only: the shell user may count :u events (perf_event_paranoid), software events
# cannot take :u, and faults per token are already in the engine CSV. The 20:01 attempt used unqualified events
# and every row failed at simpleperf start-up (no data), which is why this list and the design below changed.
EV="instructions:u,cpu-cycles:u,raw-dtlb-walk:u,raw-l2d-cache-refill:u"
cat $M > /dev/null
run() {  # arm n rep
  tag=$1_n$2_rep$3; NTOK=$2
  g=$(thermal_wait 30); mr=$(mem_ready 6000 120)
  foreign=$(ps -A -o ARGS | grep -E "llama-bench|bmoe-cli|zcbench|com\.moephone" | grep -v grep | tr " " "_" | tr "\n" "," )
  echo "=== $tag $(date +%H:%M:%S) $mr foreign=[${foreign}] BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  case $1 in
    plain)     X="" ;;
    stream)    X="$STREAM --dense-weights anon" ;;
    streammap) X="$STREAM --dense-weights mmap" ;;
  esac
  ( cd $H/bmoe-i8mm-0016 && LD_LIBRARY_PATH=. simpleperf stat -e $EV -o "$O/$tag.perf" ./bmoe-cli -m $M $COMMON -n $NTOK $X --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-cache:' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  case $((r % 3)) in
    1) order="plain stream streammap" ;; 2) order="stream streammap plain" ;; 0) order="streammap plain stream" ;;
  esac
  for a in $order; do run $a 32 $r; run $a 160 $r; done
done
echo "done $(date)" | tee "$O/DONE"
