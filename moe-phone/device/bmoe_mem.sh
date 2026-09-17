#!/system/bin/sh
# bmoe_mem.sh — free RAM for the expert cache without touching what is computed. At 5 GB the cache hits 85.3%
# and the auto budget is capped by MemAvailable minus the floor, so every MiB handed back becomes cache:
#   base5      the current best: 5 GB ceiling, 1 GB floor (6.48 tok/s median, claim cache5000_tok_s)
#   rowstream  --row-stream: the token-embedding table is served from flash (64 MiB window) instead of RAM
#   ctx1024    -c 1024 instead of 2048: KV 192 -> 96 MiB (still larger than these 256-token runs need)
#   combo      both, with a 6 GB ceiling so the freed RAM can actually be taken
# Pinned t4 cores 4-7, I/O cores 0-3, no recycling/arena. 4 cells x 3 repeats, rotated, quiesce + thermal per row.
set -u
REPS=${1:-3}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_mem_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f"
run() {
  tag=$1_rep$3
  g=$(thermal_wait 30)
  echo "=== $tag $(date +%H:%M:%S) BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-arena && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:|row-stream' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
cells="base5 rowstream ctx1024 combo"
flags() {
  case $1 in
    base5)     echo "--cache-ceil-mb 5000" ;;
    rowstream) echo "--cache-ceil-mb 6000 --row-stream" ;;
    ctx1024)   echo "--cache-ceil-mb 6000 -c 1024" ;;
    combo)     echo "--cache-ceil-mb 6000 --row-stream -c 1024" ;;
  esac
}
for r in $(seq 1 "$REPS"); do
  set -- $cells
  i=0; while [ $i -lt $(( (r - 1) % 4 )) ]; do first=$1; shift; set -- "$@" "$first"; i=$((i + 1)); done
  for c in "$@"; do run "$c" "$(flags $c)" "$r"; done
done
echo "done $(date)" | tee "$O/DONE"
