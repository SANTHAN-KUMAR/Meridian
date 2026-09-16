#!/system/bin/sh
# bmoe_campaign.sh — reproduce BigMoeOnEdge's published Qwen3-30B-A3B phone run, then test levers.
#
# BigMoeOnEdge (github.com/Helldez/BigMoeOnEdge, Apache-2.0) measured Qwen3-30B-A3B on this exact
# phone model (OnePlus 15R) at ~4.0 tok/s with top-8 routing (Q4_K_M, 4000 MB cache, 76% hit rate,
# --overlap, --dense-weights anon); its 5.0-5.2 tok/s rows use top-6 routing or another session.
# Cell 1 is its documented command (docs/community-benchmarks.md), with the shipped build flags
# (armv8.2-a+dotprod+fp16). Cells 2-3 each add ONE lever:
#   i8mm    same engine built for this SoC (armv8.6-a+dotprod+i8mm+fp16)
#   pinned  i8mm build under taskset 0xF0: the thread-cliff fix (strict affinity; their CLI has
#           only -t, so affinity is applied from outside)
# Our model file is Qwen3-30B-A3B Q4_0 (not Q4_K_M), stated in every row. Runs from /data/local/tmp
# over adb shell, as BigMoeOnEdge's own protocol does. Interleaved by repeat.
#   sh bmoe_campaign.sh REPS
set -u
REPS=${1:-2}
cd /data/local/tmp/moe-stream || exit 1
M=/data/local/tmp/moe-stream/Qwen3-30B-A3B-Q4_0.gguf
O=/data/local/tmp/moe-stream/bmoe_out_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
run() {  # name build_dir taskset_mask(or -) rep
  name=$1; bd=$2; mask=$3; rep=$4; tag=${name}_rep$rep
  echo "=== $tag $(date +%H:%M:%S) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)" | tee -a "$O/log.txt"
  cmd="./bmoe-cli -m $M --chatml -n 256 -t 4 --ubatch 512 --moe-stream --cache-mb auto --cache-ceil-mb 4000 --io-threads 4 --overlap --dense-weights anon --csv $O/$tag.csv -p"
  if [ "$mask" = - ]; then
    ( cd "$bd" && LD_LIBRARY_PATH=. $cmd "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  else
    ( cd "$bd" && LD_LIBRARY_PATH=. taskset "$mask" $cmd "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  fi
  rc=$?
  # bmoe-cli prints its summary on STDOUT: "generation: N tokens, X s/token (Y tok/s)", "prefill: ...",
  # "moe-stream: read ... MiB/token ..."; the cache hit line may be on either stream
  echo "exit=$rc $(grep -hE 'generation:|prefill:|moe-stream:|moe-cache:' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  run reference_armv82 bmoe-ref - "$r"
  run i8mm bmoe-i8mm - "$r"
  run i8mm_pinned bmoe-i8mm f0 "$r"
done
echo "done $(date)" | tee "$O/DONE"
