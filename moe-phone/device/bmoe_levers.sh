#!/system/bin/sh
# bmoe_levers.sh — one lever per cell on top of BigMoeOnEdge's reproduced ~4 tok/s Qwen3-30B-A3B run.
#
# Reference = BigMoeOnEdge's documented phone command (reproduced 3.87 / 4.09 tok/s on this phone).
# Each other cell changes ONE thing:
#   ngram3 / ngram5  speculative decoding: n-gram drafts from prompt + generated text, verified in
#                    one batch (--ngram --draft N); decode time only, drafting cost reported separately
#   t5 / t6          more compute threads (compute is ~58% of decode time)
#   io6              more parallel expert-read lanes
#   cache_big        leave 1 GB free instead of 1.5 GB -> a larger expert cache (fewer re-reads)
#   twowave          --io-two-wave: start a layer's reads before committing the rest
# Reference build (armv8.2 + dotprod + fp16), unpinned, interleaved by repeat, same prompt as the
# published protocol. Model file Qwen3-30B-A3B Q4_0.
#   sh bmoe_levers.sh REPS
set -u
REPS=${1:-2}
cd /data/local/tmp/moe-stream/bmoe-ref || exit 1
M=/data/local/tmp/moe-stream/Qwen3-30B-A3B-Q4_0.gguf
O=/data/local/tmp/moe-stream/bmoe_levers_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-ceil-mb 4000 --overlap --dense-weights anon"
run() { # name extra-flags rep
  tag=$1_rep$3
  echo "=== $tag $(date +%H:%M:%S) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)" | tee -a "$O/log.txt"
  LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err"
  rc=$?
  echo "exit=$rc $(grep -hE 'generation:|drafts accepted|drafting costs|moe-stream:|moe-cache:' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  run reference "-t 4 --io-threads 4" "$r"
  run ngram3    "-t 4 --io-threads 4 --ngram --draft 3" "$r"
  run ngram5    "-t 4 --io-threads 4 --ngram --draft 5" "$r"
  run t5        "-t 5 --io-threads 4" "$r"
  run t6        "-t 6 --io-threads 4" "$r"
  run io6       "-t 4 --io-threads 6" "$r"
  run cache_big "-t 4 --io-threads 4 --cache-floor-mb 1024" "$r"
  run twowave   "-t 4 --io-threads 4 --io-two-wave" "$r"
done
echo "done $(date)" | tee "$O/DONE"
