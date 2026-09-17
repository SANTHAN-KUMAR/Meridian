#!/system/bin/sh
# bmoe_repack.sh — does computing streamed experts on repacked i8mm kernels speed up decode?
#
# BigMoeOnEdge decodes Qwen3-30B-A3B at ~4 tok/s on this phone with compute (0.145 s/token) as its
# largest term, running generic kernels because repacking is off for streamed tensors. The moe-phone
# patch (tools/patches/0002-*) tags expert tensors for ggml's repacked MUL_MAT_ID kernel and repacks
# each expert slice in place as it lands; on the laptop it matched the generic path's perplexity
# (5.4915 vs 5.4831 over 1458 tokens). Here: the SAME i8mm build, repack off vs on, interleaved,
# plus the reference (armv8.2) build as a session control. Same command and prompt as the published
# protocol otherwise.
#   sh bmoe_repack.sh REPS
set -u
REPS=${1:-2}
M=/data/local/tmp/moe-stream/Qwen3-30B-A3B-Q4_0.gguf
O=/data/local/tmp/moe-stream/bmoe_repack_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 256 -t 4 --ubatch 512 --moe-stream --cache-mb auto --cache-ceil-mb 4000 --io-threads 4 --overlap --dense-weights anon"
run() { # name dir extra rep
  tag=$1_rep$4
  echo "=== $tag $(date +%H:%M:%S) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)" | tee -a "$O/log.txt"
  ( cd "/data/local/tmp/moe-stream/$2" && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $3 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  rc=$?
  echo "exit=$rc $(grep -hE 'generation:|moe-stream:|moe-cache:|repack-experts:|repack failed' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  run i8mm_plain     bmoe-i8mm-repack ""                 "$r"
  run i8mm_repack    bmoe-i8mm-repack "--repack-experts" "$r"
  run reference      bmoe-ref         ""                 "$r"
done
echo "done $(date)" | tee "$O/DONE"
