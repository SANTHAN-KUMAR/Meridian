#!/system/bin/sh
# bmoe_gptoss20b.sh — the "bigger models" goal: gpt-oss-20b MXFP4 (12.1 GB file, > the 15R's ~11 GB RAM)
# streamed by BigMoeOnEdge on the phone, default top-k (unmodified routing), with the confirmed placement
# (pinned t4 cores 4-7, I/O cores 0-3) vs unpinned, 2 repeats alternating, quiesce + state per row.
#   sh bmoe_gptoss20b.sh REPS
set -u
REPS=${1:-2}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/gpt-oss-20b-MXFP4.gguf
O=$H/bmoe_gptoss20b_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="-n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-ceil-mb 4000 --overlap --dense-weights anon --io-threads 4"
run() {
  tag=$1_rep$3
  g=$(thermal_wait 30)
  echo "=== $tag $(date +%H:%M:%S) BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-defer && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  if [ $((r % 2)) = 1 ]; then run pinned "-t 4 --cpu-mask f0 --io-cpu-mask 0f" "$r"; run unpinned "-t 4" "$r"
  else run unpinned "-t 4" "$r"; run pinned "-t 4 --cpu-mask f0 --io-cpu-mask 0f" "$r"; fi
done
echo "done $(date)" | tee "$O/DONE"
