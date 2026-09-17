#!/system/bin/sh
# bmoe_lanes.sh — I/O lane count on top of the best configuration (pinned t4 cores 4-7 + --recycle-pages):
# the flash stall is ~0.069 s/token at 4 lanes while G1 measured storage scaling to 8 threads (3.2 GB/s at
# 1 MB x 8). Lanes mostly block in the kernel, so 6-8 lanes share cores 0-3.
# Cells differ only in --io-threads (4 / 6 / 8), all on --io-cpu-mask 0f.
# order alternating per repeat, quiesce + thermal/wake state per row.
#   sh bmoe_lanes.sh REPS
set -u
REPS=${1:-3}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_lanes_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-ceil-mb 4000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-cpu-mask 0f --recycle-pages"
run() {
  tag=$1_rep$3
  g=$(thermal_wait 30)
  echo "=== $tag $(date +%H:%M:%S) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo) BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-verify && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:|io_threads|recycle-pages:' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  case $((r % 3)) in
    1) run io4 "--io-threads 4" "$r"; run io6 "--io-threads 6" "$r"; run io8 "--io-threads 8" "$r" ;;
    2) run io6 "--io-threads 6" "$r"; run io8 "--io-threads 8" "$r"; run io4 "--io-threads 4" "$r" ;;
    0) run io8 "--io-threads 8" "$r"; run io4 "--io-threads 4" "$r"; run io6 "--io-threads 6" "$r" ;;
  esac
done
echo "done $(date)" | tee "$O/DONE"
