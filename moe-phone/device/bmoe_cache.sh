#!/system/bin/sh
# bmoe_cache.sh — expert cache size on top of the best configuration (pinned t4 cores 4-7  (no recycling: bmoe_pin2 found it a net loss)):
# every earlier run capped the auto budget at 4000 MiB and left 1536 MiB free; after quiescing ~6 GB is available.
# The only lever that beat reference in bmoe_levers was a larger cache. Cells differ only in --cache-ceil-mb
# (4000 / 5000 / 6000) with --cache-floor-mb 1024; the budget actually granted is in each row (moe-cache: budget).
# order alternating per repeat, quiesce + thermal/wake state per row.
#   sh bmoe_cache.sh REPS
set -u
REPS=${1:-3}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_cache_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f"
run() {
  tag=$1_rep$3
  g=$(thermal_wait 30)
  echo "=== $tag $(date +%H:%M:%S) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo) BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-verify && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:|recycle-pages:' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  case $((r % 3)) in
    1) run ceil4000 "--cache-ceil-mb 4000" "$r"; run ceil5000 "--cache-ceil-mb 5000" "$r"; run ceil6000 "--cache-ceil-mb 6000" "$r" ;;
    2) run ceil5000 "--cache-ceil-mb 5000" "$r"; run ceil6000 "--cache-ceil-mb 6000" "$r"; run ceil4000 "--cache-ceil-mb 4000" "$r" ;;
    0) run ceil6000 "--cache-ceil-mb 6000" "$r"; run ceil4000 "--cache-ceil-mb 4000" "$r"; run ceil5000 "--cache-ceil-mb 5000" "$r" ;;
  esac
done
echo "done $(date)" | tee "$O/DONE"
