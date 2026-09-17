#!/system/bin/sh
# bmoe_repack_pin.sh — retest of the repacked-kernel lever ON TOP of the best measured configuration.
# The first repack A/B (bmoe_repack.sh, 2026-09-17 11:50) ran unpinned, before page recycling existed, on a
# throttled CPU, and found no compute gain. Here both cells use pinned t4 (cores 4-7) + I/O on cores 0-3
# + --recycle-pages; the only difference is --repack-experts. Same binary (patches 0002-0006), 3 repeats,
# order alternating per repeat, quiesce + thermal/wake state per row.
#   sh bmoe_repack_pin.sh REPS
set -u
REPS=${1:-3}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_repack_pin_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-ceil-mb 4000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f --recycle-pages"
run() {
  tag=$1_rep$3
  g=$(thermal_wait 30)
  echo "=== $tag $(date +%H:%M:%S) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo) BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-verify && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:|repack-experts:|recycle-pages:' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  if [ $((r % 2)) = 1 ]; then run pin_recycle "" "$r"; run pin_recycle_repack "--repack-experts" "$r"
  else run pin_recycle_repack "--repack-experts" "$r"; run pin_recycle "" "$r"; fi
done
echo "done $(date)" | tee "$O/DONE"
