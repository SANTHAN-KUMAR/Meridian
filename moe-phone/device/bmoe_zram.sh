#!/system/bin/sh
# bmoe_zram.sh — an expert cache LARGER than the RAM the OS will grant, deliberately, so that the
# overflow lands in ZRAM instead of being re-read from flash.
#
# THE HYPOTHESIS. The cache budget is currently chosen to fit in MemAvailable, which caps the hit rate
# (see the measured hit rate in results/*/bmoe_cache.json, cell ceil5000) and leaves the rest of every
# token's routed bytes to come from UFS. This phone has a ZRAM swap device whose size and free space are
# recorded per row below. A miss served by ZRAM costs a page fault plus a zram read; a miss served by
# flash costs a UFS read at the bandwidth measured in results/*/g1_storage.json. If ZRAM is the faster
# of the two, an oversized cache trades flash reads for swap faults and wins -- even though Q4_0 weights
# are high-entropy and will barely compress, because zram stores an incompressible page raw and the
# transfer is then RAM to RAM.
#
# WHY IT MIGHT LOSE, which is why it is measured and not assumed:
#   - a swap fault is synchronous in the faulting compute thread, whereas a flash miss is read by the
#     I/O lanes and can overlap compute; trading an overlappable cost for a non-overlappable one can be
#     a net loss even at a higher raw bandwidth;
#   - writing a page out costs compression on some CPU, competing with the same cores as decode;
#   - the low-memory killer may kill the process outright, which is a failure row, not a slow row.
# Every row therefore carries MemAvailable, SwapFree, the engine's own cache budget and hit rate, the
# process's exit status, and the delta in /proc/vmstat's pswpin/pswpout so a row that did not actually
# swap cannot be reported as one that did.
#
#   sh bmoe_zram.sh REPS
set -u
REPS=${1:-3}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_zram_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
# Identical to bmoe_cache.sh's BASE except that the budget is fixed rather than auto: --force-cache is
# required because a fixed budget above MemAvailable is exactly what the engine's guard refuses.
BASE="--chatml -n 256 --ubatch 512 --moe-stream --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f --force-cache"
swp() { awk '/^pswpin|^pswpout/{printf "%s=%s ", $1, $2}' /proc/vmstat; }
mem() { awk '/MemAvailable|SwapFree|SwapTotal/{printf "%s=%s ", $1, $2}' /proc/meminfo; }
run() {
  tag=$1_rep$3
  g=$(thermal_wait 30)
  s0=$(swp)
  echo "=== $tag $(date +%H:%M:%S) $(mem) $s0 BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-verify && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  e=$?
  echo "exit=$e AFTER $(thermal_state) $(mem) $(swp) $(grep -hE 'generation:|moe-stream:|moe-cache:' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  # base5000 repeats the best measured cell inside THIS campaign, so the comparison never crosses
  # campaigns (a different day's thermal state and MemAvailable are not a control).
  case $((r % 3)) in
    1) run base5000 "--cache-mb 5000" "$r"; run zram8000 "--cache-mb 8000" "$r"; run zram10000 "--cache-mb 10000" "$r" ;;
    2) run zram8000 "--cache-mb 8000" "$r"; run zram10000 "--cache-mb 10000" "$r"; run base5000 "--cache-mb 5000" "$r" ;;
    0) run zram10000 "--cache-mb 10000" "$r"; run base5000 "--cache-mb 5000" "$r"; run zram8000 "--cache-mb 8000" "$r" ;;
  esac
done
echo "done $(date)" | tee "$O/DONE"
