#!/system/bin/sh
# bmoe_hitrate.sh — cache budget vs HIT RATE and BYTES READ, which are the two quantities in this
# project that do not depend on the clock the phone happens to be running at.
#
# WHY THIS CAMPAIGN EXISTS AND WHY IT CAN RUN HOT. Every rate comparison is invalid while the SoC is
# throttling (thermal status 3 caps the cores at 1.25/1.40 GHz of 3.80/3.32). But the engine's cache
# counters are not rates: for a fixed prompt, a fixed budget and a fixed policy, the hit rate and the
# MiB read per token are determined by the routing and the cache, and they came out identical to the
# decimal across all twelve rows of the expert-order campaign despite those rows spanning thermal status
# 0 to 3 and decode rates from 5.39 to 6.99 tok/s. So this measures the MECHANISM now, and the speed
# consequence is measured later when the phone is cool.
#
# THE QUESTION. The engine has been capped at --cache-ceil-mb 5000 since a day when MemAvailable was
# about 6 GB. After a reboot and a quiesce the phone reports ~7.6 GB available, and the simulator says
# the miss traffic at these budgets falls steeply (results/2026-09-18/evict_zram_budgets.json):
#   4000 MiB -> 0.256 of the expert bytes, 7000 -> 0.448, 8500 -> 0.544, and LRU's hit rate goes
#   82.1% -> 88.0% -> 95.6% -> 98.1%. If that holds on the device, raising the ceiling removes about
#   two thirds of the flash traffic at 7000 and about five sixths at 8500.
# That is a prediction from a trace-driven simulation; this campaign checks it against the engine.
#
# Runs are -n 32 so each one is short: the hit rate converges long before then (the cache fills during
# the first pass over the layers), and a short run puts far less heat into the phone than the 256-token
# rows the rate campaigns use.
#   sh bmoe_hitrate.sh REPS
set -u
REPS=${1:-2}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_hitrate_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 32 --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f --force-cache"
run() {
  tag=$1_rep$3
  echo "=== $tag $(date +%H:%M:%S) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo) BEFORE $(thermal_state)" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-order2 && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
# A budget larger than the RAM the OS will grant is the ZRAM question; the cap that stopped the phone
# rebooting is computed the same way bmoe_zram.sh computes it and is printed for the record.
MEMTOTAL_MIB=$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo)
CAP_MIB=$((MEMTOTAL_MIB - 1024 - 1536))
echo "budgets capped at ${CAP_MIB} MiB (MemTotal ${MEMTOTAL_MIB} MiB)" | tee -a "$O/log.txt"
for r in $(seq 1 "$REPS"); do
  for mib in 5000 7000 8500; do
    if [ "$mib" -gt "$CAP_MIB" ]; then
      echo "=== c${mib}_rep${r} SKIPPED: above the cap" | tee -a "$O/log.txt"; continue
    fi
    run "c$mib" "--cache-mb $mib" "$r"
  done
done
echo "done $(date)" | tee "$O/DONE"
