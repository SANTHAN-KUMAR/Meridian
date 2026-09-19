#!/system/bin/sh
# bmoe_sustain.sh — the paper plan's per-device sustained test (research/2026-09-19_PAPER_PLAN.md §4 step 2), for ANY Android device:
# back-to-back 256-token decodes for SUSTAIN_S seconds (default 600), logging per run tok/s, the engine's compute / cache-mgmt / stall
# terms, flash MiB/token, hit rate, every cpufreq policy's cap, shell temperature, battery and power state. The steady-state number is
# the median tok/s of the runs in the second half. The device-specific engine binary and thread placement come from the environment;
# the placement follows the rule measured on the 15R (compute on the fastest cores, I/O lanes on the others).
#   BIN=bmoe-v82 CMASK=c0 NT=2 IOMASK=0f SUSTAIN_S=600 sh bmoe_sustain.sh
set -u
H=/data/local/tmp/moe-stream
BIN=${BIN:-bmoe-v82}; CMASK=${CMASK:-f0}; NT=${NT:-4}; IOMASK=${IOMASK:-0f}; SUSTAIN_S=${SUSTAIN_S:-600}
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_sustain_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
CFG="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon -t $NT --cpu-mask $CMASK --io-threads 4 --io-cpu-mask $IOMASK --expert-slru --predict-prefetch --spec-adopt-selective"
[ -e $H/.phone_busy ] && { echo "phone busy" | tee "$O/REFUSED"; exit 3; }
echo "sustain $(date +%H:%M:%S)" > $H/.phone_busy
trap 'rm -f $H/.phone_busy' EXIT
sf() { for z in /sys/class/thermal/thermal_zone*; do t=$(cat $z/type 2>/dev/null); case $t in shell_front|shell_frame|skin*|xo-therm*) cat $z/temp; return;; esac; done; }
caps() { for p in /sys/devices/system/cpu/cpufreq/policy*; do printf '%s=%s/%s ' $(basename $p) $(cat $p/scaling_max_freq) $(cat $p/cpuinfo_max_freq); done; }
state() { echo "batt=$(dumpsys battery | grep -m1 ' level' | tr -dc 0-9)% $(dumpsys battery | grep -E 'AC powered|USB powered' | tr -d ' ' | tr '\n' ',') skin_mC=$(sf) wake=$(dumpsys power | grep -m1 mWakefulness= | cut -d= -f2) $(caps)"; }
echo "device=$(getprop ro.product.model) soc=$(getprop ro.soc.model) binary=$BIN md5=$(md5sum $H/$BIN/bmoe-cli | cut -d' ' -f1) cfg=[$CFG]" >> "$O/log.txt"
t0=$(cut -d. -f1 /proc/uptime); i=0
while [ $(( $(cut -d. -f1 /proc/uptime) - t0 )) -lt $SUSTAIN_S ]; do
  i=$((i+1))
  echo "=== run$i t=$(( $(cut -d. -f1 /proc/uptime) - t0 ))s $(state)" | tee -a "$O/log.txt"
  ( cd $H/$BIN && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $CFG --csv "$O/run$i.csv" -p "$P" > "$O/run$i.out" 2> "$O/run$i.err" )
  echo "exit=$? AFTER $(state) $(grep -hE 'generation:|moe-stream:|moe-cache:|moe-overlap' "$O/run$i.out" "$O/run$i.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
done
echo "done $(date)" | tee "$O/DONE"
