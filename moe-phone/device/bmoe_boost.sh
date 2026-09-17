#!/system/bin/sh
# bmoe_boost.sh — three lossless levers on top of the best configuration so far (bmoe_cache: pinned t4 cores 4-7,
# I/O cores 0-3, 5 GB cache ceiling, 1 GB floor = 6.20 tok/s median):
#   base        the best configuration, fixed performance mode OFF
#   perfmode    same flags, Android Power HAL fixed performance mode ON (cmd power set-fixed-performance-mode-enabled)
#               — does it lift the governor's cpufreq cap (2.2 / 1.65 GHz at thermal status 0)?
#   predpf      --predict-prefetch: the next layer's gate on this layer's input, predicted misses read on idle
#               lanes, predicted residents LRU-protected (reads/eviction order only; outputs unchanged)
#   floor768    --cache-floor-mb 768 with a 6 GB ceiling: a larger cache from the same RAM
# Rotated order over 4 cells x 3 repeats; performance mode is switched immediately before each run and its state,
# the cpufreq caps and the per-core current frequency after 60 s of decode are logged per row.
#   sh bmoe_boost.sh REPS
set -u
REPS=${1:-3}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_boost_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f"
run() { # name perfmode(0/1) flags rep
  tag=$1_rep$4
  if [ "$2" = 1 ]; then cmd power set-fixed-performance-mode-enabled true; else cmd power set-fixed-performance-mode-enabled false; fi
  g=$(thermal_wait 30)
  echo "=== $tag $(date +%H:%M:%S) perfmode=$2 BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-arena && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $3 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" ) &
  bp=$!
  sleep 60
  f=$(for c in 0 4 6 7; do cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq; done | tr '\n' '/')
  wait $bp
  echo "exit=$? MIDRUN_cur_freq_c0/c4/c6/c7=$f AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:|predict' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
cells="base perfmode predpf floor768"
flags() {
  case $1 in
    base)     echo "0|--cache-ceil-mb 5000 --cache-floor-mb 1024" ;;
    perfmode) echo "1|--cache-ceil-mb 5000 --cache-floor-mb 1024" ;;
    predpf)   echo "0|--cache-ceil-mb 5000 --cache-floor-mb 1024 --predict-prefetch" ;;
    floor768) echo "0|--cache-ceil-mb 6000 --cache-floor-mb 768" ;;
  esac
}
for r in $(seq 1 "$REPS"); do
  set -- $cells
  i=0; while [ $i -lt $(( (r - 1) % 4 )) ]; do first=$1; shift; set -- "$@" "$first"; i=$((i + 1)); done
  for c in "$@"; do fl=$(flags $c); run "$c" "${fl%%|*}" "${fl#*|}" "$r"; done
done
cmd power set-fixed-performance-mode-enabled false
echo "done $(date)" | tee "$O/DONE"
