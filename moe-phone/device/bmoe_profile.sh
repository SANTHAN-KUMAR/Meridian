#!/system/bin/sh
# bmoe_profile.sh — E1 (HEADROOM.md §5): what is the ~90-105 ms/token "compute residual" (wall - stall - mgmt)?
# One decode of 256 tokens on the best configuration (pinned t4 cores 4-7, I/O cores 0-3, 5 GB cache), profiled
# with /system/bin/simpleperf: `stat` counters for the whole process, then `record` with call graphs on the
# 4 compute cores for 40 s of steady decode; `report` by symbol and by thread. Plain (no arena) and
# --slot-arena, so the ~10 ms residual shift seen in bmoe_arena can be attributed.
set -u
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_profile_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-ceil-mb 5000 --cache-floor-mb 1024 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f"
for cell in plain arena; do
  F=""; [ $cell = arena ] && F="--slot-arena"
  g=$(thermal_wait 30)
  echo "=== $cell $(date +%H:%M:%S) BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-arena && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $F -p "$P" > "$O/$cell.out" 2> "$O/$cell.err" ) &
  sleep 25   # model load + prefill; decode is steady after this
  pid=$(pidof bmoe-cli | cut -d' ' -f1)
  simpleperf stat -p $pid --duration 10 -e cpu-cycles,instructions,cache-misses,page-faults,context-switches,cpu-migrations > "$O/$cell.stat.txt" 2>&1
  simpleperf record -p $pid --duration 30 -g -f 2000 -o "$O/$cell.perf.data" > "$O/$cell.record.txt" 2>&1
  wait
  simpleperf report -i "$O/$cell.perf.data" --sort dso,symbol --percent-limit 0.5 > "$O/$cell.report_symbols.txt" 2>&1
  simpleperf report -i "$O/$cell.perf.data" --sort tid,comm --percent-limit 0.5 > "$O/$cell.report_threads.txt" 2>&1
  echo "exit AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:|slot-arena:' "$O/$cell.out" "$O/$cell.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
done
echo "done $(date)" | tee "$O/DONE"
