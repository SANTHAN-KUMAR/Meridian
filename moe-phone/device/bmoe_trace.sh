#!/system/bin/sh
# bmoe_trace.sh — attribute the compute residual with the engine's own tracing (simpleperf cannot run: this
# retail kernel supports no perf events, hardware or software, even with security.perf_harden=0).
#   layers   --compute-trace-layers: one barrier per layer, coalescing and prefetch survive, rows per layer
#   nodes    --compute-trace: every graph node isolated and timed (serialized graph: proportions only)
# Both on the best configuration (pinned t4 cores 4-7, I/O cores 0-3, 5 GB cache). Also samples per-core
# frequency every 2 s for the first 60 s of each run, so the governor's cap behaviour on battery power
# (USB unplugged 23:26) is recorded next to the timings.
set -u
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_trace_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 128 --ubatch 512 --moe-stream --cache-mb auto --cache-ceil-mb 5000 --cache-floor-mb 1024 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f"
run() { # name flags
  g=$(thermal_wait 30)
  echo "=== $1 $(date +%H:%M:%S) BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-arena && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE $2 -p "$P" > "$O/$1.out" 2> "$O/$1.err" ) &
  bp=$!
  i=0
  while [ $i -lt 30 ] && kill -0 $bp 2>/dev/null; do
    echo "$(date +%s) $(for c in 0 4 6 7; do cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq; done | tr '\n' ' ')$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq) $(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq)" >> "$O/$1.freq"
    sleep 2; i=$((i + 1))
  done
  wait $bp
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:' "$O/$1.out" "$O/$1.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
run layers "--compute-trace-layers $O/trace_layers.csv"
run nodes  "--compute-trace $O/trace_nodes.csv"
run plain  ""
echo "done $(date)" | tee "$O/DONE"
