#!/bin/bash
# bmoe_attn_gpu.sh — attention on the Adreno GPU vs the CPU, OUR engine inside the app (com.moephone.bmoe3),
# ABBA-balanced. Why now:
#   * results/2026-09-18/matmul_sweep.json (claims msweep_gpu_attn_q_speedup etc.): with weights resident the
#     GPU computes the attention projections 1.38-2.80x faster than the CPU, and attention weights ARE
#     resident (uploaded once at load), so the upload penalty that kills the GPU for experts does not apply.
#   * the only previous GPU-attention row (bmoe_attn_ab, 01:52) crashed at load; the cause was fixed in
#     patch 0009 but the fixed build was never measured. The HTP arm lost 17%, all of it in compute, which
#     was attributed to 192 backend crossings/token -- the GPU pays the same crossings, so a loss is possible
#     and this is what decides it.
# PRE-REGISTERED (2026-09-18 ~15:00, before any row):
#   outcome   decode tok/s per row (bmoe-cli "generation:" line)
#   keep row  iff it produced a generation line AND the engine's granted budget equals the cpu arm's budget
#             of the same repeat (a different budget is a different cache, not an attention effect)
#   estimate  median(gpu)/median(cpu) - 1 over kept rows, every row listed
#   verdict   "GPU attention helps" only if gpu beats its same-repeat cpu pair-mates in every repeat
# A 16-token GPU smoke run goes first: if it produces no generation line, the campaign stops (FAILED_smoke)
# instead of spending the driver's full wait on each dead row.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
PKG=com.moephone.bmoe3
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/$(date +%F)/bmoe_attn_gpu"
M=/data/data/$PKG/files/qwen3.gguf
REPS=${1:-3}
mkdir -p "$R"
exec 9>"/tmp/claude-1000/bmoe_attn_gpu.lock"
flock -n 9 || { echo "another bmoe_attn_gpu driver holds the lock" >&2; exit 3; }
log() { echo "$(date -Iseconds) $*" >> "$R/driver.log"; }
A() { adb shell "$@" </dev/null; }
PROMPT="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
COMMON="-m~~$M~~-p~~$PROMPT~~--chatml~~--ubatch~~512~~--moe-stream~~--cache-mb~~auto~~--cache-floor-mb~~1024~~--cache-ceil-mb~~5000~~--overlap~~--dense-weights~~anon~~-t~~4~~--cpu-mask~~f0~~--io-threads~~4~~--io-cpu-mask~~0f"

run_cell() {  # name  n_tokens  extra  max_polls(15 s each)
  local name=$1 n=$2 extra=$3 maxp=$4 i=0
  A "run-as $PKG sh -c 'rm -f files/out.txt files/bench.txt files/run.csv'" >/dev/null 2>&1
  A "for p in \$(pm list packages -3 | sed 's/^package://'); do [ \"\$p\" = $PKG ] || am force-stop \$p; done; am force-stop $PKG; am kill-all" >/dev/null 2>&1
  A "input keyevent KEYCODE_WAKEUP" >/dev/null 2>&1
  sleep 30
  local foreign mem
  foreign=$(A 'ps -A -o ARGS' | grep -E 'llama-bench|bmoe-cli|sh bmoe_' | grep -v grep | tr ' \n' '_,')
  mem=$(A "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo" | tr -d '\r')
  A '. /data/local/tmp/moe-stream/thermal_gate.sh; thermal_state' > "$R/$name.state_before.txt" 2>/dev/null
  log "start $name memavail=${mem}MiB foreign=[$foreign]"
  A "am start -n $PKG/com.moephone.npu.Run --es bench 'bmoe~~$COMMON~~-n~~$n~~--csv~~/data/data/$PKG/files/run.csv$extra'" >/dev/null 2>&1
  while [ $i -lt "$maxp" ]; do
    sleep 15
    A "run-as $PKG sh -c 'grep -c EXIT= files/out.txt 2>/dev/null'" 2>/dev/null | tr -d '\r' | grep -qv '^0$' && break
    i=$((i + 1))
  done
  A "run-as $PKG sh -c 'cat files/bench.txt'" > "$R/$name.txt" 2>/dev/null
  A "run-as $PKG sh -c 'cat files/out.txt'"   > "$R/$name.runner.txt" 2>/dev/null
  A "run-as $PKG sh -c 'cat files/run.csv'"   > "$R/$name.csv" 2>/dev/null
  A '. /data/local/tmp/moe-stream/thermal_gate.sh; thermal_state' > "$R/$name.state_after.txt" 2>/dev/null
  log "done $name polls=$i :: $(grep -hE 'generation:|moe-cache:' "$R/$name.txt" | tr '\n' ' ')"
}

log "START reps=$REPS pkg=$PKG"
run_cell smoke_gpu 16 "~~--attn-device~~GPUOpenCL" 40
if ! grep -q 'generation:' "$R/smoke_gpu.txt"; then
  log "FAILED_smoke: GPU attention produced no generation line; see smoke_gpu.runner.txt"
  A "logcat -d -b crash" > "$R/smoke_gpu.logcat_crash.txt" 2>/dev/null
  touch "$R/DONE"; exit 1
fi
G="~~--attn-device~~GPUOpenCL"
for r in $(seq 1 "$REPS"); do
  if [ $((r % 2)) -eq 1 ]; then
    run_cell "cpu_rep${r}a" 256 "" 60; run_cell "gpu_rep${r}a" 256 "$G" 60
    run_cell "gpu_rep${r}b" 256 "$G" 60; run_cell "cpu_rep${r}b" 256 "" 60
  else
    run_cell "gpu_rep${r}a" 256 "$G" 60; run_cell "cpu_rep${r}a" 256 "" 60
    run_cell "cpu_rep${r}b" 256 "" 60; run_cell "gpu_rep${r}b" 256 "$G" 60
  fi
done
A "am force-stop $PKG"
log "DONE"; touch "$R/DONE"
