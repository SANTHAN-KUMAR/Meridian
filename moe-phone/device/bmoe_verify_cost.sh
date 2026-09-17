#!/system/bin/sh
# bmoe_verify_cost.sh — what does a speculative VERIFY of N positions cost on the 15R, streamed?
# Teacher-forced --ppl over a fixed text with --ppl-batch N (N tokens per llama_decode, logits at all
# positions): each decode is exactly a verify pass with real routing and the real expert cache. Seconds
# per decode at N vs at 1 is the verify cost multiplier c(N); a drafter at acceptance giving T tokens per
# verify wins only if c(N) < T minus the drafting cost. On the laptop c(4) was 3.5 (results/2026-09-17).
# Interleaved N, thermal gate per run, one binary (patches 0002-0005).
#   sh bmoe_verify_cost.sh REPS
set -u
REPS=${1:-2}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_verify_$(date +%Y%m%d_%H%M); mkdir -p "$O"
BASE="--moe-stream --cache-mb auto --cache-ceil-mb 4000 --overlap --dense-weights anon -t 4 --io-threads 4 --ppl $H/verify_cost_text.txt"
for r in $(seq 1 "$REPS"); do
  for n in 1 2 3 5; do
    tag=b${n}_rep$r
    g=$(thermal_wait 30)
    echo "=== $tag $(date +%H:%M:%S) memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo) BEFORE $g" | tee -a "$O/log.txt"
    ( cd $H/bmoe-i8mm-verify && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE --ppl-batch $n > "$O/$tag.out" 2> "$O/$tag.err" )
    echo "exit=$? AFTER $(thermal_state) $(grep -h '^ppl:' "$O/$tag.out" | tr '\n' ' ')" | tee -a "$O/log.txt"
  done
done
echo "done $(date)" | tee "$O/DONE"
