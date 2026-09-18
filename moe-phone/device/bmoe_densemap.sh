#!/system/bin/sh
# bmoe_densemap.sh — the 13.6 ms/token dense-weight copy (claim ovhmech_dense_copy_ms), tested on Qwen3 itself.
#   base   the decisive stack config, --dense-weights anon (the engine default; a private copy of the dense weights)
#   stack  the same with --dense-weights mmap (dense weights read from the page cache)
# (arm names reuse gates/stack_summary.py unchanged; "stack" here means "dense mmap".)
# Why it is not free on Qwen3: the default became anon because mmap'd dense pages can be evicted under the memory
# pressure the expert cache creates, which turns into major faults; majflt per token is in every row's CSV.
# PRE-REGISTERED (2026-09-18 ~21:00, before any row):
#   PRIMARY   compute_ms/token (the term the copy's cost lands in on OLMoE); row sd ~5.5 ms
#   GUARD     stall_ms and majflt/token: faults from evicted dense pages would show here
#   SECONDARY decode tok/s; counters hit/MiB must not move; text identical in every row
#   estimate/verdict exactly as gates/stack_summary.py (paired within ABBA repeats; decisive iff SE/|diff| < 0.5
#             and sign holds in >= n-1 repeats)
#   sh bmoe_densemap.sh REPS
set -u
REPS=${1:-4}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_densemap_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
COMMON="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f --expert-slru --predict-prefetch --spec-adopt-selective"
echo "stack_flags=[--dense-weights mmap vs anon]" >> "$O/log.txt"
run() {
  tag=$1_rep$3
  g=$(thermal_wait 30); mr=$(mem_ready 6500 120)
  foreign=$(ps -A -o ARGS | grep -E "llama-bench|bmoe-cli|zcbench|com\.moephone" | grep -v grep | tr " " "_" | tr "\n" "," )
  echo "=== $tag $(date +%H:%M:%S) $mr foreign=[${foreign}] BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-0016 && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $COMMON $2 --csv "$O/$tag.csv" -p "$P" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:|moe-overlap' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
for r in $(seq 1 "$REPS"); do
  if [ $((r % 2)) -eq 1 ]; then run base "--dense-weights anon" "${r}a"; run stack "--dense-weights mmap" "${r}a"; run stack "--dense-weights mmap" "${r}b"; run base "--dense-weights anon" "${r}b"
  else run stack "--dense-weights mmap" "${r}a"; run base "--dense-weights anon" "${r}a"; run base "--dense-weights anon" "${r}b"; run stack "--dense-weights mmap" "${r}b"; fi
done
first=""
for f in "$O"/*.out; do
  t=$(sed -n '1,/^generation:/p' "$f" | sed '$d')
  if [ -z "$first" ]; then first="$t"; echo "text_reference=$(basename "$f")" >> "$O/log.txt"; fi
  if [ "$t" = "$first" ]; then echo "text_match $(basename "$f") OK" >> "$O/log.txt"; else echo "text_match $(basename "$f") DIFFERS" >> "$O/log.txt"; fi
done
echo "done $(date)" | tee "$O/DONE"
