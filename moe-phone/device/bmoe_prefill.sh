#!/system/bin/sh
# bmoe_prefill.sh — prefill throughput as a function of PROMPT LENGTH, which is the number an agentic
# use of this engine would live or die on and the one number we have never measured properly.
#
# WHY THE NUMBER WE HAVE IS NOT THE NUMBER. Every campaign so far used a ~26-token prompt, so the
# prefill figures in results/*/bmoe_cache.json (5.20-7.50 tok/s) are a COLD-START measurement: the cache
# is empty, the first chunk touches whatever experts those 26 tokens route to, and the per-token cost is
# dominated by the startup. Prefill on a streamed MoE should get CHEAPER per token as the prompt grows,
# because one read of an expert serves every token in the chunk that routes to it -- but it should also
# saturate, because a large enough chunk touches essentially all 128 experts per layer and then pays the
# full 16.4 GB of expert traffic per chunk no matter how many tokens share it.
#
# Where those two effects cross is unmeasured, and it decides whether an agent can afford to put a file
# in the context. At 96 KB of KV per token this also interacts with the cache: a long prompt takes RAM
# away from the expert cache that the same run needs to be fast.
#
# Cells: prompt lengths 128 / 512 / 2048 / 8192 tokens, -n 8 so the run is dominated by prefill, at the
# best known configuration. The prompt is built by repeating a fixed paragraph so the token count is
# roughly proportional to the repetition count and no cell gets a semantically easier prompt.
# Per row: prefill tok/s, decode tok/s, the granted cache budget, the hit rate and the bytes read.
#   sh bmoe_prefill.sh REPS
set -u
REPS=${1:-2}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_prefill_$(date +%Y%m%d_%H%M); mkdir -p "$O"
UNIT="The history of computing begins with mechanical calculation and proceeds through stored program machines to the present day. "
BASE="--chatml -n 8 --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f"

mkprompt() {  # repetitions -> file
  n=$1; f=$2; : > "$f"
  i=0; while [ $i -lt "$n" ]; do printf '%s' "$UNIT" >> "$f"; i=$((i + 1)); done
}

run() {  # name reps_of_unit rep
  tag=$1_rep$3
  mkprompt "$2" "$O/$tag.prompt"
  g=$(thermal_wait 30)
  echo "=== $tag $(date +%H:%M:%S) prompt_bytes=$(wc -c < "$O/$tag.prompt") memavail=$(awk '/MemAvailable/{print $2}' /proc/meminfo) BEFORE $g" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-order2 && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE -c 16384 --csv "$O/$tag.csv" -f "$O/$tag.prompt" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|prefill:|moe-stream:|moe-cache:' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}

for r in $(seq 1 "$REPS"); do
  # ~20 tokens per repetition of UNIT, so these are roughly 128 / 512 / 2048 / 8192 tokens
  case $((r % 2)) in
    1) run p128 6 "$r"; run p512 26 "$r"; run p2048 102 "$r"; run p8192 410 "$r" ;;
    0) run p8192 410 "$r"; run p2048 102 "$r"; run p512 26 "$r"; run p128 6 "$r" ;;
  esac
done
echo "done $(date)" | tee "$O/DONE"
