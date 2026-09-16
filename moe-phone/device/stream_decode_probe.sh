#!/data/data/com.termux/files/usr/bin/sh
# stream_decode_probe.sh — decode rate of an expert-STREAMING engine on the phone.
#
# llama.cpp PR #25294's streaming flags (--moe-stream-cache, --moe-stream-direct,
# --moe-stream-io-threads) exist only in the common-args tools, not in llama-bench, so this drives
# llama-cli non-interactively: greedy (--temp 0), fixed seed, --ignore-eos so every run decodes
# exactly NGEN tokens, and llama.cpp's own perf lines are parsed ("eval time ... tokens per
# second" for decode, "prompt eval time" for prefill). The prompt is BigMoeOnEdge's published
# benchmark prompt (docs/community-benchmarks.md), so the two engines are measured on the same text.
#
# The binary is llama-COMPLETION, not llama-cli, and stdin is /dev/null (fixed 2026-09-16): this
# branch's llama-cli ignored -no-cnv, fell into interactive chat with no input, and wrote output
# until the laptop's /tmp filled. llama-completion is the non-interactive tool, and a closed stdin
# makes any interactive fallback exit instead of spin.
# /proc/<pid>/io, status and stat are sampled every 0.5 s: flash bytes, peak RSS, CPU seconds and
# MAJOR faults, as in llm_decode_probe.sh.
#
# Usage (Termux, screen on, Termux foregrounded):
#   sh stream_decode_probe.sh MODEL THREADS NGEN REPS OUTDIR "ENGINE FLAGS" [BIN]
set -u
MODEL=$1; T=$2; N=$3; REPS=$4; OUT=$5; FLAGS=${6:-}
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${7:-$HERE/llm-stream/llama-completion}
PROMPT="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
TCK=$(getconf CLK_TCK 2>/dev/null || echo 100)
mkdir -p "$OUT"; CSV=$OUT/runs.csv
[ -f "$CSV" ] || echo "tag,model,threads,n_gen,rep,flags,t_wall_s,decode_tok_s,prefill_tok_s,read_bytes,max_vmrss_kb,cpu_s,majflt,memavail_kb_start,memavail_kb_min,exit" > "$CSV"
for rep in $(seq 1 "$REPS"); do
  ftag=$(echo "$FLAGS" | tr -c 'A-Za-z0-9' '_' | sed 's/_\{1,\}/_/g; s/^_//; s/_$//')
  tag=$(basename "$MODEL" .gguf)_t${T}_n${N}_r${rep}${ftag:+_$ftag}
  mem=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  t0=$(date +%s.%N)
  "$BIN" -m "$MODEL" -t "$T" -n "$N" --temp 0 --seed 1 --ignore-eos -no-cnv --simple-io \
         --no-display-prompt --perf -p "$PROMPT" $FLAGS < /dev/null > "$OUT/$tag.out" 2> "$OUT/$tag.err" &
  pid=$!
  : > "$OUT/$tag.io"
  while kill -0 $pid 2>/dev/null; do
    if [ -r /proc/$pid/io ]; then
      rb=$(awk '/^read_bytes/{print $2}' /proc/$pid/io 2>/dev/null)
      rss=$(awk '/^VmRSS/{print $2}' /proc/$pid/status 2>/dev/null)
      st=$(awk '{print $14+$15, $12}' /proc/$pid/stat 2>/dev/null)
      ma=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
      [ -n "$rb" ] && echo "$(date +%s.%N) $rb ${rss:-0} $st $ma" >> "$OUT/$tag.io"
    fi
    sleep 0.5
  done
  wait $pid; ex=$?
  t1=$(date +%s.%N)
  last=$(tail -1 "$OUT/$tag.io")
  rb=$(echo "$last" | awk '{print $2}')
  cpu=$(echo "$last" | awk -v t="$TCK" '{print $4/t}')
  majf=$(echo "$last" | awk '{print $5}')
  mrss=$(awk 'BEGIN{m=0} {if ($3>m) m=$3} END{print m}' "$OUT/$tag.io")
  mmin=$(awk 'BEGIN{m=-1} {if (m<0 || $6<m) m=$6} END{print m}' "$OUT/$tag.io")
  dec=$(grep "eval time" "$OUT/$tag.err" | grep -v "prompt eval" | tail -1 | sed -n 's/.*, *\([0-9.]*\) tokens per second.*/\1/p')
  pre=$(grep "prompt eval time" "$OUT/$tag.err" | tail -1 | sed -n 's/.*, *\([0-9.]*\) tokens per second.*/\1/p')
  wall=$(awk "BEGIN{print $t1-$t0}")
  echo "$tag,$(basename "$MODEL"),$T,$N,$rep,\"$FLAGS\",$wall,${dec:-NA},${pre:-NA},${rb:-0},$mrss,${cpu:-0},${majf:-0},$mem,$mmin,$ex" >> "$CSV"
  tail -1 "$CSV"
done
