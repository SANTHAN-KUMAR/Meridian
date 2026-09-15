#!/data/data/com.termux/files/usr/bin/sh
# llm_decode_probe.sh — measured decode rate AND flash traffic of an unmodified
# engine (llama.cpp) on the phone, per ESTIMAND.md §1-2.
#
# ESTIMAND §2 defines B_flash from /proc/<pid>/io read_bytes (block-layer reads;
# page-cache hits excluded). llama-bench prints only tok/s, so this wrapper
# samples the engine's /proc/<pid>/io and /proc/<pid>/status every 0.25 s while
# it runs. The decode-only quantities are obtained by DIFFERENCING two runs of
# different length from the same starting state (n_short vs n_long): load and
# first-token costs cancel, leaving marginal seconds and bytes per decoded token
# over tokens n_short+1 .. n_long — the steady-state window of ESTIMAND §1.
#
# Cold start is MEASURED, not assumed: fadvdrop evicts the checkpoint and prints
# its mincore() residency before every run (a run whose post-drop residency is
# above 5% is marked not_cold).
#
# Usage (in Termux, screen on, Termux foregrounded):
#   sh llm_decode_probe.sh MODEL THREADS "N_SHORT N_LONG" REPS OUTDIR [cold|warm]
# Writes OUTDIR/runs.csv (one row per run) and OUTDIR/<tag>.io (raw samples).
set -u
MODEL=$1; THREADS=$2; NS=$3; REPS=$4; OUT=$5; MODE=${6:-cold}
HERE=$(cd "$(dirname "$0")" && pwd)
BENCH=$HERE/llm/llama-bench
DROP=$HERE/fadvdrop
mkdir -p "$OUT"
CSV=$OUT/runs.csv
[ -f "$CSV" ] || echo "tag,model,threads,n_gen,rep,mode,resident_after_drop,t_wall_s,read_bytes,rchar,max_vmrss_kb,max_rssfile_kb,tg_tok_s,memavail_kb_start,batt_level,batt_temp_dC,exit" > "$CSV"
batt() { dumpsys battery 2>/dev/null | awk '/^  level:/{l=$2} /^  temperature:/{t=$2} END{printf "%s,%s", l, t}'; }
for rep in $(seq 1 "$REPS"); do
  for n in $NS; do
    tag=$(basename "$MODEL" .gguf)_t${THREADS}_n${n}_r${rep}_${MODE}
    res=NA
    if [ "$MODE" = cold ]; then
      res=$("$DROP" "$MODEL" | sed -n 's/.*resident_after=\([0-9.]*\).*/\1/p')
    fi
    mem=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    b=$(batt)
    t0=$(date +%s.%N)
    "$BENCH" -m "$MODEL" -p 0 -n "$n" -t "$THREADS" -r 1 --no-warmup -o csv > "$OUT/$tag.csv" 2> "$OUT/$tag.err" &
    pid=$!
    : > "$OUT/$tag.io"
    rb=0; rc=0; mr=0; mf=0
    while kill -0 $pid 2>/dev/null; do
      if [ -r /proc/$pid/io ]; then
        s=$(awk '/^read_bytes/{r=$2} /^rchar/{c=$2} END{print r, c}' /proc/$pid/io 2>/dev/null)
        v=$(awk '/^VmRSS/{r=$2} /^RssFile/{f=$2} END{print r+0, f+0}' /proc/$pid/status 2>/dev/null)
        [ -n "$s" ] && echo "$(date +%s.%N) $s $v" >> "$OUT/$tag.io"
      fi
      sleep 0.25
    done
    wait $pid; ex=$?
    t1=$(date +%s.%N)
    last=$(tail -1 "$OUT/$tag.io")
    rb=$(echo "$last" | awk '{print $2}'); rc=$(echo "$last" | awk '{print $3}')
    mr=$(awk 'BEGIN{m=0} {if ($4>m) m=$4} END{print m}' "$OUT/$tag.io")
    mf=$(awk 'BEGIN{m=0} {if ($5>m) m=$5} END{print m}' "$OUT/$tag.io")
    # llama-bench csv: avg_ts is the tok/s column
    ts=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="\"avg_ts\""||$i=="avg_ts") c=i} NR==2{gsub(/"/,"",$c); print $c}' "$OUT/$tag.csv")
    echo "$tag,$(basename "$MODEL"),$THREADS,$n,$rep,$MODE,$res,$(echo "$t1 - $t0" | bc -l 2>/dev/null || awk "BEGIN{print $t1-$t0}"),$rb,$rc,$mr,$mf,$ts,$mem,$b,$ex" >> "$CSV"
    tail -1 "$CSV"
  done
done
