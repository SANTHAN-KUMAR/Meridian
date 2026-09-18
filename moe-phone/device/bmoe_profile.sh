#!/system/bin/sh
# bmoe_profile.sh — WHERE does the engine's extra 29% of user-space instructions go (claim
# ovhmech_extra_instructions)? Sampling profile (simpleperf record, cpu-cycles:u) of plain llama.cpp decode and of
# the engine with dense weights mmap'd, both on fully-cached OLMoE, -n 160; report by shared object and symbol.
# Descriptive: it names the functions, it does not estimate an effect.
set -u
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/olmoe-1b-7b-0924-q4_0.gguf
O=$H/bmoe_profile_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
COMMON="--chatml -n 160 --ubatch 512 -t 4 --cpu-mask f0"
STREAM="--moe-stream --cache-mb 4500 --force-cache --overlap --io-threads 4 --io-cpu-mask 0f --dense-weights mmap"
cat $M > /dev/null
for arm in plain streammap; do
  X=""; [ $arm = streammap ] && X="$STREAM"
  g=$(thermal_wait 30); mem_ready 6000 120 >/dev/null
  echo "=== $arm $(date +%H:%M:%S) BEFORE $g" | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-0016 && LD_LIBRARY_PATH=. simpleperf record -e cpu-cycles:u -f 2000 -o "$O/$arm.data" ./bmoe-cli -m $M $COMMON $X -p "$P" > "$O/$arm.out" 2> "$O/$arm.err" )
  echo "exit=$? $(grep -h 'generation:' "$O/$arm.out" "$O/$arm.err")" | tee -a "$O/log.txt"
  ( cd $H/bmoe-i8mm-0016 && simpleperf report -i "$O/$arm.data" --sort dso,symbol --percent-limit 0.3 > "$O/$arm.report.txt" 2>&1 )
done
echo "done $(date)" | tee "$O/DONE"
