#!/system/bin/sh
# bmoe_fullclock.sh — what would COOLING buy? Measures the engine's per-token terms at FULL clock directly,
# instead of extrapolating: below ~31 C shell temperature the phone runs the prime cores uncapped (claim
# cap_full_below_31C: 23 of 23 samples), so a run started on a cold phone decodes its first tokens at full
# clock, and later tokens at whatever cap the heat brings. Each run samples both caps every 1 s against
# /proc/uptime; gates/fullclock_analyze.py maps every decode token to the cap it ran at (from the CSV's
# per-token wall_ms and the engine's load/prefill times) and reports compute, stall and mgmt per cap level.
# Config = the decisive stack (SLRU + predict-prefetch + selective adoption, patch-0015 engine), because that
# is what would be deployed.
# PRE-REGISTERED (2026-09-18 ~18:40, before any row):
#   outcome   per-token compute_ms, stall_ms, mgmt_ms at FULL clock (cap6 == hw6 and cap0 == hw0) vs capped,
#             tokens >= 16 only (cache warm-up); the full-clock decode rate is 1000 / median(wall_ms) of
#             full-clock tokens -- MEASURED, not projected
#   validity  needs >= 20 full-clock decode tokens pooled; fewer -> "not measurable this way", no number
#   runs      3, each started only when shell temperature <= 30.5 C (waits up to 40 min per run, logged)
#   sh bmoe_fullclock.sh RUNS
set -u
RUNS=${1:-3}
H=/data/local/tmp/moe-stream
. $H/thermal_gate.sh
M=$H/Qwen3-30B-A3B-Q4_0.gguf
O=$H/bmoe_fullclock_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
BASE="--chatml -n 128 --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f --expert-slru --predict-prefetch --spec-adopt-selective"
# the shell_front thermal zone, found once: reading one sysfs file per sample keeps the sampler cheap
# (thermal_state calls dumpsys, far too heavy to run every second during a measurement)
SZ=$(for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type 2>/dev/null)" = shell_front ] && echo $z/temp; done | head -1)
shell_mc() { cat "$SZ" 2>/dev/null; }
C0=/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq
C6=/sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq
for r in $(seq 1 "$RUNS"); do
  w=0; t=$(shell_mc)
  while [ "${t:-99999}" -gt 30500 ] && [ $w -lt 2400 ]; do sleep 30; w=$((w + 30)); t=$(shell_mc); done
  mr=$(mem_ready 6500 120)
  echo "=== run$r $(date +%H:%M:%S) waited_cool=${w}s shell_mC=$t $mr BEFORE $(thermal_state)" | tr '\n' ' ' | tee -a "$O/log.txt"; echo | tee -a "$O/log.txt"
  echo "launch_uptime $(cut -d' ' -f1 /proc/uptime)" > "$O/run$r.caps"
  ( cd $H/bmoe-i8mm-0015 && LD_LIBRARY_PATH=. ./bmoe-cli -m $M $BASE --csv "$O/run$r.csv" -p "$P" > "$O/run$r.out" 2> "$O/run$r.err" ) &
  eng=$!
  while kill -0 $eng 2>/dev/null; do echo "$(cut -d' ' -f1 /proc/uptime) $(cat $C0) $(cat $C6) $(shell_mc)" >> "$O/run$r.caps"; sleep 1; done
  wait $eng; e=$?
  echo "exit=$e AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:' "$O/run$r.out" "$O/run$r.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
done
echo "hw0=$(cat /sys/devices/system/cpu/cpufreq/policy0/cpuinfo_max_freq) hw6=$(cat /sys/devices/system/cpu/cpufreq/policy6/cpuinfo_max_freq)" >> "$O/log.txt"
echo "done $(date)" | tee "$O/DONE"
