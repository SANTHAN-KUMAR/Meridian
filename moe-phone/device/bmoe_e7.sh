#!/system/bin/sh
# bmoe_e7.sh — research spec E7 (added 2026-09-19 at the user's request, after E4's kill): the Adreno GPU as the MAIN compute engine.
# E4 closed the GPU as a HELPER (per-layer dispatch of ~2 experts: the ~0.3 ms fixed round trip alone exceeds the CPU's cost).
# E7 asks the other question: with the whole decode graph resident on the GPU (llama.cpp's OpenCL backend with its Adreno kernels,
# build llama.cpp/build-android-ocl, commit 9f31776), how many ms/token does Qwen3-30B-A3B's arithmetic cost, and does the GPU hold
# that rate under sustained load (resident OLMoE: GPU 47.1 -> 46.1 tok/s while the CPU fell 50.9 -> 23.7, ceiling_ledger capped_floor)?
# Specimens: Qwen3-30B-A3B-Q4_0 truncated to its first L = 4 and 8 blocks (host/e7/truncate_gguf.py; byte-identical layers, plus
# the embedding and output head). Qwen3's full 17 GB cannot be resident.
# PRE-REGISTERED (2026-09-19 ~20:45, before any E7 row):
#   rows     for L in 4, 8: llama-bench -p 0 -n 128 -r 5, GPU (-ngl 99) and CPU (-ngl 0, -t 4), the phone held Awake and unplugged,
#            GPU and CPU rows interleaved. Then a sustained GPU L8 run: llama-bench -n 256 -r 1 repeated for 10 min, logging caps and
#            shell temperature per repeat.
#   fit      t(L) = a + b*L (ms/token) from the two sizes; full-model projection C_gpu48 = a + 48*b, from the median of the last
#            5 minutes of the sustained run if it differs from the -r 5 rows (throttling), else the -r 5 medians.
#   decision (a GPU-main engine still needs, per layer, the routed ids back on the CPU for the flash reads: 48 x 0.31 ms = 14.9 ms,
#            the gx host-minus-device overhead with spin, gx_m6_020545/082901; plus the measured stall 34.6 + mgmt 22.0 ms):
#     POSITIVE  C_gpu48 + 14.9 + 56.6 <= 100, i.e. C_gpu48 <= 28.5 ms: a GPU-main engine is a credible 10 tok/s architecture
#               (a multi-day build, decided by the user).
#     KILL      C_gpu48 > 50.5 ms: 10 tok/s is out of reach even if cache management vanished entirely (C + 14.9 + 34.6 > 100).
#     MIDDLE    28.5 < C_gpu48 <= 50.5: reachable only if a GPU-main design ALSO removes most of the 22 ms cache management,
#               which no measurement supports; reported as "not supported", and the research closes.
#   The CPU rows cross-check the projection against E1's generic arm (same arithmetic on the CPU).
# Guards: shared lock, battery >= 25%, the phone held Awake (PIN from GT_PIN).
#   GT_PIN=... sh bmoe_e7.sh
set -u
H=/data/local/tmp/moe-stream
E=/data/local/tmp/e7
. $H/thermal_gate.sh
O=$H/bmoe_e7_$(date +%Y%m%d_%H%M); mkdir -p "$O"
if [ -e $H/.phone_busy ]; then echo "phone busy: $(cat $H/.phone_busy)" | tee "$O/REFUSED"; exit 3; fi
echo "e7 $(date +%H:%M:%S)" > $H/.phone_busy
PIN=${GT_PIN:-}
awake() {
  settings put system screen_off_timeout 1800000
  dumpsys power | grep -q 'mWakefulness=Awake' || { input keyevent KEYCODE_WAKEUP; sleep 1; }
  if [ -n "$PIN" ] && dumpsys window | grep -q 'isKeyguardShowing=true'; then
    input swipe 540 1900 540 700 200; sleep 1; input text "$PIN"; input keyevent 66; sleep 2
  fi
}
trap 'rm -f $H/.phone_busy; input keyevent KEYCODE_SLEEP' EXIT
batt() { dumpsys battery | grep -m1 ' level:' | tr -dc 0-9; }
sf() { for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type 2>/dev/null)" = shell_front ] && cat $z/temp; done | head -1; }
state() { echo "batt=$(batt)% $(dumpsys battery | grep -E 'AC powered|USB powered' | tr -d ' ' | tr '\n' ',') shell_mC=$(sf) caps=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq)/$(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq) wake=$(dumpsys power | grep -m1 mWakefulness= | cut -d= -f2)"; }
echo "llama-bench md5=$(md5sum $E/llama-bench | cut -d' ' -f1) L4=$(ls -l $E/L4.gguf | awk '{print $5}') L8=$(ls -l $E/L8.gguf | awk '{print $5}')" >> "$O/log.txt"
run() {  # tag model ngl n reps
  b=$(batt); [ "${b:-0}" -lt 25 ] && { echo "STOP battery ${b}%" | tee -a "$O/log.txt"; exit 4; }
  mr=$(mem_ready 6000 120); awake
  echo "=== $1 $(date +%H:%M:%S) $(state) $mr" | tee -a "$O/log.txt"
  ( cd $E && ./llama-bench -m $2 -ngl $3 -t 4 -p 0 -n $4 -r $5 -o csv > "$O/$1.csv" 2> "$O/$1.err" )
  echo "exit=$? AFTER $(state) $(grep -v '^build_commit' "$O/$1.csv" | tail -1 | cut -c1-400)" | tee -a "$O/log.txt"
}
for L in 4 8; do
  run gpu_L$L L$L.gguf 99 128 5
  run cpu_L$L L$L.gguf 0 128 5
done
# sustained GPU L8, 10 minutes
t0=$(cut -d. -f1 /proc/uptime); i=0
while [ $(( $(cut -d. -f1 /proc/uptime) - t0 )) -lt 600 ]; do i=$((i+1)); run sus_L8_$i L8.gguf 99 256 1; done
echo "done $(date)" | tee "$O/DONE"
