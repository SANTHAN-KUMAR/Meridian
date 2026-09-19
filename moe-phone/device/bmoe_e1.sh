#!/system/bin/sh
# bmoe_e1.sh — research spec E1 (research/2026-09-19_RESEARCH_SPEC.md §6 E1; closing rule §0): the engine's compute floor on
# Qwen3-30B-A3B with every flash term removed.
#   --fixed-routing routes every layer to experts 0..7, so after warm-up every expert is a cache hit: stall, reads and cache
#   management read ~0, and decode time is the arithmetic (attention, router, dense and expert matmuls, small ops) alone.
#   The text is meaningless; only the arithmetic is real. Validity check per row: MiB/token ~0 and stall ~0.
# Arms:  G = generic Q4_0 kernels (the shipped file)
#        R = llama.cpp's repacked i8mm kernels: the pre-repacked file (repack_gguf on this phone) + --experts-prerepacked --repack-dense
# PRE-REGISTERED (2026-09-19 ~19:40, before any E1 row):
#   capped   ABBA x3 (12 rows), -n 256, the phone held Awake, rows back to back (the steady capped state IS the condition), caps and
#            shell temperature logged per row. C = decode ms/token (1000 / tok/s), median per arm; the repacked gain is the paired
#            difference per repeat.
#   long     one row per arm with the ~3,000-token E6 KL text as the prompt (-c 4096), -n 64: the compute floor at long context.
#   cold     one row per arm (-n 24) started only when shell_front <= 30.5 C (waits up to 25 min; if not reached the row is still run,
#            and labelled "not cold"): the floor at the hardware clock.
#   counters one row per arm under simpleperf stat -e instructions:u,cycles:u (-n 64): instructions per token.
# DECISION (spec E1 and §0; the I/O terms are the measured awake/unplugged ones, gtier A/B 0447 base arm: stall 34.6 + mgmt 22.0 ms):
#   10 tok/s lossless at the capped clock needs C_R(capped) + 56.6 - S_gpu <= 100 ms, where S_gpu is E4's measured saving
#   (0 if E4 kills). With S_gpu <= 10 ms (E4's bound), that is C_R(capped) <= 53.4 ms at best. C_R > 53.4 ms => 10 tok/s is not
#   reachable losslessly on this phone at its sustained clock, with every lever this project has measured.
#   Also reported: C_G - C_R (the repacked gain in the engine, E1's positive/kill: C_R <= 80 positive, >= 95 kill).
# The phone runs on USB power unless unplugged over wireless adb (logged per row as ac/usb flags). Shared lock; battery >= 25%.
#   GT_PIN=... sh bmoe_e1.sh
set -u
H=/data/local/tmp/moe-stream
BIN=bmoe-i8mm-0027
. $H/thermal_gate.sh
Q4=$H/Qwen3-30B-A3B-Q4_0.gguf
QR=$H/Qwen3-30B-A3B-Q4_0.repacked.gguf
O=$H/bmoe_e1_$(date +%Y%m%d_%H%M); mkdir -p "$O"
P="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
STACK="--chatml --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f --expert-slru --predict-prefetch --spec-adopt-selective --fixed-routing"
if [ -e $H/.phone_busy ]; then echo "phone busy: $(cat $H/.phone_busy)" | tee "$O/REFUSED"; exit 3; fi
echo "e1 $(date +%H:%M:%S)" > $H/.phone_busy
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
power() { dumpsys battery | grep -E 'AC powered|USB powered' | tr -d ' ' | tr '\n' ','; }
sf() { for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type 2>/dev/null)" = shell_front ] && cat $z/temp; done | head -1; }
echo "binary=$BIN md5=$(md5sum $H/$BIN/bmoe-cli | cut -d' ' -f1) q4=$(ls -l $Q4 | awk '{print $5}') qr=$(ls -l $QR | awk '{print $5}') marker=[$(head -3 $QR.repacked | tr '\n' ' ')]" >> "$O/log.txt"
run() {  # tag arm pre extra...
  tag=$1; arm=$2; pre=$3; shift 3
  b=$(batt); if [ "${b:-0}" -lt 25 ]; then echo "STOP battery ${b}% before $tag" | tee -a "$O/log.txt"; exit 4; fi
  mr=$(mem_ready 6500 120); awake
  ws=$(dumpsys power | grep -m1 'mWakefulness=' | cut -d= -f2)
  if [ "$arm" = R ]; then m=$QR; X="--experts-prerepacked --repack-dense"; else m=$Q4; X=""; fi
  echo "=== $tag $(date +%H:%M:%S) batt=${b}% $(power) wake_start=${ws} shell_mC=$(sf) caps=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq)/$(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq) $mr" | tee -a "$O/log.txt"
  ( cd $H/$BIN && LD_LIBRARY_PATH=. $pre ./bmoe-cli -m $m $STACK $X "$@" --csv "$O/$tag.csv" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE 'generation:|moe-stream:|moe-cache:|moe-overlap|repack|prerepacked|Performance counter|instructions|cycles|FATAL' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
# capped ABBA x3
for r in 1 2 3; do
  if [ $((r % 2)) -eq 1 ]; then o="G R R G"; else o="R G G R"; fi
  i=0; for a in $o; do i=$((i+1)); run cap_${a}_r${r}_$i $a "" -n 256 -p "$P"; done
done
# long context
LP=$(cat $H/e6/kl_text.txt)
run long_G G "" -c 4096 -n 64 -p "$LP"
run long_R R "" -c 4096 -n 64 -p "$LP"
# counters
run cnt_G G "simpleperf stat -e instructions:u,cycles:u --" -n 64 -p "$P"
run cnt_R R "simpleperf stat -e instructions:u,cycles:u --" -n 64 -p "$P"
# cold
for a in G R; do
  w=0; while [ "$(sf)" -gt 30500 ] && [ $w -lt 1500 ]; do sleep 30; w=$((w + 30)); done
  echo "cold wait ${w}s shell_mC=$(sf) $( [ "$(sf)" -gt 30500 ] && echo 'NOT COLD')" | tee -a "$O/log.txt"
  run cold_$a $a "" -n 24 -p "$P"
done
echo "done $(date)" | tee "$O/DONE"
