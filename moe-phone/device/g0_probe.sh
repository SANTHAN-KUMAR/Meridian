#!/bin/sh
# g0_probe.sh — gate G0: record the device facts every later measurement depends on.
#
# Runs in Termux (as an ordinary app — the regime we care about) or in `adb shell`.
# Read-only: it changes nothing on the phone. Some lines print "denied" — that is
# itself a finding (e.g. power rails not readable without root) and is kept.
#
#   sh g0_probe.sh > g0_$(date +%Y%m%d).txt
#
# Then copy the file to moe-phone/results/<date>/ on the computer and run
#   python moe-phone/gates/g0_summarize.py <file>

say() { printf '\n== %s ==\n' "$1"; }
try() { sh -c "$1" 2>/dev/null || echo "denied/absent: $1"; }

say "identity"
for p in ro.product.manufacturer ro.product.model ro.soc.manufacturer ro.soc.model \
         ro.board.platform ro.build.version.release ro.build.version.sdk ro.build.fingerprint; do
  printf '%s=%s\n' "$p" "$(getprop $p 2>/dev/null)"
done
try "uname -a"
echo "context: $(id 2>/dev/null)"

say "memory"
try "grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree|Cached' /proc/meminfo"
try "cat /proc/self/oom_score_adj"
try "cat /sys/block/zram0/disksize"
try "getprop ro.lmk.use_minfree_levels; getprop ro.config.low_ram"

say "storage"
try "mount | grep -E ' /data '"
for q in scheduler nr_requests read_ahead_kb max_sectors_kb rotational; do
  for d in /sys/block/sda /sys/block/sdc /sys/block/dm-0; do
    [ -e "$d/queue/$q" ] && printf '%s/%s=%s\n' "$d" "$q" "$(cat $d/queue/$q 2>/dev/null || echo denied)"
  done
done
try "ls /sys/class/ufs* /sys/bus/platform/drivers/ufshcd* 2>/dev/null | head -20"
try "df -h /data"

say "npu (Hexagon) presence"
for f in /vendor/lib64/libcdsprpc.so /system/lib64/libcdsprpc.so /vendor/lib64/libadsprpc.so; do
  [ -e "$f" ] && echo "present: $f" || echo "absent: $f"
done
# Skeleton library names carry the HTP version (e.g. ...V81Skel.so); list what the image ships.
try "ls /vendor/lib/rfsa/adsp /vendor/dsp /vendor/dsp/cdsp 2>/dev/null | grep -iE 'htp|skel|v7[0-9]|v8[0-9]' | head -40"
try "ls -l /dev/fastrpc* /dev/adsprpc* 2>/dev/null"

say "gpu"
try "ls /sys/class/kgsl/kgsl-3d0/ 2>/dev/null | head -5"
try "cat /sys/class/kgsl/kgsl-3d0/gpu_model"

say "power measurement"
for f in current_now voltage_now charge_counter status; do
  printf 'battery/%s=%s\n' "$f" "$(cat /sys/class/power_supply/battery/$f 2>/dev/null || echo denied)"
done
try "ls /sys/bus/iio/devices/ 2>/dev/null"
try "dumpsys powerstats 2>/dev/null | head -40"

say "thermal zones"
for z in /sys/class/thermal/thermal_zone*; do
  printf '%s %s %s\n' "$(basename $z)" "$(cat $z/type 2>/dev/null)" "$(cat $z/temp 2>/dev/null)"
done | head -60

say "cpu"
try "grep -E 'processor|CPU part' /proc/cpuinfo | paste - - | head -16"
for c in /sys/devices/system/cpu/cpu[0-9]*; do
  printf '%s max_khz=%s\n' "$(basename $c)" "$(cat $c/cpufreq/cpuinfo_max_freq 2>/dev/null)"
done

say "root"
if command -v su >/dev/null 2>&1; then echo "su present: $(command -v su)"; else echo "su absent (unrooted)"; fi
try "getprop ro.boot.verifiedbootstate; getprop ro.boot.flash.locked"

say "done"
