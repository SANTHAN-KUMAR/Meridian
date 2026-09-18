#!/bin/bash
# clock_sampler.sh — record the CPU frequency CAPS alongside every campaign, because they move on their
# own and they move the result. Across 74 committed rows, decode at cap6 >= 3.0 GHz has a median of
# 6.77 tok/s against 5.48 at cap6 <= 1.7 GHz: a 1.24x swing, larger than any engine lever measured so
# far, and 67 of those 74 rows ran capped. A campaign that does not record the cap cannot tell its own
# effect from the SoC's power state.
# Columns: unix time, policy0/policy6 scaling_max and current, thermal status, battery temp, shell temp.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
OUT=${1:?output csv}
INT=${2:-2}
echo "t,p0_max,p6_max,p0_cur,p6_cur,thermal_status,batt_decideg,shell_mC" > "$OUT"
while :; do
  adb shell 'echo "$(date +%s),$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq),$(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq),$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq),$(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq),$(dumpsys thermalservice 2>/dev/null | awk -F": " "/^Thermal Status/{print \$2; exit}"),$(dumpsys battery | awk "/temperature/{print \$2}"),$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1)"' 2>/dev/null | tr -d '\r' >> "$OUT"
  sleep "$INT"
done
