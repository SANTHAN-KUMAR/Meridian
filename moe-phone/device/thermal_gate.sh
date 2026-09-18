#!/system/bin/sh
# thermal_gate.sh — sourced by phone campaigns. Every measured run on the 15R starts only when the SoC is
# NOT throttled: Android thermal status 0 and both cpufreq policies at their hardware maximum. On
# 2026-09-17 the phone sat at thermal status 2 with scaling_max_freq capped to 1.90/1.65 GHz (hardware
# 3.32/3.80), which halved CPU decode between repeats (moe-phone/results/2026-09-17/ocl_split). A run
# that starts throttled measures the cooling, not the engine.
#
#   thermal_state          -> "wake=<Awake|Dozing|Asleep> status=<n> cap0=<kHz> cap6=<kHz> hw0=<kHz> hw6=<kHz> skin=<mC> cpu_max=<C>"
#   thermal_wait MAX_S     -> waits (idle) until unthrottled or MAX_S seconds; prints the state and
#                             "gate=ok|timeout waited=<s>"
P0=/sys/devices/system/cpu/cpufreq/policy0
P6=/sys/devices/system/cpu/cpufreq/policy6
thermal_state() {
  st=$(dumpsys thermalservice 2>/dev/null | awk -F': ' '/^Thermal Status/{print $2; exit}')
  sk=$(for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type 2>/dev/null)" = shell_front ] && cat $z/temp; done | head -1)
  wk=$(dumpsys power 2>/dev/null | awk -F= '/mWakefulness=/{print $2; exit}')
  echo "wake=${wk:-NA} status=${st:-NA} cap0=$(cat $P0/scaling_max_freq) cap6=$(cat $P6/scaling_max_freq) hw0=$(cat $P0/cpuinfo_max_freq) hw6=$(cat $P6/cpuinfo_max_freq) shell_front_mC=${sk:-NA}"
}
thermal_ok() {
  st=$(dumpsys thermalservice 2>/dev/null | awk -F': ' '/^Thermal Status/{print $2; exit}')
  [ "${st:-9}" = 0 ] && [ "$(cat $P0/scaling_max_freq)" = "$(cat $P0/cpuinfo_max_freq)" ] && [ "$(cat $P6/scaling_max_freq)" = "$(cat $P6/cpuinfo_max_freq)" ]
}
# quiesce — before every measured run: force-stop every third-party app (all users), stop the clone
# profile (user 10), kill cached background processes. Prints "quiesce: stopped=<n> top_other_cpu=<pct>"
# where cpu_busy_of_800 is total busy CPU (8 cores) in a 1-s sample after quiescing, with the 3 busiest
# processes, so residual interference is part of the row. System services cannot be stopped without
# root; they stay visible in that number. (Added 2026-09-17 on user direction.)
quiesce() {
  n=0
  for pkg in $(pm list packages -3 2>/dev/null | sed 's/^package://'); do
    am force-stop "$pkg" >/dev/null 2>&1; am force-stop --user 10 "$pkg" >/dev/null 2>&1; n=$((n + 1))
  done
  am stop-user -f 10 >/dev/null 2>&1
  am kill-all >/dev/null 2>&1
  sleep 2
  # second 1-s sample of toybox top (the first sample has no CPU deltas): busy = 800 - idle over 8 cores,
  # plus the three busiest processes by name
  t=$(top -b -n 2 -d 1 -s 1 -o %CPU,NAME 2>/dev/null)
  busy=$(echo "$t" | awk '/%idle/{i=$0} END{sub(/%idle.*/,"",i); n=split(i,a," "); printf "%d", 800-a[n]}')
  top3=$(echo "$t" | awk '$1 ~ /^%CPU/ {blk++; c=0; next} blk==2 && c<3 {printf "%s:%s,", $2, $1; c++}')
  echo "quiesce: stopped=$n cpu_busy_of_800=$busy top3=$top3"
}
thermal_wait() {
  quiesce
  max=${1:-900}; w=0
  while ! thermal_ok; do
    [ "$w" -ge "$max" ] && { echo "$(thermal_state) gate=timeout waited=$w"; return 1; }
    sleep 10; w=$((w + 10))
  done
  echo "$(thermal_state) gate=ok waited=$w"
}


# mem_ready MIN_MIB MAX_S — before a row, make sure the phone has the memory the row needs. The engine's
# cache is sized by --cache-mb auto from what is free at start, so a row that starts with less free memory
# silently gets a smaller cache and is not the same cell as its neighbours (2026-09-18 14:3x: free memory
# crept down between rows as services restarted, and one row got a 1595 MiB cache and read 821 MiB/token).
# Force-stops every running app except the phone's own UI/connectivity and our apps, then waits up to
# MAX_S for MemAvailable to reach MIN_MIB. Prints what it reached; the row logs it either way.
mem_ready() {
  _min=${1:-6500}; _max=${2:-120}; _t=0
  for _p in $(ps -A -o NAME | grep "\." | sed "s/:.*//" | sort -u); do
    case "$_p" in
      com.moephone.*|com.android.systemui|com.android.launcher*|android|system|com.android.shell|com.android.networkstack*|com.android.wifi*|com.google.android.networkstack*|com.android.phone|com.android.bluetooth|com.android.se|com.android.nfc|com.qualcomm.*|vendor.*|com.android.providers.*) continue ;;
    esac
    am force-stop "$_p" 2>/dev/null
  done
  am kill-all 2>/dev/null
  _m=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  while [ "$_m" -lt "$_min" ] && [ $_t -lt "$_max" ]; do sleep 10; _t=$((_t + 10)); _m=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); done
  echo "mem_ready=${_m}MiB waited=${_t}s"
}
