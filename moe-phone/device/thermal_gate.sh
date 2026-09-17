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
thermal_wait() {
  max=${1:-900}; w=0
  while ! thermal_ok; do
    [ "$w" -ge "$max" ] && { echo "$(thermal_state) gate=timeout waited=$w"; return 1; }
    sleep 10; w=$((w + 10))
  done
  echo "$(thermal_state) gate=ok waited=$w"
}
