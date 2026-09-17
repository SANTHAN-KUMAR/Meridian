#!/system/bin/sh
# keep_awake.sh — during measurement campaigns the phone must stay Awake (a dozing 15R halved decode speed,
# 2026-09-17). Every 15 s: if wakefulness is not Awake, wake the screen; if the keyguard shows, unlock
# with the PIN given as $1 (argv only; never written to disk).
#   sh keep_awake.sh PIN
# Interventions are logged so a row measured across one can be identified. Stop: rm keep_awake.run
H=/data/local/tmp/moe-stream
PIN=${1:-}
touch $H/keep_awake.run
while [ -f $H/keep_awake.run ]; do
  w=$(dumpsys power 2>/dev/null | awk -F= '/mWakefulness=/{print $2; exit}')
  if [ "$w" != "Awake" ]; then
    input keyevent KEYCODE_WAKEUP; sleep 1
    echo "$(date +%H:%M:%S) woke from $w" >> $H/keep_awake.log
  fi
  if [ -n "$PIN" ] && dumpsys window 2>/dev/null | grep -q "isKeyguardShowing=true"; then
    input swipe 636 2380 636 840 200; sleep 1.5; input text "$PIN"; sleep 0.5; input keyevent KEYCODE_ENTER
    echo "$(date +%H:%M:%S) unlocked keyguard" >> $H/keep_awake.log
  fi
  sleep 15
done
