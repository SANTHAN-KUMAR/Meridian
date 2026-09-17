#!/system/bin/sh
# keep_awake.sh — during measurement campaigns the phone must stay Awake (a dozing 15R halved decode speed,
# 2026-09-17). Every 15 s: if wakefulness is not Awake, wake the screen and dismiss the keyguard.
# Interventions are logged so a row measured across one can be identified. Stop: rm keep_awake.run
H=/data/local/tmp/moe-stream
touch $H/keep_awake.run
while [ -f $H/keep_awake.run ]; do
  w=$(dumpsys power 2>/dev/null | awk -F= '/mWakefulness=/{print $2; exit}')
  if [ "$w" != "Awake" ]; then
    input keyevent KEYCODE_WAKEUP; sleep 1; wm dismiss-keyguard >/dev/null 2>&1
    echo "$(date +%H:%M:%S) woke from $w" >> $H/keep_awake.log
  fi
  sleep 15
done
