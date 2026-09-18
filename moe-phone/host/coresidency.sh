#!/bin/bash
# coresidency.sh — can the engine stay alive while ANOTHER app is in the foreground?
#
# WHY THIS IS THE DECIDING EXPERIMENT for on-device task automation. An agent that drives the phone must
# run while the app it is driving is in front: the target app is foreground, the agent is not. Every
# measurement in this project so far had the opposite arrangement -- our process foreground, 76 other
# packages force-stopped -- so none of them says whether the arrangement an agent needs is possible.
#
# The reason to doubt it: the engine's expert cache is ~5 GB of anonymous memory on an 11.4 GB device.
# Android's low-memory killer ranks a backgrounded process far below a foreground one, and 5 GB is the
# largest single allocation on the phone. Either the agent is killed when it backgrounds, or the system
# evicts the app the agent is supposed to be driving, and both are fatal in different ways.
#
# ARMS (each runs the same streamed decode inside the app, and differs only in what else is happening):
#   alone        our app foreground, everything else stopped. The control -- this is every previous row.
#   home         our app backgrounded by going to the launcher.
#   target       our app backgrounded by launching a real, memory-hungry app on top of it.
# Recorded per arm: whether our process was still alive at the end, the decode rate if it finished, our
# RSS over time, MemAvailable, the LMK's own kills from logcat, and whether the target app survived.
#
# A "foreground service" would be the production answer (a persistent notification raises the process's
# LMK rank), and it is NOT tested here: our harness is an activity. If the backgrounded arms die, that is
# the next thing to build, and this campaign says how much has to change.
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
PKG=${PKG:-com.moephone.bmoe3}
TARGET=${TARGET:-com.android.chrome}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/$(date +%F)/coresidency"
M=/data/data/$PKG/files/qwen3.gguf
REPS=${1:-2}
mkdir -p "$R"
log() { echo "$(date -Iseconds) $*" >> "$R/driver.log"; }

PROMPT="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
FLAGS="--chatml -n 128 --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f"

run_arm() {  # tag  what_to_bring_forward ("" | home | target)
  local tag=$1 bring=$2 i=0
  adb shell "am force-stop $PKG" >/dev/null 2>&1 </dev/null
  adb shell "run-as $PKG sh -c 'rm -f files/out.txt files/bench.txt'" >/dev/null 2>&1 </dev/null
  adb logcat -c >/dev/null 2>&1 </dev/null
  local jargs=$(echo "bmoe -m $M -p $PROMPT $FLAGS" | sed 's/ /~~/g')
  adb shell "input keyevent KEYCODE_WAKEUP; am start -n $PKG/com.moephone.npu.Run --es bench '$jargs'" >/dev/null 2>&1 </dev/null
  # let the model load and the cache fill before anything else takes the foreground
  sleep 45
  case $bring in
    home)   adb shell 'am start -a android.intent.action.MAIN -c android.intent.category.HOME' >/dev/null 2>&1 </dev/null; log "$tag: went to launcher" ;;
    target) adb shell "monkey -p $TARGET -c android.intent.category.LAUNCHER 1" >/dev/null 2>&1 </dev/null; log "$tag: launched $TARGET" ;;
  esac
  : > "$R/$tag.rss"
  while [ $i -lt 100 ]; do
    pid=$(adb shell "pidof $PKG" 2>/dev/null </dev/null | tr -d '\r' | awk '{print $1}')
    rss=$([ -n "$pid" ] && adb shell "awk '/VmRSS/{print \$2}' /proc/$pid/status" 2>/dev/null </dev/null | tr -d '\r')
    avail=$(adb shell "awk '/MemAvailable/{print \$2}' /proc/meminfo" 2>/dev/null </dev/null | tr -d '\r')
    echo "$(date +%s),${pid:-DEAD},${rss:-0},${avail:-0}" >> "$R/$tag.rss"
    [ -z "$pid" ] && { log "$tag: OUR PROCESS IS GONE after $((i*5+45))s"; break; }
    if adb shell "run-as $PKG sh -c 'grep -c EXIT= files/out.txt 2>/dev/null'" 2>/dev/null </dev/null | tr -d '\r' | grep -qv '^0$'; then break; fi
    i=$((i + 1)); sleep 5
  done
  adb shell "run-as $PKG sh -c 'cat files/bench.txt'" > "$R/$tag.txt" 2>/dev/null </dev/null
  adb logcat -d -b main -t 2000 2>/dev/null </dev/null | grep -iE "lowmemorykiller|lmkd|am_kill|Killing" > "$R/$tag.lmk.txt"
  local alive_target=$(adb shell "pidof $TARGET" 2>/dev/null </dev/null | tr -d '\r')
  log "$tag :: $(grep -oE 'generation: .*tok/s\)' "$R/$tag.txt" | tail -1)  | lmk lines $(wc -l < "$R/$tag.lmk.txt") | target alive: ${alive_target:-no}"
}

log "START reps=$REPS pkg=$PKG target=$TARGET"
for rep in $(seq 1 "$REPS"); do
  run_arm "alone_rep$rep"  ""
  run_arm "home_rep$rep"   home
  run_arm "target_rep$rep" target
done
log "DONE"
touch "$R/DONE"
