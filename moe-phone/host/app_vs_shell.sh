#!/bin/bash
# app_vs_shell.sh — what does running our engine INSIDE an app process cost, measured with the thermal
# state and the cache budget held as equal as this phone allows?
#
# WHY. The first in-app row of the attention A/B came in at 4.59 tok/s against the 6.20 tok/s shell
# baseline (results/2026-09-17/bmoe_cache.json, cell ceil5000) — but its own log says why, and neither
# reason is the SELinux domain:
#   * the cache auto-budget was granted 4127 MiB, not ~5000, because only 5150 MiB was free right after
#     a 17 GB model copy; the hit rate fell to 78.8% and the per-token read rose to 185 MiB;
#   * the cores were thermally capped at 1.6516 GHz of 3.8016 (cap6) after the preceding campaign.
# A row taken under a 43% frequency cap measures the cap. So the app-vs-shell question gets its own
# campaign, in which the arms alternate within a repeat and each row carries its caps and its granted
# budget, and the whole thing starts only after a long idle.
#
# Arms, identical flags (the ceil5000 string), alternating, 3 repeats:
#   app    bmoe_main in com.moephone.bmoe3's process, via the JNI shim
#   shell  the same engine build invoked from adb shell
# A row whose cap6 is below the threshold below, or whose granted budget differs from its pair's by more
# than 5%, is still recorded but flagged: the comparison is only between rows that were given the same
# machine. Flagging happens in the log, never by dropping a row.
set -u
export ANDROID_SERIAL=192.168.0.65:5555
PKG=${PKG:-com.moephone.bmoe3}
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/$(date +%F)/app_vs_shell"
H=/data/local/tmp/moe-stream
APP_M=/data/data/$PKG/files/qwen3.gguf
SH_M=$H/Qwen3-30B-A3B-Q4_0.gguf
IDLE=${IDLE:-600}      # seconds of quiet before the campaign, so it does not start throttled
SETTLE=${SETTLE:-180}  # seconds between rows
REPS=${1:-3}
mkdir -p "$R"
log() { echo "$(date -Iseconds) $*" >> "$R/driver.log"; }

PROMPT="Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field"
FLAGS="--chatml -n 256 --ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f"

state() { adb shell '. /data/local/tmp/moe-stream/thermal_gate.sh; thermal_state' 2>/dev/null | tr -d '\r'; }

run_app() {  # tag
  local tag=$1 i=0
  local jargs=$(echo "bmoe -m $APP_M -p $PROMPT $FLAGS" | sed 's/ /~~/g')
  adb shell "run-as $PKG sh -c 'rm -f files/out.txt files/bench.txt'" >/dev/null 2>&1
  adb shell "input keyevent KEYCODE_WAKEUP; am start -n $PKG/com.moephone.npu.Run --es bench '$jargs'" >/dev/null 2>&1
  while [ $i -lt 80 ]; do
    sleep 15
    if adb shell "run-as $PKG sh -c 'grep -c EXIT= files/out.txt 2>/dev/null'" 2>/dev/null | tr -d '\r' | grep -qv '^0$'; then break; fi
    i=$((i + 1))
  done
  adb shell "run-as $PKG sh -c 'cat files/bench.txt'" > "$R/$tag.out" 2>/dev/null
  : > "$R/$tag.err"
}

run_shell() {  # tag
  local tag=$1
  adb shell "input keyevent KEYCODE_WAKEUP; cd $H/bmoe-i8mm-verify && LD_LIBRARY_PATH=. ./bmoe-cli -m $SH_M $FLAGS -p '$PROMPT'" > "$R/$tag.out" 2> "$R/$tag.err"
}

row() {  # tag kind
  local tag=$1 kind=$2
  local before=$(state)
  sleep "$SETTLE"
  before=$(state)
  case $kind in app) run_app "$tag" ;; shell) run_shell "$tag" ;; esac
  local after=$(state)
  local gen=$(grep -hoE 'generation: .*tok/s\)' "$R/$tag.out" "$R/$tag.err" | tail -1)
  local budget=$(grep -hoE 'budget [0-9]+ MiB' "$R/$tag.out" "$R/$tag.err" | tail -1)
  local hit=$(grep -hoE 'moe-cache: [0-9.]+% hit' "$R/$tag.out" "$R/$tag.err" | tail -1)
  log "$tag [$kind] BEFORE $before AFTER $after :: $gen :: $budget :: $hit"
}

log "START reps=$REPS idle=${IDLE}s settle=${SETTLE}s"
# quiesce and idle first: this campaign is about the domain, so it must not also be about temperature
adb shell "for p in \$(pm list packages -3 | sed 's/^package://'); do [ \"\$p\" = $PKG ] || am force-stop \$p; done; am kill-all" >/dev/null 2>&1
log "idling ${IDLE}s to shed heat; state now: $(state)"
sleep "$IDLE"
log "idle done; state now: $(state)"
for rep in $(seq 1 "$REPS"); do
  if [ $((rep % 2)) -eq 1 ]; then order="app shell"; else order="shell app"; fi
  for k in $order; do row "${k}_rep${rep}" "$k"; done
done
log "DONE"
touch "$R/DONE"
