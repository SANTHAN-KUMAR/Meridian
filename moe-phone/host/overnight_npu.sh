#!/bin/bash
# overnight_npu.sh — 2026-09-18 night. Drives the in-app (untrusted_app) engine runs, because the Hexagon
# session only opens in an app's own process, then hands the phone back to the CPU-engine campaign.
# Each run: am start -> wait for EXIT= -> copy files/bench.txt into the results tree with its own name.
set -u
export ANDROID_SERIAL=192.168.0.65:5555
R="/run/media/santhankumar/New Volume/identifying-variation/moe-phone/results/2026-09-18"
PKG=com.moephone.npu2
Q=/data/data/$PKG/files/qwen3.gguf
O=/data/data/$PKG/files/olmoe.gguf
mkdir -p "$R/app_engine"
log() { echo "$(date -Iseconds) $*" >> "$R/app_engine/driver.log"; }

app_run() {   # name  args(~~ separated)
  local name=$1 args=$2 i=0
  adb shell "run-as $PKG sh -c 'rm -f files/out.txt files/bench.txt'" >/dev/null 2>&1
  # quiesce without killing our own app: stop other third-party packages only
  adb shell "for p in \$(pm list packages -3 | sed 's/^package://'); do [ \"\$p\" = $PKG ] || am force-stop \$p; done; am kill-all" >/dev/null 2>&1
  adb shell "input keyevent KEYCODE_WAKEUP; am start -n $PKG/com.moephone.npu.Run --es bench '$args'" >/dev/null 2>&1
  while [ $i -lt 120 ]; do
    sleep 15
    if adb shell "run-as $PKG sh -c 'grep -c EXIT= files/out.txt 2>/dev/null'" 2>/dev/null | tr -d '\r' | grep -qv '^0$'; then break; fi
    i=$((i + 1))
  done
  adb shell "run-as $PKG sh -c 'cat files/bench.txt'" > "$R/app_engine/$name.txt" 2>/dev/null
  adb shell "run-as $PKG sh -c 'cat files/out.txt'" > "$R/app_engine/$name.runner.txt" 2>/dev/null
  adb shell '. /data/local/tmp/moe-stream/thermal_gate.sh; thermal_state' > "$R/app_engine/$name.state.txt" 2>/dev/null
  log "done $name rate=$(grep -oE 'eval time =.*tokens per second' "$R/app_engine/$name.txt" | tail -1)"
}

COMMON="-m~~$Q~~-p~~Write a long detailed essay about the history of computing including its origins its key milestones the people involved and the future directions of the field~~-n~~128~~-no-cnv~~--temp~~0~~-c~~1024~~-b~~64~~-ub~~64~~-t~~4"
log "START"
for rep in 1 2; do
  app_run "qwen_htp_24s_rep$rep"  "completion~~$COMMON~~-dev~~HTP0~~-ngl~~99~~--moe-stream-cache~~24s~~--moe-stream-direct~~--moe-stream-io-threads~~6"
  app_run "qwen_htp_32s_rep$rep"  "completion~~$COMMON~~-dev~~HTP0~~-ngl~~99~~--moe-stream-cache~~32s~~--moe-stream-direct~~--moe-stream-io-threads~~6"
  app_run "qwen_gpu_32s_rep$rep"  "completion~~$COMMON~~-dev~~GPUOpenCL~~-ngl~~99~~--moe-stream-cache~~32s~~--moe-stream-direct~~--moe-stream-io-threads~~6"
  app_run "qwen_cpu_32s_rep$rep"  "completion~~$COMMON~~-ngl~~0~~--moe-stream-cache~~32s~~--moe-stream-direct~~--moe-stream-io-threads~~4"
done
# OLMoE (fits in RAM) on all three devices, for the compute-only comparison
for rep in 1 2; do
  app_run "olmoe_htp_rep$rep" "-m~~$O~~-p~~64~~-n~~64~~-r~~1~~-t~~4~~-dev~~HTP0~~-ngl~~99"
  app_run "olmoe_gpu_rep$rep" "-m~~$O~~-p~~64~~-n~~64~~-r~~1~~-t~~4~~-dev~~GPUOpenCL~~-ngl~~99"
  app_run "olmoe_cpu_rep$rep" "-m~~$O~~-p~~64~~-n~~64~~-r~~1~~-t~~4~~-ngl~~0"
done
log "APP RUNS DONE"
# hand the phone back to the CPU-engine campaign (BigMoeOnEdge, shell domain)
adb shell 'cd /data/local/tmp/moe-stream && echo "overnight: bmoe_mem start $(date)" >> phone_queue.log && (setsid nohup sh bmoe_mem.sh 3 > bmoe_mem_nohup.log 2>&1 < /dev/null &)' >/dev/null 2>&1
log "bmoe_mem launched"
touch "$R/app_engine/DRIVER_DONE"
