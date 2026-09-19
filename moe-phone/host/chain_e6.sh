#!/bin/bash
# chain_e6.sh — research spec E6 on the phone: push (engine 0026, corpus, Q8_0 reference), KL phase, score, knowledge phase for
# REF + FLOOR + KL-passing arms, score. The speed A/B is a separate step decided on the scored result (spec section 0).
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-$(adb devices | awk 'NR>1 && $2=="device"{print $1; exit}')}
MP="/run/media/santhankumar/New Volume/identifying-variation/moe-phone"
R="$MP/results/2026-09-19"; mkdir -p "$R/bmoe_e6"
SP=/tmp/claude-1000/-run-media-santhankumar-New-Volume-identifying-variation/2d806460-912e-4de7-a437-dc77597a5bb6/scratchpad
Q8L="/run/media/santhankumar/New Volume/moe-work/models/Qwen3-30B-A3B-Q8_0.gguf"
H=/data/local/tmp/moe-stream
LOG="$R/chain_e6.log"; log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }
A() { adb shell "$@" </dev/null; }
lock() { A "cat $H/.phone_busy 2>/dev/null" | tr -d '\r'; }
restore() { A 'svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable' >/dev/null 2>&1; log "phone restored"; }
campaign() {  # mode [arms]
  local d
  A "cd $H && (GT_PIN=$(cat "$SP/phone_pin") setsid nohup sh bmoe_e6.sh $1 \"${2:-}\" > bmoe_e6_$1_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
  sleep 60
  while :; do
    d=$(A "ls -d $H/bmoe_e6_${1}_*/ | tail -1" | tr -d '\r')
    [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
    [ -n "$d" ] && A "[ -f ${d}REFUSED ]" && { log "REFUSED: $(A "cat ${d}REFUSED")"; break; }
    [ -z "$(A 'ps -A -o ARGS' | grep 'sh bmoe_e6' | grep -v grep)" ] && { log "bmoe_e6 $1 exited without DONE ($d)"; break; }
    sleep 60
  done
  adb pull "$d" "$R/bmoe_e6" >/dev/null 2>&1 </dev/null; log "pulled $d"
  LAST="$R/bmoe_e6/$(basename "$d")"
}
log "start (device $ANDROID_SERIAL)"
[ -n "$(lock)" ] && { log "phone busy: $(lock)"; exit 1; }
orph=$(A 'ps -A -o ARGS' | grep -E 'bmoe-cli|gx_|sh bmoe_' | grep -v grep); [ -n "$orph" ] && { log "FATAL foreign: $orph"; exit 1; }
# engine, corpus, script
A "mkdir -p $H/bmoe-i8mm-0026 $H/e6/mmlu" >/dev/null
for f in "$SP/bmoe-i8mm-0026"/*; do adb push "$f" "$H/bmoe-i8mm-0026/" >/dev/null 2>&1 </dev/null; done
want=$(grep bmoe-cli "$SP/bmoe-i8mm-0026/MD5" | cut -d' ' -f1); got=$(A "md5sum $H/bmoe-i8mm-0026/bmoe-cli" | cut -d' ' -f1)
[ "$want" = "$got" ] || { log "FATAL 0026 md5 $got != $want"; exit 1; }
A "chmod 755 $H/bmoe-i8mm-0026/bmoe-cli" >/dev/null
adb push "$R/e6_corpus/kl_text.txt" "$R/e6_corpus/mmlu_list_phone.txt" "$H/e6/" >/dev/null 2>&1 </dev/null
adb push "$R/e6_corpus/mmlu/." "$H/e6/mmlu/" >/dev/null 2>&1 </dev/null
[ "$(A "ls $H/e6/mmlu | wc -l" | tr -d '\r ')" = 100 ] || { log "FATAL mmlu files not all pushed"; exit 1; }
adb push "$MP/device/bmoe_e6.sh" "$H/" >/dev/null 2>&1 </dev/null
# reference model: laptop sha256 must match the published one, phone copy must match the laptop's size and sha256
exp=$(cat "$Q8L.sha256.expected")
if [ ! -f "$Q8L.sha256.ok" ]; then
  got=$(sha256sum "$Q8L" | cut -d' ' -f1); [ "$got" = "$exp" ] || { log "FATAL Q8_0 download sha256 $got != published $exp"; exit 1; }
  echo "$got" > "$Q8L.sha256.ok"; log "Q8_0 download verified against the published sha256"
fi
if [ "$(A "sha256sum $H/Qwen3-30B-A3B-Q8_0.gguf 2>/dev/null" | cut -d' ' -f1)" != "$exp" ]; then
  log "pushing Q8_0 (32.5 GB)"; adb push "$Q8L" "$H/Qwen3-30B-A3B-Q8_0.gguf" >/dev/null 2>&1 </dev/null
  [ "$(A "sha256sum $H/Qwen3-30B-A3B-Q8_0.gguf" | cut -d' ' -f1)" = "$exp" ] || { log "FATAL phone Q8_0 sha256 mismatch"; exit 1; }
fi
log "Q8_0 on the phone verified"
A "svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable" >/dev/null 2>&1
campaign kl; K="$LAST"
python3 "$MP/gates/e6_score.py" --kl "$K" --answers "$R/e6_corpus/mmlu_answers.json" --out "$R/e6_kl_summary.json" > "$R/e6_kl_summary.txt" 2>&1
log "KL phase: $(tr '\n' ' ' < "$R/e6_kl_summary.txt")"
PASS=$(python3 -c "
import json; d=json.load(open('$R/e6_kl_summary.json'))['verdict']
print(' '.join(a for a,v in d.items() if v['passes_kl']))" 2>/dev/null)
log "arms passing the KL criteria: [${PASS}]"
campaign mc "REF FLOOR $PASS"; MC="$LAST"
python3 "$MP/gates/e6_score.py" --kl "$K" --mc "$MC" --answers "$R/e6_corpus/mmlu_answers.json" --out "$R/e6_summary.json" > "$R/e6_summary.txt" 2>&1
log "E6 scored: $(tr '\n' ' ' < "$R/e6_summary.txt")"
restore
touch "$R/CHAIN_E6_DONE"
