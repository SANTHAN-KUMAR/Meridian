#!/bin/bash
# chain_e6b.sh — E6 resumed after the heat mitigation (bmoe_e6.sh REVISION 17:40): klresume in the existing KL dir, score, knowledge phase, score; screen off, cooldown-gated. From chain_e6.sh — research spec E6 on the phone: push (engine 0026, corpus, Q8_0 reference), KL phase, score, knowledge phase for
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
  A "cd $H && (E6_NOWAKE=1 GT_PIN=$(cat "$SP/phone_pin") setsid nohup sh bmoe_e6.sh $1 \"${2:-}\" > bmoe_e6_$1_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
  sleep 60
  while :; do
    if [ "$1" = klresume ]; then d="$2/"; else d=$(A "ls -d $H/bmoe_e6_${1}_*/ | tail -1" | tr -d '\r'); fi
    [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
    [ -n "$d" ] && A "[ -f ${d}REFUSED ]" && { log "REFUSED: $(A "cat ${d}REFUSED")"; break; }
    [ -z "$(A 'ps -A -o ARGS' | grep 'sh bmoe_e6' | grep -v grep)" ] && { log "bmoe_e6 $1 exited without DONE ($d)"; break; }
    sleep 60
  done
  adb pull "$d" "$R/bmoe_e6" >/dev/null 2>&1 </dev/null; log "pulled $d"
  LAST="$R/bmoe_e6/$(basename "$d")"
}
log "[e6b] start: klresume"
KD=$(A "ls -d $H/bmoe_e6_kl_*/ | tail -1" | tr -d '\r'); KD=${KD%/}
campaign klresume "$KD"; K="$LAST"
python3 "$MP/gates/e6_score.py" --kl "$K" --answers "$R/e6_corpus/mmlu_answers.json" --out "$R/e6_kl_summary.json" > "$R/e6_kl_summary.txt" 2>&1
log "[e6b] KL phase: $(tr '\n' ' ' < "$R/e6_kl_summary.txt")"
PASS=$(python3 -c "
import json; d=json.load(open('$R/e6_kl_summary.json'))['verdict']
print(' '.join(a for a,v in d.items() if v['negligible_kl']))" 2>/dev/null)
log "[e6b] arms passing the KL criteria: [${PASS}]"
if [ -n "$PASS" ]; then
  campaign mc "REF FLOOR $PASS"; MC="$LAST"
  python3 "$MP/gates/e6_score.py" --kl "$K" --mc "$MC" --answers "$R/e6_corpus/mmlu_answers.json" --out "$R/e6_summary.json" > "$R/e6_summary.txt" 2>&1
else
  log "[e6b] no arm is negligible on the KL/PPL measures: knowledge probe skipped (it cannot change the verdict)"
  cp "$R/e6_kl_summary.json" "$R/e6_summary.json"; cp "$R/e6_kl_summary.txt" "$R/e6_summary.txt"
fi
log "[e6b] E6 scored: $(tr '\n' ' ' < "$R/e6_summary.txt")"
restore
touch "$R/CHAIN_E6_DONE"
