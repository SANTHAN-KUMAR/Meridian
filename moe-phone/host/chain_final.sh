#!/bin/bash
# chain_final.sh — the two paper-only measurements (research/2026-09-19_CLOSURE.md §4), unplugged on Wi-Fi adb, engine bmoe-i8mm-0027:
#   1. device/bmoe_prerepack.sh: fidelity gate (|dPPL|/PPL <= 0.2%, hits change <= 1%), smoke, ABBA x6 (pre-registered 09:05)
#   2. device/bmoe_h2h.sh: same-session head-to-head vs BigMoeOnEdge (pre-registered 09:15); our arm = repack only if (1) is decisive positive
set -u
MP="/run/media/santhankumar/New Volume/identifying-variation/moe-phone"; R="$MP/results/2026-09-19"; mkdir -p "$R/bmoe_prerepack" "$R/bmoe_h2h"
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.0.65:5555}
H=/data/local/tmp/moe-stream; LOG="$R/chain_final.log"; log() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }
A() { adb shell "$@" </dev/null; }
reconnect() { adb shell 'echo up' </dev/null 2>/dev/null | grep -q up || adb connect "$ANDROID_SERIAL" >/dev/null 2>&1; }
PIN=$(cat "$HOME/.config/moe-phone/phone_pin")
campaign() {  # script mode env
  local d
  A "cd $H && ($3 GT_PIN=$PIN setsid nohup sh $1 $2 > ${1%.sh}_$2_nohup.log 2>&1 < /dev/null &)" >/dev/null 2>&1
  sleep 60
  while :; do
    reconnect
    d=$(A "ls -d $H/${1%.sh}_${2}_*/ | tail -1" | tr -d '\r')
    [ -n "$d" ] && A "[ -f ${d}DONE ]" && break
    [ -n "$d" ] && A "[ -f ${d}REFUSED ]" && { log "REFUSED: $(A "cat ${d}REFUSED")"; break; }
    [ -z "$(A 'ps -A -o ARGS' | grep "sh $1" | grep -v grep)" ] && { log "$1 $2 exited without DONE ($d)"; break; }
    sleep 60
  done
  sleep 5; adb pull "$d" "$R/${1%.sh}" >/dev/null 2>&1 </dev/null; log "pulled $d"
  LAST="$R/${1%.sh}/$(basename "$d")"
}
reconnect
[ -n "$(A "cat $H/.phone_busy 2>/dev/null" | tr -d '\r')" ] && { log "phone busy"; exit 1; }
orph=$(A 'ps -A -o ARGS' | grep -E 'bmoe-cli|gx_|sh bmoe_|llama-bench' | grep -v grep); [ -n "$orph" ] && { log "FATAL foreign: $orph"; exit 1; }
adb push "$MP/device/bmoe_prerepack.sh" "$MP/device/bmoe_h2h.sh" "$H/" >/dev/null 2>&1 </dev/null
A "svc power stayon true; settings put system screen_off_timeout 1800000; dumpsys deviceidle disable" >/dev/null 2>&1
ENVR="GT_BIN=bmoe-i8mm-0027"
log "start (serial $ANDROID_SERIAL, power: $(A 'dumpsys battery | grep -E "AC powered|USB powered| level"' | tr -s ' ' | tr '\n' ' '))"
# 1a fidelity
campaign bmoe_prerepack.sh fidelity "$ENVR"; F="$LAST"
pb=$(grep -o "ppl_base exit=0 ppl: [0-9.]*" "$F/log.txt" | awk '{print $4}'); ps=$(grep -o "ppl_stack exit=0 ppl: [0-9.]*" "$F/log.txt" | awk '{print $4}')
hb=$(grep -o "ppl_base.*hits: [0-9]*/[0-9]*" "$F/log.txt" | grep -o "[0-9]*/[0-9]*$"); hs=$(grep -o "ppl_stack.*hits: [0-9]*/[0-9]*" "$F/log.txt" | grep -o "[0-9]*/[0-9]*$")
fid=$(python3 -c "
pb,ps,hb,hs='$pb','$ps','$hb','$hs'
try:
  r=abs(float(ps)-float(pb))/float(pb); nb,t=map(int,hb.split('/')); ns,_=map(int,hs.split('/')); f=abs(ns-nb)/t
  print('ok' if (r<=0.002 and f<=0.01) else 'FAIL', 'dppl=%.4f%% dhits=%d/%d'%(100*r,abs(ns-nb),t))
except Exception as e: print('FAIL parse', e)")
log "[prerepack] fidelity: base $pb ($hb), repacked $ps ($hs) -> $fid"
OURS=plain
case "$fid" in
  ok*)
    campaign bmoe_prerepack.sh smoke "$ENVR"; S="$LAST"; ok=1
    [ "$(grep -c '^exit=0' "$S/log.txt")" = 2 ] || ok=0
    grep -q FATAL "$S"/*.err && ok=0
    grep -q "marker ok" "$S/stack_reps1.err" || ok=0
    [ "$(grep -c 'wake_start=Awake' "$S/log.txt")" = 2 ] || ok=0
    log "[prerepack] smoke ok=$ok :: $(grep -h 'generation:' "$S"/*.out | tr '\n' ' ')"
    if [ $ok = 1 ]; then
      campaign bmoe_prerepack.sh ab "$ENVR"; P="$LAST"
      python3 "$MP/gates/stack_summary.py" "$P" --any-budget-prereg --require-awake --out "$R/prerepack_ab_summary.json" > "$R/prerepack_ab_summary.txt" 2>&1
      log "[prerepack] A/B: $(grep -E '^decode_tok_s|^compute_ms|^stall_ms|n_kept' "$R/prerepack_ab_summary.txt" | tr '\n' ' ' | cut -c1-700)"
      OURS=$(python3 -c "
import json; d=json.load(open('$R/prerepack_ab_summary.json'))['results']['decode_tok_s']
print('repack' if d.get('decisive') and d.get('mean_diff',0)>0 else 'plain')")
    fi ;;
  *) log "[prerepack] fidelity gate failed: no smoke, no A/B; h2h uses the plain stack" ;;
esac
# 2 head-to-head
log "[h2h] ours=$OURS"
campaign bmoe_h2h.sh smoke "$ENVR H2H_OURS=$OURS"; S="$LAST"; ok=1
[ "$(grep -c '^exit=0' "$S/log.txt")" = 2 ] || ok=0
[ "$(grep -c 'wake_start=Awake' "$S/log.txt")" = 2 ] || ok=0
log "[h2h] smoke ok=$ok :: $(grep -h 'generation:' "$S"/*.out | tr '\n' ' ')"
b=$(A "dumpsys battery | grep -m1 ' level' | tr -dc 0-9")
if [ $ok = 1 ] && [ "${b:-0}" -ge 30 ]; then
  campaign bmoe_h2h.sh ab "$ENVR H2H_OURS=$OURS"; HH="$LAST"
  python3 "$MP/gates/stack_summary.py" "$HH" --any-budget-prereg --require-awake --out "$R/h2h_ab_summary.json" > "$R/h2h_ab_summary.txt" 2>&1
  log "[h2h] A/B: $(grep -E '^decode_tok_s|decode_ratio|n_kept' "$R/h2h_ab_summary.txt" | tr '\n' ' ' | cut -c1-700)"
else log "[h2h] A/B not started (smoke ok=$ok, battery ${b}%)"; fi
A 'svc power stayon false; settings put system screen_off_timeout 60000; dumpsys deviceidle enable; input keyevent KEYCODE_SLEEP' >/dev/null 2>&1
touch "$R/CHAIN_FINAL_DONE"; log "done"
