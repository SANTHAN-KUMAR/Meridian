#!/system/bin/sh
# bmoe_e6.sh — research spec E6 (research/2026-09-19_RESEARCH_SPEC.md §6 E6 and §0 closing rule): the quality cost of the
# engine's training-free lossy routing options on Qwen3-30B-A3B, measured against a reference model. Speed is a separate step.
#
# PRE-REGISTERED (2026-09-19 ~16:00, before any Qwen3 E6 row):
#   Reference  REF   = Qwen3-30B-A3B Q8_0 (unsloth, sha256 a68fe734…), top-8. A PROXY for BF16; that limitation is stated.
#   Margin     FLOOR = the shipped Q4_0 file (unsloth, 17,379,988,032 B), top-8: its KL to REF defines "the loss already accepted".
#   Arms (Q4_0, one flag each):  T7 --n-expert-used 7 | T6 --n-expert-used 6 | RA1 --route-ahead 1 |
#                                DC05 --drop-cold-experts 0.5 | DC10 --drop-cold-experts 1.0 | SUB --expert-substitute 0.15
#   KL text    e6/kl_text.txt (human-written: 2 Wikipedia extracts, CPython heapq.py, one oasst1 turn; ~3,000 tokens), -c 4096,
#              --ppl-step (the decode regime every cache-dependent policy acts in), --ppl-dump-k 8192 (KL over REF's top-8192 plus one
#              tail bucket: a lower bound of the full KL; REF's largest tail mass is reported).
#   Knowledge  e6/mmlu (100 MMLU test questions, 51 subjects, zero-shot), first-token log-prob of " A"/" B"/" C"/" D" (mode: see below).
#              Run for REF, FLOOR and every arm that passes the three KL criteria (a fail on any criterion is a fail).
#              Batch mode for REF/FLOOR/T7/T6/RA1; --ppl-step only for DC05/DC10/SUB, whose policies act on cache residency and
#              barely fire in a batch. The batch/step numeric difference is floating-point reordering (~1e-6), far below a choice.
#   PASS (all four; gates/e6_score.py):  KL_mean(arm) <= 2 x KL_mean(FLOOR);  KL_p99(arm) <= 2 x KL_p99(FLOOR);
#              flips(arm) <= 2 x flips(FLOOR);  MMLU accuracy(arm) >= the lower end of FLOOR's 95% Wilson interval.
#   Adoption (§0): a passing arm is adopted only if it ALSO decodes >= 10.0 tok/s (median, awake, unplugged, steady state) in the
#              speed A/B that follows; that A/B is a separate, pre-registered script.
# All arms use the deployed stack flags, except where a flag excludes them (route-ahead excludes --predict-prefetch and
# --spec-adopt-selective); the cache budget is the deployed one. The phone is held Awake (waker below). Shared lock, battery >= 25%.
# REVISION 2026-09-19 17:40 (after the REF row, before any arm row was scored): the REF row ended at thermal status 3,
# shell 49 C, battery 47.8 C (on USB power, screen held on). Quality numbers do not depend on clock or wake state (a suspend
# pauses the run), so from here: E6_NOWAKE=1 leaves the screen off, and cooldown() waits (up to 20 min, logged) before every
# row until thermal status < 3 and shell_front <= 43 C. Mode klresume DIR finishes the KL phase in DIR with its existing
# ref.bkl, skipping arms already scored. The scoring rule is unchanged.
# CONTROL added 2026-09-19 19:00 (after the T7/T6/RA1 rows): mode ident dumps FLOOR (Q4_0 top-8) and re-scores FLOOR against it
# on the phone. Expected: KL exactly 0 and 0 flips. Any other value means run-to-run nondeterminism in the phone build, and every
# E6 KL number is then suspect.
#   GT_PIN=... sh bmoe_e6.sh kl | klresume DIR | mc "ARM ARM ..." | ident
set -u
MODE=${1:-kl}; MC_ARMS=${2:-}
NOWAKE=${E6_NOWAKE:-0}
H=/data/local/tmp/moe-stream
BIN=bmoe-i8mm-0026
. $H/thermal_gate.sh
Q4=$H/Qwen3-30B-A3B-Q4_0.gguf
Q8=$H/Qwen3-30B-A3B-Q8_0.gguf
E=$H/e6
if [ "$MODE" = klresume ]; then O=$MC_ARMS; else O=$H/bmoe_e6_${MODE}_$(date +%Y%m%d_%H%M); mkdir -p "$O"; fi
STACK="--ubatch 512 --moe-stream --cache-mb auto --cache-floor-mb 1024 --cache-ceil-mb 5000 --overlap --dense-weights anon -t 4 --cpu-mask f0 --io-threads 4 --io-cpu-mask 0f --expert-slru"
PRED="--predict-prefetch --spec-adopt-selective"
KLA="-c 4096 --ppl $E/kl_text.txt --ppl-step --ppl-dump-k 8192"
if [ -e $H/.phone_busy ]; then echo "phone busy: $(cat $H/.phone_busy)" | tee "$O/REFUSED"; exit 3; fi
echo "e6 $MODE $(date +%H:%M:%S)" > $H/.phone_busy
PIN=${GT_PIN:-}
cooldown() {
  w=0
  while [ $w -lt 1200 ]; do
    st=$(dumpsys thermalservice | grep -m1 'Thermal Status' | tr -dc 0-9); sf=$(for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type 2>/dev/null)" = shell_front ] && cat $z/temp; done | head -1)
    [ "${st:-9}" -lt 3 ] && [ "${sf:-99999}" -le 43000 ] && break
    sleep 30; w=$((w + 30))
  done
  echo "cooldown waited=${w}s status=${st} shell_front_mC=${sf}"
}
awake() {
  [ "$NOWAKE" = 1 ] && return 0
  settings put system screen_off_timeout 1800000
  dumpsys power | grep -q 'mWakefulness=Awake' || { input keyevent KEYCODE_WAKEUP; sleep 1; }
  if [ -n "$PIN" ] && dumpsys window | grep -q 'isKeyguardShowing=true'; then
    input swipe 540 1900 540 700 200; sleep 1; input text "$PIN"; input keyevent 66; sleep 2
  fi
}
trap 'rm -f $H/.phone_busy; [ "$NOWAKE" = 1 ] || input keyevent KEYCODE_SLEEP' EXIT
batt() { dumpsys battery | grep -m1 ' level:' | tr -dc 0-9; }
echo "binary=$BIN md5=$(md5sum $H/$BIN/bmoe-cli | cut -d' ' -f1) q4=$(ls -l $Q4 | awk '{print $5}') q8=$(ls -l $Q8 2>/dev/null | awk '{print $5}')" >> "$O/log.txt"
flags() {  # arm -> model and flags
  case $1 in
    REF)   echo "$Q8 $STACK $PRED" ;;
    FLOOR) echo "$Q4 $STACK $PRED" ;;
    T7)    echo "$Q4 $STACK $PRED --n-expert-used 7" ;;
    T6)    echo "$Q4 $STACK $PRED --n-expert-used 6" ;;
    RA1)   echo "$Q4 $STACK --route-ahead 1" ;;
    DC05)  echo "$Q4 $STACK $PRED --drop-cold-experts 0.5" ;;
    DC10)  echo "$Q4 $STACK $PRED --drop-cold-experts 1.0" ;;
    SUB)   echo "$Q4 $STACK $PRED --expert-substitute 0.15" ;;
  esac
}
run() {  # tag arm extra...
  tag=$1; arm=$2; shift 2
  b=$(batt); if [ "${b:-0}" -lt 25 ]; then echo "STOP battery ${b}% before $tag" | tee -a "$O/log.txt"; exit 4; fi
  cd_=$(cooldown); g=$(thermal_wait 30); mr=$(mem_ready 6500 120); awake
  ws=$(dumpsys power | grep -m1 'mWakefulness=' | cut -d= -f2)
  set -- $(flags $arm) "$@"
  m=$1; shift
  echo "=== $tag $(date +%H:%M:%S) batt=${b}% wake_start=${ws} $mr $cd_" | tee -a "$O/log.txt"
  ( cd $H/$BIN && LD_LIBRARY_PATH=. ./bmoe-cli -m $m "$@" > "$O/$tag.out" 2> "$O/$tag.err" )
  echo "exit=$? AFTER $(thermal_state) $(grep -hE '^ppl:|^ppl-kl:|^ppl-policy|perplexity failed' "$O/$tag.out" "$O/$tag.err" | tr '\n' ' ')" | tee -a "$O/log.txt"
}
if [ "$MODE" = ident ]; then
  run FLOOR_dump FLOOR $KLA --ppl-dump $O/floor.bkl
  run FLOOR_self FLOOR $KLA --ppl-ref $O/floor.bkl --ppl-kl-out $O/FLOOR_self.kl
  rm -f $O/floor.bkl
elif [ "$MODE" = klresume ]; then
  [ -s $O/ref.bkl ] || { echo "FATAL no ref.bkl in $O" | tee -a "$O/log.txt"; exit 5; }
  echo "klresume $(date +%H:%M:%S)" >> "$O/log.txt"
  for a in FLOOR T7 T6 RA1 DC05 DC10 SUB; do
    grep -q '^ppl-kl:' $O/$a.out 2>/dev/null && continue
    run $a $a $KLA --ppl-ref $O/ref.bkl --ppl-kl-out $O/$a.kl
  done
  rm -f $O/ref.bkl
elif [ "$MODE" = kl ]; then
  run REF REF $KLA --ppl-dump $O/ref.bkl
  [ -s $O/ref.bkl ] || { echo "FATAL no reference dump" | tee -a "$O/log.txt"; exit 5; }
  for a in FLOOR T7 T6 RA1 DC05 DC10 SUB; do run $a $a $KLA --ppl-ref $O/ref.bkl --ppl-kl-out $O/$a.kl; done
  rm -f $O/ref.bkl   # 200+ MB; the per-token KL files and logs are the record
else
  for a in $MC_ARMS; do
    st=""; case $a in DC05|DC10|SUB) st="--ppl-step";; esac
    run mc_$a $a -c 4096 --ppl-list $E/mmlu_list_phone.txt --ppl-choices " A, B, C, D" --ppl-skip 0 $st
  done
fi
echo "done $(date)" | tee "$O/DONE"
