#!/data/data/com.termux/files/usr/bin/sh
# phone_campaign_pin.sh — the thread-cliff FIX, swept (2026-09-16 night).
#
# Campaign 4 (rep 1) found stock llama.cpp OLMoE Q4_0 at 6.7 tok/s with 4 unpinned threads and
# 31.6 tok/s with the same 4 threads pinned to cores 4-7 (-C 0xF0 --cpu-strict 1), with a similar
# major-fault count — so placement, not faults, set the rate, and --poll 0 was catastrophic
# (~0.5 tok/s). This sweeps WHERE and HOW MANY: cores 0-5 are 3.32 GHz, 6-7 are 3.8 GHz primes.
# Interleaved by repeat; the page cache is warmed once first.
#   tmux new-session -d -s moe5 'sh ~/moe/phone_campaign_pin.sh > ~/moe/campaign5.log 2>&1'
set -u
cd ~/moe
O=~/moe/out5_$(date +%Y%m%d_%H%M); mkdir -p "$O"
echo "campaign5 start $(date)" | tee "$O/START"
OL=models/olmoe-1b-7b-0924-q4_0.gguf
P="sh ./llm_decode_probe.sh"
./llm/llama-bench -m $OL -p 0 -n 16 -t 2 -r 1 -C 0xF0 --cpu-strict 1 > /dev/null 2>&1
for r in 1 2; do
  EXTRA="-C 0xF0 --cpu-strict 1" $P $OL 4 "64 320" 1 "$O/t4_c4to7_rep$r" warm
  EXTRA="-C 0xC0 --cpu-strict 1" $P $OL 2 "64 320" 1 "$O/t2_primes_rep$r" warm
  EXTRA="-C 0xFC --cpu-strict 1" $P $OL 6 "64 320" 1 "$O/t6_c2to7_rep$r" warm
  EXTRA="-C 0xFF --cpu-strict 1" $P $OL 8 "64 320" 1 "$O/t8_all_rep$r" warm
  EXTRA="-C 0x0F --cpu-strict 1" $P $OL 4 "64 320" 1 "$O/t4_c0to3_rep$r" warm
  EXTRA=""                       $P $OL 4 "64 320" 1 "$O/t4_unpinned_rep$r" warm
done
echo "campaign5 done $(date)" | tee "$O/DONE"
