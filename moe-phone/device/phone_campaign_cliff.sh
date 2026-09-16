#!/data/data/com.termux/files/usr/bin/sh
# phone_campaign_cliff.sh — diagnose and fix the stock-llama.cpp thread cliff (2026-09-16).
#
# Campaign 3 found warm OLMoE Q4_0 at 20.8 / 17.4 / 2.3 / 2.1 tok/s at 1-4 threads, with no flash
# traffic to explain it. Two mechanisms fit: (a) worker threads SPIN at op barriers (llama.cpp
# --poll, default 50), so one blocked or preempted worker makes the rest burn CPU and starve the
# kernel's reclaim and I/O threads; (b) the scheduler migrates workers across the 6+2 cores.
# Each cell switches one lever; cells are interleaved by repeat so device drift cannot pose as an
# effect, and every run logs major faults. The page cache is warmed once before the first cell.
#   tmux new-session -d -s moe4 'sh ~/moe/phone_campaign_cliff.sh > ~/moe/campaign4.log 2>&1'
set -u
cd ~/moe
O=~/moe/out4_$(date +%Y%m%d_%H%M); mkdir -p "$O"
echo "campaign4 start $(date)" | tee "$O/START"
OL=models/olmoe-1b-7b-0924-q4_0.gguf
P="sh ./llm_decode_probe.sh"
./llm/llama-bench -m $OL -p 0 -n 16 -t 2 -r 1 > /dev/null 2>&1
for r in 1 2; do
  EXTRA=""                              $P $OL 2 "64 320" 1 "$O/t2_default_rep$r" warm
  EXTRA=""                              $P $OL 4 "64 320" 1 "$O/t4_default_rep$r" warm
  EXTRA="--poll 0"                      $P $OL 4 "64 320" 1 "$O/t4_poll0_rep$r" warm
  EXTRA="-C 0xF0 --cpu-strict 1"        $P $OL 4 "64 320" 1 "$O/t4_mask_rep$r" warm
  EXTRA="--poll 0 -C 0xF0 --cpu-strict 1" $P $OL 4 "64 320" 1 "$O/t4_poll0_mask_rep$r" warm
  EXTRA="--poll 0"                      $P $OL 3 "64 320" 1 "$O/t3_poll0_rep$r" warm
done
echo "campaign4 done $(date)" | tee "$O/DONE"
