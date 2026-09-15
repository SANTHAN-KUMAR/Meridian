#!/data/data/com.termux/files/usr/bin/sh
# phone_campaign_threads.sh — the third on-device campaign of 2026-09-16: a thread sweep.
#
# Why: in campaign 2, warm OLMoE Q4_0 decoded at ~30 tok/s steady state with 2 threads and
# ~8 tok/s with 4, in the same memory state and with near-zero flash bytes per token. Two
# pairs are not a finding. This sweeps 1-4 threads for OLMoE (warm) and for the resident
# reference granite, INTERLEAVED by repeat, so a drift in device state over the campaign
# cannot masquerade as a thread effect. One (threads, repeat) cell per output directory.
#
#   tmux new-session -d -s moe3 'sh ~/moe/phone_campaign_threads.sh > ~/moe/campaign3.log 2>&1'
set -u
cd ~/moe
O=~/moe/out3_$(date +%Y%m%d_%H%M)
mkdir -p "$O"
echo "campaign3 start $(date)" | tee "$O/START"
P="sh ./llm_decode_probe.sh"
OL=models/olmoe-1b-7b-0924-q4_0.gguf
GR=models/granite-3.1-1b-a400m-instruct-Q4_0.gguf
for r in 1 2 3; do
  for t in 1 2 3 4; do
    $P $OL $t "64 320" 1 "$O/olmoe_warm_t${t}_rep${r}" warm
  done
done
for r in 1 2; do
  for t in 1 2 3 4; do
    $P $GR $t "64 320" 1 "$O/granite_warm_t${t}_rep${r}" warm
  done
done
echo "campaign3 done $(date)" | tee "$O/DONE"
