#!/data/data/com.termux/files/usr/bin/sh
# phone_campaign_cpu.sh — the second on-device campaign of 2026-09-16.
#
# Why: in the first campaign, OLMoE Q4_0's steady-state decode through stock llama.cpp
# varied 4.6x between repeats while its flash bytes per token did not change, so flash
# traffic does not explain the variance. llm_decode_probe.sh now also samples the
# engine's CPU time (utime+stime) and the minimum MemAvailable during each run, so a
# slow run can be classified as descheduled (low CPU share) or stalled under memory
# pressure (low MemAvailable, high CPU share) — rather than guessed.
#
#   tmux new-session -d -s moe2 'sh ~/moe/phone_campaign_cpu.sh > ~/moe/campaign2.log 2>&1'
set -u
cd ~/moe
O=~/moe/out2_$(date +%Y%m%d_%H%M)
mkdir -p "$O"
echo "campaign2 start $(date)" | tee "$O/START"
P="sh ./llm_decode_probe.sh"
OL=models/olmoe-1b-7b-0924-q4_0.gguf
GR=models/granite-3.1-1b-a400m-instruct-Q4_0.gguf
$P $GR 4 "64 320" 2 "$O/granite_warm_t4" warm
$P $OL 4 "64 320" 3 "$O/olmoe_cold" cold
$P $OL 4 "64 320" 3 "$O/olmoe_warm" warm
$P $OL 2 "64 320" 2 "$O/olmoe_warm_t2" warm
echo "campaign2 done $(date)" | tee "$O/DONE"
