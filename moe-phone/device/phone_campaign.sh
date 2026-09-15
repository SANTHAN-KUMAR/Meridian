#!/data/data/com.termux/files/usr/bin/sh
# phone_campaign.sh — the unattended on-device measurements of 2026-09-16, in order.
# Run inside tmux in Termux (screen on, Termux foregrounded, wake lock held):
#   tmux new-session -d -s moe 'sh ~/moe/phone_campaign.sh > ~/moe/campaign.log 2>&1'
#
#  1. OLMoE-1B-7B-0924 Q4_0 (3.93 GB) through unmodified llama.cpp mmap — the
#     "stock engine" baseline of POSITION.md §6. COLD = checkpoint evicted first.
#     Beyond DRAM by construction: the app's file-backed budget is 1.6-4.8 GB.
#  2. The same, WARM (whatever the page cache kept) — the steady state a user sees.
#  3. granite-3.1-1b-a400m Q4_0 (0.77 GB, 32 experts top-8), fully RESIDENT —
#     the non-flash term: decode time when every weight is already in DRAM (S11).
#  4. Prefill (pp512) on the resident model — a compute-bound rate for comparison.
#  5. ufsbench at 16/32 threads for bulk sizes — whether the 2.8 GB/s plateau of
#     G1 is a queue-depth limit (ARCHITECTURE §5b) or the device ceiling.
set -u
cd ~/moe
O=~/moe/out_$(date +%Y%m%d_%H%M)
mkdir -p "$O"
echo "campaign start $(date)" | tee "$O/START"
P="sh ./llm_decode_probe.sh"
OL=models/olmoe-1b-7b-0924-q4_0.gguf
GR=models/granite-3.1-1b-a400m-instruct-Q4_0.gguf
$P $OL 4 "64 320" 3 "$O/olmoe_cold" cold
$P $OL 4 "64 320" 2 "$O/olmoe_warm" warm
$P $GR 4 "64 320" 3 "$O/granite_warm_t4" warm
$P $GR 2 "64 320" 2 "$O/granite_warm_t2" warm
./llm/llama-bench -m $GR -p 512 -n 0 -t 4 -r 3 -o csv > "$O/granite_pp512_t4.csv" 2> "$O/granite_pp512_t4.err"
./llm/llama-bench -m $OL -p 512 -n 0 -t 4 -r 2 -o csv > "$O/olmoe_pp512_t4.csv" 2> "$O/olmoe_pp512_t4.err"
./ufsbench --file ~/moe/ufs.bin --size-mb 8192 --create --seconds 2 --repeats 3 \
    --sizes-kb 512,1024,4096 --threads 1,4,8,16,32 --patterns rand --modes direct,buffered \
    --out "$O/ufs_qd.csv" > "$O/ufs_qd.log" 2>&1
rm -f ~/moe/ufs.bin
echo "campaign done $(date)" | tee "$O/DONE"
