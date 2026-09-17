#!/system/bin/sh
# phone_queue.sh — runs the remaining 2026-09-17 campaigns on the phone without the laptop:
# waits for the running verify-cost campaign, then GPU correctness v2, then the rotated confirmation.
H=/data/local/tmp/moe-stream
cd $H || exit 1
until ls $H/bmoe_verify_2*/DONE >/dev/null 2>&1; do sleep 30; done
echo "gpu2 start $(date)" >> $H/phone_queue.log
chmod 755 $H/ocl-stream2/*
sh gpu_stream_check2.sh > gpu2_nohup.log 2>&1 < /dev/null
echo "pin2 start $(date)" >> $H/phone_queue.log
sh bmoe_pin2.sh 4 > bmoe_pin2_nohup.log 2>&1 < /dev/null
echo "ALL DONE $(date)" >> $H/phone_queue.log
