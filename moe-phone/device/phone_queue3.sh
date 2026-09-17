#!/system/bin/sh
# phone_queue3.sh — critical path first: GPU bisect (short, unblocks debugging), then the rotated confirmation.
# A laptop-side step pushes gpt-oss-20b between the two, while no campaign is measuring.
H=/data/local/tmp/moe-stream
cd $H || exit 1
echo "bisect start $(date)" >> $H/phone_queue.log
sh gpu_bisect.sh > gpu_bisect_nohup.log 2>&1 < /dev/null
echo "bisect done; waiting for gptoss push marker $(date)" >> $H/phone_queue.log
until [ -f $H/PUSH_DONE ]; do sleep 10; done
echo "pin2 start $(date)" >> $H/phone_queue.log
sh bmoe_pin2.sh 4 > bmoe_pin2_nohup.log 2>&1 < /dev/null
echo "ALL DONE 3 $(date)" >> $H/phone_queue.log
