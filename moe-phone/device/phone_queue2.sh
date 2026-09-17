#!/system/bin/sh
# phone_queue2.sh — rerun of the rotated confirmation after the doze contamination, then the GPU bisect.
H=/data/local/tmp/moe-stream
cd $H || exit 1
echo "pin2 start $(date)" >> $H/phone_queue.log
sh bmoe_pin2.sh 4 > bmoe_pin2_nohup.log 2>&1 < /dev/null
echo "bisect start $(date)" >> $H/phone_queue.log
sh gpu_bisect.sh > gpu_bisect_nohup.log 2>&1 < /dev/null
echo "ALL DONE 2 $(date)" >> $H/phone_queue.log
