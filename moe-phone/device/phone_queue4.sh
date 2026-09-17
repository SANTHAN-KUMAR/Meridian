#!/system/bin/sh
H=/data/local/tmp/moe-stream
cd $H || exit 1
until [ -f $H/PUSH_DONE ]; do sleep 10; done
echo "gpu_speed start $(date)" >> $H/phone_queue.log
sh gpu_speed.sh 2 > gpu_speed_nohup.log 2>&1 < /dev/null
echo "pin2 start $(date)" >> $H/phone_queue.log
sh bmoe_pin2.sh 4 > bmoe_pin2_nohup.log 2>&1 < /dev/null
echo "ALL DONE 4 $(date)" >> $H/phone_queue.log
