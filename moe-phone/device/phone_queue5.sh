#!/system/bin/sh
# phone_queue5.sh — after phone_queue4 (gpu_speed, rotated confirmation): clean retests of two failed levers.
H=/data/local/tmp/moe-stream
cd $H || exit 1
until grep -q "ALL DONE 4" $H/phone_queue.log 2>/dev/null; do sleep 30; done
echo "verify_cost clean start $(date)" >> $H/phone_queue.log
sh bmoe_verify_cost.sh 3 > bmoe_verify_nohup.log 2>&1 < /dev/null
echo "repack_pin start $(date)" >> $H/phone_queue.log
sh bmoe_repack_pin.sh 3 > bmoe_repack_pin_nohup.log 2>&1 < /dev/null
echo "ALL DONE 5 $(date)" >> $H/phone_queue.log
