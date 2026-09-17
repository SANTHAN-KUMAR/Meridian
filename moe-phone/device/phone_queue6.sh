#!/system/bin/sh
H=/data/local/tmp/moe-stream
cd $H || exit 1
until grep -q "ALL DONE 5" $H/phone_queue.log 2>/dev/null; do sleep 30; done
echo "npu_compare (fixed env) start $(date)" >> $H/phone_queue.log
sh npu_compare.sh 3 > npu_compare_nohup.log 2>&1 < /dev/null
echo "defer start $(date)" >> $H/phone_queue.log
sh bmoe_defer.sh 3 > bmoe_defer_nohup.log 2>&1 < /dev/null
echo "lanes start $(date)" >> $H/phone_queue.log
sh bmoe_lanes.sh 3 > bmoe_lanes_nohup.log 2>&1 < /dev/null
echo "cache start $(date)" >> $H/phone_queue.log
sh bmoe_cache.sh 3 > bmoe_cache_nohup.log 2>&1 < /dev/null
echo "ALL DONE 6 $(date)" >> $H/phone_queue.log
