#!/system/bin/sh
# phone_queue7.sh — the remainder of queue6 after an NPU diagnostic slot: defer -> lanes -> cache
H=/data/local/tmp/moe-stream
cd $H || exit 1
echo "defer start $(date)" >> $H/phone_queue.log
sh bmoe_defer.sh 3 > bmoe_defer_nohup.log 2>&1 < /dev/null
echo "lanes start $(date)" >> $H/phone_queue.log
sh bmoe_lanes.sh 3 > bmoe_lanes_nohup.log 2>&1 < /dev/null
echo "cache start $(date)" >> $H/phone_queue.log
sh bmoe_cache.sh 3 > bmoe_cache_nohup.log 2>&1 < /dev/null
echo "ALL DONE 7 $(date)" >> $H/phone_queue.log
