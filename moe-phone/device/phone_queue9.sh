#!/system/bin/sh
H=/data/local/tmp/moe-stream
cd $H || exit 1
until grep -q "ALL DONE 8" $H/phone_queue.log 2>/dev/null; do sleep 30; done
echo "boost start $(date)" >> $H/phone_queue.log
sh bmoe_boost.sh 3 > bmoe_boost_nohup.log 2>&1 < /dev/null
echo "ALL DONE 9 $(date)" >> $H/phone_queue.log
