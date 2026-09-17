#!/system/bin/sh
H=/data/local/tmp/moe-stream
cd $H || exit 1
until grep -q "ALL DONE 10" $H/phone_queue.log 2>/dev/null; do sleep 30; done
mv $H/bmoe_boost_20260917_2231 $H/bmoe_boost_20260917_2231_FAILED_argsplit 2>/dev/null
echo "boost rerun start $(date)" >> $H/phone_queue.log
sh bmoe_boost.sh 3 > bmoe_boost_nohup.log 2>&1 < /dev/null
echo "ALL DONE 11 $(date)" >> $H/phone_queue.log
