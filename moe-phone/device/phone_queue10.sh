#!/system/bin/sh
H=/data/local/tmp/moe-stream
cd $H || exit 1
until grep -q "ALL DONE 9" $H/phone_queue.log 2>/dev/null; do sleep 30; done
echo "profile start $(date)" >> $H/phone_queue.log
sh bmoe_profile.sh > bmoe_profile_nohup.log 2>&1 < /dev/null
echo "ALL DONE 10 $(date)" >> $H/phone_queue.log
