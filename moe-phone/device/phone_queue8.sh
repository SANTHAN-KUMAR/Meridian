#!/system/bin/sh
H=/data/local/tmp/moe-stream
cd $H || exit 1
until grep -q "ALL DONE 7" $H/phone_queue.log 2>/dev/null; do sleep 30; done
echo "arena start $(date)" >> $H/phone_queue.log
sh bmoe_arena.sh 3 > bmoe_arena_nohup.log 2>&1 < /dev/null
echo "ALL DONE 8 $(date)" >> $H/phone_queue.log
