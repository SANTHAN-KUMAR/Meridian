#!/bin/sh
# g0_probe.sh — gate G0: record the device facts every later measurement depends on.
#
# Runs in Termux (as an ordinary app — the regime we care about) or in `adb shell`.
# Read-only: it changes nothing on the phone. Lines that report DENIED or ABSENT
# are themselves findings (e.g. power rails not readable without root) and are kept.
#
#   sh g0_probe.sh > g0_$(date +%Y%m%d).txt
#
# Then copy the file to moe-phone/results/<date>/ on the computer and run
#   python moe-phone/gates/g0_summarize.py <file>
#
# This file must be checked out with LF endings (see /.gitattributes): Android's
# /bin/sh fails on CRLF at the first blank line and at every `\` continuation.

say() { printf '\n== %s ==\n' "$1"; }

# try CMD [MAXLINES] — run CMD and distinguish three outcomes that the usual
# `cmd || echo denied` conflates:
#   FAILED  non-zero exit — denied, or the tool does not exist
#   EMPTY   exit 0 with no output — the query ran and matched nothing
#   output  the answer
# Truncation happens HERE, not as `cmd | head` inside CMD: a trailing pipe makes
# the pipeline's exit status head's, so the failure of the command that matters
# is silently reported as success. That masking hid `dumpsys powerstats` being
# unavailable to an app in the 2026-09-14 run of the previous version.
try() {
  _max=${2:-200}
  _out=$(sh -c "$1" 2>&1); _rc=$?
  if [ "$_rc" -ne 0 ]; then
    printf 'FAILED rc=%s: %s\n' "$_rc" "$1"
    [ -n "$_out" ] && printf '  msg: %s\n' "$_out"
  elif [ -z "$_out" ]; then
    printf 'EMPTY rc=0: %s\n' "$1"
  else
    printf '%s\n' "$_out" | head -n "$_max"
  fi
}

# kv PATH LABEL — print LABEL=<contents> or LABEL=DENIED_OR_ABSENT. Never blank:
# a missing line and a denied read must not look the same in the artifact.
kv() {
  _v=$(cat "$1" 2>/dev/null)
  if [ -n "$_v" ]; then printf '%s=%s\n' "$2" "$_v"; else printf '%s=DENIED_OR_ABSENT\n' "$2"; fi
}

say "identity"
for p in ro.product.manufacturer ro.product.model ro.soc.manufacturer ro.soc.model \
         ro.board.platform ro.build.version.release ro.build.version.sdk ro.build.fingerprint; do
  printf '%s=%s\n' "$p" "$(getprop $p 2>/dev/null)"
done
try "uname -a"
echo "context: $(id 2>/dev/null)"
echo "probe_epoch_s: $(date +%s)"

say "memory"
try "grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree|^Cached' /proc/meminfo"
kv /proc/self/oom_score_adj oom_score_adj
kv /sys/block/zram0/disksize zram0_disksize_bytes
printf 'ro.lmk.use_minfree_levels=%s\n' "$(getprop ro.lmk.use_minfree_levels 2>/dev/null)"
printf 'ro.config.low_ram=%s\n' "$(getprop ro.config.low_ram 2>/dev/null)"
printf 'dalvik.vm.heapgrowthlimit=%s\n' "$(getprop dalvik.vm.heapgrowthlimit 2>/dev/null)"

say "storage"
# /proc/mounts, not `mount`: Termux's PATH has no mount(8), and the previous
# version therefore reported only "mount: not found" where the f2fs mount
# options (inlinecrypt, fsync_mode, read-ahead) are the fact we need.
try "grep -E ' (/data|/data/user/0) ' /proc/mounts"
# Resolve the device that actually backs this process's storage. A hardcoded
# sda/sdc/dm-0 list (this script until 2026-09-14) collected NOTHING on the
# OnePlus 15R, whose /data is dm-79 over UFS LUNs sda..sdf, and said so nowhere.
DATA_MOUNT=/data/user/0
[ -d "$DATA_MOUNT" ] || DATA_MOUNT=${HOME:-/data/local/tmp}
DATA_DEV=$(df "$DATA_MOUNT" 2>/dev/null | awk 'NR==2{print $1}')
printf 'data_mount=%s\ndata_device=%s\n' "$DATA_MOUNT" "${DATA_DEV:-UNKNOWN}"
try "df -h $DATA_MOUNT"
DEVS=$(basename "${DATA_DEV:-none}")
for d in /sys/block/sd[a-z] /sys/block/dm-0; do
  [ -e "$d" ] && DEVS="$DEVS $(basename $d)"
done
for b in $DEVS; do
  [ "$b" = none ] && continue
  for q in scheduler nr_requests read_ahead_kb max_sectors_kb rotational logical_block_size; do
    kv "/sys/block/$b/queue/$q" "/sys/block/$b/queue/$q"
  done
done
try "ls -d /sys/devices/platform/soc/*ufs* /sys/class/scsi_host/* 2>/dev/null" 20

say "npu (Hexagon) presence"
for f in /vendor/lib64/libcdsprpc.so /system/lib64/libcdsprpc.so /vendor/lib64/libadsprpc.so; do
  [ -e "$f" ] && echo "present: $f" || echo "absent: $f"
done
# Skeleton library names carry the HTP version (e.g. ...V81Skel.so); list what the
# image ships. An app usually cannot read /vendor/dsp — EMPTY/FAILED says which.
try "ls /vendor/lib/rfsa/adsp /vendor/dsp /vendor/dsp/cdsp 2>/dev/null" 40
try "ls -l /dev/fastrpc-cdsp /dev/fastrpc-adsp /dev/adsprpc-smd 2>/dev/null" 10
# NSP thermal zones are readable without privilege and are independent evidence
# that Hexagon HVX/HMX blocks exist on this die.
try "grep -l -E 'nsph' /sys/class/thermal/thermal_zone*/type 2>/dev/null" 20

say "gpu"
try "ls /sys/class/kgsl/kgsl-3d0/" 5
kv /sys/class/kgsl/kgsl-3d0/gpu_model gpu_model

say "power measurement"
for f in current_now voltage_now charge_counter status capacity; do
  kv "/sys/class/power_supply/battery/$f" "battery/$f"
done
try "ls /sys/bus/iio/devices/ 2>/dev/null" 20
try "dumpsys powerstats" 40

say "thermal zones"
for z in /sys/class/thermal/thermal_zone*; do
  printf '%s %s %s\n' "$(basename $z)" "$(cat $z/type 2>/dev/null)" "$(cat $z/temp 2>/dev/null)"
done | head -120

say "cpu"
try "grep -E 'processor|CPU part' /proc/cpuinfo | paste - -" 16
for c in /sys/devices/system/cpu/cpu[0-9]*; do
  printf '%s max_khz=%s\n' "$(basename $c)" "$(cat $c/cpufreq/cpuinfo_max_freq 2>/dev/null)"
done

say "root"
# `command -v su` is NOT a root test: Termux ships its own su shim in $PREFIX/bin,
# so the previous version of this script reported "su present" on a locked,
# unrooted, verified-boot-green OnePlus 15R. Only an su that actually returns
# uid 0 is evidence of root.
if command -v su >/dev/null 2>&1; then
  SU_PATH=$(command -v su)
  SU_UID=$(su -c 'id -u' 2>/dev/null | tr -dc '0-9')
  if [ "$SU_UID" = "0" ]; then
    echo "ROOT: su at $SU_PATH returned uid 0"
  else
    echo "NOT_ROOT: su at $SU_PATH exists but 'su -c id -u' gave '${SU_UID:-<nothing>}'"
  fi
else
  echo "NOT_ROOT: no su on PATH"
fi
printf 'ro.boot.verifiedbootstate=%s\n' "$(getprop ro.boot.verifiedbootstate 2>/dev/null)"
printf 'ro.boot.flash.locked=%s\n' "$(getprop ro.boot.flash.locked 2>/dev/null)"
printf 'ro.secure=%s\n' "$(getprop ro.secure 2>/dev/null)"

say "done"
