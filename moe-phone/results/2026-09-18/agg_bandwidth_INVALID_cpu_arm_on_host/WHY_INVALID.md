# Why this directory is invalid

Two defects. Either one voids every row.

1. **The CPU arm never ran on the phone.** `host/agg_bandwidth.sh`'s `shell_cpu` ran
   `cd /data/local/tmp/moe-stream/ocl && ./llama-bench ...` on the laptop, because the line had no `adb shell`.
   Every `solo_cpu_*.txt` and `pair_*_cpu_*.txt` holds only `cd: ... No such file or directory` / `exit=1`.
   So each "pair" was one device running alone, and the question "do two devices add bandwidth?" was
   never asked.
2. **Two drivers wrote into this directory.** An orphaned copy of the driver ran from START 13:40 to
   DONE 14:35:36, left behind by an earlier chain. The chain_next copy ran from START 14:34:30 to
   DONE 14:42:24. They overlapped, and both wrote the same file names, so the per-row files are a mix
   of the two runs. See `driver.log`.

The orphan also launched `llama-bench` in `com.moephone.npu2` while `bmoe_slru_20260918_1408` was
running. That is why the SLRU decode rates are not interpreted; see `../bmoe_slru/DECODE_NOT_INTERPRETED.md`.

**Fix:** `shell_cpu` now runs through `adb shell` and logs `FAILED <tag>` when no tg row comes back. The
driver takes an flock and refuses to run if another copy holds it. Re-run queued in `host/chain_after.sh`.
No number from this directory is quoted anywhere.
