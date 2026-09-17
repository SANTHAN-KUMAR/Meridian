# The phone stopped answering at 02:38 — what happened, and what to do

## What to do first (one of these, takes a minute)

The laptop is already waiting: `moe-phone/host/chain_resume.sh` is running and blocks until `adb`
answers, then restores the phone's state and runs the rest of the night's queue by itself. So the only
thing needed is to make the phone reachable again:

* **plug the phone in over USB** and run `adb tcpip 5555` (then the cable can come out), **or**
* **turn Wireless debugging off and on** in Developer options and, if it hands out a different port,
  run `adb connect 192.168.0.65:<port>` and restart the resume chain with
  `ANDROID_SERIAL=192.168.0.65:<port> BMOE_PIN=147853 bash moe-phone/host/chain_resume.sh &`.

Nothing was lost: `/data/local/tmp/moe-stream` is persistent, every completed campaign is already
pulled and committed, and the resume chain re-pushes the scripts before it starts.

## What happened

The ZRAM campaign (`device/bmoe_zram.sh`) asks for an expert cache deliberately larger than RAM, so
that the overflow lands in the 12.6 GB ZRAM swap instead of being re-read from flash. Its third cell
requested **10000 MiB on an 11366 MiB device**. During that cell:

* ping latency to the phone rose from ~55 ms to 90–885 ms (the machine was thrashing),
* then latency returned to normal, but TCP port 5555 was **refused**, no adb service was advertised over
  mDNS, and no other adb port was open.

A killed-and-restarted `adbd` comes back on the same port, because `service.adb.tcp.port` survives it.
A **reboot** does not: that property is non-persistent, so after a reboot the port is simply gone —
which is exactly the symptom. So the most likely reading is that the phone rebooted under the memory
pressure of that cell, and the least it can be is that the device became unreachable and stayed that
way.

## This is a result, not just an accident

It was pre-registered. `PREREG_zram_and_aggregate.md`, Q1, lists among the possible outcomes:

> `zram*` rows fail (LMK kill, non-zero exit) — the configuration is not runnable, which is itself the
> answer at that budget → report the failure rate per arm, do not substitute a smaller budget and call
> it the same arm.

So: **at a 10000 MiB budget on this device the configuration is not runnable** — it does not merely run
slowly, it takes the machine down. The cell is retired rather than repeated, and `bmoe_zram.sh` now
computes a cap from `/proc/meminfo` at run time (MemTotal − the 1024 MiB floor the engine is told to
leave free − the ~1.5 GB of dense weights and context that live outside the cache) and **skips any cell
above it**, printing the skip. On this phone that cap is 8806 MiB, so the surviving cells are 5000
(control), 7000 and 8500. The retired cell is named in the script with the reason, so the next reader
does not re-run it by accident.

Note what is *not* being claimed: this says nothing yet about whether a ZRAM-backed cache is faster or
slower at a runnable size. The one row that completed before the phone went down was the **control**
(`base5000`), and it reproduced the committed baseline almost exactly — 6.187 tok/s at an 85.3% hit rate
and 119.80 MiB/token read, against 6.199 / 85.3% / 119.80 in `results/2026-09-17/bmoe_cache.json`. That
agreement is worth having on its own: the baseline is stable across days and across a 17 GB model copy.

## What the resume chain will run, in this order

1. `bmoe_order3.sh` — the **balanced** (ABBA) expert-order A/B, because the two earlier campaigns
   disagreed and both had an unbalanced rotation (see `bmoe_order/FINDING.md`).
2. `bmoe_arena2.sh` — `--slot-arena` re-measured at the current operating point, sampling MemAvailable
   and SwapFree every 2 s. The first campaign showed it cutting cache management from 34 ms to 2 ms per
   token — the largest single saving found in this project — while losing overall to a 14 ms rise in
   compute and a 24 ms rise in stall. The memory hypothesis for that penalty is what the sampling tests.
3. `bmoe_zram.sh` — capped as above.
4. `matmul_sweep.sh` (after reinstalling the app with the OpenCL fix), then `agg_bandwidth.sh`,
   `thread_sweep.sh`, `app_vs_shell.sh`, and finally `bmoe_mem.sh`.
