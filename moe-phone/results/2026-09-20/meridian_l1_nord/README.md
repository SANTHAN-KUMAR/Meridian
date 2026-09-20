# Meridian L1 device profile — OnePlus Nord (AC2001, SM7250), 2026-09-20

Produced by `platform/impl/meridian/cli.py profile`, driving the
cross-compiled `device/{devprobe,dramprobe,memprobe,ufsbench}` binaries over
adb against a live device. See `platform/impl/STATUS.md` for what this does
and does not measure.

**Device note.** This is the OnePlus Nord used in the prior `moe-phone`
phase (`results/2026-09-19/nord/`), not the OnePlus 15R that `00_PROBLEM.md`
is written around. The device profiler is device-generic by design
(`03_INTERFACES.md`), so this does not change any code path — only the
numbers that come out.

**Regime note.** The phone was USB-charging throughout (required for adb).
Decision D4 (`02_ARCHITECTURE.md`) requires `unplugged` for a deployment-
regime measurement, so every affected field in `DeviceProfile.json` is
`provenance: prior` with a widened interval, not `measured` — see
`validity.conditions.power` in the JSON and the note in
`capability_report.md`.

## Files

| file | produced by | contents |
|---|---|---|
| `g0_static.txt` | `device/g0_probe.sh` | T0 identity, CPU topology, storage mount, thermal zones, root check |
| `devprobe.txt` | cross-compiled `device/devprobe.c` | accelerator device-node open() results, as the adb-shell process |
| `dram.csv` | cross-compiled `device/dramprobe.c` | sequential DRAM read bandwidth, 1/2/4/6/8 threads x 3 repeats |
| `ufs.csv` | cross-compiled `device/ufsbench.c` | random-read curve, buffered, 4KiB-4MiB x 1/4/8 threads |
| `ufs_direct.csv` | same | O_DIRECT confirmation at 4KiB and 1MiB, 4 threads |
| `memprobe.txt` | cross-compiled `device/memprobe.c` | anonymous-memory grant-and-hold curve to the resident plateau |
| `battery_dumpsys.txt`, `power_state.txt` | `adb shell dumpsys battery / power` | validity conditions (power source, wakefulness, battery %) |
| `DeviceProfile.json` | `meridian/profiler.py` | the assembled contract, per `03_INTERFACES.md` §1 |
| `capability_report.md` | `meridian/report.py` | the plain-language summary, per `02_ARCHITECTURE.md` §9 |

## Retraction

`DeviceProfile.json` and `capability_report.md` from this directory were
deleted: they carried fabricated uncertainty intervals (STATUS.md, D-1). The
raw artifacts above are genuine and are used as parser test fixtures. This
run was USB-charging with an unchecked foreground state; use
`../meridian_l1_nord_wireless/` instead.
