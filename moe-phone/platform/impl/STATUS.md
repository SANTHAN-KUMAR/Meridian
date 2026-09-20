# Implementation status

What exists under `platform/impl/`, what conformance level it reaches, and
what it does not do. This extends the stub registry of
[`../10_EXTENSION_POINTS.md`](../10_EXTENSION_POINTS.md) section 4 — an
unregistered stub is a false claim (CLAUDE.md section 7.5), so everything
this code skips is listed here, not silently omitted.

## Conformance reached: **L1 — profiled runner**, incomplete

Per [`../10_EXTENSION_POINTS.md`](../10_EXTENSION_POINTS.md) section 1:

> L1 = L0 (loads a model, runs it) + the device profiler at T0/T1 + a
> capability report.

**L0 is not built.** There is no engine, no session protocol, no app shell.
This pass builds the layer *below* L0 — the device profiler — because it is
the cheapest thing to build honestly and the thing every other layer depends
on, and because building an inference engine or an Android app in one pass
without the ability to test either on a UI would itself violate this
project's "verify before claiming done" rule.

So this is more precisely: **T0+T1 device profiling and a capability
report, with no engine underneath it yet.** It satisfies L1's *profiling*
requirement and none of L0's *running* requirement.

## What was actually measured, on a real device, over adb

Device: OnePlus Nord (AC2001), Qualcomm SM7250, Android 12. Not the OnePlus
15R named in `00_PROBLEM.md` — the 15R was unavailable this session; the
Nord already has prior calibration history in this repository
(`results/2026-09-19/nord/`), and the architecture is device-generic, so the
choice of device does not change any code path (`03_INTERFACES.md` — the
whole point of `DeviceProfile` is that layer 0 is where device specifics
live).

- **T0 static inventory** (`meridian/probes/static_inventory.py`, driving
  the existing `device/g0_probe.sh` + cross-compiled `device/devprobe.c`):
  identity, CPU topology (3 clusters resolved from `/proc/cpuinfo` part IDs
  and `cpuinfo_max_freq`), storage mount/filesystem resolution, accelerator
  device-node presence.
- **T1 fast probes**, cross-compiled with the NDK
  (`platform/impl/build_probes.sh`) from the existing, already-validated C
  sources in `device/`:
  - `dramprobe.c` → `memory.dram_read_gbps` (12.4 GB/s median, 3 repeats x
    5 thread counts; 8 of 15 rows at 8 threads rejected as descheduled by
    the probe's own `min_cpu_over_wall` check — expected on an unpinned
    8-core device with no `--cpus` mask given, since thread placement is a
    separate, unimplemented probe, see below).
  - `memprobe.c` (anonymous mode) → `memory.grantable_quiesced` (8194 MiB
    plateau before `MemAvailable` hit its floor).
  - `ufsbench.c` → `storage.random_read` curve and `efficient_request_size`
    (four sizes x three thread counts, buffered + one direct-I/O
    confirmation pair).
- **ValidityGuard** (`meridian/validity.py`): reads `dumpsys battery` and
  `dumpsys power` to attach real `ValidityConditions` to every measurement,
  and discovered a genuine regime violation — the phone was USB-charging
  throughout this session (needed for the adb connection), which is **not**
  the deployment regime decision D4 requires (`unplugged`). The profiler
  downgrades every affected `Measured` from `provenance: measured` to
  `provenance: prior` with a widened interval for exactly this reason,
  rather than reporting a charging-state number as if it were the sustained
  unplugged operating point. **Re-running this profiler on battery power
  would upgrade these fields to `measured`** — that is the one substantive
  follow-up this document recommends.

Raw artifacts: [`../../results/2026-09-20/meridian_l1_nord/`](../../results/2026-09-20/meridian_l1_nord/)
(every CSV/text file the parsers above read, plus the assembled
`DeviceProfile.json` and `capability_report.md`).

## Registered stubs (extends `10_EXTENSION_POINTS.md` section 4)

| id | what | current state | removed when |
|---|---|---|---|
| `PL-E10` | L0 engine + session protocol | does not exist in `platform/impl/` | an engine binary is wired to `LocalApi` per `06_EXECUTION_ENGINE.md` §11 |
| `PL-E11` | per-cluster `matmul_gbps` (T2 compute probe) | `Measured.unknown` in every `CpuCluster`; requires `host/app/ggml_matmul_bench.cpp`, which needs a ggml build this pass did not attempt | the ggml-based compute probe is cross-compiled and wired in |
| `PL-E12` | thread-placement A/B (`recommended_compute_mask`/`recommended_io_mask`) | `Measured.unknown`; the 4.0 vs 30.1 tok/s pinned/unpinned gap in `00_PROBLEM.md` §4 was never re-measured on the Nord | a same-process pinned/unpinned A/B runs and picks a mask |
| `PL-E13` | per-accelerator decode/prefill rate, dispatch overhead | `Measured.unknown` for CPU/GPU/NPU; requires an engine (`PL-E10`) | T2 profiling ships |
| `PL-E14` | thermal derate curve, time-to-throttle, recovery (T3) | not attempted; would need a sustained decode load this pass has no engine to generate | T3 profiling ships, per `04_DEVICE_PROFILING.md` §3.5 |
| `PL-E15` | thermal-status OS field in `ValidityConditions` | left `None`; only wakefulness/power/battery are read from `dumpsys` | a thermal-status API call is added to `validity.py` |
| `PL-E16` | ISA feature detection (`i8mm`, `dotprod`, etc.) per cluster | `isa_features: []` in every `CpuCluster` | `/proc/cpuinfo` `Features:` line or `getauxval(AT_HWCAP)` is parsed |
| `PL-E17` | sequential write throughput | `Measured.unknown`; `ufsbench` was only run in random-read mode | a `--patterns seq --modes buffered` sweep is added to the T1 run |
| `PL-E18` | swap/zram fault cost | `Measured.unknown`; `memprobe --file` mode (file-backed residency) was not run | the file-backed regime is added and its fault cost measured |
| `PL-S2` | foreground-coexistence memory grant (`grantable_foreground`) | still never measured anywhere in this project, per `01_RESEARCH.md`/`10_EXTENSION_POINTS.md` — this pass does not change that, since it needs a real target app foregrounded and no app exists | experiment X1 runs, per the docs' own priority order in §7 |
| `PL-E19` | per-rail power | not readable by an unprivileged process on this device either (`battery/current_now` etc. all `DENIED_OR_ABSENT` in `g0_static.txt`) — consistent with `04_DEVICE_PROFILING.md` §7's documented limitation on the 15R | a device grants rails, or a validated battery-counter method is built (`PL-S5`) |
| ~~`PL-E20`~~ | thermal-zone data quality | **fixed**: `validity.valid_thermal_zones()` ports `ufsbench.c`'s own `max_thermal_c()` filter (magnitude + name exclusion) and applies it to every zone in `g0_static.txt`, not one hand-picked name. `ufsbench`'s in-process picks still surface `mmw-pa4-usr` (81.5 °C, unchanging across every row this session) because that zone genuinely passes the magnitude filter — it is real millidegree-range data from a plausible-sounding name, just not a useful proxy for skin temperature. The filter removes clearly-invalid sensors; it does not and cannot validate that a passing zone tracks anything physically meaningful, which only a T3 sustained-load run against a reference could do | T3 profiling ships and cross-checks zone readings against measured throttle behaviour |

## What is deliberately NOT a stub

- The four cross-compiled probes (`dramprobe`, `memprobe`, `ufsbench`,
  `devprobe`) are unmodified from `device/*.c` — their own correctness
  rules (descheduling detection, zram-residency refusal, permutation-based
  random reads, O_DIRECT fallback detection) are inherited intact, not
  reimplemented and possibly weakened.
- `contracts.py`'s `Measured` dataclass enforces its own invariant in
  `__post_init__`: a non-`measured` value must carry an interval. This is a
  runtime check, not a docstring promise — the profiler would crash, not
  silently emit a bad `DeviceProfile`, if a probe tried to report a `prior`
  point value with no interval.

## How to reproduce

```sh
NDK=$HOME/Android/Sdk/ndk/<version> sh platform/impl/build_probes.sh /tmp/meridian_bin
cd platform/impl
python3 -m meridian.cli profile --bin-dir /tmp/meridian_bin \
    --out-dir ../../results/$(date +%F)/meridian_l1_<device>
```

Requires an adb-authorized device attached. Rebuilding the profile from
already-captured artifacts (no device needed): add `--from-existing` and
drop `--bin-dir`.
