# Implementation status

What exists under `platform/impl/`, what conformance level it reaches, and
what it does not do. This extends the stub registry of
[`../10_EXTENSION_POINTS.md`](../10_EXTENSION_POINTS.md) section 4 — an
unregistered stub is a false claim (CLAUDE.md section 7.5), so everything
this code skips is listed here, not silently omitted.

## Conformance reached (per `../10_EXTENSION_POINTS.md` section 1)

**Delivered: `app/dist/meridian.apk`** (build: `app/build.sh`; details: [`app/README.md`](app/README.md)). One arm64 APK containing the
engine, the probe suite, the planner and the agent runtime. Models are downloaded or imported, never bundled.

| level | state | what exists |
|---|---|---|
| **L0 runner** | done | research engine (`bmoe-cli`, session protocol) + resident tier + Android app shell + foreground service; typed refusals (`EngineUnsupported`, `ArchitectureUnsupported`, `Infeasible`, `IntegrityFailed`) |
| **L1 profiled runner** | done | on-device T0/T1 profile with a per-second regime sampler; every number carries measured/prior/unknown; capability report |
| **L2 planned runner** | **partial** | ModelCard from real GGUF headers; feasibility verdicts (sound `Infeasible` or `NotCalibrated`); measured thread-placement A/B with the real engine. **No PerformanceModel, no memory lease protocol, no degradation ladder** (so no rate is ever predicted) |
| **L3 governed platform** | **no** | no governor, no streamed tier exposed, no T2/T3 profiling, no fidelity-gated transforms |
| **L4 agent platform** | **partial** | ToolSpec registry, grammar-constrained plan/act/answer loop, consent gate, postcondition verification, audit trail. **No frozen task suite and no three baselines (PL-S1)**, so no success rate `S` is claimed |

The engine in the APK is the research fork (BigMoeOnEdge + `tools/patches`, version string 0.23.0) with one more patch,
`tools/patches/0020-*` (per-request grammar). The app currently runs it in resident mode only; its streaming flags exist in
the binary but are not exposed (`PL-E25`).

## Remaining stubs introduced or changed by the app

| id | what | state |
|---|---|---|
| ~~`PL-E7`~~ | grammar from ToolSpecs | **closed**: `Tools.grammar*()` + engine patch 0020; string-typed arguments only (others refused at generation time) |
| `PL-E23` | i8mm engine variant | not shipped: two engine builds collide on library names inside one APK; the portable armv8.2 build runs everywhere with dotprod+fp16 |
| `PL-E24` | engine crash isolation | the engine is a child process of the app; a foreground service keeps priority, but there is no separate-process supervisor/restart |
| `PL-E25` | streamed tier in the app | engine flags exist; not exposed, because cache sizing needs the memory-lease protocol and `grantable_foreground` (PL-S2) |
| `PL-E26` | Java tests | only `app/test/test_parity.py` (Java planner == Python planner on 3 real GGUFs); no UI or agent unit tests. Agent behaviour was verified by hand on a Nord |
| `PL-S1` | frozen task suite + baselines | still open; the agent's success rate is unmeasured. Observed: with grammar-constrained plan/act/answer, one 2-tool task succeeded and verified on a 3B model; a free-form loop failed on the same model |

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
  - `dramprobe.c` → `memory.dram_read_gbps` (rows failing the probe's own
    `min_cpu_over_wall` check are rejected and counted, not averaged).
  - `memprobe.c` (anonymous mode, 2 runs) → `memory.grantable_quiesced`.
  - `ufsbench.c` (3 repeats) → `storage.random_read` and
    `efficient_request_size`; one direct-I/O confirmation pair.
  Values: read them from `DeviceProfile.json`, not from this file.
- **ValidityGuard** (`meridian/validity.py`): reads `dumpsys battery` and
  `dumpsys power` to attach real `ValidityConditions` to every measurement,
  and gates `measured` on power=unplugged, foreground=none and
  wakefulness=awake. Any other state downgrades to `prior` with an interval
  from real repeats. The current run (`meridian_l1_nord_awake`) was unplugged, awake, launcher
  foreground, and is `measured`; the earlier `_wireless` run (screen off) is
  `prior` and kept only as evidence for D-5.

Raw artifacts and the assembled profile:
[`../../results/2026-09-20/meridian_l1_nord_awake/`](../../results/2026-09-20/meridian_l1_nord_awake/).

## Defects found in this code after the first commit, and how they were fixed

These are recorded as defects, not as "limitations", because each one was a
mistake in code this pass wrote — not an inherent constraint of the device.

**D-1. Fabricated uncertainty intervals (commit `2fc6e48`).** When a
measurement was out of the deployment regime, `profiler.py` downgraded it
from `measured` to `prior` and attached an interval — because
`contracts.Measured` refuses a non-`measured` value without one. The
intervals were invented to satisfy that check: `[mbps * 0.7, mbps * 1.3]`
for every storage read point (a made-up ±30%, no measurement behind it),
`[knee, knee]` for `efficient_request_size` (a zero-width "interval"), and
`[0, observed]` for `grantable_quiesced` (a lower bound of zero says nothing).
This is the CLAUDE.md §0/§6.4 failure exactly: a plausible substitute that
made a check pass, with nothing asking whether it was the quantity. It also
meant the capability report displayed intervals that looked like uncertainty
quantification and were not. **Fix:** `ufsbench` now runs 3 repeats and
`memprobe` 2 independent runs, and every interval is the real observed
min/max across those repeats (`probes/storage.py`, `probes/memory.py`
`parse_multi`). Regression test:
`test_storage_interval_is_real_spread_not_fabricated_multiplier`. The
committed `results/2026-09-20/meridian_l1_nord/` directory was produced by
the defective version and its `DeviceProfile.json` was deleted
(retraction is deletion, CLAUDE.md 7.6); raw artifacts remain as test fixtures.
**D-5.** The regime gate ignored wakefulness; a screen-off (`dozing`) run was
labelled `measured`. Gate now requires `awake`.

**D-2. Foreground state was hardcoded, not measured.** `build_conditions`
took `foreground="none"` as a default, so a run taken while the user was
watching YouTube would have been reported as quiesced. The user's own
session was in exactly that state. **Fix:** foreground is now read from
`dumpsys activity activities` before and after the T1 window and the worse
of the two is used (`validity.classify_foreground`); empty input yields
`unknown`, never `none`. The first wireless run correctly caught YouTube
(`concurrent_load: ['com.google.android.youtube']`) and downgraded every
field.

**D-3. `ROOT:` substring match.** `"ROOT:" in text` is also true of
`"NOT_ROOT:"`; every unrooted device would have been reported as rooted.
Caught by a test written before the first commit; fixed in the same commit.

**D-4. kHz/GHz mislabel in the capability report** (÷1000 labelled GHz).
Fixed before the first commit.

**Not a defect, recorded for the next reader:** over wireless adb,
`g0_probe.sh` takes ~37 s against <1 s over USB (many forked subshells,
higher per-exec latency on the TCP transport). It looked like a hang; it is
not one. Timeout raised to 90 s.

## Planner slice (L2, feasibility only)

`meridian/gguf.py` reads real GGUF headers (sizes computed from dims and the
ggml block table, cross-checked against file layout); `meridian/planner.py`
derives a `ModelCard` and issues per-tier verdicts. It never returns
"feasible" and never predicts a rate: it refuses `Infeasible` only via a
sound lower-bound argument (assumption A1: a foreground app cannot raise what
one process may keep) and otherwise returns `NotCalibrated` naming every
missing measurement. Registry rows (olmoe, gpt-oss, granitemoe) were observed
in real local checkpoints; qwen3moe is refused because no local file verifies
its pattern. Defect found while testing: offset-gap sizing over-stated tensor
bytes by alignment padding (D-6, fixed).

## Registered stubs (extends `10_EXTENSION_POINTS.md` section 4)

| id | what | current state | removed when |
|---|---|---|---|
| `PL-E21` | engine working-set bytes | excluded from tier need (makes need a lower bound; verdicts stay sound) | measured from a real engine session |
| `PL-E22` | further registry rows | qwen3moe added (observed); others refused `ArchitectureUnsupported` | a real checkpoint's expert pattern is observed |
| ~~`PL-E10`~~ | L0 engine + session protocol | **closed**: engine bundled in the APK and driven over its session protocol (`app/src/com/meridian/Engine.java`) | - |
| `PL-E11` | per-cluster `matmul_gbps` (T2 compute probe) | `Measured.unknown` in every `CpuCluster`; requires `host/app/ggml_matmul_bench.cpp`, which needs a ggml build this pass did not attempt | the ggml-based compute probe is cross-compiled and wired in |
| ~~`PL-E12`~~ | thread-placement A/B | **closed for the compute mask** (I/O mask still unknown): app `Placement.java` A/Bs masks with the real engine (ABBA, 2 reps). On the Nord, olmoe: cores 6-7 = 17.3 tok/s vs default unpinned-4 = 8.3 (results in the app's `profile.json`, `cpu.recommended_compute_mask.arms`) | - |
| `PL-E13` | per-accelerator decode/prefill rate, dispatch overhead | `Measured.unknown` for CPU/GPU/NPU; requires an engine (`PL-E10`) | T2 profiling ships |
| `PL-E14` | thermal derate curve, time-to-throttle, recovery (T3) | not attempted; would need a sustained decode load this pass has no engine to generate | T3 profiling ships, per `04_DEVICE_PROFILING.md` §3.5 |
| `PL-E15` | thermal-status OS field in `ValidityConditions` | left `None`; only wakefulness/power/battery are read from `dumpsys` | a thermal-status API call is added to `validity.py` |
| `PL-E16` | ISA features per cluster | parsed from `/proc/cpuinfo` (intersection over cluster cores); captured in `meridian_l1_nord_awake` (cpuinfo read after the run; static file) | done | |
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
