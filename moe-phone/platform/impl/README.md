# `platform/impl/` — Meridian, as far as this pass actually built it

This is the first code under `moe-phone/platform/`, which until now was
documentation only (`../README.md`: "Status: architecture and design
documentation. No implementation exists yet.").

**Read [`STATUS.md`](STATUS.md) before trusting anything in here.** It
states the conformance level reached (T0/T1 device profiling — below full
L1, since there is no engine yet), lists every registered stub, and says
which numbers in the capability report are `measured` versus downgraded to
`prior` because the phone was USB-charging for the whole session (not the
`unplugged` deployment regime decision D4 requires).

## What this is

A device profiler that runs the existing, already-validated C probe suite
(`../../device/{devprobe,dramprobe,memprobe,ufsbench}.c` — see
`00_PROBLEM.md` section 5 and `04_DEVICE_PROFILING.md`'s own description of
them as "already exists ... and becomes a library") against a real attached
Android device over adb, applies the `ValidityGuard` from
`04_DEVICE_PROFILING.md` section 5, and assembles a `DeviceProfile` per the
schema in `03_INTERFACES.md` section 1 — with real provenance, real
intervals, and real refusals (`Measured.unknown(...)`) for anything it did
not measure.

```
meridian/
  contracts.py    DeviceProfile / Measured<T> / ValidityConditions, transcribed from 03_INTERFACES.md
  adb.py          thin adb subprocess wrapper
  validity.py     ValidityGuard: conditions parsing, descheduled-row rejection, thermal-zone sanity filter
  probes/
    static_inventory.py   parses g0_probe.sh + devprobe output (T0)
    dram.py                parses dramprobe.csv (T1)
    storage.py             parses ufsbench.csv (T1)
    memory.py              parses memprobe stdout (T1)
  profiler.py     orchestrates: push probes, run on device, pull results, build DeviceProfile
  report.py       renders the plain-language capability report from a DeviceProfile
  cli.py          `python3 -m meridian.cli profile ...`
build_probes.sh   cross-compiles the four C probes with the Android NDK
tests/            parser + invariant tests against the real captured fixtures
```

## Why it stops here

`02_ARCHITECTURE.md`'s layer diagram has eight layers above this one
(hardware abstraction, execution engine, decision layer, inference service,
agent runtime, experience). Building any of them for real — not as a
plausible-looking stand-in — requires things this pass does not have:

- an actual GGUF-loading inference engine with a session protocol
  (`06_EXECUTION_ENGINE.md` section 11) to run models at all;
- the ggml-based matmul bench (`../../host/app/ggml_matmul_bench.cpp`)
  built against an actual ggml checkout, for per-cluster/per-backend
  throughput (`PL-E11`/`PL-E13`);
- an installed Android app to foreground, for the foreground-coexistence
  memory grant (`PL-S2`) that the docs identify as **the single cheapest
  experiment that could kill the whole architecture** (R1 in
  `10_EXTENSION_POINTS.md`) — building a fake version of this measurement
  would be worse than not building it, since a false "yes it coexists" is
  exactly the failure mode `CLAUDE.md` section 0 describes.

Per `10_EXTENSION_POINTS.md` section 7 ("where to start"), the honest next
step after this one is **X1** — the foreground-coexistence experiment —
before the planner or engine get built at all, because it is cheap and it
can invalidate the whole design. This pass does not run X1 either, because
X1 needs a real target app to foreground and drive, which does not exist.

## Reproducing the measured profile

```sh
NDK=$HOME/Android/Sdk/ndk/<version> sh build_probes.sh /tmp/meridian_bin
python3 -m meridian.cli profile --bin-dir /tmp/meridian_bin \
    --out-dir ../../results/$(date +%F)/meridian_l1_<device_label>
python3 tests/test_meridian.py   # parser + invariant tests against real fixtures
```

Needs one adb-authorized device attached. The committed run is
[`../../results/2026-09-20/meridian_l1_nord/`](../../results/2026-09-20/meridian_l1_nord/)
(OnePlus Nord, SM7250).
