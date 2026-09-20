# 4 — Device profiling: measuring a phone you have never seen

Layer 0. It produces the `DeviceProfile` of [`03_INTERFACES.md`](03_INTERFACES.md) §1, and it is the
component everything above it depends on. A wrong profile does not produce a slow system; it
produces a confident, wrong promise.

The probe suite already exists in this repository as shell and C
([`../device/g0_probe.sh`](../device/g0_probe.sh), `memprobe.c`, `dramprobe.c`, `devprobe.c`, the
storage bench, [`../device/bmoe_sustain.sh`](../device/bmoe_sustain.sh)). This document specifies
what it becomes as a library, and the rules it must keep.

---

## 1. The governing principle

> **Measure the thing you will do, in the regime you will do it in, with the code that will do it.**

Every violation of that sentence in the prior project produced a wrong number, and the violations
were never obviously wrong at the time. A synthetic matmul is not the engine's matmul. A read of the
block device is not a read of the model's path. A two-second burst is not a sustained rate. A
quiesced phone is not a phone running the app the agent is driving.

---

## 2. Probe tiers, because measurement costs time and heat

Profiling is itself a load. It warms the device, spends battery and delays first use, so it is
staged. Each tier produces a usable profile; later tiers narrow intervals and upgrade fields from
`prior` to `measured`.

| tier | budget | when | what it establishes |
|---|---|---|---|
| **T0 static** | < 100 ms | every launch | identity, topology, ISA features, RAM total, filesystem, which accelerators and vendor runtimes are even present |
| **T1 fast** | ~5–10 s | first launch, before the first capability report | storage random-read at two sizes, a short per-cluster compute probe, available memory, one grant probe |
| **T2 standard** | ~60 s | after first launch, or before first binding a new model | the full storage curve and its knee, per-backend decode and prefill rates on model-shaped kernels, dispatch overhead, a first derate sample |
| **T3 deep** | ~10 min | opportunistic: charging, screen off, cool, idle | the sustained derate curve, time-to-throttle, recovery, swap-fault cost, foreground-coexistence grant (RQ2), per-model warm hit-rate samples |

**T3 is where the honest numbers come from**, and the scheduler's job (§6) is to acquire it without
the user ever noticing. Until T3 exists for a device, every sustained prediction is `calibrated` or
`prior` at best, and the interface must say so.

---

## 3. The probes

### 3.1 Static inventory (T0)

Reads properties, `/proc`, `/sys` and the filesystem. Three rules learned the hard way:

- **Distinguish "denied" from "absent" from "empty".** A probe that reports nothing where a read was
  refused makes a permissions problem look like a hardware fact. Every key returns one of a value,
  `DENIED`, or `ABSENT`.
- **Resolve the device you actually use.** Hardcoded block-device names collected *nothing* on the
  primary phone, whose data partition sits behind a device-mapper node over multiple storage units.
  Resolve from the mount that backs the model directory.
- **Verify capabilities by exercising them.** Root is detected by executing a privileged command and
  checking the result, not by finding a binary named `su` on the path — a sandboxed terminal app
  ships its own.

### 3.2 Compute probe (T1/T2)

Runs the **engine's own kernels** on **the candidate model's shapes**, per cluster and per backend.

- Per cluster, with threads pinned: matmul throughput in weight-bytes per second at batch 1 (decode
  shape) and batched (prefill shape). These differ enough to change decisions, so both are measured.
- Per accelerator: the same, plus **dispatch overhead** — the fixed per-call cost. That single number
  decides whether an accelerator can be used as a *helper* for small per-layer work at all: on the
  primary device the fixed round trip alone exceeded the CPU's entire cost for the work being
  offloaded, which is why the helper role was abandoned there.
- Thread placement is probed, not assumed: a short A/B over candidate masks derived from the cluster
  topology. The measured spread between placements on the primary device was 4.0 [E:cliff_unpinned]
  against 30.1 [E:cliff_pinned] tok/s on the same four cores, so this probe pays for itself many
  times over and must not be skipped.

### 3.3 Storage probe (T1/T2)

Run **on the directory where models will live**, never on a synthetic path.

- Random reads with direct I/O at 4 KiB, 64 KiB, 256 KiB, 1 MiB and 4 MiB, across 1/4/8/16 threads.
- Output: the full curve, plus `efficient_request_size` — the smallest request reaching bulk rate.
  On the primary device the ratio between bulk and small reads was 8.9x [E:dev_flash_ratio]
  (2806 [E:dev_flash_bulk_mbps] against 315 [E:dev_flash_4k_mbps] MB/s), which is the whole reason
  the engine reads whole experts rather than rows.
- **Record the filesystem.** A userspace-bridged filesystem measured 14x slower than the same
  hardware natively, and got *worse* with concurrency. The profile must carry the measured rate for
  the actual path, and the planner must refuse streaming plans on a path that cannot serve them.
- Concurrency is swept rather than assumed: the right lane count is where the curve plateaus, and on
  the primary device that plateau started at four lanes with no gain past eight.

### 3.4 Memory grant probe (T1/T3)

The question is not "how much RAM does this phone have" but "how much will it give me and let me
keep". Three separate quantities, measured separately:

1. `available_at_probe` — what the OS reports. Informational.
2. `grantable_quiesced` — allocate in increments, **verify resident set after each step**, stop on a
   residency plateau. Pages that have been compressed into swap are not memory: the allocator will
   happily report success while every later touch becomes a synchronous fault.
3. `grantable_foreground` — the same, with a real target app foregrounded and active. **This is the
   number the agent lives on and no measurement of it exists anywhere in this project** (`PL-S2`).

Safety rules: increment with a margin below the last successful step, register the process so the
low-memory killer's decisions are observable, and abort on the first sign of system-wide pressure.
A grant probe that gets the app killed has measured something true and lost the profile with it, so
the result is checkpointed after every increment.

### 3.5 Thermal probe (T3)

A fixed sustained load — the engine decoding a fixed prompt, repeatedly, for a set period — with
clock caps, skin temperature, throughput, battery state and thermal status logged per run. Outputs:

- the **derate curve**: throughput fraction against skin temperature;
- **time-to-throttle** from a cool start;
- **recovery time** back to full clock.

The curve is what makes a *sustained* prediction possible, and the prior project's central benchmark
finding is that without it every number is a burst number: the clock cap tracked skin temperature,
so the same configuration measured materially differently depending on how long it had been running
and whether it was charging.

### 3.6 Warm-sample probe (T3, per model)

For a streamed configuration, the expert-cache hit rate is the dominant unknown and it is **not**
transferable across models ([`00_PROBLEM.md`](00_PROBLEM.md) §8.3). So it is measured: a short warm
generation at the planned cache size, reporting hit rate and bytes read per token. A few hundred
tokens is enough for the cache to converge, and this replaces an entire class of scaling assumptions
with one cheap measurement.

---

## 4. Contamination: the failures that make measurements lie

These are not hypotheticals. Each one invalidated real runs in this project, and each is now a check.

| contaminant | what it does to the number | detection |
|---|---|---|
| **device autosuspend** while unplugged | runs become bimodal; a campaign averages two regimes and reports neither | sample wakefulness during the run; reject runs that were not continuously awake |
| **process descheduled** | a fixed-duration probe takes far longer and reports a fraction of the true rate, with latency percentiles unchanged — so the storage looks broken when it was fine | compare elapsed time to the intended deadline; reject beyond a fixed margin |
| **foreground loss / lock screen** | the process leaves the foreground scheduling group mid-run | sample foreground and keyguard state |
| **charging state** | changes both thermal headroom and governor behaviour | record; never compare across it |
| **thermal carryover between arms** | arm B inherits arm A's heat, so order becomes the effect | rotate order (ABBA), gate on a start temperature, record start and end |
| **orphaned processes from a previous run** | a second engine competes for memory and cores | scan for known process names before starting |
| **a sensor that is not a temperature** | a state-of-charge zone reporting a bare integer becomes the maximum of every temperature row | validate units and range; record which zone produced the value |

**The rule: a contaminated run is discarded, never averaged, and the rejection is counted in the
profile.** A profile that silently drops half its runs is telling you something about the device.

---

## 5. The validity guard

One component, consulted by every probe, that answers: *does this measurement count?*

```
guard.begin(expected_conditions) -> handle
guard.sample(handle)                         # during the run, at a fixed interval
guard.end(handle) -> ValidityConditions + verdict{ valid | rejected(reason) }
```

Rejection reasons are enumerated (§4) and stored. A probe never decides its own validity, and the
guard never looks at the measured value — it only looks at the conditions, so it cannot reject a
number for being inconvenient.

---

## 6. The calibration scheduler

Device time is the scarcest resource in this design, and the prior project's most repeated mistake
was spending it on the wrong measurement. The scheduler enforces three rules taken from that
experience:

1. **Largest uncertainty first.** Rank pending calibration cells by how much they would narrow a
   prediction the user actually depends on. A cell that cannot change a decision is not run.
2. **Opportunistic for anything expensive.** T3 runs only when charging, cool, screen off and idle,
   and yields immediately if the user returns.
3. **Falsification pays for itself.** Every governor invalidation ([`02_ARCHITECTURE.md`](02_ARCHITECTURE.md) §3.7)
   produces a free calibration datum, because the run already happened. These are the cheapest
   measurements in the system and the profile should be dominated by them over time.

Budgets are explicit: a maximum share of battery per day, a maximum device-temperature rise, and a
hard stop when the user is interacting.

---

## 7. What an unprivileged app cannot have, and what to do instead

The platform must work as an ordinary installed app with no root. Known limits on the primary
device, each with the fallback:

| wanted | availability | fallback |
|---|---|---|
| per-rail power | not readable by an app | battery current × voltage, with the display and radio contribution named as a limitation (`PL-S5`) |
| memory locking | needs privilege | accept reclaim; design the cache to be re-readable rather than pinned — an attempt to pin the cold cache on the primary device pushed hot data into swap and cost more than it saved |
| block-device queue tuning | not writable | choose request size and lane count instead; both are under our control |
| some thermal zones and governor internals | partially denied | measure the *effect* (the derate curve) rather than reading the mechanism |
| NPU access | reachable by ordinary apps through the vendor's unsigned process domain; this project captured a direct existence proof of a full language model running on the NPU of an unrooted retail phone | if unavailable, demote to GPU or CPU by measured rate, never by assumption |

The profile records which of these were denied, because a denial is a fact about the device that
changes what the planner may consider.

---

## 8. Profile lifecycle

- **Created** at first launch (T0+T1), upgraded by T2 and T3.
- **Invalidated** by an OS build change, a probe-suite version bump, a change of model storage path,
  or repeated prediction drift flagged by the governor.
- **Never merged across devices.** Device-class priors are a separate artifact with a separate type
  ([`05_PERFORMANCE_MODEL.md`](05_PERFORMANCE_MODEL.md) §2), and they may inform a prediction but may
  never be written into a profile as if measured.
- **Exportable**, so that a measured profile can be attached to a bug report or a paper and so the
  platform's own claims stay reproducible.
