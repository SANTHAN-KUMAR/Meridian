# 9 — Hardware abstraction, vendor acceleration, and the iQOO/Snapdragon integration map

What stays neutral, what is allowed to be vendor-specific, where the seams are, and what deeper
silicon integration would actually buy — each with the measurement that would prove it.

The governing decision is **D9** ([`02_ARCHITECTURE.md`](02_ARCHITECTURE.md) §1): vendor
acceleration is **discovered**, **measured**, and **never on the correctness path**.

---

## 1. Why this needs a discipline at all

A platform that hard-codes one vendor's accelerator is a demo. A platform that ignores vendor
accelerators leaves most of a modern phone unused: on the primary device the NPU prefilled a
resident model at 284.1 tok/s [E:olmoe_npu_prefill] against the CPU's 20.4 [E:olmoe_cpu_prefill] —
a factor of fourteen that no portable path recovers.

Both failure modes are avoided by the same structure: a **narrow port**, a **capability
declaration**, a **measured admission test**, and a **fidelity gate**. A vendor path that passes
all four is used; one that does not is demoted automatically, without anyone editing policy.

This matters concretely because the ecosystem is unreliable in both directions: an NPU backend for
the portable runtime exists upstream, is described as experimental, and has been reported to produce
corrupted output on some models — while, on the same class of hardware, a vendor runtime runs the
same file correctly and much faster. Neither "trust it" nor "avoid it" is the right policy. **Measure
it, gate it, and record the verdict per device and per model.**

---

## 2. The ports

Four ports, each with a neutral default implementation that works on any Android device, and
optional vendor implementations selected at runtime.

### 2.1 `Backend`

```
interface Backend {
  id: BackendId
  capabilities(): { regions: [Region], dtypes: [string], max_batch: int,
                    supports_moe_gather: bool, weights_shared_with_cpu: bool }
  probe(model_shapes): { decode_gbps, prefill_gbps, dispatch_overhead_ms }
  prepare(model, region): PreparedGraph | Unsupported
  execute(PreparedGraph, inputs): outputs
  fidelity_class(model): "bit-exact" | "tolerant" | "unverified"
}
```

`weights_shared_with_cpu` is load-bearing: on a unified-memory phone an accelerator that requires a
**copy** of the weights pays twice, once in memory it takes from the expert cache and once in the
copy itself. On the primary device a generic GPU path's upload cost exceeded the compute it was
meant to accelerate by a large factor. Any backend declaring `false` here is admitted only for
regions whose weights are resident for the session, never for streamed experts.

### 2.2 `ThermalSource`, `PowerSource`, `MemoryPressureSource`

Read-only signals. Each reports `available: bool` and never fabricates a value when a rail or a zone
is unreadable — the profile records the denial as a fact about the device
([`04_DEVICE_PROFILING.md`](04_DEVICE_PROFILING.md) §7).

### 2.3 `PerfHintSink`

Write-only. Where the OS offers performance and thermal hint APIs, the platform declares its work
type and target duration so the governor can cooperate with the scheduler rather than fight it. A
missing implementation is a no-op, never a failure.

---

## 3. What is vendor-neutral, and stays that way

These are the parts that must work on any Android phone from any silicon vendor, because they are
the platform's actual product:

| neutral forever | why |
|---|---|
| the contracts of [`03_INTERFACES.md`](03_INTERFACES.md) | a vendor pack fills fields; it never changes their meaning |
| the probe suite and validity guard | measuring a phone must not require the phone's cooperation |
| the performance model, planner and governor | the whole value proposition is that it adapts, which it cannot do if it is written for one part |
| the expert-cache design, residency and lease protocol | measured properties of MoE routing and of Android memory, not of a vendor |
| the agent runtime, tools and safety gate | Android APIs, not silicon |
| the CPU backend | the fallback that must always exist, and the reference the fidelity gate compares against |
| the app | Kotlin and Compose on any device |

**The CPU backend is the correctness reference.** Every accelerated path is validated against it.
That is what makes it safe to adopt an experimental vendor path at all.

---

## 4. What is allowed to be vendor-specific

Packaged as a `VendorPack`: a discovered module contributing `Backend` implementations, optional
signal sources, and optional model-preparation steps.

```
VendorPack {
  id, vendor, version
  detect(): bool                     // libraries present, device nodes openable, runtime responds
  backends(): [Backend]
  signals(): { thermal?, power?, memory_pressure?, perf_hint? }
  prepare_model(ModelCard): TransformRecord | Unsupported   // e.g. a compiled bundle
  admission_test(): measured results, fed to the profile
}
```

Rules:

1. **Detection executes**, it does not infer. A library on disk is not a working path; open the
   device node and complete a round trip.
2. **Admission is measured** into the profile's `accelerators` array, under the same validity guard
   as any other probe.
3. **Adoption requires a fidelity gate** for that model on that device
   ([`06_EXECUTION_ENGINE.md`](06_EXECUTION_ENGINE.md) §8).
4. **Demotion is automatic** on gate failure, numerical drift, or repeated runtime errors; the
   planner simply stops proposing it.
5. **No vendor pack may be required to build or run.** A build with zero packs must pass the whole
   test suite.

---

## 5. The Snapdragon and iQOO integration map

Concrete, and ordered by how much each is worth given what has been measured. iQOO's current
flagships pair Snapdragon 8 Elite-class silicon with LPDDR5X memory and UFS 4.1 storage, which
changes two of the three terms in the cost model relative to the device all of this was measured on.

### 5.1 Tier 1 — available to any app today, high value

| hook | what it gives | evidence it matters | how to verify on a new device |
|---|---|---|---|
| **Vendor GenAI runtime** (Qualcomm's Genie / GenieX line) for the resident tier, especially prefill | the NPU's prefill advantage without writing an NPU backend; it already runs common quantised formats across NPU, GPU and CPU and exposes a local API | 284.1 tok/s [E:olmoe_npu_prefill] prefill against 20.4 [E:olmoe_cpu_prefill] on the CPU | admission test + fidelity gate; compare against the CPU reference on the same file |
| **Precompiled model bundles** from the vendor's model hub | NPU execution without on-device compilation cost | removes a multi-minute preparation step from first use | hash-pinned bundles; gate as a `vendor_compile` transform |
| **NPU access from an ordinary app** via the vendor's unsigned process domain and its asynchronous DSP queue | no root required; this project captured a direct existence proof of a full language model running on the NPU of an unrooted retail phone, including the asynchronous queue interface | it is the difference between the NPU being a research curiosity and a product path | open the device node, complete a round trip, measure dispatch overhead |
| **OpenCL on the GPU** for resident-tier decode | the fastest single unit at decode on the primary device, 50.0 tok/s [E:olmoe_gpu_decode] | measured | standard admission test |
| **OS performance and thermal hint APIs** | cooperate with the governor instead of fighting the scheduler; read thermal headroom rather than inferring it | the derate curve is currently measured the slow way | compare hinted against unhinted sustained rate |

### 5.2 Tier 2 — plausible, needs measurement on the target device

| hook | hypothesis | the measurement that decides it |
|---|---|---|
| **UFS 4.1 storage** | a higher bulk rate and a lower knee shift the streamed tier's stall term materially; the primary device measured 2806 MB/s [E:dev_flash_bulk_mbps] bulk against 315 [E:dev_flash_4k_mbps] at 4 KiB | run the storage probe; re-derive `efficient_request_size`; re-plan |
| **LPDDR5X bandwidth** | raises the resident tier's ceiling; the primary device gave an app 59.7 GB/s [E:dev_dram_gbps] | DRAM probe |
| **Larger memory configurations (12–16 GB)** | models that stream on a 12 GB device may become resident, which is a tier change, not a speed change | grant probe, quiesced and foreground |
| **Vendor memory-extension features** | may increase the grant — or may be compressed swap wearing a friendly name, which this project measured as actively harmful when the engine's pages landed in it | grant probe plus major-fault rate per token |
| **A second gaming/display co-processor** | irrelevant to inference unless it frees the main GPU; do not assume | measure GPU contention with the display pipeline active |

### 5.3 Tier 3 — what a deeper partnership could unlock

This is the section to put in front of a hardware partner. Each item names what the platform would
do with it and how the benefit would be proven, so none of it is a wish.

| ask | why it is blocked today | what the platform would do with it | proof |
|---|---|---|---|
| **A larger, retained memory grant for a registered on-device AI service** | the OS grants what it grants; the largest budget this project ever obtained across 238 runs was 5806 MiB [E:dev_budget_max_mib], on a quiesced phone | a bigger expert cache moves the streamed tier's hit rate directly, and a large enough grant changes tier entirely | grant probe before and after; hit rate and sustained rate at each budget |
| **Zero-copy weight access for the NPU/GPU** | a backend that needs its own copy of the weights pays memory the cache needed and a copy cost that, measured on a generic path, exceeded the compute it replaced | the accelerator becomes admissible for **streamed experts**, not just resident regions — the single largest structural change available to the streamed tier | dispatch-overhead probe and an end-to-end A/B under the autotune rules |
| **Lower-latency small-dispatch path to the NPU** | at batch one the fixed per-call cost dominates, which is why the NPU lost at decode (40.7 [E:olmoe_npu_decode] against the CPU's 44.5 [E:olmoe_cpu_decode]) while winning prefill by 14x | decode moves to the NPU, and the whole placement calculus changes | `dispatch_overhead_ms` measured; the §3.2 selection rule admits it automatically |
| **Storage read-priority or stream hints** | the engine's reads compete with everything else on the device | the un-hidden stall term, measured at 34 ms/token [E:qwen3_h2h_stall_ms] on the primary device, is directly attackable | stall term under matched conditions, with and without the hint |
| **A published thermal budget and derate curve** | the platform measures it over ~10 minutes per device because it is not exposed | T3 profiling collapses from minutes to instant, and predictions become `measured` on first launch | compare the published curve against the measured one; the gap is itself a result |
| **Per-rail power access for a registered service** | unavailable to an unprivileged app, so energy accounting is coarse (`PL-S5`) | joules per completed task becomes a real metric rather than an estimate | validate the coarse method against the rail method on the same workload |
| **Early silicon access / a reference device** | — | the performance model gains a calibration point on a new part before launch, and the device ships with accurate predictions on day one | pre-registered prediction, then measurement — the protocol already exists |

### 5.4 What a partner gets back

Stated plainly, because an integration argument runs both ways: a measured, reproducible
characterisation of their silicon under **sustained** load on real model workloads — the derate
curve, the grant behaviour, the storage knee, the dispatch costs, and the prediction errors — using
a harness whose validity discipline has already caught benchmark artifacts that ordinary on-device
benchmarks are exposed to, including devices that autosuspend mid-run and report bimodal results.

---

## 6. Portability beyond one vendor

The same ports cover the rest of the space, and the platform must be tested against at least one
non-Qualcomm device before any generality is claimed:

- **Other Android SoC vendors**: different CPU topologies, different NPU stacks, different storage
  controllers. The neutral path (CPU plus OpenCL) works; vendor packs are additive. The prior phase
  already measured a second, much older Snapdragon device and found both major terms mispredicted,
  which is the correct warning about assuming a family behaves like its flagship.
- **Desktop and laptop**: the same three-term model applies, and the measured lesson transfers
  intact — filesystem choice dominated storage behaviour, with one drive measuring an order of
  magnitude slower through a userspace bridge. The niche differs: on phones every large mixture
  model exceeds memory, while on PCs only the very large ones do.
- **iOS**: a separate port with a different accelerator stack and much tighter background execution
  limits. Out of scope for this document beyond noting that the contracts are platform-neutral and
  the agent layer is not.

---

## 7. Invariants

| invariant | test |
|---|---|
| a build with zero vendor packs passes the full suite | CI configuration without packs |
| no vendor type appears above layer H | architecture lint on imports |
| every admitted backend has a measured admission result in the profile | profile audit |
| every adopted accelerated path has a passing fidelity gate for that model and device | registry audit |
| a vendor pack that fails detection leaves no trace in the plan | negative test with a stubbed absent runtime |
| the CPU path can run every supported model on every device | conformance test |
