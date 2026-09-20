# 0 — The problem, the evidence, and what the prior work does and does not support

**Read this before the architecture.** It is the argument that the architecture answers. Every
measured number is quoted through [`EVIDENCE.md`](EVIDENCE.md) and re-verified by
`tools/evidence.py`; every device those numbers came from is named, because a number measured on one
phone is not a property of phones.

---

## 1. Where the prior work ended, stated plainly

`moe-phone` asked one question: can a OnePlus 15R (Snapdragon 8 Gen 5 class, 12 GB) decode an
unmodified Qwen3-30B-A3B checkpoint at 10 tok/s by streaming experts from flash? It answered **no**,
by measurement, and closed ([`../research/2026-09-19_CLOSURE.md`](../research/2026-09-19_CLOSURE.md)).

The closure is not a failure to build something. It is a map of where the time goes, and it is the
most useful input this platform has:

| what was established | number | why it matters to a platform |
|---|---|---|
| the engine beats the published baseline, controlled, same session | 1.22x [E:qwen3_h2h_ratio] (6.68 [E:qwen3_h2h_ours] vs 5.47 [E:qwen3_h2h_base] tok/s) | the streaming engine is real and competitive, but the margin is engineering, not a category difference |
| the lossless compute floor, routing replayed so no I/O is charged | 71.0 ms/token [E:qwen3_e1_repack_ms] (93.7 [E:qwen3_e1_generic_ms] with generic kernels) | **arithmetic, not flash, is the wall on this device** once the cache works |
| that floor plus the measured stall and cache management | 127.6 ms/token [E:qwen3_e1_close_ms] → 7.8 tok/s [E:qwen3_e1_close_tok_s] | the ceiling of the CPU-centric streamed architecture on this phone |
| the same compute floor at ~3,000 tokens of context | 144.3 ms/token [E:qwen3_e1_long_ms] | **at agent-sized context the model is under 7 tok/s before any flash cost is paid** |
| the GPU as the main engine, 8 layers, sustained | 51.5 ms/token [E:e7_gpu_l8_ms] against the CPU's 17.5 [E:e7_cpu_l8_ms] | the Adreno is not a rescue for streamed MoE decode on this class of part |
| training-free routing shortcuts, against an 8-bit reference | best arm p99 KL 2.06 [E:e6_best_arm_p99] against the 4-bit format's own 1.76 [E:e6_floor_p99] | there is no free quality to spend; lossy routing was measured, not assumed away |
| a second device, same model and engine | 1.67 tok/s [E:nord_decode] against a pre-registered 2.6 [E:nord_pred_point] (range from 1.8 [E:nord_pred_lo]) | **spec-sheet scaling across devices was falsified on the first try** |

That last row is the single most important fact in this document, and §4 returns to it.

## 2. The measurement the closure did not foreground, and which changes the product

The same phone, the same app process, a *small resident* MoE (OLMoE-1B-7B, Q4_0, held entirely in
RAM), measured across all three compute units
([`../results/2026-09-18/app_engine.json`](../results/2026-09-18/app_engine.json)):

| | decode | prefill |
|---|---|---|
| CPU, 4 threads | 44.5 tok/s [E:olmoe_cpu_decode] | 20.4 tok/s [E:olmoe_cpu_prefill] |
| Adreno GPU, OpenCL | 50.0 tok/s [E:olmoe_gpu_decode] | 135.3 tok/s [E:olmoe_gpu_prefill] |
| Hexagon NPU | 40.7 tok/s [E:olmoe_npu_decode] | **284.1 tok/s** [E:olmoe_npu_prefill] |

Two comparisons follow from this table, and they must be kept apart because only one of them is
clean.

**Clean: decode, across tiers.** The resident small model decodes about 6.7x faster than the
streamed 30B on the same phone (44.5 [E:olmoe_cpu_decode] against 6.68 [E:qwen3_h2h_ours] tok/s).
**This compares two different models as well as two tiers** — a 1B-active checkpoint against a
3B-active one — so it is not a measurement of what streaming costs. It is a measurement of what the
platform would actually bind, which is the decision at hand, and the confound is stated rather than
hidden.

**Clean: prefill, within the resident tier.** The NPU is last at decode and first at prefill by a
factor of 14 over the CPU on the same model and the same campaign. That is a latency-versus-
throughput signature, not a bandwidth one, and it is the reason placement must be chosen separately
for the two phases.

**Not available: prefill across tiers.** The streamed tier's prefill has never been measured at any
realistic prompt length (`PL-S3`, §8.6), so *no* ratio may be quoted for it. What bounds it instead
is a measurement that needs no prefill number at all: at ~3,000 tokens of context the streamed
model's compute floor alone is 144.3 ms/token [E:qwen3_e1_long_ms], which puts it under 7 tok/s
before a single byte is read from flash.

This is not an argument that big models are pointless. It is an argument about **defaults**, and it
is decisive for a product whose unit of work is a task rather than a token:

> A phone-task agent turn is a large-ish prompt and a short output, so it is **prefill-dominated**.
> The prior project optimised **decode**, for a model whose compute floor at agent-sized context
> already exceeds the budget. The product's default execution tier must therefore be a resident
> model on the compute unit that wins *prefill*, and the streamed large model must become the
> **escalation tier** — reached when a task demonstrably needs it, at a cost the system predicted
> before paying it.

The premise "prefill-dominated" is itself an assumption about the workload, not yet a measurement
(`PL-S3`), and experiment X2 in [`01_RESEARCH.md`](01_RESEARCH.md) §6 exists to test it. If it is
false, the tiering inverts, and that is a cheap thing to find out early rather than a reason to
delay the decision.

That reframing is the central architectural decision of this platform, and it converts the prior
project's closed negative result into a component with a defined role rather than a sunk cost.

## 3. What the actual core problem is

Not "run a big model on a phone". That is solved well enough to measure, and its ceiling is known.
The core problem is:

> **On an arbitrary phone, with an arbitrary open-weights model, choose and hold an execution
> configuration that maximises completed user tasks per unit of sustained time and energy, under
> memory the operating system may revoke at any moment and a clock that falls as the device heats —
> and tell the user what to expect, in advance, without lying.**

Four properties of that problem make it hard, and each is a component in [`02_ARCHITECTURE.md`](02_ARCHITECTURE.md):

1. **The capability question is empirical.** What a phone can run is not derivable from its spec
   sheet (§4). It has to be measured, cheaply, on the device, in the regime the product runs in.
2. **The binding resource is not RAM — it is *grantable, retainable* RAM.** On the 15R, total RAM
   reads 10.84 GiB [E:dev_ram_total_gib], `MemAvailable` at probe time was 3360048 kB
   [E:dev_ram_avail_kb], and across 238 logged runs the largest expert-cache budget the device ever
   actually **granted** was 5806 MiB [E:dev_budget_max_mib]. Those are three different numbers and
   only the third is a budget. All three were measured on a *quiesced* phone; an agent drives a
   foreground app, so the real figure is smaller and revocable.
3. **The device is a thermal machine.** Clock caps track skin temperature, so a burst benchmark
   describes a state the product never operates in. Sustained rate, time-to-throttle and a planned
   degradation path are the quantities that matter.
4. **The user must not have to know any of this.** Quantisation, expert placement, cache budgets,
   thread masks and backend selection are the platform's job. The user's inputs are a task and,
   optionally, a model they want.

## 4. Why a spec-sheet planner is not good enough, proven on our own data

The cheapest possible design is the one every existing "what can my hardware run" tool uses: take
nameplate memory bandwidth and model size, divide, print a tokens-per-second estimate. The prior
project ran exactly that experiment as a pre-registered prediction for a second phone, and **it was
falsified below its own stated range**: predicted 2.6 tok/s [E:nord_pred_point] with a floor of
1.8 [E:nord_pred_lo], measured 1.67 [E:nord_decode]
([`../results/2026-09-19/nord/README.md`](../results/2026-09-19/nord/README.md)).

The two error mechanisms are both recorded, and both are structural rather than bad luck:

- **compute** came in at 376 ms/token [E:nord_compute_ms] against ~290 assumed — two big cores did
  not deliver the capacity that core-count scaling predicted;
- **stall** came in at 171 ms/token [E:nord_stall_ms] against ~50 assumed — the older storage's
  reads did not hide behind compute the way the faster device's did.

A planner that ships those two errors to a user is worse than no planner, because it is a promise.
The architecture's answer is in [`05_PERFORMANCE_MODEL.md`](05_PERFORMANCE_MODEL.md): an analytic
model is a **prior**, on-device micro-calibration is the **estimate**, and a prediction outside the
calibrated range is **refused** rather than extrapolated.

Two further measured facts say the same thing from different directions:

- **Read granularity, not bytes.** Random 1 MiB reads run at 2806 MB/s [E:dev_flash_bulk_mbps];
  random 4 KiB reads at 315 MB/s [E:dev_flash_4k_mbps], a ratio of 8.9x [E:dev_flash_ratio]. Any
  model of streaming that counts bytes without counting request sizes is wrong by most of an order
  of magnitude.
- **Placement, not arithmetic.** The same model on the same four cores decodes at 4.0 tok/s
  [E:cliff_unpinned] unpinned and 30.1 tok/s [E:cliff_pinned] pinned. A 7.6x difference that no
  spec sheet contains, and that a platform must therefore discover per device.

## 5. What survives the pivot unchanged

These are assets, not legacy. They were measured, they generalise, and the architecture keeps them:

| asset | where it lives now | why it survives |
|---|---|---|
| **per-layer expert cache, LRU, event-atomic fetch** | [`06_EXECUTION_ENGINE.md`](06_EXECUTION_ENGINE.md) §4 | measured against every classical alternative; a shared pool across layers gets 0.0 [E:sim_global_pool] hits at the same total bytes where per-layer gets 0.28 [E:sim_lru_10pct] |
| **the architecture-registry seam** (a new MoE family is one table row, not a code change) | [`03_INTERFACES.md`](03_INTERFACES.md) §3 | this is what makes "download any model" a real feature rather than a curated list |
| **persistent sessions with a warm cache and KV prefix reuse** | [`06_EXECUTION_ENGINE.md`](06_EXECUTION_ENGINE.md) §6 | an agent takes many short turns; the serving primitives it needs already exist and were read in the engine source |
| **the fidelity toolchain** (per-token KL, top-1 flip rate, perplexity against a reference) | [`06_EXECUTION_ENGINE.md`](06_EXECUTION_ENGINE.md) §8 | every automatic optimisation the platform performs must prove it did not change the model's answers |
| **the validity instrumentation** (autosuspend detection, granted-vs-requested memory, descheduling, thermal state per row) | [`04_DEVICE_PROFILING.md`](04_DEVICE_PROFILING.md) §5 | this is the difference between a profiler and a random number generator, and it was learned by having two campaigns invalidated |
| **the probe suite** (device facts, storage curve, DRAM, thermal caps) | [`04_DEVICE_PROFILING.md`](04_DEVICE_PROFILING.md) | it already exists as shell and C; it becomes a library |

## 6. What must be generalised

Everything below is currently correct *for the OnePlus 15R* and encoded as a constant. Each becomes
a measured decision:

| hard-coded today | generalises to |
|---|---|
| compute threads pinned to `cpu4-7`, I/O lanes on `cpu0-3` | a placement derived from the cluster topology and confirmed by a short on-device A/B, stored in the device profile |
| expert-cache ceiling of 5000 MiB | a memory **lease** negotiated against measured grant behaviour and released under pressure |
| an `i8mm` build and its repacked kernels | ISA feature detection plus an on-device layout transform, adopted only if it passes a fidelity gate |
| 4 I/O lanes, 1 MiB requests | a request size and lane count read off the device's own measured storage curve |
| one model, one quantisation, one context size | a portfolio, with the choice made per task by the planner |
| "the phone" | a device class with a profile, a thermal curve and a calibration state |

## 7. What must be redesigned, not generalised

Four things in the prior work are not merely device-specific; they are the wrong shape for a
platform and must be rebuilt.

1. **The optimisation target.** Tokens per second is not the product's quantity. The platform's
   estimand is task latency and task success, defined in [`01_RESEARCH.md`](01_RESEARCH.md) §2, and
   a token-averaged rate may never be substituted for it. A configuration that decodes faster and
   completes fewer tasks has lost.
2. **The tiering.** The engine was built as one model streamed from flash. The platform is a
   portfolio with an escalation rule (§2).
3. **Memory.** The engine sizes a cache once at load. The platform must treat memory as a revocable
   lease with a shrink path, because the agent's operating condition is *another app in the
   foreground* — which no measurement in the prior project ever covered.
4. **Performance claims.** The prior work's per-term ledger is a diagnostic. The platform's
   predictions must be end-to-end, interval-valued and falsifiable, for the reason in §8.2.

## 8. Defects in the prior research this architecture must not inherit

The user asked for the architecture to be improvised where the existing experiments are flawed.
These are the flaws that matter to the design. Each is recorded in the project's own audit trail;
none of them is a criticism of the people who found them, since finding them is what the audit
trail is for.

**8.1 The compute term is a residual, not a measurement.** The engine's `compute_ms` is
`wall − stall − mgmt`, and the engine's own telemetry contract says it silently absorbs page faults,
scheduler stalls and frequency caps ([`../HEADROOM.md`](../HEADROOM.md) §0.2, and the engine's
`docs/telemetry.md`). **Consequence for the design:** the performance model may be *fitted* only on
wall-clock quantities. Residual terms are diagnostics for humans, never regression inputs, and any
calibration cell that needs a true compute number uses the direct trace mode instead.

**8.2 Term ceilings were summed although the terms share one machine.** The project's own hostile
review ([`../research/2026-09-19_HOSTILE_REVIEW.md`](../research/2026-09-19_HOSTILE_REVIEW.md) §1)
showed that adding the best case of each term produced an optimistic total that five measured levers
had already refuted: removing 15 ms of cache management returned it as 12 ms of compute and 4 ms of
stall, for a net of approximately zero. **Consequence:** the platform predicts **end-to-end wall
time** and validates end-to-end. A decomposition may explain a prediction; it may never constitute
one. Any composite prediction built by summing independently-measured savings is labelled a lower
bound on cost, never an estimate.

**8.3 Cross-model transfer of hit rate was refuted and then partly relied on anyway.** The
"equal-rho" rule for moving a cache curve between models with different expert geometry failed its
pre-registered test. **Consequence:** the planner never derives a hit rate from a scaling law across
model geometries. It uses a measured warm sample on the device, or a trace-derived curve for that
exact checkpoint, or it refuses. See [`05_PERFORMANCE_MODEL.md`](05_PERFORMANCE_MODEL.md) §4.3.

**8.4 The simulator is systematically optimistic against the phone.** The offline cache simulator
sits above the device at both calibration points. **Consequence:** simulated inputs enter the
planner only with a measured correction attached, and the correction's provenance is carried in the
prediction's `basis` field.

**8.5 The agent's operating condition has never been measured.** The prior side-track's own analysis
([`../agentic/ESTIMAND.md`](../agentic/ESTIMAND.md) §3.3) names the killer assumption: every
measurement in the project was taken on a quiesced phone with third-party apps force-stopped, while
a phone-task agent by definition runs *alongside* the app it drives. **Consequence:** the
foreground-coexistence budget is a first-class runtime signal in this architecture
([`06_EXECUTION_ENGINE.md`](06_EXECUTION_ENGINE.md) §5) **and** the first experiment in
[`01_RESEARCH.md`](01_RESEARCH.md) §6, because it is the cheapest one that can kill the design.

**8.6 Prefill was never measured as a function of prompt length.** Every prefill figure in the
project was taken at a 26-token [E:gptoss_prompt_tokens] prompt, which makes 6.0 tok/s
[E:gptoss_prefill_short] a cold-start artifact rather than an ingestion rate. For a
prefill-dominated product this is the largest unmeasured quantity in the entire record. The probe
script exists and has never been run. **Consequence:** prefill-versus-length is a **required**
calibration cell; no plan may be issued for an agent workload without it
([`05_PERFORMANCE_MODEL.md`](05_PERFORMANCE_MODEL.md) §3.1).

**8.7 Storage was assumed to be storage.** On a PC the same NVMe drive measured 14x slower through a
FUSE filesystem than natively, and *degraded* with added concurrency
([`../research/2026-09-19_PC_SCOPE.md`](../research/2026-09-19_PC_SCOPE.md) §1). **Consequence:** the
platform measures the filesystem path a model actually sits on, and refuses to plan a streaming
configuration on a path whose measured random-read rate is below what the plan requires.

**8.8 An optimisation was adopted on the strength of a microbenchmark and then failed its fidelity
gate end to end.** The repacked-kernel path won its isolated bench and then missed its pre-registered
perplexity gate in the full engine. **Consequence:** every automatic transform in this platform —
repack, requantisation, KV policy, any lossy routing — is gated by an on-device fidelity check
before adoption, and a failing transform is *not used*, not "used with a note".

## 9. The one-paragraph statement of what is being built

A phone-resident service that measures its own hardware honestly, keeps a small portfolio of local
models, predicts with stated uncertainty what each of them will actually do on this device under
sustained load, chooses and holds a configuration under revocable memory and a falling clock, and
exposes that as a single capability call to an on-device agent that does everyday tasks for its
owner. The heavy MoE-streaming engine from `moe-phone` sits inside it as the tier that runs models
larger than memory, used when a task needs one and the cost has been predicted first.
