# 2 — System architecture

The system, its layers, what each component is responsible for, how they talk, and how they behave
at runtime. Contracts are named here and specified in [`03_INTERFACES.md`](03_INTERFACES.md).

The argument that motivates these decisions is in [`00_PROBLEM.md`](00_PROBLEM.md); this document
assumes it.

---

## 1. Architectural decisions, stated up front

Each decision names what it rules out, so an implementer can tell when they are violating one.

| # | decision | rules out |
|---|---|---|
| **D1** | **The platform is a portfolio, not a model runner.** Several models of different sizes coexist; a task is bound to one of them per turn | a single-model app; "the model" as a global |
| **D2** | **Prediction is a contract with an interval and a provenance, never a scalar** | printing an estimated tokens-per-second from a formula |
| **D3** | **Measured beats inferred beats assumed, and the consumer can see which it got** | silently substituting a prior for a measurement |
| **D4** | **The quantity is sustained, in the deployment regime** — awake, unplugged, thermally settled, app foreground | burst benchmarks; numbers taken on a plugged-in idle phone |
| **D5** | **Memory is a lease, not an allocation.** It is granted, verified, held with a heartbeat, shrunk on pressure, and revoked | sizing a cache once at load and assuming it survives |
| **D6** | **Every plan carries its own falsification rule.** The runtime continuously compares outcome to prediction and invalidates the plan when it drifts | open-loop configuration |
| **D7** | **The agent never sees compute detail.** It declares intent and constraints; the planner answers with a binding | an agent that picks a quantisation |
| **D8** | **Any automatic transform must pass a fidelity gate on the device before adoption** | adopting an optimisation because a microbenchmark liked it |
| **D9** | **Vendor acceleration is discovered, measured, and never on the correctness path** | a build that only works on one silicon vendor; a vendor path whose output is trusted untested |
| **D10** | **Refusing to answer is a supported outcome** at every layer: refuse to predict, refuse to plan, refuse to run | returning a number outside the calibrated range |

---

## 2. The layer diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│  E  Experience          task console · model shelf · capability report   │
│                         consent prompts · live telemetry · audit view    │
├──────────────────────────────────────────────────────────────────────────┤
│  D  Agent Runtime       intent → plan → act → verify → recover           │
│                         ToolRegistry · ContextManager · SafetyGate       │
│                         Recorder (audit + evaluation data)               │
├══ contract ══ CapabilityRequest  ▶  ModelBinding ════════════════════════┤
│  C  Inference Service   SessionManager · admission · preemption          │
│                         local API (in-process; optional localhost HTTP)  │
├══ contract ══ ExecutionPlan  ▶  EngineSession ═══════════════════════════┤
│  B  Decision Layer      PerformanceModel · ConfigurationPlanner          │
│                         ModelRegistry · Acquisition · Transform · GC     │
├══ contract ══ DeviceProfile · ModelCard  ▶  PerfPrediction ══════════════┤
│  A  Execution Engine    PlacementController · ResidencyManager           │
│                         ExpertStream · LayoutService · FidelityService   │
│                         KVManager · Telemetry                            │
├──────────────────────────────────────────────────────────────────────────┤
│  H  Hardware Abstraction  Backend(CPU|GPU|NPU|Vendor) · ThermalSource    │
│                           PowerSource · MemoryPressureSource · PerfHint  │
├──────────────────────────────────────────────────────────────────────────┤
│  0  Device Profiler     StaticInventory · MicroBench · ThermalProbe      │
│                         GrantProbe · ValidityGuard · ProfileStore        │
└──────────────────────────────────────────────────────────────────────────┘
        ▲                                                        │
        └──────────── RuntimeGovernor: observed vs predicted ─────┘
                      (invalidates plans, schedules calibration)
```

Layer 0 and layer H are the only places that touch device specifics. Layer A is the only place that
touches model weights. Layers C, D and E never see either.

---

## 3. Components and responsibilities

### 3.1 Layer 0 — Device Profiler

The platform's sensory system. Detail in [`04_DEVICE_PROFILING.md`](04_DEVICE_PROFILING.md).

| component | responsibility | must not |
|---|---|---|
| `StaticInventory` | facts readable without running anything: SoC and board, cluster topology, ISA features, RAM total, storage type and **filesystem of the model directory**, OS and API level, presence of vendor runtimes and device nodes, root state | be trusted as a performance claim |
| `MicroBench` | measured throughput on the **engine's own kernels and I/O path**: per-cluster compute on model-shaped matmuls, storage random reads across request sizes, DRAM read rate, page-fault and swap behaviour | use synthetic kernels that the engine does not run |
| `GrantProbe` | how much memory this device will actually **grant and let us keep**, measured by allocating and verifying residency, under a named foreground condition | report the requested size |
| `ThermalProbe` | the sustained-load derate curve: throughput and clock caps against skin temperature, time-to-first-throttle, recovery time | run for two seconds and call it sustained |
| `ValidityGuard` | decides whether a measurement counts: wakefulness, foreground, charging, thermal state, descheduling, orphan processes | average a contaminated run into a clean one |
| `ProfileStore` | versioned `DeviceProfile` with per-field provenance and confidence; invalidation on OS update, storage move, or repeated prediction drift | keep a profile whose probe-suite version no longer matches |

### 3.2 Layer H — Hardware Abstraction

A narrow set of ports so that layer A is written once. Every port has a neutral implementation and
optional vendor implementations, selected by capability detection ([`09_VENDOR_INTEGRATION.md`](09_VENDOR_INTEGRATION.md)).

- `Backend` — a compute device that can run a named graph region: `capabilities()`, `estimate()`,
  `execute()`, `fidelity_class()`. Implementations: CPU (always present), GPU (OpenCL), NPU
  (QNN/Genie or vendor), and vendor-specific packs.
- `ThermalSource`, `PowerSource`, `MemoryPressureSource` — read-only signals, each reporting
  availability rather than pretending a missing rail is zero.
- `PerfHintSink` — write-only performance and thermal hints to the OS where the platform offers them.

### 3.3 Layer A — Execution Engine

The generalised `moe-phone` engine. Detail in [`06_EXECUTION_ENGINE.md`](06_EXECUTION_ENGINE.md).

| component | responsibility |
|---|---|
| `PlacementController` | which backend runs which graph region, **separately for prefill and decode**, because the measured winner differs between them |
| `ResidencyManager` | what is in RAM: resident weights, expert cache, KV — all under one lease, with a defined shrink order |
| `ExpertStream` | reads experts from storage at the device's measured efficient request size, on lanes pinned by topology; per-layer cache, LRU/SLRU, event-atomic fetch |
| `LayoutService` | on-device weight transforms (kernel repack, expert pre-layout), each produced once and gated before use |
| `KVManager` | KV allocation, quantisation policy, prefix reuse, eviction |
| `FidelityService` | KL, top-1 flip rate and perplexity against a reference, as a runtime capability |
| `Telemetry` | per-token and per-turn events with **wall-additive** semantics and explicit residual labelling |

### 3.4 Layer B — Decision Layer

| component | responsibility | detail |
|---|---|---|
| `ModelRegistry` | the portfolio: cards, derived byte budgets, feasibility per device, integrity, retention | [`03_INTERFACES.md`](03_INTERFACES.md) §3 |
| `Acquisition` | resumable download, storage-path verification, hash check, quarantine until verified | [`06_EXECUTION_ENGINE.md`](06_EXECUTION_ENGINE.md) §9 |
| `PerformanceModel` | predicts `PerfPrediction` from `DeviceProfile` × `ModelCard` × configuration, with interval, basis and calibration state; **may refuse** | [`05_PERFORMANCE_MODEL.md`](05_PERFORMANCE_MODEL.md) |
| `ConfigurationPlanner` | searches the configuration space under constraints and emits an `ExecutionPlan` with a degradation ladder and a falsification rule | [`05_PERFORMANCE_MODEL.md`](05_PERFORMANCE_MODEL.md) §6 |
| `CalibrationScheduler` | decides when to spend device time on measurement, under a time, heat and battery budget | [`04_DEVICE_PROFILING.md`](04_DEVICE_PROFILING.md) §6 |

### 3.5 Layer C — Inference Service

| component | responsibility |
|---|---|
| `SessionManager` | one warm session per bound model: model loaded once, expert cache warm, KV prefix preserved across turns |
| `Admission` | accepts, queues or rejects a request against the current lease and thermal state; rejection is a typed outcome, not an exception |
| `Preemption` | when the foreground app needs memory, shrink or suspend a session; checkpoint what can be restored |
| `LocalApi` | in-process binding for the agent; optionally an OpenAI-compatible localhost endpoint so third-party tools and other runtimes can use the same portfolio |

### 3.6 Layer D — Agent Runtime

Detail in [`07_AGENT_RUNTIME.md`](07_AGENT_RUNTIME.md).

`IntentParser` → `TaskPlanner` → `Executor` (observe → decide → act → verify) with `ToolRegistry`,
`ContextManager`, `SafetyGate`, `Recorder`.

### 3.7 The RuntimeGovernor — the component that makes the system adaptive

The governor sits outside the layer stack because it observes all of it. It is the mechanism behind
decision **D6**, and it is the piece that makes this a *platform* rather than a configuration file.

Every `ExecutionPlan` contains a prediction and a **falsification rule**: a tolerance band and a
minimum number of observations. The governor:

1. receives per-turn and per-token telemetry;
2. compares observed sustained rate, memory retention and energy against the plan's prediction;
3. if observations fall inside the band, does nothing;
4. if they fall outside, **invalidates the plan**, moves to the next rung of its degradation ladder
   to keep serving, records a calibration datum, and asks the planner to re-plan with the new
   measurement as a *measured* input;
5. if the device state changes materially (thermal level, charging, foreground app, storage move),
   re-plans without waiting for drift.

> **The runtime is a continuously re-validated experiment.** A plan is a registered prediction, an
> execution is the measurement, and a falsification is not an error — it is the event that upgrades
> a device's profile from inferred to measured. This is the repository's research method turned
> into a control loop, and it is the answer to "how should runtime optimisation work".

---

## 4. The two control loops

**Slow loop — planning.** Triggered at install, on a new model, on a device-state change, on plan
invalidation, or on a scheduled calibration window.

```
DeviceProfile ─┐
               ├─▶ PerformanceModel ─▶ PerfPrediction{point, interval, basis, calibration_state}
ModelCard ─────┘                                │
                                                ▼
                       ConfigurationPlanner ─▶ ExecutionPlan{config, ladder, falsification_rule}
                                                │
                                                ▼
                                          SessionManager binds
```

**Fast loop — governing.** Per token, per turn.

```
EngineSession ─telemetry─▶ RuntimeGovernor ─┬─ within band ──▶ continue
                                            ├─ outside band ─▶ step down ladder + record datum
                                            ├─ memory trim ──▶ shrink lease (ladder order)
                                            └─ state change ─▶ request re-plan
```

The loops are deliberately asymmetric: planning is allowed to be slow and expensive and to run off
the critical path; governing must be cheap enough to run every token and must never block decode.

---

## 5. Data flow: one agent turn, end to end

1. **Intent.** The user states a task. `IntentParser` produces a `TaskSpec` with a task class, a
   latency target, a quality floor and a privacy class.
2. **Capability request.** The agent issues a `CapabilityRequest` — *what* it needs (for example:
   structured tool-call output, a 4,000-token context, under 3 seconds to first action, strictly
   on-device) — never *which model*.
3. **Binding.** The planner returns a `ModelBinding`: a model, a configuration, a predicted
   prefill and decode rate with intervals, a memory lease, and the degradation ladder. If nothing
   satisfies the constraints, it returns an explained refusal with the nearest feasible relaxation.
4. **Admission.** `SessionManager` acquires the lease. If the lease is refused, the binding's ladder
   is consulted before the request is rejected.
5. **Observation.** `ContextManager` builds the prompt with a **stable prefix** (system instructions
   and tool schemas, unchanged across turns so the KV is reusable) followed by the **volatile
   suffix** (the current screen or state snapshot). This ordering is a hard requirement, not an
   optimisation: it is what makes prefix reuse possible at all on a workload whose observation
   changes every turn.
6. **Prefill and decode.** The engine runs the turn on the bound placement. Telemetry streams to the
   governor.
7. **Action.** The model emits a structured tool call. `SafetyGate` classifies it: reversible actions
   proceed; irreversible or outward-facing ones require consent and are logged.
8. **Verification.** The tool's postcondition is checked against observed device state. A failed
   verification is a retry with evidence, not a silent continue.
9. **Escalation, if needed.** On repeated failure or on an explicit low-confidence signal, the agent
   re-issues the `CapabilityRequest` with a higher quality floor. The planner may bind the streamed
   large tier — and **the predicted cost of that escalation is returned before it is paid**, so the
   agent (or the user) can decline it.
10. **Record.** `Recorder` writes the turn: intent, binding, prediction, observation, action,
    verification, consent. This is both the audit trail and the evaluation dataset, and it is the
    only source from which `T` and `S` ([`01_RESEARCH.md`](01_RESEARCH.md) §2) may be computed.

---

## 6. Runtime behaviour under pressure

The three pressures a phone applies, and the defined response to each. These are the behaviours that
distinguish a platform from a demo, and none of them has been measured yet (`PL-S2`).

### 6.1 Memory pressure

Trigger: an OS trim callback, a drop in available memory, or a failed lease heartbeat.

Response, in this fixed order — **shrink before you swap, swap nothing you can re-read**:

1. drop speculative and prefetch structures;
2. shrink the expert cache to the plan's floor (it re-reads from storage; it costs rate, not
   correctness);
3. quantise or trim KV for inactive sessions;
4. suspend the largest inactive session, checkpointing its KV;
5. unload the escalation tier entirely and rebind to the resident tier;
6. reject new admissions with a typed `MemoryUnavailable`.

Rungs 1–3 are reversible without reloading a model. Rung 5 costs a model load and is therefore the
last resort before refusal. **Anonymous memory that the OS can push to swap is the enemy**: the
prior project measured a configuration whose cache changes pushed hundreds of megabytes into
compressed swap and turned every touch into a synchronous fault, with a compute penalty that
erased the gain.

### 6.2 Thermal pressure

Trigger: a thermal-status change or a measured drop in sustained rate.

Response: step down the plan's **thermal ladder**, which was computed at plan time and is ordered by
*quality preserved per millisecond recovered*: reduce context window, then reduce cache (fewer
active lanes, less heat), then change placement, then change tier, then defer non-urgent work until
the device cools. Hysteresis is mandatory — a ladder that oscillates costs more than the throttling
it avoids, and X6 in [`01_RESEARCH.md`](01_RESEARCH.md) §6 exists to measure exactly that.

### 6.3 Interactivity pressure

Trigger: the user is waiting.

Response: the agent's latency target is part of the binding, so the planner has already chosen a
configuration that meets it or refused. At runtime, the governor may **preempt a background job**
(summarising, indexing, a queued task) to give an interactive turn the whole device. Background work
is always checkpointable by construction; that constraint is imposed on tool authors rather than
handled afterwards.

---

## 7. Failure modes, and the behaviour each one gets

An explicit list, because the failure behaviour is where a platform is judged.

| failure | detected by | behaviour |
|---|---|---|
| device profile is stale (OS update, storage moved, probe-suite version bump) | `ProfileStore` version check | mark profile `stale`; predictions downgrade to priors with wide intervals; schedule re-profiling |
| prediction drifts persistently from observation | `RuntimeGovernor` | invalidate plan, re-plan, record calibration datum; after repeated drift, mark the *profile* suspect, not just the plan |
| model file corrupt or truncated | hash check at acquisition and at load | quarantine, do not bind, offer re-download |
| transform fails its fidelity gate | `FidelityService` | discard the transform, keep the untransformed weights, record the failure; **never adopt with a caveat** |
| backend produces wrong numerics (a real risk on experimental NPU paths) | fidelity gate per device and model | demote that backend for that model, fall back to CPU, record |
| memory lease revoked mid-turn | lease heartbeat | ladder response §6.1; if the turn cannot complete, fail the turn with a typed error and preserve the task state |
| storage path too slow for the bound plan | acquisition probe and runtime read-rate monitor | refuse streaming configurations on that path; recommend moving the model |
| device sleeps mid-run | wakefulness sampling | mark the measurement invalid (never average it in); hold a wake lock for foreground work only |
| agent loops without progress | `Executor` progress detector | stop after a bounded number of non-progressing steps, report state, ask the user |
| tool action is irreversible | `SafetyGate` classification | require explicit consent; never infer consent from a prior grant in another context |

---

## 8. Process and threading model

- **One foreground service** hosts the inference session; it holds the lease and the wake lock, and
  it is the only component that may pin threads.
- **The engine runs in its own process**, so that a native crash or an OOM kill does not take the app
  with it, and so the OS accounts its memory separately. The existing engine already supports this
  shape: a persistent process serving structured requests over a stream.
- **Compute threads are pinned** to the cluster chosen by the profile; **I/O lanes are pinned
  elsewhere**. This is not tuning. The measured difference between pinned and unpinned on the same
  four cores was 30.1 [E:cliff_pinned] against 4.0 [E:cliff_unpinned] tok/s.
- **The governor runs on a separate low-priority thread** and never blocks decode; telemetry is
  written to a lock-free ring and drained asynchronously.
- **Probes never run concurrently with serving.** A measurement taken while the engine is working
  measures the engine, not the device.

---

## 9. What "plug and play" means concretely

The user-visible contract, which the architecture exists to support:

1. Install. The platform profiles the device in seconds and shows a capability report in plain
   language: what this phone can run, how fast, and how confident the estimate is.
2. Pick a model — from a catalogue or by pasting a URL. The platform says, before downloading, what
   it expects: tokens per second at a stated context, memory it will need, whether it must stream,
   and how much of the estimate is measured versus inferred.
3. Download. The platform verifies the storage path, transforms the weights for this device if that
   helps, and proves the transform did not change the model's answers.
4. Use. The agent does tasks. When it needs more capability than the current tier provides, it says
   what escalation would cost and asks or proceeds according to policy.
5. Over time, every run makes the profile more measured and the intervals narrower. **The platform
   gets more accurate with use, not more confident with use** — those are different, and only the
   first one is honest.
