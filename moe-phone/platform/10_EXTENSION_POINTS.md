# 10 — Conformance levels, extension seams, stubs and open risks

What a partial implementation is allowed to be, where the system is designed to grow, what does not
exist yet, and what could still be wrong with the whole thing.

---

## 1. Conformance levels

An implementation does not have to be complete to be correct. These levels exist so that a team can
build a defensible subset and say precisely what it is — rather than shipping something that looks
complete and silently skips the parts that keep it honest.

### L0 — Runner
Loads a model, runs it, streams tokens to a UI.
**Required:** the engine, the resident tier, the session protocol, the app shell.
**Must still hold:** typed refusals instead of crashes; no performance claims of any kind.

### L1 — Profiled runner
L0, plus the device profiler at T0/T1 and a capability report.
**Required:** `DeviceProfile`, the validity guard, feasibility decisions.
**Must still hold:** every number carries a basis chip; nothing outside the measured range gets a
point value.

### L2 — Planned runner
L1, plus the performance model, the planner, and `ExecutionPlan` with a degradation ladder.
**Required:** intervals, calibration states, refusals, the memory lease protocol.
**Must still hold:** the planner beats the spec-sheet baseline on the devices it has been checked on,
or the app says it is estimating.

### L3 — Governed platform
L2, plus the runtime governor, falsification-driven re-planning, calibration scheduling, the
streamed tier, transforms with fidelity gates.
**Required:** T2/T3 profiling, autotune, `TransformRecord`s.

### L4 — Agent platform
L3, plus the agent runtime, tools, safety gate, verification, audit trail and the evaluation harness.
**Required:** a frozen task suite and the three baselines; `T` and `S` reported together.

**A demo should state its level.** An L1 implementation that honestly reports what it measured is
worth more than an L4 that fakes the parts it skipped, and the difference is visible to anyone who
looks for a refusal and does not find one.

---

## 2. Extension seams

Places the architecture is explicitly built to grow, with the contract each new thing must satisfy.

| seam | to add | contract it must satisfy |
|---|---|---|
| **model architecture registry** | a new mixture-of-experts family | one row naming its expert-tensor suffixes, plus a byte-identity check. No change to the streaming path |
| **`Backend`** | a new accelerator or runtime | capability declaration, measured admission, fidelity gate, automatic demotion ([`09_VENDOR_INTEGRATION.md`](09_VENDOR_INTEGRATION.md) §4) |
| **`VendorPack`** | a silicon vendor's stack | detection by execution; nothing required at build time |
| **probe** | a new device measurement | a validity condition, a rejection rule, a profile field with provenance, and a tier budget |
| **transform** | a new on-device weight optimisation | a `TransformRecord` with pre-registered thresholds and a gate that runs before adoption |
| **ladder rung** | a new degradation | an expected cost, a reversibility flag, a hysteresis window, and a bounded quality cost |
| **tool** | a new device capability for the agent | a `ToolSpec` with effects, reversibility, consent class, cost hint, named postcondition, checkpointability |
| **skill** | a deterministic procedure for a recurring task | the same postcondition contract as a tool, plus a match rule against `TaskSpec` |
| **task class** | a new kind of work | a capability request profile, a step budget, and suite tasks with programmatic predicates |
| **baseline** | a new comparison | a fixed definition that does not depend on the platform's own choices |

Anything added at a seam without its contract is a stub, and belongs in §4.

---

## 3. Deliberate non-goals

Named so nobody adds them by drift.

- **Training, fine-tuning or adapters on device.** The whole premise is stock checkpoints. Adding
  training changes the memory, thermal and quality arguments entirely.
- **A cloud fallback that silently answers when the device cannot.** If it exists at all it is an
  explicit, per-task, user-visible choice, because `on_device_only` is the default and the product's
  point.
- **A curated model whitelist as the only path.** The architecture registry seam exists so that any
  compatible checkpoint works; a catalogue is a convenience over it, never a gate.
- **Beating cloud models on quality.** The claim is usefulness on a phone, privately, at a known
  cost.
- **A general computer-use agent.** Scope is everyday phone tasks
  ([`07_AGENT_RUNTIME.md`](07_AGENT_RUNTIME.md) §1).

---

## 4. Stub registry

Per [`../../CLAUDE.md`](../../CLAUDE.md) §7.5: location, current behaviour, correct behaviour,
removal condition. This extends the research stubs of [`01_RESEARCH.md`](01_RESEARCH.md) §8 with the
engineering ones.

| id | what | current state | removed when |
|---|---|---|---|
| `PL-S1` | task suite and success predicates | does not exist | a frozen, versioned suite exists |
| `PL-S2` | foreground-coexistence memory grant | never measured, anywhere | experiment X1 runs |
| `PL-S3` | prefill versus prompt length | script exists, never run | experiment X2 runs |
| `PL-S4` | cross-device calibration set | one device fully, one partially | the device matrix is measured |
| `PL-S5` | energy per completed task | coarse battery-counter method only | validated against a per-rail method, or a device grants rails |
| `PL-S6` | NPU numerical fidelity per device and model | one existence proof; upstream reports corruption on some models | the gate runs per device and model |
| `PL-S7` | small-tier task quality | unmeasured | experiment X5 runs |
| `PL-E1` | the planner itself | does not exist; the analytic prior is a formula in a document | L2 conformance |
| `PL-E2` | the runtime governor | does not exist | L3 conformance |
| `PL-E3` | grant probe | never written; the prior project observed grants passively | T1 profiling ships |
| `PL-E4` | thermal derate curve as a reusable artifact | measured once, ad hoc, on one device | T3 profiling ships |
| `PL-E5` | device-class priors | none; there is no fleet | a calibration set exists across ≥ 3 devices |
| `PL-E6` | vision path for extraction tasks | no vision model bound, no OCR path | a vision-capable tier is profiled and gated |
| `PL-E7` | grammar-constrained decoding wired to the tool schemas | the engine supports constrained decoding; nothing generates grammars from `ToolSpec`s | the tool registry emits grammars |

**An unregistered stub is a false claim.** Anything an implementer skips gets a row here.

---

## 5. Open risks, ranked by how much they would cost

### R1 — Foreground coexistence kills the premise
If an inference session cannot hold a useful model while the target app is foreground, the agent
cannot work as designed and the product becomes a chat app with tools that only run when it is the
only thing running. **Cost if true: the product.** **Mitigation: X1 first, before anything else is
built.** A negative here is a genuine, publishable finding about on-device agents, and the
architecture degrades to an L2 "local model platform" that is still worth shipping.

### R2 — The small tier is not capable enough
Deployment-viable small models are reported in the literature to fall below practical usability on
GUI-agent benchmarks. If the resident tier cannot drive tasks and the streamed tier is too slow at
agent context lengths — where compute alone measured 144.3 ms/token [E:qwen3_e1_long_ms] — then the
useful task set is narrow. **Mitigation:** the skill library (deterministic procedures) carries the
tasks a model is not needed for, and the escalation path is measured rather than assumed; X5 sizes
the gap early.

### R3 — Prediction does not beat the spec sheet
RQ1 answers "no". **Mitigation:** the product pivots from prediction to fast on-device measurement,
which the architecture already supports as the preferred basis. Smaller claim, same code.

### R4 — Thermal oscillation
The ladder fights the thermal controller and the user sees a system that speeds up and slows down
unpredictably, which is worse than being uniformly slower. **Mitigation:** hysteresis is part of the
rung contract; X6 measures it over 30 minutes.

### R5 — Accelerator fidelity
A vendor or experimental path silently produces wrong numbers — a documented risk on at least one
upstream NPU backend. **Mitigation:** the CPU reference plus a per-device, per-model gate, with
automatic demotion. This risk is *managed*, not eliminated, and the gate's cost is real.

### R6 — Permissions and platform policy
Accessibility-service and notification-listener access is increasingly scrutinised by app stores,
and an agent that drives other apps sits in exactly that category. **Mitigation:** every capability
degrades to intents and deep links; the audit trail and on-device-only default are the argument; no
single permission is load-bearing for the whole product.

### R7 — The engine's own overhead
The streaming engine was measured running identical arithmetic 1.9x [E:overhead_ratio] slower than
the plain runtime on a fully cached model. For the **resident** tier — now the default — that
overhead is pure loss, and the resident path should use the plain runtime unless streaming is
actually needed. **Mitigation:** treat tier selection as also selecting the code path, and measure
the resident tier against the plain runtime as a standing baseline.

### R8 — Scope
The architecture describes a large system. Every layer has a conformance level precisely so that a
small team can build L1 or L2 completely rather than L4 partially.

---

## 6. What would make this architecture wrong

Stated so that a future reader can check rather than assume. The design rests on four claims; if any
falls, the corresponding part must be redesigned rather than patched.

| claim | if it is false |
|---|---|
| **a phone's usable performance is measurable in seconds and stable enough to plan against** | the planner is pointless; measure per run and cache nothing (R3) |
| **the resident tier is the right default for agent work** | if prefill at agent lengths turns out to be dominated by something the small tier handles badly, the tiering inverts. X2 and X5 decide it, and both are cheap |
| **memory grants are stable enough for a lease protocol** | if grants are chaotic under foreground pressure, plans cannot be held and the system must become reactive rather than planned (R1) |
| **an agent can verify its own steps from observable device state** | if most useful actions have no checkable postcondition, the loop cannot self-correct and the product needs a human in the loop per step, which is a different product |

---

## 7. Where to start

For an implementer with no prior context, the shortest path to something real and defensible:

1. Read [`00_PROBLEM.md`](00_PROBLEM.md) and [`02_ARCHITECTURE.md`](02_ARCHITECTURE.md).
2. Build the L0 runner against the existing engine's session protocol
   ([`06_EXECUTION_ENGINE.md`](06_EXECUTION_ENGINE.md) §11), with the app skeleton and performance
   budget of [`08_APP_AND_UX.md`](08_APP_AND_UX.md).
3. Add T0/T1 profiling and the capability report — this is the first thing that is visibly *this
   product* and not another local-model app.
4. Run **X1** ([`01_RESEARCH.md`](01_RESEARCH.md) §6). It is cheap and it can invalidate the design.
   Do it before building the planner.
5. Add the planner with intervals and refusals; validate against the spec-sheet baseline.
6. Only then the governor, the streamed tier, and the agent.

Each step ends at a conformance level that can be demonstrated and defended on its own.
