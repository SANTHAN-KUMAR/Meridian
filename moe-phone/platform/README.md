# Meridian — a mobile AI compute and agent platform

**Status: architecture and design documentation. No implementation exists yet. Nothing here has been
measured; everything measured is from the `moe-phone` research phase and is quoted through
[`EVIDENCE.md`](EVIDENCE.md).**

Meridian is the evolution of `moe-phone` from a single-device systems study into a platform: a phone
decides for itself what models it can run, how to run them, and how fast they will actually be, and
an on-device agent uses that to do everyday work for its owner.

> **The name is a placeholder.** It appears only in prose and in the module prefixes suggested in
> [`03_INTERFACES.md`](03_INTERFACES.md). Renaming it is a find-and-replace, and doing so changes no
> design decision.

---

## Read in this order

| # | document | what it decides |
|---|---|---|
| 0 | [`00_PROBLEM.md`](00_PROBLEM.md) | What the core problem actually is, what the measured evidence supports, what in `moe-phone` survives the pivot, what is generalised, what is redesigned, and the defects in the prior research that this architecture must not inherit |
| 1 | [`01_RESEARCH.md`](01_RESEARCH.md) | The research layer: the estimands, what is identified and estimable, the novelty position with its dated search protocol, the gate list, and the experiments that would falsify the design |
| 2 | [`02_ARCHITECTURE.md`](02_ARCHITECTURE.md) | The system: layers, components, responsibilities, the two control loops, data flow, runtime behaviour, failure modes |
| 3 | [`03_INTERFACES.md`](03_INTERFACES.md) | The contracts, in schema form: `DeviceProfile`, `ModelCard`, `PerfPrediction`, `ExecutionPlan`, `CapabilityRequest`, `ModelBinding`, the engine, tool and telemetry APIs, and the error taxonomy |
| 4 | [`04_DEVICE_PROFILING.md`](04_DEVICE_PROFILING.md) | How a device is measured, how a measurement is known to be valid, and what to do when it is not |
| 5 | [`05_PERFORMANCE_MODEL.md`](05_PERFORMANCE_MODEL.md) | How performance is predicted, calibrated, bounded, falsified — and when the honest answer is to refuse to predict |
| 6 | [`06_EXECUTION_ENGINE.md`](06_EXECUTION_ENGINE.md) | The compute layer: tiers, placement, residency, MoE streaming, memory and storage, autotuning, fidelity |
| 7 | [`07_AGENT_RUNTIME.md`](07_AGENT_RUNTIME.md) | The agent: the loop, context management, tools, permissions, verification, recovery, and the compute contract it holds with the planner |
| 8 | [`08_APP_AND_UX.md`](08_APP_AND_UX.md) | The Android app: stack, process model, screens, how uncertainty is shown on screen, and the frame and memory budget the UI must hold |
| 9 | [`09_VENDOR_INTEGRATION.md`](09_VENDOR_INTEGRATION.md) | Hardware abstraction, what stays vendor-neutral, the Snapdragon/iQOO integration map, and what each deeper hook would buy |
| 10 | [`10_EXTENSION_POINTS.md`](10_EXTENSION_POINTS.md) | Conformance levels, extension seams, the stub registry, the open risks, and where to start |

An implementer who reads 0, 2 and 3 can start building; add 8 to build the app. An implementer who
reads only 2 will build something that returns confident numbers it has no right to.

**Shortest defensible path:** [`10_EXTENSION_POINTS.md`](10_EXTENSION_POINTS.md) §7 names the order,
and §1 of the same file defines the conformance levels a partial implementation may honestly claim.

---

## The rules these documents follow

This repository's working method is in [`../../CLAUDE.md`](../../CLAUDE.md) and it applies here.
Three of its rules shape every page:

1. **No number is written by hand.** Each figure appears as `` value [E:id] `` and is verified against
   the artifact that produced it:
   ```bash
   python moe-phone/platform/tools/evidence.py          # verify every document against the artifacts
   python moe-phone/platform/tools/evidence.py --emit   # regenerate EVIDENCE.md
   ```
   A number without a marker is either arithmetic shown inline, or an external literature value with
   its source, or a design parameter with no empirical claim attached. There are no other cases.

2. **Measured, inferred and assumed are never mixed silently.** Every field of every contract in
   [`03_INTERFACES.md`](03_INTERFACES.md) carries its provenance, and code paths that consume them
   must branch on it. This is not documentation hygiene — it is the mechanism that stops a
   spec-sheet guess from being served to a user as a performance promise.

3. **A negative result is a result.** The `moe-phone` phase closed with a measured "no" to its
   headline goal, and that "no" is the most valuable input this architecture has, because it says
   precisely where the compute is and is not. A design that quietly assumes the "no" away is worse
   than no design.

## What this is not

- Not an implementation plan, a schedule, or a task breakdown.
- Not a claim that any component here is novel; [`01_RESEARCH.md`](01_RESEARCH.md) §5 states the
  novelty position with its search protocol and date, and most of the platform is deliberately
  assembled from existing parts.
- Not a promise that the architecture is right. It is falsifiable, and §6 of
  [`01_RESEARCH.md`](01_RESEARCH.md) lists the cheapest experiments that would kill it.
