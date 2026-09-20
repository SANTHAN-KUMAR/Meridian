# 1 — Research concepts: what is being claimed, what is estimable, and what would falsify it

This document separates the **research** content of the platform from its engineering. It follows
the validity levels of [`../../CLAUDE.md`](../../CLAUDE.md) §1: a quantity is defined (L1), shown to
be a functional of something observable (L2), shown to be estimable at the sample sizes available
(L3), and only then estimated.

Nothing in this document has been measured. It states what *would* be measured and what each
outcome would mean. Where the prior phase already measured something, it is quoted through
[`EVIDENCE.md`](EVIDENCE.md).

---

## 1. The three research questions the platform contains

The platform is an engineering artifact, but three genuinely open questions sit inside it. Each is
stated so that both outcomes are informative.

**RQ1 — Predictability.** *Can the sustained performance of a given (model, configuration) on an
unseen phone be predicted, before running it, to a usefully narrow interval, from probes that cost
seconds?* A "no" is a strong result: it says on-device benchmarking is unavoidable, and the product
must be built around measuring rather than predicting.

**RQ2 — Coexistence.** *How much memory and compute can an inference service actually hold while the
app it is driving is in the foreground, and how does that change the achievable configuration?*
This has never been measured by anyone in this project's record and it gates the whole agent
premise ([`00_PROBLEM.md`](00_PROBLEM.md) §8.5).

**RQ3 — Escalation value.** *On real phone tasks, does a model too large for RAM, streamed from
flash, complete tasks that a resident small model fails — by enough to pay for being roughly 7x
slower at decode and far slower at prefill?* A "no" retires the streaming tier to a niche and
simplifies the product enormously. A "yes" is the strongest possible justification for the
`moe-phone` engine's existence.

RQ3 is the question the prior project never asked, because it optimised the streamed model's speed
without ever testing whether the streamed model was needed.

---

## 2. L1 — The estimands

### 2.1 Primary: task latency and task success, jointly

Adopted from [`../agentic/ESTIMAND.md`](../agentic/ESTIMAND.md) §1, which derived them properly and
whose reasoning is not repeated here.

For a task `k` from a fixed suite `K`, on device `d`, under platform configuration `c`:

- **`T(k, d, c)`** — wall-clock seconds from a complete task instruction to the final action that
  completes the task correctly. **Unit of analysis: one task.** Reported as median with IQR over
  `K`, defined only over tasks completed correctly.
- **`S(d, c)`** — the fraction of `K` completed correctly, judged by per-task success predicates
  fixed before the run.

**`T` and `S` are reported together or not at all**, because the cheapest way to lower `T` is to
fail faster. A configuration that is faster at lower `S` has not won.

> **Prohibition, inherited and restated:** a token-averaged decode rate may never be substituted for
> `T` at any point. They average over different populations — tokens within long generations versus
> tasks composed of many short turns, each paying a fresh prefill and a fresh cache disturbance. The
> `aegis/` project in this repository lost two papers to exactly this class of substitution.

### 2.2 Secondary: energy and retention

- **`J(k, d, c)`** — joules per completed task, from the battery counter (current × voltage) with
  the display and radio contribution named. A phone agent that drains the battery is not a product.
- **`R(d, c)`** — **memory retention**: the fraction of a session during which the platform held its
  granted memory lease without revocation. This is an availability metric, and it is the quantity
  RQ2 is really about.

### 2.3 The planner's own estimand, which is a different thing

The planner is itself an estimator, so it has an estimand of its own:

> **`P(m, c, d)`** — the sustained steady-state rate (prefill tok/s at length `n`, decode tok/s at
> context `n`) that configuration `c` of model `m` will deliver on device `d`, averaged over the
> deployment regime: awake, unplugged, thermally settled, with the target app foreground.

The averaging measure is stated explicitly because it is where prior work in this area quietly
cheats: a burst number taken on a cool, plugged-in, quiesced phone is a different quantity, usually
by a large factor, and the prior project's own campaigns were invalidated twice by exactly this
([`../results/2026-09-19/`](../results/2026-09-19/), the autosuspend finding).

**The planner is evaluated on prediction error against measured `P`**, and separately on whether
using its choice improves `T` and `S` over the baselines in §4.

---

## 3. L2 — Identification: what has to be true for these to be estimable

Stated as assumptions over named quantities, each marked testable or not, each pointing at what
would enforce it.

| # | assumption | testable? | enforced by |
|---|---|---|---|
| A1 | the probe suite's measurements are of the same hardware regime the engine later runs in (same thread placement, same request sizes, same thermal state) | yes | the probes use the engine's own kernels and I/O path, not synthetic stand-ins ([`04_DEVICE_PROFILING.md`](04_DEVICE_PROFILING.md) §3) |
| A2 | a run's validity conditions are observable at the time of the run | yes | the validity guard: wakefulness, foreground, charging, thermal status, descheduling ([`04_DEVICE_PROFILING.md`](04_DEVICE_PROFILING.md) §5) |
| A3 | the task suite `K` is representative of the intended workload | **no** — a judgement | declared with the suite; the suite is public and fixed before any run |
| A4 | success predicates are decidable from observable device state | yes | each task ships a programmatic predicate; tasks that need a human judge are marked and reported separately |
| A5 | the device grants memory in a way that is stable enough to plan against | yes | RQ2; the retention metric `R` |
| A6 | model quality is preserved by every automatic transform the platform performs | yes | the fidelity gate ([`06_EXECUTION_ENGINE.md`](06_EXECUTION_ENGINE.md) §8) |

**Positivity / overlap, in this setting.** The planner can only learn which configuration is better
if the design actually varies configuration. A platform that always picks the same tier produces no
variation and can never validate its own choices. **Consequence for the design:** the evaluation
harness must run a *fixed, pre-registered* configuration grid, not the configurations the planner
happens to prefer. This is a real constraint on how the product's telemetry may be used as evidence.

---

## 4. L3 — Estimability, and the baselines that make a result mean anything

### 4.1 The three mandatory floors

[`../../CLAUDE.md`](../../CLAUDE.md) §8 requires three baselines. Instantiated here, they are
unusually sharp, because the first of them is exactly what every competing product does:

| floor | instantiated | rules out |
|---|---|---|
| **no-data baseline** | the spec-sheet calculator: nameplate memory bandwidth ÷ active bytes per token, with a fixed quantisation rule of thumb | that the platform's measured profiling has **negative** information content. This is the strong baseline: the prior phase's own attempt at it missed a second device below its stated range ([`00_PROBLEM.md`](00_PROBLEM.md) §4) |
| **best constant** | always choose the same (model, quantisation, placement) for every device and task | that the platform's "adaptivity" beats not adapting |
| **one-parameter shrinkage** | the spec-sheet estimate multiplied by a single scalar fitted per device class | that the whole performance model beats one number of calibration |

A planner that does not beat all three on held-out (device, model) cells has demonstrated nothing,
however good its engineering is.

### 4.2 The oracle check, run first

Before any comparison is reported, run the oracle: a planner handed the *measured* outcome for every
candidate configuration. If the oracle's task-level advantage over the best constant is small, then
**the benchmark cannot distinguish a good planner from a bad one** and no planner result may be
reported from it ([`../../CLAUDE.md`](../../CLAUDE.md) §4.3). This is cheap and it should be the
first experiment, not the last — it is the check the prior `aegis/` project failed to run, at the
cost of two papers.

### 4.3 The estimability question the platform will actually face

The decisive ratio is `SE(prediction error) / |difference between configurations|`. If two candidate
configurations differ by less than the run-to-run noise of the device, the planner cannot choose
between them and must not pretend to: it reports them as tied and picks by a declared tie-break
(lower energy, then lower memory). Device run-to-run spread is measured by the probe suite itself —
the storage probe already reports a 9% median run-to-run relative spread on the primary device — and
the planner's resolution is derived from it rather than assumed.

---

## 5. Novelty position, with its search protocol

Per [`../../CLAUDE.md`](../../CLAUDE.md) §10.4, a novelty claim needs a reproducible protocol and a
date, and must distinguish *verified absent* from *not found*.

**Protocol.** Tool: `WebSearch`, 2026-09-20. Queries run, verbatim:
1. `on-device Android agent framework accessibility tree LLM AndroidWorld benchmark 2026`
2. `Qualcomm QNN Genie SDK on-device LLM Snapdragon NPU third-party app unsigned PD 2026`
3. `local LLM app device capability detection recommend model RAM quantization automatic "tokens per second" estimate`
4. `small on-device model mobile GUI agent function calling 2026 Qwen3 4B UI-TARS mobile tool use Android`
5. `llama.cpp Android Hexagon NPU backend ggml-hexagon 2026 status Adreno OpenCL MoE`

No venue index, issue tracker or repository history was crawled. **Every verdict below is therefore
"not found", never "verified absent",** and a paper claim would need the fuller protocol.

### 5.1 Found — and therefore not claimable

| prior work | what it already does |
|---|---|
| Qualcomm **GenieX** (BSD-3) and the Genie SDK | runs GGUF models on Snapdragon NPU, GPU and CPU, with a CLI, Android bindings and an OpenAI-compatible local server. **Running local models on the Snapdragon NPU is not novel, and this is a component to adopt rather than reimplement** ([`09_VENDOR_INTEGRATION.md`](09_VENDOR_INTEGRATION.md) §3) |
| **Qwen-UI-Agent**, **GUI-Owl-1.5**, **Mobile-Agent-v3.5**, **V-Droid**, **AndroidWorld**, **SPA-Bench** | mobile GUI agents, their benchmarks and their evaluation harnesses exist and are mature. A phone agent is not novel; the benchmark suites are assets to reuse |
| **llmfit**, **ModelFit**, **LLM Configurator** | "which model fits my hardware" tools. All PC-oriented; all derive tokens per second from nameplate bandwidth and parameter count; none measures the device, models thermal derating, or distinguishes requested from granted memory |
| `ggml-hexagon` / llama.cpp Snapdragon backends | an NPU backend exists upstream and is described as experimental, with reports of corrupted output on some models; the OpenCL/Adreno path is the working alternative. **Consequence: the NPU path must be validated by the fidelity gate on every device, not trusted** |

A useful negative from the same search: a technical report evaluating deployment-viable 3B-class GUI
agents places their success rate below a practically usable threshold, which is direct external
support for the escalation tier in RQ3.

### 5.2 Not found (2026-09-20) — the candidate contributions

Stated as candidates, in descending order of confidence that they are real:

1. **A measured, interval-valued, falsifiable device-to-configuration planner for phones**, whose
   predictions carry a calibration state and which refuses to extrapolate. The tools found in §5.1
   are spec-sheet calculators; none was found that measures the device, none reports uncertainty,
   and none was found to model sustained thermal derating.
2. **Sustained-regime performance as the contracted quantity**, with a time-to-throttle and a
   pre-planned degradation ladder rather than a burst number, and with validity instrumentation
   (autosuspend, foreground loss, descheduling) as a product component.
3. **Memory modelled as a revocable lease held jointly over models, experts and KV**, sized from
   measured grant behaviour rather than from requests.
4. **Beyond-RAM MoE streaming used as an escalation tier inside an agent runtime**, with the
   escalation decision made on *predicted task latency* rather than on model size.

None of these is a modelling advance. They are systems contributions of the form "the quantity
everyone reports is the wrong quantity, here is the right one and here is what it costs to get it" —
which is the same shape as this project's benchmark-validity finding, and the shape its record is
strongest at defending.

### 5.3 The honest framing for a hackathon versus a paper

For a hackathon, novelty claims are not the currency and should not be made: the demonstrable facts
are that the platform measures a phone it has never seen, predicts before it runs, and is right or
visibly says it does not know. For a paper, §5.2 needs the verified-absent protocol, a device matrix
and pre-registered predictions per cell — which is exactly the plan the prior phase already wrote
([`../research/2026-09-19_PAPER_PLAN.md`](../research/2026-09-19_PAPER_PLAN.md) §4).

---

## 6. The experiments that would falsify this architecture, cheapest first

Ordered by how early they can kill the design, per [`../../CLAUDE.md`](../../CLAUDE.md) §4.3.

| # | experiment | kills what, if it fails |
|---|---|---|
| **X1** | **Foreground coexistence.** Drive a real app while the engine holds a model; measure granted memory, revocations, and the achievable tier over a 10-minute session | the whole agent premise. If a useful model cannot be held while an app is foreground, the product is a chat app, and that is a publishable negative result about on-device agents |
| **X2** | **Prefill versus prompt length**, at 128 / 512 / 2048 / 8192 tokens, per tier and per backend | the tiering rationale in [`00_PROBLEM.md`](00_PROBLEM.md) §2, which currently rests on one 64-token benchmark row per backend and on a prefill figure measured at a 26-token [E:gptoss_prompt_tokens] prompt |
| **X3** | **The oracle check** (§4.2) on a small device × model × configuration grid | the entire planner evaluation. Run before the planner is built |
| **X4** | **Planner versus the three floors** (§4.1) on held-out cells | the claim that measured profiling beats a spec-sheet formula — i.e. candidate contribution 1 |
| **X5** | **Escalation value** (RQ3): resident small tier versus streamed large tier on the same task suite, success and latency | the streaming tier's role in the product |
| **X6** | **Thermal ladder fidelity**: does the degradation ladder hold the promised rate over 30 minutes, or does it oscillate? | the sustained-rate contract |

X1 and X2 use scripts that already exist in the repository and have never been run
([`../device/bmoe_prefill.sh`](../device/bmoe_prefill.sh) is written for X2 and documents why the
existing number is not the number).

---

## 7. Gates, derived for this problem

Not copied from the engine's gate list — derived, per [`../../CLAUDE.md`](../../CLAUDE.md) §9.2, by
asking what each gate class means for a *planner and a runtime* rather than for a kernel. These run
on 100% of cells, never a sample.

| class | gate | its expected value is fixed by |
|---|---|---|
| **Invariance** | doubling a model's active bytes per token, at fixed placement and hit rate, must halve the predicted decode rate; halving KV bytes per token must move predicted memory by exactly the bytes saved | arithmetic, independent of whether the model is any good |
| **Recovery** | for a model that fits entirely in RAM with no streaming, the planner must reproduce the measured plain-runtime rate within its stated interval | the measurement itself |
| **Degeneracy** | a plan whose predicted interval spans the decision boundary between two configurations must return `tied` or `calibrate`, never a choice | the definition of a decision under uncertainty |
| **Monotonicity** | more cache must not lower predicted hit rate; faster measured storage must not lower predicted rate; the oracle plan must not lose to the planner | the definition of a cache and of an oracle |
| **Physical plausibility** | no predicted rate may exceed the device's own measured bandwidth ceiling for that model; energy per token must fall inside the published external range | external measurement |
| **Floor** | the planner beats all three baselines of §4.1 on held-out cells | [`../../CLAUDE.md`](../../CLAUDE.md) §8 |
| **Artifact** | prediction error must not correlate with thermal state, run order, or battery level across the campaign | the campaign's own state logs |
| **Fidelity** | every automatically adopted transform passes its KL / flip-rate / perplexity gate against the untransformed model on that device | the format floor: the quantisation the user already accepted, measured at 0.118 nats [E:e6_floor_kl] mean KL on the primary device |

The **Degeneracy** gate is the one that will be inconvenient in practice and it is the one that must
not be relaxed. Returning "I do not know, let me measure for 40 seconds" is a feature; returning a
confident number from an uncalibrated model is the failure mode this whole repository exists to
prevent.

---

## 8. Registered stubs — what does not exist

Per [`../../CLAUDE.md`](../../CLAUDE.md) §7.5. Everything in this platform is currently a stub; these
are the ones whose absence would otherwise be invisible in the architecture documents, because the
documents describe them as though they work.

| id | what is missing | current state | removed when |
|---|---|---|---|
| `PL-S1` | the task suite `K` and its success predicates | does not exist | a suite is written and frozen before any run; candidates from the public benchmarks in §5.1 |
| `PL-S2` | any measurement of foreground coexistence | never attempted | X1 runs |
| `PL-S3` | prefill versus prompt length, any device, any tier | script exists, never run | X2 runs |
| `PL-S4` | the cross-device calibration set the performance model is fitted on | one device fully, one device partially | the device matrix of the paper plan is measured |
| `PL-S5` | energy accounting per completed task | coarse battery-counter method only; per-rail access is unavailable to an unprivileged app on the primary device | a device grants rail access, or the coarse method is validated against one that does |
| `PL-S6` | the NPU path's numerical fidelity on any target device | one existence proof that an LLM runs on the NPU of an unrooted retail phone; upstream reports corrupted output on some models | the fidelity gate is run per device and per model |
| `PL-S7` | quality evaluation of the small tier on real tasks | none | X5 runs |
