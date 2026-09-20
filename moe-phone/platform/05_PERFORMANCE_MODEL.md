# 5 — The performance model and the configuration planner

The two components that turn a `DeviceProfile` and a `ModelCard` into a decision. This is the part
of the platform with the most ways to be confidently wrong, so most of this document is about the
discipline rather than the arithmetic.

---

## 1. What is being predicted, and why it is not tokens per second

The planner's estimand, restated from [`01_RESEARCH.md`](01_RESEARCH.md) §2.3:

> **`P(m, c, d)`** — the **sustained** prefill and decode rates that configuration `c` of model `m`
> delivers on device `d`, in the **deployment regime**: awake, unplugged, thermally settled, with
> the target app in the foreground.

Three consequences that are easy to skip and expensive to skip:

1. **Prefill and decode are different predictions with different bases.** On the primary device the
   ranking of compute units inverts between them. A single "speed" number is not a simplification;
   it is an error.
2. **Sustained, not burst.** A prediction that does not pass through the derate curve describes the
   first thirty seconds of a phone's life.
3. **The regime includes another app running.** Every measurement inherited from the prior phase was
   taken on a quiesced device, so every prediction derived from them is optimistic by an unmeasured
   amount until RQ2 is answered (`PL-S2`).

The user-facing quantity is still task latency ([`01_RESEARCH.md`](01_RESEARCH.md) §2.1); `P` is the
intermediate the planner optimises, and the two must never be conflated in reporting.

---

## 2. Three bases, with a strict precedence

Every field of a `PerfPrediction` is produced by exactly one of these, and says which:

| basis | what it is | when it may be used |
|---|---|---|
| **measured** | this device, this model, this configuration, valid conditions | always preferred |
| **calibrated** | the analytic model of §3, fitted to measurements *on this device*, evaluated inside the fitted range | when no direct measurement exists |
| **prior** | a device-class prior, or the analytic model outside its fitted range | only with a widened interval and a `calibration_state` of `extrapolated`, and never as the sole basis for a promise shown to a user |

Below that there is a fourth state, **uncalibrated**, which is not a basis but a `Refusal`.

**The precedence is enforced in code, not by convention.** The prediction carries `basis` as a list
of the profile fields and calibration cells it consumed, so a reviewer can see exactly which
measurements a number rests on.

---

## 3. The analytic model — a prior, and explicitly labelled as one

The structure below is the prior. It is useful because it has the right shape and the right
invariances, and it is **not** trusted as an estimate: the same structure, applied across devices
with spec-sheet inputs, was falsified on the first second device it met
([`00_PROBLEM.md`](00_PROBLEM.md) §4).

### 3.1 Turn-level decomposition

```
T_turn  =  t_admit  +  t_prefill(n_new)  +  t_decode(n_out)  +  t_tool
```

`n_new` is the prompt tokens **not** already in the KV cache. On an agent workload the stable prefix
(system instructions, tool schemas) is reused and the volatile suffix (the screen snapshot) is not,
which is why prompt construction order is a hard requirement in
[`02_ARCHITECTURE.md`](02_ARCHITECTURE.md) §5.

**`t_prefill` as a function of length is the single largest unmeasured quantity in this project**
(`PL-S3`). Every prefill figure in the record was taken at a 26-token [E:gptoss_prompt_tokens]
prompt, making 6.0 tok/s [E:gptoss_prefill_short] a cold-start artifact rather than an ingestion
rate. For a streamed MoE the expectation — registered here so it can be falsified — is that
per-token prefill cost *falls* with prompt length, because one read of an expert serves every token
in the chunk that routes to it, and then *saturates*, because a large enough chunk touches
essentially every expert. Where those cross decides whether an agent can afford to put a screen
description in its context at all.

### 3.2 Per-token cost

```
ms_per_token  ≈  ( compute_ms + stall_ms + mgmt_ms ) / derate(skin_temp)
```

with

```
compute_ms  ≈  active_bytes_per_token / kernel_throughput(placement, cluster)
stall_ms    ≈  (1 − hit) × expert_bytes_per_token / effective_read_rate(request_size, lanes)
mgmt_ms     ≈  per-engine, measured; it is not negligible and it is not constant
```

Every input on the right is a **measured** field of the `DeviceProfile` or a measured property of
the model file. None is a nameplate figure.

### 3.3 The prohibition that makes this usable

> **The three terms may never be optimised independently and then summed.**

This is the prior project's most expensive structural error, caught by its own hostile review
([`00_PROBLEM.md`](00_PROBLEM.md) §8.2): five separate levers each removed time from one term and
returned it in another, for a net of approximately zero, because the terms share one wall clock, one
memory system and one power budget. Concretely, one configuration removed 15 ms of cache management
and got back 12 ms of compute and 4 ms of stall.

Therefore:

- the decomposition **explains** a prediction and may be shown to a developer;
- the **prediction** is end-to-end wall time, fitted and validated end-to-end;
- any figure constructed by adding best cases across terms is labelled a **bound**, never an
  estimate, and never reaches a user.

### 3.4 Fitting rules

- **Fit on wall-clock quantities only.** The engine's compute term is a residual that absorbs faults,
  scheduler stalls and clock caps ([`00_PROBLEM.md`](00_PROBLEM.md) §8.1). It is not a regression
  input. Where true compute is needed, the tracing mode measures it directly.
- **Never fit a constant to make a prediction come out right.** A calibration constant comes from a
  probe that never sees the quantity being predicted, and the probe is reported
  ([`../../CLAUDE.md`](../../CLAUDE.md) §6.4).
- **Recompute derived quantities from data.** Any field whose name asserts a relationship — hit rate,
  effective read rate, bytes per token — is recomputed from the run that produced it and checked
  against its name.

---

## 4. Handling the four inputs that are hardest to get right

### 4.1 Thermal derate

`derate()` comes from the measured curve (T3). Until that curve exists, sustained predictions are
`prior` with an interval wide enough to contain the difference between a cool and a hot device — on
the primary phone, sustained clocks sat around half of hardware maximum, so the interval is wide,
and saying so is the honest behaviour.

### 4.2 Memory

The planner plans against `grantable_foreground`, **never** against total RAM, and re-checks against
the lease actually granted ([`03_INTERFACES.md`](03_INTERFACES.md) §5). Requested and granted
diverged routinely on the primary device, where the largest budget ever granted across 238 runs was
5806 MiB [E:dev_budget_max_mib] and the median was lower.

### 4.3 Expert-cache hit rate

**No cross-model scaling law.** The pre-registered transfer rule across expert geometries was
refuted ([`00_PROBLEM.md`](00_PROBLEM.md) §8.3). Permitted sources, in order:

1. a **measured warm sample** for this model at this cache size on this device (§4 of
   [`04_DEVICE_PROFILING.md`](04_DEVICE_PROFILING.md), a few hundred tokens);
2. a simulated curve from this model's **own** routing trace, with the measured simulator-versus-device
   correction attached and carried in `assumptions` — the simulator is known to sit above the device
   at both points where the two were compared;
3. refusal.

A worked reason to care: on the primary device, one streamed model sat at 75.3% [E:gptoss_hit] hit
rate and still read 283 MiB [E:gptoss_read_mib] per token, re-reading 22 [E:gptoss_rereads] experts
per token that the cache had already paid for once. Hit rate alone does not determine traffic, and a
planner that predicts traffic from a hit-rate rule of thumb will be wrong in the direction that
flatters it.

### 4.4 Storage

`effective_read_rate` is read off the measured curve at the **planned request size**, on the
**model's actual path**, and the plan is refused if the path cannot serve it. Nameplate storage
class is never an input.

---

## 5. Intervals, calibration state, and refusal

Every prediction carries an interval, and the interval is constructed from measured sources of
variation, not chosen:

- run-to-run spread from repeated probe cells (the primary device's storage probe showed a ~9%
  median relative spread, and that is a floor, not a ceiling);
- thermal state uncertainty, from the derate curve's own spread;
- the simulator or prior correction, where one was used.

**Calibration state** is mechanical:

| state | condition |
|---|---|
| `measured` | a valid measurement of this exact cell exists |
| `interpolated` | inside the convex hull of measured cells on this device |
| `extrapolated` | outside it, but the model's inputs are all measured |
| `uncalibrated` | a required input is `prior` or `unknown` |

**Refusal rules — the part that must not be softened.**

1. `uncalibrated` never yields a user-facing number; it yields `NotCalibrated` plus an offer to
   measure, with the time that would take.
2. If the predicted interval **spans the decision boundary** between two candidate configurations,
   the planner returns `tied` and picks by the declared tie-break — it does not pretend to choose.
3. If the interval spans the user's stated latency target, the answer is "I do not know yet; a
   40-second calibration would tell us", not a point estimate in the middle.

> The prior phase's calibration design had only two distinct granted cache sizes, which made almost
> every derived cell an extrapolation while the artifact still printed point values. The mechanism
> above exists specifically so that this cannot happen silently again.

---

## 6. The configuration planner

### 6.1 The search space

`EngineConfig` ([`03_INTERFACES.md`](03_INTERFACES.md) §4.1): tier, quantisation, per-region
placement for prefill and decode separately, thread count and mask, I/O lanes and request size,
expert-cache floor and target, KV policy and context cap, speculation mode, weight layout.

The space is small enough to enumerate after feasibility pruning, which is deliberate — a planner
whose choices cannot be enumerated and audited cannot be trusted with a promise.

### 6.2 Pruning, in order

1. **Fits at all.** Resident bytes must fit inside the foreground grant with room for KV and the
   engine's own working set. A model whose non-streamable part alone exceeds the grant is refused
   with `Infeasible`, not given a number.
2. **Storage feasible.** Streaming tiers require a path whose measured rate supports the traffic the
   plan implies; otherwise `StorageTooSlow`.
3. **Backend available and fidelity-verified.** An accelerator with `fidelity_class: "unverified"`
   for this model is not a candidate until gated — upstream reports of corrupted output on some
   NPU paths make this a live risk, not a formality.
4. **Quality floor.** Configurations whose quality cost is unbounded (aggressive lossy routing,
   unverified requantisation) are excluded unless a fidelity gate bounds them. The prior phase
   measured the whole training-free routing-shortcut frontier and found **no** option that was
   negligible against the format's own floor of 0.118 nats [E:e6_floor_kl] mean KL — dropping a
   single expert of eight already cost 0.146 [E:e6_topk7_kl].

### 6.3 The objective

Lexicographic, because these are not commensurable and pretending they are is how a platform ends up
recommending a fast configuration that fails the task:

1. **Meet the quality floor.** Non-negotiable.
2. **Meet the latency target**, using the *lower* end of the predicted interval for the promise and
   the upper end for the risk assessment.
3. **Minimise energy per completed task.**
4. **Minimise memory footprint**, which is what preserves the user's other apps.

Ties inside the resolution of the prediction (§5, rule 2) are broken by (3), then (4), then by
preferring the configuration with the **more measured** basis — a known quantity beats a slightly
better-looking unknown one.

### 6.4 Building the ladder

The `ExecutionPlan`'s degradation ladder is computed at plan time, not improvised at runtime,
because improvising under pressure is how systems oscillate. Rungs are ordered by quality preserved
per unit of pressure relieved, each with an expected cost and a hysteresis window
([`03_INTERFACES.md`](03_INTERFACES.md) §4.2). A rung with unbounded quality cost may never be
entered automatically.

### 6.5 The falsification rule

Every plan declares what would prove it wrong: which metric, what tolerance, how many observations.
The governor enforces it ([`02_ARCHITECTURE.md`](02_ARCHITECTURE.md) §3.7). This is the mechanism
that makes the platform improve rather than merely persist: a falsified plan yields a measured cell,
and the next prediction for that cell is `measured` rather than `prior`.

---

## 7. Validating the planner

### 7.1 Protocol

1. **Pre-register** the prediction for every (device, model, configuration) cell **before** running
   it. The artifact is written and committed first; the run happens second.
2. Run the cell in the deployment regime, with the validity guard active.
3. Report **prediction error**, per cell and pooled, with the calibration state of each prediction.
4. Never revise a prediction after seeing the outcome. A wrong prediction is a datum.

This is the protocol the prior phase used, and it is why the second-device failure is usable
evidence rather than an embarrassment: the number was on record before the run.

### 7.2 What success requires

The planner must beat all three floors of [`01_RESEARCH.md`](01_RESEARCH.md) §4.1 on **held-out**
cells — most importantly the spec-sheet calculator, which is what every competing tool does — and
the oracle check must show the benchmark can distinguish planners at all.

### 7.3 What failure would mean

If measured profiling does not beat the spec-sheet baseline, that is RQ1 answered "no", and the
product's honest shape changes: the platform stops predicting and starts **measuring on demand**,
selling calibration speed rather than prediction accuracy. That is a smaller product but a true one,
and the architecture supports it without redesign, because the planner already treats measurement as
the preferred basis.

---

## 8. Anti-patterns, named so they can be refused in review

| anti-pattern | why it is fatal | the rule |
|---|---|---|
| summing per-term best cases | the terms share one machine; five measured levers refuted it | §3.3 |
| fitting on the compute residual | it absorbs faults and clock caps | §3.4 |
| transferring hit rate across expert geometries | pre-registered test refuted it | §4.3 |
| using a simulated hit rate uncorrected | the simulator sits above the device | §4.3 |
| planning against total RAM | grants are smaller and revocable | §4.2 |
| a single "speed" number | prefill and decode rank compute units differently | §1 |
| a point estimate outside the calibrated range | this is the failure this whole document exists to prevent | §5 |
| tuning a constant until predictions look right | circular; voids the result | §3.4 |
