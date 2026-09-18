# moe-phone / agentic — what an on-device phone agent's performance *is*, before anything is built

**Status: L1–L3 only. Nothing has been measured for this side track. No phone time has been spent
on it and none is scheduled.** Started 2026-09-18 at the project owner's direction as a *side*
track: the main `moe-phone` line (`../NEXT.md`, `../POSITION.md`) has priority and this document
changes nothing there.

Read [`../../CLAUDE.md`](../../CLAUDE.md) first. This file is the L1/L2/L3 record for the side
track and follows the same rules: no number appears here that is not read from an artifact by a
named script, and every level below the one being worked at must hold before the one above it means
anything.

---

## 0. The system this is about, and the one it is not about

**It is about:** an agent running entirely on the phone that performs *automated tasks on the
phone* — read and triage notifications, find and change a setting, compose and send a message,
extract a value from what is on screen, drive a sequence of taps through an app. Its inputs are a
task instruction and a machine-readable description of the current screen; its outputs are
structured actions.

**It is not about:** code generation, a coding assistant, or anything that needs to read a
repository. That reading was assumed in an earlier conversation and it was wrong. It matters
because the two have *opposite* cost profiles, and the correction moves the project toward its
strengths rather than away from them:

| | coding agent | phone-task agent |
|---|---|---|
| context length | grows without bound (files, history) | roughly bounded by one screen description |
| context reuse across turns | high — the prefix is stable | **low — the screen changes every turn** |
| output length | long (code blocks, diffs) | short (one structured action) |
| dominant cost | decode | **prefill** |
| other apps running | none that matter | **the target app must be running and foregrounded** |

Every row of the right-hand column is an assumption at this point, not a measurement. Three of
them are testable cheaply and none has been tested. That is the honest state.

---

## 1. L1 — the estimand

### 1.1 Primary quantity

For a task `k` drawn from a named task suite `K`, run on device `d` under engine configuration `c`:

> **`T(k, d, c)` = wall-clock seconds from the moment the task instruction is complete to the
> moment the agent emits the final action that completes the task correctly.**

**Unit of analysis: one task.** Not one token, not one turn. The reported quantity is the
**median of `T` over `K`**, with IQR, and it is defined **only over tasks the agent completes
correctly** (§1.2). Tokens per second is not the estimand and must never be reported as if it
were — it is an intermediate quantity that can improve while `T` gets worse, for instance if a
configuration decodes faster but needs more turns.

### 1.2 The gating quantity, reported alongside and never averaged into it

> **`S(d, c)` = the fraction of tasks in `K` completed correctly**, judged against a
> per-task success predicate fixed before the run.

A latency median computed over a set that excludes failures is meaningless without `S`, because
the cheapest way to lower `T` is to fail faster. `T` and `S` are reported together or not at all.
A configuration that is faster at lower `S` has not won.

### 1.3 The decomposition, which is what makes `T` actionable

Over the turns `i = 1..N(k)` the agent takes on task `k`:

```
T(k) = Σ_i [ t_snapshot(i) + t_prefill(n_new(i)) + t_decode(n_out(i)) + t_act(i) ]
```

| term | what it is | measured? |
|---|---|---|
| `N(k)` | turns the agent takes to finish the task | no |
| `t_snapshot` | capturing and serialising the screen description | no |
| `n_new(i)` | prompt tokens this turn that are **not** already in the KV | no |
| `t_prefill` | time to ingest them | **no — at any prompt length** (§3.1) |
| `n_out(i)` | tokens generated this turn | no |
| `t_decode` | time to generate them | only at the few-dozen-token prompts of the existing campaigns |
| `t_act` | the phone executing the action, and the UI settling | no |

**Five of the seven terms have never been measured and the sixth was measured in a regime this
system does not operate in.** That is the finding of this section, and it is why no engine work
is justified yet.

### 1.4 Units and an external plausibility range

`T` is in seconds. The range must come from outside this codebase (`CLAUDE.md` §8):

- a person doing the same task by hand: seconds to tens of seconds;
- a cloud-hosted agent doing it over the network: a few seconds per turn, dominated by round trips.

So a target of **`T` under ~60 s for a short task** is the band where an on-device agent is a
product rather than a demonstration, and **`T` over several minutes** means the answer is no
regardless of how good the engine is. This range is stated here so that a later result cannot be
declared a success by comparing it only to an earlier version of itself.

### 1.5 The averaging measure, and how it differs from the engine's

The estimand averages **over tasks**. Every engine measurement this project holds averages **over
tokens**, within a single long generation. These are different measures over different populations,
and `CLAUDE.md` §2 item 5 exists because this exact gap went unnoticed for two papers in `aegis/`.
Concretely: a token-averaged decode rate is dominated by long generations, whereas a phone-task
agent's tokens are spread over many short turns each paying a fresh prefill and a fresh cache
disturbance. **A token-averaged number may not be substituted for a task-averaged one at any point.**

---

## 2. L2 — what the decomposition assumes, and where the code enforces it

Each assumption names the code that makes it true, or is marked as a wish (`CLAUDE.md` §3).

| # | assumption | enforced by | status |
|---|---|---|---|
| A1 | the engine can serve many turns from one loaded model without reloading it | `--session` / `run_session_loop`, `cli/main.cpp` (BigMoeOnEdge) | **holds — read in source** |
| A2 | the expert cache stays warm across turns | same; the session keeps the live context and the cache | **holds — read in source** |
| A3 | a turn re-ingests only the tokens not already in the KV | `kv_tokens` prefix reuse in `core/src/engine/session.cpp`; `clear_kv` is opt-in | **holds — read in source** |
| A4 | the screen description changes every turn, so A3 saves the system prompt and history but **not** the snapshot | nothing — this is a property of the workload | **untested assumption** |
| A5 | the engine can hold its memory budget while the target app is foreground | nothing | **wish — and the most likely killer (§3.3)** |
| A6 | the model emits a parseable action at a useful rate | nothing | **wish — `S` is unmeasured** |

A1–A3 being already true is the substantive good news in this document: the serving primitives a
phone agent needs exist in the engine and did not have to be built. A4–A6 are the exposure.

---

## 3. L3 — is any of this estimable, and what would have to be measured

This is the level that matters and the level the project exists to take seriously.

### 3.1 The dominant term is the unmeasured one

Every prefill figure in this project was measured at a prompt of a few dozen tokens (the exact
count is in each campaign's own artifact), which makes it a
cold-start artifact rather than an ingestion rate (see `../device/bmoe_prefill.sh`, written for
exactly this and **not yet run**). A phone-task agent's turn is a large-ish prompt and a small
output, so `t_prefill` is expected to dominate `T`. **Until prefill is measured as a function of
prompt length, the primary estimand cannot be predicted, and no configuration can be preferred
over another.** This is the single blocking measurement.

The expectation, registered here before the run: on a streamed MoE, per-token prefill cost should
*fall* with prompt length, because one read of an expert serves every token in the chunk that
routes to it — and then *saturate*, because a large enough chunk touches essentially every expert
and pays the full expert traffic per chunk regardless. Where those cross decides whether the agent
can afford to put a screen description in the context at all.

### 3.2 The context tax has two mechanisms and only one of them matters here

Registered prediction: [`predict_context_tax.py`](predict_context_tax.py) →
`prereg/context_tax_prediction.json`, generated before any long-context cell exists on any device.

It separates the two ways context costs the engine — **eviction** (KV takes RAM from the expert
cache, so more expert bytes are read from flash) and **traffic** (attention re-reads the whole KV
from DRAM every token) — because they have different fixes and the existing campaigns, which never
varied context at all, could not tell them apart.

What it says, and the correction it forces: at the context lengths a phone-task agent actually
uses, **the KV traffic term is a small minority of the cost and the eviction term dominates.**
An earlier conversation claimed the reverse and used it to argue that KV quantisation was the
decisive lever. The registered numbers do not support that: the artifact carries the traffic
share per cell, and quantising KV buys a modest amount at the contexts of interest, not a
transformation. **That claim is withdrawn here rather than annotated** (`CLAUDE.md` §7.6).

The prediction's own weaknesses are in the artifact's `limitations` and are not small: the
calibration design has only **two** distinct granted cache budgets, so almost every predicted cell
is flagged `extrapolated`; the hit-rate input is simulated and systematically optimistic against
the phone at both calibration points; and the traffic term is an uncalibrated roofline. It carries
a recovery check (`CLAUDE.md` §9.2) showing it reproduces its own calibration points within
replicate noise, and an `SE/|slope|` figure per `CLAUDE.md` §4.1.

**The design that would fix the leverage** is a cache-budget sweep with more than two distinct
*granted* sizes, spanning from well below the smallest measured budget up to the largest the device
will actually grant — note *granted*, because the phone refused the one
cell that asked for more than it would give, and a campaign that records the request rather than
the grant will silently fit a line through a point that does not exist.

### 3.3 The assumption most likely to kill the plan, and it is not about speed

A phone-task agent must coexist with the app it is driving: that app is foreground, resident and
active, by definition of the task. Every measurement this project holds was taken on a **quiesced**
phone with third-party apps force-stopped. The gap is not incidental — it is the operating
condition of the entire product.

The unmeasured quantity is: **what expert-cache budget is actually grantable, and held, while a
target app is foreground and being driven?** There is already indirect evidence that the ceiling is
low: on a *quiesced* device the largest cache cell of `bmoe_cache` was refused and granted no more
than the cell below it (the granted sizes are in the prediction artifact's
`calibration.granted_cache_mib_observed`), and a later cell that asked for more than the device had
took it down entirely (`../results/2026-09-18/PHONE_UNREACHABLE.md`).
With an app live, the budget is smaller and the agent is also a background process competing for
survival with the thing it is operating.

**Per `CLAUDE.md` §4.3 this is the cheapest experiment that could kill the project, so it should be
among the first, not the last.** If the answer is "the engine cannot hold a useful cache while
driving an app", no amount of prefill or decode work matters, and that is a publishable negative
result about streamed MoE inference on phones.

### 3.4 Where the workload might be *kinder* than the benchmark

Not every difference runs against us, and the same L3 discipline applies to the favourable
directions — they are hypotheses with cheap tests, not reasons for optimism:

- **Routing concentration.** Every cache curve here was measured on generic prose. Screen
  descriptions and structured actions are repetitive, templated and low-entropy. If they route to
  fewer distinct experts, the hit rate at a fixed cache is higher and decode is better than measured
  for free. Testable entirely on the laptop with the existing `../gates/cache_sim.py` and
  `../gates/llamacpp_traces.py` — **no phone time** — but it needs a corpus of real screen
  descriptions, which does not exist yet and cannot be captured without the phone.
- **Copy-heavy output.** A structured action echoes element ids, field names and literal strings
  straight out of the prompt. That is the regime prompt-lookup drafting is built for, and the
  engine already implements it (`--ngram`, `core/src/engine/ngram_draft.cpp`, which searches the
  live context for a matching span). But the ceiling is fixed by this project's own measured verify
  cost (`../results/2026-09-17/verify_cost.json`): verifying `N` tokens reads the union of `N`
  tokens' experts, so the achievable speedup is bounded by `N / c(N)` and is modest even at perfect
  acceptance. **That bound is stated before any measurement so a later result cannot be oversold.**
- **Device inversion.** The accelerator ranking measured so far is a *decode* ranking. A
  prefill-dominated workload may invert it. Any claim that it does needs the §3.1 measurement first.

### 3.5 Substitution hazards specific to this side track

`CLAUDE.md` §0's failure mode is a plausible substitute standing in for the quantity. The
substitutions available here, named now so they are refused later:

1. **A coding-agent transcript standing in for a phone-task trace.** Transcripts of this very
   repository's sessions are on disk and would make the harness run today. They are a different
   distribution — different context growth, different reuse, different output length — and using
   them would measure the workload this project just established it is *not* about.
2. **A synthetic screen description standing in for a real one.** Hand-written UI trees are shorter,
   cleaner and more repetitive than real ones, all in the direction that flatters the result.
3. **Token-averaged decode standing in for task-averaged latency** (§1.5).
4. **A simulated hit rate standing in for a measured one.** The simulator is already known to be
   optimistic against the phone at both calibration points; the size of that gap is in the
   prediction artifact and must be carried into any claim that rests on it.

---

## 4. Gates, derived for this problem

Not copied from the main line (`CLAUDE.md` §9.2). None of these can run yet; each names what it
needs.

| class | gate | expected value fixed by | needs |
|---|---|---|---|
| Recovery | the cost model returns the measured value at its own calibration points | the measurements themselves | **exists, passes** — in the prediction artifact |
| Degeneracy | the calibration design has ≥3 distinct *granted* cache sizes and `SE/\|slope\| < 1` | `CLAUDE.md` §4.1 | a cache sweep; currently 2 distinct sizes |
| Invariance | halving KV bytes per token at fixed context must move the predicted cache by exactly the KV bytes saved | arithmetic | exists, trivial |
| Monotonicity | more expert cache must not lower the hit rate; a longer context must not raise it | the definition of a cache | a context sweep |
| Physical plausibility | median `T` against the human-hand and cloud-agent ranges of §1.4 | external | the whole harness |
| Floor | §5's baselines | — | a task suite |
| Artifact | reported task latency must not correlate with thermal state or clock caps across the run | the campaign's own state logs | a harness that logs them per turn |

## 5. Baselines this must beat before any number means anything

`CLAUDE.md` §8's three floors, instantiated for a phone-task agent. A configuration that does not
beat all three has demonstrated nothing.

| floor | here | rules out |
|---|---|---|
| **no-data** | a deterministic scripted automation (a fixed rule per task, no model) | that the model has negative information content on tasks a script already does |
| **best constant** | always emitting the single most common action | that the agent's "understanding" beats not understanding |
| **one-parameter shrinkage** analogue | **a model small enough to be resident in RAM** (OLMoE, or a small dense model) at the same task suite | that streaming a 30B from flash beats not streaming at all |

The third is the one that decides whether this project's engine is the right engine for this
product. It is a *task-success* comparison, not a speed comparison: the streamed 30B is slower by
construction, so it must be more capable by enough to pay for it, on this suite, or the honest
recommendation is the small resident model.

## 6. What does not exist

Registered stubs (`CLAUDE.md` §7.5), each with what would remove it.

| id | what is missing | current state | removed when |
|---|---|---|---|
| `AG-S1` | the task suite `K` and its per-task success predicates | does not exist | a suite is written down, before any run |
| `AG-S2` | a corpus of real screen descriptions | does not exist; needs the phone | captured on a device with the owner's phone time |
| `AG-S3` | the agent harness (snapshot → prompt → action → execute loop) | does not exist | built against the engine's `--session` protocol |
| `AG-S4` | prefill vs prompt length | script written (`../device/bmoe_prefill.sh`), never run | the owner allocates phone time |
| `AG-S5` | grantable cache budget with a target app foreground | not attempted | §3.3 is run |
| `AG-S6` | calibration leverage: ≥3 distinct granted cache sizes | 2 exist | a cache sweep is run |

**A note on phone time.** `chain_agentic.sh` was written and briefly armed on 2026-09-18 to queue
the prefill row behind the main chain; it was **stopped before it ran anything** at the owner's
direction, and this side track is not to touch the device. The script is kept because it is the
correct way to do it later, and `chain_agentic.log` records that it waited and was killed.
