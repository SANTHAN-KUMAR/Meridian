# moe-phone — the engine, derived from the measurements

**Status 2026-09-14.** This document states the decode loop the measurements imply, and *only*
what they imply. Every number is produced by a named script into `results/<date>/` and checked
against the artifact by `gates/claims_check.py`; nothing here is typed by hand (`CLAUDE.md` §7.1).
Assumptions the design leans on are named inline and registered in [`POSITION.md`](POSITION.md)
§11 — **S9** (geometry transfer), **S10** (acceptance rate), **S11** (compute term), **S12**
(predictor accuracy).

Read [`POSITION.md`](POSITION.md) §3c first: it has the corrected G2 and the two simulator defects
that produced the previous, wrong version of this design.

---

## 1. The reduction: one free variable

G3 found no gate-first neuron density below 1.0 that stays inside the pre-registered Q4_0 Tier-A
fidelity margin, so the engine reads **whole experts**. Every expert in every candidate model is
1.77–99 MB, and G1 measured full bulk bandwidth at and above 512 KB. So **every read the engine
issues is already a bulk read**: the 8.9× small-read penalty does not apply, and co-activation
layout (G4) is not on the critical path.

What is left is one equation:

```
seconds / token  =  E_tok_bytes × (1 − h) / bulk_bandwidth
```

`E_tok_bytes` is fixed by the checkpoint and the quantisation. `bulk_bandwidth` is a device
constant. **`h`, the expert-cache hit rate, is the only free variable, and everything the engine
does exists to raise it.**

> ```
> python moe-phone/gates/engine_target.py --bulk-gbps 2.806 --ram-gb 4.85 --target-tps 5
> ```
> prints, per candidate model, the achieved tok/s at the measured LRU and Belady hit rates.
> **S9 applies to every row.**

---

## 2. What raises `h`, in the order the measurements rank them

### 2.1 Scope the cache per layer, not as one shared pool

Layer *l*'s experts are never candidates for layer *l+1*, so a shared pool spends its capacity
holding the whole `L × k` cycle before anything recurs — the cyclic-scan worst case.

This is not a close call, and it is worth showing because the shared pool is the intuitive design:

| 10% cache | per-layer | global pool |
|---|---|---|
| LRU | **0.272** | 0.000 |
| + lookahead of 4 | **0.437** | 0.179 |
| + lookahead of 16 | 0.437 | 0.363 |
| Belady (offline optimum) | 0.437 | **0.490** |

*(`gates/scope_compare.py`, 8000-token sub-sample — the shared-pool lookahead scans the whole pool
on every eviction, so the full trace is expensive. The sub-sample is recorded in the artifact. The
per-layer column on the full trace is 0.279 / 0.447; the ordering is unchanged.)*

Note the horizon means different things in the two columns, and that is the comparison rather
than an inconsistency: per-layer counts **tokens**, global counts **events** (layer-steps). The
global column is given the *easier* signal on purpose — a within-token cross-layer predictor
supplies a few events with no drafter at all — because if the shared pool could reach its optimum
from cheap lookahead it would win.

It cannot. Its offline optimum is genuinely higher (0.490 vs 0.437; it has more freedom), but
16 events is a whole token of future routing and it still lands below what the per-layer cache
gets from four. So: **per-layer quotas, out of the same total byte budget.** `split_capacity()`
preserves the total exactly, so "10% cache" means the same bytes either way.

### 2.2 Fetch a layer's misses as one event

Deciding residency for a layer's whole top-k *before* evicting anything stops a miss from evicting
an expert the same fetch still needs. This is not an optimisation — it is what the hardware does
anyway; simulating otherwise was a defect.

### 2.3 Evict by recency, not by popularity

Measured on a held-out split against LFU, cache warming, and static pinning at every reserve
fraction: **plain LRU wins at every cache size from 10% up**, including against a static oracle
that cheats by knowing the whole trace's expert frequencies. Warm-starting is indistinguishable
from plain LRU to three decimals once past the warmup, because LRU re-converges to the same state
within a few hundred tokens.

Cache warming is therefore a **time-to-first-token** optimisation here, not a throughput one.
Ship it for latency; do not count it in the byte budget.

### 2.4 The only thing left is lookahead

The gap between LRU and Belady is not an eviction-rule problem — §2.3 exhausted the rules.
Belady's advantage is that it sees the future. Sweeping a bounded horizon — horizon 0 means the
engine knows only the token it is executing, an unbounded horizon is exactly Belady (asserted as
an identity in `tests/test_gates.py`; horizon 0 is bracketed rather than identified, because
knowing the current token already makes the rule event-atomic):

> **Four tokens of lookahead reaches Belady exactly at cache fractions up to 12.5%; at 20% it
> reaches 97% of it and eight tokens closes the rest.** The phone-relevant models sit at 6–24%.

Four tokens is a remarkably short horizon, and it is the load-bearing result of this document.

---

## 3. The design: per-layer LRU, with a predictor that may only veto

**The primary recommendation is not speculative decoding.** It is the cheapest thing that
delivers most of the lookahead benefit with no acceptance risk:

```
  per-layer LRU cache
    + a next-token, same-layer expert predictor
    + eviction in PROTECT mode: the predictor may VETO a candidate for
      eviction, never propose one; recency still does the ranking
```

Measured on OLMoE at a 10% cache (`results/<date>/cache_pred_OLMoE-1B-7B-0924.json`):

| accuracy | h | tok/s | vs plain LRU |
|---|---|---|---|
| — (plain LRU) | 0.284 | 8.6 | 1.00× |
| 0.9 | 0.383 | 10.0 | **1.16×** |
| 0.7 | 0.365 | 9.7 | 1.13× |
| 0.5 | 0.342 | 9.4 | 1.09× |
| 0.3 | 0.311 | 8.9 | 1.04× |

**The shape of that column is the point.** It is monotone, it is never below 1.00×, and it
degrades gracefully. A design whose worst case is "no worse than the baseline" can be shipped and
tuned in the field; one whose worst case is a regression cannot.

### 3.1 Why `protect` and not `rank`

The obvious way to spend a lookahead is to evict the expert whose predicted next use is farthest —
call it `rank`. With an *exact* lookahead that is optimal: it reaches Belady. With a wrong one it
is a disaster, and an earlier version of this document asserted the opposite.

**The claim that was wrong:** that a bad predictor is harmless here because a wrong *eviction*
spends no bandwidth, unlike a wrong *prefetch*. The first half is true, and it is why the
published prefetch negatives do not transfer. The second half does not follow, and the measurement
refutes it: ranking victims by a wrong prediction **discards recency, which is itself real
information.** At accuracy 0, `rank` scores 0.097 against LRU's 0.284. Break-even is accuracy
≈ 0.70 — published expert-routing predictors report 86–91%, so they clear it, but not by much, and
on their models rather than ours.

`protect` keeps recency as the ranking signal and lets the predictor only remove candidates. It
gives up the peak (0.390 vs 0.451 with an exact lookahead) and buys the entire robustness column
above.

> **Rule: a lookahead that is exact should rank. A lookahead that is predicted should only veto.**

### 3.2 What predictor, and what we do *not* know about it (S12)

The cache is per-layer, so the useful prediction is **cross-token, same-layer**: *which experts
will token t+1 want at layer l?* That is **not** the quantity G2 measured. G2's 0.88 recall is
**cross-layer** lookahead (layer *l+1* from layer *l*'s input, the Fate / Mixtral-offloading
trick), which a per-layer cache cannot use at all — §2.1 is why.

So the accuracy of the predictor this design needs is **unmeasured**, and that is stub **S12**. It
is also the cheapest open measurement in the project, because it is a property of traces already
committed and needs no GPU.

What *is* known: the measured adjacent-token expert overlap on this trace is **0.366**, so even
the trivial "token t+1 will want what token t wanted" persistence predictor lands near the bottom
of the table above — around 1.04–1.09×, still positive. **That is the floor, not the estimate.**

One caveat on the noise model itself: corruption replaces a wrong prediction with a *uniformly
random* expert. A real predictor's errors are not uniform — they are plausible experts — so these
figures are a conservative reading of a given accuracy, not a calibrated one.

---

## 4. Speculative decoding: a second option, conditional rather than free

If a drafter is available, verifying W tokens in one pass supplies an **exact** lookahead over its
own window — and the exactness is worth being precise about, because the obvious explanation is
wrong. It is *not* that the draft model routed those tokens: a self-draft restricted to
cache-resident experts is a different model, so its routing would only be a prediction. The
exactness comes from the shape of the verification pass. Layer *l* routes all W positions in one
batched matmul, and that happens **before** layer *l*'s expert weights are touched, so the engine
reads true top-k sets off the router it has just run. The horizon is exactly the window and not
one token more, which is why `engine_sim.py` defaults to `--lookahead-beyond 0`.

That window is spent two ways at once on the same cache — each distinct expert fetched **once**
(the union), and eviction by farthest-next-use — so **the two gains are not independent and must
not be multiplied.** `engine_sim.py` replays the whole loop in one pass and prices it per
**accepted** token, because a rejected draft token still cost its share of the fetches:

| W, α | h | tok/s | vs plain LRU |
|---|---|---|---|
| 4, α=0.8 | 0.451 | 8.3 | **0.97×** |
| 4, α=0.9 | 0.451 | 9.7 | 1.13× |
| 8, α=0.9 | 0.582 | 10.5 | 1.23× |
| 8, α=0.95 | 0.582 | 12.5 | 1.45× |
| 16, α=0.8 | 0.714 | 6.6 | **0.76×** |
| 16, α=0.95 | 0.714 | 15.1 | 1.76× |

> **Read against §3: at α = 0.9, speculation with W=4 delivers 1.13×, which a predictor in protect
> mode matches at 1.16× — with no drafter, no verification compute, and no downside risk.**
> Speculation only clearly wins above α ≈ 0.95, and it goes *negative* at α ≤ 0.8.

`alpha` is swept, not measured (**S10**), and the sign of the effect flips inside the plausible
range of published drafters. So **the number to measure next is `alpha` for a specific drafter —
not another cache policy.**

Three external facts that sharpen this:

- The hard ceiling on union savings is `M·k/E` accepted tokens per pass, so break-even needs
  `M = E/k`: 8 for OLMoE, 16 for Qwen3-30B-A3B, **51 for Qwen3-Next-80B-A3B**. **This inverts the
  model preference in `POSITION.md` §3** — fine-grained MoE has the best byte budget and the worst
  speculation payoff.
- Every published MoE speculative-decoding system independently converged on γ ≈ 3–5.
- A 19-configuration llama.cpp sweep on an MoE found *no* speculative configuration faster than
  baseline, at 100% draft acceptance. See [`LEADS.md`](LEADS.md) §5.

---

## 5. What this engine is *not* claimed to do

- **It does not beat the flash roofline.** It raises `h`; the equation in §1 is unchanged.
- **It does not transfer across expert counts for free** (**S9**). Every per-model tok/s figure
  goes through `engine_target.at_rho()`. The one external check available is consistent (an
  independently reported 512-expert top-10 model reaches 0.693 where our curve interpolates 0.669
  at the same `rho`), but one point is not a validation.
- **It does not include a compute term** (**S11**). Valid only while the device is flash-bound.
- **Its predictor's accuracy is unmeasured** (**S12**), and it is not the quantity G2 measured.
- **It is not novel in its parts.** Union-of-experts loading exists in llama.cpp PR #25294 for
  *prefill*; self-speculative MoE decode exists (S2-MoE, DraftExpert — the latter on this exact
  SoC class). What is *not found* in the literature — not found, not verified absent, per
  `CLAUDE.md` §10.4 — is using a bounded lookahead as an **eviction oracle** rather than as a
  prefetcher or a compute amortiser, and the `rank`/`protect` distinction that makes a noisy one
  safe.

---

## 5b. Levers not yet in the number, and what each is worth

The tok/s figures above use **one** lever: raising `h`. Four more exist. Each is listed with its
status, because the difference between *measured*, *assumed* and *untested* is the whole point of
the table — and **they have never been measured together, so this is not a prediction.** Anyone
stacking them owes a fidelity check per row.

| lever | multiplier | status | what it would cost to find out |
|---|---|---|---|
| **raise `h`** via lookahead (§3) | 1.16× with a predictor at accuracy 0.9; up to ~2.2× with an exact lookahead | **measured** | done |
| **deeper I/O queue** | 1.25–1.5× (est.) | **untested.** G1 stopped at 8 threads; at 1 MB the 1→8 scaling was ×1.81 and still rising | one afternoon: extend `ufsbench` to 16/32/64. **Cheapest lever, and it scales every figure linearly** |
| **lower precision**, 4.5 → ~3.0 bpw | ~1.5× | **untested against our margin** | rerun G3's harness at Q3_K/IQ3 against the same pre-registered Q4_0 Tier-A margin |
| **whole-expert skipping** (ACE, arXiv 2609.05228: 50%, training-free *and* calibration-free) | up to 2× | **untested, and a different axis from G3** — G3 killed intra-expert *neuron* sparsity on a gate-first criterion; this drops whole experts, so S8 does not gate it | same harness. ACE's headline is measured against other skipping methods, not against the full model, so our margin is the real test |
| **union fetch** across a verification window | ~1.2× at α=0.9 | **measured, but can be negative** (§4) | measure `alpha` |

Row four is the largest single multiplier available and the least examined — which is exactly why
it should be run early rather than assumed.

---

## 6. The next four measurements, in order of how much they move the answer

1. **The cross-token, same-layer predictor's accuracy** (**S12**). It is the input to the
   *primary* design and it is currently a blank. Cheap: a property of traces already committed,
   no GPU needed. Build the predictor, score it on held-out tokens, read the tok/s off §3.
2. **Routing traces for a second expert count** (Qwen3-30B-A3B, `E`=128; Qwen3-Next-80B-A3B,
   `E`=512). Closes **S9**, which gates *every* per-model tok/s figure. The Kaggle notebook
   already runs the full sweep; it needs a second model argument.
3. **A drafter's acceptance rate `alpha`** (**S10**). Decides whether §4 is worth building at all,
   since the sign flips inside the plausible range.
4. **Extend the G1 thread sweep past 8.** The bandwidth constant in §1 is a measurement limit, not
   a device limit. Cheapest of the four, and it scales everything.

---

## 7. Reproducing every number in this document

```
python moe-phone/gates/claims_check.py          # verify each claim against its artifact
python moe-phone/gates/claims_check.py --emit   # regenerate CLAIMS.md from the artifacts
```

`tests/test_gates.py` runs the check, so a stale number in this file or in
[`CLAIMS.md`](CLAIMS.md) fails the suite rather than surviving into a paper.
