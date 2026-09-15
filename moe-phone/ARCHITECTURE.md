# moe-phone — the engine, derived from the measurements

**Status 2026-09-14.** This document states the decode loop the measurements imply, and *only*
what they imply. Every number is produced by a named script into `results/<date>/` and checked
against the artifact by `gates/claims_check.py`; nothing here is typed by hand (`CLAUDE.md` §7.1).
Assumptions the design leans on are named inline and registered in [`POSITION.md`](POSITION.md)
§11 — **S9** (geometry transfer), **S10** (acceptance rate), **S11** (compute term). **S12**
(predictor accuracy) is closed, negative — see §3.1.

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
constant. `h`, the expert-cache hit rate, is the only free variable **of this equation**.

> **Tested 2026-09-16 against an independent engine, and the equation is incomplete**
> (`gates/external_validation.py`, G-VALID-3). On colibri's published disk-bound rows with
> speculation off it over-predicts tok/s by 2.8x (median); at a 98% hit rate by 4.4x. It leaves
> out DRAM reads of resident weights and cache hits, and compute — and on real hardware those
> are the same order as the flash term. So raising `h` is necessary and not sufficient.
> `engine_target.py` now charges DRAM at the measured 59.74 GB/s (S1) and, with
> `--nonflash-gbps`, the measured on-device non-flash rate (S11): a resident MoE on the 15R
> consumes its per-token bytes at 23.2 GB/s, 39% of the DRAM probe, so compute and engine
> overhead — not DRAM — set the non-flash term here.

> ```
> python moe-phone/gates/engine_target.py --bulk-gbps 2.806 --dram-gbps 59.74 --ram-gb 4.85 --target-tps 5
> ```
> prints, per candidate model, the flash-only tok/s, the serial flash + DRAM tok/s, and the
> serial figure under S9's competing hypothesis. Fully resident models are reported as
> flash-free and models outside the measured `rho` range get no number. **S9 applies to every
> row.**

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

## 3. The design, and the lever that turned out not to exist

```
  per-layer expert cache
    + LRU eviction
    + event-atomic fetch (a layer's whole top-k decided before any eviction)
```

That is the whole deployable design. An earlier version said it was "worth double digits of tok/s
on three candidate models"; that came from three `engine_target.py` defects (README defect table)
and is withdrawn. Corrected: Qwen3-30B-A3B at 11.5 tok/s flash-only, 8.8 with DRAM charged, and **6.3 with the
measured on-device non-flash rate charged (S11) — 4.2 if S9's floor-additive hypothesis holds**.
The reading-speed target (5 tok/s) sits between the two, so the S9 trace decides this model. Everything below is about the 2.2× that §2.4 says is still on the table,
and whether it can be reached.

### 3.1 A cheap predictor was the primary recommendation. It is not.

An earlier version of this document proposed a **next-token, same-layer expert predictor** used in
`protect` mode — the predictor may veto an eviction candidate, never propose one — on the strength
of an abstract accuracy sweep. Measured with real predictors (`gates/predictor.py`, closing
**S12**), the lever is not there:

| predictor (10% cache, held out) | slot accuracy | h | vs plain LRU |
|---|---|---|---|
| plain LRU | — | 0.287 | 1.00× |
| persistence ("t+1 wants what t wants") | 0.387 | 0.284 | **1.00×** |
| first-order Markov, fitted | 0.338 | 0.262 | 0.97× |
| frozen popularity | 0.218 | 0.237 | 0.93× |
| *exact one-step oracle* | 1.000 | 0.383 | *1.16×* |
| *Belady* | — | 0.455 | *1.36×* |

**Persistence is not weak — it is informationally empty**, and that is the finding. The eviction
rule already knows the expert set of the token it is *executing*, for free, because the engine is
running that layer right now. Persistence asserts the next token wants that same set, so it
contributes nothing the rule did not already have. It reproduces the horizon-0 column to three
decimals at every cache fraction, which `tests/test_gates.py` asserts as a property rather than a
measurement.

Its slot accuracy of 0.387 against 0.125 under independence is genuinely 3× chance. **Good
prediction, useless information** — the two are not the same thing, and that gap is the result.

### 3.2 Why the sweep looked better, and what the real constraint is

Two explanations were considered. One was tested and **rejected**: that the abstract sweep flatters
a predictor because `corrupt_routing()` replaces a wrong guess with a *uniformly random* expert —
rarely resident, so a wrong veto rarely fires — whereas a real predictor errs toward *plausible*
experts that often are resident. Corrupting from the trace's own routing at matched accuracy scores
the same to within 0.2%. The noise model is sound.

The supported explanation is **horizon**:

| predictor accuracy | protect, horizon 1 | protect, horizon 4 |
|---|---|---|
| 1.00 | 0.383 | 0.394 |
| 0.50 | 0.286 *(= LRU)* | 0.347 |
| 0.35 | 0.257 | 0.324 |

At horizon 4 even a 35%-accurate predictor beats LRU. At horizon 1, accuracy ≈ 0.5 is needed merely
to break even. **A one-step predictor can only honestly supply horizon 1** — standing at token *t*
it cannot form token *t+2*'s prediction from token *t+1*'s routing, which it does not have.

And the agreement is close once both are matched: uniform corruption at accuracy 0.35, horizon 1
scores 0.257; the measured Markov predictor at slot accuracy 0.338 scores 0.262. The abstract sweep
predicts the real predictor correctly — it was being read at the wrong horizon.

> **The lookahead lever is real, and it is not reachable by prediction.** It needs a horizon of
> about four tokens, and a horizon of four needs the next four tokens' routing. Only a multi-token
> verification pass actually has that.

### 3.3 What survives of `protect`

The `rank` / `protect` distinction still holds and still matters, because §4's lookahead is exact
over its window but the window ends: `rank` collapses below LRU when its lookahead is wrong
(0.097 vs 0.284 at accuracy 0), `protect` degrades gracefully. **A lookahead that is exact should
rank; anything uncertain should only veto.** That is a rule about how to spend a lookahead, and it
is unaffected by S12 — what S12 removes is the claim that a cheap predictor can *supply* one.

---

## 4. Multi-token verification: the only supplier of the lookahead, and it is conditional

§3 leaves exactly one way to obtain a horizon of four. Verifying W tokens in one pass supplies an
**exact** lookahead over its own window — and the exactness is worth being precise about, because the obvious explanation is
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

> **Read against §3: this is the only route to the lookahead lever, and it is not free.** It is
> a regression at α ≤ 0.8, roughly 1.13–1.28× at α = 0.9, and clearly worth it only above
> α ≈ 0.95. There is no cheaper substitute — S12 closed that door.

**And the table above is flash-only.** A self-draft runs W−1 extra forward passes per window and
the verify pass is not free either. `engine_sim.py --fwd-ms` charges them (c seconds per pass,
verify = c·(1 + beta·(W−1))): at 10 ms per pass, W=4, alpha=0.9 falls from 1.13x to 1.10x, and to
1.02x if the batched verify is compute-bound (beta = 1). c is swept until the on-device
forward-pass time is measured.

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
  goes through `engine_target.at_rho()`. An earlier version cited an "independently reported"
  0.693 for a 512-expert model as a consistent external check; no source for it was ever
  recorded, so it is deleted. The two competing transfer rules are pre-registered in
  `results/2026-09-16/s9_prereg_Qwen3-30B-A3B.json`.
- **It does not include a compute term** (**S11**). Valid only while the device is flash-bound —
  and G-VALID-3 says real engines are not purely flash-bound even at modest hit rates.
- **It does not include a predictor.** S12 is closed, negative: no cheap cross-token, same-layer predictor reaches the lookahead lever, because one step is the wrong horizon (§3.2).
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
| **raise `h`** via lookahead | up to ~2.2×, but ONLY from an exact 4-token window (§3, §4) | **measured, and its cheap route is closed** (S12) | done; what remains is α, below |
| **deeper I/O queue** | 1.0× | **closed 2026-09-16, negative.** At 1 MB direct reads 16 or 32 threads reach 0.99x of 8 threads (`g1_storage_qd.json`); the plateau starts at 4 threads. It is a device ceiling, not a queue-depth limit | — |
| **lower precision**, 4.5 → ~3.0 bpw | ~1.5× | **untested against our margin** | rerun G3's harness at Q3_K/IQ3 against the same pre-registered Q4_0 Tier-A margin |
| **whole-expert skipping** (ACE, arXiv 2609.05228: 50%, training-free *and* calibration-free) | up to 2× | **untested, and a different axis from G3** — G3 killed intra-expert *neuron* sparsity on a gate-first criterion; this drops whole experts, so S8 does not gate it | same harness. ACE's headline is measured against other skipping methods, not against the full model, so our margin is the real test |
| **union fetch** across a verification window | ~1.2× at α=0.9 | **measured, but can be negative** (§4) | measure `alpha` |

Row four is the largest single multiplier available and the least examined — which is exactly why
it should be run early rather than assumed.

---

## 6. The next measurements, in order of how much they move the answer

0. **Done 2026-09-16 (S11):** a resident MoE's non-flash rate is 23.2 GB/s on the 15R. Next is a
   clean *resident* OLMoE run to test the linear-in-bytes assumption it rests on (predicted
   33.3 tok/s), and the S9 trace, which now decides whether Qwen3-30B-A3B clears 5 tok/s.

1. **A drafter's acceptance rate `alpha`** (**S10**). S12's closure makes this the *only*
   remaining route to the 2.2×, and the sign of the effect flips inside the plausible range of
   published drafters, so it decides whether §4 gets built at all.
2. **Routing traces for a second expert count** (Qwen3-30B-A3B, `E`=128; Qwen3-Next-80B-A3B,
   `E`=512). Closes **S9**, which gates *every* per-model tok/s figure. The Kaggle notebook
   already runs the full sweep; it needs a second model argument.
3. **Whole-expert skipping at the pre-registered fidelity margin** (§5b row four). The largest
   untested multiplier, and S8 does not gate it.
4. ~~Extend the G1 thread sweep past 8.~~ Run 2026-09-16: no gain past 8 threads (§5b).

---

## 7. Reproducing every number in this document

```
python moe-phone/gates/claims_check.py          # verify each claim against its artifact
python moe-phone/gates/claims_check.py --emit   # regenerate CLAIMS.md from the artifacts
```

`tests/test_gates.py` runs the check, so a stale number in this file or in
[`CLAIMS.md`](CLAIMS.md) fails the suite rather than surviving into a paper.
