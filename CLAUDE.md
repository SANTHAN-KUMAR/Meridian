# CLAUDE.md — how to do quantitative research here without producing a false claim

This is a research repository. The deliverable is **a true claim about the world, with evidence that
survives a hostile reviewer.** Working code, passing tests, and good-looking numbers are not the
deliverable and are not evidence — this project has produced all three while being wrong, twice.

This file is deliberately **domain-general**. It states how to work, not what is true about any
particular model, dataset, or estimator. Project-specific facts — the current problem statement, the
simulator's equations, the active gate list, commands — live in [`README.md`](README.md)
and must not be duplicated here. **If the project pivots to a different problem, this file should
still be correct.**

---

## 0. The failure mode this file exists to prevent

Every prior agent on this project did competent local work and produced a global falsehood. The
pattern never varies:

> A quantity was hard to compute, so a plausible-looking substitute was written instead. The
> substitute satisfied the test. **Nothing ever asked whether the substitute was the quantity.**

The substitutes were each locally reasonable: a regulariser that made a singularity go away, a ratio
that had the right name but the wrong units, a weight that blended toward the expected answer, a
constant chosen so coverage came out at 95%, a test that asserted the conclusion. None was malicious.
None was incompetent. All of them shipped, and all of them survived a fully green test suite.

The countermeasure is not more tests. It is **asking the questions in the right order**, which is §1.

---

## 1. The order of validity

Six levels. Each is meaningless unless every level below it holds. **Work bottom-up. Never repair a
level before the one beneath it is verified.**

| level | question | if it fails |
|---|---|---|
| **L1 Definition** | Is the target quantity written down unambiguously? | You are estimating an unknown |
| **L2 Identification** | Is it a functional of the observable distribution, under assumptions you have stated? | No amount of data helps |
| **L3 Estimability** | Does the data contain enough of the *right* variation to estimate it at the sample size you have? | Every number you produce is noise wearing a point estimate |
| **L4 Specification** | Does the estimator target *that* functional, or a different one? | You measured something real, but not the thing |
| **L5 Computation** | Is the arithmetic right, and did the solver answer the question you asked? | The number is not the estimator's output |
| **L6 Reporting** | Does the claim match the evidence? | Everything below was wasted |

The costliest errors in this repository's history were **L3 defects diagnosed as L4 or L5 defects**.
Two audits recommended fixing the estimator; the target was not estimable from the design at any
sample size, so a corrected estimator would have produced a different wrong number, indistinguishable
from the old one. **When something does not work, walk down the levels, not sideways.**

Name the level you are working at in every response. If you cannot name it, you do not yet understand
the task.

---

## 2. L1 — Define before you estimate

No estimator is written before the estimand exists in writing. The estimand document states:

1. The target in explicit notation (potential outcomes, do-notation, or a stated functional). Include
   the **unit of analysis** — per-individual, per-subgroup, or population — and say which.
2. The conditioning set, and the **averaging measure**: over what distribution, at what times, under
   what baseline, weighted how.
3. Units, and a plausible magnitude range taken from outside this codebase.
4. The identification assumptions, each with a citation or a structural argument.
5. **The estimator's implicit weighting, and how it differs from the estimand's averaging measure.**

Item 5 is the one that gets skipped, and it is where the deepest error in this project lived: the
reference quantity and the estimated quantity were averages over different regimes, and the gap was
invisible for two papers because nobody wrote both down side by side.

**A quantity you cannot write in notation, you cannot estimate.** If writing it exposes an ambiguity —
which time points, which baseline, whose distribution — that ambiguity is a finding. Resolve it in
the document, not silently in the code.

---

## 3. L2 — Identification

State assumptions as **conditional independences over named variables**, not as prose. For each:

- **Say which are testable and which are not.** Test the testable ones and report the result. Name the
  untestable ones as limitations in the same paragraph where you rely on them.
- **Point at the line of code that enforces or violates it.** If a generative process is supposed to
  satisfy an exclusion restriction, name the line that makes it true. If nothing makes it true, the
  assumption is a wish.
- **Positivity/overlap is an identification assumption and it is the one that gets forgotten.** Write
  it down explicitly and then check it at L3.

**Never implement a method from a paper without quoting the equation.** Put the citation, section
number, and the equation as it appears in the source directly above the implementation, then map each
symbol to a variable name. A method attributed to a paper that does not contain it is a fabrication,
regardless of whether the code runs.

**Never describe a causal pathway in a docstring that is not asserted in code.** If a docstring claims
`X` affects `Y`, there is a test measuring it. Documented arrows that turn out to be disconnected
from the running model are how this project spent two papers building proxies for a variable that did
not affect the outcome.

---

## 4. L3 — Estimability: is there anything to estimate?

**This level is the one this project did not have, and its absence cost two papers.** It sits between
"identified in principle" and "estimated in practice", and it is where most quantitative research
silently dies.

### 4.1 The denominator rule

Every causal and regression estimator is a ratio whose denominator is **the variation in the exposure
that survives conditioning on everything you condition on**. Weak instruments, weak proxies, poor
overlap, near-collinear designs and deterministic assignment are not four problems. They are one
problem — *the denominator is near zero* — seen from four directions.

Before running any estimator, compute the standard error the design permits:

```
SE(θ̂)  ≳  sd(outcome residual) / ( sd(exposure residual) × √n_eff )
```

with both residuals taken after partialling out the **full** adjustment set, including every variable
you hope to adjust for successfully. That makes it a best case, so it is a lower bound on achievable
SE. Compare it to |θ| and **report the ratio**.

- `SE/|θ| ≳ 1` — the target is not estimable at this sample size. **Stop.** Do not tune an estimator
  against noise. Report the ratio as the finding, and state the `n` that would be required.
- Report `n_eff`, not `n`. Autocorrelated, clustered or repeated-measures data has an effective sample
  size far below its row count. Sampling a process faster than it varies adds rows and no information.

### 4.2 If the exposure is assigned by a rule, assume there is no variation until you measure some

Protocols, clinical guidelines, control algorithms, dosing formulas, pricing rules and recommender
policies all make the exposure a near-deterministic function of the observed state. Then, by
construction, there is nothing left to identify an effect with.

**The measurement:** regress the exposure on the rule's own inputs. `1 − R²` is the fraction of
exposure variation available to you. If the rule is known, do this analytically too.

This cuts both ways, and the second edge is useful: **a known assignment rule is a known propensity**,
which is a gift for policy evaluation even when it is fatal for individual effect estimation. When L3
blocks one estimand, ask which estimand the same design makes *easier*.

### 4.3 The oracle is a property of the design, not of the method

An oracle estimator — handed the unmeasured confounder, the true nuisance functions, or the true model
— bounds what any feasible method can achieve.

> **If the oracle cannot beat a trivial baseline, the design is uninformative, and no result computed
> on it may be reported.** This is not a statement about your estimator. It says the benchmark cannot
> distinguish a good method from a bad one, so every row in the table is noise.

Run the oracle first, not last. It is the cheapest experiment that can kill a project, which is
exactly why it should be the earliest.

### 4.4 Choose problems where the assignment is hard and the outcome is easy

Causal machinery removes **confounding bias**. It does nothing about **outcome-model error**. So the
value of a causal method is visible only when confounding bias is the larger of the two.

> **Before adopting a problem, estimate both. If the outcome is a complicated dynamical object —
> a stiff ODE, a long-memory process, a system where the response depends on the whole recent history
> — the outcome-model error will dominate and no causal method can demonstrate value on it, however
> correct the identification argument is.**

Causal inference earns its keep where the *assignment mechanism* is complicated and the *outcome
model* is simple: a messy selection process producing a scalar response. It is a poor fit for the
reverse, and the reverse is easy to mistake for a rich research problem, because complex dynamics
look like depth.

The diagnostic is cheap and it is a special case of §4.3: run an oracle that sees the confounder.
If oracle error ≈ naive error, the residual is misspecification, not confounding, and the entire
comparison you were planning is unobservable. Do this **before** committing to a problem, not after.

### 4.5 Estimators must fail loudly on unidentified data

Construct a dataset where the target is provably not identified — deterministic exposure, zero proxy
strength, no overlap — and feed it to the estimator. **It must refuse, warn, or return a non-finite
value.** An estimator that returns a confident number there is reporting its own regularisation, and
it will do exactly the same on the real data without telling you.

---

## 5. L4 — Specification

**An estimator's target is whatever its estimating equation identifies, which is not necessarily what
you meant.** Write out the equation as implemented — not as documented — and check it against L1.

- **Misspecification bias is usually much larger than the bias you are trying to remove.** Before
  deploying machinery that corrects a second-order bias, measure the first-order one. If a
  functional-form error is an order of magnitude larger than the confounding your method targets, the
  comparison measures the functional-form error and your method's contribution is unobservable.
- **Rank deficiency means your model is wrong, not that your solver needs help.** See §6.1.
- If the data-generating process is dynamic and the estimator is static, say so and quantify the gap.
  Do not assume a snapshot covariate set stands in for a history.
- **A method must be distinguishable from its baselines.** Two "different" estimators that are
  algebraically identical are one estimator and a bug in the comparison. Check explicitly — agreement
  beyond a few significant figures between nominally different methods is a defect, not a validation.

---

## 6. L5 — Computation

1. **Never let a numerical solver mask a specification error.** `pinv`, `lstsq` with `rcond`, SVD
   truncation and ridge all return an answer for a rank-deficient design. That answer is not an
   estimate. **Assert rank and condition number at every fit**, before the solve, and fail rather than
   proceed.
2. **Never suppress warnings at module or package scope.** Narrow `catch_warnings` at the exact call
   site, with a comment naming the warning and why it is expected. Global suppression is the mechanism
   by which every other defect in §0 stayed invisible.
3. **Never catch an exception without logging and counting it.** Every fallback increments a counter
   that appears in the results. `except: use something else` is how a pooled estimate was silently
   relabelled a per-unit estimate for an unknown number of units.
4. **Never tune a constant to make a metric come out right.** Not "calibrated", not "empirically
   selected", not "chosen for stability". If a value was chosen by looking at the output it
   influences, the result is circular and void. Constants come from physics, from a cited source, or
   from a selection procedure that (a) never sees the target quantity and (b) is itself reported.
5. **Recompute every derived constant from data.** Any name asserting a relationship — `snr`, `r2`,
   `coupling`, `explained_variance`, `effective_n` — is recomputed from actual output and asserted to
   match its name. Two dimensional errors of exactly this class shipped here; both were a variance
   term omitted from a ratio, and both were later quoted as fact by audits.
6. **Never let a test write to a results path.** Results directories are write-once and stamped with
   timestamp, code version and `n`. Tests write to temporary directories.

---

## 7. L6 — Reporting

1. **Never write a number into prose, a README, a table or a paper by hand.** Every reported figure is
   read from a generated artifact by a named script. One claim, one script, one named cell, listed in
   a claims map.
2. **Report distributions and failures, never just means.** Every results table carries bias, empirical
   SD, error **and error as a fraction of the mean |target|**, median and IQR, measured coverage, and
   the **failure rate** — the fraction of units where the estimator was non-finite, rank-deficient,
   fell back, or landed outside the plausible range.
3. **State magnitudes with their scale.** "MAE 2.12" is uninterpretable. "MAE 2.12, 39% of the mean
   effect size, against a decision threshold of 10%" is a claim someone can evaluate.
4. **Never widen an interval until it covers.** Under-coverage is a finding.
5. **Never ship a stub silently.** Register it: location, current behaviour, correct behaviour,
   removal condition, verifying test. A registered stub is honest engineering; an unregistered one is
   a false claim.
6. **Retraction is deletion, not annotation.** A retracted number is removed from every table, figure
   and summary containing it. A banner at the top of a document whose body still asserts the number is
   worse than no banner: it reads as due diligence while the falsehood stays quotable.

---

## 8. Validate against the world, not against yourself

Internal consistency is not validation. For every quantity the project computes there must be a check
against something outside the code: a published measurement, a conservation law, a dimensional
analysis, or a limiting case with a known closed form.

**If a computed quantity is orders of magnitude from a published value for the same thing, that is the
finding, and it outranks whatever you were working on.**

Three baselines are mandatory in every comparison. A method that does not beat all three has
demonstrated nothing:

| baseline | what it rules out |
|---|---|
| **the no-data baseline** — a prediction using none of the observed outcomes (a formula, a prior, a published rule of thumb) | that your estimator has negative information content |
| **the best constant**, and the oracle constant that cheats by knowing the target's mean | that your "individualisation" is worse than not individualising |
| **one-parameter shrinkage** of the simplest estimator | that your machinery beats one scalar of regularisation |

These are floors, not the scientific claim. Clearing them is a precondition for a number meaning
anything. The scientific claim is *reported*, never *asserted*.

---

## 9. Tests and gates

### 9.1 The only rule that matters

**A test asserts a property whose expected value is fixed by mathematics, physics, or external
literature — independently of whether the method works.**

`assert method_a_error < method_b_error` is not a test. It is a filter that makes the negative result
unrepresentable, and it passes on a broken implementation as readily as a correct one.

Before writing a test, answer: **what would have to break for this to fail?** If the answer is
"nothing realistic", delete it. Then answer: **would this still pass if the component under test were
replaced by a constant, by half of its input, or by a random number in the plausible range?** Work
through it concretely. If yes, it is not testing what you think it is.

### 9.2 Deriving gates for a new problem

Gates are CI-blocking checks whose expected values are fixed independently of the result. Do not copy
a gate list from a previous problem — **derive one** by asking what each class means here:

| class | form | example question |
|---|---|---|
| **Invariance** | rescale, relabel or change units of an input; the output must transform correspondingly | does scaling the exposure by `k` scale the effect by `1/k`? |
| **Recovery** | a limit where the answer is known in closed form | at zero effect, does every estimator return zero? under randomised assignment, is the truth recovered? |
| **Degeneracy** | the design must be non-degenerate before any estimate means anything | is every design matrix full rank? is there overlap? is `SE/\|θ\|` below 1? |
| **Monotonicity** | more information must not hurt | does the oracle beat the feasible method? do strong proxies beat weak ones? |
| **Physical plausibility** | magnitude, sign and range against external literature | is the sign right for 100% of units? is the median within 2× of the published value? |
| **Floor** | §8's three baselines | is it beaten by a constant? |
| **Artifact** | the result must not be a property of the numerics | is the heterogeneity you report correlated with the integrator's clamp rate? |

Gates run on **100% of units**, never a sample. A validator that inspects a subset will miss the
pathological unit, and the pathological unit is the finding.

### 9.3 When a check fails

**The first hypothesis is that the check is right and the code is wrong.** Changing a threshold to
make a test pass requires the same justification as changing the science, because it is the same act.
When criteria genuinely need revising, record the revision, the reason and the date — a documented
threshold change is transparency and a strength; an undocumented one is fraud.

---

## 10. Working method

### 10.1 Definition of done

Not "tests pass". A change is done when:

1. You name the validity level (§1) you worked at, and confirm the levels below it still hold.
2. The relevant gates are green, and you say which ones and what they measured.
3. Any number you report is reproduced by a committed script whose path you name.
4. Any quantity with a physical meaning is stated with its value and the external range it falls in.
5. Anything you could not do is written down — in the stub registry if it is a placeholder, in your
   response if it is a limitation.

### 10.2 Before reporting a result, try to break it

- Does it survive rescaling the exposure?
- Does it survive setting the true effect to zero?
- Does an oracle with strictly more information do better? If not, the benchmark is broken.
- Is it beaten by a constant, by a no-data formula, or by one-parameter shrinkage?
- Is the magnitude physically plausible against an external source?
- Would the sign hold under a different random seed?
- **Is `SE/|θ|` below 1?**

Every one of those questions, asked once, would have caught a central defect in this project.

### 10.3 When you are blocked

The correct move is to **say so and stop**. The wrong move — the one that produced every finding in
the audit — is to write something plausible that lets the pipeline run.

- **Do**: implement it correctly but slowly; register a stub with its removal condition; or report
  that the component is not implemented and say what it would take.
- **Do not**: substitute a proxy that "captures the same idea", add a regularisation that makes the
  singularity go away, or relax a threshold until the check passes.

A negative result, reported clearly, is a contribution. A positive result that does not survive §10.2
is a liability that compounds — every subsequent audit will quote it as established fact, as happened
here twice.

### 10.4 Claims about novelty

Treat "nobody has done this" as a claim requiring evidence, at the same standard as a numerical
result. Cite the searches you ran and the terms you used, distinguish *verified absent* from *not
found*, and state the date — absence of evidence decays. A literature claim that cannot be reproduced
from a written search protocol is not a finding, and "greenfield" asserted without one has the same
status as a hand-transcribed number.

### 10.5 Communication

Report what happened, including what did not work. If a result got worse, say it got worse and by how
much. If you are uncertain whether something is a bug or a real effect, say that rather than picking
the flattering interpretation.

The reader is a co-author who will defend this in review. Give them what they need to be attacked.

---

## 11. Hard prohibitions — index

Violating any of these is a defect regardless of test status. If you believe an exception is
warranted, stop and ask; do not proceed and document it.

1. No constant tuned against the metric it influences. §6.4
2. No test asserting the hypothesis. §9.1
3. No solver masking a rank-deficient or ill-posed design. §6.1
4. No module- or package-scope warning suppression. §6.2
5. No uncounted, unlogged exception handler. §6.3
6. No hand-transcribed number in any document. §7.1
7. No documented causal pathway that is not asserted in code. §3
8. No method implemented from a paper without the quoted equation. §3
9. No downstream fix before its upstream level is validated. §1
10. No interval widened until it covers. §7.4
11. No unregistered stub. §7.5
12. No test writing to a results path. §6.6
13. No estimate reported from a design where `SE/|θ| ≳ 1`. §4.1
14. No result reported from a benchmark whose oracle loses to a trivial baseline. §4.3
15. No novelty claim without a reproducible search protocol and a date. §10.4
16. No problem adopted before measuring whether confounding bias or outcome-model error dominates. §4.4

---

**Project-specific context lives with the project, not here.** This repository holds one:

- [`moe-phone/`](moe-phone/) — running large Mixture-of-Experts language models on Android phones
  by streaming expert weights from flash, plus the Meridian on-device agent app built on that
  engine. The research phase (10 tok/s target on a OnePlus 15R) is **closed**: see
  [`moe-phone/research/2026-09-19_CLOSURE.md`](moe-phone/research/2026-09-19_CLOSURE.md). The
  platform/app phase is active: start at [`moe-phone/platform/README.md`](moe-phone/platform/README.md).

Read [`README.md`](README.md) for the 60-second overview and routing, then
[`moe-phone/README.md`](moe-phone/README.md) (the gate record) or
[`moe-phone/HANDOFF.md`](moe-phone/HANDOFF.md) (live state for a new session), before your first
edit.
