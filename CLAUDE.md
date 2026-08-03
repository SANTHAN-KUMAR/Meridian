# CLAUDE.md — operating rules for this repository

This is a **causal inference research project** targeting a peer-reviewed venue. The deliverable is
not working code. The deliverable is **a true claim about the world, with evidence that survives a
hostile reviewer.** Code that runs, tests that pass, and numbers that look good are worth nothing on
their own — and this repository has already produced all three while being wrong.

Read [`audit/CRITICAL_REVIEW.md`](audit/CRITICAL_REVIEW.md) before your first edit. It documents 34
findings, of which the central one is that a published headline number was an artifact of a silently
pseudo-inverted singular matrix, and the ground truth it was scored against was an artifact of an ODE
integrator clamp. Both survived 33 passing tests.

---

## 0. The failure mode this file exists to prevent

Every prior agent on this project did competent local work and produced a global falsehood. The
pattern was always the same:

> A quantity was hard to compute, so a plausible-looking substitute was written instead. The
> substitute satisfied the test. Nothing ever asked whether the substitute was the quantity.

Concretely, what actually happened here:

| the shortcut | what it produced |
|---|---|
| Stage 2 regression included every regressor that "should help" | exactly singular design matrix; the SVD solver silently returned a min-norm solution equal to **naive OLS ÷ 2**, and that halving *was* the published 47.5% improvement |
| `snr = beta / sigma` — looks like a signal-to-noise ratio | actual SNR was **7× smaller**; the project's stated central limitation ("only 28% of patients exceed F=10") was downstream of this one line |
| `w = 1 − exp(−F/10)` blended toward naive when proxies were weak | manufactured proxy sensitivity for a point estimate that had none — the *appearance* of the desired result |
| `effective_n = n/5.0`, `sensitivity_gamma = 0.15` | interval widths tuned until coverage hit 95%; the source comment says so outright |
| `test_proximal_beats_all_alternatives` | the conclusion asserted as a build precondition, so a negative result could not ship |
| `warnings.filterwarnings('ignore')` at module scope | the mechanism by which all of the above stayed invisible |
| ground truth averaged over hourly samples | 63% of those samples were **exactly 0.0** because the glucose state hit a hard clamp; 58% of the "order-of-magnitude patient heterogeneity" was integrator saturation |

None of these were malicious and none were incompetent. Each was locally reasonable. The rules below
are the specific countermeasures.

---

## 1. Hard prohibitions

Violating any of these is a defect regardless of test status. If you believe an exception is
warranted, stop and ask — do not proceed and document it.

1. **Never tune a constant to make a metric come out right.** Not "calibrated", not "empirically
   selected", not "chosen for stability". If a value was chosen by looking at the output it
   influences, it is circular and the result is void. Constants come from physics, from a cited
   paper, or from a procedure (cross-validation on held-out data) that is itself reported.

2. **Never write a test that asserts the hypothesis.** `assert proximal_mae < naive_mae` is not a
   test, it is a filter that makes the negative result unrepresentable. Tests assert properties whose
   expected value is fixed by mathematics, physics, or external literature — *independently of whether
   the method works*. See §3.

3. **Never let a numerical solver mask a specification error.** `pinv`, `lstsq` with `rcond`, SVD
   truncation, and ridge regularisation all return an answer for a rank-deficient design. That answer
   is not an estimate. Assert rank and condition number at every fit. Rank deficiency means your
   model is wrong, not that your solver needs help.

4. **Never suppress warnings at module or package scope.** No `warnings.filterwarnings('ignore')`, no
   bare `np.errstate(all='ignore')` wrapping a whole function. Narrow `catch_warnings` at the exact
   call site, with a comment naming the warning and why it is expected.

5. **Never catch an exception without logging and counting it.** `except Exception: fall back to
   something else` is how a pooled estimate got relabelled as a per-patient estimate for an unknown
   number of patients. Every fallback increments a counter that appears in the results.

6. **Never write a number into prose, a README, a table, or a paper by hand.** Every reported figure
   is read from a generated artifact by a script. The README claimed P95 = 5.60 while `results.csv`
   said 6.187, and `summary_statistics.txt` offered the range of the *estimator* as evidence for the
   heterogeneity of the *ground truth*. Both are transcription-class errors.

7. **Never describe a causal pathway in a docstring that is not asserted in code.** Two documented
   arrows in this repository (`U → glucose via cortisol`) operate on a variable that is
   uncorrelated (r = −0.009) with the actual confounder. If a docstring claims `X` affects `Y`, there
   is a test measuring it.

8. **Never implement a method from a paper without quoting the equation.** The proximal estimator was
   attributed to Cui et al. (JASA 2024), which contains no such estimator; the method it gestured at
   (P2SLS) puts a *different variable* in the endogenous slot. Put the citation, the section number,
   and the equation as it appears in the source directly above the implementation, then map each
   symbol to a variable name.

9. **Never fix a downstream component before its upstream input is validated.** Both prior audits
   recommended fixing the estimator. The ground truth it is scored against is invalid, so a correct
   estimator would have produced a different wrong number, indistinguishable from the current state.
   Work in the order of the data flow: simulator → ground truth → data generation → estimator →
   inference → reporting.

10. **Never widen an interval until it covers.** Under-coverage is a finding. Report it.

11. **Never ship a stub silently.** If a component is a placeholder, register it (this repository's
    `verification/STUB_REGISTRY.md` is a genuinely good model: location, current behaviour, correct
    behaviour, removal condition, verifying test). A registered stub is honest engineering. An
    unregistered one is a false claim.

12. **Never let a test write to a results path.** Running `pytest` currently overwrites the committed
    50-patient `results.csv` and diagnostics with a 10-patient run, and the figure generator reads
    those same paths — so the published figures can silently become figures of the test run. Results
    directories are write-once and stamped with timestamp, git SHA and `n`. Tests write to `tmp_path`.

---

## 2. Positive practices

### 2.1 Before writing an estimator, write the estimand

No estimator PR is accepted without a current `docs/ESTIMAND.md` stating:

- the potential-outcome notation for the target quantity;
- the conditioning set;
- the averaging measure (over what distribution, at what times, at what baseline dose);
- the units, and a plausible range from the clinical literature;
- the identification assumptions, each with a citation;
- **what the estimator's implicit weighting is, and how it differs from the ground truth's averaging
  measure.**

That last item is the one that gets skipped and it is where this project's deepest problem lives: the
ground truth was a derivative at insulin dose 0 with basal suspended, on a trajectory averaging
341 mg/dL glucose, while the data came from a trajectory averaging 151 mg/dL with mean dose 0.33 U.
Those are different quantities and nothing noticed for two papers.

### 2.2 Validate against the world, not against yourself

Internal consistency is not validation. For every quantity the project computes, there must be a check
against something outside the code:

| quantity | external anchor |
|---|---|
| simulated glucose distribution | real T1D cohorts average 150–200 mg/dL; time-in-range 70–180 is a reported clinical statistic |
| insulin sensitivity τ | clinical ISF is 30–80 mg/dL/U (Walsh 2000-rule; Davidson 1700-rule); scale by the horizon fraction |
| day-to-day variability of τ | 38–79% CV (Ruan/Hovorka, *IEEE TBME* 2017, PMID 28113240) |
| diurnal variation of ISF | numerators 1736 (am) / 1873 (pm) / 2035 (evening) (Hegab, *Front Pediatr* 2022) |
| what counts as an error that matters | ~10% (titration increment); ~20% (insulin-dose error grid) |

If a computed quantity is two orders of magnitude off a published clinical value, that is the finding,
and it outranks whatever you were working on. Adults in the current cohort have τ = 1.65 mg/dL/U
against a clinical 30–80. Nobody checked for two papers.

### 2.3 Recompute every derived constant from data

Any constant whose name asserts a relationship (`snr`, `r2`, `coupling`, `explained_variance`) must be
recomputed from simulated output and asserted to match its name. Two dimensional errors of exactly
this class shipped here:

```python
'snr': BETA_STRONG / SIGMA_STRONG           # claims 4.0; measured SNR_Z = 0.546
# and, in a docstring:
# "fraction of fatigue variance explained by stress ~ 0.10²/(0.10²+0.03²) ≈ 92%"
#                                                    measured R² = 0.529
```

Both omit `var(stress)`. Both were quoted as fact by two subsequent audits.

### 2.4 Report distributions and failures, never just means

Every results table carries: bias, empirical SD, MAE **and MAE as a fraction of mean |target|**,
median and IQR, measured coverage, and the **failure rate** — the fraction of units where the
estimator was non-finite, rank-deficient, fell back, or landed outside the plausible range. Heavy
tails are the normal behaviour in this problem class; a mean alone hides them.

### 2.5 One claim, one script, one artifact

Every claim maps to a named cell in a generated CSV, produced by a named script, listed in a
claims-map file. `paper.pdf` claims this infrastructure exists; it does not. Build it before claiming
it, and do not count hypothesis-asserting tests in the assertion total.

---

## 3. The twelve gates

These are the CI-blocking checks. Every one has an expected value fixed **independently of whether the
method works**. Full specification and current status: [`audit/REMEDIATION_PLAN.md`](audit/REMEDIATION_PLAN.md) §2.

**Benchmark validity**
- `G-PHYS` — cohort mean glucose ∈ [120, 200] mg/dL; median time-in-range ≥ 0.50
- `G-CLAMP` — < 1% of trajectory at a state clamp; no finite difference computed across a clamp
- `G-PLAUS` — 2 ≤ |τ_i| ≤ 60 mg/dL/U, correct sign, cohort median within 2× of clinical ISF
- `G-HETERO` — `corr(frac_clamped, τ_i)² < 0.05` — heterogeneity must not be a numerical artifact

**Estimator correctness**
- `G-RANK` — every design matrix full rank, condition number < 1e10
- `G-NULL` — at zero confounding, every estimator recovers the truth
- `G-EQUIVAR` — scaling `A` by `k` scales `τ̂` by `1/k`, to 1%
- `G-ORACLE` — the oracle that observes `U` has strictly the lowest MAE in the table; **if it does
  not, the benchmark cannot separate bias removal from approximation error and no other row may be
  reported**
- `G-DISTINCT` — no two estimators agree beyond 6 significant figures
- `G-FLOOR` — the method beats the best constant, the oracle constant, and CV-tuned scalar shrinkage

**Inference and process**
- `G-CALIB` — all coverage measured from ≥200 Monte Carlo replications; any constant tuned to a
  coverage target is a build failure
- `G-ESTIMAND` — `docs/ESTIMAND.md` exists and is current

Gates run on **100% of the cohort**. The current ground-truth validator inspects 10 of 50 patients
and therefore misses the patient whose τ is exactly zero.

`G-FLOOR` is a floor, not the scientific claim. Beating a constant is a precondition for a number
meaning anything. The scientific claim is *reported*, never *asserted*.

---

## 4. Workflow

### Definition of done

A change is done when all of the following hold. Not "tests pass".

1. The relevant gates are green, and you say which ones and what they measured.
2. Any number you report is reproduced by a committed script whose path you name.
3. If you touched a quantity with a physical meaning, you state its value and the external range it
   falls in.
4. If you could not do something, it is written down — in the stub registry if it is a placeholder, or
   in your response if it is a limitation.

### When you are blocked

The correct move is to **say so and stop**. The wrong move — and the one that produced every finding
in the audit — is to write something plausible that lets the pipeline run. If a quantity is hard to
compute:

- **Do**: implement it correctly but slowly; or register a stub with its removal condition; or report
  that the component is not implemented and explain what it would take.
- **Do not**: substitute a proxy that "captures the same idea", add a regularisation that makes the
  singularity go away, or relax a threshold until the check passes.

If a check fails, the first hypothesis is that **the check is right and the code is wrong**. Changing
a threshold to make a test pass requires the same justification as changing the science, because it
is the same act. When criteria genuinely need revising, record the revision and the reason — this
repository does that well in `test_success_criteria.py`, and that transparency is a strength worth
keeping.

### Reviewing your own work

Before reporting a result, try to break it:

- Does it survive scaling the treatment by 2? (unit equivariance)
- Does it survive setting the effect to zero? (null test)
- Does an oracle with more information do better? (if not, the benchmark is broken)
- Is it beaten by a constant, or by one-parameter shrinkage? (if so, it is not a finding)
- Is the number's magnitude physically plausible?
- If I ran this with a different random seed, would the sign hold?

Every one of those questions, asked once, would have caught the central defect.

---

## 5. Project map

```
causal_eval/  (Stage 0.5/0.6 done — package rename to aegis/ still open, cosmetic only)
  simulator/        Hovorka 10-state ODE. Moved here from verification/simulator/patient.py
                    (Stage 0.5); normal package import, no more sys.path injection. This is
                    the copy that gets edited; archive/v1/ keeps its own frozen copy so audit
                    finding line numbers (e.g. CRITICAL_REVIEW.md T0-1's patient.py:318-323)
                    stay valid against the state they were found in.
  dgp/              cohort construction, confounders, proxies, ground truth
  estimators/       one file per estimator; each states its estimating equation and citation
  evaluation/       experiment orchestration, metrics, gates. results_runs/ holds every run
                    (timestamp+git-sha+n, write-once); results/ is the published pointer,
                    updated only by explicit publish_results() (Stage 0.6, fixes T3-8).
docs/ESTIMAND.md    prerequisite for any estimator work (does not exist yet — Stage 2)
audit/              CRITICAL_REVIEW.md, REMEDIATION_PLAN.md, RESEARCH_POSITIONING.md — historical
                    record as of the audit date; line-number references there describe
                    archive/v1/verification/'s frozen copy, not causal_eval/'s current one.
audit/repro/        every audit claim, as a runnable script
archive/v1/         the 5-layer closed-loop system, retired (moved from verification/, Stage 0.5)
```

State vector is **10** elements: `[S1, S2, I, x1, x2, x3, Q1, Q2, G1, G2]`. `Q1` is index 6. The
README says 11 and is wrong.

### Commands

```bash
python -m pytest causal_eval/tests/ -q          # currently 33 pass; see §0 for why that is not reassuring
python -m causal_eval.evaluation.experiment     # full run; DO NOT run for results until Gate 1 is green
python audit/repro/rev_static_checks.py         # ~5 s, no ODE — fastest way to see the core defects
```

`pytest` must run with warnings enabled. If it does not, someone reintroduced a `filterwarnings` call.

### Environment

Python 3.13, numpy 2.5, scipy 1.18, scikit-learn 1.6.1, pandas 2.2.3, matplotlib 3.10.1. There is no
dependency manifest yet — add `pyproject.toml` with pins (Stage 0.5) and record `pip freeze` into
every results directory.

---

## 6. Domain facts worth knowing before you touch the DGP

- Insulin **lowers** glucose, so τ < 0 in the sign convention used here. A τ near zero means insulin
  has no effect, which is not a thing in Type 1 Diabetes.
- The Hovorka ODE has hard switches: renal clearance `FR` activates above 162 mg/dL, `F01c` saturates
  below 81 mg/dL, `EGP ∝ max(1−x₃, 0)`, all states are floored at 0 each substep, and `Q1` is clamped
  to [40, 400] mg/dL equivalent. **Finite differences near any of these are meaningless.** Use a
  1-Unit dose contrast, not an infinitesimal derivative — 1 U is also the clinically meaningful unit.
- Insulin action peaks around 55 minutes (`tmaxI`) and persists 3–5 hours. A 60-minute outcome window
  captures a fraction of the total effect; say which fraction when comparing to clinical ISF.
- The 60-minute glucose change depends on the **entire recent insulin and carbohydrate history**, not
  on the current dose. Static regression on `[glucose, carbs, hour]` is misspecified, and the
  resulting bias is an order of magnitude larger than the confounding bias the proximal machinery
  targets. Partial out lagged treatment and carbs with cross-fitted flexible learners.
- Stress raises glucose *and* causes insulin resistance. The current simulator only does the first
  (additive on `dQ1`), so there is no treatment–confounder interaction anywhere in the DGP. The
  clinically real case is the untested one.
- A per-patient scalar τ is a questionable estimand: within-patient temporal variation in τ(t) is
  1.8–2.1× the between-patient variation this project is trying to detect.

---

## 7. Communication

Report what happened, including what did not work. If a result got worse, say it got worse and by how
much. If you are uncertain whether something is a bug or a real effect, say that rather than picking
the flattering interpretation.

State magnitudes with their scale: "MAE 2.12 mg/dL/U" is uninterpretable; "MAE 2.12 mg/dL/U, 39% of
the mean effect size, against a clinical titration increment of 10%" is a claim someone can evaluate.

The reader is a co-author who will be defending this in review. Give them what they need to be
attacked.
