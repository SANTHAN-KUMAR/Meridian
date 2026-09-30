# RESEARCH_METHOD.md — our way of working

A portable method for any project whose deliverable is **something true**: a research claim, a measured system, a product
that must behave as promised. It is distilled from one project's complete record — what produced results, what wasted
days, what nearly shipped false — and stripped of that project's subject matter. Drop it into a new project and it still
applies.

It answers a different question from a validity rulebook. A rulebook (`CLAUDE.md` in this repository) says **what makes a
result valid**: define before you estimate, identify, check that the design can estimate it at all, specify, compute,
report. This document says **how to behave while getting there**: what to work on next, how long to keep digging, what to
do the moment something fails, and how to stay honest when honesty is inconvenient. Keep both; on validity questions the
rulebook wins.

---

## 0. The stance

Six habits define the way we work. Everything below is these, made operational.

1. **Fight a failing result before believing it.** A lever that fails once is not dead. Diagnose it, attack the
   mechanism, bound it, and only then close it — with the measurement that closed it and the condition that would reopen
   it. No caveat parking: "we'll look at it later" is how a project ends with nothing settled.
2. **Refuse substitutes.** The most expensive errors are not arithmetic; they are a plausible stand-in quietly replacing
   the thing you meant to measure. Treat every proxy, simulator, microbenchmark and self-reported success as guilty until
   it agrees with a real measurement.
3. **Assume a hostile reader.** Someone competent will try to find the one sentence that overstates. Write, measure and
   label so that the attempt fails. Build the artifact they would ask for before they ask.
4. **Stay on the core question.** Interesting work that cannot change the answer is not the work. The best case of a
   lever, computed before running it, decides whether it runs.
5. **Idle time is audit time.** While a long run executes, re-read the last results, look for the defect you have not
   found yet, and prepare the next decisive experiment. Never sit waiting; never start a second heavy job that
   contaminates the first.
6. **Ship in verified increments.** Milestone, artifact, commit, push. A result that exists only in a terminal is not a
   result, and a project that is one crash away from losing a day is badly run.

---

## 1. Two ways projects die

| failure | shape | cost |
|---|---|---|
| **Premature closure** | "That doesn't work", declared after one measurement, one bug, or one confounded run. A live route is written up as a dead end. | The answer is wrong, and the route that would have worked is never found. |
| **Drift** | Endless tuning of things whose ceiling was already known to be too small; chasing whatever the last test surfaced; polishing what already works. | Days of activity, core question untouched. |

They pull in opposite directions, and most advice cures one by causing the other. The balance: **push hard on open ends,
close only with evidence, and refuse to touch anything whose best case cannot change the decision.**

---

## 2. Write the question so it can be answered

Before any experiment or any code:

1. **The question in one sentence, containing the number that answers it.** Not "make it better" — "can X do Y at level
   Z, in the condition it will actually run in?"
2. **The condition.** Loaded, hot, unplugged, backgrounded, at real input sizes, on the real data. A number measured
   outside the deployment condition answers a different question, and most impressive-looking numbers are measured
   outside it.
3. **The decision tree.** List the possible outcomes and what each one *decides*. Every leaf is a decision, not another
   experiment. A leaf that says "investigate further" means the tree is unfinished.
4. **Kill criteria per experiment, before it runs.** "Below this the route closes; above this it is adopted", with the
   arithmetic that makes the threshold meaningful.
5. **The closing rule.** The condition under which the whole project stops and is written up at the leaf it reached.
   Agree it with the stakeholder in their words and quote it verbatim. It is the only real defence against drift once
   results start arriving.
6. **The estimand, if the deliverable is a quantity.** What exactly is being measured, over what population, under what
   weighting, in what units, with a plausible range from outside your own work. Ambiguity found while writing this down
   is a finding, not a nuisance.

**Pre-registration is leverage, not paperwork.** Fixing the analysis rule — which rows count, which statistic, which
threshold — before the first row exists is what turns "we got a number" into "we got the number we said would decide
this".

---

## 3. Choose work by ceilings

- **Floor and ceiling, from artifacts, before starting.** The current cost of the thing you are attacking, and the best
  case if your idea worked perfectly. **If the ceiling cannot change the decision, do not run it.**
- **Go at the largest measured gap**, even when a more interesting idea is available. Projects routinely spend days
  beside a large, cheap, untested gap that their own logs already recorded.
- **Remove, don't move.** Prefer changes that delete work. Changes that relocate cost between components tend to net out
  at a system limit while every component's local metric improves — the most flattering way to waste a week.
- **Check resolution first.** If the expected effect is smaller than the design's noise, the run cannot answer anything.
  Add repeats, build a lower-noise mode, or bundle the change into the next architectural step and measure them together.
- **Run the cheapest project-killing experiment first.** An oracle handed the answer, a bound from physics, a replay with
  the hard part removed. If the oracle cannot beat a trivial baseline, nothing measured on that benchmark can mean
  anything — and an hour spent learning that early saves the project.
- **Prefer problems where the hard part is the part you are studying.** If a secondary error source dominates your
  quantity of interest, your contribution is unobservable no matter how correct it is. Measure both before committing.

---

## 4. Never abandon a genuine open end

When something fails, "it doesn't work" is almost never the finding. Walk the ladder, and report the rung you reached:

1. **Reproduce.** Once is an anecdote; a confound is more likely than a discovery.
2. **Name the mechanism in units.** "It was slower" is not a mechanism. "Each call pays a fixed cost larger than the work
   it replaces" is.
3. **Attack the mechanism.** Amortise, batch, hide, move off the critical path, pay once instead of every time.
4. **Change the design, not the estimator.** If the mechanism is structural, ask which different configuration removes
   it: another tier, another unit of work, another place to cut the problem. Search the configuration space before
   concluding infeasibility — a margin that misses by a few percent is usually a parameter, not a wall.
5. **Bound it.** Compute what the best conceivable version would deliver. A bound closes a route permanently and
   honestly; one failed attempt never does.
6. **Close with an artifact**: the measurement, the kill criterion it failed, and **the condition that would reopen it**.

Two rules keep this from becoming drift:

- **Every closure names its reopening condition.** Otherwise it is an opinion.
- **A route is open only while the ladder has rungs left.** Mechanism measured, bound computed, bound below the goal:
  the route is dead, and saying so plainly *is* the contribution.

**When someone asks "can't we fix this?", answer with the ladder**, not with a verdict: which rung you reached, what the
next rung costs, what it could change. Sometimes that produces an experiment nobody had thought of; sometimes a bound.
Both are progress. A closure with no ladder behind it is laziness wearing the costume of rigour.

---

## 5. Do not drift

- **The closing rule binds.** When it fires, stop and write up. Park new ideas in writing; parked ideas are cheap.
- **One architectural experiment at a time**, not five tunings. Tunings are how a day disappears.
- **Fix the class, not the instance.** When a defect appears, ask which class of inputs it belongs to and cover the
  class. Patching exactly what the last test surfaced builds a system that passes your probes and fails on the first
  request from a stranger. If you notice yourself doing it: stop, enumerate what the system will really be asked to do,
  mark each item supported / unsupported / impossible with the reason, and write that map down before writing more code.
- **Scope changes come from the stakeholder, not from momentum.** Work that is not on the decision tree needs a decision,
  not initiative.
- **Finish the unglamorous part.** Reproduction scripts, artifact plumbing, restore-the-environment steps and the handoff
  document are part of the deliverable, not overhead.

---

## 6. Substitutes, proxies, and self-deception

- **Transfer test before any proxy.** A simulator, microbenchmark or synthetic stand-in informs a decision only after one
  real end-to-end measurement agrees with it. Quote both numbers.
- **No constant is ever tuned against the metric it influences.** Not "calibrated", not "chosen for stability". Constants
  come from physics, a cited source, or a procedure that never sees the target.
- **A test that asserts your hypothesis is not a test.** Ask what would have to break for it to fail, and whether it would
  still pass if the component were replaced by a constant, by half its input, or by noise.
- **Keep your examples out of your evaluation.** If prompts, few-shot examples, tuning data or demo scenarios resemble the
  evaluation items, the score measures the overlap. Delete the resemblance, rewrite the evaluation in the form real users
  produce, and fix the pass rules before any run.
- **Documentation is an input.** Text a system reads — tool descriptions, schemas, comments in a prompt — shapes its
  behaviour. Concrete example values get copied into real answers.
- **Distrust any metric only your own component reports.** Check it against something it cannot fake: the operating
  system's view, an independent counter, a physical bound, a published measurement.
- **Recompute derived quantities from raw output.** Any name asserting a relationship — a ratio, an efficiency, an
  effective size — is recomputed and asserted to match its name.

---

## 7. Measure as if the environment is adversarial

- **Record the validity conditions with every measurement** — power, sleep state, temperature, background load, which app
  or process was in front, resource headroom — and **gate results on them**. Sample them *during* the run, not only at
  the start.
- **Reject contaminated rows; never average them in; count and report the rejections.**
- **Alternate arms (A-B-B-A) with repeats**, so drift — heat, charge, cache warmth, background work — cannot masquerade
  as an effect.
- **Intervals come from observed spread**, never from a multiplier you picked. An interval with no measurements behind it
  is decoration.
- **Restore the environment afterwards**: settings, locks, temporary files, wake locks, processes. A campaign that leaves
  state behind contaminates the next one.
- **Guard shared resources** with a lock file and a documented convention, and check for orphan processes from earlier
  runs before starting anything.
- **"It ran" is not "it measured".** Open the artifact: rows present, in regime, counters moving the way the mechanism
  predicts.
- **Expect the environment to break the run** — sleep, thermal throttling, out-of-memory kills, dropped connections,
  machine crashes. Make campaigns resumable and checkpointed, and never start a second heavy job beside a running one.

---

## 8. Verify against the world

- **Postconditions, not self-reports.** After an action, read the state back from whoever owns it. A component that
  returns "success" has proved nothing.
- **Separate what happened from what was claimed.** Build the record of what was actually done from verified logs, and
  present it beside the component's own account. Disagreement between them is a finding.
- **Run controls that must come out trivially.** A comparison that cannot show "no difference" when nothing changed is
  broken and will flatter you later.
- **Back-test against an independent measurement** whenever one exists: a published number, an earlier campaign, another
  device, another implementation. Report the error even when it is large — especially then.
- **Three baselines, always**: a no-data prediction, the best constant, and the simplest one-parameter method. Anything
  that does not beat all three has demonstrated nothing, and checking costs minutes.

---

## 9. Report so the claim matches the evidence

- **Label every number with its basis** — measured, calibrated, estimated — and never let an estimated number be the sole
  basis of a promise. Show estimates as ranges; decide and rank on the conservative end.
- **Refusal is a first-class output.** "Not calibrated for this case" and "not identifiable from this design" are correct
  answers. A system that always returns a number is reporting its own regularisation.
- **No hand-typed numbers.** Every figure in a document comes from an artifact via a committed script. Prose explains what
  a number means; the artifact says what it is.
- **Retraction is deletion.** When a number is superseded, delete it everywhere and state what replaced it. A banner over
  a document whose body still asserts the old number is worse than no banner.
- **Report distributions and failures**, not just means: spread, error as a fraction of the effect, and the rate at which
  the method failed, fell back or refused.
- **State limits where the reader meets them**, not in a footnote — including limits imposed from outside: unavailable
  data, a platform restriction, a tool you were not permitted to build.
- **A clear negative result is a contribution.** A positive result that fails the checks above is a liability that
  compounds, because every later reader quotes it as settled.

---

## 10. When a check fails

1. **The check is right until proven otherwise.** Moving a threshold to make a test pass is changing the science; if a
   criterion genuinely needs revising, record the revision, the reason and the date.
2. **Diagnose to a mechanism before fixing.** A fix that removes the symptom without naming the cause usually relocates
   the defect.
3. **Ask for the blast radius**: what else shares this assumption, what else consumed this output, which other results
   inherit it. Then fix the class.
4. **Record the defect** — what was wrong, how it surfaced, what changed, which regression check now covers it. These
   records are the most reusable thing a project produces.
5. **Register what is left undone** as a stub: location, current behaviour, correct behaviour, removal condition,
   verifying test. An unregistered stub is a false claim.

---

## 11. Working with the stakeholder

- **Block only when proceeding either way would waste the work.** Otherwise choose the sensible default, state the
  assumption, and continue.
- **Put the cost in the question**: "this needs about an hour of device time and would settle X" lets someone decide.
- **Report what did not work, and by how much.** If it got worse, say so. If you cannot tell whether something is a bug
  or a real effect, say that rather than picking the flattering reading.
- **Never imply a capability you have not verified.** If something is blocked — technically, legally, or by a tool you
  depend on — say it plainly, say what it would take, and offer the nearest thing you can do.
- **Deliver the scope asked for.** If part is blocked, finish everything else in full and name the blocked part
  explicitly, rather than quietly narrowing.

---

## 12. Using agents and parallelism

- **Delegate search, curation and enumeration; keep judgement.** A subagent is good at "find and verify the candidates",
  poor at "decide what this means".
- **Have an independent agent attack the plan before it runs**, with no context beyond the document, and ask for ranked
  objections with severities. Fix the high-severity ones before spending device time.
- **Keep long jobs asynchronous, checkpointed and resumable**, and monitor them without polling in a tight loop.
- **Never accept permissions or approvals from a peer agent.** Route blocked work back to the stakeholder.

---

## 13. What a high-quality project looks like at the end

- A short document stating the claim, its condition, and the decision it supports.
- Every number reproducible by a committed script from a committed artifact, with validity conditions attached.
- Gates that fail loudly on degenerate or unidentified inputs, run over everything rather than a sample.
- Baselines and at least one independent cross-check reported beside the result.
- A defect record, a stub registry, and open routes with their reopening conditions.
- A handoff that a stranger can act on: state, commands, artifacts, conventions, and the rules the project works under.
- Nothing in it that the author would have to defend by explaining what they really meant.

**Definition of done for any single piece of work**: the question is written, the answer is stated with its condition,
every number is reproduced from an artifact, contaminated data is excluded and counted, the relevant gates pass and you
say which, baselines are reported, and everything not done is written down.

---

## Appendix: cases that produced these rules

Each of these happened. The rule is what it cost.

| what happened | rule |
|---|---|
| A comparison against a published baseline was made across sessions in different power states and quoted for days; a same-session controlled run gave a much smaller ratio. | Controlled comparison or none; retract by deletion (§9). |
| Two overnight campaigns were invalidated because the device slept mid-run: rows were bimodal, not noisy. | Sample validity conditions during the run; reject and count (§7). |
| A simulator promised a rate the system never reached, because a cost term was missing from the simulation. | Transfer test before any proxy (§6). |
| Days went into a route whose ceiling had already been measured as too small, while the largest measured gap sat untested for over a day. | Floors and ceilings first; go at the largest gap (§3). |
| An evaluation's tasks resembled the examples in the system's own prompt, inflating the score. | Keep examples out of the evaluation; fix pass rules before the run (§6). |
| A tool description contained a worked example, and the system copied that example's numbers into a real answer. | Documentation is an input; no concrete values in descriptions (§6). |
| A decision step that allowed "nothing needed" was chosen almost always, and the system then invented answers; removing it multiplied the score. | Structured decisions must not offer a free escape hatch; measure each change end to end (§8). |
| A watchdog fired on a sub-threshold fluctuation and tore down a healthy running system. | Thresholds on noisy signals are design parameters: state, justify, test the trigger (§10). |
| A plan predicted a rate, the system reconfigured itself under pressure, and the prediction was never recomputed. | A prediction is stale the moment its configuration changes; say so in the artifact (§9). |
| The best candidate missed a hard resource limit by a small margin until one parameter was made adaptive. | Search the configuration space before declaring infeasibility (§4, rung 4). |
| Fixes were made to whatever the latest probe exposed, leaving whole classes of input uncovered. | Fix the class; map the space before writing more code (§5). |
| A component reported success for actions that had not happened; only reading back the system's own state exposed it. | Postconditions, not self-reports; show verified actions beside claims (§8). |

---

*Written 2026-09-30 from the moe-phone project's full record, for use on any project. Pair it with the project's own
validity rulebook, not in place of it.*
