# moe-phone — developer guide

Practical companion to [`ARCHITECTURE.md`](ARCHITECTURE.md) (what to build and why),
[`POSITION.md`](POSITION.md) (the research claim and the stub registry) and
[`README.md`](README.md) (the gate record). This file is the one to open when you want to *run*
something or *write* something.

Read [`../CLAUDE.md`](../CLAUDE.md) first. It is not optional here: this project has retracted
three claims, and two of them came from code that ran fine and tested green.

---

## 1. Setup

```bash
# the venv is required -- system pip is broken on this machine
.venv/Scripts/python.exe -m pytest moe-phone/tests -q     # 44 tests, ~30 s, no GPU needed
```

The offline analysis (everything under `gates/` except `traces_sparsity.py`) needs only numpy.
`traces_sparsity.py` needs torch + `transformers==4.56.2` and a GPU; it is the only piece that
does.

---

## 2. What each script is for

### The engine's design, in dependency order

| script | answers | reads | writes |
|---|---|---|---|
| `gates/traces_sparsity.py` | which experts does the model's own router pick, and what does gate-first sparsity cost? | a HF checkpoint (GPU) | `traces_<model>.npz`, `g3_<model>.json` |
| `gates/cache_sim.py` | what hit rate does a policy get, under which cache **scope**, which **replay**, which **lookahead horizon**, at which **predictor accuracy**? | `traces_*.npz` | `cache_*.json` |
| `gates/scope_compare.py` | per-layer quotas vs one shared pool, at equal total bytes | `traces_*.npz` | `scope_compare_*.json` |
| `gates/expert_policy.py` | do any classical policies (LFU, warming, pinning) beat LRU? | `traces_*.npz` | `expert_policy_*.json` |
| `gates/engine_sim.py` | how many **seconds per token** does the proposed loop take, priced per *accepted* token? | `traces_*.npz` | `engine_sim_*.json` |
| `gates/byte_budget.py` | the flash roofline across candidate checkpoints | HF configs | `byte_budget*.json` |
| `gates/engine_target.py` | per model: what does the measured policy deliver, and what would it need? | `byte_budget_measured.json`, `cache_fcrit_*.json` | `engine_target.json` |
| `gates/claims_check.py` | does every number in the prose still match its artifact? | all of the above | `CLAIMS.md` |

### The device track

`device/g0_probe.sh`, `device/memprobe.c`, `device/ufsbench.c`, `device/devprobe.c` run on the
phone; `gates/g0_summarize.py` and `gates/g1_analyze.py` parse their output. Procedure:
[`protocols/RUNBOOK.md`](protocols/RUNBOOK.md).

---

## 3. Reproducing every number

```bash
P=.venv/Scripts/python.exe
T=moe-phone/results/2026-09-14/traces_OLMoE-1B-7B-0924.npz

# G2: hit rate by scope, replay, control and lookahead horizon
$P moe-phone/gates/cache_sim.py $T --controls --both-scopes --both-replays \
    --lookahead 0,1,2,4,8 --fractions 0.05,0.10,0.125,0.20,0.30 \
    --out-name cache_OLMoE-1B-7B-0924

# the pre-registered sweep across f_crit = k/E
$P moe-phone/gates/cache_sim.py $T --fractions-around-crit --both-replays \
    --out-name cache_fcrit_OLMoE-1B-7B-0924

# how far a NOISY lookahead gets, under each eviction rule
$P moe-phone/gates/cache_sim.py $T --lookahead 4 \
    --pred-accuracy 1.0,0.9,0.8,0.7,0.5,0.3,0.0 --fractions 0.10,0.20 \
    --max-tokens 16384 --out-name cache_pred_OLMoE-1B-7B-0924

# per-layer vs shared pool, equal bytes
$P moe-phone/gates/scope_compare.py $T --max-tokens 8000 --fractions 0.10,0.20

# classical policies on a held-out split
$P moe-phone/gates/expert_policy.py $T --fractions 0.05,0.10,0.20,0.30,0.50

# the decode loop, priced per accepted token
$P moe-phone/gates/engine_sim.py $T --bulk-gbps 2.806 --expert-mb 3.54

# per-model verdicts on the measured 15R numbers
$P moe-phone/gates/engine_target.py --bulk-gbps 2.806 --ram-gb 4.85 --target-tps 5

# and finally: does the prose still match all of that?
$P moe-phone/gates/claims_check.py
```

**Cost warning.** `--lookahead` scans the cache on every eviction, so runtime grows with cache
size. The full trace at `--fractions ...,0.50` takes tens of minutes; the grids above are chosen
to stay reasonable, and nothing above 0.30 changed a conclusion when swept.

---

## 4. Build order for the engine

From [`ARCHITECTURE.md`](ARCHITECTURE.md). Each step is independently useful, and each one's
benefit was measured before it was put on the list.

1. **Per-layer expert cache with LRU eviction, event-atomic fetch.** This is the baseline and it
   is already worth double digits of tok/s on three candidate models. Do not build a shared pool
   (§2.1) and do not build static pinning (§2.3); both were measured and both lose.
2. **Cache warming at load.** Buys time-to-first-token, not throughput. Do not count it in the
   byte budget.
3. **A next-token, same-layer expert predictor, used in `protect` mode.** The predictor may only
   veto an eviction candidate; recency still ranks. Measure its accuracy *first* (**S12**) — it
   is the cheapest open measurement in the project and it is the input to this step.
4. **Only then**, if a drafter with α ≥ 0.9 exists, multi-token verification (§4). It is a
   regression below α ≈ 0.85, so measure α before writing any of it.

**What not to build:** expert *prefetch*. Three independent published negatives, including a
trace-driven oracle prefetching perfectly one token ahead that gained ~8%. See
[`LEADS.md`](LEADS.md) §8. Prediction used for *eviction* is a different thing and is step 3.

---

## 5. Adding a number to a document

1. Make the script write it into `results/<date>/`.
2. Add an entry to `CLAIMS` in `gates/claims_check.py` — id, the sentence, the artifact, an
   accessor, the expected value, a tolerance.
3. `python moe-phone/gates/claims_check.py --emit` to regenerate `CLAIMS.md`.
4. Cite it in prose.

`tests/test_gates.py` then fails if the artifact stops producing that number, which is the point:
a re-run gate cannot silently orphan a sentence in a paper. This already caught a hand-typed
`0.490` where the artifact said `0.5015`.

---

## 6. Adding a gate or a policy

Read `CLAUDE.md` §9 first, then:

- **A test asserts a property fixed by mathematics, physics or external literature — not the
  result.** `assert new_policy > old_policy` is not a test; it is a filter that makes the negative
  result unrepresentable. The existing suite has worked examples of the alternative: identities
  (`lookahead_hits(H=None) == belady_hits_scoped`), bracketing (`sequential ≤ any horizon ≤
  Belady`), monotonicity (more lookahead cannot hurt), recovery (LRU on locality-free routing
  returns the cache fraction), and one asserted **negative** (`rank` mode with a wrong predictor
  scores *below* the no-lookahead rule) so a refuted claim cannot quietly come back.
- **Simulate the machine you will build.** Both retracted G2 numbers came from simulating a
  *different* machine than the one the policy assumed — a shared pool against a per-layer policy,
  and per-access eviction against batched fetch. Neither was a coding error. Both passed tests.
- **Run the control.** A hit rate above the floor means nothing until you know what the floor is
  for *that* policy: the cache fraction is the correct null for LRU and the wrong one for Belady,
  which beats it even on locality-free routing.

---

## 7. Scripted edits to this repo

Several of the defects in `README.md` came from tooling, not analysis. If you edit files with a
script rather than by hand:

- **Assert the pattern is present and unique before substituting.** `str.replace` on a missing
  pattern is a silent no-op that reports success — that is how a corrected table shipped under a
  stale header, and it is §0's failure mode in miniature.
- **Parse the result before writing it.** `ast.parse` on Python, and check the file still renders.
- Shell heredocs eat a level of backslash escaping. Write patch scripts to a file rather than
  piping them through a heredoc when they contain `\n` inside string literals.

---

## 8. Where the open work is

The four ranked open measurements are in [`ARCHITECTURE.md`](ARCHITECTURE.md) §6; the full stub
registry with removal conditions is in [`POSITION.md`](POSITION.md) §11; external work that bears
on each failure mode, with provenance marks, is in [`LEADS.md`](LEADS.md).

Shortest path to the next real result: **measure the cross-token, same-layer predictor's accuracy
(S12)**. No GPU, the traces are already committed, and it is the input to the primary design.
