# Hostile review of `research/2026-09-19_RESEARCH_SPEC.md`

**Reviewer stance.** Independent, adversarial, no prior context beyond `CLAUDE.md`, `ESTIMAND.md`, `CLAIMS.md`,
`research/2026-09-19_STATE_OF_RESEARCH.md`, `results/2026-09-19/NIGHT_SUMMARY.md`, and the artifacts the spec
itself cites. I did not run anything; every number below is read from a file already on disk. Where I recomputed
arithmetic, I show it.

**Bottom line up front.** The spec is unusually well-sourced compared to most systems documents — most of its
numeric citations check out verbatim. But it has one severe, structural defect (§2 below): it explicitly names
non-additivity as an assumption to stop making (§3.1), then performs additive arithmetic on exactly the same
three quantities to produce its one optimistic scenario ("10 tok/s reachable"), using evidence that the spec's
*own* preceding section (§2.1) shows to be non-additive in practice. That defect, plus two decision-tree gaps and
one experiment (E4) whose "decisive" k is not the k the engine actually dispatches at, mean the tree as written
will not close in one pass. It will very likely re-open exactly the cycle `STATE_OF_RESEARCH.md` §5 already
diagnosed: cheap wins early, then a widening set of conditional branches.

---

## Ranked objections

### 1. [SEVERITY: HIGH — invalidates the headline "10 tok/s is reachable" branch] §5's additive arithmetic uses the assumption §3 tells you to stop making, and §2.1's own evidence refutes it for one of the three terms being added

**Claim.** §5: *"A + E + C at their ceilings: 148 − 19 − 17 − 22 = 90 ms → 11 tok/s. A + C alone: 107 ms → 9.3."*
This is the arithmetic basis for the entire "(a) 10 tok/s Tier E is reachable" leaf of the decision tree (§7) and
for the recommendation to build H-A + H-B + H-C (§8).

**Why it is wrong.** §3, assumption 1, states as a thing to *stop assuming*: *"That the three terms are
independent and additive. They share one wall clock, one power budget and one memory."* §2.1 then gives five
concrete, measured counterexamples where a lever that removed ms from one term added ms to another, net near
zero: the slot arena (mgmt −15 → compute +12, stall +4, twice), kgsl pinning (cache faults → hot-data faults,
compute +78), the swap guard (faults −15% → its own checks +2.3 ms), the GPU tier (expert compute −6 →
CPU wait +7), run-time repack (compute −15 → stall +17). Every one of these *is* a case of two of {compute,
stall, mgmt} interacting through the shared wall clock/memory budget that §3 warns about.

Mechanism E (cache-management removal, one of the three terms summed to reach "11 tok/s") is not a hypothetical
lever — it is literally the slot arena, which the spec's own §2.1 and the arena3 result (`results/2026-09-19/arena3_summary.json`,
also quoted at `research/2026-09-19_STATE_OF_RESEARCH.md` §4, row "anonymous slot arena, awake") already ran and
found net-neutral: mgmt −15.3 ms, compute +12.2 ms, stall +3.6 ms, decode unchanged (5.48 → 5.46 tok/s). The
"~17 ms" ceiling the spec assigns to mechanism E in the §5 table is the ceiling *before* the compute/stall
rebound that its own arena experiment measured. Summing E's raw ceiling against A and C's raw ceilings, when E's
own measured net effect on wall time is ≈0, is not "arithmetic of the Tier-E route" — it is re-adding a term the
project has already shown does not survive contact with the shared resource.

**Evidence.** `research/2026-09-19_RESEARCH_SPEC.md` §3 item 1 vs §5 "Arithmetic of the Tier-E route"; §2.1;
`results/2026-09-19/arena3_summary.json`; `research/2026-09-19_STATE_OF_RESEARCH.md` §4 row "anonymous slot
arena, awake" and §4.3.

**What would fix it.** Either (a) do not sum ceilings across terms that share a resource — report only the sum
of *measured, wall-clock* deltas from controlled A/Bs that were run simultaneously (which for E is ≈0, not 17),
or (b) if the spec wants to keep a hypothetical additive projection, label it explicitly as violating its own
§3.1 assumption and give the reader the arena's demonstrated counterexample in the same paragraph, not two
sections earlier. As written, a reader who trusts §3 and then reads §5 gets two contradictory instructions from
the same document.

---

### 2. [SEVERITY: HIGH — undermines E4's claim to be decisive] E4 tests the GPU kernel at k=8, but the engine's actual GPU dispatch runs at k≈2, where the same cited logs show the rate is much lower

**Claim.** E4 (§6): *"does the expert FFN kernel reach ≥ 25 GB/s on the Qwen3 shapes at k = 8... Kills: < 15 GB/s:
the Adreno cannot beat the repacked CPU per byte; the GPU branch is closed permanently."*

**Why it is wrong.** At an 88% cache hit rate (`stack_hit`, confirmed below), the expected number of *misses*
per layer of 8 routed experts is 8 × (1 − 0.88) ≈ 1. The already-run GPU tier experiment reports exactly this:
`results/2026-09-19/bmoe_gtier/bmoe_gtier_ab_20260919_0447/README.md` — *"Device: 0 failures, 0 risk recomputes,
~2.1 experts per dispatch."* So the real workload dispatches the GPU kernel at k ≈ 1–3, not k = 8.

The two files the spec cites for the "7-11 GB/s" bit-exact figure (`gx_m6_020545`, `gx_m6_082901`) contain
per-k breakdowns, and the rate is strongly k-dependent and much worse at low k:

| file | variant | k=1 GB/s | k=2 GB/s | k=3 GB/s | k=4 GB/s |
|---|---|---|---|---|---|
| gx_m6_020545 | 0 | 4.41 | 6.45 | 7.45 | 7.92 |
| gx_m6_020545 | 1 | 4.43 | 7.23 | 8.63 | 10.03 |
| gx_m6_082901 | 1 | 3.72 | 6.19 | 7.28 | 8.89 |
| gx_m6_082901 | 3 | 3.68 | 6.25 | 7.96 | 8.82 |

(`bench.out` in each directory, `BENCH spin=1 ... GBps_at_median=`.) At k=1 — closer to the real dispatch size —
the bit-exact kernel runs at 3.7–4.4 GB/s, roughly a third of the "7-11 GB/s" figure the spec quotes, which is
in fact the k=4 number. E4's own construction (§6.4) uses k=8, the single point in the whole sweep with the
*best* per-byte rate, and the spec's positive/kill thresholds (≥25 / <15 GB/s) are set relative to that best
point. A reordering-tolerant kernel that clears 25 GB/s at k=8 could easily still run at well under 15 GB/s at
k=1–2 — the actual regime — and E4 would still report "positive," triggering a day of engineering (§5.1, "then a
1-day port of the whole graph") on a kernel that cannot help the real dispatch pattern. This is a case of "cheaper
or more direct experiment exists": E4 should sweep k=1..3 with realistic weighting by the measured miss
distribution, not test at the favorable end of the curve and call it decisive.

**Evidence.** `results/2026-09-19/bmoe_gtier/bmoe_gtier_ab_20260919_0447/README.md`; `results/2026-09-19/gx_m6_020545/bench.out`;
`results/2026-09-19/gx_m6_082901/bench.out`; `CLAIMS.md` row `stack_hit`.

**What would fix it.** Re-derive E4's decisive threshold as a *weighted* average GB/s over the realistic
per-layer miss-count distribution (mostly k=0-2, occasionally higher), not a single k=8 number, and report the
k=1 rate explicitly since it dominates the expected dispatch count.

---

### 3. [SEVERITY: MEDIUM-HIGH — one L1 quantity is internally inconsistent by ~7-8%, and it is the anchor of the whole budget] "Weight bytes per token, total ~2.0 GB" does not match the project's own directly-measured active-bytes-per-token

**Claim.** §1.1: expert 1.02 GB + dense ~0.95 GB (the "anon copy") = **"~2.0 GB"** total weight bytes per token.
This total feeds every downstream ms figure in §1.2 (30 ms expert + 29 ms dense = 59 ms of the 80 ms compute
floor).

**Why it is wrong / unreconciled.** `results/2026-09-18/gguf_active_qwen_olmoe.json` (the source of CLAIMS.md's
own `qwen3_active_mb_per_token` = 1839.97 "MB", checked and "ok") gives, for the *same file*:
`active_bytes_per_token: 1839971456`, `expert_bytes_per_token: 1023934464`. Expert bytes match the spec's 1.02 GB
exactly (confirmed below). But `active_bytes_per_token − expert_bytes_per_token = 816,036,992` bytes = **778.4
MiB**, not the 945 MiB the spec adds for dense weights. The spec's dense figure comes from a *different*
artifact — a runtime log line (`bmoe: dense-weights=anon — 945 MiB in 435 anon buffers`, confirmed below) that
measures the engine's allocated anonymous-copy footprint, not the GGUF's own dense-tensor byte count. These two
numbers for "dense bytes touched per token" disagree by 21% (778 MiB vs 945 MiB), and the spec's own headline
total (1.02 + 0.95 = 1.97 GB) disagrees with its own claims-map total (1.84 GB) by 7%, without any note that two
different artifacts were combined or that they disagree. Either the 945 MiB anon allocation includes bytes that
are not "touched" every token (padding, alignment, buffers for tensors not on the hot path), in which case the
compute-floor's "29 ms dense" row is inflated by a similar 7-20%, or the GGUF-derived active-bytes figure is
missing something the runtime actually pays for — either way this is an L1 definitional gap (CLAUDE.md §2 item
5: "the estimator's implicit weighting... is the one that gets skipped") that was skipped here.

**Evidence.** `results/2026-09-18/gguf_active_qwen_olmoe.json` (`active_bytes_per_token`, `expert_bytes_per_token`
fields); `results/2026-09-17/bmoe_verify/b1_rep1.err:1735` (`dense-weights=anon — 945 MiB in 435 anon buffers`);
`CLAIMS.md` row `qwen3_active_mb_per_token`.

**What would fix it.** Reconcile the two dense-byte numbers explicitly (are the anon buffers padded/duplicated,
or is the GGUF number missing tensors?) before using either in a floor computation, and state which one the 29
ms dense-matmul floor actually uses.

---

### 4. [SEVERITY: MEDIUM-HIGH — the dense-matmul floor's uniform-rate assumption is contradicted by the *same cited artifact*] "Dense matmuls: 0.95 GB / 33 GB/s = 29 ms... not measured on these shapes [H]" — the node ledger the spec cites elsewhere shows attention shapes running far below 33 GB/s

**Claim.** §1.2 table, "dense matmuls" row: 29 ms, derived by applying the *expert* kernel's repacked rate
(33 GB/s) to dense weights, flagged `[H]` (hypothesis) because it is "not measured on these shapes."

**Why it is unsafe.** The spec cites `ceiling_ledger.json`'s `node_ledger` elsewhere in the same table (for the
"attention, norms, router, small ops" row). That same `node_ledger.median_GB_s` block — sitting in the file the
spec already opened — gives the *generic*-kernel per-byte rates for the individual dense/attention projections:
`Qcur = 28.26 GB/s`, `Vcur = 17.15 GB/s`, `Kcur = 9.02 GB/s`. Kcur in particular runs at less than a third of
the 33 GB/s the spec assumes the *repacked* dense matmuls will reach. A rate this low, on a small matrix
(head_dim-sized), is a signature of fixed per-call overhead dominating rather than memory bandwidth — exactly
the kind of bottleneck that kernel repacking (which improves per-byte throughput on large matmuls) does not fix.
Assuming a uniform 33 GB/s across all dense matmuls, including the worst-performing one in the project's own
ledger, is optimistic by a factor that could matter: if Kcur-like ops stay near 9-17 GB/s after repacking, the
29 ms dense floor understates itself by perhaps 10-15 ms, which alone would move E1's "≤ 80 ms" positive
threshold into its own kill zone (≥ 95 ms).

**Evidence.** `results/2026-09-18/ceiling_ledger.json` → `node_ledger.median_GB_s` (`Qcur`, `Vcur`, `Kcur`);
compare to §1.2's dense-matmul row and §11's own item 2 ("The dense-matmul rate of the repacked kernels on
Qwen3's attention shapes (assumed equal to the expert rate)" — registered as unknown, but not flagged against
the contradicting evidence sitting in the very same artifact).

**What would fix it.** Before E1 runs, pull `node_ledger`'s per-op rates for every dense matmul (not just the
three named categories) and check whether any of them are overhead-bound rather than bandwidth-bound; if so,
repacking will not lift them to the expert rate, and the dense-matmul floor needs a per-op breakdown, not one
number.

---

### 5. [SEVERITY: MEDIUM] "Full clock" is not the hardware maximum, so the "50 ms at full clock" floor and mechanism F's ceiling are computed against an already-partially-throttled reference point

**Claim.** §1.2: *"compute floor... ~50 ms at full clock (compute scales with clock: 77 vs 121-124 ms [CL:
trace_runs.plain, capped_terms])."* Table row F (§5): energy-per-token → higher sustained clock, "ceiling up to
40 ms."

**Why it is imprecise.** `ceiling_ledger.json`'s `trace_runs.plain` entry — the source of the "77 ms" figure —
was captured under `"caps_before": "cap0=3033600 cap6=3436800"`. The hardware maximum, per the same file's
`clock_state.p0_levels_kHz` / the phone's own `hardware_max_ghz` (from `thermal_caps.json`: policy0 = 3.3216
GHz = 3,321,600 kHz, policy6 = 3.8016 GHz), is higher than the clock the "full clock" run actually used
(3,033,600 / 3,436,800 kHz — about 91% and 90% of true hardware max respectively). So the spec's "full clock"
reference is itself a throttled state relative to the phone's true unthrottled ceiling; it happens to be less
throttled than the "capped" state used elsewhere, but calling it "full clock" invites the reader to treat 50 ms
(and F's 40 ms ceiling, computed as `compute ×(cap/hw)`) as the best case the hardware allows, when a genuinely
unthrottled run (only reachable, per `cap_full_below_31C`, below 31°C shell temperature) could in principle do
better still. This does not change any conclusion by itself, but it means mechanism F's stated ceiling is
computed from a ratio (`cap/hw`) using a `cap` value in the numerator that both other places in the same table
already call "full," creating an internal terminology collision that a hostile reader will notice and use to
question every ratio downstream.

**Evidence.** `results/2026-09-18/ceiling_ledger.json` (`trace_runs.plain.caps_before`, `clock_state.p0_levels_kHz`);
`results/2026-09-19/thermal_caps.json` (`hardware_max_ghz`, `cap_full_below_31C`).

**What would fix it.** Rename the "full clock" row to the actual clock it was measured at (e.g. "cap0=3.03 GHz,
91% of hardware max"), and add a genuinely unthrottled (<31°C) trace run as a third reference point before
computing mechanism F's ceiling.

---

### 6. [SEVERITY: MEDIUM] Decision tree has unaddressed ranges — it does not terminate for all possible experimental outcomes, contradicting its own claim

**Claim.** §7 header: *"Decision tree (terminates; every leaf is a decision, not an experiment)."*

**Why it is wrong.** The tree only branches at the two threshold values named in each experiment's own
"positive"/"kill" criteria, and those two criteria are not exhaustive:
- E1: branches are "≥ 95 ms" and "≤ 80 ms." A result of, say, 87 ms (a perfectly plausible outcome — the
  positive/kill gap of 15 ms is not small relative to the noise the project has documented, e.g. `overhead_ratio`'s
  own SE) falls into neither branch. The tree gives no instruction.
- E4: branches are "< 15 GB/s" and "≥ 25 GB/s." A result of 20 GB/s — again plausible, since the spin-locked
  bit-exact kernel already reaches 7-11 GB/s and a 2-3x gain from dropping bit-exactness is not an unusual
  order of magnitude — falls into neither branch.

For both, the experiment sections (§6, item 7/8) at least *name* these as "positive" and "kills" without saying
what an in-between result means, and the tree in §7 inherits the gap. A tree that does not cover its own
sample space is not "terminates" — it is "terminates on 2 of the ≥3 outcomes each experiment can plausibly
produce." Given `STATE_OF_RESEARCH.md` §5's diagnosis that under-resolved campaigns were exactly the project's
historical failure mode (§4.4 of `CLAUDE.md`'s own reasoning about resolution), an ambiguous middle result is
not a corner case to hand-wave; it is close to the modal outcome for a first engine measurement of a
never-before-run replay mode (E1) or a never-built kernel variant (E4).

**Evidence.** `research/2026-09-19_RESEARCH_SPEC.md` §6 (E1 items 7-8, E4 items 7-8), §7 (the tree itself).

**What would fix it.** Add an explicit middle branch for each experiment ("80 < C ≤ 95: ambiguous, resolve with
N more replay rows before deciding" / "15 ≤ GB/s < 25: ambiguous, ...") with its own stopping rule, so the tree
is provably exhaustive before it is run, not discovered to have a gap after a phone-hour is spent landing in it.

---

### 7. [SEVERITY: MEDIUM] The 6.76 tok/s anchor number is explicitly flagged by its own source as "not a controlled comparison," but the spec treats it as the fixed operating point for a 5%-accurate reconstruction

**Claim.** §0 header and §1.3: *"today's best measured: 148 ms (6.76 tok/s)"*; §1.3: *"The reconstruction
accounts for the measured number to within 5%."*

**Why this overstates precision.** The source, `results/2026-09-19/bmoe_gtier/bmoe_gtier_ab_20260919_0447/README.md`,
reports the 6.76/6.755 tok/s figure as a labeled **"Observation, not a controlled comparison"** — a single base
arm from an A/B whose primary purpose was the GPU-tier comparison, run unplugged with the screen on, and
explicitly contrasted against the *same stack's* charger-session result of 5.3-6.3 tok/s from the day before,
with the note that "X3 (off-charger equilibrium) deserves its own run" (i.e., not yet run). `STATE_OF_RESEARCH.md`
§2's own table shows the identical "stack" configuration giving four different numbers across three days (6.20,
"+4.3%" on top of ~5.25, 5.46-5.48 on the charger, 6.76 unplugged) depending on thermal/charging state — a
20-30% spread on the *same lever set*. Building a "the reconstruction accounts for the measured number to
within 5%" claim on top of a single uncontrolled row, when the same configuration has shown a 20-30% spread
across sessions, is precision the underlying measurement cannot support. This does not mean the reconstruction
is wrong, but the "within 5%" framing implies a level of validation the single anchor point does not have.

**Evidence.** `results/2026-09-19/bmoe_gtier/bmoe_gtier_ab_20260919_0447/README.md` ("Observation, not a
controlled comparison"); `research/2026-09-19_STATE_OF_RESEARCH.md` §2 table and §4.3.

**What would fix it.** Either run the X3 off-charger-equilibrium row before quoting a 5% reconstruction accuracy,
or report the reconstruction's agreement against the *range* (5.3-6.76 tok/s) the same stack has produced, not
a single most-favorable sample.

---

### 8. [SEVERITY: MEDIUM] Missing bottleneck: prompt/context length and KV-cache growth are not in the token-budget reconstruction at all

`ESTIMAND.md` §5 explicitly requires context lengths 512 and 4096 to be reported separately, "because KV-cache
traffic grows with context and competes for DRAM." The entire §1 reconstruction (expert bytes, dense bytes,
attention small-ops budget) is stated as a single per-token constant, with no term for KV-cache read/write
bytes, no context-length dependence, and no acknowledgment that the DRAM bandwidth budget (`dram_app_gbps` =
59.7 GB/s, cited in §1.1) is shared between weight streaming and KV-cache access. At longer contexts the
attention small-ops term (currently folded into "~20 ms... more when capped") would grow, and DRAM contention
between the flash-fed expert-weight path and a growing KV cache is exactly the kind of "memory bandwidth
contention between I/O lanes and compute" the review brief asks about. None of E1-E5 varies context length; all
appear to be single fixed-length decode runs (E1's "256 tokens," E2's "256 tokens," etc.), so the entire budget
is only validated at one point on an axis the project's own estimand document says must be reported separately.

**Evidence.** `ESTIMAND.md` §5 ("Context lengths 512 and 4096 reported separately"); `research/2026-09-19_RESEARCH_SPEC.md`
§6 (all experiments specify a single token count, no context-length arm).

**What would fix it.** State the context length at which 6.76 tok/s and the E1-E5 floors were/will be measured,
and add at least one context-length arm (or an explicit limitation that the whole spec is a short-context
result) before generalizing to "the goal."

---

### 9. [SEVERITY: LOW-MEDIUM] Two of the "dead" mechanisms in §5's table are marked dead on evidence gathered under conditions the spec elsewhere says are unrepresentative

Row J (attention/KV on GPU) is marked dead via `why_levers_flip` §4 ("192 crossings/token cost more"), and row
K (NPU experts) via `msweep_htp_*` (0.39-0.68x CPU). Both of these matmul-sweep and crossing measurements were
taken outside the engine, in isolated benches (`matmul_sweep.json`'s own claim text: *"the sweep's CPU clock
caps were not logged"* for the GPU gate/up row, and *"the engine's GPU expert path and zero-copy upload were NOT
measured"*). CLAUDE.md §9.3 rule / the spec's own §9 rule 5 ("Transfer test before any proxy... used only after
one engine measurement agrees with it within 20%") is not obviously satisfied for these two "dead" verdicts —
unlike E1-E4, which are explicitly designed as in-engine transfer tests, J and K are retired from the plan on
microbenchmark evidence alone, the exact failure mode §2.3 spends a whole subsection warning about ("proxies
were used before their transfer was checked"). This is a smaller objection than 1-2 because the underlying
numbers (0.39-0.68x, 162x copy overhead) are large enough that a 20-30% transfer-test uncertainty likely
wouldn't flip the verdict — but the spec does not say that explicitly, and CLAUDE.md's own bar has not been
formally cleared for these two closures.

**Evidence.** `CLAIMS.md` rows `msweep_gpu_expert_gate_up_speedup`, `msweep_htp_expert_down_speedup` (both note
scope limitations in their own claim text); §5 table rows J, K.

**What would fix it.** Note explicitly, next to J and K, that they are closed on microbenchmark margin-of-error
grounds (large effect size) rather than an in-engine transfer test, so a reader does not mistake them for E1-E4-grade evidence.

---

### 10. [SEVERITY: LOW] E5 depends on a same-session run that has been queued and not executed for two days, and its "decisive" framing (ABBA ×6) does not itself validate the ~1.7x headline number if E1-E4 change the engine between now and then

E5 is "already pre-registered" (`device/bmoe_h2h.sh`) but has been sitting unrun since at least the 09-17/09-18
reproduction (`bmoe_reproduced_decode`, 3.98 tok/s uncontrolled comparison flagged in `STATE_OF_RESEARCH.md` §1).
If E1-E4 change the engine's configuration (pre-repacked kernels, a GPU split), E5 needs to run *again* against
BigMoeOnEdge with the *final* configuration, not the current one, or the "1.7x the published result" claim will
be quoting a stale engine build against the reference. The spec's phone-time budget ("~5 hours... order E1, E4,
E2, E3, E5") puts E5 last, which is correct, but does not say E5 must be re-run if any earlier experiment changes
the default build (it likely will, per H-A being "primary").

**Evidence.** §6 E5; §10 ordering; `research/2026-09-19_STATE_OF_RESEARCH.md` §1 ("same-session head-to-head...
was written and queued... and not run").

**What would fix it.** State explicitly that E5 runs against whatever configuration §7's tree lands on, not
against today's stack, and that a second E5 run is required if the landed configuration differs from what was
current when E5 was drafted.

---

## Citation spot-check (≥10 required; all checked against the artifact named)

| # | citation (as used in the spec) | verdict | note |
|---|---|---|---|
| 1 | `device_ceiling_qwen3_cpu` = 16.9 tok/s (§1.3) | **CONFIRMED** | `results/2026-09-18/device_bandwidth.json`: `implied_target_tok_s: 16.8538` (olmoe_cpu arm) |
| 2 | repack_bench generic/repacked rates: 20.0/33.5 (gate/up), 26.6/33.9 (down) (§1.1) | **CONFIRMED** | `results/2026-09-19/repack_bench/repack_bench_phone.out`, verbatim |
| 3 | "dense-weights=anon — 945 MiB" (§1.1) | **CONFIRMED but misapplied** | line exists verbatim in `results/2026-09-17/bmoe_verify/b1_rep1.err:1735`; disagrees by 21% with the GGUF's own dense-tensor byte count (see objection 3) |
| 4 | expert weights per token = 1.02 GB, "48 × 8 × 3 × (2048·768·18/32 B)" (§1.1) | **CONFIRMED** (arithmetic and artifact) | 2048×768×18/32 = 884,736 B; ×8×48×3 = 1,019,430,912 B ≈ 1.02 GB; matches `gguf_active_qwen_olmoe.json` `expert_bytes_per_token: 1023934464` (the small residual gap is the 6 Q4_1 down layers at 983,040 B/expert vs the spec's uniform 884,736 B assumption — a ~0.5% simplification, not material) |
| 5 | Adreno MEMPROBE 31-33 GB/s (§1.1) | **CONFIRMED (rounded)** | `gx_m6_012604/bench.out`: MEMPROBE range 30.72-33.18 GB/s across orderings; spec's "31-33" excludes the 30.72 SoA reading, a minor rounding-up |
| 6 | Adreno bit-exact gx kernel 7-11 GB/s (§1.1) | **MISQUOTED (selective range)** | at k=1-2 (the actual dispatch regime, ~2.1 experts/dispatch per `gtier_ab_summary` README) the cited files show 3.7-7.2 GB/s, well below "7-11"; see objection 2 |
| 7 | DRAM 4-thread app bandwidth = 59.7 GB/s (§1.1) | **CONFIRMED** | `CLAIMS.md` `dram_app_gbps` = 59.7410 |
| 8 | flash bytes/token at 88% hit = 120-127 MiB (§1.1) | **CONFIRMED** | `results/2026-09-19/gtier_ab_summary.json` rows: `read_MiB_per_token` 119.8-127.2 |
| 9 | measured token floor 148 ms / 6.76 tok/s (§1.2) | **CONFIRMED numerically, but see objection 7** | `bmoe_gtier_ab_20260919_0447/README.md`: base decode 6.755 tok/s, self-labeled "Observation, not a controlled comparison" |
| 10 | "compute scales with clock: 77 vs 121-124 ms" (§1.2) | **CONFIRMED** | `ceiling_ledger.json`: `trace_runs.plain.compute_residual_ms = 77.0`; `capped_terms.base/stack.compute_ms = 120.83 / 123.82` |
| 11 | `ledger_best_case_tok_s` ≈ 7.9 tok/s (§ referenced via ceiling handoff logic, cross-checked) | **CONFIRMED** | `ceiling_ledger.json`: `projection.best_case_tok_s_width_plus_gate_up = 7.8810`; `CLAIMS.md` agrees |
| 12 | GPU tier "~2.1 experts per dispatch" used implicitly to justify the E4 k=8 test point | **NOT SUPPORTED BY THE SPEC** (finding, not a quote) | the spec never mentions this number even though it directly undercuts E4's choice of k=8; see objection 2 |

12 of 12 numeric citations resolve to real values in the named artifact (no fabrication found). One (#3) combines
two artifacts that disagree without saying so. One (#6) quotes the favorable end of a k-dependent range as if
it were the whole range. One (#9) omits its own source's explicit "not controlled" caveat when re-quoted as a
5%-accurate anchor.

---

## Answering the review brief's specific questions

**Is 10 tok/s achievable? Independent estimate.** Using only measurements the spec itself treats as solid
(repack_bench's isolated 33.5/33.9 GB/s, the node ledger's per-op rates including the slow attention ops, the
capped-clock stall/mgmt floors from `capped_terms`), and *not* assuming the disputed additivity of §5: the
single best fully-measured wall-clock number in the whole corpus is 6.76 tok/s (148 ms), itself uncontrolled
(objection 7); the best *controlled, decisive, ABBA-scored* wall-clock number is the stack's 5.48 tok/s
(`stack_decode_gain`, capped_terms.stack). Every proposed improvement beyond that (A, B, C, E, F) has so far
either not been run end-to-end (A, B, F) or been run and found net-neutral in aggregate (E, and a related tier
variant of B in `bmoe_gtier_ab_20260919_0447`). My independent estimate, using only wall-clock-measured
deltas: **plausible range 7-9 tok/s if the pre-repacked kernels (A) transfer cleanly to the engine and do not
trigger a compensating rise elsewhere (as E did) — 10 tok/s requires that AND at least one of {B, C, F} to land
simultaneously with no offset, which no lever combination has done yet in this project's own history (§2.1).**
I would not bet on 10 tok/s at the capped clock; I would bet on 10 tok/s being reachable only via F (a clock
response that has not been shown to exist) or G (cooling, explicitly out of scope).

**Alternative explanations for 6.76 tok/s and the neutral results.** The reconstruction in §1.3 is one
consistent story, but the arena/gtier/kgsl pattern (mgmt or compute term falls, another rises by nearly the same
amount) is also exactly what you would see if the engine is close to a *single* shared resource ceiling (DRAM
bandwidth, or an OS-level memory-pressure response) rather than three separable ceilings that happen to trade off
1:1. §2.2 already says this ("compute, stall and mgmt are one wall time split by bookkeeping"), but the project
has not tried to identify what the single resource is (DRAM bandwidth reaches 59.7 GB/s max, but at 2.0 GB/token
+ up to 127 MiB/token flash, the DRAM-bound floor should be far below current wall times — so DRAM itself is
probably not saturated). A missing candidate: Linux's `MemAvailable` reclaim and major-fault behavior is already
implicated (`r1_arena_majflt`, `r1_base_majflt` — the base configuration itself runs with ~370 MiB swapped and
~21-23 major faults/token even *without* any lever under test). This is a strong, already-measured, and
under-discussed candidate mechanism: the phone may be memory-pressured at baseline, and every lever that touches
memory layout is really modulating *how much of the fixed swap-fault cost lands on which term*, not adding or
removing real work. The spec's own §2.1 rule ("a lever counts only if it deletes bytes, instructions or serial
dependencies") does not obviously apply to reducing baseline memory pressure (e.g., closing background apps,
increasing zram, or reducing the model's resident footprint) — that path is not in §5's mechanism list at all.

**Architecture hypotheses / decision tree (§8, §7).** Beyond objections 1, 2 and 6 above: H-B is explicitly
"conditional on E4," but as shown in objection 2, E4's own positive criterion is measured at an unrepresentative
k. If E4 reports "positive" under its current design, H-B still has to clear a *second*, unstated gate — does
the split's aggregate throughput hold at the realistic k≈1-2 distribution — before the "one wall-time A/B"
promised in §5.1's row for this path is actually decisive. That is a fourth branch the tree does not show,
meaning H-B could spawn its own sub-experiment rather than terminate.

**Would this repeat the cycle `STATE_OF_RESEARCH.md` documents?** Partially, and for a specific reason:
`STATE_OF_RESEARCH.md` §5's diagnosis is "effort went to levers whose ceiling was already known to be small,
and the biggest lever (kernels) sat untested for 30 hours." This spec correctly reprioritizes kernels first
(E1). But its two structural gaps — the additive arithmetic that revives an assumption the same document
disowns (objection 1), and a decisive threshold on E4 set at a non-representative k (objection 2) — are exactly
the kind of "locally reasonable substitute that nobody asked whether it was the quantity" that `CLAUDE.md` §0
describes as this project's recurring failure mode. If E1 comes back ambiguous (objection 6's gap) or E4 comes
back "positive" at k=8 but the real split doesn't help (objection 2), the natural next move is another
campaign, not a stop — which is the cycle repeating with better bookkeeping.

---

## Verdict

**(a) Collapse or expand?** As written, the spec would likely **collapse most of the tree in one pass on its
negative branches** (E1 killing at ≥95 ms is plausible given the dense-floor concerns in objections 3-4, and
would cleanly stop most of the optimistic architecture work) — which is the spec's main virtue: it is honestly
built to be falsifiable and cheap to falsify (~5 phone-hours). But if any of E1/E4 lands in its **positive**
branch, the tree **would expand**, because (1) the additive arithmetic that licenses "10 tok/s reachable" is
unsound per objection 1 and the real number would need re-deriving from controlled deltas, not summed ceilings,
and (2) E4's own positive result would immediately raise the unstated k-representativeness question (objection
2), which is itself a new experiment. The ambiguous-middle gap (objection 6) is also a nontrivial-probability
outcome that the tree currently has no leaf for at all. Net: **more likely to expand than the document claims,
specifically along its optimistic branches** — its pessimistic branches are sound and would genuinely stop the
work.

**(b) The three changes that matter most:**
1. **Delete or explicitly caveat the additive "A + E + C" arithmetic in §5** (objection 1). Replace it with
   controlled, simultaneous wall-clock deltas only, and state next to it that mechanism E (cache management) has
   already been measured net-neutral in isolation (the arena), so its ceiling cannot be freely added to A and C.
2. **Redesign E4 to test at the realistic dispatch size** (k≈1-3, weighted by the measured miss distribution),
   not k=8, before calling any result decisive for the GPU-split mechanism (objection 2).
3. **Add the missing middle branch to every decisive experiment's decision rule** (objection 6), and reconcile
   the two disagreeing dense-byte-per-token numbers before trusting the 80 ms / 95 ms compute-floor thresholds
   that a 7-20% error in that single input could move across (objections 3-4).
