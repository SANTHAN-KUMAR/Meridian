# Paper framing if the 10 tok/s goal is not reached (2026-09-19)

**Context this document assumes as fact, each cited to its artifact:**
- BigMoeOnEdge's own documented Qwen3-30B-A3B run reproduces on this phone at 3.98 tok/s median decode
  (`bmoe_reproduced_decode`, `CLAIMS.md`), at 69.5% expert-cache hit rate (`bmoe_reproduced_hit`).
- This project's stack (SLRU + predictive prefetch with selective adoption) reached 6.76 tok/s steady
  state, awake, unplugged, in the base arm of the GPU-tier A/B (`results/2026-09-19/NIGHT_SUMMARY.md`,
  `bmoe_gtier/bmoe_gtier_ab_20260919_0447/README.md`), but **this is not yet a controlled head-to-head
  against BigMoeOnEdge's own build** — the two numbers come from different campaigns, different nights,
  different thermal/charging states, and the document that reports 6.76 explicitly flags "that
  comparison is not controlled" (`NIGHT_SUMMARY.md` line 27; the pre-registered stack A/B that matches
  conditions between the two engines, `stack_summary.json`, compares our own base vs stack, not
  BigMoeOnEdge's reference build vs ours, under matched conditions).
- The 10 tok/s ceiling analysis (`research/2026-09-18_ceiling_handoff.md`) finds the wall is capped-clock
  CPU arithmetic (124 ms/token compute at capped clocks alone exceeds the 100 ms budget for 10 tok/s;
  `[CL: capped_terms.stack]`), not I/O — every I/O lever (eviction, prefetch, lanes, GPU tier, kgsl
  pinning) tops out worth a few ms to slightly negative, while the two live compute levers (repacked
  kernels, wide 6-thread scheduling) project a best case of ~7.9 tok/s (`ceiling_ledger.json`,
  `projection.best_case_tok_s_width_plus_gate_up`).
- The repacked-kernel lever is mid-measurement: isolated kernel bench shows 0.596x/0.786x the generic
  kernel's time on gate/up and down respectively (`results/2026-09-19/repack_bench/README.md`), and the
  pre-repack-on-flash path is verified lossless within a stated tolerance on the laptop
  (`results/2026-09-19/repack_laptop/README.md`), but the in-engine A/B on the phone that decides whether
  this becomes a decode-rate gain is queued, not run (commit `0591919`: "chain_prerepack queued (convert,
  fidelity gate, smoke, ABBA x6)").
- The GPU-expert-tier lever is closed net-neutral at current kernel speed: decode 6.70 vs 6.76 tok/s
  (not resolved), compute +6.9 ms/token (decisive, worse), stall -5.3 ms/token (decisive, better)
  (`bmoe_gtier_ab_20260919_0447/README.md`).
- The kgsl-pinning lever, applied to the cold expert cache, was stopped after 4 rows with no verdict
  because it moves reclaim pressure onto the engine's hot pages and ABBA cannot cancel the carryover
  (`bmoe_kgsl_ab_20260919_0131/README.md`).
- Novelty findings behind these framings are in `research/2026-09-19_novelty_search.md` (this
  session) and `POSITION.md` §9 (earlier sessions); no framing below claims a "first" that either
  document marks "found" or leaves as "not found" without flagging it.

---

## Framing 1 — "Where the time goes: a measurement paper on flash-streamed MoE on a capped phone"

**Title:** *Where the Milliseconds Go: A Term-by-Term Ledger for Flash-Streamed Mixture-of-Experts
Inference on a Thermally Capped Phone*

**Abstract (one paragraph).** Running an unmodified 30B-parameter Mixture-of-Experts model beyond a
phone's DRAM by streaming experts from flash is usually reported as a single decode-rate number. We
decompose that number into its constituent costs — I/O stall, cache management, and CPU compute — for
Qwen3-30B-A3B (Q4_0) on a OnePlus 15R, and show that on this device the wall is not I/O, contrary to
what a systems intuition (and prior work's emphasis on read granularity and locality) would predict:
capped-clock CPU arithmetic alone consumes more than the entire 100 ms/token budget implied by a
10 tok/s target, while every I/O-side lever we tested (segmented LRU eviction, predictive prefetch,
core placement, a GPU expert-compute tier, GPU-driver-pinned unswappable memory) is worth at most a few
milliseconds, and several — a naively-placed pinned cache, an under-strength GPU kernel — are net
negative once their second-order costs (reclaim pressure displaced onto hot pages, per-dispatch fixed
cost exceeding the compute it saves) are measured. We report the full ledger, the diagnostic method
(pre-registered ABBA A/Bs judged on a lever's own physical term rather than on decode rate), and the two
compute-side levers (kernel repacking, wide low-clock scheduling) that remain open.

**Contributions (each tied to an artifact, or labelled "still needed"):**
1. A measured, per-term cost ledger (compute / stall / cache-management) for a 30B-A3B MoE decoding on
   a real phone under its actual thermal cap — `ceiling_ledger.json`, `research/2026-09-18_ceiling_handoff.md`.
2. A negative-results catalogue for six I/O-side and placement-side levers, each closed by a measured
   number rather than by omission: NPU expert matmul (0.39-0.68x CPU, `msweep_htp_*`), generic-path
   Adreno GPU expert matmul (1.09x CPU with incorrect down-projection results on the fallback path,
   `msweep_gpu_*`), speculative decoding (best case 1.015x, `draft_best_speedup`), a slot arena
   (compute +14 ms/token from induced major faults, `arena2_compute_delta_ms`, `r1_arena_majflt`), a
   bit-exact concurrent GPU expert-compute tier (net-neutral, `gtier_cap8_awake_summary.json`), and
   kgsl-pinned cold cache (reclaim pressure displaced onto hot pages, stopped with no verdict,
   `bmoe_kgsl_ab_20260919_0131/README.md`).
3. Identification of the specific dominant anomaly inside compute: expert gate/up matmuls run ~1.5x
   slower per byte than the down projection at the same quant type on the same cores
   (`ledger_up_vs_down_same_type_ratio`), with three candidate mechanisms narrowed by the repacked-kernel
   result (`repack_bench_phone.out`: repacking removes the anomaly, 33.5 vs 33.9 GB/s).
4. **Still needed:** the in-engine decode-rate effect of the repacked kernel (queued, chain_prerepack)
   and the wide 6-thread compute lever at capped clocks (X1 in `ceiling_handoff.md`, not yet run per the
   files read for this document) — without these, contribution 3 is a kernel-level finding, not yet a
   decode-rate one.
5. **Still needed:** a controlled, matched-condition head-to-head against BigMoeOnEdge's own reference
   build (not just our own before/after), since the two headline numbers cited above (3.98 vs 6.76 tok/s)
   come from different nights and thermal states.

**Baselines beaten, and whether controlled:** beats naive stock llama.cpp `mmap` streaming (uncontrolled
in the sense that it is a different codebase, but same device, same model — `app_engine.json`
`upstream_inapp_qwen3_best`, 2.36 tok/s at best on GPU, 0.12 on CPU) and beats BigMoeOnEdge's documented
run **on this phone, on our own reproduction of their run** (`bmoe_reproduced_decode`), but that
comparison against our stack's 6.76 tok/s is explicitly **not yet controlled** for thermal/charging
state (`NIGHT_SUMMARY.md`). The paper must either run the matched-condition A/B before submission or
report the comparison as suggestive, not decisive, and say so in the same sentence as the number
(CLAUDE.md §7.6 forbids hedging in a footnote while the body asserts the number plainly).

**Threats a hostile reviewer would raise:**
- *"Your 6.76 vs their 3.98 isn't a controlled comparison — you might just be running warmer/cooler."*
  Answer: run the matched-condition ABBA (same night, same charge state, same `--require-awake` rule)
  before citing the ratio as a result; until then, report both numbers with their conditions stated
  separately, not as a ratio.
- *"Single-device, single-model result — is any of this general?"* Answer: the term decomposition method
  (ABBA on physical terms, not decode rate) and the specific finding that compute, not I/O, is the wall
  at capped clocks are the generalizable claims; the tok/s numbers are one device's data point. Say this
  explicitly rather than let the reader over-generalize from n=1 device.
- *"The gate/up anomaly is 'unexplained' in your own document — that's a hole, not a finding."* Answer:
  report it as a finding about magnitude and locus (which op, how much) with the mechanism narrowed to
  candidates, not resolved; that is honest and still useful — but do not claim the mechanism until X1
  (the kernel bench) or the repack A/B closes it.
- *"Every lever you tried failed — why should a reader believe the next one won't too?"* Answer: that is
  the paper's actual finding (§4.4 of `CLAUDE.md`'s framing: outcome-model error / arithmetic dominates,
  confounding-style I/O tricks don't), not a weakness to paper over.

**Minimal remaining experiments, ranked by value per phone-hour:**
1. X1 expert-kernel bench at capped clocks (~1 h) — decides whether width and gate/up restructuring are
   real before any more engine code; already queued and specifically designed to be cheap.
2. In-engine repack A/B (chain_prerepack, already queued) — the single biggest already-measured
   kernel-level lever (33-41% per-op speedup) still unconfirmed at decode-rate level.
3. Matched-condition BigMoeOnEdge-vs-stack A/B (X3-adjacent; needs an off-charger, awake, same-night run
   of both engines) — without this, contribution 5 above is not defensible and Framing 1's strongest
   comparative claim is unusable.

**Honesty check:** this framing's central number (10 tok/s not reached) is itself the finding, and the
document says so already (`ceiling_handoff.md` §0.8: "lossless 10 tok/s ... is unlikely; lossless
7.5-8 is realistic"). This framing does not need the goal to be hit; it is weakest exactly where the
project is weakest right now — the uncontrolled head-to-head — and that must be fixed or explicitly
scoped out, not glossed over.

---

## Framing 2 — "Benchmark validity: what makes an unplugged phone LLM benchmark trustworthy"

**Title:** *Awake, Off-Charger, and Cool Enough to Trust: Confounds in On-Device LLM Benchmarking, and
a Pre-Registered Protocol That Survives Them*

**Abstract.** On-device LLM benchmarks on phones are routinely reported as single numbers, but we find
at least three device-state confounds that silently bimodalize or bias decode-rate measurements on a
real Android phone: (1) the SoC entering Doze/autosuspend during an unplugged run when `svc power
stayon` fails to hold, producing bimodal per-row decode rates within one campaign; (2) a thermal governor
that caps CPU clocks as a function of *skin* temperature, not the framework's reported thermal status,
so two rows at the same reported "thermal status" can be running at markedly different clocks; and (3) a
charging-vs-unplugged difference in the SoC's steady-state clock cap large enough to change which
regime a reported decode rate describes. We give a pre-registered protocol (per-row wake enforcement,
clock and skin-temperature logging every 2 seconds, an ABBA design judged on the lever's own physical
term rather than on decode rate, and a documented keep/discard rule fixed before the first row) that
detects and excludes the confounded rows rather than averaging over them, and we show, on our own prior
campaigns, how much each confound would have silently biased a naively-averaged number.

**Contributions:**
1. The Doze/autosuspend confound, found and fixed in this project: an uncontrolled run went bimodal
   (`bmoe_gtier_ab_20260919_0235/README.md`, "not resolved: SoC autosuspend made rows bimodal in both
   arms"); the fix (per-row waker, `--require-awake` keep rule) and a negative control (the old Dozing
   run keeps 0/24 rows under the new rule) are both artifact-backed (`NIGHT_SUMMARY.md`).
2. The skin-temperature-tracking clock cap, measured directly: 100% of samples below 31 C run at full
   clock, 97.3% of samples from 33 C up are capped, and 84.7% of samples at the framework's own
   thermal-status 0 are *already* capped (`cap_full_below_31C`, `cap_throttled_from_33C`, `CLAIMS.md`) —
   i.e., the OS-reported thermal status is not a reliable proxy for the clock the CPU is actually running
   at, which is a benchmark-validity finding independent of anything to do with MoE or flash.
3. The charging-vs-unplugged difference, observed but **explicitly flagged as uncontrolled** in this
   project's own documents: base-arm decode was 5.3-6.3 tok/s on the charger one night and 6.76 tok/s
   unplugged another night (`NIGHT_SUMMARY.md`: "That comparison is not controlled; X3 (off-charger
   equilibrium) deserves its own run"). **Still needed:** X3 itself — the controlled, same-day,
   reversed-order AC-vs-unplugged run that would turn this from an anecdote into a result.
4. A general pre-registration and physical-term-judging methodology (ABBA, decisions on ms/token of a
   named physical quantity rather than on decode rate, decision rules fixed before the first row) applied
   consistently across ~15 A/B campaigns in this project (`why_levers_flip` §1; every campaign cited in
   this document's "already decided" table, `ceiling_handoff.md` §1.4).
5. **Still needed:** an explicit comparison against the general thermal-throttling literature (found:
   `arXiv 2603.23640` reports 30-minute sustained throttling on a Snapdragon 8 Gen 3) to establish which
   parts of this project's finding are device-specific corroboration and which (the Doze/autosuspend
   confound specifically) are, per this session's novelty search, not found elsewhere and may be the
   paper's actual new contribution.

**Baselines/comparisons:** this framing is not a speed comparison paper, so §8's three-baseline rule
does not directly apply to a headline number; what it must instead defend is that the *confound* is real
and *material* — i.e., show the size of the bias a naive protocol would have produced (a table: naive
mean including Dozed rows vs the pre-registered protocol's result) for at least the Doze case, where the
before/after numbers already exist (`bmoe_gtier_ab_20260919_0235` bimodal vs `..._0447` clean).

**Threats a hostile reviewer would raise:**
- *"This is n=1 device, n=1 engine — how do you know Doze/autosuspend generally biases LLM benchmarks,
  not just yours?"* Answer: be explicit that the claim is a demonstrated mechanism and a protocol, not a
  population estimate; invite replication; do not claim generality the single device does not support.
- *"Isn't 'phones throttle' already well known?"* Answer: yes for thermal (cite `arXiv 2603.23640` and
  similar), and the paper must foreground that this part is corroboration, not novelty. The Doze/
  autosuspend-specific confound is the part the novelty search did not find elsewhere
  (`research/2026-09-19_novelty_search.md` §c) — the paper's novelty claim must be scoped to exactly that
  piece, dated, and re-searched before submission (the search here was general-WebSearch only, not a
  targeted crawl of Android systems venues).
- *"You fixed it in your own harness — is the fix itself validated, or just 'looks fine now'?"* Answer:
  the negative control is real (old Dozing run: 0/24 rows pass the new rule) — cite it, and note that a
  positive control (a run known to be confound-free by an independent measurement, e.g. power-rail data)
  would strengthen it further; register that as a limitation if not available.

**Minimal remaining experiments, ranked by value per phone-hour:**
1. X3 (off-charger equilibrium, ~40 min, already designed in `ceiling_handoff.md` §3.3/§4) — turns
   contribution 3 from an anecdote into a controlled result; cheapest fix to this framing's biggest hole.
2. A naive-vs-protocol bias table, computable from **already-collected** data (no new phone time) —
   quantify how far off a naively-averaged number would have been for the Doze campaign; pure analysis.
3. A targeted literature re-search specifically for "Doze" / "App Standby" / SoC autosuspend in the
   Android systems literature (no phone time; a research-agent task) — needed before the novelty claim
   in contribution 5 can be asserted rather than merely "not found by a general search".

**Honesty check:** this framing is defensible even at 6-7 tok/s, because its claim is about measurement
validity, not peak speed — but it is the weakest of the three if the "not found" verdicts in the novelty
search turn out, on a targeted re-search, to be "found" (Doze/background-execution effects on
benchmarking are a known general concern in Android systems research even if not tied to LLMs
specifically). Do the targeted search before committing to this as the headline framing.

---

## Framing 3 — "A negative-results catalogue: why plausible levers for phone MoE inference don't pay"

**Title:** *Six Plausible Levers That Don't Pay: A Negative-Results Study of Flash-Streamed
Mixture-of-Experts Inference on a Phone*

**Abstract.** A large space of engineering levers looks, on paper, like it should speed up flash-
streamed MoE decoding on a phone: heterogeneous compute (NPU or GPU for expert matmuls), speculative
decoding, smarter eviction and prefetch, wider thread pools, and GPU-driver memory pinning to dodge
Android's swap. We measured all six on an unmodified Qwen3-30B-A3B (Q4_0) running on a OnePlus 15R and
found each one delivers at most a few milliseconds per token, and two are net negative once their
second-order costs are accounted for: a memory-pinning technique that avoids zram entirely for the
pinned region (measured directly: 0 vs 196,608 major faults after a forced reclaim) nonetheless makes
engine decode *slower* in-context, because pinning the wrong region (the cold expert cache) displaces
the system's reclaim pressure onto the engine's own hot pages; and a concurrent CPU/GPU expert-compute
tier, verified bit-exact against the CPU kernel over more than 16,000 dispatched experts, is
net-neutral because its own fixed per-dispatch cost (~0.5 ms) is comparable to the CPU time it saves
(~0.2-0.25 ms). We report the measured cost of each lever, the causal mechanism identified for the two
negative results, and the diagnostic protocol (judge on a lever's own physical term, not on end-to-end
decode rate) that made the mechanisms visible rather than just the net verdict.

**Contributions:**
1. NPU expert matmul: 0.39-0.68x the CPU's speed on Qwen3's expert shapes even with weights already
   resident (`msweep_htp_expert_down_speedup`, `msweep_htp_*`, `CLAIMS.md`) — closed.
2. Adreno GPU expert matmul, generic OpenCL path: 1.09x CPU on gate/up but produces **incorrect** results
   on the down-projection (max relative error 16.8, `msweep_gpu_expert_down_incorrect`) — closed, and a
   caution that a naive GPU-offload paper claim here would be reporting on a broken kernel.
3. A bit-exact concurrent GPU expert-compute tier (this project's own `gx` component): verified
   byte-identical to the CPU across 16,449+ dispatched experts and multiple negative controls
   (`gtier_m4_text_identical`, `gx_down_diff_bits` et al.), but net-neutral on decode
   (`gtier_cap8_awake_summary.json`: compute +6.9 ms decisive-worse, stall -5.3 ms decisive-better,
   decode not resolved) — the mechanism (fixed per-dispatch cost ~0.5 ms vs ~0.2-0.25 ms CPU-time saved
   per expert) is identified and quantified, not just observed.
4. kgsl-backed pinned memory: mechanism verified (immune to `MADV_PAGEOUT` reclaim, reads at malloc
   speed, `pinprobe/README.md`) but the naive application (pinning the cold cache) is a **regression**
   with an identified cause (reclaim pressure displaced onto hot anonymous pages, `bmoe_kgsl_ab_20260919_0131/README.md`)
   — closed as "wrong target", with the right target (`--dense-weights ahwb`, the hot set) named and
   explicitly flagged as never measured.
5. Speculative decoding: closed at best case 1.015x (`draft_best_speedup`), because each verified
   position routes to its own experts, so a streamed MoE pays near-full I/O cost per speculated position
   regardless of acceptance.
6. **Still needed if this framing is chosen:** the sixth lever (eviction/prefetch) is already closed
   (`slru_decode_clean`: -0.3%, not resolved at n=3, despite an 11.1% byte reduction) but could use one
   more repeat to either resolve the null cleanly or confirm it stays null — cheap, and tidies the
   catalogue rather than adding new content.

**Baselines beaten, and whether controlled:** this framing does not need to beat a baseline in the
usual sense — its claim is that six specific alternatives *don't* beat the project's own existing stack,
and every one of those six comparisons is a controlled, pre-registered, same-device ABBA (per
`why_levers_flip` §1's method, applied consistently, as in Framing 1/2). This is the strongest framing
for CLAUDE.md §9.1's own standard ("what would have to break for this to fail?") because each result is
a measured physical term, not an assertion of the hypothesis.

**Threats a hostile reviewer would raise:**
- *"A catalogue of negative results is a weak paper — did you just fail to engineer these well?"*
  Answer: for at least two levers (GPU tier, kgsl pinning) the paper identifies the *specific
  quantitative reason* (fixed dispatch cost vs. saved compute; reclaim pressure displaced to hot pages)
  rather than reporting "it didn't work" — that mechanistic identification, verified by a targeted
  measurement, is the actual contribution, not the negative sign itself.
- *"Six negatives on one device don't tell me anything about the next device."* Answer: true, and the
  paper should state which findings are device-specific magnitudes (e.g. the exact ms figures) versus
  which are structural (e.g. "a per-dispatch fixed-cost heterogeneous-compute tier only pays when its
  fixed cost is smaller than the compute-time of the smallest reasonable batch of work it's given" is a
  general design principle, illustrated by, not limited to, this device).
- *"Why should I believe the GPU tier is truly bit-exact and not just 'close enough that you didn't
  notice'?"* Answer: cite the specific negative controls (`gx_quant_mismatch_blocks`,
  `gx_swiglu_mismatches`, `gx_div_mismatches` — millions of crafted near-tie cases with 0 mismatches) —
  this is the strongest-evidenced claim in the whole project and should be foregrounded, not buried.
- *"Is 'net-neutral' actually just 'not enough repeats to see a real effect'?"* Answer: report the
  SE/|diff| for every claimed "not resolved" (already done in the source artifacts, e.g. decode
  SE/|diff|=1.04 in `gtier_cap8_awake_summary.json`) so the reader can judge power, not just sign.

**Minimal remaining experiments, ranked by value per phone-hour:**
1. None are strictly required to publish this framing — it is the most "already done" of the three. The
   highest-value optional addition is the in-engine repack A/B (queued) as a **seventh, currently-open**
   lever, which would let the paper end on an open question rather than purely closed ones.
2. One more SLRU-decode repeat to firmly resolve or firmly null the eviction-policy result, for
   completeness of the catalogue.
3. A `--dense-weights ahwb` (hot-set pinning) run, since the kgsl finding currently ends on "here is the
   right target, untested" — running it, even if negative, completes the catalogue's own logic.

**Honesty check:** this is the framing that best survives the goal not being reached, because it does
not depend on a headline decode-rate number at all — its unit of contribution is "lever X costs/saves Y
ms, here is why." The main risk is reviewer fatigue with "yet another negative-results paper"; the
mechanistic identification (not just the sign) for the GPU tier and kgsl pinning is what should be led
with to distinguish it from a bare list of failed attempts.

---

## Recommendation

**Framing 3** is the safest and most immediately publishable at 7-8 tok/s: every contribution is already
measured, mechanistically explained (not just signed), and internally consistent with `CLAIMS.md`. It
needs essentially no further phone time.

**Framing 1** is the most ambitious and the one with a real headline claim ("compute, not I/O, is the
wall at capped clocks on this SoC"), but it is currently blocked on one uncontrolled comparison
(BigMoeOnEdge vs stack, matched conditions) and two queued-but-unrun experiments (repack A/B, X1 width
bench); do not submit it with the BigMoeOnEdge ratio quoted as a result until that comparison is
controlled.

**Framing 2** has the most interesting single new claim (the Doze/autosuspend confound), but its novelty
is currently only "not found by a general search" — it needs the targeted re-search named in
`2026-09-19_novelty_search.md` before that claim can be made with confidence, and it needs X3
(off-charger equilibrium) to close its own internal "not controlled" flag on the charging-state finding.

A combined submission is plausible: Framing 3's catalogue plus Framing 2's protocol section (the
protocol is what made the catalogue's negative results trustworthy in the first place) is a coherent
single systems paper; Framing 1's compute-is-the-wall claim is better held for a second paper once the
repack/width experiments and the controlled BigMoeOnEdge comparison land, since it is the one making a
comparative speed claim rather than reporting internally-consistent measurements.
