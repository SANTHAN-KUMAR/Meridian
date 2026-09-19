# State of the research, 2026-09-19: where moe-phone stands, what worked, what did not, and why

**Scope.** Everything measured from 2026-09-14 to 2026-09-19 on the goal *"Qwen3-30B-A3B, unmodified, ≥ 10 tok/s decode
on the OnePlus 15R"*. Every number below carries its source: a claim id from `CLAIMS.md` (checked by
`gates/claims_check.py`) or a results path. Numbers without a source are labelled as estimates.
**Status.** Phone testing stopped at the user's request after the anonymous-arena A/B (§4.3).

---

## 1. The answer first

- **Best measured decode, lossless:** 6.76 tok/s steady state. That is the stack (pinned threads, 5 GB SLRU cache,
  selective predictive prefetch) on the phone, awake and unplugged
  (`results/2026-09-19/bmoe_gtier/bmoe_gtier_ab_20260919_0447`, base arm, 12 rows).
- **Published baseline on the same phone:** BigMoeOnEdge's documented run reproduces at **3.98 tok/s**
  (`bmoe_reproduced_decode`). We are about **1.7x** faster, but that comparison is **not controlled** (different sessions and thermal
  and charging states). The same-session head-to-head was written and queued (`device/bmoe_h2h.sh`) and not run.
- **Goal gap:** 10 tok/s needs 100 ms/token against today's ~148. **Not reached.**
- **The largest untested lever is the one found last.** On the phone, ggml's repacked (i8mm) kernels run Qwen3's expert matmuls at
  0.60x (gate/up) and 0.79x (down) the time of the generic kernels our engine uses (`results/2026-09-19/repack_bench`).
  The version that removes its I/O cost (pre-repacked experts on flash) is built and laptop-verified. Its phone A/B was
  cancelled with the stop. It is the one experiment that could move the number materially (§6).

## 2. How the decode rate actually moved

| date | tok/s | what changed | source |
|---|---|---|---|
| 09-17 | 3.98 | BigMoeOnEdge reference build, documented command, our phone | `bmoe_reproduced_decode` |
| 09-17 | 3.71 | the same engine built for i8mm: no gain (generic kernels, repack off) | `bmoe_i8mm_no_gain` |
| 09-17 | 5.36 | compute pinned to cpu4-7, I/O lanes on cpu0-3 (the thread-cliff fix) | `pin2_pinned_best` |
| 09-17 | 6.20 | 5 GB expert cache | `cache5000_tok_s` |
| 09-18 | +4.3% | stack: SLRU + predictive prefetch with selective adoption (decisive, 5/6 repeats) | `stack_decode_gain` |
| 09-18 | -12.3 ms stall | I/O lanes on cpu0,1,6,7 (decisive); compute on cpu2-5 not resolved | `cores_3c_stall` |
| 09-19 | 6.76 | the stack, awake and unplugged (the conditions differ from the rows above) | gtier A/B 0447, base arm |

**+70% came in one day (09-17), from two cheap engineering changes: thread placement and cache size.** From 09-18 to 09-19, one
decisive +4.3% and one decisive stall cut were added. Everything else was null or negative (§4).

## 3. What is established (positive, decisive or measured)

| finding | evidence |
|---|---|
| the 4-thread "cliff" is thread placement, not memory: pinned threads recover it | `cliff_fix_pinned_t4` (4.0 -> 30.1 tok/s on OLMoE) |
| pinning plus separate I/O cores is the largest single engine gain on Qwen3 | `pin2_pinned_best` vs `pin2_unpinned` |
| SLRU reads 11.1% fewer flash bytes, exactly as the simulator predicted | `slru_phone_read_mib` |
| the stacked I/O levers are worth +4.3% decode, lossless (24/24 identical texts) | `stack_decode_gain`, `stack_lossless` |
| the CPU clock cap tracks skin temperature (mechanism unresolved: thermal engine or load clamp) | `results/2026-09-19/thermal_caps.json` (gx session) |
| compute, not flash I/O, is the wall at capped clocks | `research/2026-09-18_ceiling_handoff.md` §1.2 |
| **the engine runs the same arithmetic 1.9x slower than plain llama.cpp on cached OLMoE** | `overhead_ratio` |
| **ggml's repacked kernels are 1.3-1.7x faster than the generic ones on Qwen3 expert shapes, on this phone** | `results/2026-09-19/repack_bench/README.md` |
| unplugged, the phone autosuspends during runs unless held awake: rows become bimodal (a benchmark-validity hazard) | `results/2026-09-19/bmoe_gtier/bmoe_gtier_ab_20260919_0235/README.md` |
| the bit-exact GPU expert kernel works: identical text, 0 failures, 0 risk recomputes in the engine | gtier A/Bs 0447 and 0235 |

## 4. What was tested and did not help (the negative results, each with its mechanism)

| lever | result | mechanism | evidence |
|---|---|---|---|
| speculative decoding (Qwen3-0.6B draft) | best case 1.015x | a verify of 2 positions costs 1.71x a decode; 71% agreement | `draft_best_speedup`, `verify_cost_n2` |
| confidence-gated prefetch | none meets its rule | the gates are precise but keep ≤ 32% of the stall saving | `specgate_verdict_none` |
| Hexagon NPU for expert matmuls | 0.39-0.68x the CPU | latency-bound at batch-1 decode | `msweep_htp_expert_down_speedup` |
| Adreno GPU, generic OpenCL path | 1.09x, upload a copy | copy costs 162x the compute | `msweep_gpu_expert_gate_up_speedup` |
| GPU expert tier (bit-exact gx kernel), awake | **net neutral**: 6.70 vs 6.76 tok/s | the CPU waits on the device about as long as the device saves; fixed cost ~0.6 ms per dispatch vs ~0.2 ms needed | `bmoe_gtier_ab_20260919_0447/README.md`; `host/gpu_ffn/NOTE.md` §9 |
| 6 compute threads | not resolved (-8 ms, SE 6.8) | barrier pacing across clock domains | `cores_6thread_compute` |
| swap guard | no net gain | its own checks cost what the fault saving gains | commit 421afe6 |
| kgsl-pinned expert cache | stopped: compute +78 ms | pinning the cold cache pushes reclaim onto hot anonymous data | `bmoe_kgsl_ab_20260919_0131/README.md`, `pinprobe/README.md` |
| run-time expert repack on the I/O lanes | compute -15, stall +17 ms (smoke, one row) | repack cost delays readiness | `bmoe_repack_smoke_20260919_0844` |
| anonymous slot arena, awake (4600 vs 5000 MiB) | **net neutral**: 5.46 vs 5.48 tok/s | mgmt -15 ms, compute +12, stall +4, flash +46 MiB/token | §4.3 |

### 4.3 Anonymous slot arena, held awake (the last run)
Pre-registered (`device/bmoe_arena3.sh`), scored with `stack_summary.py --any-budget-prereg --require-awake` (24/24 rows
kept, text identical in 24/24; `results/2026-09-19/arena3_summary.json`, rows in `results/2026-09-19/bmoe_arena3/bmoe_arena3_ab_20260919_0851`):

| term | base (5000 MiB) | arena (4600 MiB) | diff | verdict |
|---|---|---|---|---|
| decode tok/s | 5.48 | 5.46 | -0.02 | not resolved (~0) |
| cache mgmt ms | 24.3 | 9.0 | **-15.3** | decisive, 6/6 |
| compute ms | 123.8 | 135.9 | **+12.2** | decisive, 5/6 |
| stall ms | 35.1 | 38.7 | **+3.6** | decisive, 6/6 |
| flash MiB/token | 120.7 | 166.7 | **+46.0** | decisive, unexplained: the hit rate ROSE (88.0 -> 89.8) |

**Net-neutral.** The arena removes cache management, as in R1, and the saving reappears in compute and stall. The +46 MiB/token
of flash traffic with a higher hit rate is an unexplained engine behaviour, recorded here and not investigated.
This run was on USB power (the phone was plugged into the laptop): its base arm decoded 5.48 tok/s, against 6.76 for the same stack
unplugged this morning. That is consistent with a charging penalty, but it is a cross-session comparison, not a controlled one.

## 5. Is the gut feeling right? "Too many tests, too little progress, wrong direction or wrong method"

### Evidence that it is RIGHT
1. **The progress curve flattened after day one.** +70% on 09-17 from two cheap changes. Two further days added one
   decisive +4.3% and one decisive stall cut (§2). Most campaigns after 09-17 ended null, negative or invalid.
2. **Effort went to levers whose ceilings were already known to be small.** The ceiling handoff (09-18, §2) had already shown
   that I/O levers could not reach 10 tok/s and that the GPU tier's ceiling was ~14 ms/token. The night of 09-18 to 09-19 was still
   spent on the GPU tier (3 A/Bs, 7 kernel benches) and on memory pinning.
3. **The biggest measured gap was not followed up.**
   - `overhead_ratio` (09-18) showed our engine running the same arithmetic 1.9x slower than plain llama.cpp.
   - `bmoe_i8mm_no_gain` (09-17) had recorded that our kernels are "generic with repacking off".
   - The ceiling handoff flagged the gate/up per-byte anomaly as "the largest specific, lossless, untested compute lever" and
     designed X1 to test it.
   - X1 was never run. The repack microbench that finally tested it (09-19 08:44) took ~2 minutes and found gate/up 0.60x.
   Two days of GPU and memory work went around the biggest lever in plain sight.
4. **Much of the testing budget was spent on validity rather than results.** Invalid or confounded runs, each costing phone hours:
   - a CANNOT-LINK smoke;
   - a flag that swallowed its argument;
   - the Doze confound, which voided two overnight A/Bs;
   - kgsl carryover;
   - the adb loss;
   - an orphan driver (09-18);
   - a hex-float JSON.
   The pre-registration discipline caught all of them, but it did not prevent them.
5. **Earlier optimism came from simulation.** The simulator's 11.5 tok/s for Qwen3-30B (`qwen3_30b_lru_tok_s`) left out compute
   cost. When measured compute was charged, it fell to 6.3 (`qwen3_30b_serial_nonflash_tok_s`), which is close to where the
   engine is now. The 10 tok/s hope was not revised down when that measurement arrived.

### Evidence that it is WRONG, or only partly right
1. **The engine is already the fastest lossless Qwen3-30B-A3B we know of on this phone class:** 6.76 against the published
   3.98 on the same phone. That is ~1.7x, uncontrolled, but large. Day-one engineering was driven by measurement (the thread cliff
   was diagnosed, not guessed).
2. **The tests prevented false claims.** Several would otherwise have shipped:
   - resident-first +6% (`order2_residentfirst_decode`, a rotation artifact);
   - "half of compute is overhead";
   - SLRU's decode gain;
   - the kgsl arena;
   - an overflow-map mechanism.
   Each was caught and retracted. Under this repository's rules (CLAUDE.md §0), that is the job.
3. **10 tok/s is not physically ruled out.** The CPU's measured weight throughput puts the device ceiling at 16.85 tok/s
   (`device_ceiling_qwen3_cpu`). The gap is kernel efficiency, and the repack result shows that efficiency is recoverable.
4. **The negative results are publishable knowledge:** speculation, the NPU, the GPU tier, pinning and the Doze confound, each with a
   mechanism (`research/2026-09-19_paper_framing.md`, framing 3).

### Verdict
The methodology (pre-registration, ABBA, decisive rules) is sound and caught real errors. **The prioritisation was wrong.**
After day one, the ranking "which lever has the largest measured headroom" was written down (ceiling handoff, overhead
ratio) and then not followed. Easier-to-start or more interesting work came first: GPU kernels and memory tricks. The fix
is a rule, not more discipline: **before any campaign, rank every open lever by its measured ceiling in ms/token, and run
the largest one first.** Applied on 09-18, that rule would have tested the repacked kernels about 30 hours earlier.

## 6. The remaining route, ranked by measured ceiling (ms/token of the ~148 awake budget)

| lever | ceiling | basis | state |
|---|---|---|---|
| pre-repacked experts (i8mm kernels, no run-time repack) | ~19 | repack_bench 0.60/0.79x on ~55 ms of expert matmuls | built, laptop-verified (identical to runtime repack; ppl -0.046%), phone conversion + A/B cancelled |
| dense weights repacked at load | ~6-7 | same kernels on ~25 ms of dense matmuls (estimate) | built (`--repack-dense`), laptop-verified |
| engine overhead vs plain llama.cpp beyond kernels | up to ~13 on OLMoE | `ovhmech_dense_copy_ms`, `ovhmech_extra_instructions` | not attacked |
| GPU tier with a lower fixed cost | ≤ 14 | gx NOTE §9 | kernel v4 is 3x too slow per dispatch |
| cache management (arena) | ~0 net | mgmt -15 ms is repaid in compute/stall (§4.3) | measured, closed |
If the first two land near their ceilings, the estimate is ~148 -> ~122 ms (~8.2 tok/s). 10 tok/s would further need the overhead
and GPU items, or the clock headroom (unresolved cap mechanism). That is a conjunction, not a plan with a known outcome.

## 7. If the work continues, the smallest honest next steps
1. The pre-repacked A/B (`host/chain_prerepack.sh`, ~1.5 h: conversion, fidelity gate, smoke, ABBA x6). It is the one
   measurement that decides whether the compute wall moves.
2. The same-session head-to-head against BigMoeOnEdge (`device/bmoe_h2h.sh`, ~1 h). It is the one measurement that makes
   the "~1.7x the published result" claim defensible.
3. Stop. Write up with `research/2026-09-19_paper_framing.md` (negative-results catalogue plus benchmark validity,
   and the speed result if steps 1-2 are positive).

## 8. Where everything is
- claims: `CLAIMS.md` (241 checked); design: `research/2026-09-18_gpu_expert_path_design.md`; ceiling analysis:
  `research/2026-09-18_ceiling_handoff.md`; novelty search: `research/2026-09-19_novelty_search.md` (topics
  "not found", none "verified absent"); framing: `research/2026-09-19_paper_framing.md`.
- tonight: `results/2026-09-19/NIGHT_SUMMARY.md`; GPU kernel: `host/gpu_ffn/NOTE.md`; engine source: the verified cumulative
  snapshot `tools/patches/0019-SNAPSHOT-bigmoeonedge-engine-tree-vs-74ba18f.patch` (+ NOTE).
