# Research specification: why 10 tok/s has not been reached, and the decisive experiments that settle it

**Date:** 2026-09-19. **Status:** v2, after the hostile review (`2026-09-19_HOSTILE_REVIEW.md`; §12 lists each objection and what changed). **This is the single source of truth for the next phase.**
**Goal under test:** Qwen3-30B-A3B, weights unmodified, Q4_0 GGUF, single-stream greedy decode at ≥ 10 tok/s on the OnePlus 15R,
fidelity Tier E (`ESTIMAND.md` §3: bit-for-bit, **or within a floating-point reordering tolerance declared in advance**).
**Sources.** Every number is from `CLAIMS.md` (claim id in backticks), `results/2026-09-18/ceiling_ledger.json` (`[CL: …]`), or a
named results path. Arithmetic on those numbers is marked **[T]**. Anything else is marked **[H]** (hypothesis).

---

## 1. Independent reconstruction of the token budget

### 1.1 What one decode token must do (from the model file and the measured device)

| quantity | value | source |
|---|---|---|
| expert weights touched per token | 48 layers × 8 experts × 3 slices × (2048·768·18/32 B) = **1.02 GB** | model shape; `results/2026-09-18/gguf_expert_types.json` |
| dense weights touched per token | **0.78 GB** (active bytes 1.84 GB − expert bytes 1.02 GB). The engine's anon copy is 945 MiB, but that includes tables not touched per token (embeddings) | `results/2026-09-18/gguf_active_qwen_olmoe.json`; the 945 MiB log line |
| weight bytes per token, total | **1.84 GB** | `gguf_active_qwen_olmoe.json: active_bytes_per_token` |
| CPU kernel throughput, generic Q4_0, 4 threads | **31.0 GB/s** resident; 20.0 (gate/up) / 26.6 (down) GB/s on the expert shapes | `device_ceiling_qwen3_cpu`; `results/2026-09-19/repack_bench` |
| CPU kernel throughput, repacked i8mm | **33.5 / 33.9 GB/s** on the expert shapes | `results/2026-09-19/repack_bench` |
| Adreno, memory-only read of the slot layout | 31-33 GB/s | `gx_m6_012604` MEMPROBE |
| Adreno, bit-exact gx kernel | 3.7-4.4 GB/s at k=1, 7.2 at k=2, 8.6 at k=3, 10 at k=4 (compute-bound, strongly k-dependent) | `gx_m6_020545`, `gx_m6_082901` |
| DRAM, 4 threads, app process | 59.7 GB/s | `dram_app_gbps` |
| flash, 1 MB random reads, 8 lanes | ~2.8-3.2 GB/s | `byte_budget_measured.json: flash_gbps`; G1 |
| expert cache | 5000 MiB budget, **88% hit** | `stack_hit`; gtier A/B 0447 |
| flash bytes per token at 88% hit | **120-127 MiB** | `read_MiB_per_token` in every stacked run |
| CPU clock, sustained | policy0 1.9-2.27 GHz, policy6 1.65 GHz (hardware 3.32 / 3.80) in 96% of samples. The "full clock" reference run below was at 3.03 / 3.44, ~91% of hardware max | `[CL: clock_state]`; `results/2026-09-19/thermal_caps.json` |
| GPU dispatch size in the engine | **~2.1 experts per dispatch** (the misses of a layer at 88% hit) | `bmoe_gtier_ab_20260919_0447/README.md` |
| context length of every number here | 512-class prompts, ≤ 300 tokens of KV. `ESTIMAND.md` §5 requires 512 and 4096; **4096 is unmeasured** | ESTIMAND §5 |

### 1.2 The lower bound, term by term [T]

A decode token is a strict chain: attention(L) → router(L) → experts(L) → attention(L+1). The experts of layer L are unknown until
router(L) has run. So a miss at layer L is paid **serially** unless the miss was predicted a layer earlier.

| term | floor | derivation |
|---|---|---|
| expert matmuls | 1.02 GB / 33.5 GB/s = **30 ms** | repacked CPU kernel rate, if it holds at the capped clock |
| dense matmuls | 0.78 GB at 33 GB/s = **24 ms**; but the node ledger shows attention shapes at 9-28 GB/s generic (Kcur 9.0, Vcur 17.1, Qcur 28.3 `[CL: node_ledger.median_GB_s]`), i.e. partly overhead-bound, which repacking does not fix. Range **24-45 ms** [H] |
| attention, norms, router, small ops | **~20 ms** | node ledger: small ops 14 + flash-attn 4 + head 8 typical at full clock, more when capped `[CL: node_ledger]` |
| **compute floor** | **75-95 ms at capped clock** | sum; **~50-60 ms at the 3.0/3.4 GHz reference** (compute scales with clock: 77 vs 121-124 ms `[CL: trace_runs.plain, capped_terms]`). The range is the dense uncertainty; E1 measures it |
| stall floor | 0.12 × 1.02 GB / 3.2 GB/s = **38 ms** | miss bytes serial at flash rate. Measured stall: 35-48 ms. This matches numerically, but one measured fact contradicts the simple model: computing resident experts first (which should hide ~0.9 ms of hit compute under each ~0.8 ms miss read) had **no effect** (`order3_balanced_residentfirst`). So the per-layer miss cost is not a single read latency; E2 must show the per-layer stall histogram, not only the total |
| cache management | **22-24 ms** measured; floor unknown | `mgmt_ms` in every stacked run; the arena cut it to 9 ms but compute rose 12 and stall 4 (§4.3 of the state report) |
| **token floor at capped clock** | **135-155 ms → 6.5-7.4 tok/s** | today's best measured: 148 ms (6.76 tok/s), i.e. the engine is already at the floor of its architecture at this clock |
| token floor at the 3.0/3.4 GHz reference | 110-120 ms → 8.3-9.1 tok/s | |

**These floors are NOT additive across levers** (§2.1 and §3.1): the table says what each term cannot go below, not that savings in two terms add. The arena is the measured counterexample (mgmt −15 → wall 0).

**Reading.** At the capped clock, 10 tok/s (100 ms) is **below the sum of the floors** of the current architecture, even with the best
kernels measured. Compute can, at most, fall from 92 to ~80 ms. The other 60 ms are I/O terms whose floors are set by the miss rate,
the flash rate and the bookkeeping, not by the CPU. **No compute lever reaches 10 tok/s alone, and no I/O lever does either.** Only a
change in the *structure* of a term (its floor) does.

### 1.3 The gap: theory → expected → measured → end-to-end

| stage | number | where the loss went |
|---|---|---|
| device ceiling (weights at CPU kernel rate) | 16.9 tok/s | `device_ceiling_qwen3_cpu` |
| minus the clock cap (×0.62 on compute) | ~11 | `[CL: trace_runs.plain vs capped_terms]` |
| minus serial misses at 88% hit | ~8 | stall 38 ms |
| minus cache management | ~7.1 | mgmt 22 ms |
| minus kernel inefficiency (generic vs repacked, ×1.5 on gate/up) | ~6.5 | `repack_bench` |
| **measured** | **6.76** | gtier A/B 0447 |

The reconstruction accounts for the measured number to within 5% for the awake, unplugged condition; the same stack measured 5.3-6.3 on the charger and 5.48 on USB power, so the anchor carries a ±20% condition effect that E3 must control. There is **no hidden ceiling**; there are four visible ones, and the
project spent most of its effort on the smallest (I/O bookkeeping) and on a fifth that is not in the table (a GPU tier whose
ceiling is bounded by the expert term it could take, ~14 ms).

## 2. Diagnosis: what has actually been going wrong

### 2.1 The system is at a resource limit, so moving work does not remove it
Every lever after 09-17 that *relocated* work was net-neutral, and the ledger shows where it went:
- slot arena: mgmt −15 ms → compute +12, stall +4 (twice: `arena2_*`, `results/2026-09-19/arena3_summary.json`);
- kgsl pinning: faults on the cache → faults on the hot anon data, compute +78 ms;
- swap guard: faults −15% → its own checks +2.3 ms;
- GPU tier: expert compute −6 ms → CPU wait +7 ms (`bmoe_gtier_ab_20260919_0447`);
- run-time repack: compute −15 → stall +17.
The only decisive wins **removed** work: thread placement (removed contention), cache size and SLRU (removed flash bytes), prefetch with
selective adoption (removed stall), repacked kernels (remove instructions). **Rule that follows: a lever counts only if it deletes
bytes, instructions or serial dependencies. Anything that relocates them will be net-neutral on this phone, and should not be A/B'd.**

### 2.2 "Compute", "stall" and "mgmt" are one wall time split by bookkeeping, and we optimised the split
`compute` is a residual (wall − stall − mgmt; BigMoeOnEdge `docs/telemetry.md`, quoted in `hr_pinned_compute_ms`). An A/B on a single
term measures partly how the accounting moved. The arena results are the proof: mgmt fell decisively, compute rose decisively, wall
did not move. **Only wall time per token is a result. Terms are diagnostics.**

### 2.3 Proxies were used before their transfer was checked
- The simulator's 11.5 tok/s (`qwen3_30b_lru_tok_s`) omitted compute; with compute charged it became 6.3 (`qwen3_30b_serial_nonflash_tok_s`), which is where the engine is. The 10 tok/s expectation was never revised.
- The GPU pool test's map/unmap cost (1.4/2.7 ms) did not transfer to the engine (0.09 ms per expert); a mechanism was built on it and retracted.
- The k=3 latency gate assumed no overlap with CPU work; the engine waits 0.24 ms per dispatch, not 1 ms.
- The 6-thread negative on OLMoE did not transfer to Qwen3 (and was re-run to no resolution).

### 2.4 Campaigns were sized below their own resolution
Decode noise is ~5% per row; ABBA ×6 resolves ~4-6%. Levers with a known ceiling of 3-8 ms/token (≈2-5%) were still run as 24-row A/Bs
(swap guard, arena, expert order ×3, SLRU decode). Each cost 1-2 phone-hours to conclude "not resolved".

### 2.5 The largest measured gap was recorded twice and not pursued for ~30 hours
`overhead_ratio` (engine 1.9× slower than plain llama.cpp on the same arithmetic, 09-18) and `bmoe_i8mm_no_gain` ("generic kernels,
repack off", 09-17) both pointed at kernels. The ceiling handoff named the gate/up anomaly the largest untested lever and designed X1.
X1 was never run; the 2-minute microbench that finally tested it (09-19 08:44) found 0.60×.

### 2.6 A silent over-constraint: "lossless" was implemented as "bit-exact with ggml-cpu"
`ESTIMAND.md` §3 allows a declared reordering tolerance. The GPU kernel was built bit-exact (integer dot, exact fp16 decode, ggml's
accumulation order) and is compute-bound at 7-11 GB/s on a device that reads the same bytes at 31-33 GB/s. The repacked CPU kernels
(a reordering) were accepted under the tolerance clause. The two standards were applied inconsistently, and the strict one cost the
GPU path 3-4×.

### 2.7 The clock cap was treated as an external constant
Compute scales with the clock (77 → 124 ms). The cap tracks skin temperature (`thermal_caps.json`), and skin temperature is a function
of the phone's own power draw. Every spinning thread (ggml barriers, GPU spin-wait, I/O polling) and every wasted instruction
(`ovhmech_extra_instructions`: 1.29× plain llama.cpp's instructions) is heat that lowers the clock for the *next* seconds. Joules per
token were never measured. **This is the one coupling that could make efficiency gains compound instead of cancel, and it is unmeasured.**

### 2.8 Methodology, in one sentence
The pre-registration and ABBA discipline is sound and caught five false claims. What was missing is the step **before** a campaign:
a written floor for the term the lever attacks, and a refusal to run anything whose ceiling is below the design's resolution or
whose mechanism relocates rather than removes work.

## 3. Assumptions to stop making
1. That the three terms are independent and additive. They share one wall clock, one power budget and one memory.
2. That a microbenchmark transfers to the engine without a transfer test.
3. That "lossless" means bit-exact with ggml-cpu. It means Tier E as defined: a declared reordering tolerance, measured by teacher-forced perplexity.
4. That the clock cap is exogenous.
5. That a 24-row ABBA can resolve a 3% lever.
6. That the simulator's tok/s is an expectation. It is an upper bound that omits compute.
7. That the phone on the charger is the deployment condition. It is the worst thermal condition measured.

## 4. What the evidence proves, and what it does not

**Proves.** The measured 6.76 tok/s; the four ceilings of §1.3; that relocation levers are neutral; that repacked kernels are 1.3-1.7× faster on the expert shapes in isolation; that the bit-exact GPU tier is neutral; that unplugged runs doze; that BigMoeOnEdge reproduces at 3.98.

**Does not prove.** That the repacked rate holds at capped clocks inside the engine (never run end-to-end without the I/O-lane cost);
that the stall floor is the miss rate (never tested with an oracle predictor); that mgmt has a floor above ~0 (the arena moved it, but
the +46 MiB/token of extra flash traffic it caused is unexplained and may be the real cost); that the clock cap responds to our power
draw; that the GPU is memory-bound under a reordering-tolerant kernel; that we are faster than BigMoeOnEdge in a controlled comparison.

## 5. The mechanisms that could reach 10 tok/s, and the ones that cannot

Budget at capped clock: 148 ms measured → 100 ms needed. Each entry is a *floor change*, with the ceiling it could add [T] and its
Tier.

| mechanism | term | ceiling (ms/token) | Tier | status |
|---|---|---|---|---|
| A. pre-repacked experts + repacked dense (llama.cpp's own kernels for every weight) | compute 92 → ~80 (capped) | 12-19 | E (reordering) | built, laptop-verified; **decisive E1** |
| B. CPU + GPU concurrent expert split with a reordering-tolerant GPU kernel | expert compute 30 → ~17 at 60 GB/s aggregate | ~13 | E (declared tolerance) | needs a non-bit-exact kernel; **decisive E4** |
| C. oracle-grade next-layer prediction (hit 88 → 95%) | stall 38 → 16 | up to 22 | E | predictor exists at 88%; **decisive E2** |
| D. faster flash (queue depth, larger reads, 3.2 → 4+ GB/s) | stall 38 → 30 | up to 8 | E | never swept above 8 lanes × 1 MB |
| E. cache bookkeeping removed rather than moved | mgmt 22 → ~5 | up to 17 | E | the arena's +46 MiB/token must be explained first |
| F. energy-per-token → higher sustained clock | compute ×(cap/hw) | up to 40 | E | unmeasured; **decisive E3** |
| G. cooling (external) | compute 92 → ~55 | ~37 | E | rejected by the user as a deliverable; still the largest single lever measured |
| H. top-6 routing / expert pruning / 2-3-bit experts | all terms ×0.75 | ~35 | **A** | rejected by the user |
| I. self-speculation (Medusa/EAGLE heads) | ×(accepted/verify cost) | ≤ 0 with measured verify costs | A | dead: `verify_cost_n2` = 1.71× |
| J. attention/KV on GPU | dense compute | negative | E | dead on an in-engine measurement (`gpu_dense_cpu_experts` slower than CPU-only, `why_levers_flip` §4) |
| K. NPU experts | expert compute | negative | E | dead on a microbenchmark only (`msweep_htp_*`, 0.39-0.68× with caps unlogged); no in-engine transfer test. The effect size (≥ 1.5× slower) makes a reversal implausible, but this is the one closure that rests on a proxy |

**Arithmetic of the Tier-E route.** The ceilings in the table cannot be summed: they share one wall clock, one power budget and
one memory (§2.1, §3.1), and E's ceiling has already been measured at **~0 net** (the arena, twice). What can be said [T]:
- A alone, at its measured kernel rate: 148 → ~130-136 ms (7.4-7.7 tok/s), **if** the rate holds inside the engine at the capped clock (E1).
- A + C, where C is bounded by an oracle that has not been run: at best 148 − 15 − 20 = ~113 ms (8.8), only if the two do not interact (they touch different resources: CPU instructions vs flash queue; the interaction is unmeasured).
- 10 tok/s at the capped clock needs A + C **and** either a working GPU split (B, whose realistic-k rate is unmeasured, E4) or a clock response to lower power (F, E3). **No combination whose parts are all measured reaches 100 ms.** That is the honest state: 10 tok/s Tier E is *not excluded* and *not supported*; E1-E4 decide which.

### 5.1 Unconventional paths with the arithmetic that keeps or kills them [T unless marked]

| path | arithmetic | verdict |
|---|---|---|
| **Bigger cache.** MemAvailable before a run is ~7.0 GB; the budget is 5.0 GB. Push to 6.0-6.5 GB. | The hit curve is measured only to 5 GB (85.3 → 88.0% from 4 to 5 GB, `cache4000_tok_s`, `stack_hit`); extrapolated +1.5 GB → ~90-91% [H] → stall 38 → ~30 ms. The kgsl run showed what happens when free RAM drops below ~1.5 GB: reclaim moves to hot data. | worth one row in E2 (cache 6000 arm), not a campaign |
| **Bigger, fewer reads.** The three slices of one expert are three 0.88 MB reads. Reading the expert's 2.65 MB as one extent (a repacked-on-flash layout that stores gate/up/down contiguously) cuts requests 3×. | G1: bulk random at 1 MB × 8 lanes ≈ 2.8-3.2 GB/s; the interface ceiling is ~4 GB/s. If 2.65 MB extents reach 3.8 GB/s: stall 38 → 32. | +6 ms; bundle with the pre-repack conversion (the file is rewritten anyway) |
| **Expert-parallel threading.** Each thread computes whole experts (gate→SwiGLU→down) with one join per layer instead of 3+ barriers per op (`ceiling_handoff` §3.2). | Barrier count per token: 144 MUL_MAT_ID ops × ≥ 2 barriers ≈ 300+; at ~20-40 µs of straggler wait each on two clock domains → 6-12 ms [H]. Also lets cached experts start while misses land (hides part of the stall). | ≤ 12 ms; the instruction count from E1 says whether the overhead is there at all |
| **Drop the dense anon copy, mmap the dense weights, give the RAM to the cache.** | +0.95 GB cache → hit +~1.5 pt → stall −5; but `densemap_sensitivity_compute` measured mmap'd dense **+22.5 ms** compute on Qwen3 (faults on hot pages). | dead unless E3 shows the phone keeps them resident when awake and not charging |
| **Cut the dense term.** 0.95 GB/token of dense weights is 29 ms at 33 GB/s. Attention is 240 matmuls/token. The output head alone is 8 ms (`node_ledger`), 155k vocab × 2048. | The head is a lookup of the argmax under greedy decoding; computing only the top-k candidate rows is Tier A. Q4_0 → Q4_0 for the head is the model as shipped. Nothing lossless here except the kernel rate (A). | no Tier-E lever beyond A |
| **Pipelining across layers with a stale router** (compute experts of L+1 predicted from h(L), verify at L+1). | Every wrong prediction is a wasted 2.65 MB of compute and a serial recompute; at 88% prediction accuracy the expected cost exceeds the saving unless prediction is ≥ 97% [T]. E2's oracle bounds the accuracy that exists. | subsumed by E2 |
| **Two tokens per forward pass with a self-draft** (Medusa-style heads). | Tier A (extra trained heads), and the verify cost is measured: a 2-position verify is 1.71× a decode (`verify_cost_n2`), so the break-even acceptance is > 71%. | dead under Tier E; marginal under Tier A |
| **Run everything on the GPU at its unthrottled clock** (the CPU swings 2.1×, the GPU 2% across the thermal drift, `[CL: capped_floor]`). | Only if the GPU reaches memory rate on the FFN (E4) *and* attention on the GPU stops costing 192 crossings/token (a fully GPU-resident graph, i.e. llama.cpp's OpenCL backend with the expert cache in GPU-mapped memory). Ceiling: ~2 GB/token at ~30 GB/s = 67 ms + dispatch overhead of ~3000 nodes. | only if E4 is positive; then a 1-day port of the whole graph, measured as one wall-time A/B against H-B |
| **Energy shaping**: fewer threads for the memory-bound ops, no spin barriers, GPU idle-wait. | Bounded by E3; if the cap follows our draw, every joule saved raises the clock; nominal instruction savings ×1.6. | E3 decides |

The pattern: every Tier-E path is one of **{fewer bytes per token, fewer instructions per byte, fewer serial misses, more sustained
clock}**, and each of the four has exactly one decisive experiment (E2/D, E1, E2, E3). There is no fifth category.

**Other possibilities considered and rejected as inputs to this spec** (each in one line, with the reason):
- Two-sequence batching: doubles throughput per weight byte but not single-stream latency (the goal is single-stream).
- Compressing expert slices on flash (LZ4): trades flash bytes for CPU instructions; the CPU is the wall (§1.2).
- Expert deduplication/merging, distillation, 2-bit: Tier A or a modified model.
- Layer-skipping / early exit: Tier A.
- Running the router of layer L+1 on a stale hidden state ("pre-gating"): this is mechanism C; the oracle in E2 bounds it.
- Whole-model on the GPU (all experts + attention): the GPU is ~1× the CPU on the dominant op and the cache would have to be GPU-mapped;
  bounded by mechanism B's aggregate, which E4 measures.
- fp16 dense weights: more bytes; wrong direction.
- Larger cache via smaller KV / dropping the anon dense copy (mmap the dense weights instead): +0.95 GB of cache → hit ~90-91% [H];
  but `densemap_sensitivity_compute` measured mmap'd dense weights **+22.5 ms** compute on Qwen3. Dead unless E3 shows faults are cheap awake.

## 6. The decisive experiments

Rules for all of them: **wall ms/token is the result**; terms are diagnostics. Phone held Awake (`device/bmoe_gtier.sh` waker),
caps and skin temperature logged every second, unplugged unless stated. Each is designed so that either outcome closes branches.

### E1. The compute floor of this engine on Qwen3, isolated from all I/O
1. **Question.** With no misses, no cache management and no flash traffic, how many ms/token does the engine spend on arithmetic,
   generic vs repacked, at the capped clock and at full clock?
2. **Why it matters.** It is the only term the project has been able to move by removing work, and its floor decides whether 10
   tok/s is arithmetically possible at all at the capped clock (needs C_floor ≤ ~60 ms with I/O at its floors).
3. **Hypothesis.** H1: C_floor(repacked, capped) ≤ 80 ms and C_floor(repacked, full) ≤ 55 ms. H1-null: repacking gains < 10 ms in the engine.
4. **Construction.** A replay mode: the engine runs Qwen3 with a **fixed routing set** of ≤ 384 experts (one 8-set per layer, chosen from a
   real trace) so that after warm-up every token is a 100% hit. Text is meaningless; only the arithmetic is real. Arms: generic
   kernels; pre-repacked experts + repacked dense (`--experts-prerepacked --repack-dense`). Each arm at (a) the capped steady state
   (pre-heated 60 s), (b) the first 20 tokens of a cold phone (< 30 °C, hardware clocks: `cap_full_below_31C`). 256 tokens, 3 rows each, interleaved.
5. **Measure.** wall ms/token; caps per token; instructions and cycles per token (`simpleperf stat -e instructions:u,cycles:u`);
   stall and mgmt must read ~0 (that is the validity check of the replay).
6. **Not a proxy.** The `repack_bench` microbench (it has no attention, no small ops, no barriers). OLMoE (different shapes and overhead).
7. **Positive signal.** C_floor(repacked, capped) ≤ 80 ms and the repacked gain ≥ 12 ms.
8. **Kills.** C_floor(repacked, capped) ≥ 95 ms: at the capped clock 10 tok/s is impossible for this engine regardless of every I/O
   lever (since stall + mgmt floors ≥ 40 ms). Then the only routes are F (clock) or Tier A.
9. **Conclusions.** The number C_floor enters §1.2 and fixes the compute row for every later decision. The instructions/token count
   against plain llama.cpp's (`ovhmech_extra_instructions`) says how much engine overhead remains beyond kernels.
10. **Branches closed.** All compute micro-levers (thread count, expert order, fusion, GPU tier as a compute saver) are closed by the
    gap between C_floor and 60 ms: if the gap is ≤ 10 ms none of them is worth a campaign; if it is ≥ 30 ms the GPU split (E4) is the only one with that ceiling.

### E2. The stall floor: is prediction the limit, or bandwidth?
1. **Question.** With a *perfect* next-layer predictor, what is the stall? Equivalently: is the stall the miss rate or the flash?
2. **Why it matters.** Stall is 35-48 ms of the budget; mechanism C's ceiling (22 ms) is the largest Tier-E I/O lever, and it is only
   real if prediction quality, not flash bandwidth, sets the floor.
3. **Hypothesis.** H2: oracle prefetch cuts stall by ≥ 20 ms. H2-null: oracle stall ≥ 30 ms (bandwidth-bound; predictors are dead).
4. **Construction.** Greedy decode is deterministic. Run the stack once, record the routed expert ids per layer per token. Run again with
   the recorded ids fed to the prefetcher **as the prediction one layer ahead** (an oracle; CLAUDE.md §4.3), everything else unchanged.
   Add a third arm with the oracle **two** layers ahead. ABBA ×3 against today's predictor, 256 tokens, held Awake.
5. **Measure.** wall ms/token, stall ms, flash MiB/token, hit rate, per-layer stall histogram (the node trace tool).
6. **Not a proxy.** The simulator's hit rate; the predictor's offline accuracy.
7. **Positive.** Oracle-1 stall ≤ 18 ms and wall −15 ms or more: the predictor's quality is the lever and its ceiling is now measured.
8. **Kills.** Oracle-1 stall ≥ 30 ms: the flash rate is the floor; every predictor/prefetch/adoption/gating lever is closed for good,
   and only D (faster flash) or a larger cache changes the stall.
9. **Conclusions.** Fixes the stall row of §1.2 as a measured floor rather than a derived one.
10. **Branches closed.** Predictor work (if null); flash-lane work (if positive, since bandwidth was not the limit).

### E3. Is the clock cap a function of our own power draw?
1. **Question.** Does a configuration that executes fewer instructions per token (repacked kernels, no spin-waits) run at a higher
   sustained clock, and what are its joules per token?
2. **Why it matters.** Compute scales with the clock (×1.6 between full and capped). If the cap responds to our draw, every
   efficiency gain compounds and the compute floor moves; if it does not, cooling is the only clock lever and it is out of scope.
3. **Hypothesis.** H3: the repacked arm settles ≥ one cap step higher (policy0 median ≥ 2.19 GHz vs 1.90) over the last 10 of 15 min. H3-null: identical cap trajectories.
4. **Construction.** Two arms, 15 min sustained decode each, same prompt loop, unplugged at 60-80% battery, order reversed on the
   second day: (i) generic kernels, ggml spin barriers as shipped; (ii) pre-repacked + repacked dense, `--gpu-spin-wait 0`, I/O lanes
   blocking. Log every second: both caps, skin temperature, `current_now × voltage_now`. Same total tokens, so heat is comparable.
5. **Measure.** cap trajectories; J/token; tok/s over the last 10 min. Also the single-variable control: arm (i) at 2 threads (less
   power, less work) to separate "less heat" from "less instructions".
6. **Not a proxy.** Thermal status; a short run (< 5 min, before equilibrium); the charger.
7. **Positive.** ≥ one cap step and ≥ 10% tok/s in the last 10 min: energy per token is a first-class metric and every instruction saving is worth ~1.6× its nominal ms.
8. **Kills.** Identical caps within a step: the cap is skin-temperature-driven by total dissipation and does not care how the work
   is done. Then F is closed; only G (cooling) or fewer joules by fewer bytes (Tier A) moves the clock.
9. **Conclusions.** Decides whether the capped or the full-clock row of §1.2 is the planning row.
10. **Branches closed.** All "power-aware scheduling" ideas (if null); the cooling debate (if positive, since software then owns part of the clock).

### E4. Is the Adreno memory-bound under a reordering-tolerant kernel?
1. **Question.** With the bit-exact requirement replaced by a declared tolerance (fp32 accumulation, fp16 decode by the hardware),
   is the expert FFN dispatch, **at the sizes the engine actually dispatches (k = 1-3, ~2.1 on average)**, faster end-to-end (host-visible)
   than the repacked CPU computing the same experts (~0.08-0.1 ms per expert at 33 GB/s), and is its output within the tolerance?
2. **Why it matters.** Mechanism B's ceiling (~13 ms) exists only if the GPU adds bandwidth (memprobe says 31-33 GB/s is available;
   the bit-exact kernel uses 7-11). Together with the CPU at 33 GB/s the expert term could halve.
3. **Hypothesis.** H4: host-visible dispatch time at k=2 ≤ 0.25 ms and at k=1 ≤ 0.15 ms (the CPU's time for the same experts), with
   max |Δ| vs ggml within the declared tolerance (the repacked CPU kernel's own ~1e-6 relative, `repack_bench`) and teacher-forced PPL within 0.2%.
   The k=8 rate is reported but is NOT the criterion: the engine never dispatches k=8 at 88% hit.
4. **Construction.** In the existing gx harness (`host/gpu_ffn`), a variant that uses native fp16 loads, fp32 fma, and no Q8
   quantisation of x (x in fp32). Same bench (`gx_bench`, warm-up, interleaved), same test cases, tolerance gate instead of bit-exact
   gate, PPL gate in the engine on `defer_ppl_text.txt`. ~15 min of phone time.
5. **Measure.** device ms at k=1..8, host ms, GB/s, max abs/rel error on real cases, PPL.
6. **Not a proxy.** MEMPROBE (no arithmetic); k=1 timings (fixed cost dominates).
7. **Positive.** k=2 host-visible ≤ 0.25 ms within tolerance: build the split and measure it as ONE wall-time A/B (E4b, ~1 h); the
   miss-weighted expected saving is then ≤ 48 layers × (CPU time − dispatch time) ≈ ≤ 10 ms/token [T], which bounds what E4b can show.
8. **Kills.** k=2 host-visible ≥ 0.40 ms (today: 0.72): the fixed dispatch cost exceeds the CPU's whole cost for the experts it would take;
   the GPU branch is closed permanently, including every dispatch-overhead lever, since at k≈2 the fixed cost is the whole cost.
   **Middle (0.25-0.40 ms):** treat as kill; the ceiling (≤ 10 ms/token) is below the resolution of any A/B the project can run.
9. **Conclusions.** Fixes the expert-compute floor: 30 ms (CPU only) or ~17 ms (split).
10. **Branches closed.** gx fixed-cost work (fusion, doorbell, staging) if null; if positive, the bit-exact kernel line is retired.

### E5. The controlled head-to-head against the published engine (for the claim, not the goal)
Same session, same Q4_0 file, BigMoeOnEdge's documented command vs our best Tier-E configuration, ABBA ×6, held Awake, unplugged.
Already pre-registered in `device/bmoe_h2h.sh`. Positive: ratio decisive by the rule. It is the only experiment that makes
"1.7× the published result on the same phone" a claim.

**Total phone time for E1-E5: ~6 hours** (E4b adds ~1 h if E4 is positive). E5 runs LAST, against the final configuration the tree
selects, so the head-to-head compares the engine we would actually report. Every experiment reports at context 512 **and** one
4096-context row (ESTIMAND §5): a floor that only holds at short context is not a floor.

## 7. Decision tree (terminates; every leaf is a decision, not an experiment)

```
E1 → C_repacked_capped
 ├─ ≥ 95 ms ──► capped-clock 10 tok/s impossible for this engine.
 │              E3 decides: cap responds to draw → plan on the full-clock row; else → 10 tok/s needs Tier A or cooling. STOP Tier-E goal, write up.
 └─ ≤ 80 ms ──► E2 → oracle stall
                 ├─ ≥ 30 ms (bandwidth-bound) ──► stall floor fixed at ~30-38; then 10 tok/s needs C_floor + 30 + mgmt ≤ 100
                 │                               → mgmt must reach ≤ 10 (E-mechanism) AND C ≤ 60 → only with E4-positive split. Else STOP.
                 └─ ≤ 18 ms (prediction-bound) ──► build the best predictor (ceiling now measured); 10 tok/s reachable if
                                                  C_floor + 18 + mgmt ≤ 100 → with A alone (C~80) needs mgmt ≤ 2: no → needs E4 or F.
E4 → GPU rate
 ├─ < 15 GB/s ──► GPU closed forever.
 └─ ≥ 25 GB/s ──► expert floor 17 ms; build the split; one wall-time A/B.
E3 → cap response
 ├─ none ──► all planning at the capped row; software cannot change the clock.
 └─ ≥ 1 step ──► J/token becomes a gate; re-derive §1.2 at the observed clock.
```
**Middle-range rule for every experiment:** a result between the positive and the kill threshold is resolved by 3 more interleaved rows; if
it is still in the middle, it is a **kill** (the lever's ceiling is then below the resolution of any campaign this project can afford).
E1 middle (80-95 ms): plan on C = the measured value and require the other terms to close the gap in the tree below; no compute campaign.

Every path ends in one of three states within ~6 phone-hours: **(a) 10 tok/s Tier E is reachable and the architecture is named** (A +
predictor + split, with measured floors summing ≤ 100), **(b) reachable only at a clock we do not control**, or **(c) not reachable
without Tier A**, and the write-up is the negative-results/benchmark-validity paper (`2026-09-19_paper_framing.md`).

## 8. Architecture hypotheses to carry forward
- **H-A (primary):** llama.cpp's own repacked kernels for every weight, experts pre-repacked on flash, no run-time repack. Lossless under the declared tolerance.
- **H-B (conditional on E4):** experts split per layer between the CPU and a memory-bound Adreno kernel over one shared, GPU-mapped cache; the CPU takes its share first and never waits more than the device's fixed cost.
- **H-C (conditional on E2):** a next-layer predictor whose target is the oracle's stall, not a hit rate.
- **H-F (conditional on E3):** energy per token as a gate on every change; no spinning anywhere.
- Retired: bit-exact GPU kernels; the GPU tier as a cache; pinned expert caches; memory relocation levers; speculative decoding with a separate draft; NPU experts; attention on the GPU; 6 threads; expert order.

## 9. Method rules for the next phase (the fix, stated as gates)
1. **Floor first.** No campaign starts without a written floor (ms/token) for the term it attacks and the ceiling of the lever, both from artifacts. A lever with ceiling < 10 ms is not A/B'd; it is bundled into the next architecture step and measured as part of it.
2. **Wall time is the only result.** Term deltas are diagnostics and cannot be a verdict.
3. **Remove, don't move.** A lever must delete bytes, instructions or a serial dependency; relocation levers are rejected at design time.
4. **Oracle before predictor, ceiling before kernel** (CLAUDE.md §4.3, applied every time).
5. **Transfer test before any proxy.** A microbenchmark or simulator number is used only after one engine measurement agrees with it within 20%.
6. **Resolution check.** Expected effect ≥ 2× the design's SE, or the design is changed (more rows, or a replay mode with lower noise), or the lever is bundled.
7. **Energy is logged** (current × voltage, caps, skin) on every row from now on.
8. **One architecture A/B per phone-day**, not five tunings.

## 10. Final recommendation
Run E1-E5 (~5 phone-hours), in the order E1, E4, E2, E3, E5 (E1 and E4 are the cheapest and each can end the Tier-E goal on its own).
Then stop and write up according to the leaf reached. Do not build anything not named in §8 before the tree has been walked.

## 11. What this spec does not know
- Whether the +46 MiB/token of flash traffic under the arena is a bug in the engine's accounting or real reads; if real, mechanism E is smaller than its ceiling.
- The dense-matmul rate of the repacked kernels on Qwen3's attention shapes (assumed equal to the expert rate).
- Whether the replay mode of E1 perturbs the compute path (the ids are fixed but the graph is the same; a check is that stall and mgmt read ~0).
- The GPU's power draw; the split of E4/H-B could lower the clock (E3's logging covers it when the split is measured).

## 12. Hostile review (independent Sonnet reviewer, no project context; `2026-09-19_HOSTILE_REVIEW.md`)

| # | objection | accepted? | change in v2 |
|---|---|---|---|
| 1 | §5 summed ceilings across terms that share a resource, which §3.1 forbids; mechanism E was already measured net-neutral | **yes, the most important one** | the additive arithmetic is deleted; §1.2 states the floors are not additive; §5 now says "not excluded and not supported" and names the decisive tests |
| 2 | E4 tested the GPU at k=8; the engine dispatches at k≈2, where the bit-exact kernel is 3-4× slower per byte | **yes** | E4's criterion is now host-visible time at k=1-2 against the CPU's time for the same experts; k=8 is reported, not judged; the split's ceiling is bounded (≤ 10 ms) |
| 3 | dense bytes: 945 MiB (anon copy) vs 778 MiB (active per token); total 1.97 vs 1.84 GB | **yes** | §1.1 uses the active bytes (1.84 GB) and explains the 945 |
| 4 | the 29 ms dense floor assumed 33 GB/s on attention shapes that run at 9-28 GB/s generic (overhead-bound) | **yes** | the dense floor is a range (24-45 ms); E1 measures it |
| 5 | "full clock" was 3.03/3.44 GHz, ~91% of hardware max | **yes** | labelled as the reference clock throughout |
| 6 | no rule for middle-range outcomes | **yes** | the middle-range rule in §7 (3 more rows, else kill) |
| 7 | the 6.76 anchor has a ±20% condition effect the "within 5%" hid | **yes** | stated in §1.3; E3 controls it |
| 8 | no context-length term (ESTIMAND §5 requires 512 and 4096) | **yes** | every experiment reports a 4096-context row |
| 9 | J and K were closed on microbenchmarks | partly | J has an in-engine measurement; K is labelled as proxy-only, with the reason it is still closed |
| 10 | E5 must run against the final configuration | **yes** | E5 is last, stated |

The reviewer's own verdict on v1: the negative branches would stop work; the positive branches would have re-opened the cycle.
v2 removes the positive branch's arithmetic; the only positive claims left are conditional on E1-E4's measured outcomes.
