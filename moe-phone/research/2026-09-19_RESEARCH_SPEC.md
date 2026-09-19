# Research specification: why 10 tok/s has not been reached, and the decisive experiments that settle it

**Date:** 2026-09-19. **Status:** v1, before the hostile review (§12 records the review and what it changed).
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
| dense weights touched per token | **~0.95 GB** (the anon copy the engine makes) | `bmoe: dense-weights=anon — 945 MiB` in every run log |
| weight bytes per token, total | **~2.0 GB** | [T] |
| CPU kernel throughput, generic Q4_0, 4 threads | **31.0 GB/s** resident; 20.0 (gate/up) / 26.6 (down) GB/s on the expert shapes | `device_ceiling_qwen3_cpu`; `results/2026-09-19/repack_bench` |
| CPU kernel throughput, repacked i8mm | **33.5 / 33.9 GB/s** on the expert shapes | `results/2026-09-19/repack_bench` |
| Adreno, memory-only read of the slot layout | 31-33 GB/s | `gx_m6_012604` MEMPROBE |
| Adreno, bit-exact gx kernel | 7-11 GB/s (compute-bound) | `gx_m6_020545`, `gx_m6_082901` |
| DRAM, 4 threads, app process | 59.7 GB/s | `dram_app_gbps` |
| flash, 1 MB random reads, 8 lanes | ~2.8-3.2 GB/s | `byte_budget_measured.json: flash_gbps`; G1 |
| expert cache | 5000 MiB budget, **88% hit** | `stack_hit`; gtier A/B 0447 |
| flash bytes per token at 88% hit | **120-127 MiB** | `read_MiB_per_token` in every stacked run |
| CPU clock, sustained | policy0 1.9-2.27 GHz, policy6 1.65 GHz (hardware 3.32 / 3.80) in 96% of samples | `[CL: clock_state]`; `results/2026-09-19/thermal_caps.json` |

### 1.2 The lower bound, term by term [T]

A decode token is a strict chain: attention(L) → router(L) → experts(L) → attention(L+1). The experts of layer L are unknown until
router(L) has run. So a miss at layer L is paid **serially** unless the miss was predicted a layer earlier.

| term | floor | derivation |
|---|---|---|
| expert matmuls | 1.02 GB / 33.5 GB/s = **30 ms** | repacked CPU kernel rate, if it holds at the capped clock |
| dense matmuls | 0.95 GB / 33 GB/s = **29 ms** | same kernel class; not measured on these shapes [H] |
| attention, norms, router, small ops | **~20 ms** | node ledger: small ops 14 + flash-attn 4 + head 8 typical at full clock, more when capped `[CL: node_ledger]` |
| **compute floor** | **~80 ms at capped clock** | sum; **~50 ms at full clock** (compute scales with clock: 77 vs 121-124 ms `[CL: trace_runs.plain, capped_terms]`) |
| stall floor | 0.12 × 1.02 GB / 3.2 GB/s = **38 ms** | miss bytes serial at flash rate. Measured stall: 35-48 ms. **The stall is the miss rate times the flash rate, and nothing else** |
| cache management | **22-24 ms** measured; floor unknown | `mgmt_ms` in every stacked run; the arena cut it to 9 ms but compute rose 12 and stall 4 (§4.3 of the state report) |
| **token floor at capped clock** | **80 + 38 + 22 = 140 ms → 7.1 tok/s** | today's best measured: 148 ms (6.76 tok/s) |
| token floor at full clock | 50 + 38 + 22 = 110 ms → 9.1 tok/s | |
| token floor at full clock, 95% hit, mgmt 10 | 50 + 16 + 10 = 76 ms → 13 tok/s | every term at its floor simultaneously |

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

The reconstruction accounts for the measured number to within 5%. There is **no hidden ceiling**; there are four visible ones, and the
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
| J. attention/KV on GPU | dense compute | negative | E | dead: 192 crossings/token cost more (`why_levers_flip` §4) |
| K. NPU experts | expert compute | negative | E | dead: 0.39-0.68× CPU (`msweep_htp_*`) |

**Arithmetic of the Tier-E route [T].** A + E + C at their ceilings: 148 − 19 − 17 − 22 = 90 ms → 11 tok/s. **A + C alone:** 107 ms → 9.3.
**A alone:** 129-136 ms → 7.4-7.8. So 10 tok/s at the capped clock requires **three** of {A, B, C, E, F} to land near their ceilings, and
each has a decisive test below. That is not a plan with a known outcome; it is a conjunction with four measurable branches.

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
   does the expert FFN kernel reach ≥ 25 GB/s on the Qwen3 shapes at k = 8, and is its output within the tolerance?
2. **Why it matters.** Mechanism B's ceiling (~13 ms) exists only if the GPU adds bandwidth (memprobe says 31-33 GB/s is available;
   the bit-exact kernel uses 7-11). Together with the CPU at 33 GB/s the expert term could halve.
3. **Hypothesis.** H4: device time at k=8 ≤ 0.85 ms (≥ 25 GB/s) and max |Δ| vs ggml ≤ the tolerance that the repacked CPU kernel
   itself shows (~1e-6 relative on the outputs, `repack_bench`) with teacher-forced PPL within 0.2%.
4. **Construction.** In the existing gx harness (`host/gpu_ffn`), a variant that uses native fp16 loads, fp32 fma, and no Q8
   quantisation of x (x in fp32). Same bench (`gx_bench`, warm-up, interleaved), same test cases, tolerance gate instead of bit-exact
   gate, PPL gate in the engine on `defer_ppl_text.txt`. ~15 min of phone time.
5. **Measure.** device ms at k=1..8, host ms, GB/s, max abs/rel error on real cases, PPL.
6. **Not a proxy.** MEMPROBE (no arithmetic); k=1 timings (fixed cost dominates).
7. **Positive.** ≥ 25 GB/s at k=8 within tolerance: build the split (CPU + GPU concurrently over the shared cache; the zcbench
   aggregate 1.535× is the model) and measure it as one wall-time A/B.
8. **Kills.** < 15 GB/s: the Adreno cannot beat the repacked CPU per byte; the GPU branch is closed permanently, including every
   dispatch-overhead lever, because the device term itself would not pay.
9. **Conclusions.** Fixes the expert-compute floor: 30 ms (CPU only) or ~17 ms (split).
10. **Branches closed.** gx fixed-cost work (fusion, doorbell, staging) if null; if positive, the bit-exact kernel line is retired.

### E5. The controlled head-to-head against the published engine (for the claim, not the goal)
Same session, same Q4_0 file, BigMoeOnEdge's documented command vs our best Tier-E configuration, ABBA ×6, held Awake, unplugged.
Already pre-registered in `device/bmoe_h2h.sh`. Positive: ratio decisive by the rule. It is the only experiment that makes
"1.7× the published result on the same phone" a claim.

**Total phone time for E1-E5: ~5 hours.** Nothing else runs until they are done.

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
Every path ends in one of three states within ~5 phone-hours: **(a) 10 tok/s Tier E is reachable and the architecture is named** (A +
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

## 12. Hostile review
(Filled after the independent review: the objections, which were accepted, and what changed. v1 has none.)
