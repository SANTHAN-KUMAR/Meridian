# moe-phone — claim audit and performance headroom (2026-09-17)

**Written for:** the moe-phone co-authors and the hostile reviewer they expect. Read
[`CLAUDE.md`](../CLAUDE.md) first; this document follows its levels (L1–L6) and its rules.
Every number below is either (a) a claim id from [`CLAIMS.md`](CLAIMS.md), (b) a field of the
artifact `results/2026-09-17/headroom_sims.json` produced by `gates/headroom_sims.py` (the new
claims with prefix `hr_` in `gates/claims_check.py` pin them), or (c) a quotation from an
external source with its location. Nothing is typed from memory.

Analysis only: no phone run and no laptop model run was made for this document. The offline
simulations re-use the project's own simulators (`cache_sim.py`, `prefetch_sim.py`) unchanged,
on the committed traces, at the phone's actual operating point.

---

## 0. Verdict in six lines

1. **Internal consistency holds.** 102/102 claims matched their artifacts at the start of this
   audit (139/139 with the `hr_` claims added below), the 48-test gate suite passes, and `G-REPRO` reproduces every offline artifact bit for bit. That is L5/L6 hygiene
   and it is genuinely good; it says nothing about whether the right quantity was measured.
2. **Two quantities are mislabelled (L4).** The engine's "compute" term is a *residual* (wall −
   stall − management), and BigMoeOnEdge's own telemetry contract says it "silently absorbs page
   faults and scheduler stalls" and frequency caps. Every "compute is the largest term" sentence
   in `README.md`/`NEXT.md` inherits that. And every 2026-09-17 phone run was taken with the CPU
   capped at 1.4–2.2 GHz (hardware 3.32/3.80), so the throttled residual is the number the plan
   for "10 tok/s" is built on.
3. **The load-bearing offline result does not transfer to the phone's regime (L3).** "Four
   tokens of lookahead reaches Belady" is true at rho ≤ 1 (a cache smaller than one token's
   working set) and false at the rho ≈ 4 the phone runs at: at rho 4 a 4-token horizon closes
   under half of the LRU-to-Belady gap and 16–32 tokens are needed. The same operating-point
   error runs the other way for within-token prefetch: judged "closed, negative" at a 10% cache
   with 30 ms of compute, it is **+24–35%** in the project's own simulator at rho 2.4–4 with the
   phone's 60–100 ms of compute.
4. **One prose claim is false against its own artifact (L6).** `POSITION.md` §3c says a 4-token
   horizon "reaches Belady at every cache fraction up to 30%"; the artifact says 70% of the gap
   at 30%. Not in the claims map, so the check could not catch it.
5. **The S9 story has an on-device counter-example.** The engine's measured Qwen3-30B-A3B hit
   rate at its own rho lands on the equal-rho curve (H_rho) and far from the floor-additive rule
   (H_floor) that the documents now call "the better estimate".
6. **Where the headroom is.** Per token at steady state: ~100 ms residual (throttled), ~54 ms
   I/O stall of which ~20–30 ms is a fixed per-layer head-of-line latency, 13–35 ms cache
   management. The levers the project has not tried, ranked by evidence, are in §4; the
   experiments that would settle them, with pre-registered predictions, are in §5.

---

## 1. Claim-by-claim audit

Method: for each claims-map group I read the producing script, the raw logs it parses, and the
device script that produced the logs, and asked the §10.2 questions (does it survive rescaling,
zero effect, an oracle, a constant, a second seed, `SE/|θ|`). Verdicts: **holds** = the number
is what the sentence says and the method is sound; **holds with caveat** = the number is right
but the sentence claims more than the design supports; **defect** = the sentence and the
evidence disagree.

### 1.1 Offline cache simulation (G2, S12, scope, policy, lookahead, predictor) — holds

`cache_sim.py` is correct for what it simulates: per-layer quotas that preserve the total
byte budget, event-atomic residency, Belady solved per layer, controls that recover the closed
forms (`test_lru_on_locality_free_routing_returns_the_cache_fraction`,
`test_cyclic_scan_lru_zero_belady_closed_form`). The decomposition into floor + skew + recency
is arithmetically what the sentences say. S12 (no cheap cross-token predictor) is sound and the
horizon explanation is supported by the matched-accuracy comparison.

**Caveat that changes the conclusion (L3, see §3.1):** every lookahead, predictor and prefetch
verdict was evaluated at cache fractions 5–20% of OLMoE, i.e. rho 0.4–1.6. The phone-relevant
models sit at rho 2.9–4.1 (`engine_target.json`: Qwen3-30B-A3B 3.91, gpt-oss-20b 2.86; the
BigMoeOnEdge run: 4.12). In the rho < 1 regime the cache cannot hold one token's top-k, so the
eviction decision is dominated by the very next event and a short horizon is trivially enough.
That is a property of the regime, not of MoE routing.

### 1.2 Engine simulator and engine target (spec_*, qwen3_*, s11_*) — holds with caveats

- `engine_sim.py` prices bytes per verified token and credits per accepted token with the
  Leviathan formula; the i.i.d.-acceptance assumption is stated (S10). Sound.
- `engine_target.py` reads the curve, refuses extrapolated rows, and reports fully resident
  models as flash-free. Sound. Its per-model figures go through `at_rho()`, which S9 refuted;
  the documents say so.
- `s11_nonflash` (23.2 GB/s effective on granite) is a measurement of *one* tiny model at 4
  threads; its linear-in-bytes transfer was tested and failed (`s11_matched_t2_ratio` 1.33×).
  The documents say so. It should not be called "compute-bound" without the profile that
  would show what bounds it (§4.1).

### 1.3 Device instruments (G0, G1, S1, G-RAM-SWEEP) — holds

`ufsbench.c`/`g1_analyze.py` exclude descheduled rows by a deadline rule fixed from the loop
structure, not from results; the 4 KB cell's 39% spread is reported (`g1_4k_cell_spread`). The
bulk constant 2.806 GB/s (`bulk_rand_best`, 1 MB × 8 threads) is 13% below the same cell in
the queue-depth run (`g1_storage_qd.json`: 3.23 GB/s, 22% spread); the roofline uses the lower
one and says so. `dramprobe` excludes rows with descheduling; 59.7 GB/s at 4 threads holds.

### 1.4 Thread cliff and pinning (cliff_*) — holds with caveat

`cliff_fix_pinned_t4` (30.1) and `cliff_fix_no_primes_needed` (28.1) read llama-bench's own
generation rate over 320 tokens; sound. `cliff_unpinned_t4` = 4.0 tok/s is the median of two
runs that read 5.9 and 2.0 (`decode_15r_pin.json`, `by_threads`): the sentence should carry
that spread. The marginal (differenced) rates in the same artifact are flagged
`marginal_suspect` for every pinned cell (negative flash bytes), which is why the claims
correctly use `tg` and not the marginal figure.

### 1.5 BigMoeOnEdge reproduction and levers (bmoe_*, levers_*, repack_*) — holds with defects in the prose

- `bmoe_reproduced_decode` 3.98 (n=2) and the 69.5% hit: what the artifact says.
- **Defect (L4): `bmoe_compute_share` "decode spends 0.145 s/token in compute, the largest
  single term".** `session.cpp:173` computes `compute_ms = wall_ms − stall_ms − mgmt_ms` and
  clamps at 0. BigMoeOnEdge `docs/telemetry.md` (read at the commit the patches target):
  *"`compute_ms` is a **residual, not a measured quantity** … a catch-all that silently absorbs
  page faults and scheduler stalls, not just matmul"* and *"low CPU occupancy points at a
  throttled/preempted core (a frequency cap …) rather than heavy math"*. Same for
  `pin2_pinned_compute` and `repack_compute`. The numbers are right; the noun is wrong, and the
  10 tok/s plan in `NEXT.md` treats the residual as matmul time. `--compute-trace` exists for
  the direct measurement and has never been run.
- **`bmoe_i8mm_no_gain` / `repack_*` / the tracker's "repack lever DEAD":** overstated. In the
  pinned retest (`bmoe_repack_pin.json`, peer commit `fa0f620`) the repacked cell's *compute
  residual* is 0.095–0.097 s vs 0.097–0.103 s (slightly lower), its I/O stall is 0.069 vs
  0.054 s, and — decisive — it decoded **different text** (80.1% hit, 172 MiB/token, 62
  re-reads/token vs 78.0%, 194, 70): the repacked kernel's accumulation order changed the greedy
  output, so the two cells are not the same workload. Its stderr also carries ~229 `repack:`
  log lines per token (58,537 lines per run vs 1,740), a contaminant in the read path. What
  the data support: *in-read-path repacking* loses; the kernel itself does not, and offline
  pre-repacked experts (§4.2) were never tested.
- **`bmoe_pinning_starves_io`:** the 0.23 tok/s cell pinned the *whole process* (`taskset f0`,
  I/O lanes included) — a correct and important negative, correctly worded.

### 1.6 Rotated confirmation (pin2_*) — holds; two reporting caveats

- Pinned 5.36 vs unpinned 4.75, n=4 rotated: the IQRs do not overlap (`ESTIMAND.md` §7), and
  the routing/bytes are identical across cells (deterministic greedy output; 193.78 MiB/token
  in every run), so the comparison is same-work. Holds.
- **Steady state (L6):** `ESTIMAND.md` §1 defines the reported rate over tokens 65+. The claims
  quote the 256-token mean. Over tokens 65–256 the same runs give 5.30 (pinned) and 4.72
  (unpinned) (`hr_pinned_steady_tok_s`, `hr_unpinned_steady_tok_s`); the compute residual
  drifts up by ~7% from the first 32 tokens to the last 32 inside every run
  (`hr_compute_drift_pinned`). Small, but the sustained-30-minute figure `ESTIMAND.md` also
  requires has never been measured and will be lower again.
- **Thermal state (L6):** every one of the 20 runs was at thermal status 2–3 with
  `scaling_max_freq` 1.40–1.75 GHz on cores 0–5 and 1.25–1.65 GHz on the primes
  (`bmoe_pin2/log.txt`; hardware 3.32/3.80). Across all 2026-09-17 logs only 3 rows ran at
  hardware max clocks. The rotated design makes the *comparison* fair; the *absolute* 5.36 is
  a throttled number and should be labelled as such wherever it is quoted.
- **Regime (L6):** the runs are from the adb `shell` domain with 6.4–6.9 GB available and a
  4000 MiB cache. `G-RAM-SWEEP` found an unprivileged app gets 1.0–5.0 GB depending on device
  state, and BigMoeOnEdge's own table gives 2.4 tok/s at a 2000 MiB cache. F6 ("beyond-DRAM as
  an unprivileged app") is the regime the project says matters, and no number exists in it.
- **Prompt (L1/L6):** one prompt, a `<think>`-style planning text with high repetition; the
  78% hit rate is prompt-specific (ESTIMAND §6). The per-token CSV shows tokens 65–256 reading
  *more* per token than the 256-token mean (198 vs 194 MiB), i.e. locality was higher in the
  cold-start segment than in the steady-state segment.
- `pin2_recycle_not_a_gain`: holds. Recycling cuts management from 35 to 14 ms/token and the
  residual rises from 100 to 115 (`hr_recycle_mgmt_ms`, `hr_recycle_compute_ms`): the cost moved
  into the compute cores (TLB shootdowns / mm lock), which is exactly what a residual hides.

### 1.7 Verify cost (verify_cost_*) — holds; interpretation corrected

c(2)=1.71, c(3)=2.37, c(5)=3.71 are what `bmoe_verify_clean/log.txt` says (3 repeats, awake,
identical nll per N so the instrument is deterministic). Each extra verified position costs
~0.135 s, i.e. about 70% of a whole single-token decode. The documents read the near-linearity
as a property of the CPU. **The traces say bytes alone predict it:** at rho 4 the union of a
5-token window fetches 4.17× (OLMoE), 4.29× (gpt-oss-20b), 4.18× (granite-3b) what one token
fetches (`hr_union_c5_*`), because at a cache that already holds the reusable experts the
window's union buys almost nothing. So c(N) ≈ N·0.75 is what a purely I/O-bound engine would
also show, and **moving the verify to a GPU would not make speculation pay on these models**
unless the cache is small (rho ≈ 1, where c_io(5) ≈ 3.6). The break-even α ≈ 0.85 for a
4-token draft stands, and is out of reach of any lossless drafter measured here.

### 1.8 S9 (s9_*) — holds; one on-device data point the documents do not mention

Pre-registration, tolerance from split-half noise, and the confound control are exemplary. The
result stands: curves are model-specific. **But** the engine run gives a measurement the S9
files do not use: at cache 4000 MiB Qwen3-30B-A3B's rho is 4.12, the measured hit is 0.780,
H_rho predicts 0.771 and H_floor 0.514 (`hr_engine_hit_vs_h_rho`, `hr_engine_hit_vs_h_floor`).
It is a coarse check (one prompt, cumulative over prefill and cold start, BigMoeOnEdge's own hit
definition) and not the pre-registered test, but it is the only Qwen data point on the phone,
and it sides with the rule the documents call refuted. Until the Qwen trace is scored,
`ARCHITECTURE.md` §3's "4.2 is now the better estimate" is not supported by the phone.

### 1.9 GPU / NPU rows (README) — holds, one addition

`gpu_speed` (1.5–1.9 tok/s) is the PR #25294 engine without overlap: not a GPU result, a
missing-overlap result, as the README says. `ocl_split` (OLMoE resident) and the peer's
`npu_compare` show the Adreno rate **did not move** across the thermal drift that halved the
CPU rate (46–48 tok/s in three repeats vs 51 → 24 for the CPU). The documents record the GPU
as "beats the throttled CPU"; the stronger, testable statement is that the GPU is the
sub-system that does not throttle on this phone, which bears directly on §4.3.

---

## 2. Where a token's 187 ms actually go

From the 16 clean pinned/unpinned runs (`headroom_sims.json` → `phone_csv`, tokens 65–256):

| cell | tok/s (65–256) | residual "compute" ms | I/O stall ms | mgmt ms | stall = a + b·MiB |
|---|---|---|---|---|---|
| pinned t4 (c4–7) / io4 (c0–3) | `hr_pinned_steady_tok_s` | `hr_pinned_compute_ms` | `hr_pinned_stall_ms` | `hr_pinned_mgmt_ms` | a ≈ `hr_pinned_stall_intercept_ms`, b ≈ `hr_pinned_stall_slope` |
| unpinned t4 / io4 | `hr_unpinned_steady_tok_s` | `hr_unpinned_compute_ms` | `hr_unpinned_stall_ms` | 28 | a ≈ 12, b ≈ 0.19 (r = 0.98) |

Three readings that the engine's own summary line hides:

1. **The residual is not clock-independent and not core-count-bound.** It fell 133 → 100 ms
   from pinning alone; 4 → 6 compute threads gave nothing (`cliff_t6_pinned`, `pin_t6*` cells);
   the cool-clock OLMoE rate was 2.1× the throttled one. The consistent reading is a
   latency/overhead-bound kernel at low clock, not an ALU-bound or DRAM-bound one (it moves
   ~1.9 GB of active weights per token at ~19 GB/s against 59.7 GB/s DRAM). A profile
   (§5, E1) is the cheapest measurement in this whole project and has not been taken.
2. **The stall has a fixed part.** The intercept (11–27 ms) is the per-layer head-of-line
   wait: a layer's first missing expert cannot be requested before its router runs and each
   layer has ~2 ms of compute to hide a ~1 ms read behind. The slope (0.14–0.20 ms/MiB, a
   marginal 5–7 GB/s) is the part overlap already hides ~40% of. Perfect within-token
   prediction bounds the removable stall at ~50 ms/token; perfect bandwidth at ~30.
3. **Management is a real 13–35 ms** and it is *not* zero at steady state as BigMoeOnEdge's
   docs expect; recycling halves it and pays the difference in the compute cores. Committed
   slot pools that never decommit (§4.4) remove it rather than move it.

Ideal bound for this engine on this model at this cache and *these* clocks: max(residual,
I/O) ≈ 100 ms → ~10 tok/s; at cool clocks (residual ~50 ms) ≈ 70 ms → ~14 tok/s; anything
above needs fewer bytes or a faster kernel. That is the honest ceiling of the current design,
and it is where "10 tok/s" sits: reachable only with the clock cap lifted or the compute moved.

---

## 3. The offline results re-run at the phone's operating point

All from `headroom_sims.json`; 16384 tokens; the project's simulators unchanged.

### 3.1 Lookahead horizon (OLMoE and gpt-oss-20b)

| model | rho | LRU | H=4 | H=8 | H=16 | Belady | gap closed by H=4 |
|---|---|---|---|---|---|---|---|
| OLMoE | 2.4 | 0.594 | 0.716 | 0.766 | 0.771 | 0.771 | `hr_la_olmoe_rho24_gap4` |
| OLMoE | 4.0 | 0.774 | 0.833 | 0.868 | 0.895 | 0.899 | `hr_la_olmoe_rho4_gap4` |
| gpt-oss-20b | 4.0 | 0.910 | 0.931 | 0.944 | 0.954 | 0.958 | `hr_la_gptoss_rho4_gap4` |

The horizon that reaches Belady grows with rho, as it must: Belady's advantage is knowing
reuse distances up to roughly the cache size in events. At the phone's rho the exact
lookahead lever needs 16–32 tokens of known routing, which no verifier supplies. The
LRU-to-Belady gap itself shrinks with rho (12.5 pp OLMoE, 4.8 pp gpt-oss-20b, 3.2 pp
granite-3b at rho 4: `hr_gap_*`), so **eviction policy is a small lever at phone cache sizes**,
consistent with BigMoeOnEdge's own offline replay ("~5 pp over LRU").

### 3.2 Union fetch (the I/O side of speculation) — §1.7. Dead at rho ≥ 2 for E/k ≥ 5.

### 3.3 Miss structure per layer-event (what a one-expert prefetch could remove)

At rho 4, P(a layer-event has ≥ 1 miss) is 0.75 (OLMoE), 0.28 (gpt-oss-20b), 0.31
(granite-3b) with mean misses 1.8 / 0.36 / 0.41 per event (`hr_p_miss_*`). On Qwen3-30B-A3B
(E/k = 16, between these) the measured 70 re-reads over 48 layers is 1.5 per event. The
head-of-line stall is paid per event with ≥ 1 miss, so its removable part scales with these
probabilities, not with bytes.

### 3.4 Within-token cross-layer prefetch, re-priced

| rho | fwd ms | speed-up | hit | flash reads/token |
|---|---|---|---|---|
| 0.8 (the closed verdict) | 30 | 0.76 | 0.27 → 0.61 | 93 → 153 |
| 2.4 | 60 / 100 | `hr_pf_rho24_fwd60` / `hr_pf_rho24_fwd100` | 0.57 → 0.89 | 55 → 76 |
| 4.0 | 60 / 100 | `hr_pf_rho4_fwd60` / `hr_pf_rho4_fwd100` | 0.76 → 0.93 | 31 → 46 |

The sign flips because two things the closed verdict assumed are false on the phone: compute
is 60–100 ms not 30, and at rho ≥ 2 the extra reads no longer churn a cache too small to hold
them. BigMoeOnEdge measured its own `--predict-prefetch` as a loss (−21%, +37% bytes), but on a
thermally drifting device and with a per-layer gate GEMV plus barrier on the eval thread that
the simulator does not charge. The simulator is optimistic (uniform wrong guesses, no CPU cost
for the prediction); the engine measurement is contaminated. Neither settles it; §5 E3 does.

### 3.5 Lexical routing: how much routing the token id alone predicts

Table fitted on the first half of the wikitext trace, scored on the second (`lexical`):

| model | lexical (token id) | bigram | persistence | independence |
|---|---|---|---|---|
| OLMoE (Q4_0, llama.cpp) | `hr_lex_olmoe` | 0.45 | 0.37 | 0.125 |
| gpt-oss-20b | `hr_lex_gptoss` | 0.48 | 0.49 | 0.125 |
| granite-3b | `hr_lex_granite` | 0.60 | 0.47 | 0.20 |

Early layers are strongly lexical (granite L0 0.80, gpt-oss L0 0.61 vs persistence 0.41 /
0.23); late layers of gpt-oss are contextual (persistence wins). This is *information a
drafter's token guesses carry at zero I/O*, which S12's predictors did not have (they saw only
routing). With it, a `protect`-mode veto at horizon 8 and 50% slot accuracy gives +2.6 pp at
rho 4 and +5.6 pp at rho 2.4 (`hr_protect_rho4_acc05`, `hr_protect_rho24_acc05`), bounded by
§3.1's gap. A modest, lossless, cheap lever — not a barrier-breaker — and honest about it.

---

## 4. Beyond the plan: where the untried headroom is

Evidence classes: **[D]** demonstrated with numbers in a cited source (on some device), **[M]**
measured in this repository, **[T]** supported by arithmetic on measured inputs, **[U]** untested.
Sources are in the three memos under [`research/`](research/) (AI literature sweeps, provenance
[U] unless a claim was verified locally; each memo lists what it could not verify). Items the
project has already closed are not repeated. The ordering is by expected ms/token on this device
× evidence; the experiments that decide each item are in §5.

### 4.1 The compute residual (~100 ms/token) — the largest term, and the least understood

1. **Instrument it before touching it [D elsewhere, U here].** llama.cpp issue #26200 profiled
   *this model's* decode graph (Qwen3-30B-A3B, 4 threads, CPU) with a per-node and per-barrier
   ledger: ~2,000 barriers per token, costing 10.8 ms/token with ggml's spin barrier and 30.8 ms
   with OpenMP, at ~4 GHz on homogeneous cores. The phone build is `GGML_OPENMP=OFF` (verified in
   `build-android-i8mm/CMakeCache.txt`), so it pays the smaller case, but at 1.75 GHz and with
   I/O lanes spinning on the other cluster the arithmetic expectation is 20–35 ms of the residual.
   Neither this nor a hardware-counter profile (`simpleperf` on the compute threads only) has
   been taken. Until it is, every compute lever below is a guess with a sign.
2. **The clock cap is an OEM governor, not thermal, and it is the single largest measured gap
   on this device [D on a sibling device, M here].** OnePlus ships `cpufreq_bouncing` (clamps
   `scaling_max_freq` through `freq_qos` after ~50 ms of sustained load) and
   `oplus_bsp_task_overload` (a `uclamp.max` clamp on "abnormal" threads, 466/792 of mid-cluster
   capacity ≈ 2.08 GHz on the OnePlus 13), documented in
   `wyl2607/oneplus13-performance-investigation`. The 15R logs show `status=0` with caps at
   2.0 GHz in 29 rows and hardware max in 3. The measured consequence on this device is the
   2.1× swing in resident OLMoE decode (50.9 → 24 tok/s, `ocl_split/runs.csv`). The sibling
   investigation found no unprivileged workaround; the candidates (OnePlus High Performance
   Mode, Game Space allow-list, an ADPF `PerformanceHintManager` session) are cheap to falsify
   with a frequency histogram, and a cooled phone gives the *limit* measurement the paper needs
   anyway (`ESTIMAND.md` requires sustained and peak side by side).
3. **Adreno hybrid: resident hits on the GPU by zero-copy, misses on the CPU [D mechanism, U
   combination].** Qualcomm's OpenCL exposes `cl_qcom_dmabuf_host_ptr` /
   `cl_qcom_android_ahardwarebuffer_host_ptr`, so an O_DIRECT read can land in GPU-visible
   memory with no upload; llama.cpp's OpenCL backend uses none of them, which is why patch 0005
   copied slots and the PR engine stalled. llama.cpp discussion #24528's inverted hybrid (thread
   0 dispatches cache-hit rows to the GPU while the other threads compute misses) measured
   +10–57% on desktops *with* a PCIe penalty the phone does not have. This device's own data:
   the Adreno decoded resident OLMoE at 46–48 tok/s across the thermal drift that took the CPU
   from 51 to 24 (`ocl_split`, `npu_compare`). Arithmetic at the memo's 34 GB/s Adreno figure:
   ~49 ms for the 0.87 GB dense + 0.80 GB of hits, concurrent with ~12 ms of CPU misses — a
   60–70 ms token if the I/O stall is unchanged. A single micro-benchmark (O_DIRECT into a
   dma-buf, wrap, generic `MUL_MAT_ID`, report GB/s; kill below ~25 GB/s) decides it.
4. **Repack, controlled, then offline [D conflicting].** Arm's own measurement (arXiv
   2501.00032) has Q4_0_8x8 kernels up to 2× on token generation at low core counts; the
   project's two retests show no gain, but §1.5 shows the retest compared different text and
   paid the repack inside the read path. The offline variant — experts stored on flash already
   in the interleaved layout, 4 KB-aligned, so the lanes copy and never convert — has no cited
   precedent in any streaming engine (llama.cpp removed file-level repacked types for mmap
   portability, PR #10446, a reason that does not apply to an engine that has already given up
   mmap). Cheap to gate: assert at runtime which GEMV symbol executes and whether `i8mm` was
   detected (issue #10662 documents it going undetected on NDK builds).
5. **Hexagon HTP: close it with one measurement [D negative].** BigMoMo (arXiv 2609.14643) ran
   exactly this workload — Qwen3-30B-A3B Q4_0 streamed from flash with experts on the HMX — on a
   OnePlus 15 and reports 304 ms/token after saturating the NPU; its cost model adds two
   movement stages (`TeMap`, `TeVtcm`) the CPU path does not pay, and the Snapdragon backend has
   a ~3.5 GB per-session mapping limit. The 15R's CPU path at 187 ms/token is already faster.
   Timing dma-buf registration of one freshly read 2.65 MB slice into an HTP session closes the
   branch in a day if it exceeds ~0.3 ms.
6. **LUT GEMV (T-MAC-style `tbl` instead of unpack + `sdot`) [T].** Fits a streamed expert
   unusually well (a 2048-row matrix used once, so the table build is amortised), but only pays
   if the profile in item 1 attributes the residual to the dequant stream. Conditional; do not
   build before item 1.

### 4.2 The I/O stall (~54 ms/token, ~25 fixed)

1. **Split-phase expert read [T, exact, U].** SwiGLU needs `gate` and `up` before `down`; the
   `down` projection is off the critical path by one matvec. Issuing `gate|up` (1.76 MB) first
   and `down` (0.88 MB) deferred takes a third of the head-of-line read out of the wait, up to
   ~13 ms/token against the measured 25 ms fixed term, at no fidelity cost. No published
   streaming engine does this (searched; "not found", not "verified absent").
2. **Within-token cross-layer prefetch, re-priced [M here in simulation, D-negative in one
   engine].** §3.4: +24–35% in the project's simulator at rho 2.4–4 with the phone's compute
   cost; BigMoeOnEdge's `--predict-prefetch` (stale-gate, 88.6% slot accuracy on Qwen3-30B by
   their measurement) lost 21% in a thermally matched pair while raising the hit rate 4.3 pp
   and bytes 37%. The two disagree because the engine charges a per-layer gate GEMV plus a
   barrier on the eval thread and speculates two experts per layer; the simulator charges
   neither. A thermally matched A/B on the pinned configuration with `predict_spec_max=1` and
   per-core frequency logged is the experiment; the simulator's number is the pre-registered
   upper bound.
3. **Expert-contiguous, 4 KB-aligned layout [D elsewhere].** llama.cpp discussion #27149's
   relayout cut one expert from 6,912 touched pages to 192 and BigMoMo reorganises experts on
   flash by co-loading; on this device the memo's arithmetic gives 4–11 ms/token from
   collapsing ~253 reads to ~85 and removing O_DIRECT round-out. It does not touch the fixed
   term. Cheap to build, cheap to measure by the stall regression's slope.
4. **Exact cross-token layer-0 routing [T, novel].** Layer 0's router input for a candidate
   next token is computable exactly from its embedding, layer-0 attention over the cached KV,
   and the resident router matrix — microseconds, zero I/O, Tier E by construction. It covers
   one layer of 48 (~2% of bytes) but it is the only exact lookahead that crosses the token
   boundary, where the first layer's head-of-line wait sits. No prior art found for the
   cross-token version.
5. **Flash-level tricks are closed [D negative].** `io_uring` is seccomp-blocked for apps and
   has no `shell.te` grant; UFS HPB/MCQ are kernel-side; polling needs a block-driver poll UFS
   does not implement; splitting one expert across four lanes cannot beat the 3.2 GB/s
   aggregate floor (~0.83 ms per expert). Expected effect ≤1 ms/token.

### 4.3 Cache management (13–35 ms/token)

**A pre-committed, never-decommitted slot arena [T, strong mechanism].** The engine commits and
decommits slot pages per miss (`MADV_DONTNEED` + first-touch zero-fill; recycling moved the cost
into TLB shootdowns on the compute cores, §1.6). One `mmap` of the whole cache, populated once
(`MADV_POPULATE_WRITE`, ~110 ms for 4 GB), overwritten in place by O_DIRECT, removes the term
rather than moving it: the memo's arithmetic gives 10–30 ms/token. The cost is memory shape:
a locked anonymous arena is what LMKD kills first, so it needs `mlock` headroom (shell domain has
it; an app may not) and a runtime-sized budget. `--defer-evict` (patch 0007, queued) amortises the
shootdowns only; it is strictly weaker.

### 4.4 Bytes per token and more than one token per fetch

1. **Lossless compression of Q4_0 experts on flash is closed by measurement [M].** The deployed
   Qwen3-30B-A3B GGUF's expert nibbles carry `hr_q4_nibble_entropy_bits` bits of zeroth-order
   entropy per 4-bit code and the fp16 scales `hr_q4_scale_entropy_bits` of 16, so any order-0
   coder saves at most `hr_q4_max_lossless_saving` of expert bytes (~2.5 ms/token of exposed
   stall at the measured 0.151 ms/MiB) for an ARM decode cost on the lanes. The published >10% claims for int4 are mutually
   inconsistent and none is on GGUF (bytes memo §1). Expert-to-expert delta coding is worse than
   nothing under per-block scales.
2. **MTP-head checkpoints with the verify on the GPU [D in colibri, T here].** Qwen3-30B-A3B
   ships no MTP head; Qwen3-Next-80B-A3B (stock head: per-position acceptance 0.897 / 0.719 /
   0.476, FastMTP) and GLM-4.5-Air do, and llama.cpp merged MTP (PR #22673). colibri stacks
   expert caching and MTP super-additively on a disk-streamed GLM-5.2 (+51%). On this device
   the traces set the floor: a 2-position window fetches `hr_union_c2_olmoe_rho4`× one token's
   bytes at rho 4, so on the I/O side the break-even accepted length is ~1.9 per 2-position
   pass, and the CPU's measured c(2) = 1.71 already sits there. It pays only if the batched
   verify's *compute* becomes nearly free (a GPU GEMM over N rows) *and* acceptance stays near
   0.9; then a 2-position pass costs ~44 ms of GPU compute plus ~95 ms of I/O for ~1.9 tokens,
   ~73 ms/token against ~94 — about 1.3×, conditional on 4.1.3. Two hazards: the head must
   survive GGUF conversion (unverified), and batched verification under quantised kernels is
   not byte-identical (Tier E becomes Tier A).
3. **Dynamic top-k / expert skipping at the KL level [D at perplexity, U at KL].** ACE (arXiv
   2609.05228) reports 50% skipping on Qwen3-30B-class models at perplexity and task accuracy;
   no paper in this line reports KL or top-1 flip rate. The project already owns the instrument
   (G3's harness) and the pre-registered margin; this is Tier A, planned in `ARCHITECTURE.md`
   §5b, and the memo confirms nobody has done the measurement. Largest possible byte cut; the
   result is publishable either way.
4. **Lexical routing table as a zero-I/O eviction hint [M offline].** §3.5: +2.6 pp at rho 4 for
   a 50%-accurate horizon-8 hint in `protect` mode, and the token id alone supplies 0.45–0.59
   slot accuracy given the token. A published shared-prefix study on Qwen3.5-35B-A3B reports
   same-token Jaccard 0.83 at layer 0 and 0.65 on average (arXiv 2604.17182), consistent with
   this. Cannot hurt fidelity; worth a few ms; not a barrier-breaker.

### 4.5 The regime the number is quoted in

Everything above is measured from the adb shell with 6.7 GB free and a 4 GB cache on a
throttled SoC over one prompt. The paper's F6 claim needs the unprivileged-app regime
(`G-RAM-SWEEP`: 1.0–5.0 GB, bimodal), a fixed prompt set (`ESTIMAND.md` §5), and a 30-minute
sustained run. None exists. That is not a lever; it is the difference between a bring-up number
and a claim.

---

## 5. Experiments that settle it, cheapest first, with the prediction written before the run

Each row names the measurement, the artifact field it would move, and what is expected. A
prediction that fails is a result (`CLAUDE.md` §8). None has been run.

| # | experiment | cost | pre-registered expectation | kills the lever if |
|---|---|---|---|---|
| E1 | per-node + per-barrier ledger (#26200 port) and `simpleperf` counters on the 4 compute threads, pinned config | 1 day, phone | barriers 20–35 ms of the 100 ms residual; backend-stall cycles > 50% | — (classification, not a lever) |
| E2 | `scaling_cur_freq` histogram at 10 Hz under default / High Performance Mode / Game Space / ADPF session; then one run on an actively cooled phone | 2 h, phone | default median ≤ 2.1 GHz; app-side toggles most likely 1.0× (memo); cooled run: residual ≤ 60 ms, i.e. ≥ 8 tok/s at the same cache | frequency histogram does not move |
| E3 | arena allocator (never decommit, populate once) behind a flag; `min_flt` per token before/after | 2 days | mgmt 35 → ≤ 5 ms/token, residual unchanged; if the residual rises, the cost moved again | mgmt does not fall |
| E4 | split-phase read (`gate\|up` first, `down` deferred) | 2 days | stall intercept 25 → ≤ 17 ms; slope unchanged | intercept unchanged |
| E5 | thermally matched A/B of `--predict-prefetch` with `predict_spec_max` 1 and 2 on the pinned config, per-core frequency logged, rotated order | 3 h, phone | simulator bound +24% (rho 4, 100 ms); engine result between −20% and +24%; positive only if the gate GEMV + barrier cost < 15 ms/token | stall falls but wall time does not |
| E6 | expert-contiguous 4 KB-aligned relayout | 1 day | stall slope 0.151 → ≤ 0.12 ms/MiB; intercept unchanged | slope unchanged |
| E7 | O_DIRECT into dma-buf + `cl_qcom_dmabuf_host_ptr` + generic OpenCL `MUL_MAT_ID` micro-benchmark | 2 days | ≥ 25 GB/s effective on the hits; kill below | < 25 GB/s |
| E8 | repack control: assert GEMV symbol and `i8mm` detection; repack on the lanes; same-text comparison (teacher-forced `--ppl` path) | 1 day | compute residual −5–20%, stall unchanged | residual unchanged with the repacked symbol confirmed |
| E9 | HTP map cost: dma-buf registration of a freshly read 2.65 MB slice, 1000×, HTP v81 session | 1 day | > 0.3 ms per expert (closes the NPU branch) | — |
| E10 | KL-instrumented ACE / Elbow dynamic top-k on Qwen3-30B-A3B against the Q4_0 margin | GPU, 2 days | unknown; the memo found no KL number anywhere | KL above the Q4_0 margin at every skip fraction |
| E11 | GPU batched verify c(2), c(3), c(5) with the CPU protocol, on a checkpoint with an MTP head, after confirming the GGUF carries it | 1 week | c(2) ≤ 1.3 on compute; I/O side ≥ 1.9 (traces) | c(2) on the GPU ≥ 1.7 |
| E12 | the regime: same engine as a Termux app, foreground service, cache sized from `getMemoryInfo`, 30-minute sustained run, fixed prompt set | 1 day, phone | ≥ 25% below the shell-domain number | — (this is the reportable number) |

Order: E1 and E2 first (they decide what every other row means), E3/E4/E6 next (lossless,
engine-only, no fidelity check), E5 and E7 after E1 says where the residual is, E8–E11 as
branch-closing measurements, E12 before any number is quoted outside this repository.

## 6. What this document could not verify

- The GPU-path OLMoE thermal stability (§1.9) rests on 3 + 3 repeats in two campaigns; it has
  not been run on a streamed model.
- `hr_*` trace sims use 16384 tokens; the 32768-token G2 artifacts agree within 0.005 where
  both exist (checked at rho 4 against `cache_fcrit_OLMoE-1B-7B-0924.json`).
- The lexical section needs the token-id files from `gates/llamacpp_traces.py tokens`, which
  are not committed (their sha256 is recorded in the artifact).
- The Qwen3-30B-A3B routing trace was being collected by the other session while this was
  written; §1.8 is a coarse check, not the S9 score.
