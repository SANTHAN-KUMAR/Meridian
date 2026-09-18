# Handoff: the 10 tok/s ceiling on the capped 15R, and the decision plan to break it or close it (2026-09-18)

**For:** the two other working sessions (phone/engine track, and analysis/GPU track).
**Status of this document:** analysis of committed artifacts only. No phone run and no model run was made.
**Validity level (`CLAUDE.md` §1):** L3 (does the design permit 10 tok/s?) and L4 (is "compute" the
quantity we think it is?). Nothing here is an L5/L6 claim until it enters `CLAIMS.md`.

**Numbers.** Every number below is a field of `results/2026-09-18/ceiling_ledger.json`, produced by
`gates/ceiling_ledger.py` from committed inputs (node trace, `ocl_split`, `stack_summary.json`,
`clock_trace`). Fields are cited as `[CL: path]`. Values are shown rounded for reading; the artifact
is the source, and none of them is a project claim until it is added to `gates/claims_check.py`.

**Evidence labels.** **[M]** measured in this repository. **[T]** arithmetic on measured inputs.
**[H]** hypothesis with a named mechanism, untested. **[X]** closed by measurement, do not reopen.

---

## 0. The answer in ten lines

1. The clock cap is the phone's **normal state**, not an event. Design for it. **[M]**
2. At capped clocks the token costs compute ≈ 124 ms + stall ≈ 35 ms + management ≈ 25 ms
   `[CL: capped_terms.stack]`. Everything the project has won so far came out of the 59 ms of I/O
   terms. **The 124 ms has not moved under any lever**, and it alone exceeds the 100 ms budget. **[M]**
3. With today's I/O terms, 10 tok/s allows ≈ 41 ms of compute: a 67% cut
   `[CL: capped_terms.compute_cut_needed_fraction]`. No I/O lever can reach the goal by itself. **[T]**
4. That compute is **arithmetic, not overhead**: in-graph, a 0.59 MB matmul takes tens of
   microseconds, small ops are a minor share, and major faults explain ≈ 3 ms/token. **[M]**
   *(This withdraws a statement I made earlier in conversation, that a fixed ≈ 0.4 ms per matmul
   dominated. That figure came from the one-node benchmark harness, not from the decode graph.)*
5. Expert matmuls are the dominant arithmetic, and **gate/up run 1.7× slower per byte than down**
   on the same cores in the same run `[CL: node_ledger.up_vs_down_per_byte_ratio]`. Unexplained.
   It is the largest specific, lossless, untested compute lever in the repository. **[M → H]**
6. A frequency cap limits clock, not core count. The capped CPU uses well under a third of the
   measured DRAM bandwidth, so **width (6 compute threads on the policy0 cores) is the other real
   lever**. It has never been tested on six same-domain cores in the logged capped state (§3.1). **[H]**
7. **Best case, both levers landing perfectly: ≈ 8.0 tok/s** at today's I/O terms
   `[CL: projection.best_case_tok_s_width_plus_gate_up]`. Reaching 10 then *also* needs I/O cut from
   59 to ≈ 35 ms `[CL: projection.io_ms_allowed_at_10_tok_s_after_both]`. **[T]**
8. So: **lossless (Tier E) 10 tok/s at capped clocks is a triple conjunction and is unlikely;
   lossless 7.5–8 is realistic.** 10 tok/s has two credible routes: (a) Tier A top-6 routing on top
   of the compute levers, with its KL measured; (b) a materially higher cap step when the phone is not
   charging. Both are one decisive experiment each (§4).
9. GPU/NPU placement, speculation, eviction policy, lanes, ZRAM, in-read-path repack: closed (§1.4).
10. Five experiments decide everything (§4). Two are already queued. Nothing else should run.

---

## 1. What is established

### 1.1 The cap is permanent, stepped, and tracks skin temperature **[M]**

- 93 minutes of `scaling_max_freq` samples: policy6 (cpu6-7, the prime cores) is at or below
  1.65 GHz in 96% of samples; policy0 (cpu0-5) is at or below 2.27 GHz in 96%
  `[CL: clock_state.frac_p6_at_or_below_1651200, frac_p0_at_or_below_2265600]`. Median caps
  1.90 / 1.65 GHz `[CL: clock_state.p0_max_median_kHz, p6_max_median_kHz]`.
- The very first sample, phone idle, thermal status 0, battery 35.5 °C, is **already capped**
  `[CL: clock_state.first_sample]`. The phone was on AC.
- `cap_vs_temp.json`: full clocks in every sample below 31 °C shell temperature; capped in 97% of
  samples from 33 °C. The cap moves through discrete steps with temperature
  `[CL: clock_state.p0_levels_kHz]`.
- **Consequence.** The deployable number is the capped number. And because the step follows skin
  temperature, **joules per token decide which step the engine lives on**: an engine that wastes
  power (spinning threads during stalls, charging heat) pays for it twice.
- The prime cores are capped *harder* than the policy0 cores. With ggml's per-op barrier the slowest
  thread paces all four, and the current mask (cpu4-7) spans both domains. The queued
  `device/bmoe_cores.sh` tests exactly this; it stays in the plan (§4, X2).

### 1.2 The ceiling is compute, and compute is arithmetic **[M]**

Node trace of Qwen3-30B-A3B decode (`results/2026-09-17/bmoe_trace`, 64 steady-state tokens,
3030 nodes per token, taken near full clock: caps 3.03 / 3.44 GHz `[CL: trace_runs.nodes.caps_before]`).
"Typical" = sum of per-node-type medians; "tail" = mean minus median (waits, preemption, clock steps).

| category | calls/token | typical ms | tail ms | field |
|---|---|---|---|---|
| expert matmuls (MUL_MAT_ID) | 144 | 73 | 28 | `node_ledger.categories.expert_matmul` |
| dense matmuls (q, k, v, o, router) | 240 | 23 | 7 | `…dense_matmul` |
| small ops (add, norm, div, view, …) | 2549 | 14 | 29 | `…small_ops` |
| `ffn_moe_weights` GET_ROWS (engine's load hook lands here) | 48 | 14 | 6 | `…moe_weights_get_rows` |
| output head | 1 | 8 | 0 | `…output_head` |
| flash attention | 48 | 4 | 9 | `…flash_attn` |

Read with two cautions. The trace is serialised: the traced run decodes at 5.0 tok/s against 7.2 for
the untraced one `[CL: trace_runs]`, so absolute values are inflated and only shares and ratios are
used here. And `ffn_moe_gate`'s median carries the expert-ready wait; substituting `up` (same shape,
same type) for `gate` gives ≈ 58 ms of expert arithmetic `[CL: node_ledger.expert_arith_ms_gate_as_up]`.

What the ledger rules out:

- **Per-op fixed overhead is not the problem.** In-graph medians: V projection (0.59 MB) 34 µs,
  K 65 µs, Q (4.7 MB) 167 µs `[CL: node_ledger.median_us]`. 2549 small ops total 14 ms typical.
  Graph fusion (q/k/v fusion, router-chain fusion) is therefore worth single-digit ms. **Do not build it.**
- **Major faults are not the problem.** ≈ 650 per token, but concentrated in ≈ 10 nodes that together
  take ≈ 3 ms/token `[CL: node_ledger.majflt]`.
- **The residual scales with the clock.** 77 ms near full clock `[CL: trace_runs.plain]` against
  121–124 ms capped `[CL: capped_terms]`. Same engine, same model. The resident-OLMoE control shows
  the same thing in isolation: the CPU swings 2.1× across a thermal drift while the GPU moves 2%
  `[CL: capped_floor.cpu_max_over_min, gpu_max_over_min]`.

**So the L3 statement is:** with 4 compute threads and the current kernels, the capped CPU's own
arithmetic occupies more than the whole 100 ms budget. The throttled resident-model rate gives the
same answer independently: ≈ 110 ms for Qwen3's bytes `[CL: capped_floor.qwen3_ms_per_token_at_throttled_cpu_rate]`
(an OLMoE-derived rate, optimistic for Qwen3 per `results/2026-09-18/README.md`). **No amount of I/O,
cache or eviction work reaches 10 tok/s.** Only three things can: more arithmetic throughput at the
capped clock, a higher cap step, or fewer weight-operations per token.

### 1.3 A specific anomaly in the dominant op **[M]**

Per byte of weights, in the same run on the same cores: `ffn_moe_down` 26 GB/s, `attn_q` 28 GB/s,
`ffn_moe_up` 16 GB/s `[CL: node_ledger.median_GB_s]`. Gate and up are two thirds of expert bytes.
If they ran at down's rate, expert arithmetic falls from ≈ 58 to ≈ 40 ms at the trace's clock
`[CL: node_ledger.expert_ms_if_gate_up_ran_at_down_rate]`, and proportionally more at capped clocks.

Candidate mechanisms **[H]**, all checkable in one bench session (§4, X1):

1. **Thread granularity.** Gate/up are 768 rows per expert; split across 4 threads that is 192 rows
   of work per thread per expert between barriers, against 512 for down. Short chunks on cores in two
   clock domains amplify the pacing loss.
2. **Kernel path.** Down is Q4_1 and gate/up are Q4_0 in this GGUF; they take different `vec_dot`
   paths, and HEADROOM §4.1.4 notes `i8mm`/repack detection has gone wrong on NDK builds before.
   Assert which symbol executes.
3. **Input handling.** For gate/up the 2048-wide activation is shared by all 8 experts; for down each
   expert has its own 768-wide input. If the shared input is re-quantised or re-fetched per expert,
   that is pure waste.

The independent per-op sweep shows the same direction (gate/up slower per byte than down on the CPU,
`matmul_sweep.json`), so this is not a tracing artefact.

### 1.4 Already decided by the project's own measurements **[X]**

| branch | why it is closed | source |
|---|---|---|
| Hexagon for expert matmuls | 0.39–0.68× the CPU on Qwen3's shapes | claims `msweep_htp_*` |
| Adreno for streamed experts | upload per miss; `ffn_down` (Q4_1) incorrect; at best 1.09× CPU on gate/up | claims `msweep_gpu_*` |
| attention on a second device | 192 crossings/token cost more than the op saves; `gpu_dense_cpu_experts` is *slower* than CPU-only | `why_levers_flip` §4, `ocl_split/runs.csv` |
| speculative decoding | c(2) = 1.71, 71% agreement, best case 1.015× | commit `8e6014f` |
| eviction policy, lanes, defer-evict, ZRAM, in-read-path repack | each measured; ≤ 5 ms or negative | `HEADROOM.md`, `NEXT.md` tracker |
| graph fusion / per-op overhead | §1.2 above | this document |

---

## 2. The budget, stated once **[T]**

| | ms/token | field |
|---|---|---|
| compute, capped, stacked engine | 124 | `projection.compute_ms_stack` |
| stall + management, stacked engine | 59 | `projection.io_ms_stack` |
| compute with *perfect* 4→6 thread scaling | 83 | `projection.compute_ms_perfect_6_thread_scaling` |
| gate/up recovered fully (at trace clock; larger when capped) | −18 | `projection.gate_up_saving_ms_at_trace_clock` |
| **best case, both** | **124 → 8.0 tok/s** | `projection.best_case_*` |
| I/O allowed for 10 tok/s after both | 35 | `projection.io_ms_allowed_at_10_tok_s_after_both` |

Both compute rows are best cases by construction. The honest reading: the compute levers are worth
**5.5 → 7–8 tok/s**; the remaining gap to 10 needs one of §3.3–§3.4 in addition.

---

## 3. What to build, and why each has a real chance

### 3.1 Wide-and-slow: 6 compute threads on policy0, I/O on cpu6-7 **[H]**

- **Why it can work.** A frequency cap leaves core count untouched. At 1.65–1.9 GHz the Q4 GEMV is
  compute-bound (the residual tracks the clock, §1.2), and DRAM has headroom: 59.7 GB/s measured at
  4 threads (`dram_15r.json`) against ≈ 17 GB/s used when throttled
  `[CL: capped_floor.throttled_cpu_effective_GB_s]`. Low-clock cores also run at the bottom of the
  voltage curve, so width is the energy-efficient direction, which matters under a skin-temperature cap.
- **Why the earlier negative does not apply.** "6 threads gives nothing" (`cliff_t6_pinned`) was
  resident OLMoE with ggml's row-split threading, in a run whose clock state was not logged, and the
  `pin_t6*` engine cells put the extra threads on the mixed-domain mask. Neither tested six
  same-domain cores in the logged capped state, which is the regime that matters.
- **Risk.** More threads means shorter chunks per barrier (mechanism 1 of §1.3). That is why §3.2's
  expert-granular threading should land with it, and why X1 measures both together.

### 3.2 Expert-granular threading with fused gate|up **[H]**

- Each thread runs whole experts to completion (gate|up as one 1536-row GEMV → SwiGLU → down); one
  join per layer instead of three-plus barriers. Work is handed out per expert, so a slower core
  takes fewer experts instead of pacing every op.
- Each row's dot product is unchanged, so output should stay byte-identical; gate G10 (text identity)
  confirms it. Cached-hit experts can start while misses are still arriving, which hides part of the
  stall inside compute for free.
- **Why it has a chance:** it attacks the one measured anomaly (§1.3) at its most likely cause, on
  the op that is over half the arithmetic.

### 3.3 Operate off-charger, and measure joules per token **[H]**

- The idle phone on AC is already above the cap threshold (§1.1). Charging dissipates heat inside the
  same skin the governor reads. If the unplugged equilibrium sits one or two cap steps higher
  (2.19–2.27 instead of 1.75–1.90 GHz on policy0), the clock-scaled 124 ms shrinks by about the ratio of the steps, for free,
  and it is a legitimate deployment condition (phones are mostly used unplugged).
- Every campaign so far that ran on AC carries this as an uncontrolled factor.

### 3.4 Tier A bridge: top-6 routing **[M elsewhere, KL unmeasured]**

- Cuts expert arithmetic *and* expert bytes by a quarter in one move; BigMoeOnEdge measured 4.0 → 5.0
  tok/s on this phone class. It is the only lever in the repository with a ≥ 20% measured effect that
  is still open. It is not lossless, so it is reportable only with KL against the pre-registered
  Q4_0 margin (`ESTIMAND.md` Tier A), and always labelled as such next to the Tier E number.

---

## 4. The experiments. Five. Each one ends a branch or commits engineering time.

Rules for all of them (from `why_levers_flip` §1, which stands): judge on the lever's **own term**
(compute ms), not on tok/s; pre-heat 60 s so the run is in the capped steady state; log both caps
every 2 s; decision rule fixed before the first row; n sized so SE/|effect| < 0.5 for the threshold
stated.

### X1. Expert-kernel bench at capped clocks (new; ~1 h of phone time; run first)

- **Determines:** whether width and gate/up restructuring are real, before any engine code is written.
- **Why it matters:** these two are the only lossless levers sized in tens of ms (§2).
- **How:** `host/app/ggml_matmul_bench.cpp` (already has `--threads`; add a CPU-mask flag and a
  `--fused-gate-up` shape of 1536 rows, plus an expert-parallel mode where each thread runs whole
  experts). Shapes: `ffn_gate_up`, fused gate|up, `ffn_down`. Arms: 4 threads on cpu4-7 (today),
  4 on cpu2-5, 6 on cpu0-5. Pre-heated, caps logged, 10 repeats per cell, rotated. Print the GEMV
  symbol that executes.
- **Decision rule (layer-level expert ms = gate + up + down for 8 experts):**
  - 6-thread policy0 cell ≤ 0.75× today's cell → **build §3.1**. ≥ 0.9× → width is dead on this SoC; stop.
  - fused + expert-parallel ≤ 0.80× the unfused cell at the same threads → **build §3.2**. ≥ 0.95× →
    the anomaly is in the kernel path; check mechanism 2 (symbol) and stop if the right symbol runs.
  - both fail → **lossless compute is closed at ≈ 5.5–6 tok/s capped.** Go straight to X4.

### X2. `bmoe_cores.sh` (already pre-registered and queued; keep unchanged)

- **Determines:** at engine level, whether moving compute off the harder-capped prime cores cuts
  compute ms (arms `f0` / `3c` / `3f`).
- **Decision:** its own pre-registered rule. If `3f` is decisive, it confirms X1's width result inside
  the real engine with I/O running; if X1 said yes and X2 says no, the I/O lanes are what breaks
  scaling, and the fix is lane placement, not more kernel work.

### X3. Off-charger equilibrium (new; ~40 min; run once)

- **Determines:** whether charging heat is setting the cap step.
- **How:** same engine command, 15 minutes sustained, once on AC and once unplugged at 60–80%
  battery over wireless adb, second order reversed on another day if the first is positive. Log caps,
  shell temperature, and `current_now × voltage_now` (unplugged arm) for joules per token.
- **Decision:** unplugged median policy0 cap ≥ one step higher for the last 10 minutes → all further
  campaigns and the reportable number move off-charger, and the capped-compute target shrinks
  accordingly. No difference → thermal environment is closed as a lever; never revisit.

### X4. Top-6 KL (new; GPU track on Kaggle, no phone needed; run in parallel now)

- **Determines:** whether the Tier A bridge exists.
- **How:** the G3 harness, Qwen3-30B-A3B, k = 8 vs 7 vs 6, mean and p99 KL against BF16, against the
  pre-registered Q4_0 margin.
- **Decision:** within margin at k = 6 (or 7) → a labelled Tier A route to 10 tok/s exists and X1's
  levers are built with it in view. Outside the margin → the project's honest headline is the
  lossless capped rate, and "10 tok/s" is reported as not reached on this model.

### X5. `bmoe_overhead.sh` (already queued, commit `d0f5973`; keep)

- **Determines:** whether our engine adds compute cost over plain llama.cpp on a fully cached model.
- **Decision:** > 10% slower than plain → there is engine-side waste in the compute path and it is
  found before building §3.1/§3.2 on top of it. ≤ 10% → the arithmetic reading of §1.2 is confirmed
  and X1 is the whole story.

**Not on the list, on purpose:** any further n = 3 wall-clock A/B; any new prefetch/eviction/arena
variant (their terms total 59 ms and the goal cannot be reached from there); `agg_bandwidth` v2
(a positive result would cost weeks of zero-copy GPU engineering for a device that is ≈ 1.09× the CPU
on the dominant op; demote unless X1 *and* X4 both fail).

---

## 5. Decision tree

```
X1 width yes ── X1 gate/up yes ──► build 3.1 + 3.2 (judge with patch 0014 in-process toggle, compute ms)
      │                 │              expected 7–8 tok/s capped, Tier E
      │                 └─ no ────► build 3.1 only; expected ≈ 6.5–7
      └─ no ──── gate/up yes ─────► build 3.2 only; expected ≈ 6–6.5
                 └─ no ───────────► lossless compute closed at ≈ 5.5–6

then:  X3 positive  → re-measure off-charger (the reportable Tier E number)
       X4 positive  → add top-6 as the labelled Tier A row; this is the realistic 10 tok/s cell
       X3 and X4 negative → report the Tier E capped rate; 10 tok/s not reached on Qwen3-30B-A3B
```

## 6. Who does what

| session | now | then |
|---|---|---|
| phone / engine | X5 and X2 as queued; build the X1 bench flags; run X1; run X3 | implement whichever of §3.1 / §3.2 X1 selects, behind flags, byte-identity gated |
| analysis / GPU | X4 on Kaggle; add `ceiling_ledger.json` fields to `claims_check.py` and `repro_check.py`; write X1 and X3 pre-registrations before their first row | score X1–X3 by their fixed rules; update `NEXT.md` |

## 7. What this document could not verify

- The node trace is one run, near full clock, serialised. The up/down disparity is corroborated by the
  per-op sweep, but its *capped-clock* size is unmeasured; X1 measures it.
- "Perfect 4→6 scaling" is an upper bound, not an expectation.
- Whether `gate`'s extra median time is entirely expert-ready wait was assumed, not shown.
- No power measurement exists in the repository; §3.3's mechanism is physical reasoning until X3.
- I did not read the MUL_MAT_ID source at the patched commit; the three mechanisms in §1.3 are
  candidates, not findings.

Reproduce: `python moe-phone/gates/ceiling_ledger.py` → `results/2026-09-18/ceiling_ledger.json`.
