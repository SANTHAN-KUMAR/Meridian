# Why the speed levers keep coming back neutral or negative, and what can still move the rate (2026-09-18)

Analysis only (no phone run, no model run). Sources: the committed artifacts and logs named inline,
BigMoeOnEdge source at the patched commit (`moe-work/BigMoeOnEdge`, base 74ba18f), and one small
simulation on the committed Qwen3-30B-A3B trace (command at the end). Numbers quoted from logs are
marked with their file; none is a claim of this project until it enters `CLAIMS.md`.

## 1. The A/B design cannot see the effects being tested (L3)

`bmoe_order3.json`, arm `idorder`, six repeats of one binary and one configuration, identical bytes
and hit rate in every row: 5.83, 6.71, 6.55, 6.55, 6.58, 6.05 tok/s. That is a 5.5% run-to-run
standard deviation from the clock governor alone. With that noise, an 80%-power two-arm test needs
about 5 rows per arm for a 10% effect, 19 for 5%, and 53 for 3%. The campaigns use 3.

Every lever tested since the 5 GB cache was expected to be worth 3 to 7% (SLRU, arena, expert
order, floor768, ub64, lanes). So "no gain" and "a loss" are both uninformative at n=3, and the sign
flips between campaigns are what that design produces. This is `CLAUDE.md` §4.1: the denominator is
too small, and no amount of careful scripting fixes it.

What does work on this phone, in order of cost:

1. **Judge a lever on its clock-independent counters first** (bytes per token, hit rate, experts
   speculated and used, minor faults per token). These were identical to the decimal across thermal
   states 0 to 3. SLRU's 11% byte cut is confirmed this way; its decode effect is then arithmetic.
2. **Judge it on its own time term, not only on wall time.** The three terms are a serial chain and
   the "compute" term is a residual, so a lever that removes cost from one term can push it into
   another (recycling, the arena). Report the term it targets and the residual side by side.
3. **Make the comparison inside one process.** For any lever that can be switched at run time
   (prefetch, expert order, defer-evict), toggle it every 16 tokens over a teacher-forced text
   (`--ppl-step`). Both arms then share the same clock state, and the per-block paired difference
   has a fraction of the row-to-row variance. This needs one small flag, and it turns a 50-row
   campaign into one run.
4. **Measure in the regime the governor settles into.** The caps return under sustained load
   (`bmoe_boost` commit 8fc0f58). Pre-heat with a 60 s dummy decode, then measure 512 tokens. The
   capped steady state is also the only deployable number.

## 2. Cross-layer prefetch was never actually exercised

Both on-device negatives (BigMoeOnEdge's own, and `bmoe_boost` predpf) measured an implementation
that destroys its own speculation.

Evidence from `results/2026-09-17/bmoe_boost/predpf_rep*.out`: `moe-prefetch: 3894.7 MiB speculative,
563/802 experts useful`. Over 256 tokens that is 3.1 experts integrated per token, against roughly 48
misses per token, and the 802 completed experts account for about 2030 MiB of the 3895 MiB read. So
about half the speculative bytes went into entries that were thrown away, the stall moved from 50 ms
to 39 to 50 ms, and cache management rose by 8 ms (the quiesce waits inside the management window).

The cause is in `core/src/moe/expert_stream_source.cpp`:

- line 68: `spec_adopt_ = cfg.route_ahead > 0;` with the comment "Off for every guessing predictor".
- `quiesce_spec()` without adoption (line 689 on): bumps the generation, clears the queue, waits for
  in-flight reads, and **releases the pages of every entry that had not finished**, including
  correct guesses. It runs at every layer's load (line 1349).
- `settle_spec()` runs the same destructive quiesce **before every issue**, so issuing layer l+1's
  prediction cancels whatever is still pending.
- The source's own comment records the consequence for the unadopted path: "at depth 2 every early
  read was destroyed ... for 0% useful and double flash per token". It was fixed only for
  `--route-ahead`, which is lossy.

A speculative read gets one layer of wall time (about 2.8 ms) on lanes that also serve the demand
reads; anything not complete by the next load is discarded and read again on demand.

**The lossless fix is selective adoption.** At layer `il`'s load the true ids are known:

1. compute `staged_` before the quiesce (it is a pure function of `ids`);
2. speculative jobs of layer `il` whose expert is in `staged_` are adopted exactly as the
   route-ahead branch does (move to the front, wait, integrate hot);
3. unstarted jobs of layer `il` for experts not in `staged_` are erased from the queue and their
   pages released; in-flight ones finish and integrate cold;
4. other layers' jobs stay queued; `settle_spec()` becomes integrate-only in this mode.

The objection in the source ("adopting a wrong guess would make the load wait on reads it does not
need") does not apply, because only guesses confirmed by the router are waited on. Byte identity is
unchanged (gate G10). The existing adoption-wait counter already reports the cost.

Expected size: the stall is 48 ms and is almost exactly the miss traffic at full flash bandwidth
(`decode_budget.json`), and the lanes are idle for most of the 93 ms residual. With BigMoeOnEdge's
measured 88.6% stale-gate slot accuracy on this model, 20 to 35 ms per token is the plausible range.

One caution from the simulator on the Qwen trace: prefetching all eight predicted experts reads 94.8
experts per token against 47.8, because with 128 experts a wrong guess is almost always uncached.
Keep the cap on predicted misses per layer (start at 2, then 3), and rank by predicted gate weight.

## 3. The slot arena removed its term and lost it again for a fixable reason

`bmoe_arena` (commit c8d85ea): management 34 ms to 2 ms, residual up about 14 ms, with the pool
spread over 169 separate 32-slot reservations. Fresh kernel allocations tend to be physically
contiguous and can use contiguous-PTE or multi-size huge page mappings; recycled slots scattered
over 169 mappings cannot, so the expert GEMVs take more TLB misses. Test one contiguous reservation
with slots aligned to 64 KB, `MADV_HUGEPAGE` if `/sys/kernel/mm/transparent_hugepage/enabled` allows
it, populated once. Judge it on minor faults per token and on the residual of a fully resident
layer, then on wall time with the in-process design. Worth about 15 ms if the residual stays put.

## 4. Stop the accelerator placement work for decode

The per-op sweep (commit 8fbb43f) settles it: the CPU is the fastest device for the expert matmuls,
the GPU needs a 99.9% hit rate to pay for uploads and gets the down-projection wrong, and attention
on the NPU lost 17% through 192 backend crossings per token. Attention on the GPU can win at most a
share of 18.6% of node time, minus the same crossings. None of this reaches 10 tok/s.

## 5. An honest budget to 10 tok/s, lossless

Start: 161 ms per token at the 5 GB cache, capped clocks (`decode_budget.json`).

| lever | evidence | expected |
|---|---|---|
| prefetch with selective adoption | mechanism confirmed in source; never tested | 20 to 35 ms |
| contiguous arena | management cut measured; residual regression has a named cause | about 15 ms |
| segmented LRU | 11% fewer bytes, clock-independent, measured | about 5 ms |
| 6 GB cache when memory allows | simulator matches the phone within 3 points | about 10 ms |

All four landing gives roughly 100 to 110 ms, which is 9 to 10 tok/s at capped clocks and above 10
in the uncapped state where the residual was 79 ms instead of 93. Without the prefetch fix the
lossless ceiling on this model is about 8 tok/s.

If a Tier A lever is acceptable, routing to 6 experts instead of 8 is the largest measured one
(BigMoeOnEdge: 4.0 to 5.0 tok/s on this phone class). It needs its KL measured against the Q4_0
margin before it is quoted.

## Reproduce the simulation

`prefetch_sim.replay` on `results/2026-09-16/traces_Qwen3-30B-A3B-q4_0-llamacpp.npz`, 4096 tokens,
cache 5000 MiB, expert 2.654 MB, 2.806 GB/s, uniform recall 0.886, compute 93 ms: 7.23 to 8.14 tok/s,
reads 47.8 to 94.8 experts per token, 31 wasted. At recall 0.75 it loses. Not yet a committed
artifact; fold into `gates/headroom_sims.py` before quoting.
