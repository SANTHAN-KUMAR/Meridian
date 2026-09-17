# Pre-registration — two questions measured on the night of 2026-09-18

Written and committed **before** either campaign produced a row, so that the outcome cannot be narrated
after the fact. Each question states the arms, the decision rule, and what would make the result
uninterpretable (a run that did not do the thing it is named after).

Reference for both: `results/2026-09-18/gguf_active_qwen_olmoe.json` (claims `qwen3_active_mb_per_token`,
`olmoe_active_mb_per_token`) and the best measured streaming cell so far,
`results/2026-09-17/bmoe_cache.json` cell `ceil5000` (claims `bcache_ceil5000_decode`,
`bcache_ceil5000_hit`, `bcache_ceil5000_read`).

---

## Q1 — Does an oversized, ZRAM-backed expert cache beat one sized to MemAvailable?

`device/bmoe_zram.sh`. Arms: `base5000` (the current best budget), `zram8000`, `zram10000`, all with
`--force-cache`, 3 rotated repeats, everything else identical to the `bmoe_cache.sh` flag string.

**Mechanism under test.** The budget is currently capped so the cache fits in RAM, which caps the hit
rate; the remaining misses are UFS reads issued by the I/O lanes and are partly hidden by `--overlap`.
An oversized cache converts some of those misses into anonymous-page faults served by ZRAM. Q4_0 pages
are near-incompressible, and zram stores an incompressible page raw, so the transfer is RAM-to-RAM.

**Predictions, in the order of how much they would change the project:**

| outcome | what it means | decision |
|---|---|---|
| `zram*` median decode **> base5000** by more than the base arm's own spread | swap faults are cheaper than the flash reads they replace; RAM is no longer the cache ceiling | raise the budget, re-measure the hit-rate curve, and re-derive the ceiling with the new miss cost |
| `zram*` **≈ base5000** (within spread) | the two costs cancel: cheaper bytes, but paid synchronously in the compute thread instead of by the I/O lanes | report as a negative; the cache ceiling stays MemAvailable |
| `zram*` **< base5000** | the synchronous fault plus compression work costs more than a lane-issued UFS read | report as a negative and stop pursuing cache size on this device |
| `zram*` rows fail (LMK kill, non-zero exit) | the configuration is not runnable, which is itself the answer at that budget | report the failure rate per arm, do not substitute a smaller budget and call it the same arm |

**What would make a row uninterpretable, and is therefore recorded per row:** `delta_pswpout` /
`delta_pswpin` from `/proc/vmstat` (a `zram*` row that did not swap is a `base*` row with a different
label), the engine's own granted budget (`moe-cache: budget`), `MemAvailable` before, `SwapFree` before
and after, and the exit status. `gates/bmoe_analyze.py --swap-log` attaches all of them.

**Stated in advance:** the ZRAM arms are expected to *lose*, on the reasoning that `--overlap` already
hides a large part of the flash cost while a page fault cannot be overlapped. The cell is run because
that reasoning predicts the sign, not the magnitude, and the hit-rate gain is large (the simulated
curve in `cache_qwen3_phone_sizes.json` rises steeply between 30% and 50% cache fraction).

---

## Q2 — Do two compute devices ADD weight-byte throughput, or share one bottleneck?

`host/agg_bandwidth.sh`. The same fully resident model (OLMoE) decodes on two devices at once and on
each alone: `solo_cpu`, `solo_gpu`, `solo_htp`, `pair_cpu_with_gpu`, `pair_cpu_with_htp`, 3 repeats.
Throughput of a run = active bytes per token × its tg rate; a pair's aggregate = the sum over its two
concurrent members.

**Why this question decides the project's direction.** A token's weight bytes are fixed, so the goal
rate is a required throughput. Every single-device throughput measured on this phone is far below the
SoC's DRAM peak. If devices add, the expert FFN can be split across them and the goal is reachable in
principle; if they share one bottleneck, no engine change reaches it and the honest deliverable is the
ceiling plus the reason.

| outcome | decision |
|---|---|
| aggregate of a pair **> best solo by more than the solo arms' spread** | devices add; design the split-FFN patch, and re-derive the target rate from the aggregate |
| aggregate **≈ best solo**, with each member's rate falling to roughly half its solo rate | one shared bottleneck (DRAM or fabric); the single-device ceiling is the real ceiling, and that is the finding |
| one member collapses while the other keeps its solo rate | not a bandwidth result but scheduling/priority; report as such and re-run with the roles swapped before concluding anything |

**Known weakness, stated now:** the two members' measurement windows overlap but are not identical
(model load and prefill differ per device), so a pair's aggregate is a **lower bound** on what the
hardware can deliver concurrently. That is the safe direction for concluding "they do not add" and the
unsafe direction for concluding "they do" — so a positive result will need the per-row start/end stamps
(written next to every row) to show the windows genuinely overlapped before it is reported as one.

**Also stated now:** `solo_cpu` runs in the `adb shell` domain while `solo_gpu`/`solo_htp` run in the
app process, because the DSP session opens only in an app. Domain is therefore confounded with device
in the *solo* comparison. It is not confounded in the *pair-vs-solo* comparison, which is the question
being asked, because each arm is compared against its own solo row in the same domain.

---

## Q3 — What can `--attn-device` show, and what can it not? (added before the A/B produced a row)

`host/bmoe_attn_ab.sh`, arms `attn_cpu` / `attn_htp` / `attn_gpu`, 3 rotated repeats.

**This A/B is underpowered for speed, by construction, and that is written down here rather than
discovered afterwards.** From the committed node trace (`results/2026-09-17/bmoe_trace`, the source of
`compute_trace.json`), the matmuls `--attn-device` can move — `Qcur`, `Kcur`, `Vcur` and the 48 unnamed
output projections — are **13.5% of decode node time**. The device that would take them is at best
around 1.3x the CPU on resident weights, so the predicted end-to-end change is **~3% of node time**,
against a base arm whose own three repeats spread 6.32 / 5.97 / 6.20 tok/s (≈3%). `SE/|θ| ≈ 1`: the
comparison cannot resolve its own effect, and adding repeats until it does would be tuning against
noise (CLAUDE.md §4.1).

**So the A/B is run for what it CAN establish, and only that:**

| question | how this A/B answers it |
|---|---|
| does the engine run at all with attention on another device, in an app process, with a 5 GB streamed expert cache? | the run completes, or it is a counted failure |
| is the placement lossless? | the generated text is compared against `attn_cpu`'s, character by character (`app_engine_analyze.py --text-compare`). A different device is not bit-identical, so exact equality is not required; the divergence point is reported, and an early divergence or incoherent text is a defect, not a rounding difference |
| does it catastrophically regress? | a large loss (well outside the spread) would be real and would kill the idea; a small win or loss will be reported as **not resolvable at this n**, with the n that would be required |
| where does the time actually go? | each row's own `compute / cache mgmt / flash I/O` split is recorded, so a change in the compute term can be seen even when the end-to-end rate cannot resolve it |

**The speed question is moved to the right instrument:** `host/matmul_sweep.sh` measures each shape on
each device directly, with enough iterations and enough rotated weight copies to be both precise and
DRAM-bound, and additionally reports the upload (repack) cost that decides whether *streamed* experts
could ever compute on the DSP. Predictions for it, on record now:

- attention shapes: the device is faster by a factor near the resident-model ratio (order 1.1–1.5x), and
  since they are only 13.5% of node time, no placement of them reaches the project's goal;
- expert shapes (`MUL_MAT_ID`, the majority of decode node time): the DSP's compute may well be faster,
  but `ggml_hexagon_is_repack_type` covers Q4_0/Q4_1, so every uploaded expert is repacked tiled on the
  CPU — the prediction is that `upload_over_compute` is large enough that the break-even hit rate
  exceeds 1, i.e. **streamed experts on the DSP never win**, and the honest route to the goal is the
  aggregate-bandwidth question (Q2), not this one.
