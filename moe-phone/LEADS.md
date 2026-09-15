# moe-phone — external work that bears on each failure mode

**Compiled 2026-09-14** from two literature sweeps. Provenance is marked on every row, because
`CLAUDE.md` §10.4 requires distinguishing *verified* from *not found*, and absence of evidence
decays:

| mark | meaning |
|---|---|
| **[V]** | source fetched and read this session |
| **[M]** | measured this session on a committed artifact in this repo |
| **[E]** | derived arithmetically from [V] or [M] inputs |
| **[U]** | single secondary source, or sources conflict — treat as unverified |
| **[N]** | not reached |

This file is a **lead list, not a results table.** Nothing here is a claim of this project.
Numbers attributed to other systems are theirs, measured on their hardware, at their
quantisation. Where one of our own measurements bears on the same quantity it is marked [M] and
the comparison is stated explicitly, because comparing across quantisation and device is the
error that produced a retraction here once already.

---

## 1. The strongest external check we have

**Revised 2026-09-16 — neither of the two checks this section used to lead with is evidence.**

- Our G1 **bulk-to-4 KB ratio of 8.9×** [M] matches PowerInfer-2's **8.89×** [V] to three
  figures, but that is a coincidence, not an instrument validation: the 4 KB cell has a 39%
  run-to-run spread (`g1_storage.json`), and the ratio depends on where the thread sweep stopped.
- An "independently reported **0.693** hit rate for a 512-expert top-10 model", cited as the one
  external check on S9, had **no recorded source** [N]. Deleted; S9 is pre-registered instead
  (`results/2026-09-16/s9_prereg_Qwen3-30B-A3B.json`).

The strongest external check now is **G-VALID-3** [E from V]: colibri's published rows, run
through our flash-only formula, are over-predicted by 2.8x (median) on disk-bound rows — the
formula is incomplete, which is a result about our model, not about theirs.

---

## 2. F4 — routing locality and cache policy

**Closed as a binding failure mode** (see `POSITION.md` §3c). What the literature adds:

- **Local Routing Consistency of MoE Models** (arXiv 2505.16056) [V] measures 20 MoE LLMs on two
  metrics. Segment routing best-F1 at segment length 16: **Qwen3-30B-A3B 55.90, OLMoE-1B-7B
  50.17, Mixtral-8x7B 49.27**, DeepSeek-V2-Lite 37.79, Qwen1.5-MoE 30.88.
  **Two structural findings that should drive model selection:** models applying MoE on *every*
  layer with *no shared experts* have the highest routing consistency; shared experts and a dense
  first layer reduce it. Their operating point — a cache of ≈2× the active expert count — is
  `rho ≈ 2`, which our own sweep also finds sufficient.
  *Why this matters to us:* it says our OLMoE trace is **mid-pack, not best-case**, so a result
  measured on OLMoE is not flattering itself.
- **LRU's collapse on MoE request streams has a published proof** (Angelopoulos et al., EURO-PAR
  2025, Thm 3.3) [U — cited by a sweep, primary not fetched]. Worth fetching before we describe
  our cyclic-scan argument as our own.
- **llama.cpp #24528** [V]: at *matched memory budget*, resident expert layers beat an expert
  cache **2.34× on prefill and 1.80× on decode**, even at an 85.3% hit rate. Caching is a
  fallback for weights that do not fit, not an accelerator for weights that do. Our byte budget
  already assumes this; it is worth stating in the paper.

## 3. F2 — bytes saved becoming time saved

- **Our G1 thread sweep stops at 8** [M]. At 4 KB the device scaled ×7.26 from 1→8 threads with
  **flat p50 latency** (86.7 → 93.0 µs), i.e. small reads are latency-bound and had **not
  saturated** at the top of our sweep. At 1 MB the scaling is only ×1.81 and at 4 MB ×1.24, so
  whole-expert reads are already near the ceiling. **Re-read 2026-09-16:** the bulk cells are flat
  from 4 threads and fall at 2–4 MB from 4 to 8, which looks like a device ceiling rather than a
  queue-depth limit. The 16/32-thread run is queued because it is cheap, not because it is
  expected to move the constant.
- **EStream** (arXiv 2609.06551, 8 days old at time of writing) [V] runs on a **OnePlus 15 with
  UFS 4.1** — our device class — and explicitly names **MoE decode streaming as future work**.
  Closest neighbour; read fully before claiming novelty.
- **Budgeting Bytes** (arXiv 2609.04238) [V] proposes a "windowed storage roofline" treating
  bytes-per-token as a first-class design axis, on an 8 GB edge board running Qwen3-30B-A3B from
  eMMC. **This is a direct novelty threat to G-ROOF itself.** It contains no mention of
  speculative decoding, batching, or multi-token verification — single-token autoregressive only.
  That absence is the opening, and it is narrow.

## 4. F1 / A1 — sparsity, and the criterion that gates it

Already folded into stub **S8**, which blocks generalising the A1 kill. The load-bearing points:

- Across **32 activation-sparsity papers checked by full-text grep, none reports KL divergence or
  top-1 flip rate** [V]. The field's bar is perplexity or benchmark average. Our G3 result is a
  *measurement-sensitivity* finding, not a contradiction.
- 2605.08575's headline **90% is Llama-4-Maverick**; its **OLMoE-1B-7B cutoff is 71.2%** [V].
- **TEAL** (2408.14690) §5.4.1 [V] tested gate-first head-to-head and measured **~1.6× the
  activation error** at matched sparsity, and runs **60%** intermediate sparsity using
  `|h| = |SiLU(g)·u|` where we thresholded `|SiLU(g)|`. **G3 may have killed a criterion, not the
  lever.**
- **ACE** (arXiv 2609.05228) [V]: **50% expert skipping, training-free *and* calibration-free.** A
  different axis from intra-expert neuron sparsity — it drops whole experts from top-k, so it
  composes with everything in `ARCHITECTURE.md` and is directly testable against our
  pre-registered Q4_0 margin. Caveat: its +4.15 pp / −7.96% perplexity figures are against *other
  skipping methods*, not against the full model.

## 5. Speculative decoding — the ceiling, and the one measurement that surprised us

- **Measured on our own committed trace** [M]: adjacent-token expert overlap **0.366** vs **0.125**
  under independence — **2.93× chance**. Cohere reports **0.381 vs 0.118 (3.2×)** on a
  128-expert top-8 model [V]. Different model, different expert count, same answer.
- **But the decomposition matters and cuts against the intuition** [M]. Against a shuffled control
  (popularity preserved, order destroyed), most of the union saving at large windows is
  **birthday collision plus expert-popularity skew**, not temporal correlation: at W=16, 67.4%
  total sub-linearity of which only **6.6 points** come from temporal correlation.
  *Good:* the saving is robust to a bad drafter and survives tree branching.
  *Bad:* it is hard-bounded by `E/k` — you cannot beat "read the whole layer once".
- **The design rule** [E]: the hard ceiling is `gain = M·k/E` for `M` tokens emitted per pass, so
  break-even needs `M = E/k` accepted tokens — 4 for Mixtral, 8 for OLMoE, 16 for Qwen3-30B-A3B,
  **51 for Qwen3-Next-80B-A3B**.
  > **This inverts our model preference.** `POSITION.md` §3 favours fine-grained MoE because it
  > touches fewer bytes per token. But fine-grained = high `E/k` = the regime where verification
  > amortisation is worth *least*. The family where our byte budget is worst is the family
  > speculation rescues most. Reportable and falsifiable.
- **The strongest negative, which must be cited** [V]: a 19-configuration llama.cpp sweep on
  Qwen3.6-35B-A3B + RTX 3090. Baseline 139.9 tok/s; ngram-cache 119.1 (−15%); worst 65.0 (−54%).
  **All configurations showed 100% draft acceptance and none was faster**, attributed to expert
  saturation at ~94 tokens ≫ realistic K.
- **Composition, measured in a real engine** [V]: on GLM-5.2, expert caching alone **+21.6%**, MTP
  alone **+28%**, stacked **+51%** (13.92 → 29.35 tok/s). Super-additive.

## 6. llama.cpp PR #25294 — the engineering wedge, and three hazards

The PR (`llama: stream MoE routed experts from disk`, open) [V] **already implements the union
mechanism — for prefill only**: *"when a ubatch touches more experts than the cache holds, the
expert GEMMs run in W waves … each touched expert is loaded once per ubatch."* A speculative
verification batch *is* a ubatch. **The path exists and has never been pointed at decode-time
verification.** That is the concrete engineering wedge.

| hazard | what it means for us |
|---|---|
| **single context** [V] — concurrent decode of multiple `llama_context` from one streamed model shares a cache and can corrupt output; `--parallel N` within a single context is safe | llama.cpp's classic `llama-speculative` uses a *second context* for the draft. **A separate-model drafter is unsafe against this PR; a same-context self-draft or MTP head is safe.** The single most important compatibility fact found. |
| **slot sizing** [V] — auto cache sizing aborts for `n_expert_used ≥ 6`; multi-pass wave sizing needs `≥ 3×n_expert_used` dynamic slots | verification needs **union-sized**, not `3k`-sized, slots |
| **parity** [V] — a contributor reported byte-identical streamed-vs-resident output, then **retracted it**; parity degrades with generation length (~1 in 15 runs match at 96 tokens), consistent with FP accumulation differing across multi-pass wave GEMMs | **Tier-E (exact) may not be achievable through the waved path.** Know this before pre-registering a Tier-E claim. |

## 7. Prior art closest to us — read before any novelty claim

1. **DraftExpert** (arXiv 2607.24434) [V] — self-speculative MoE decode on **Snapdragon 8 Elite,
   Hexagon HTP v81, Q4_0, llama.cpp, experts flash-resident**. Our device class, our NPU, our
   runtime, our storage tier. Flash→NPU 10.18 → 15.47 TPS (1.52×). Acceptance 84–87%.
   **Its Table 4 is the number we most need:** verification cost at K=4 relative to one AR step is
   **3.10×** with fixed K, **1.70×** with truncation, **1.35×** with prefetch. (3.10/4 = 0.775
   bytes/token → 1.29×, independently consistent with our own union measurement [M].)
2. **S2-MoE** (arXiv 2608.15018) [V] — self-speculative MoE on edge devices, built on llama.cpp
   with SSD expert offloading, **code public**, models include OLMoE and Qwen3-30B-A3B. Already
   does union-aware candidate admission.
3. **AcceptMoE** (2608.02989) [V] — H2D 2394 → 633 MiB/token (−73.6%), 2.06× under offloading —
   **but explicitly not distribution-preserving**: *"an approximation rather than a
   distribution-preserving optimization"*. **Tier-A only, never Tier-E.**
4. **EVICT** (2605.00342) [V] — training-free, hyperparameter-free, **lossless** draft-tree
   truncation. The lossless counterpart to AcceptMoE and the correct first thing to implement if
   we tree at all.
5. Also verified present: EcoSpec, MoE-Spec, MoE-SpAc, SpecMoE (DAC'26), ELMoE-3D, EdgeXpert,
   MoESD, MoE-SpeQ, SpecMoEOff, SP-MoE, Utility-Driven SD [all V].

6. **colibri** ([github.com/JustVugg/colibri](https://github.com/JustVugg/colibri), Apache-2.0)
   [V, read in full at commit a8f2ca6 on 2026-09-16] — open C engine streaming MoE experts from
   disk: per-layer LRU, learned pinned hot store, one-layer-ahead router prefetch, io_uring /
   O_DIRECT read path for its GLM engine, CUDA/Metal/Vulkan tiers, and a table of community
   measurements (our G-VALID-3 input). Its OLMoE engine reads synchronously at queue depth 1 and
   keeps dense weights and KV in fp32 (~1.8 GB dense), so it is **not** a phone base as it stands.
   Its hit rate counts every lookup (OLMoE) or every distinct expert per forward (GLM) and
   prefetch loads never count as misses, so its hit rates are not ours; its `[PROF]` "GB fetched"
   is the comparable quantity. Its "71.6%" next-layer recall has no recorded provenance and a code
   comment beside it says 75.8% [U].
7. Found 2026-09-16 by a research subagent [V by the subagent, numbers not re-checked]: **mllm /
   EdgeMoE** (Android, QNN + OpenCL, expert streaming from storage — the closest shipping peer);
   **llama.cpp discussion #27149** (expert-aware SSD streaming, Qwen3-30B-A3B, reports 4.10 tok/s
   at an 88% hit rate on a laptop); **slipstream** (Apple M5, per-layer LFU-with-decay, reports
   bytes read per token); **scale-snu/SSD-offloading** (an analytical SSD cost model — read its
   functional form before finalising ours); **FlashMoE** (2601.17063, learned replacement, paper
   only). Each is a candidate held-out validation point once its conditions are checked.

**Design constraint worth remembering** [V]: per AcceptMoE, *"a wide shallow tree tends to select
overlapping expert sets, yielding smaller unions; a deep tree can activate many distinct experts
despite fewer total tokens."* **If we tree at all, go wide and shallow.** This is the opposite of
the intuition.

## 8. Do not pursue

- **Expert prefetch/prediction as a bandwidth saver.** Three independent negatives [V]:
  *Budgeting Bytes* — on an RK3588 + eMMC, temporal-locality prefetch is **net-negative** (0.19 →
  0.12 tok/s), and a **trace-driven oracle prefetching perfectly one token ahead** gives only
  0.12 → 0.13; *"prefetch changes when and how bytes are read, never how many"*. *WiSP* — *"does
  not help in single-stream decode: the bottleneck is bandwidth, not prediction quality"*.
  llama.cpp #24528 — offline replay with near-perfect prediction caps at **+12.8%**.
  **This is a §4.3 oracle result and it should discipline how much we invest here.** Note it does
  *not* rule out prediction used for **eviction**, which is a different use of the same signal —
  see `ARCHITECTURE.md` §4, where we measure the break-even accuracy rather than assume it.
- **Lookahead Decoding / wide n-gram pools** [E] — feeds 100+ tokens per pass while emitting ~2;
  the union saturates to the whole layer. Structurally wrong for a byte-bound machine.
- **XShare in-batch expert sharing** [V] — explicitly N/A at batch size 1.

## 9. Not reached [N]

Union curves for any model but OLMoE; union growth for **tree-shaped** candidate sets and for
**actual draft tokens including rejected ones** (our proxy is ground-truth consecutive tokens,
which is optimistic); whether Qwen3-Next-80B's NextN block carries its own 512-expert MoE layer
(the cached `config.json` has no `num_nextn_predict_layers`, contradicting one secondary source
[U]); expert **output** caching (caching FFN outputs rather than weights) — searched, essentially
nothing found, plausibly a genuine gap but **not verified absent**; whether the 15R's UFS scales
past queue depth 8.
