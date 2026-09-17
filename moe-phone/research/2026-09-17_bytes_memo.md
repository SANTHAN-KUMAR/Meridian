# Cutting flash bytes/token or getting >1 token per expert fetch, no training — research memo (2026-09-17)

*AI research subagent output (Claude Opus, web search). Provenance [U] per `research/README.md`;
the memo's own "not verified" marks are kept. Reproduced verbatim except for formatting.*

Evidence class per item: **(a) demonstrated** (numbers in a cited source), **(b) theoretical**
(arithmetic shown), **(c) untested**. "Not verified" marks anything I could not confirm from primary
text.

## 1. Lossless compression of already-quantised expert weights

**What is demonstrated, and on what.** All strong published numbers are for *16-bit or 8-bit*
formats, not Q4:

- **DFloat11** (arXiv 2504.11651, NeurIPS'25, github.com/LeanModels/DFloat11): Huffman-codes the
  **BF16 exponent field only**, sign+mantissa stored raw. ~30% size reduction, bit-identical outputs,
  custom **GPU** kernel with SRAM-resident hierarchical LUTs. Not applicable to Q4_0 — there is no
  exponent field.
- **ZipNN** (arXiv 2411.05239, IEEE CLOUD'25): ~33% on BF16, >55% on "clean" FP32, same
  exponent-skew mechanism.
- **On the Compressibility of Quantized LLMs** (arXiv 2403.01384) — the closest measurement. INT8
  weights: **tensor-wise SmoothQuant 1.66–2.22× (zstd), channel-wise LLM.int8() only 1.11–1.31×**.
  Decompression 2300 MB/s (zstd), 1350 MB/s (Huffman), 440 MB/s (FSE). **The paper does not test
  4-bit.** The mechanism it identifies is decisive for us: *finer-grained scaling destroys
  compressibility*, because per-group rescaling re-spreads each group's codes over the full code
  range. Q4_0 (per-32 block) and Q4_K (per-256 superblock, 6-bit sub-scales) are the extreme
  fine-grained case — strictly worse than channel-wise INT8's 1.1–1.3×.
- **Approaching Shannon Bound** (arXiv 2606.15789, ISCA'26, NUS/ETH): claims effective entropy
  "2–10× lower" than stored bitwidth across bf16→int4/AWQ/SQ8, ANS decode within 0.01–0.1 bits of
  Shannon, Mixtral-176B batch 20→95. **I could not verify its int4 number**: one fetch of the paper
  reported "int4 retains only 0.6–1.0 bits of true entropy", a second fetch of the same paper's
  author-hosted PDF reported "~3.5–3.8 bits of actual entropy". These are incompatible. **Not
  verified — treat the int4 claim as unestablished.** The decode is warp-cooperative rANS on CUDA;
  the authors give no CPU/ARM path.
- **EntroLLM** (arXiv 2505.02380): claims uint4 → **1.39–1.62 effective bits** (65% reduction),
  decode on Jetson ARM Cortex-A57, 1.66 s for the uint4 model, 31.9–146.6% throughput gain. **I am
  sceptical and flag it**: a 1.4-bit entropy on a 4-bit code means ~85% of weights share ~2 values,
  which is a property of their particular mixed-precision quantiser, not of a max-normalised Q4
  nibble. Do not port this number to GGUF.

**Arithmetic for Q4_0 specifically (b).** Q4_0 = 32 weights, one fp16 scale `d = max|w|/(-8)`,
nibbles `q-8 ∈ [-8,7]`. For 32 approximately-Gaussian samples, `E[max]/σ ≈ 2.2`, so the quantised
code has `sd ≈ 8/2.2 ≈ 3.6` on a 16-symbol support — ratio 3.6/15 = 0.24 versus uniform's 0.289.
Discrete entropy is then **≈3.8–3.9 of 4 bits: 2–5% savings on the nibbles.** The fp16 scales are
16 bits per 32 weights = 0.5 bits/weight of the 4.5 bits/weight total, and *those* are
DFloat11-compressible by ~30% → **0.15 bits/weight ≈ 3.3% of the file**. Total realistic Q4_0
lossless gain: **3–7%, below your 10% bar.** Q4_K is slightly better (min-offset nibbles are more
skewed, 6-bit scales already dense): plausibly 8–12%, still marginal, and you pay ~1.3 GB/s Huffman
decode against a UFS4 read you are already only 50 ms/token stalled on.

**Expert-to-expert delta/dedup.** Everything published is lossy and needs SVD/Fisher calibration:
D²-MoE (arXiv 2502.17298, ICML'25, github.com/lliai/D2MoE, base+delta, 40–60% compression), MoBE
(2508.05257), PuzzleMoE (2511.04805, training-free but merges weights), Camera (2508.02322). **No
published lossless XOR/delta coding of Q4 expert nibbles exists.** Arithmetic (b): a lossless delta
only helps if paired experts' nibbles agree better than chance; with per-expert independent scales,
agreement is ~1/16 and the XOR residual has *higher* entropy than the original. Dead unless experts
share a scale, which GGUF does not do.

**Status: demonstrated (for BF16/INT8) / theoretical (for Q4).** Expected effect on bytes/token:
**3–7% for Q4_0, 8–12% optimistic for Q4_K**, at a decode cost of ~1 ARM core. Experiment that
settles it: compute the empirical symbol histogram and Shannon entropy of the nibble stream and of
the scale stream in your actual Qwen3-30B-A3B Q4_0 GGUF expert tensors, and report bits/weight
versus 4.5. That is a one-pass measurement with no model run, and it is a **kill experiment** — if
nibble entropy > 3.8 bits, close the whole line.

## 2. Built-in MTP heads for lossless self-speculation

**Which checkpoints ship a usable head (a).**
- **Qwen3-Next-80B-A3B** (Instruct/Thinking, HF): 48 layers, 512 experts, hybrid
  Gated-DeltaNet/Gated-Attention, ships MTP (`mtp_num_layers`); vLLM exposes it as
  `{"method":"qwen3_next_mtp"}`. FastMTP (arXiv 2509.18362) measures its *native* head's
  per-position acceptance on Qwen3-Next-80B-A3B-Instruct: **pos0 0.897, pos1 0.719, pos2 0.476**
  (untrained head; their fine-tuned version reaches 0.912/0.776/0.616 — that variant requires
  training, so it is out for you, but **the 0.897/0.719/0.476 figures are the stock-checkpoint
  numbers you can use**).
- **GLM-4.5-Air (106B-A12B)**: `num_nextn_predict_layers: 1`, one MoE MTP layer (arXiv 2508.06471).
- **DeepSeek-V3 style**: 1 MTP head, reported ~85–90% next-token acceptance, ~1.8× throughput (arXiv
  2412.19437).
- **gpt-oss-120b**: no MTP head found. **Qwen3-30B-A3B (your current model): no MTP head** —
  standard GGUFs do not contain prediction heads; MTP-converted variants exist only for newer
  Qwen3.6/3.8.

**Engine support (a).** MTP is **merged in llama.cpp**, PR ggml-org/llama.cpp#22673 (`llama + spec:
MTP Support`, am17an), merged 2026-05-16, tested on Qwen3.6-27B and Qwen3.6-35B-A3B; reported
**~75% steady-state acceptance at 3 draft tokens, >2× speedup**; community 38→65 tok/s on a 5090.
MoE disk streaming is also upstream (PR #25294, `--moe-stream`, `--moe-stream-cache`,
`--moe-stream-io-threads`). There is a known post-merge bug (xhinker, Medium). Qwen3-Next-80B GGUFs
exist (unsloth, lmstudio-community) but I found **no confirmation that the Qwen3-Next MTP head
survives GGUF conversion — verify before planning on it.**

**The number you asked for — union-expert growth per verified position (a).** colibrì
(github.com/uv-genai/colibri, GLM-5.2 744B, experts streamed from disk) reports: **int8 MTP head →
39–59% draft acceptance, 2.2–2.8 tokens/forward**; expert loads per forward rise from **~660 (MTP
off) to ~1100 (MTP on) on a cold cache**, i.e. **1.67× bytes for 2.2–2.8× tokens ⇒ 0.60–0.76×
bytes per accepted token (a 24–40% cut)**. It uses **batch-union MoE**: each unique expert in the
verify batch is read once and applied to every position routing to it. It also warns MTP is a *net
time loss* until the cache warms, and that int4 heads collapse to 0–4% acceptance. Honest caveat
from their docs: batched verify is "lossless in exact arithmetic" but **not byte-identical** to
single-token greedy, because quantised kernels round shape-dependently — that is a **Tier E
violation** you would have to check on your own kernels.

**Arithmetic for your design (b).** Qwen3-30B: E=128, k=8, E/k=16. Measured adjacent-token overlap
≈3× chance → shared ≈ 3·(8/128)·8 ≈ 1.5 experts → **union(2 positions) ≈ 14.5 = 1.81×**. Break-even
on bytes needs mean accepted length > 1.81, i.e. single-draft acceptance > 81%. With your measured
verify cost c(2)=1.71, break-even on *time* needs accepted length > 1.71. DeepSeek/llama.cpp
acceptance (0.75–0.90) puts mean accepted length at 1.75–1.90 — **straddling both break-evens.**
This is why CPU-side MTP is marginal for you and colibrì still wins: their 11 GB/token I/O makes
bytes dominate, yours does not.

**Would GPU verify change it?** Yes, and this is the decisive point. Your own measurement (commit
db11ba1) says the Adreno GPU already beats the throttled CPU. c(N) is near-linear on the CPU because
the CPU is compute-bound at 100 ms/token; a batched verify of N positions is one GEMM of N rows,
which on a GPU is nearly free relative to N=1. If GPU verify gives c(3) ≈ 1.1–1.3 instead of 2.37,
MTP pays on time at any acceptance above ~25%, and the byte question becomes the only one. **(b),
and it is the highest-value untested lever here.**

*(Main-session note, not the memo's: `HEADROOM.md` §1.7 and §4.4 correct the last paragraph — at
rho 4 the union of 2 positions fetches 1.90× what one token fetches on the committed traces, so
the I/O side alone sets the break-even, and "any acceptance above ~25%" does not hold.)*

**Status: demonstrated (acceptance rates, engine support, colibrì's 660→1100/2.2–2.8) / theoretical
(that it pays on *your* device).** Expected effect: **0.6–0.76× bytes per accepted token; 1.3–2×
tok/s if and only if the batched verify runs on the Adreno.** Experiment that settles it: on the
15R, measure c(2), c(3), c(5) for a batched verify executed on the GPU backend (same protocol that
produced 1.71/2.37/3.71 on the CPU), and separately measure the union-of-experts count at 2 and 3
positions from your existing routing traces. Both are offline/cheap. Then pick a model that
actually has a head (Qwen3-Next-80B-A3B or GLM-4.5-Air), after confirming GGUF carries it.

## 3. Exact next-token lookahead from the model itself

**Architecture facts (a).** Verified from configs: **Qwen3-30B-A3B** — `num_hidden_layers: 48`,
`num_experts: 128`, `num_experts_per_tok: 8`, `decoder_sparse_step: 1`, `mlp_only_layers: []` →
**layer 0 is an MoE layer, no dense prefix**. **OLMoE**: `first_k_dense_replace: 0`. By contrast
**DeepSeek-V3 / GLM-4.5**: 3 dense layers before the first router (Raschka, *The Big LLM
Architecture Comparison*); DeepSeekMoE and Moonlight: 1.

**But no architecture puts a router before attention.** Every one of these is a standard pre-norm
block: attention then MLP/MoE. So layer-0 routing is *not* a function of token id alone. **Your
premise 3 is false as stated.**

**However (b), the useful version survives.** The layer-0 router input for a candidate next token
`x` is `h = LN( e(x) + Attn₀( LN(e(x)), KV₀^{≤t} ) )`, and `KV₀^{≤t}` is already in cache. Given a
candidate token id, **every term is known**, so the layer-0 (and, with one MoE execution, layer-1)
expert set for that candidate is computable **exactly**, at a cost of one attention row plus one
2048×128 router GEMV — microseconds, zero flash I/O. Evaluated over the top-M candidates of the
current step's logits you get an exact, weighted distribution over next-token layer-0 experts. This
is **Tier E by construction**: it changes *when* bytes are fetched, never which experts execute.

**Ceiling (b).** It covers layer 0 of 48 exactly, layer 1 approximately. That is ~2–4% of
per-token expert traffic — as a *byte* reduction it is negligible. Its value is **latency hiding**:
it extends prefetch depth across the token boundary, which is exactly where your 50 ms/token I/O
stall sits. Closest published work is one layer *within* a token, not across tokens: PILOT / FATE
(arXiv 2502.12224) report **78.8% plain top-k coverage, 97.15% with confidence-expanded candidates,
training-free, negligible CPU cost**; YALIS (2603.19289) **83–90% hit rate on Qwen3 and GPT-OSS**;
colibrì issue #200 improves PILOT from **73.6% → 76.7% recall** on GLM-5.2 over 13,616 samples by
adding the shared expert's output to the residual before running the next router (3 small GEMMs,
zero disk I/O). **I found no publication doing the cross-token exact-layer-0 version.**

**Status: theoretical, untested.** Expected effect: ~2–4% of bytes made prefetchable one token
early, at 100% precision; main payoff is hiding part of the 50 ms stall. Experiment: offline, from
traces — for each step, compute layer-0 routing for the top-M actual candidates and measure (i) the
size of the union over M, (ii) the fraction of the realised layer-0 experts covered as a function
of M.

## 4. Token-identity-conditioned routing statistics

**The one direct measurement I found (a).** arXiv 2604.17182, *Layer-wise MoE Routing Locality
under Shared-Prefix Code Generation*, on **Qwen3.5-35B-A3B-FP8** (256 experts, top-8). Jaccard
similarity of expert sets:

| | layer 0 | layer 14 | layer 38 | mean |
|---|---|---|---|---|
| same token id, different context | **0.828** | 0.618 | 0.616 | **0.649** |
| different token id | 0.087 | 0.223 | 0.207 | 0.179 |

So token identity alone pins **~83% of layer-0 routing and ~65% averaged over layers**, while
context contributes most in the middle layers (L14–20). Supporting but weaker: Mixtral routing is
driven more by syntax than domain, especially in first and last layers; Qwen3-30B-A3B shows the same
token taking distinct trajectories across 48 layers by context (arXiv 2604.09780 argues routing
reflects hidden-state *geometry*, not domain).

**No paper reports per-layer mutual information between token id and expert choice.** That
measurement does not exist in the literature as of today. Note the numbers above are Jaccard under
a *shared prefix*, which inflates them relative to your setting.

**Implication for your idea (b).** An n-gram draft (your measured ~30% acceptance on free text) plus
a token-id→expert table built offline from traces gives a multi-token eviction hint at **zero I/O
and zero training**. Expected quality: at Jaccard 0.65 mean, a hint of size ~k/0.65 ≈ 12 experts
per layer covers most of the true 8. Used only as an eviction *priority* it cannot damage fidelity
at all (Tier E). Used as a prefetch trigger it costs bytes on misses.

**Status: demonstrated (the routing statistic) / theoretical (that it improves your 78% hit
rate).** Expected effect: bounded by your cache-miss rate; at best it converts some of the 22%
misses into hits — plausibly a few percent of bytes/token, and **it cannot hurt fidelity**.
Experiment: from existing traces build the table, then replay your cache simulator with the table
as an eviction-priority term and report hit rate versus the LRU baseline. Pure offline; no device
time.

## 5. Calibration-free expert skipping/merging measured in KL

**The honest finding: nothing in this literature reports KL or any logit-level fidelity.** I checked
every 2025–2026 candidate:

- **ACE** (arXiv 2609.05228, calibration-free, checkpoint-preserving, token-adaptive): evaluated on
  **Qwen3-30B-A3B-Instruct-2507**, Qwen3.6-35B-A3B, Gemma-4-26B-A4B. At 50% skipping on
  Qwen3.6-35B-A3B: WikiText-2 PPL **8.67** vs MoDES 9.42; avg downstream accuracy **75.57%** vs
  71.42%. At 60% on Qwen3-30B: +1.07/+1.66 accuracy points. **Metrics are perplexity and 7
  downstream accuracies only — no KL, no top-1 flip rate.**
- **MoDES** (arXiv 2511.15690): training-free, globally-modulated local gating + dual-modality
  thresholds; needs a frontier search over thresholds. PPL/accuracy only.
- **Elbow-based MoE Routing** (arXiv 2608.04401): training-free inference-time dynamic-k, picks the
  elbow of the sorted router-probability curve. Closest to your "dynamic top-k by router
  confidence". No KL reported.
- **Training-Free Halving of Activated Experts in Fine-Grained MoE** (arXiv 2609.04575): could not
  extract results from the PDF. **Not verified.**
- **Router calibration is necessary** (arXiv 2603.02217) argues retraining-free MoE compression
  *needs* router calibration — a direct warning against your no-calibration constraint.
- Merging: PuzzleMoE (2511.04805, training-free dual-mask), Sub-MoE, HC-SMoE, MEO (online merge at
  inference), SERE (arXiv 2602.07616, similarity-based re-routing, up to **2.0×**, precomputed
  similarity matrix from a general calibration set, no retraining). All report task metrics, none
  report KL.

So the entire field is invisible at your fidelity tier. Your gate-first neuron-sparsity result (KL
0.18 vs Q4_0's 0.05 at density 0.5 on OLMoE) is, as far as I can tell, **the only KL-level
measurement of intra-expert sparsity in existence** — that is a publishable negative result, and the
same instrument applied to ACE/Elbow would be a second one.

**Status: demonstrated at the perplexity level, untested at the KL level.** Expected effect on
bytes/token: 50% expert skipping ⇒ ~50% of expert bytes, i.e. potentially ~97 MiB/token — the
largest single lever available, *if* it survives your margin, which nobody has checked. Experiment:
run ACE and Elbow dynamic-k on Qwen3-30B-A3B and report KL to bf16 and top-1 flip rate against the
Q4_0 margin, sweeping skip fraction. This is the highest-information experiment in the memo and it
reuses the harness you already built for the neuron-sparsity result.

## 6. Unconventional

- **Expert *output* caching across tokens with an input-similarity threshold: nothing exists for LLM
  decode.** MoECa (2606.15615) is diffusion-transformer timestep caching, not autoregressive. The
  nearest is a "Consecutive Tokens Pattern" observation — at least one expert per layer is reused
  across adjacent tokens with **40–60% probability** — and ReMoE (ICML'26), which raises
  adjacent-token expert reuse ~26% and cuts decode 43.6–49.8% but does it by **router fine-tuning**
  (disqualified). **(c) untested; confirms your earlier negative search.**
- **Routing-aware batching across concurrent sessions**: real and measured, but for servers.
  **Opportunistic Expert Activation** (arXiv 2511.02237, Dao et al.): batch-aware routing that lets
  tokens piggyback experts already loaded for other tokens in the batch, **no retraining**,
  evaluated on Qwen3-30B and Qwen3-235B at batch 16 — but it reports perplexity/MATH-500/GPQA, **not
  KL**, and it *changes routing*, so it is lossy. ELDR (2607.00466) groups expert-similar requests
  into a micro-batch, cutting distinct active experts per step (up to 20% less all-to-all). **On a
  phone at batch size 1 these are inapplicable**; they only matter if you commit to a multi-session
  chat server on-device.
- **Second-cluster resident small model as a zero-training proxy router: nothing exists.**
  Self-Routing (arXiv 2604.00421) shows routing can come from a hidden-state subspace with no router
  projection, and arXiv 2604.09780 argues hidden-state similarity is necessary *and sufficient* to
  explain expert usage — both make a proxy plausible in principle, but neither builds one, and any
  cross-model proxy needs a learned map between two different representation spaces, i.e. training.
  **(c) untested; report as nothing found.**
- **Worth knowing**: the batch-union-MoE trick in colibrì (read each unique expert once per verify
  batch) is the enabling mechanism for §2 and is already implemented in a public C engine you can
  read.

**Status: untested / not applicable.** Expected effect: expert-output caching unquantified;
routing-aware batching ~0 at batch 1. Experiment that settles expert-output caching: from your
traces, measure the distribution of cosine similarity between successive inputs to the *same*
expert at the same layer, and the resulting output error — if inputs are not >0.99 similar, close
it immediately.

## Ranked top-5 testable interventions

1. **MTP head + batched verify on the Adreno GPU, on a checkpoint that ships a head**
   (Qwen3-Next-80B-A3B or GLM-4.5-Air). *(a) for acceptance rates and for the 660→1100-experts /
   2.2–2.8 tok-per-forward byte accounting (colibrì); (b) for whether it pays on your device.* Only
   lever with a demonstrated >1 token per expert fetch. Gate: measure GPU c(2)/c(3) against the
   CPU's 1.71/2.37, and union-of-experts at 2–3 positions offline. Watch the Tier-E hazard: batched
   verify is not byte-identical under quantised kernels.
2. **KL-instrument ACE and Elbow dynamic-top-k on Qwen3-30B-A3B.** *(a) at perplexity, (c) at KL.*
   Largest possible byte cut (~50%) and the literature is silent exactly where your criterion
   lives; you already own the measuring instrument. Either a big win or a second clean negative
   result.
3. **Nibble/scale entropy measurement on your Q4_0 GGUF expert tensors.** *(b), with (a) evidence
   only for adjacent formats.* One pass over a file, no model run. My arithmetic predicts 3–7%
   total and I expect this to kill the line; published >10% claims for int4 are mutually
   contradictory (0.6–1.0 vs 3.5–3.8 bits from the same paper) and none are on GGUF.
4. **Token-id lexical routing table + n-gram draft as a zero-I/O eviction hint.** *(a) for the
   statistic (same-token Jaccard 0.828 at layer 0, 0.649 mean, Qwen3.5-35B-A3B), (c) for the cache
   gain.* Cannot damage fidelity; costs only offline replay of traces you already have.
5. **Exact cross-token layer-0 router lookahead over top-M next-token candidates.** *(c) untested;
   no prior art found for the cross-token version.* Tier E by construction, ~2–4% of bytes made
   prefetchable a full token early — small on bytes, aimed at the 50 ms stall. Also the cheapest
   thing on this list to implement.

**Not worth pursuing**: lossless delta/dedup coding across Q4 experts (per-expert scales destroy
the correlation), expert-output caching (no evidence, and nothing found in two independent
searches), routing-aware multi-session batching (batch 1 on a phone), any proxy-router scheme
(requires training a cross-space map).

**Claims I could not verify**: the int4 entropy figure in arXiv 2606.15789; EntroLLM's uint4 →
1.39–1.62 bits; whether Qwen3-Next's MTP head survives GGUF conversion; the results section of
arXiv 2609.04575. Each is flagged inline.
