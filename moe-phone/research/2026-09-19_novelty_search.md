# Novelty search — 2026-09-19

Protocol per `CLAUDE.md` §10.4: every query string below, the date run (2026-09-19), and a per-topic
verdict of **verified absent** (searched, and checked specific venues/repos/issue trackers, found
nothing), **not found** (WebSearch only, general queries, no targeted repo/venue check), or **found**
(with citations). This search used the `WebSearch` tool only (no full-text fetches of the hits below
beyond what the search snippets show); where a claim needs a primary-source read before it can be
cited in a paper, that is flagged. This supersedes nothing in `POSITION.md` §9 — it is additive, and
narrower (five specific sub-questions the project needs answered now), not a full re-run of that
broader record.

---

## (a) Flash-streamed MoE inference on phones — prior systems and their reported rates

**Queries run (2026-09-19):**
- `PowerInfer-2 mobile smartphone MoE flash offloading decode tok/s 2026`
- `on-device MoE inference flash streaming expert cache phone "tokens per second" 2026`
- `EdgeMoE AdapMoE HOBBIT Fiddler ktransformers MoE-Infinity phone Android decode tok/s benchmark`

**Verdict: found**, and it mostly confirms and dates the record `POSITION.md` §9 already built (searches
there ran 2026-09-14, -16, -17; this repeats the check two days later and finds no new phone-class system).

- **PowerInfer-2** ([arXiv 2406.06282](https://arxiv.org/pdf/2406.06282)): TurboSparse-Mixtral-47B,
  11.68 tok/s at 19 GB budget, 2.13 tok/s at 7 GB, on OnePlus 12 (24 GB) / OnePlus Ace 2 (16 GB), with
  root, `mlock`, and a retrained+predictor-augmented model. Matches `POSITION.md` §1's table; no new
  number found. Code still unreleased (project's own tracker: SJTU-IPADS/PowerInfer#207 unanswered).
- **BigMoeOnEdge** (this project's direct baseline, [github.com/Helldez/BigMoeOnEdge](https://github.com/Helldez/BigMoeOnEdge)):
  one search hit restates "up to 4.1 tok/s on Gemma-4-26B-A4B (17 GB) using pure CPU inference with 4
  CPU cores and phone flash" — a **different model** from this project's Qwen3-30B-A3B row
  (`bmoe_reproduced_decode`, 3.98 tok/s). Not a discrepancy to resolve, just a reminder the repo reports
  multiple model rows; the project's own reproduction (its documented Qwen3-30B-A3B command) is the
  number to cite for that model, not this search snippet.
- **BigMoMo** ([arXiv 2609.14643](https://arxiv.org/html/2609.14643), 2026-09-13, already logged in
  `POSITION.md` 2026-09-17 evening entry) resurfaced in this search too — closest prior art, 4 days old
  at last check, still the closest. No newer prior art displaced it in this pass.
- **EStream** ([arXiv 2609.06551](https://arxiv.org/html/2609.06551)) — mobile-NPU MoE **prefill**
  virtualization; already logged as prefill-only in `POSITION.md`.
- **EdgeMoE** ([arXiv 2308.14352](https://arxiv.org/pdf/2308.14352)) — tested on MiniCPM MoE 8x2B on a
  Xiaomi 14 phone. This is a genuinely phone-class MoE result not previously named with a device in this
  project's record (`POSITION.md` names EdgeMoE only as "Android, QNN and OpenCL backends" without the
  device). **Action for the project:** read EdgeMoE's full text for its decode rate before citing a
  number; this search only surfaces that a phone (Xiaomi 14) was used, not a tok/s figure comparable to
  ours (model, quantization and budget all differ from Qwen3-30B-A3B/Q4_0).
- **HOBBIT** ([arXiv 2411.01433](https://arxiv.org/pdf/2411.01433)) — reported speedups over
  llama.cpp/MoE-Infinity are measured on **Jetson AGX Orin**, not a phone. Confirms `POSITION.md`'s
  existing classification ("needs model changes, so not our regime" / not phone).
- **AdapMoE, Fiddler, ktransformers, MoE-Infinity, SP-MoE, FreeToken** — all found, all on
  workstation/server GPUs (RTX 3090/4060/5090) or Jetson-class edge, none on an Android phone. No new
  phone-class competitor.

**Not found in this pass:** any system reporting >= 8 tok/s decode for an unmodified >= 20B-total MoE
checkpoint on an Android phone. This matches `POSITION.md` §9's standing "still not found" line
(last updated 2026-09-17) and this search adds no counter-example, so that line is not restated as new
here — it is simply not overturned.

---

## (b) Repacked (interleaved Q4_0 i8mm) kernels applied to streamed/cached experts, incl. pre-repacking on flash

**Queries run (2026-09-19):**
- `llama.cpp repacked Q4_0 4x8 i8mm kernel MoE expert streamed cache pre-repack flash`
- `"expert cache" llama.cpp github discussion streaming SSD MoE Qwen3-30B-A3B phone`

**Verdict: not found** (general search only; not verified absent — no full crawl of llama.cpp's issue
tracker or commit history for "repack" + "MoE" + "stream" together was done, so this is a weaker claim
than "verified absent").

- General repack/i8mm kernel work exists upstream: `fastpath64` (GitHub, Marc-Dvci) adds a missing
  `smmla` repack kernel for **IQ4_XS** on Arm, reporting 2.12x faster prefill on dense models and 1.13x
  on MoE — but this is a **prefill** number, on a **different quant format**, and nothing there
  addresses experts that are streamed from flash rather than resident.
  `github.com/Marc-Dvci/fastpath64`.
  - Also found: recent llama.cpp work "transposed scales for Q4_K weights repack to allow coalesced
    load", improving dense-model prefill/decode — again dense, not streamed MoE experts.
- The MoE-expert-cache discussions found (`ggml-org/llama.cpp` #24528 RFC, #27149 "Expert-Aware SSD
  Streaming", #27861 "GPU-resident LRU cache for host-offloaded MoE expert weights", PR #25294, issue
  #27562 "Just-in-time MoE expert streaming from storage") are all about **caching/streaming policy**
  (which experts to keep resident, LRU eviction, VRAM residency), not about **repacking the on-flash or
  in-cache byte layout** for a faster matmul kernel. None mentions `i8mm`, interleaved `Q4_0_4x8`, or
  pre-repacking a GGUF's expert tensors before deployment.
- This project's own committed measurements (`results/2026-09-19/repack_bench/README.md`,
  `results/2026-09-19/repack_laptop/README.md`) — repacked kernels on Qwen3-30B-A3B expert shapes give
  0.596x/0.786x the generic kernel's time (gate/up, down) on the phone, and the pre-repack-on-flash path
  (`repack_gguf`, marker-checked, byte-identical to runtime repack, laptop perplexity 8.2637 vs 8.2675)
  — is not contradicted or duplicated by anything found. No prior public report of pre-repacking expert
  tensors for a streamed-from-flash MoE engine was found.

**Statement for the paper, if this framing is used:** "not found" only, dated 2026-09-19, general
WebSearch protocol above; do not upgrade to "verified absent" without a targeted search of the
llama.cpp GitHub issue/discussion/PR full-text index and a check of PowerInfer-2's and BigMoeOnEdge's
own repos for "repack".

---

## (c) Android autosuspend/Doze, thermal or charging state confounding on-device LLM benchmark validity

**Queries run (2026-09-19):**
- `Android Doze autosuspend benchmark validity thermal LLM on-device inference bimodal`

**Verdict: not found** for the specific Doze/autosuspend-causes-bimodal-decode-rate claim; **found** for
the closely related thermal-throttling-invalidates-peak-benchmarks claim, which the search surfaced
strongly.

- **Found, thermal:** "LLM Inference at the Edge: Mobile, NPU, and GPU Performance Efficiency Trade-offs
  Under Sustained Load" ([arXiv 2603.23640](https://arxiv.org/html/2603.23640v2)) reports a Snapdragon
  8 Gen 3 dropping from 12.4 to 3.8 tok/s (-69%) over 30 minutes, and describes iPhone 16 Pro and S24
  Ultra settling into governor-regulated thermal plateaus after a handful of iterations. This is the
  same phenomenon this project measured directly (`cap_vs_temp.json`, claims `cap_full_below_31C`,
  `cap_throttled_from_33C`: 97.3% of samples above 33 C shell temperature capped, tracking skin
  temperature rather than the framework's own thermal-status field) — a genuine external corroboration,
  useful for a benchmark-validity framing, but it is about **thermal governors**, not **Doze/autosuspend**.
- **Related, not the same mechanism:** "Dissecting the Impact of Mobile DVFS Governors on LLM Inference"
  ([arXiv 2507.02135](https://arxiv.org/pdf/2507.02135)) — governor policy effects on throughput/energy,
  again thermal/DVFS, not Doze.
- **Not found:** any paper describing Android's Doze mode or SoC autosuspend (the CPU/SoC entering a
  low-power suspend state under screen-off + `svc power stayon` not holding, causing **bimodal**
  per-row decode rates within a single benchmark campaign) as a benchmark-validity confound for LLM
  inference specifically. **This project's own finding — that unplugged rows without a per-row waker
  went Dozing and produced bimodal decode rates, fixed by a per-row waker plus a pre-registered
  `--require-awake` keep rule (`results/2026-09-19/NIGHT_SUMMARY.md`, `bmoe_gtier/bmoe_gtier_ab_20260919_0235/README.md`)
  — is, as far as this search shows, not previously reported in the on-device-LLM literature.** This is
  stated as "not found", dated 2026-09-19, general search only; it is not a claim of absolute novelty
  and should be re-run against systems venues (MobiSys/MobiCom papers on Android background-execution
  limits generally, e.g. Doze/App Standby literature from the Android systems community, which was not
  specifically queried here) before a paper asserts it as new.

---

## (d) Mobile GPU (Adreno/OpenCL) as a concurrent co-processor for a subset of MoE experts, bit-exact vs CPU

**Queries run (2026-09-19):**
- `Adreno OpenCL GPU concurrent co-processor MoE expert offload bit-exact CPU GPU LLM`

**Verdict: not found** for the bit-exact-concurrent-tier design; **found**, background only, for
Adreno/OpenCL MoE support existing at all.

- **Found:** Qualcomm's own OpenCL backend for llama.cpp on Adreno GPUs exists and is documented
  (ProAndroidDev writeup); it "supports MoE expert matmul" generically, which is exactly the backend
  this project already benchmarked (`msweep_gpu_expert_gate_up_speedup` etc., `POSITION.md` §1.4 /
  `ceiling_handoff.md` §1.4: closed, 0.39-1.09x the CPU depending on op).
  `github.com/ggml-org/llama.cpp` OpenCL/Adreno backend; ProAndroidDev article "Introducing the new
  OpenCL GPU Backend in llama.cpp for Qualcomm Adreno GPUs".
- **Found, general (desktop dGPU, not Adreno):** CPU/GPU **concurrent** expert offloading designs exist
  in the literature — "while the CPU computes one expert, another can be simultaneously loaded into GPU
  memory via PCIe" (DAOP, [arXiv 2501.10375](https://arxiv.org/pdf/2501.10375)) — but these are
  **desktop/server GPU over PCIe**, not a **mobile SoC's unified-memory GPU tile via `kgsl`**, and none
  claims **bit-exact** agreement with the CPU kernel; DAOP and similar systems are Tier-A (approximate/
  accuracy-preserving-in-aggregate), not Tier-E.
- **Not found:** any system pairing a mobile GPU with the CPU concurrently for a subset of MoE experts
  per token **and** proving bit-exact (Tier-E) numerical equivalence to the CPU kernel. This project's
  own `gx` component does claim exactly that — 0 of 262,144 down-projection outputs differ from
  ggml-cpu's ARM output across both Qwen3 expert down types, verified with negative controls
  (`gtier_m4_text_identical`, `gx_down_diff_bits`, `gx_row_down_diff_bits` in `CLAIMS.md`). **Statement:
  not found, general search only, dated 2026-09-19; not verified absent** (no targeted search of GPU
  driver / ggml-opencl issue trackers, or of systems venues specifically for "bit-exact heterogeneous
  MoE dispatch", was run).

---

## (e) Pinning (unswappable) memory for LLM weights on Android via GPU-driver (kgsl) allocations

**Queries run (2026-09-19):**
- `Android kgsl GPU driver pinned memory unswappable LLM weights avoid zram swap`
- `ashmem ION dmabuf pinned memory Android prevent swap machine learning inference mlock alternative unrooted`

**Verdict: not found** for the specific technique (using `kgsl` GPU-driver allocations, unrelated to
any GPU compute, purely as an unswappable CPU-readable memory pool for LLM weights on an unrooted
phone); **found**, background only, on the general mechanisms (kgsl exists, ashmem pinning exists as an
`mlock`-alternative for unrooted apps, ION/DMA-BUF is being deprecated in favor of DMA-BUF heaps).

- **Found, mechanism background:** kgsl is documented as the standard Android/Qualcomm kernel-mode GPU
  driver (submits Adreno UM-driver commands to the GPU) — nothing in the search ties it to a technique
  for pinning general-purpose (non-graphics) memory against reclaim.
- **Found, closest analogous idea:** Ashmem's pinning mechanism is described as an `mlock()`-equivalent
  reachable **without root**: "when creating anonymous shared memory, all memory is pinned by default,
  and only when the user tells the Ashmem driver to unpin a certain block of memory will the Ashmem
  driver unpin this memory" — i.e., the *general concept* of an unrooted app obtaining unswappable
  memory on Android via a non-`mlock` kernel path is documented, but for **ashmem**, not **kgsl**, and
  not in an LLM-inference context.
- **Not found:** any prior report of allocating `kgsl`-backed (GPU-driver) memory purely to get
  CPU-side, zram-immune storage for LLM expert weights or KV cache on an unrooted Android phone. This
  project's own `pinprobe` measurement (`results/2026-09-19/pinprobe/README.md`) — kgsl reads at
  malloc speed (62.9 vs 61.7 GB/s) and is immune to `MADV_PAGEOUT` reclaim (0 vs 196,608 major faults)
  — appears, from this search, to be a technique not previously reported for this purpose. **Caveat,
  and it is a real one for a paper:** the project's own follow-on engine A/B
  (`results/2026-09-19/bmoe_kgsl/bmoe_kgsl_ab_20260919_0131/README.md`) found the naive application
  (pinning the *cold* expert cache) **backfires** — it pushes the system's reclaim pressure onto the
  engine's *hot* anonymous pages (dense weights, KV, buffers), and the campaign was stopped after 4 rows
  with no verdict because ABBA carryover made the design invalid. So the finding, if reported, is a
  **negative/nuanced** one: the mechanism is real and measurable, but the naive engineering use of it
  is worse, not better, and the correct target (hot set, not cold cache) is untested here
  (`--dense-weights ahwb` flagged as never measured). **Statement: not found, general search only,
  dated 2026-09-19; not verified absent** (no targeted search of Android systems venues, XDA/kernel
  forums, or existing on-device-inference engines' source for `kgsl`+`mlock` was run).

---

## Summary table

| topic | verdict | strength | re-run before submission |
|---|---|---|---|
| (a) flash-streamed on-device MoE, phone decode rates | found (existing record confirmed, no new phone-class beater) | general search | yes — re-check BigMoMo and any successor within days of submission (it was 4 days old at last check) |
| (b) repacked/i8mm kernels applied to streamed/pre-repacked MoE experts | not found | general search only, not verified absent | yes — full-text search of llama.cpp issues/discussions/PRs for "repack" + "expert"/"MoE" |
| (c) Doze/autosuspend as a benchmark-validity confound (vs. the well-documented thermal-throttling confound) | not found (thermal confound is found and corroborates our thermal measurement; Doze specifically is not) | general search only, not verified absent | yes — search Android systems literature (MobiSys/MobiCom) on Doze/App Standby specifically |
| (d) mobile GPU as bit-exact concurrent co-processor for MoE experts | not found | general search only, not verified absent | yes — search ggml-opencl issue tracker and heterogeneous-inference systems venues |
| (e) kgsl-backed unswappable memory for LLM weights on unrooted Android | not found | general search only, not verified absent | yes — search Android kernel/XDA forums and existing engines' source |

None of the five topics reached "verified absent" — every negative above is bounded to the queries
listed and the WebSearch tool's result set on 2026-09-19. A paper claiming novelty on any of (b)-(e)
needs the targeted re-run named in the table, per `CLAUDE.md` §10.4.
