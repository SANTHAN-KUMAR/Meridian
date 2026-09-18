# GPU expert path — engineering design (2026-09-18)

Status: **design, no code yet.** Nothing here is a measured claim unless it cites a claim id from
`gates/claims_check.py`. The phone is resting: every milestone below up to M5 is verified on the laptop.
M6 is the first phone step, and it waits until the user releases the phone.

## 1. Goal and the budget it has to meet

The target is Qwen3-30B-A3B, unmodified, at 10 tok/s: **100 ms per token**. The measured terms today:

| term | now | source |
|---|---|---|
| compute: dense part (0.82 GB/token) + experts (1.02 GB/token) | ~92 ms at status-1 caps | `densemap_sensitivity_compute` (anon rows), `gguf_active_qwen_olmoe.json` |
| stall + cache management (stack) | ~55–60 ms | `stack_stall_mgmt_ms` |
| stall saving from I/O lanes on cpu0,1,6,7 | −12.3 ms | `cores_3c_stall` |

The one lever that moves compute on this phone is **putting a second device on the expert matmuls,
concurrently with the CPU**:
- The GPU zero-copy benchmark measured CPU 22.5 + GPU 13.0 = 35.5 GB/s, 1.54× the CPU alone, with proven
  overlap, zero-copy reads and correct results (teammate commit 35c4c16).
- Applied to the expert bytes only (1.02 GB/token), rate-proportional splitting takes the expert matmuls
  from ~44 ms to ~29 ms: **about −15 ms/token**, before sync costs.
- A faster GPU kernel raises that. Moving the dense attention matmuls to the GPU as well is the
  extension (§8).

Projection, not a measurement: 92 − 15 compute + 45 stall/mgmt ≈ 122 ms, about 8.2 tok/s with experts only.
About 9–10 tok/s needs a GPU kernel at CPU speed and/or attention on the GPU. The design has to make
both extensions cheap.

## 2. Architecture in one paragraph

Per MoE layer and decode token, the router selects 8 experts. The engine splits them into two sets:
- **G:** experts whose three projections are already resident in GPU-visible memory. The GPU computes
  their whole FFN (gate, up, SwiGLU, down) in ONE dispatch.
- **C:** the rest, including every miss still being read. The CPU computes them through ggml's normal
  `MUL_MAT_ID`, which has been told to skip G.

When the CPU finishes the layer's down projection, one thread waits for the GPU and writes G's down
outputs into the `MUL_MAT_ID` destination rows. Every op after that (expert weighting, sum) is untouched
ggml. Cost: one GPU dispatch and one wait per layer, i.e. 48 per token.

## 3. Components and ownership

| # | component | owner | where |
|---|---|---|---|
| A | **GPU-visible expert pool**: fixed-size blocks allocated with `CL_MEM_ALLOC_HOST_PTR` and mapped once (persistent CPU pointer). Slots are handed out like the slot arena (patch 0008). O_DIRECT reads go straight into a slot. | me | engine `expert_stream_source`, new `gpu_pool.{h,cpp}` |
| B | **ggml-cpu hooks** (fork patch): `mmid_begin` (thread 0, before the row-grouping barrier; returns a per-expert skip mask), a skip test in the expert loop, and `mmid_end` (thread 0, after its own chunks; waits for the GPU and fills G's dst rows) | me | `ggml-cpu.c`, `ggml-cpu.h` |
| C | **Fused expert-FFN OpenCL kernel + C API** (no ggml dependency): `gx_init`, `gx_dispatch(layer, down_type, k, slots[], x, out)`, `gx_wait`; native on-flash layout: gate/up Q4_0 in every layer, down **Q4_1 in layers 0-5 and Q4_0 in layers 6-47** (read from the GGUF tensor table; teammate, 2026-09-18), so the down type is passed per dispatch | teammate | `moe-phone/host/gpu_ffn/` → linked into the engine |
| D | **Assignment policy**: which selected experts go to G. Resident ones only, count proportional to measured rates (≈3 of 8 at 13 vs 23 GB/s), `--gpu-experts K|auto`, 0 = off | me | engine |
| E | **Telemetry**: per token, experts on the GPU, dispatch→ready latency, wait time charged to the CPU, GPU bytes; one shutdown summary line | me | engine CSV + stderr |
| F | **Correctness harness** (laptop) | me + teammate | `gates/`, `host/` |

## 4. Data flow for one layer (decode, n_tokens = 1)

1. **Routing node (existing hook).** `ffn_moe_topk-<il>` is observed and `load_layer` stages misses as
   today.
2. **Gate `MUL_MAT_ID`, `mmid_begin` on thread 0.**
   - Read ids. For each selected expert that is resident in the pool, including its gate/up/down slots,
     pick up to K for G.
   - Call `gx_dispatch` with x = src1 (the normalised hidden state, fp32, 2048 floats) and the G slots.
     It returns immediately.
   - Set skip bits for G.
   - This happens before the existing barrier, so every thread sees the mask.
3. **Gate and up `MUL_MAT_ID`.** Threads skip G's experts. G's rows in dst are left unwritten; the
   engine zero-fills them in `mmid_begin` so no NaN can propagate.
4. **SwiGLU (ggml GLU op).** Runs over all 8 slots. G's slots compute on zeros, which is harmless:
   down skips them.
5. **Down `MUL_MAT_ID`.** Threads skip G. On thread 0, `mmid_end` calls `gx_wait` and copies G's 2048-float
   outputs into dst rows `(slot id, token 0)`. Those rows are disjoint from every row a CPU thread
   writes, so no barrier is needed.
6. **Weighting and sum:** unchanged ggml ops read a complete dst.

Prefill (n_tokens > 1) keeps the CPU path. `mmid_begin` returns an empty set.

## 5. Why a block pool, not the current per-layer reservation

The LRU cache reserves one virtual buffer per (layer, projection) sized for all 128 experts, and rebinds
`tensor->data` to it (`expert_stream_source.cpp:185-207`). Three facts rule that layout out for the GPU:
1. **Zero-copy.** The teammate's benchmark found `CL_MEM_USE_HOST_PTR` on ordinary memory is **not**
   zero-copy on Adreno (copy-speed sync, stale data without it). Only `CL_MEM_ALLOC_HOST_PTR`
   allocations are.
2. **Size.** A GPU allocation cannot reserve 128 × slice per layer, which is the model's full 16 GB of
   expert bytes.
3. **The data hook already exists.** The slot arena already addresses experts through
   `ggml_cpu_set_expert_data_hook`, so ggml reads a slot wherever it lives.

**Known risk carried in:** the slot arena measured compute +14 ms/token on the phone (`arena2_compute_delta_ms`),
cause unresolved. The OLMoE counters point at page faults (48–76 major faults/token in the engine vs 0.4
plain, `ovhmech_*` rows). Anonymous slots can be compressed into zram; GPU-mapped memory is normally pinned,
so a GPU-visible pool may REMOVE that penalty rather than inherit it. M6's first phone check measures
exactly this before anything else.

## 6. Numerics (what "lossless" means here)

The model is unmodified: the same weights, the same operations, the same routing. The GPU's float rounding
differs from the CPU's:
- ggml-cpu quantizes the activation to Q8_0/Q8_1 before an integer dot.
- The GPU kernel may not, and its exp in SiLU differs.

So bit-identical output is **not** promised. It is a stretch goal for component C: mirror ggml's
activation quantization and block order. Fidelity is verified, not assumed:
- On the laptop (OLMoE, and Qwen3 layer slices, never the whole model): per-layer max relative error of the
  FFN output vs the CPU path.
- Top-1 agreement and mean KL of the final logits over a fixed prompt set.
- Pre-stated acceptance: mean KL < 1e-4 and top-1 agreement ≥ 99.5%. Those are the ranges at which
  llama.cpp's own CPU and GPU backends are treated as equivalent. If the result is worse, it is a defect,
  not a tradeoff.

## 7. Milestones and acceptance tests

Each one ships with a test whose expected value is fixed by construction, not by the result (CLAUDE.md §9.1).

| M | deliverable | acceptance (laptop unless stated) |
|---|---|---|
| M1 | Hooks B, plus a **CPU stand-in for the GPU** that computes G's FFN with ggml-cpu's own routines | A `MUL_MAT_ID` graph with and without the hooks, stand-in on: dst must be **bit-identical**. This proves the plumbing: skip mask, row mapping, end-fill, thread safety, over 1000 random id sets, k = 0..8 |
| M2 | Pool A (OpenCL `ALLOC_HOST_PTR` blocks, persistent map) behind the existing arena interface, with an anon fallback | The engine on OLMoE with the pool: text identical to the non-pool arena and to plain LRU, 128 tokens |
| M3 | Kernel C and its API | vs ggml-cpu on random and real Qwen3 expert slices: max relative error reported; within the §6 bound. A laptop OpenCL functional run only (the NVIDIA driver copies, so no speed claim) |
| M4 | Policy D, telemetry E, and C wired into `mmid_begin`/`mmid_end` | OLMoE end to end on the laptop: logits KL and top-1 vs CPU-only within §6. Counters: G per token equals K when enough experts are resident |
| M5 | Failure paths: GPU init fails → CPU-only with a counted warning; a dispatch or wait error → that layer falls back to the CPU and it is counted (CLAUDE.md §6.3); `--gpu-experts 0` is byte-identical to today | Fault injection on the laptop: forced init and dispatch failures produce correct text plus a nonzero fallback counter |
| M6 | Phone (when released): pool pinning and arena-penalty check, then per-dispatch latency, then the pre-registered A/B (K = 0 vs auto; primary compute ms/token; guard: CPU caps and stall) | Pre-registered before the run, as for every campaign |

## 8. Extensions that the design keeps open

- **Attention on the GPU:** the same pool plus a dense GEMV dispatch at the attention matmuls; `--attn-device`
  exists (patch 0009) but crosses ggml backends 192 times per token. A direct dispatch avoids that.
- **Faster GPU kernel:** ggml-opencl's Adreno MoE kernels need a transposed layout (trans4). A layout
  conversion at read time can be evaluated once M3 gives a baseline.
- **Better split:** assign by measured per-layer GPU latency instead of a fixed K.

## 9. Risks, each with how it is detected

| risk | detection |
|---|---|
| Per-layer dispatch and wait latency eats the saving (48 per token) | M6 per-dispatch latency; telemetry E charges the CPU's wait to the token |
| GPU heat lowers the CPU clock caps (shared skin-temperature budget) | M6 logs both CPU caps every second; the A/B's guard |
| Pool memory is pinned and counts against LMK differently | M6 logs MemAvailable and swap; the budget auto-sizer must see the pool |
| Driver limits (max single allocation, total mappable) | Queried at init, logged; the pool sizes its blocks from them |
| Numerics exceed §6 | M3/M4 acceptance fails loudly; no speed number is reported from a failing build |

## 10. Failures reopened, not dismissed (user direction, 2026-09-18 ~22:00)

Each of these was recorded as "no gain" or left unexplained. None is closed until its cause is understood.
Each gets a hypothesis, a check that does not need the phone (where one exists), and a fix attempt.

| # | failure as recorded | why it is suspicious | hypothesis | next check |
|---|---|---|---|---|
| R1 | Slot arena: cache mgmt −18 ms, compute **+14 ms** (`arena2_compute_delta_ms`) | the arena removes work; compute should not rise | anon slots are compressed into zram and faulted back (the OLMoE engine shows 48–76 major faults/token vs 0.4 plain); the pool in §5 is pinned GPU memory and may remove it | count faults per token for arena vs LRU from existing CSVs (`majflt` column); M6 phone check |
| R2 | Dense weights: mmap **saves 13.6 ms** on OLMoE (`ovhmech_dense_copy_ms`) but **costs 22.5 ms** on Qwen3 (`densemap_sensitivity_compute`) | the same change with opposite signs means something model-specific decides it | the mmap'd Qwen3 rows had FEWER major faults (18 vs 61/token), so faults are not it. Candidates: file-page eviction plus readahead re-reads the fault counter misses, or TLB reach on 0.82 GB of dense weights | per-token minor faults and read bytes from `/proc/pid/io` for both arms (a counter to add to the engine); page-cache residency of the dense ranges (the engine's existing `dense_resident_frac`) |
| R3 | Engine 1.9× slower than plain on OLMoE (`overhead_ratio`), mostly spin at barriers plus kernel time | on Qwen3 the gap is only 12–25% | the gap tracks faults (kernel time while the other threads spin) | M-work: pinned pool (§5) plus fault counters per layer |
| R4 | Repacked i8mm kernels on streamed experts were **slower** (compute 0.170 vs 0.146 s/token, n=2, 2026-09-17) | i8mm GEMV is designed to be faster than the generic q4_0 dot; the CPU is the compute wall | the in-place repack runs in `read_slice` on the I/O path and was charged to compute, OR the repacked path skipped the fork's ready/probe hooks and serialized; 2 rows at mismatched budgets (3448 vs 3748 MiB) decide nothing | laptop: per-op GEMV rate of the repacked vs generic kernel on a Qwen3 expert shape (ggml_matmul_bench, CPU); read repack.cpp's hook coverage. **Also: the new split hooks (§3 B) exist only in the generic path; the repacked MUL_MAT_ID needs them too before both can be combined** |
| R5 | 6 threads: compute −8 ms, not resolved (`cores_6thread_compute`) | 0.84× in the teammate's pin2 data | the second clock domain paces barriers | bound it by the GPU split's effect on the CPU share first; revisit after M4 |
| R6 | SLRU cut flash bytes 11.1% and decode did not resolve (`slru_decode_clean`) | bytes are the stall's input | the stall is set by per-layer read latency, not bytes | per-layer stall histogram from the node trace (existing tool); in the stack, SLRU+prefetch DID resolve −10.7 ms |
| R7 | GPU per-op sweep measured the generic OpenCL path (weights misnamed) | the Adreno MoE kernels were never run | — | rebuild the app's matmul bench with `--wname` (committed 91e877f) when the phone returns |
| R8 | The OEM performance-mode setting did not lift the caps | the UI toggle may engage a service that `settings put` does not | the Settings UI calls horae/oplus-perf over binder | read which binder call the toggle makes (dumpsys during a manual toggle), when the phone returns |

## 11. Component G — swap-aware cache (from R1; independent of the GPU)

**Finding** (claims `r1_arena_majflt`, `r1_base_majflt`): Android compresses unused anonymous memory into zram
even with ~1.4 GB available.
- The engine runs with ~370 MiB of itself swapped and ~21–23 major faults per decode token.
- The slot arena, which never releases slots, doubles the swap (~770 MiB) and quadruples the faults (~90/token).

The engine's hit counter calls a swapped-out expert a HIT. It is the most expensive case there is: its ~216
pages (a 884 KB slice) fault back one by one, synchronously, on a compute thread, while the other threads
wait at the next barrier. A real miss is one ~1.3 ms O_DIRECT read that the engine overlaps with compute.

**Design:**
1. **Detect.** In `load_layer`, for each selected expert counted as resident, `mincore()` its slice ranges.
   Anon pages report 0 when swapped out. One syscall per (expert, projection): ~1,150 per token, µs each.
   Sample-first variant: check the first and last page, full range only if they differ. The cost is
   measured, not assumed.
2. **Act.** An expert not fully resident is demoted to a MISS:
   - `madvise(MADV_DONTNEED)` its slices, which drops the swap entries without swapping in;
   - then stage the normal O_DIRECT read, which the existing overlap path hides like any miss.
   - The decision happens before the layer computes, so no thread can be reading the slice.
3. **Prevent.** Feed the process's own swap (`VmSwap` in `/proc/self/status`) back into the budget through
   the existing `set_cache_budget`. If VmSwap grows over a window of N tokens, shrink the budget by a step;
   never grow it past the auto ceiling.
4. **Telemetry.** Swapped-hits detected per token, bytes re-read because of them, VmSwap per token, budget
   changes.

**Acceptance (laptop, no phone):** Run OLMoE in a memory cgroup small enough that the kernel swaps the cache
(`systemd-run -p MemoryMax=<small> -p MemorySwapMax=<large>`; the laptop's swap is zram too). Then:
- the guard detects swapped hits (counter > 0);
- the text is byte-identical to the unguarded run and to plain;
- major faults per token fall.

Speed is judged only on the phone (M6).

**Why before M2:** the GPU pool is pinned memory, so it cannot be swapped. That protects cached experts, but
it also pushes more of everything else (the dense copy, KV, buffers) into zram. The guard and the budget
feedback are what keep that from turning into faults elsewhere.

## 12. Revision after M3 and the coherence facts (2026-09-18 ~23:30)

**M3 is done and bit-exact** (teammate, 0225f99). gx reproduces ggml-cpu's ARM arithmetic exactly: 0 of 262,144
down outputs differ, across both down types and real Qwen3 slices, and all 10 negative controls are
detected. The GPU split is therefore verified by **byte-identical text**, like every CPU-side change. §6's
KL tolerance is no longer needed; it stays only as a fallback if the device's arithmetic (denormals, fma)
turns out to differ at M6.

**Two tiers, not one shared pool.** The only phone-verified coherence protocol is map(WRITE_INVALIDATE) →
write → unmap → kernel, with nothing mapped during a dispatch (teammate's answers from `zcbench.out`). A
CPU thread therefore cannot compute from pool memory while the GPU runs. So:
- **CPU tier:** today's anonymous cache, unchanged. The CPU computes only experts resident here.
- **GPU tier:** a pool of slots in `ALLOC_HOST_PTR` blocks, each slot holding one expert's three slices,
  4096-aligned. It is filled by **promotion**: when an expert becomes hot in the CPU tier (SLRU's
  protected segment), it is copied into a free slot (map, memcpy, unmap) and **released from the CPU
  tier**, so nothing is held twice. A full pool evicts its LRU slot, which leaves both tiers and is
  re-read from flash when next needed.
- **Per layer, decode only:** every selected expert that the GPU tier owns goes to G. There is no fixed K.
  The pool's size sets the GPU's expected share, and telemetry (GPU wait charged to the CPU vs the CPU's
  own layer time) is what tunes it.
- **Prefill / multi-token batches:** v1 flushes the GPU tier before any multi-token batch. Its experts
  become misses and are re-read. That is correct and simple, and it is counted. v2 maps pool slots
  READ-only for the CPU while no dispatch runs.
- **All OpenCL stays inside libgx** (requested pool API: slot alloc/free/map_write/unmap on gx's
  context). The engine sees an abstract `GpuExperts` interface with two implementations:
  - **libgx** (the phone);
  - **a CPU stand-in** that runs the same per-expert FFN as a small ggml graph (mul_mat, swiglu_split,
    mul_mat; plain MUL_MAT, so the split hooks never recurse) on its own thread.

  The stand-in is bit-identical to the main graph by construction (M1 showed mul_mat and mul_mat_id rows
  agree). So the whole engine's text with the tier ON must equal the tier OFF: the M4 plumbing acceptance
  on the laptop, before gx is linked at all.

## 13. M4 result and the policy it exposed (2026-09-19 ~00:30)

**Plumbing: accepted** (claim `gtier_m4_text_identical`, patch 0019). With the CPU stand-in as the device, the
engine's text is byte-identical with the tier ON and OFF: OLMoE, 96 tokens, the full stack config, and 16,449
experts computed by the "device" across two runs, with no failure and no fail-loud abort. Patch 0019 also fixes a
latent 0011 bug:
- `evict_tail` cleared `cprot_` before `lru_unlink`, so a protected-tail eviction unlinked from the wrong list.
- It was never reached on the phone (0 protected evictions in every SLRU row), but the tier makes it reachable.

**Policy: not acceptable yet** (claim `gtier_m4_policy_churn`):
1. **Churn.** 2,498 promotions and 2,350 tier evictions in 96 tokens. Promotion on a plain hit count admits
   experts no hotter than the ones they evict. On the phone every promotion is a ~2.6 MB copy plus a
   map/unmap. **Fix:** admission control. Keep a per-entry hit-frequency estimate (decayed counts) and admit
   only if it exceeds the tier's LRU tail's; otherwise leave the expert in the CPU tier.
2. **Device overload.** 4.6–6.2 of 8 experts per dispatch went to the device. The GPU measured ~half the
   CPU's rate, so it would become the layer's critical path; the target share is ~3/8 until M6 measures
   gx's real rate (variant 0 vs 1). **Fix:** a per-layer cap K on device experts. Owned experts beyond K
   must still be computed, and their CPU pages are gone, so the CPU reads them from their GPU-tier slot.
   With one expert per cl_mem (the teammate's recommendation), mapping a slot's buffer READ-only while the
   GPU reads OTHER buffers is within the verified rules; the data hook points ggml at the mapped slot, and
   it is unmapped after the op. Cost: one map/unmap per overflow expert, counted.
3. **Balance.** Pick K per layer from measured timings rather than fixed: the device's per-expert time
   (gx_stats) against the CPU's per-expert time (compute term / experts), both from telemetry.
