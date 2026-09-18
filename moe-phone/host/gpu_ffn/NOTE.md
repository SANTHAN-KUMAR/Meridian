# gx: fused MoE expert FFN on the GPU, bit-exact with ggml-cpu (design component C, milestone M3)

Status, 2026-09-18: **M3 laptop acceptance passed.** gx's output is bit-identical to ggml-cpu's ARM
path on every value compared, and every negative control is detected. Nothing here is a speed claim.
The laptop GPU is an NVIDIA part with a copying driver, so no throughput figure in this directory
means anything for the phone. The phone is resting; §6 is the M6 plan.

## 1. What it is

`gx.h` + `gx.cpp` + `gx_kernels.cl` (embedded as `gx_kernels.inc`). There is no ggml dependency;
OpenCL is loaded with `dlopen`.
- `gx_init(cl_context, cl_device_id, {n_embd, n_ff, debug_h})`
- `gx_dispatch(g, layer, down_type, k, slots[k], x, out)`: enqueues and returns.
- `gx_wait(g)`: one host wait per dispatch.
- `gx_get_stats`: counts dispatches, experts and errors, for CLAUDE.md §6.3.

For k ≤ 8 experts of one layer and one token, gx computes `out[s] = down_s(silu(gate_s(x)) * up_s(x))`.
The weights are read in place, in the GGUF block layout, from `cl_mem` blocks the caller owns (the
pool, component A), at byte offsets given per slot.

**Down type is per layer.** In `Qwen3-30B-A3B-Q4_0.gguf`, `ffn_down_exps` is Q4_1 only in layers 0–5
and Q4_0 in layers 6–47 (claim `gguf_down_q4_1_layers`, from `gates/gguf_expert_types.py`). Gate and
up are Q4_0 everywhere. The caller passes `GX_DOWN_Q4_0` or `GX_DOWN_Q4_1`. Down slice size is
n_embd · n_ff/32 · 18 or · 20 bytes.

**One dispatch = two kernels, one wait.**
1. The host quantizes x to Q8_0. That is 2048 floats, using exactly ggml's formula.
2. `gx_gate_up` computes the gate and up rows, SwiGLU, and the quantization ggml applies to h before
   the down `MUL_MAT_ID`: Q8_1 for a Q4_1 down, Q8_0 for a Q4_0 down.
3. `gx_down` computes the down rows. A single kernel would need a grid-wide barrier between up and
   down, which OpenCL does not provide.

A work-group is 256 work-items: 32 rows × 8. `gx_init` fails loudly if the kernel's work-group limit is
below 256.

## 2. Numerics: what "bit-exact with ggml-cpu" required (L4/L5)

The target is the ggml-cpu build the engine ships: `armv8.6-a+dotprod+i8mm+fp16`, no SVE, no KleidiAI.
- Streamed experts never go through CPU_REPACK (session.cpp:624, `use_extra_bufts = false`, confirmed
  by the engine session).
- At n_tokens = 1, `MUL_MAT_ID` calls `vec_dot` with nrc = 1, so the i8mm `smmla` path never runs.
- n_embd/32 = 64 and n_ff/32 = 24 are both even, so the scalar tail loops never run.

The operations reproduced are these, from `ggml-cpu/arch/arm/quants.c`, `vec.h` and `vec.cpp`, with
each fused multiply-add confirmed in the disassembly of the engine's `libggml-cpu.so`:

| step | ggml (ARM NEON) | gx |
|---|---|---|
| activation → Q8_0 / Q8_1 | amax; d = amax/127 and id = 1/d (two IEEE fdiv, *not* 127/amax); d stored fp16 RNE; q = `vcvtnq` (ties to even); Q8_1 s = fp16(fp32 d · Σq) | same. On the GPU the divisions are correctly rounded by an fma residual check (`cr_div`), because OpenCL's `/` may be off by 2.5 ulp |
| q4_0·q8_0, q4_1·q8_1 | two float32x4 accumulators (even and odd blocks). Per lane: `fmla` (fused) of float(sdot lane) × fp32(dx·dy). Sum: `faddp`, `faddp`, `fadd` | work-item w of a row = accumulator w>>2, lane w&3, in the same block order; explicit `fma`; the same reduction tree |
| q4_1 `summs` | `summs + fma(m0, s0, m1·s1)` (an `fmadd`); result = summs + (hsum0 + hsum1) | same |
| SwiGLU | `ggml_v_expf` (a polynomial with 7 fused ops), silu = x / (1 + e) (IEEE fdiv), times up | `gx_expf`, a lane-exact port; `cr_div` |

The kernels use `#pragma OPENCL FP_CONTRACT OFF` and must never be built with `-cl-mad-enable` or
fast-math, so that every unfused multiply-add stays unfused.

## 3. Acceptance (laptop): method and result

Produced by `run_accept.sh` into `results/2026-09-18/gx_m3_220511/` (first run) and
`gx_m3_221524/` (the current kernel file, with the pool test added):
- `gx_test.json`, `gx_test.out`, `controls.out`, `pool_test.out`, `disasm_identity.txt`,
  `manifest.json`, `STAMP`.

**Reference.** `ggml_ref.cpp` runs ggml's own graph (`mul_mat_id` → `ggml_swiglu_split` →
`mul_mat_id`, as llama.cpp's `build_moe_ffn` emits for Qwen3-MoE). It is built as a static aarch64
binary and run under `qemu-aarch64-static`, whose softfloat is IEEE-exact.
- `disasm_identity.txt` shows that the five numeric functions in that binary are instruction-identical
  to the engine's Android `libggml-cpu.so` (addresses stripped): `quantize_row_q8_0`,
  `quantize_row_q8_1`, `vec_dot_q4_0_q8_0`, `vec_dot_q4_1_q8_1` and `vec_swiglu_f32`.
- So the reference is the phone's arithmetic, not a re-implementation of it.

**Cases** (`make_cases.py`, 27 + 1):
- **20 random cases:** k = 1..8, both down types, and activations of five kinds: normal; wide (8
  decades per block, including fp16-subnormal d); zero blocks; rounding ties; outliers.
- **7 real cases:** Qwen3-30B-A3B expert slices from layers 0, 5, 6, 12, 24, 36 and 47, read from
  the GGUF by byte offset. Both down types are covered. x is randn · `ffn_norm.weight`.
- **65,536 crafted quantizer blocks:** elements within 2 ulps of rounding half-way points; half of
  them with an amax where 127/amax ≠ 1/(amax/127); fp16-subnormal d; fp32-subnormal elements; zero
  blocks.

| check | result | claim |
|---|---|---|
| down outputs, bitwise vs ggml ARM | 0 differ of 262,144 | `gx_down_values_compared`, `gx_down_diff_bits` |
| SwiGLU h, bitwise vs ggml ARM | 0 differ of 98,304 | `gx_h_diff_bits` |
| Q8_0/Q8_1 quantizer, GPU and host, vs ggml's `from_float` | 0 blocks differ | `gx_quant_mismatch_blocks` |
| `cr_div` vs IEEE division, 16.8M pairs | 0 differ | `gx_div_mismatches` |

For scale, the per-case numbers are in `gx_test.out`:
- **Against an fp64 FFN on the same dequantized weights, with x unquantized:** ggml's own error and
  gx's are identical, about 1–3% max-relative. This is the Q8 activation error that the CPU path
  already has. gx adds nothing to it.
- **Against x86 ggml:** within about 1e-4 on real cases, and about 1.3% on the rounding-tie cases.
  That is expected: x86's AVX quantizer computes id = 127/amax, not 1/(amax/127). This is why the
  reference must be the ARM build.

**Negative controls** (`negative_controls.sh`, CLAUDE.md §9.1). Each run changes one thing in the
kernel, rebuilds, and reruns the whole test. All 10 are detected (`controls.out`):
- each fused fma unfused (gate/up, down);
- q4_1 summs with two roundings;
- a sequential horizontal sum;
- a single accumulator per lane;
- libm/native `exp` in SiLU;
- a plain divide in SiLU;
- round-toward-zero quantization;
- 127/amax scaling;
- s computed from the fp16 d.

Two findings came from controls that were initially missed:
- An "unfused summs" mutation was undetectable **because it changes nothing**: fp16 × fp16 products
  are exact in fp32. It was replaced by a mutation that actually changes the rounding.
- The 127/amax mutation escaped the end-to-end cases, because h rarely lands near a rounding tie.
  That gap is why the crafted-block quantizer test exists. That test detects it in about half of the
  blocks aimed at it.

## 3b. The expert pool (API for the engine; all OpenCL stays inside libgx)

- `gx_pool_create(g, block_bytes, n_blocks)`: allocates `READ_ONLY | ALLOC_HOST_PTR` blocks and
  returns how many it got. It touches each block once through a map, so allocation failures surface
  here. It logs `max_mem_alloc` and `global_mem`.
- `gx_slot_alloc(g, gate, up, down)`: one expert = three slices, contiguous in one block, each offset
  4096-aligned for O_DIRECT. First fit, with coalescing on `gx_slot_free`.
- `gx_slot_map_write`: maps exactly the slot's range with `WRITE_INVALIDATE_REGION`.
- `gx_slot_unmap`: unmaps and waits for completion on its own event, so every later `gx_dispatch`
  sees the bytes.
- Map and unmap run on a second command queue and may be called from I/O threads while
  `gx_dispatch` runs on the compute thread. The allocator and counters are mutex-guarded.
- `gx_stats` adds: pool blocks and bytes, live slots, allocation failures, maps, unmaps, map errors,
  bytes mapped, and map/unmap time.

`gx_pool_test` (laptop, NVIDIA; `pool_test.out`) runs every check for two pool shapes, one slot per
block and four slots per block:
- **Allocator:** 20,000 random alloc/free operations over the two slot sizes (Q4_0-down and
  Q4_1-down experts), with no alignment, bounds or overlap violation. Capacity after freeing
  everything equals capacity when new.
- **Correctness:** dispatch from pool slots is bit-exact against ggml ARM on all 7 real cases.
- **Concurrency:** 200 dispatches stayed bit-exact while an I/O thread mapped, wrote and unmapped
  other slots in the same blocks.
- **Negative control:** a dispatch issued while the slot is still mapped with new bytes saw the
  **old** bytes. So on a copying driver, a protocol violation is detectable. After the unmap, the new
  bytes are seen.
- **What the laptop cannot show:** on Adreno's zero-copy memory, the same violation might show the
  new bytes, or a mix, and go unnoticed.
- **The region question stays open for Adreno:** can a kernel read one region of a buffer while
  another region of it is mapped? The spec wording is ambiguous.
- **Recommendation:** one slot per block until M6's stress test answers both points.

## 3c. Variant 1 (row per work-item): wired and bit-exact

`gx_params.variant = 1` selects `gx_gate_up_row` and `gx_down_row`. They do the same arithmetic with
one work-item per output row. Each work-item holds all 8 NEON lane accumulators and walks the blocks
in order, reading whole 18- or 20-byte blocks. The activation blocks are staged in local memory, and
work-groups are 64 work-items.

- **Tests.** `gx_test` runs every case on both variants. Variant 1 is bit-exact too: 0 of 262,144
  down values and 0 of 98,304 h values differ (claims `gx_row_down_diff_bits`, `gx_row_h_diff_bits`;
  `results/2026-09-18/gx_m3_222005/`).
- **Negative controls.** Four target this variant, and all are detected:
  - an unfused lane fma;
  - a sequential horizontal sum;
  - wrong accumulator parity;
  - summs with two roundings.
- **Why parity shows in h only.** The wrong-parity control changes only h, by about an ulp. The 8-bit
  quantization of h absorbs that before down, and the bitwise h comparison is what catches it.
- **Speed is untested.** Which variant is faster on Adreno is an M6 question. The laptop's relative
  timings are not evidence for it.

## 4. What is NOT established## 4. What is NOT established

- **Adreno arithmetic.** The laptop GPU reports fp32 denormals, correct fma and a correctly rounded
  divide. Adreno may flush fp32 denormals, or its `fma` may be slow or emulated.
  - Denormals appear only in tiny-value corners: fp32-subnormal activations, or SiLU at |x| > 87.
  - Bit-exactness on the phone is unknown until gx_test runs there (§6 step 1).
- **qemu equals hardware.** This rests on IEEE semantics. §6 step 1 closes it directly by running
  `ggml_ref_arm` natively on the phone.
- **Speed.** None of it is measured. The kernel's memory pattern (8 work-items per row, 4-byte loads)
  was chosen for exact lane correspondence, not for bandwidth.
- **The engine path end to end.** That is M4 (logits KL and top-1 on OLMoE).
- **The remaining SwiGLU vector branch.** gx reproduces ggml's special-case branch for |n| > 126
  lane by lane. It is exercised only by extreme gate values (|g| ≳ 87), which neither the random nor
  the real cases are guaranteed to hit. The crafted-block test covers the quantizer, not exp.

## 5. Integration notes (for the engine side, M4)

- x must be exactly the fp32 `src1` ggml passes to the gate `MUL_MAT_ID` (the normed hidden state).
  gx quantizes it to Q8_0 on the host exactly as ggml does, so no Q8 buffer is shared with ggml.
- Row s of `out` is the down output for `slots[s]`. Write it to dst row (slot id, token 0) in
  `mmid_end`.
- **Buffers must not be mapped by the CPU while a dispatch runs.** The only zero-copy protocol tested
  on the phone is map (`WRITE_INVALIDATE_REGION`), then write, then unmap, then kernel (see
  `host/gpu_zerocopy/NOTE.md`).
- `gx_dispatch` finishes any pending dispatch first: its staging arrays are reused.
- A context is single-threaded.
- `build.sh android` produces `out/android/libgx.a` from `gx.cpp`. `gx_kernels.inc` is committed.
  Regenerate it with `build.sh inc` after editing the `.cl` file.

## 6. M6 plan on the phone (only when the user releases it; to be pre-registered with the engine session)

1. **Correctness on Adreno, before any speed row.** Build `gx_test` for Android, push a subset of the
   cases, and run `ggml_ref_arm` natively on the phone (a static binary).
   - Run `gx_test` against that output. Require `BIT-EXACT`, with the same decision as on the laptop.
   - If it fails, find the mechanism (denormal flush? `fma`?) before measuring speed.
   - Also dump the full `CL_DEVICE_EXTENSIONS` string and `CL_DEVICE_SINGLE_FP_CONFIG`.
2. **Dispatch latency**, from `gx_dispatch` until `gx_wait` returns:
   - for k = 1..8 and both down types, weights in an `ALLOC_HOST_PTR` pool;
   - median and p90 over at least 200 dispatches, with CPU caps and the GPU clock logged.
   - This is the number that decides K: it must stay below the CPU time the same experts would take.
3. **Throughput variants that keep the arithmetic identical.** The lane mapping is free; only each
   accumulator's block order and the reduction tree are fixed.
   - Candidates: one work-item per row holding all 8 accumulators with 18- or 20-byte block loads;
     2–4 rows per work-item; the `trans4` layout converted at read time.
   - Each candidate must pass the same test and controls before its speed is read.
   - Reference points: zcbench's native kernel (10–12 GB/s) and the capped CPU (about 23 GB/s).
4. **Concurrent split with Q4_0 down.** zcbench's 1.535× used Q4_1 down in every layer, which is the
   type of 6 of 48 layers. Re-measure with Q4_0 down.

## Files

| file | role |
|---|---|
| `gx.h`, `gx.cpp`, `gx_kernels.cl`, `gx_kernels.inc` | the component |
| `ggml_ref.cpp` | ggml-cpu reference: the FFN graph, and `--quant` for the quantizers |
| `gx_test.cpp` | acceptance: bit comparison, errors, division and quantizer checks |
| `make_cases.py` | random, real (GGUF by offset) and crafted cases |
| `negative_controls.sh` | 10 kernel mutations that must each be detected |
| `run_accept.sh` | everything above, into a new `results/<date>/gx_m3_<time>/` |
| `gx_pool_test.cpp` | pool allocator, dispatch from pool slots, concurrent map/unmap, dispatch-while-mapped control |
| `gx_bench.cpp` | M6 only: dispatch latency and GB/s per variant, down type and k; slot write cost |
| `cl_shim.cpp` | OpenCL entry points by `dlopen`, for the Android builds of the three programs above |
| `run_phone_m6.sh` | M6, prepared and **not run**: correctness first, then the bench |
| `build.sh` | `inc`, `host`, `arm-ref`, `android` (libgx.a plus the three M6 programs) |
