# GPU zero-copy expert micro-benchmark (`zcbench`) — question, decision rule, how to read it

**Status (2026-09-18): built (arm64 + x86) and self-tested on the laptop; NOT yet run on the phone.** The
decision rule below is fixed before any phone row exists. Goal (user, via the phone session):
Qwen3-30B-A3B, unmodified, at 10 tok/s. The measured blocker is CPU compute at the governor's clock
caps (about 121 ms/token of 190). The only route in the data that could take that term below 100 ms is
the Adreno GPU computing expert matmuls, provided an expert read from flash can reach the GPU without
an upload copy.

## What it answers

| | question | how |
|---|---|---|
| (a) | Can an expert slice read from flash be handed to OpenCL without a copy? | Four mechanisms: `copy` (device buffer + `clEnqueueWriteBuffer`, the baseline), `use_host_ptr` (page-aligned host memory), `alloc_host_ptr` (driver memory, mapped, O_DIRECT into the mapping), `ion_dmabuf` (a `/dev/dma_heap` buffer imported with `cl_qcom_ion_host_ptr`). Per mechanism: does a map return the pointer the bytes were read into; after new bytes are `pread` into that memory, does the kernel see them with no sync and with only a map/unmap; and what does that sync cost for one expert slice and for the whole region, against a `memcpy` of the same bytes. A copy scales with bytes; a zero-copy handoff does not. |
| (b) | GPU expert GEMV throughput on Qwen3 shapes | gate/up q4_0 k=2048 n=768 and down **q4_1** k=768 n=2048 (the file's own down type), 8 of the region's experts per layer, n = 1. Weights stay in the file's native block layout (18-byte q4_0, 20-byte q4_1 blocks), which is what zero-copy of file bytes gives. Reported per layer with a `clFinish` per layer (as an engine must) and batched (48 layers, one finish). Work-group width is chosen from 16/32/64 on correctness first, then speed. |
| (c) | CPU and GPU computing disjoint experts at the same time | CPU = ggml's own `MUL_MAT_ID` (the engine's kernels, 4 threads strictly pinned to cores 4–7); GPU host thread pinned to core 3. Every iteration is time-stamped inside its worker; aggregate GB/s counts only iterations inside the window where both ran; overlap is reported as the share of each worker's busy time during which the other was also busy. CPU-alone runs before and after bracket drift. |
| (d) | Correctness | Every GPU and CPU result is compared with a double-precision scalar reference over the same bytes; max \|error\| / max \|reference\|. GPU must be < 1e-3; ggml quantises activations to q8, so its error is reported but not held to that bound. |

## Decision rule (fixed before any phone run)

The GPU route is **viable** only if all four hold, in one run, with the CPU clock caps sampled
every second (`CLKS` lines, tagged with the phase in progress) and the median cap of each CPU
policy identical across the compared phases (CPU alone, concurrent, CPU alone again):

1. **Zero-copy:** at least one of `use_host_ptr`, `ion_dmabuf` or `alloc_host_ptr` reports
   `zero_copy=1`, is correct after an in-place update from flash, and its whole-region sync runs at
   ≥ 10× the `memcpy` rate of the same bytes. That is, the handoff costs under 10% of a copy.
2. **Correct:** GPU `max_rel_err < 1e-3` on gate, up and down for the mechanism used in (c).
3. **Concurrent gain:** `aggregate_GBps ≥ 1.5 × cpu_alone_GBps`, with `cpu_alone_GBps` the mean of
   the two bracketing CPU-alone runs.
4. **Proven overlap:** `overlap_frac_of_cpu_busy ≥ 0.8`. Without that, an aggregate could come from
   time-slicing.

Otherwise the route is **not viable**, and this note will say which condition failed. The 1.5×
threshold comes from the phone session's request. The 10× and 0.8 figures are set here, before data.

## What was found before running anything (read from source, 2026-09-18)

- **The per-op sweep's "OpenCL gets ffn_down wrong" did not test the engine's path.** ggml-opencl
  uses its Adreno MoE kernels (`kernel_gemv_moe_q4_{0,1}_f32_ns`, with a transposed weight layout
  produced by `kernel_convert_block_q4_*_trans4_ns`) only for tensors whose name contains `ffn` and
  `exps`, or `as` (`use_adreno_moe_kernels`, ggml-opencl.cpp). The sweep's `ggml_matmul_bench.cpp`
  names its weights `weight0`, `weight1` and so on, so it ran the generic `MUL_MAT_ID` fallback. That
  fallback is what disagreed on Q4_1 (the down type). The engine's tensors are named `ffn_down_exps`
  and would take the Adreno path. So the wrong result is real for the fallback and says nothing about
  the Adreno MoE path.
- **ggml's Adreno MoE path cannot be zero-copy as written:** it converts each expert tensor to a
  transposed struct-of-arrays layout on the GPU after upload. Zero-copy of file bytes therefore needs
  either a kernel on the native layout (what `zcbench` uses), or that transposed layout stored on
  flash per expert (possible: the transpose is within an expert). `zcbench` measures the native-layout
  kernel. If it is too slow, the stored-transposed layout is the next variant, not a reason to give up.
- **Tensor names do not matter to `zcbench`'s GPU arm.** Its GPU kernels are its own (native-layout
  GEMV), not ggml-opencl, so ggml's name gate never applies. The ggml CPU tensors carry the engine's
  names (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`). Whether ggml's *Adreno MoE* path is correct
  on this phone is a separate check, cheapest as a rerun of `host/matmul_sweep.sh` with the weights
  named like the engine's (`blk.N.ffn_down_exps.weight`).
- Qualcomm's dma-buf and AHardwareBuffer host-pointer extensions have no public spec (the Khronos
  registry has neither). `zcbench` uses the published `cl_qcom_ion_host_ptr` with a dma-heap fd, and
  prints the device's extension list so a missing path is visible.

## Running it on the phone (only after the phone session clears a slot)

```
host/gpu_zerocopy/build.sh android
host/gpu_zerocopy/run_phone.sh            # pushes, creates the 264 MB weight file once, runs, pulls
python3 host/gpu_zerocopy/analyze.py results/<date>/zcbench/zcbench.out
```

About 2–3 minutes of phone time. Memory: two 132 MB regions per mechanism tested (≤ ~800 MB). No model
file is touched.

## Laptop self-test

`build.sh host` then `out/host/zcbench --make-file --experts 16 --secs 1 --file <scratch>/zc.bin` on the
laptop's NVIDIA OpenCL. That checks the kernels and the harness, not Adreno, and its throughput
numbers mean nothing for the phone.

Result on 2026-09-18 (RTX 4060 laptop, 16 experts per region, 1 s phases):
- GPU kernels agree with the double-precision reference to about 1.5e-7 on gate, up and down, at
  work-group widths 16, 32 and 64. ggml's CPU result agrees to 5.2e-3, from its q8 activations.
- The zero-copy test behaves as intended on a copying driver. With `use_host_ptr` the map returns the
  same pointer, but an update from flash is invisible to the kernel until a map/unmap (error 1.46
  without, 1.6e-7 with). That sync runs at copy speed (12.7 GB/s against `memcpy` at 18 GB/s), so
  `handoff=1 zero_copy=0`, and `analyze.py` returns **NOT viable: failed 1 zero-copy**. That is the
  right answer for a discrete GPU, and it shows the rule can fail. `ion_dmabuf` reports
  "unavailable" (no Qualcomm extensions).
- The concurrent phase runs fully overlapped (overlap 1.0) at about 2× the CPU alone, as expected
  with separate CPU and GPU memories. The phone, where both share one memory bus, is the actual test.
- Two harness bugs were found and fixed on the way. Activations were never uploaded (every output
  was 0). And `CL_MAP_WRITE` reads the stale device copy back over freshly read bytes on a copying
  driver; all maps now use `CL_MAP_WRITE_INVALIDATE_REGION`.
