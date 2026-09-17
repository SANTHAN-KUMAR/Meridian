# Cutting the ~100 ms/token CPU compute residual on SM8845 — research memo (2026-09-17)

*AI research subagent output (Claude Opus, web search). Provenance [U] per `research/README.md`;
the memo's own "not verified" marks are kept. Reproduced verbatim except for formatting.*

**Framing note first (L3, estimability).** The 100 ms is a *residual*, not a measurement. Before
ranking levers I did the arithmetic on what it can be made of:

- 1.9 GB active weights / 100 ms = **19 GB/s**, vs **59.7 GB/s** DRAM measured on this device —
  3.1× headroom, so not DRAM-bound.
- Per core at 1.75 GHz: 4.75 GB/s = **2.71 bytes/cycle**. llama.cpp's Q4_0 dot path costs roughly
  8–10 NEON ops per 18-byte block (load, and/shift/sub unpack, 2× `sdot`/`smmla`, scale FMA) ≈
  1.8–2.25 bytes per vector op. So the kernel is issuing **~1.3–1.5 vector ops/cycle against 4
  128-bit vector ALUs** on Oryon v3 (Chips and Cheese, Oryon Gen 3: "4 128b Vector ALUs all capable
  of FMAs", chipsandcheese.com/p/qualcomms-snapdragon-x2-elite). Roughly **3× off vector issue and
  3× off DRAM simultaneously** — the classic signature of stall, not of either ceiling.
- Your own cool-clock measurement (OLMoE resident, 50.9 vs 24 tok/s) implies ~35 GB/s effective on
  this CPU. 1.9 GB / 35 GB/s = **54 ms**. So a 1.85× on the compute term is already demonstrated
  *on this device* — most of it by clocks, not kernels. Items 1 and 5 are partly the same lever and
  must not be double-counted.

## 1. What limits llama.cpp ARM Q4_0 `MUL_MAT_ID` at batch 1

**The strongest single finding in this memo is the barrier count.** llama.cpp issue #26200
(github.com/ggml-org/llama.cpp/issues/26200) profiles **Qwen3-30B-A3B** decode with a per-node +
per-barrier ledger validated to 0.3% of wall clock, on an i5-7600K, 4 threads, `-ngl 0`:

- "A MoE decode graph executes **~2,000 barriers per token**" (~1,640 inter-node barriers measured).
- libgomp build: barriers cost **30.8 ms/token = 33% of the token** (11.92 tok/s).
- `-DGGML_OPENMP=OFF` (ggml's own spin barrier): barriers still cost **10.8 ms/token**; throughput
  16.64 tok/s (**+40%**).

Scaling the *good* case to your device: 10.8 ms was measured at ~4 GHz on 4 homogeneous cores. At
1.75 GHz, and with 4 I/O lanes spinning on cores 0–3 while 4 compute threads spin on 4–7, **20–35 ms
of your 100 ms residual being barrier/sync is the arithmetic expectation, not a guess.** It also
explains the two facts you already have: 4→6 threads gave 0% (barrier cost is superlinear in
threads, and the extra threads land on I/O cores), 2→4 gave +30%.

Two immediate checks: (a) is BigMoEOnEdge built with `GGML_OPENMP=ON`? The NDK ships libomp and
CMake will pick it up; if so you may be paying the 33% case, not the 11% case. (b) `--poll` level —
ggml's threadpool polls 128K rounds per level before a condvar wait (llama.cpp discussion #21112);
spinning I/O-blocked threads is pathological in a streaming engine.

**Dequant/ALU:** not the binding constraint per the arithmetic above. **Gather latency:** plausible
contributor but unquantified anywhere I found. **Page faults:** you already ruled out the zero-fill
component via mremap.

**GB/s-effective bounds for "well-optimised CPU" on this class of core.** I could not find a clean
published Q4_0 llama-bench table for 8 Elite / 8 Gen 5 — flag as **not verified**. The nearest
usable anchors: Adreno 830 Termux run, Qwen2.5-Coder-1.5B **Q8_0: 21.3 tok/s GPU vs 7.95 tok/s
CPU** (llama.cpp discussion #23736) → ≈34 GB/s GPU, ≈12.7 GB/s CPU (unpinned, throttled Termux).
And your own 35 GB/s cool-clock figure. So **~35 GB/s is the demonstrated CPU ceiling on this
silicon class**, not 59.7.

**KleidiAI / SME2.** llama.cpp's KleidiAI decode path selects
`kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot` (Arm Learning Path,
learn.arm.com, performance_llama_cpp_sme2/kleidiai_integration) — i.e. the fast GEMV is an **SME2**
kernel, and KleidiAI only covers Q4_0/Q8_0. Whether Oryon v3 exposes SME2 is **contested**: Android
Authority on 8 Elite Gen 5 reports "SME extension, **but not the latest SME2**" (SVE2 + SME1);
low-quality aggregators claim SME2. **Not verified.** Also note KleidiAI's trait handles `MUL_MAT`;
I found **no evidence of a KleidiAI `MUL_MAT_ID` path** — so even with SME2 the routed experts would
not be accelerated without work.

Separately: issue #10662 documents **i8mm going undetected on an Android NDK build** despite
`lscpu` showing `i8mm` (Q4_0_4_4 collapsed 16.71 → 2.28 tok/s). Cheap to check on your build.

> **Status: partly demonstrated (barriers), partly theoretical (gather latency), untested on this
> device. Expected gain: 1.2–1.4× on the compute term from barrier/threadpool work alone.
> Experiment that settles it: port the #26200 per-node + per-barrier ledger into BigMoEOnEdge and
> run it pinned, plus simpleperf `-e cpu-cycles,stalled-cycles-backend,l1d-cache-refill,inst-retired`
> on the compute threads only. This is the first experiment to run; everything below is unrankable
> until it exists.**

## 2. Offline pre-repacked expert weights on flash

**History:** file-level repacked types existed (`Q4_0_4_4/4_8/8_8`) and were **deliberately
removed** in PR #10446 (Djip007, "Refactor/online repacking"), replaced by runtime repack in
`ggml_backend_cpu_aarch64_buffer_type`; issue #10757 records the resulting `TYPE_Q4_0_4_4 REMOVED`
error. The removal reason was portability + mmap, **not** that the layout was useless. Your engine
already forfeits mmap, so the original objection does not apply to you.

**Does anyone pre-repack for a *streaming* engine?** Closest precedents, both about **contiguity,
not SIMD layout**:
- llama.cpp discussion #27149: expert-contiguous GGUF relayout tool; one expert drops from touching
  6,912 pages to 192 (36× amplification reduction); M1 MBP 16 GB, 0.9 → 4.7 tok/s, per-layer I/O
  345 → 30 MB.
- **BigMoMo** (arXiv 2609.14643): runtime-adaptive on-flash expert reorganisation by hypergraph
  clustering of co-loading patterns; flash reads/token for Qwen3-30B **131.48 → 82.65**.

Neither stores a SIMD-interleaved layout. **This is genuinely unoccupied ground.**

**Expected gain — be sceptical.** The relevant measurement is arXiv 2501.00032 (Arm, "Highly
Optimized Kernels ... on Arm CPUs"): Q4_0_8_8 improves **token generation up to 2× at low core
counts**, with the gain vanishing at 64 cores where tg becomes memory-bound. At 4 cores on a phone
you are in the regime where it *should* pay. That contradicts your measured null result — which is
why the **pinned** retest matters, and why "is the repack path actually taken?" (the
`ggml_gemv_q4_0_8x8_q8_0` entry, and i8mm detection per #10662) must be asserted, not assumed. The
offline variant additionally removes the per-token repack of 194 MiB from the critical path — but
only if the repack was on compute threads. If your I/O lanes (cores 0–3, idle during compute) do
the repack, offline storage buys only the file-size/bandwidth delta, which is ~0 for Q4_0_8x8
(same bytes, permuted).

> **Status: theoretical; no cited streaming engine does it. Expected gain: 0–1.6× on the compute
> term, entirely conditional on the pinned repack retest showing a positive. Experiment: (a) assert
> at runtime which GEMV symbol is executing and whether `i8mm`/`dotprod` were detected; (b) move
> the in-place repack onto the I/O lanes and re-measure; only if (a)+(b) show gain, build the
> offline-repacked sidecar.**

## 3. Expert compute on Adreno while the CPU streams

**Zero-copy exists and is well documented.** Adreno OpenCL exposes `cl_qcom_ion_host_ptr`,
`cl_qcom_dmabuf_host_ptr`, `cl_qcom_android_ahardwarebuffer_host_ptr`,
`cl_qcom_android_native_buffer_host_ptr`, `cl_qcom_ext_host_ptr_iocoherent` (Khronos registry,
cl_qcom_ion_host_ptr.txt; IWOCL 2022 Qualcomm extensions overview). An O_DIRECT read needs a
page-aligned destination anyway, so landing it in an ION/dmabuf allocation wrapped as a CL buffer
is mechanically compatible. **llama.cpp's OpenCL backend does not use these** — OPENCL.md contains
no zero-copy/host-pointer discussion, which is exactly why PR #25294's GPU expert cache did
`clEnqueueWriteBuffer` copies and landed at 1.5–1.9 tok/s.

**Adreno MoE kernels.** `MUL_MAT_ID` for Q4_0/Q4_1/Q4_K/MXFP4 exists in llama.cpp, but **only in the
prebuilt Adreno binary kernel library, which targets X2-90/X2-85/X2-45 (Snapdragon X2), not Adreno
840** (OPENCL.md). Adreno 840 is listed as a *verified device* for the generic path. So: generic
OpenCL `MUL_MAT_ID` yes; tuned Adreno MoE kernels no.

**The structurally right design is published but only on desktop.** Discussion #24528 proposes an
*inverted* hybrid: keep `MUL_MAT_ID` on CPU, have **thread 0 dispatch cache-hit rows to the GPU while
the other threads compute miss rows** — worst case degrades to vanilla, and crucially it does not put
a synchronous transfer on the critical path. Measured +10–57% across 16 models on desktop; but the
matched-budget control is damning (RTX 5060 Ti, Qwen3-30B, 12.5 GB both ways: resident 35.4 tok/s
vs cache 19.7 tok/s at 85.3% hit). On a phone there is **no PCIe** — the entire "convert a
bandwidth problem into a PCIe-latency problem" argument changes character, and your 78% hit rate
maps directly onto the hit/miss split.

Arithmetic for your device: 0.87 GB dense + 0.78×1.02 = 0.80 GB from GPU-resident memory at the
34 GB/s Adreno figure ≈ **49 ms**, concurrent with 0.22 GB of misses on CPU (~12 ms) and the 50 ms
I/O stall. Landing zone ~60–70 ms wall. **~1.4–1.6×, not 2× alone** — but it composes with §5, and
the Adreno is the *thermally* better unit on your own data (46–47 tok/s flat across drift vs
50.9→24 on CPU).

**No engine found** that runs routed experts on Adreno with experts arriving from storage at
runtime. Flash-MoE/DynaMoE/ds4-ssd are all Metal/Apple Silicon.

> **Status: mechanism demonstrated (zero-copy extensions, hybrid hit/miss on desktop); the
> combination untested on Adreno. Expected gain: 1.4–1.6× on the compute term, plus better thermal
> stability. Experiment: build a micro-benchmark that (1) allocates an ION/dmabuf buffer, (2)
> O_DIRECTs an expert slice into it, (3) wraps it with `cl_qcom_dmabuf_host_ptr`, (4) runs the
> generic ggml-opencl Q4_0 MUL_MAT_ID over it, and reports GB/s. If it does not clear ~25 GB/s,
> stop — the hybrid will not pay.**

## 4. Hexagon HTP v81 for decode

**The decisive published number is negative for you.** BigMoMo (arXiv 2609.14643) runs exactly
your workload — **Qwen3-30B-A3B Q4_0, streamed from flash, experts on HMX, on a OnePlus 15
(Snapdragon 8 Elite Gen 5)** — and reports **304.4 ms/token** (Table 9), vs 552.6 ms for the best
speculative baseline. **Your 187 ms/token on a lesser SoC is 1.6× faster than the state-of-the-art
NPU paper on a better SoC.** They achieve this with hypergraph flash reorganisation, speculative
multi-token verification windows, and HTP utilisation driven from 18.96–45.03% up to 83.78–98.30%
— i.e. they *saturated* the NPU and still landed slower, because the path is flash-bound, and their
cost model `Te = TeFlash + TeMap + TeVtcm + TeHMX` adds two movement stages (`TeMap` into NPU address
space, `TeVtcm` into the 8 MiB scratchpad with layout conversion) that the CPU path does not pay.
They publish no per-stage values.

Supporting evidence: llama.cpp's own Hexagon backend gets **12 tok/s on Qwen3-4B Q4_0 on 8 Elite** vs
Qualcomm's proprietary HTP engine at 26 (discussion #21702), and the Snapdragon backend README shows
a **~3.5 GB virtual address space limit per NPU session**, requiring dynamic buffer remapping or
multi-session splits — a per-token remap treadmill for a streaming engine. It does use
dmabuf/rpcmem, so *some* zero-copy is available, but "the CPU fills a host mmap and the HTP reads it
with no map cost" is **not verified** and BigMoMo's explicit `TeMap` term implies otherwise.

> **Status: demonstrated negative. Expected gain on this device: ≤1× (likely a regression).
> Experiment that would settle it: measure `TeMap` alone — time `rpcmem`/dmabuf registration of a
> 2.65 MB freshly-O_DIRECT'd buffer into an HTP session, 1000 iterations. If it exceeds ~0.3 ms per
> expert, the path costs >24 ms/token in mapping before any compute and is closed. This is a 1-day
> check that closes a large branch; worth doing for that reason alone.**

## 5. Clocks and thermal — the highest-value, least-glamorous item

**Your cap is probably not thermal.** The best source is
github.com/wyl2607/oneplus13-performance-investigation (OnePlus 13, Snapdragon 8 Elite), which
identifies two OEM kernel modules:

1. **`cpufreq_bouncing` (CFB)** — clamps via `freq_qos` after **50 ms of sustained load**: mid
   cluster 3,532,800 → **2,400,000 kHz (68%)**, prime 4,320,000 → **2,438,400 kHz (56.4%)**. Because
   `freq_qos` takes the *minimum* of all requests, **writing `scaling_max_freq` has no effect even as
   root**.
2. **`oplus_bsp_task_overload`** — clamps `uclamp.max` on threads it deems "abnormal"; a traced
   worker carried `uclamp.max = 466` against mid-cluster capacity 792, so the scheduler never
   promoted it to prime. Evidence in `/proc/task_overload/abnormal_task`.

Measured impact: Geekbench 7 single-core **947 → 1,234 (+30%)** with CFB disabled; junction stayed
40–50 °C against a 105 °C trip, thermal status **0**, all 20 cooling devices `cur_state=0`. **That is
your "capped at 1.75 GHz at thermal status 0" phenomenon, named.** Note `466/792 × 3.53 GHz ≈ 2.08
GHz` — suspiciously close to your ~2.0 GHz observation.

**Bad news:** the investigation concludes an unprivileged app can do **nothing**; all workarounds
need root. Genuine thermal throttling is real too and severe on this generation: 3DMark stability
**62.3% on 8 Gen 5** over 20 loops, 25% on 8 Elite Gen 5 (Beebom); a 30-min Snapdragon 8 Gen 3 LLM
run went **12.4 tok/s @ 3.3 GHz → 3.8 tok/s @ 1.8 GHz, −69%** (MVP Factory blog).

**Unprivileged levers actually worth testing** (all **untested**, none guaranteed): (a) an ADPF
`PerformanceHintManager` session over the compute TIDs — the framework's mechanism is `uclamp.min`
boosting via PowerHAL (AOSP Performance Hint API; Arm/LPC UtilClamp on Android), which may or may not
out-rank an OPLUS `uclamp.max` clamp — arithmetically it cannot, since uclamp takes min(max-clamps),
so **expect this to fail and say so if it does**; (b) `Window.setSustainedPerformanceMode(true)` — by
design this *lowers* peak for consistency, so it is a stability lever, not a speed lever; (c)
OnePlus **High Performance Mode** toggle + adding the app to Game Space, which is the OEM's own
allow-list and the only mechanism that plausibly bypasses `task_overload`'s "abnormal task"
heuristic; (d) **avoid looking abnormal**: the guard triggers on sustained ~100% single-thread load,
and a streaming engine whose compute threads idle 50 ms/token may already dodge it — worth checking
whether your threads appear in `/proc/task_overload/abnormal_task` (readable only with root, so use
a rooted reference device if you have one).

> **Status: mechanism demonstrated on a sibling device; app-side mitigation untested and
> theoretically weak. Expected gain if the cap lifts to hardware max: 1.8–1.9× on the compute term
> (your own 24 → 50.9 tok/s OLMoE measurement is the evidence). Experiment: log `scaling_cur_freq`
> per core at 10 Hz alongside decode, under four conditions — default, High Performance Mode on,
> ADPF session registered, app in Game Space — and report the frequency histogram, not the tok/s.
> If frequency does not move, the tok/s cannot.**

## 6. Unconventional 2025–2026 work

**LUT GEMV (T-MAC).** arXiv 2407.00088 / microsoft/T-MAC. Published 4-bit numbers on Snapdragon X
Elite (Surface Laptop 7): **12.6 tok/s at 2 cores, 18.7 at 4 cores, vs 10.4 on the NPU**. Headline
speedups (2.8× for Llama-2-7B-4bit) are against llama.cpp **on a Raspberry Pi 5** — a core without
competitive `sdot` throughput — so they do not transfer to Oryon. The README explicitly hedges on
"4-bit token generation" speedup. *However*, the structural fit to your engine is unusually good:
T-MAC's LUT is built per activation group and amortised across output rows; a streamed expert is a
2048-row matrix used exactly once, so LUT construction is ~0.05% overhead, and the inner loop
becomes `tbl` (table lookup) instead of unpack+`sdot` — which is the right direction if §1's profile
says the unpack is the stall. **Theoretical for this device.** Related, newer, ultra-low-bit only:
Vec-LUT (arXiv 2512.06443); T-MAN (arXiv 2511.11248, LUT on NPU).

**Activation-magnitude row skipping.** arXiv 2511.04477 ("Enabling Dynamic Sparsity in Quantized
LLM Inference") reports **Llama-3.1-8B at 50% sparsity: 153.5 → 179.9 tok/s (+17.2%)**, Mistral-7B
+21.4%, Qwen-2.5-7B +21.2% — but on Apple-silicon GPUs, and this is a **lossy routing change in
substance** even if not in name, so it is out of scope per your constraint. Noted only so it is not
re-proposed.

**BigMoMo's flash reorganisation** (§2) is the most transferable unconventional idea in the 2026
literature, and it attacks I/O, not compute.

> **Status: T-MAC theoretical; sparsity out of scope. Expected gain from LUT GEMV: unknown,
> plausibly 1.2–1.5× *if and only if* §1's profile attributes ≥30% of the residual to the
> unpack/dequant instruction stream. Experiment: a standalone microbenchmark of `tbl`-based vs
> `sdot`-based Q4_0 GEMV on one pinned core at fixed frequency, reporting bytes/cycle for each.
> ~2 days, no engine changes.**

## Ranked top-5 interventions for this device

| # | Intervention | Evidence class | Expected gain on the compute term |
|---|---|---|---|
| 1 | **Instrument before optimising**: port the #26200 per-node + per-barrier ledger, plus simpleperf backend-stall/L1D counters, into the pinned config. Then check `GGML_OPENMP` state and `--poll`. | **Demonstrated** that barriers cost 10.8–30.8 ms/token on this exact model's graph (#26200); untested here | 1.2–1.4× if barrier-bound; and it is the only thing that makes 2–5 rankable rather than guessed |
| 2 | **Frequency-cap forensics** (§5): log per-core `scaling_cur_freq` under High Performance Mode / Game Space / ADPF session. | **Demonstrated** mechanism on OnePlus 13 (CFB + `task_overload`); app-side fix **untested and theoretically weak** | 1.8–1.9× if the cap lifts; honestly, most likely 1.0× — but it is the largest measured gap on your own device and is cheap to falsify |
| 3 | **Adreno hybrid hit/miss** (§3): resident 78% on GPU via `cl_qcom_dmabuf_host_ptr` zero-copy, misses on CPU, preserving read/compute overlap — the #24528 inversion, adapted to a unified-memory SoC. | Zero-copy extensions **demonstrated**; hybrid inversion **demonstrated on desktop**; combination **untested** | 1.4–1.6×, and thermally the better unit on your own data |
| 4 | **Repack, properly controlled** (§2): assert which GEMV symbol runs and whether i8mm was detected (#10662 risk), move the repack onto idle I/O lanes, retest pinned; only then consider an offline-repacked sidecar. | Arm's 2× tg-at-low-core-count claim (2501.00032) is **demonstrated**, on server cores; your null result is **demonstrated** on an uncontrolled run — these conflict, which is the reason to redo it | 0–1.6×; the conflict itself is the finding |
| 5 | **Close the NPU branch with one measurement** (§4): time dmabuf/rpcmem registration of a freshly-read 2.65 MB buffer into an HTP session. | BigMoMo's **304.4 ms/token on a better SoC for your exact model** is a demonstrated negative; the map cost is **not verified** | ≤1×. Ranked here because a 1-day experiment that *closes* a branch is worth more than a speculative one that opens three |

**Explicitly not re-proposed:** draft speculation (your c(N) measurements kill it), cross-token
expert prediction, stale-gate prefetch, more threads or lanes, mremap page recycling.

**Two things I could not verify and you should not quote as fact:** (i) whether SM8845's Oryon v3
exposes **SME2** (sources conflict; KleidiAI's llama.cpp GEMV is an SME2 kernel, and there appears to
be **no KleidiAI `MUL_MAT_ID` path** regardless) — settle it with one `getauxval(AT_HWCAP2) &
HWCAP2_SME2` call; (ii) any published llama-bench Q4_0 tok/s table for 8 Elite / 8 Gen 5 from which
to derive a clean GB/s-effective bound — I found none, so §1's ceiling rests on your own 35 GB/s
cool-clock measurement and a 34 GB/s Adreno 830 third-party datapoint.
