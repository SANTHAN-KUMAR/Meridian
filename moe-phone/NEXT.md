# moe-phone — where the work stands, and the next step for whoever picks it up

**Updated 2026-09-16.** This is a handoff note, not a results document: every number lives in an
artifact and in [`CLAIMS.md`](CLAIMS.md). The gate record is [`README.md`](README.md).

## Done in the 2026-09-16 session (all committed)

| step | outcome | where |
|---|---|---|
| G-REPRO | every committed artifact re-derives bit for bit from committed inputs | `gates/repro_check.py` |
| S1 | DRAM bandwidth measured as the app; the 45 GB/s assumption was low; >4 busy threads are never clean | `dram_15r.json` |
| engine_target | three defects fixed (resident models, clamped curve, flash-only); no model reaches double digits with DRAM charged | `engine_target.json` |
| G-VALID-3 | the flash-only formula over-predicts an independent engine (colibri) by ~3x | `external_validation_colibri.json` |
| S10 (sim side) | speculation priced with draft/verify compute as a swept parameter | `engine_sim --fwd-ms` |
| S9 | pre-registered (committed before any trace) | `s9_prereg_Qwen3-30B-A3B.json` |
| docs | stale claims deleted: "double digits", G-VALID(1) "within 12%", the unsourced 0.693, A4/A5/§9/S2 | `git log` |

## Running or blocked

0. **Done since this note was first written:** the phone campaign ran clean (G6-baseline, S11,
   G1-QD in `README.md`). A second, CPU-time-instrumented campaign (`device/phone_campaign_cpu.sh`,
   tmux session `moe2`, output `~/moe/out2_*`) ran clean: 2 threads 30.1 tok/s vs 4 threads 8.0
   for warm OLMoE, no flash traffic to explain it. Campaign 3 (`phone_campaign_threads.sh`, tmux
   `moe3`) found a cliff: warm OLMoE 20.8 / 17.4 / 2.3 / 2.1 tok/s at 1-4 threads, resident granite
   scaling normally (`decode_15r_threads.json`). **Next on the phone:** explain the cliff — log
   majflt (field 12 of /proc/pid/stat) and per-thread CPU time (/proc/pid/task/*/stat) at 2 vs 3
   threads, and repeat with a model that fits with room to spare. Busy-waiting threads mean CPU
   share alone cannot tell stalls from work.
1. **Phone campaign** (`device/phone_campaign.sh`, in Termux `tmux` session `moe`): OLMoE Q4_0
   cold/warm, granite-1b-a400m resident, pp512, ufsbench 16/32 threads. Output on the phone in
   `~/moe/out_<date>_<time>/`. Pull it, then:
   ```
   python moe-phone/gates/decode_analyze.py moe-phone/results/<date>/decode_15r/ \
       --model-bytes olmoe-1b-7b-0924-q4_0.gguf=3928036960 granite-3.1-1b-a400m-instruct-Q4_0.gguf=771466560
   ```
   **Before trusting any run, check `results/2026-09-16/phone_state_log.csv`**: a run that overlaps
   a non-`Awake` state, `keyguard=true`, a non-Termux focus, or an `INTERVENTION` line is
   contaminated. The first campaign (00:23) was, and is kept only as evidence.
2. **S9 is answered for one geometry (2026-09-16): the equal-rho transfer is REFUTED**
   (`s9_result_granite-3.1-1b-a400m.json`). The top open item is now the **family confound**:
   granite differs from OLMoE in family as well as in E/k. The pre-registered separator is
   **gpt-oss-20b** (E=32, k=4 — OLMoE's E/k in a different family), where both hypotheses predict
   the same value, so any gap is family, not geometry. It needs a 12 GB download, and on
   2026-09-16 the project chose **no further downloads** on the ~384 KB/s link, so this is
   deferred by decision: the confound stands as a stated limitation in `POSITION.md` and
   `README.md` rather than as an unnoticed gap. Two ways to close it when bandwidth allows, in
   order of value: gpt-oss-20b (12 GB, decisive), or granite-3.1-3b-a800m (1.9 GB, E/k=5 — a
   within-family geometry step against the granite already measured, indirect but cheap).
   The rest of the original item: the control half is DONE (2026-09-16): tokens, the OLMoE Q4_0 trace through
   llama.cpp, its f_crit curve and the confound score all exist, and the control passes
   (0.0013 against a 0.031 limit). What remains is the target: resume
   `unsloth/Qwen3-30B-A3B-GGUF` Q4_0 (9.6 GB left of 17.4, resumable from
   `moe-work/models/.cache`), then:
   ```
   python moe-phone/gates/llamacpp_traces.py tokens --model Qwen/Qwen3-30B-A3B --out tok_qwen3.bin
   moe-work/route_trace <qwen gguf> tok_qwen3.bin qwen3.trc 20
   python moe-phone/gates/llamacpp_traces.py convert qwen3.trc --tag Qwen3-30B-A3B-q4_0-llamacpp ...
   python moe-phone/gates/cache_sim.py <npz> --fractions-around-crit --out-name cache_fcrit_Qwen3-30B-A3B
   python moe-phone/gates/s9_score.py --prereg .../s9_prereg_Qwen3-30B-A3B.json \
       --target-curve .../cache_fcrit_Qwen3-30B-A3B.json \
       --confound-curve .../cache_fcrit_OLMoE-1B-7B-0924-q4_0-llamacpp.json
   ```
3. **S11** — the on-device forward-pass time comes out of step 1 (granite resident, OLMoE warm);
   feed it to `engine_sim --fwd-ms` and to a compute column in `engine_target`.

## State at the end of the 2026-09-16 morning session

- **S9 is answered and negative** (see above). Every per-model tok/s through `at_rho()` is now a
  provisional upper estimate, and the floor-additive column is the better one for large models.
- **Both remaining "levers" were priced at the measured compute cost and shrank**: speculation to
  1.06–1.07x (negative if the verify is compute-bound), within-token prefetch to 0.76–0.99x.
- **The phone is disconnected** (`adb: no devices`). The open on-device experiment is the thread
  cliff: log majflt and per-thread CPU at 2 vs 3 threads (`NEXT` item above).
- **Downloads are stopped** at the user's direction: the laptop is on a ~384 KB/s link. A 431 MB
  partial of Qwen3-30B-A3B Q4_0 is kept in `moe-work/models/` and resumes with
  `curl -C -`. The pre-registered family-confound separator (gpt-oss-20b, 12 GB) is unfetched.

## OVERNIGHT 2026-09-17/18 — NPU IS OPEN; what runs unattended

**The NPU works, unrooted, and it needed two conditions (both verified):**
1. the engine must run in the APP's OWN process. A binary the app spawns, `adb shell`, and `run-as` are all
   refused at the vendor HAL (errno 13 -> AEE_ECONNREFUSED 0x72) even with `-ngl 0`, because the backend opens
   the session at registry init;
2. the app must declare the vendor libraries: `<uses-native-library libOpenCL.so / libcdsprpc.so /
   libadsprpc.so>` (they are in /vendor/etc/public.libraries.txt).
Harness: results/2026-09-17/npu_inprocess (AndroidManifest.xml, Probe.java, Run.java, npu_probe.cpp). The JNI
shim sets ADSP_LIBRARY_PATH / GGML_HEXAGON_ARCH in-process and calls `llama_bench(argc,argv)` or
`llama_completion(argc,argv)` by dlsym, so OUR engine stays the core and the app only opens the door.
Gotcha found the hard way: build the Android libs with `-DANDROID_STL=c++_shared`, otherwise the app's
libc++_shared and the libs' static libc++ free each other's pointers ("Pointer tag ... was truncated", SIGABRT).

**First numbers (in-app):** OLMoE Q4_0 resident on HTP0 pp32 197.8 / tg32 40.7 tok/s (Adreno 102.2 / 47.6;
4 pinned CPU cores 55.5 / 26.9 throttled). Qwen3-30B-A3B Q4_0 STREAMED from flash with a 24-slot expert cache
on HTP0: coherent output, 4.88 tok/s over 31 tokens from a cold cache (same engine: GPU 1.9, CPU 0.39).

**Running unattended (laptop driver `moe-work/overnight_npu.sh`, results in results/2026-09-18/app_engine):**
Qwen3-30B on HTP0 with 24 and 32 slots, on GPUOpenCL 32 slots, on CPU 32 slots, 128 tokens, 2 repeats each;
then OLMoE on all three devices; then it hands the phone back to `bmoe_mem.sh` (row-stream / smaller context
/ both, against the 5 GB reference). Each row carries its own thermal state file.

**Read the results like this:** the in-app engine is llama.cpp PR #25294, whose CPU path is ~10x slower than
BigMoeOnEdge's (no read/compute overlap), so compare NPU against GPU/CPU *within* these rows, and compare the
best in-app NPU number against BigMoeOnEdge's 6.48-7.25 tok/s separately - different engines.

**Next levers, in order:** (1) if HTP decode scales with slots, raise the cache and re-measure; (2) move the
output head (10.6 ms/token, claim trace_output_head_ms) and attention onto HTP/GPU while experts stream;
(3) port the NPU path into BigMoeOnEdge's overlap engine, which is the only engine that hides I/O.

## OVERNIGHT 2026-09-18 — OUR ENGINE ON THE NPU, and the ceiling question

**Patch 0009 (committed): BigMoeOnEdge itself now runs in-app and can place work on another device.**
`bmoe_main()` is the CLI as a shared-library symbol, so the app's JNI shim dlsyms the engine instead of
exec'ing a child (a child is refused by the vendor HAL). `--attn-device NAME` puts attn_q/k/v/output on
HTP0 or GPUOpenCL through llama.cpp's `tensor_buft_overrides` while the routed experts keep streaming
into host memory and computing on the CPU under `--overlap`. Verified before any rate was measured: an
unknown device name fails the run; `--attn-device CPU` reproduces the unset baseline's generated text
byte-for-byte; the in-process probe reports `HTP0 init OK`; `bmoe~~--version` returns through the shim.
Found while reading the path: the dense-weight policies would have rebound a device-resident tensor's
`data` pointer into our anon arena — such tensors are now excluded and counted. The output head cannot
move (Q6_K, and ggml-hexagon refuses `ne[1] > 32768`).

**Reproducing the app path: `moe-phone/host/build_npu_app.sh` (engine -> JNI -> APK -> install -> probe),
with the app's whole source in `moe-phone/host/app/`.** Before 0009 this existed only in shell history.

**THE CEILING QUESTION, which now drives the order of work.** A token of Qwen3-30B-A3B reads a fixed
1840.0 MB of weights (`results/2026-09-18/gguf_active_qwen_olmoe.json`), so a decode rate is equivalent
to a weight-byte throughput, and each device's own throughput is a hard ceiling on what any engine can
reach with that device alone. `gates/device_bandwidth.py` computes those from tonight's resident-model
(OLMoE) rows; `gates/app_engine_analyze.py` produces the rates it reads. The measured single-device
numbers are all well under the SoC's DRAM peak, which makes one question decisive:

> **Do two compute devices ADD weight-byte throughput on this SoC, or share one bottleneck?**

If they add, splitting the expert FFN across CPU + GPU (+ HTP) is the route to 10 tok/s and worth a
patch; if they share, no amount of engine work gets there and that is the honest finding.
`moe-phone/host/agg_bandwidth.sh` measures it directly: the same resident model decoding on two devices
at once vs each alone, with start/end stamps per row (a pair's sum is a lower bound, which is the safe
direction for a "they do not add" conclusion).

**Queued unattended, in this order (laptop drivers under `moe-phone/host/`):**
| # | step | script | produces |
|---|---|---|---|
| 1 | upstream llama.cpp in-app: Qwen3 on HTP 24/32 slots, GPU, CPU, then OLMoE on all three | `overnight_npu.sh` | `results/2026-09-18/app_engine/` |
| 2 | copy Qwen3 into the new package, then OUR engine in-app: `--attn-device` CPU vs HTP0 vs GPUOpenCL, 3 rotated repeats, flags identical to the best measured cell | `chain_attn.sh` -> `bmoe_attn_ab.sh` | `results/2026-09-18/bmoe_attn_ab/` |
| 3 | oversized ZRAM-backed expert cache (5000 vs 8000 vs 10000 MiB, `--force-cache`), with `/proc/vmstat` swap counters per row so a row that did not swap cannot be reported as one that did | `chain_after_ab.sh` -> `device/bmoe_zram.sh` | `results/2026-09-18/bmoe_zram/` |
| 4 | do two devices add bandwidth? | `chain_after_ab.sh` -> `agg_bandwidth.sh` | `results/2026-09-18/agg_bandwidth/` |
| 5 | the memory campaign queued earlier (row-stream / smaller context / both) | `device/bmoe_mem.sh` | pulled on resume |

**Why ZRAM is worth a cell:** the phone reports 11.4 GB of RAM and a 12.6 GB ZRAM swap device with most
of it free. The cache budget is currently chosen to fit MemAvailable, which caps the hit rate; Q4_0
pages barely compress, but zram stores an incompressible page raw, so the overflow becomes a RAM-to-RAM
fault instead of a UFS read. It can still lose, because a swap fault is synchronous in the compute
thread while a flash miss is read by the I/O lanes and overlaps compute — hence a measurement.

**Analysis to run on resume:** `app_engine_analyze.py` on both result trees (`--producer bmoe
--reference attn_cpu` for the A/B), `device_bandwidth.py` on the OLMoE rows, `bmoe_analyze.py
--swap-log` on the ZRAM campaign, then claims + repro gates, then this tracker.

## LIVE TRACKER (updated as results land) — 2026-09-17 evening

### Phone queue (phone-side: phone_queue4 -> 5 -> 6; log /data/local/tmp/moe-stream/phone_queue.log)
| # | step | script | status |
|---|---|---|---|
| 1 | GPU slot cache speed vs CPU, same engine | gpu_speed.sh | DONE -> results/2026-09-17/gpu_speed (GPU 1.5-1.9, CPU 0.17-0.39 tok/s) |
| 2 | rotated confirmation, pin/recycle, 5 cells x 4 | bmoe_pin2.sh | DONE -> bmoe_pin2.json: pinned 5.36 median (best), unpinned 4.75; recycle a net loss (retracted) |
| 3 | CPU vs Adreno vs Hexagon HTP v81 compute, OLMoE | npu_compare.sh | DONE -> npu_compare/: GPU tg64 34.6/47.1/47.6 vs pinned CPU 23.8/30.4/26.9 (throttled); NPU 'failed to load model' x3 -> verbose diagnostic after gpt-oss-20b |
| 4 | verify cost c(N), 3 reps | bmoe_verify_cost.sh | DONE -> verify_cost.json: c(2)=1.71 c(3)=2.37 c(5)=3.71 |
| 5 | repacked kernels on pinned (no recycle), 3 reps | bmoe_repack_pin.sh | DONE -> bmoe_repack_pin.json: 4.94 vs 5.39 tok/s — repack lever DEAD (both unpinned and pinned) |
| 5a | gpt-oss-20b (12.1 GB > RAM) pinned vs unpinned, 2 reps | bmoe_gptoss20b.sh | DONE -> bmoe_gptoss20b.json: 4.21-4.39 tok/s; pinning inconclusive (n=2). Then queue7 resumed: defer -> lanes -> cache |
| 5b | --defer-evict correctness (ppl) + speed vs pinned | bmoe_defer.sh | queued |
| 6 | I/O lanes 4/6/8 (pinned, no recycle) | bmoe_lanes.sh | queued |
| 7 | cache ceiling 4/5/6 GB | bmoe_cache.sh | queued |
Every row: quiesce (third-party apps force-stopped), wake, thermal status, cpufreq caps logged.
Queue order on the phone now (22:40): arena (running) -> boost (perfmode / predict-prefetch / floor768 / ub64) -> profile (simpleperf E1).
Latest results: cache 5 GB = 6.20 tok/s median (best; claims cache5000_*); lanes no effect; defer-evict lossless but -30%;
arena lossless, mgmt 0.034 -> 0.002 s/token but residual +10 ms (early reps); Qwen3 cache sim matches phone
(86.4 vs 85.3% at 5 GB), Belady 94.6% at the same size. Phone under AC since 21:48: thermal status 0, cpufreq caps
unchanged (governor, not temperature). NPU 0x72 fix: research agent running.

**Housekeeping when the queue is done:** restore the phone's screen timeout (`settings put system screen_off_timeout 1800000`,
its original value), `svc power stayon false`, `dumpsys deviceidle enable`, stop keep_awake.sh (`rm /data/local/tmp/moe-stream/keep_awake.run`).
Laptop: S9 Qwen3-30B-A3B trace running under capped.sh (`moe-work/s9_Qwen3-30B-A3B.log`); then simulate lookahead
eviction under verify batches on that trace (ARCHITECTURE.md §4). Parameters now measured: forward pass
~99.5 ms pinned (bmoe_pin2 compute median), verify beta = (c(N)-1)/(N-1) = 0.71/0.685/0.68 from verify_cost.json
(pessimistic: c(N) includes the union's flash reads, which engine_sim also charges). Command:
  python moe-phone/gates/engine_sim.py results/2026-09-16/traces_Qwen3-30B-A3B-q4_0-llamacpp.npz \
      --bulk-gbps 2.806 --expert-mb 2.65 --fractions 0.2,0.24,0.3 --fwd-ms 50,99.5 --verify-beta 0,0.69 \
      --out-dir results/2026-09-17
Expected reading: with a full-cost self-draft pass (~0.1 s) W>1 needs very high alpha; the lever is a
cheaper drafter (reduced top-k / layer skip) - fwd 50 ms row approximates it.

### Earlier failures — fix status
(Peer audit 2026-09-17, moe-phone/HEADROOM.md: bmoe `compute` is a RESIDUAL (wall - stall - mgmt), never measured matmul time; all 17 Sep bmoe runs were at throttled caps; see HEADROOM for the prefetch re-check at the phone's rho.)
| failure | status |
|---|---|
| S9 transfer (hit-rate curves across expert counts) | granite-3b: neither rule validated; gpt-oss-20b family CONTROL FAILS (RMSE 0.1423 vs tol 0.0291): curves are model-specific. Qwen3-30B-A3B tracing (laptop) |
| thread collapse / unpinned variance | CONFIRMED FIXED by pinning: 5.36 vs 4.75 tok/s median, n=4 rotated |
| page recycling (new failure) | net loss end to end (mm lock/IPIs on compute cores); fix attempt: patch 0007 --defer-evict, step 5b |
| repacked kernels no gain | pinned 5.39 vs pinned+repack 4.94 tok/s median, but CONFOUNDED: repacked kernels generated different text (hit 80.1% / 172 MiB/tok vs 78.0% / 194), so the runs differ in routing. Compute residual 0.096 vs 0.100; the loss is in stall (~229 per-slice repack calls/token in the read path). Verdict stands as 'no gain shown'; a teacher-forced (--ppl-step) A/B would remove the text confound |
| verify cost inconclusive (doze) | DONE clean: c(N) 1.71/2.37/3.71 -> 4-token lossless draft needs >3.71 tokens/verify |
| GPU expert cache abort / wrong ppl | FIXED (0005 v4): ppl 11.5482 = whole-model GPU; engine speed low -> move GPU compute into an overlap engine |
| PR engine slow on phone | diagnosed: per-layer load stall, no overlap |
| NPU unreachable (old reading) | llama.cpp ggml-hexagon: HTP0 registers (v81) but opening the DSP session fails with 0x72 = AEE_ECONNREFUSED ("connection refused to DSP", AEEStdErr.h:196) from BOTH the adb shell domain and Termux (untrusted_app_27). QNN APKs reach HTP v81 on this phone (G0-NPU). OPEN, next: run through QNN (libQnnHtp*) or package the backend in an APK; not a dead end |
| "flash I/O wall" claim | retracted (bmoe MiB/s is per lane-second); lanes + cache tests queued |
| G-VALID-3 formula over-predicts | deferred until speed work lands; refit with measured compute term |
| G3 sparsity outside margin | lossy by design; not pursued for headline |
| n-gram speculation | dead for free text; replaced by draft verify + lookahead eviction (needs Qwen trace + step 4) |

## RESUME HERE — 2026-09-17 afternoon (session limit), phone queue running unattended

Phone over wireless adb: `export ANDROID_SERIAL=192.168.0.65:5555` (see memory note; tcpip mode survives
until reboot). Everything writes under `/data/local/tmp/moe-stream/`; `phone_queue.sh` orchestrates on the
phone itself (no laptop needed). Progress: `adb shell cat /data/local/tmp/moe-stream/phone_queue.log`.

Queue, in order:
1. `gpu_stream_check.sh` (old ocl-stream build) — its cpu / Qwen rows are expected to fail or be superseded;
   only its gpu_full row matters (non-regression, already 11.5482 twice).
2. `bmoe_verify_cost.sh 2` -> `bmoe_verify_*/`: --ppl-batch 1,2,3,5 on Qwen3-30B-A3B = verify cost c(N).
   Analyse: seconds per decode = total s / ceil(tokens/N), vs N=1. Laptop reference c(4)=3.5.
3. `gpu_stream_check2.sh` -> `gpu_stream2_*/`: OLMoE ppl gpu_full vs gpu_stream (patch 0005 v2) — must
   match; then Qwen3-30B-A3B with the expert cache on the Adreno GPU (llama-completion eval time).
   If gpu_stream aborts: read gpu_stream.err, next suspect is the GEMV (single-token) MoE path or the
   `q_img` image over a partially written buffer.
4. `bmoe_pin2.sh 4` -> `bmoe_pin2_*/`: rotated-order confirmation (5 cells x 4). Analyse with
   `gates/bmoe_analyze.py`, then add to `repro_check.py` JOBS_NEW and to CLAIMS.md.

State of the goal (all numbers in commit messages / artifacts, none yet in CLAIMS.md):
best phone decode so far = pinned t4 + --recycle-pages (results/2026-09-17/bmoe_pin.json), throttled CPU,
n=2, fixed order. Open levers: GPU expert cache (patch 0005), verify cost -> S2-MoE-style resident-expert
drafting, bigger models (gpt-oss-20b not yet run on the phone). Audit hardening: CLAIMS.md entries for
bmoe_levers / bmoe_repack / bmoe_pin / ocl_split / recycle_laptop_check; README/POSITION narrative.
Pending: cancel the overnight watchdog cron when the plan is done.

### Update 16:30 — results already on the phone (not yet pulled into results/)
- gpu_stream2: patch 0005 v2 no longer aborts, but the GPU-cache run is WRONG: OLMoE perplexity
  gpu_stream 14.6022 vs gpu_full 11.5482 (cpu 11.5128). Correctness gate FAILED — no speed claim allowed.
  Suspects, in order: (a) the Adreno trans4 partial conversion or the q_img image over the slot buffer;
  (b) remapped-ids handling in the GEMM tile path (emap values are slot ids, check against src0->ne[2]);
  (c) the GEMV single-token path. Bisect by running with ADRENO MoE kernels off for the stream cache.
  Qwen3-30B-A3B GPU stream: exit 139 (segfault), read qwen_gpu_stream24.err.
- verify cost rep1 total s at N=1,2,3,5: 135.17 105.72 97.31 90.62; rep2: 134.83 67.10 83.17 125.06 —
  rep2 is erratic (N=5 slower than rep1 by 38%), so c(N) needs the thermal columns and more repeats.
- bmoe_pin2 (rotated confirmation) started 16:23, running.

### Update 19:40 — where the 0.1 s/token budget actually goes (corrected)
- GPU expert cache (llama.cpp PR #25294 + 0005 v4) is CORRECT (OLMoE ppl 11.5482 with 32 slots = whole-model GPU)
  and works on Qwen3-30B-A3B, but that engine decodes at 1.53-1.89 tok/s on the GPU and 0.39 on the CPU
  (results pending pull: gpu_speed_*): it stalls on expert loads every layer (no read/compute overlap).
- A claim made in conversation that flash I/O caps decode near 3.7 tok/s was WRONG: bmoe's "MiB/s" is per
  lane-second summed over lanes; G1 measured 3.2 GB/s at 1 MB x 8 threads. bmoe pin_t4+recycle per token:
  compute 0.114 + stall ~0.069 + mgmt 0.014 = 0.197 s. Compute is the largest term, measured throttled.
- Plan for 10 tok/s: (1) compute -> ~0.05 s: unthrottled CPU and/or GPU/NPU compute inside an overlap engine
  (port GPU MUL_MAT_ID into BigMoeOnEdge rather than the PR engine); npu_compare.sh ranks the accelerators;
  (2) stall -> ~0.04: 6-8 I/O lanes (G1 scales to 8) + prefetch; (3) mgmt already 0.014.

## OVERNIGHT PLAN 2026-09-16 night — resume here

**Goal (user, verbatim intent): run big MoE models on the 15R at a decent decode rate — reproduce
BigMoeOnEdge's 5.2 tok/s on Qwen3-30B-A3B, then aim for 10+; bigger models at 5+. Try real
speculative decoding. Fix the thread cliff.**

Jobs that run without the agent (check them first on resume):
| job | where | done when |
|---|---|---|
| downloads (curl) | `moe-work/dl_{qwen3,gptoss}.log`, GGUFs in `moe-work/models/` | `DL_DONE` in the log (granite-3b already done) |
| draft model Qwen3-0.6B Q8_0 | `moe-work/models/` | file present, ~0.6 GB |
| streaming engine (llama.cpp PR #25294, branch `freedomljc/feat/moe-streaming-core`) | `moe-work/llama.cpp-stream/build-{cpu,android}` | `llama-cli` present in `build-android/bin` |
| speculative binaries on the same branch | `moe-work/spec-build-{cpu,android}.log` | `llama-speculative-simple`, `llama-lookup` present |
| thread-cliff campaign 4 | phone `~/moe/out4_*`, tmux `moe4` | `DONE` file |
| BigMoeOnEdge method extraction | research subagent | its report |
| phone state monitor (auto-unlock) | `moe-work/phone_monitor.sh` -> `results/2026-09-16/phone_state_log.csv` | runs until killed |

Order of work:
1. Pull campaign 4, analyse with `decode_analyze.py` (groups by engine flags). Early rows: 4 threads
   = 566k-879k MAJOR faults in <60 s vs fewer at 2 threads -> the cliff is a 4 KB page-fault storm
   under memory pressure. Fix = stop faulting weights in at 4 KB: bulk O_DIRECT expert reads
   (the streaming engine), not a thread flag. Confirm with the --poll 0 / cpu-mask cells.
2. Streaming engine on the phone: push `build-android/bin` + OLMoE smoke test with a small
   `--moe-stream-cache` and `--moe-stream-direct`; then push Qwen3-30B-A3B Q4_0 and run beyond DRAM.
   Budget: the app's ANONYMOUS memory (median 2.45 GB) holds the slabs. Start at 2 threads.
3. Reproduce 5.2 tok/s using BigMoeOnEdge's exact flags (from the subagent), then tune one lever
   per measured step: threads, cache slots, io-threads, quant, then GPU/NPU compute.
4. Speculative decoding for real: `llama-speculative-simple` with Qwen3-0.6B draft, sweep draft
   length 2-8, record acceptance and tok/s; `llama-lookup` (no draft). Close S10 with the measured alpha.
5. Bigger models (gpt-oss-20b, then larger) once Qwen3 is solid.
Commit + push after each milestone. Phone asleep between runs (PIN via the monitor). Shut the
laptop down when done (user instruction).

## Local setup this session used (not committed)

- Linux venv in the session scratchpad; `numpy pytest nbformat transformers==4.56.2 datasets`.
- `moe-work/` beside the repo: llama.cpp clone (`build-cpu`, `build-android` via NDK r30),
  `route_trace` binary, GGUFs. Phone: Termux sshd on 8022 over `adb forward`.
