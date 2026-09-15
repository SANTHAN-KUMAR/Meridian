# moe-phone — beyond-DRAM MoE inference on a 12 GB phone, no retraining

**Start here:** [`POSITION.md`](POSITION.md) — the claim, the baselines, the plan, and what each
failure would mean. **Definitions:** [`ESTIMAND.md`](ESTIMAND.md).
This file is the **gate record and script index**. No number is transcribed into it.

Devices: **OnePlus 15R** (Snapdragon 8 Gen 5, SM8845) primary; **Galaxy S25** (SM8750) fallback
only if the 15R fails bring-up (G0).

## Gates

| gate | question | verdict | script |
|---|---|---|---|
| **G-ROOF** | best decode rate any engine could reach, from flash/DRAM traffic alone | **RUN on ASSUMED phone inputs** — superseded by G-ROOF v2 | `gates/byte_budget.py` |
| **G0** | 15R bring-up: device facts, memory one app can hold, NPU presence, power rails | **RUN 2026-09-14** → `results/2026-09-14/g0_device.json`, raw `g0_15r.txt` + `memprobe_15r*.csv` | `device/g0_probe.sh`, `device/memprobe.c` → `gates/g0_summarize.py` |
| **G1** | storage read throughput vs request size, concurrency, I/O mode; energy/GB | **RUN 2026-09-14** → `results/2026-09-14/g1_storage.json`, raw `ufs_15r.csv`. Bandwidth measured; **energy NOT measured** — no readable power rail (stub S6) | `device/ufsbench.c` → `gates/g1_analyze.py` |
| **G-ROOF v2** | the oracle rerun on measured 15R inputs | **RUN 2026-09-14** → `results/2026-09-14/byte_budget_measured.json` and `byte_budget_measured_bundled_ideal.json` (the granularity-wall comparison). Flash and RAM measured; **`--dram-gbps` still assumed** (stub S1) | `gates/byte_budget.py` |
| **G0-NPU** | can an unprivileged app reach the Hexagon NPU? | **RUN 2026-09-14 — YES, HTP v81, no root** → `results/2026-09-14/npu_selinux_denials.log` + `geekbench_ai_qnn.json`. A stock APK loaded `libQnnHtpV81Skel.so` onto the DSP over FastRPC domain 3; fp16/int8 ran 1-2 orders of magnitude faster than fp32. **A first reading of this gate said the opposite**: `device/devprobe.c` tested `open()` on `/dev/fastrpc-cdsp`, got `EACCES`, and that was reported as 'NPU unreachable' — but the app that *succeeds* logs the same denials, because FastRPC probes several nodes and uses whichever is permitted. The probe answers a different question than the gate asks | `device/devprobe.c` (insufficient alone) + logcat during a QNN workload |
| G2 | routing traces; hit rate vs cache size, scope, replay and lookahead horizon | **RUN 2026-09-14, OLMoE-1B-7B only** (Kaggle 2x T4), then **RE-RUN after two defects were found in our own simulator** (see the defect table below and POSITION.md §3c) → `results/2026-09-14/cache_OLMoE-1B-7B-0924.json`, `traces_OLMoE-1B-7B-0924.npz`. Corrected: plain per-layer LRU reaches **0.279** at a 10% cache against a 0.100 floor and a 0.447 Belady optimum. Controls decompose that into 0.100 floor + 0.034 popularity skew + 0.146 recency. No classical policy (LFU, warming, pinning) beats LRU; a **4-token lookahead reaches Belady exactly**. The cache result reproduces from the committed traces | `gates/traces_sparsity.py` → `gates/cache_sim.py` |
| G3 | fidelity (KL, flip, ECE) vs density for gate-first sparsity; Q4_0 floor | **RUN 2026-09-14, OLMoE-1B-7B only** → `results/2026-09-14/g3_OLMoE-1B-7B-0924.json`. **No density below 1.0 stays within the pre-registered Q4_0 Tier-A margin** — this is the trigger condition for finding **F1** in `POSITION.md`. One model is not the finding; the 30B-class runs are needed before generalising. Reference ran in float16, not bf16 (T4 has no native bf16) — recorded in the artifact as a stated deviation from `ESTIMAND.md` §3 | `gates/traces_sparsity.py` |
| **G-VALID** | does the roofline reproduce INDEPENDENT published measurements? | **RUN 2026-09-14.** (1) Fed PowerInfer-2's own configuration (TurboSparse-Mixtral, 7 GB, `predicted`, d=0.03) the roofline's **upper bound** is 1.90 tok/s (claim `gvalid1_bound_below_published`) and PowerInfer-2 **measured** 2.13 → `byte_budget_validate_powerinfer2.json`. **A real system above an upper bound is not agreement — it says the inputs do not correspond**: the bandwidth is the 15R's, not the OnePlus 12's; the cache is expert-level, not neuron-level; and scattered reads are charged at the small-read rate where PowerInfer-2 bundles them. An earlier version of this row read the same numbers as "within 12%" and is corrected here (`CLAUDE.md` §7.6). (2) [BigMoeOnEdge](https://github.com/Helldez/BigMoeOnEdge) measures **5.2 tok/s at a 76% hit rate** on Qwen3-30B-A3B on a 12 GB UFS 4.x phone. Our bound at h=0.7 is 9.18 tok/s and rises with h, so their measured rate sits **well below** our bound — which is what an upper bound must do. The gap is compute and engine overhead, which this roofline deliberately ignores and which they independently measure as the *dominant* term once the cache is well sized. **A first version of this row claimed a 1% agreement by inverting our roofline on their tok/s to get an 'implied' hit rate of 0.471 and matching it to our LRU curve. That was wrong and is deleted, not annotated (`CLAUDE.md` §7.6): they run Q4_K_M on their own device's bandwidth, so the inverted quantity did not correspond to ours, and their actual reported hit rate is 0.76, not 0.47.** Validation (2) is therefore a *bound check*, not a point match | `gates/byte_budget.py` |
| **G-RAM-SWEEP** | what memory budget does an unprivileged app actually get, and does it depend on device state? | **RUN 2026-09-14, 12 file-backed runs across 4 device states** → `g0_device.json`, `memprobe_15r_file_*.csv`, frontier at `byte_budget_measured_ram{1.5,1.88,2.5,4.85,7.0}.json`. **The budget is bimodal, not a constant: rested ~4.8 GB, used ~1.6 GB, 5.1x spread overall (983-5018 MB).** fresh boot [4056,4558,5018]; early session [4487,4623,4639]; after hours of use [1028,1568,1796]; after `am kill-all` [983,1452,2003]. **Only a reboot recovers it** — `am kill-all` gains ~92 MB and does not move the ceiling. Even on a fresh boot the app gets ~72% of `MemAvailable` (4.78 of 6.6 GB); the rest is kernel watermarks and other apps' resident sets. **Therefore PowerInfer-2's 7 GB budget is NOT reachable on this device in any state**, and every scale-frontier figure must name its budget | `device/memprobe.c --file`, `gates/byte_budget.py` |
| **S1 / DRAM** | sustained DRAM read bandwidth one app gets | **RUN 2026-09-16** → `results/2026-09-16/dram_15r.json`, raw `dram_15r_{app,shell}.csv`. 59.74 GB/s as the unprivileged app (Termux, `untrusted_app_27`) at 4 threads — the previously assumed 45 GB/s was low. **At 8 threads no row in either domain was free of descheduling**, so an engine here cannot count on more than ~4 busy cores | `device/dramprobe.c` → `gates/dram_analyze.py` |
| **G-REPRO** | do the committed artifacts reproduce from committed inputs? | **RUN 2026-09-16: yes, bit for bit** — every numeric leaf of all 11 artifacts (cache, fcrit, pred, scope, policy, engine_sim, predictor, engine_target, two byte-budget runs, g1_storage) matched. `results/` untouched: outputs go to a temp dir | `gates/repro_check.py` |
| **G-VALID-3** | does the flash-only formula predict an INDEPENDENT engine? | **RUN 2026-09-16 — NO.** colibri's community rows (github.com/JustVugg/colibri, commit a8f2ca6, quoted with line numbers): on disk-bound rows with speculation off the formula over-predicts tok/s by 2.8x (median); at a 98% hit rate by 4.4x. Where colibri reports the disk share of decode time, the implied in-engine bandwidth is a fraction of the benchmark figure — and on the M1 Ultra row it comes out at 0.93, the same "~93%" colibri reports, which checks the arithmetic. **The DRAM/compute terms the formula drops are the same order as the flash term** → `external_validation_colibri.json` | `gates/external_validation.py` |
| **S9 pre-registration** | equal-rho transfer vs floor-additive transfer, fixed before the second trace | **REGISTERED 2026-09-16** (committed before any Qwen3 trace) → `s9_prereg_Qwen3-30B-A3B.json`. Tolerance = 2x OLMoE's split-half noise; the hypotheses are separated by more than that everywhere on the grid, so the test can discriminate. The trace itself is NOT run: it needs the 17.4 GB Q4_0 checkpoint | `gates/s9_prereg.py`; collection via `tools/route_trace.cpp` + `gates/llamacpp_traces.py` |
| **G6-baseline** | unmodified llama.cpp decode on the 15R: tok/s and flash bytes/token | **RUN 2026-09-16** (clean: every 30 s state sample in the campaign window was Awake, unlocked, Termux focused) → `results/2026-09-16/decode_15r.json`, raw `decode_15r/`. **granite-1b-a400m Q4_0, fully resident: 107.2 tok/s** steady state at 4 threads, zero flash bytes. **OLMoE-1B-7B Q4_0 (3.93 GB) through stock mmap is unstable**: MemAvailable swung 1.9–5.7 GB during the campaign, every run faulted 8–10 GB from flash for a 3.9 GB file, and steady-state s/token varied 4.7x between cold repeats **with no change in flash bytes per token** — flash traffic does not explain it. **Campaign 2** (CPU-instrumented, also clean) → `decode_15r_cpu.json`: warm OLMoE decodes at **30.1 tok/s steady state with 2 threads and 8.0 with 4**, in the same memory state and with near-zero flash bytes per token — so at steady state the **thread count, not flash, sets the rate**. Cause not identified: preemption of one worker stalling llama.cpp's per-op barriers is consistent with the DRAM probe's descheduling from 4 threads, and unproven. A thread sweep (campaign 3) is running. An earlier campaign ran under the lock screen and is kept only as `decode_15r_screenlocked_contaminated/` | `device/llm_decode_probe.sh`, `device/phone_campaign*.sh` → `gates/decode_analyze.py` |
| **S11** | what does a token cost besides flash, on this phone? | **MEASURED 2026-09-16** → `s11_nonflash_15r.json`. A resident MoE consumes its per-token bytes (read from the GGUF's tensor table, `gguf_active.json`) at an effective **23.2 GB/s — 39% of the DRAM probe**: resident decode is compute/overhead-bound, not DRAM-bound. With it charged, Qwen3-30B-A3B is **6.3 tok/s under H_rho and 4.2 under H_floor** — either side of the 5 tok/s target, so the S9 trace now decides that model. Assumption stated in the script: non-flash time is linear in bytes touched. **Tested at matched thread count, and NOT confirmed**: at 2 threads it under-predicts OLMoE by 1.33x, and at 4 threads OLMoE runs far below it (the thread effect above). The 23.2 GB/s figure stands as a measurement of granite at 4 threads; its transfer to other models is open | `gates/gguf_active.py`, `gates/s11_nonflash.py` → `engine_target.py --nonflash-gbps` |
| **G1-QD** | does storage scale past 8 threads (ARCHITECTURE §5b)? | **RUN 2026-09-16 — NO.** At 1 MB direct reads 16 or 32 threads reach 0.99x of 8 threads; the plateau starts at 4 → `g1_storage_qd.json` | `device/ufsbench.c` → `gates/g1_analyze.py` |
| G4 | co-activation layout → request-size distribution | NOT RUN | *not written* |
| G5 | engine, built step by step with ablations | NOT STARTED | — |
| G6 | head-to-heads, scale frontier, energy, sustained thermals, non-root | NOT RUN | — |

**Evidence kept on purpose.** `results/2026-09-14/ufs_15r_screenoff_contaminated.csv` is a G1 run
whose third repeat was measured with the phone's screen off. It is retained because it is the
evidence for the descheduling effect: at identical configurations, throughput fell ~7.5x while p50
latency was unchanged. `g0_15r_shell.txt` is the same G0 probe run from the `shell` domain rather
than the app, so the app-vs-shell difference is a measurement rather than an assumption.

**Superseded artifacts.** `results/2026-09-14/byte_budget_g1prelim_ram5.json` and
`…_ram8.json` were computed from a *preliminary* on-device storage measurement whose raw CSV
was never committed, so their flash inputs cannot be reproduced from this repository. They are
kept as a record of what was run and are **superseded by `byte_budget_measured*.json`**, whose
inputs come from the committed `ufs_15r.csv` and `memprobe_15r*.csv`. Do not quote the prelim
artifacts.

## Scripts

```bash
# Phone track: see protocols/RUNBOOK.md for the full procedure.
python moe-phone/gates/g0_summarize.py moe-phone/results/<date>/g0_15r.txt \
       --memprobe moe-phone/results/<date>/memprobe_15r.csv
python moe-phone/gates/g1_analyze.py  moe-phone/results/<date>/ufs_15r.csv \
       --memprobe moe-phone/results/<date>/memprobe_15r.csv
# g1_analyze.py prints the G-ROOF v2 command with the measured values. Run that.
```

```bash
# G-ROOF on assumed inputs (the original scenario runs; needs network on first
# use for HF configs, cached in moe-phone/cache/, not committed)
python moe-phone/gates/byte_budget.py --ram-gb 8 --flash-gbps 3.5 --flash-gbps-scattered 0.45 \
    --dram-gbps 45 --assumed flash,dram,ram --threshold-tps 5
# the granularity-wall comparison: scattered reads charged at bulk bandwidth
python moe-phone/gates/byte_budget.py --ram-gb 8 --flash-gbps 3.5 --flash-gbps-scattered 3.5 \
    --dram-gbps 45 --assumed flash,dram,ram --threshold-tps 5 --tag bundled_ideal
```

```bash
python -m pytest moe-phone/tests -q      # cache-sim closed forms, G1 checks, notebook sync
python moe-phone/gates/traces_sparsity.py --selftest   # needs torch + transformers==4.56.2
python moe-phone/kaggle/build_notebook.py             # regenerate the Kaggle notebook
python moe-phone/kaggle/build_notebook.py --check     # fail if it is out of date
```

**How to run the tests on the phone and GPU:** [`protocols/RUNBOOK.md`](protocols/RUNBOOK.md).
The Kaggle notebook is **generated** from `gates/traces_sparsity.py` and `gates/cache_sim.py` by
`kaggle/build_notebook.py`; a test asserts the checked-in notebook matches, so the Kaggle run is
always a run of the code in this repository.

Artifacts go to `results/<date>/` via `gates/_paths.py` (dated, write-once; tests never write there).

## Defects found during 15R bring-up (2026-09-14)

Recorded because each one would have produced a confident, wrong number, and because the same
classes will recur on the next device.

| what | why it mattered | fix |
|---|---|---|
| Windows checkouts wrote the device scripts with **CRLF** | Android `/bin/sh` failed at the first blank line and every `\` continuation, so `g0_probe.sh` produced two lines of error instead of a probe | `/.gitattributes` forces LF for `moe-phone/device/**`, `*.sh`, `*.ipynb` |
| `g0_probe.sh` root check was `command -v su` | Termux ships its **own** `su`, so a locked, verified-boot-green, unrooted phone was reported as having root | test that `su -c id -u` actually returns 0; print the verified-boot state alongside |
| `try()` ended its commands in a pipe (`… \| head`) | the pipeline's exit status is `head`'s, so a denied command reported success with no output; `dumpsys powerstats` being unavailable to an app was invisible | `try()` truncates its own output and distinguishes `FAILED rc=N`, `EMPTY rc=0` and output |
| storage probe hardcoded `/sys/block/{sda,sdc,dm-0}` | the 15R's `/data` is `dm-79`; the probe collected **nothing** about the storage device and said so nowhere | resolve the backing device from `df`, and print `DENIED_OR_ABSENT` per key rather than nothing |
| `memprobe` reported the last **allocated** size, and stopped when a chunk took 5× the **first** chunk | the first chunk was the fastest of the run by chance, so it stopped early; and allocated bytes include pages already compressed into zram, which an engine cannot read at DRAM speed | report **max VmRSS** (resident); stop on an RSS plateau; timing guard keeps a median baseline and never sets the reported figure |
| `ufsbench` took the max over every thermal zone | `thermal_zone85` type `socd` reports a bare `75` (state-of-charge depletion, not a temperature) and so was the maximum in **every row**, making the whole temperature column void | reject `\|raw\| < 1000` (thermal sysfs is millidegrees), reject outside 0–120 °C, keep the `trip`/`bcl`/`-lvl` name filter, and record **which zone** produced the maximum |
| `g1_analyze.load_memprobe` matched the CSV by **field count** | when the collector grew from 5 to 8 columns the match silently stopped firing, and `--memprobe` contributed nothing while the script printed `--ram-gb <ASSUMED>` | parse by column name; return an explicit error when the file is from an older collector |
| a missing `# direct_io_supported=` trailer was read as "direct unsupported" | a truncated run would report **buffered** bandwidths under a direct-I/O heading | infer the mode from the rows and record `mode_determined_by` |
| nothing detected a **descheduled** benchmark process | with the phone's screen off, a 2 s configuration took 4.88 s and reported 108 MB/s where the same configuration screen-on gave 814 MB/s — p50 latency unchanged, so the storage was fine and the process simply was not running. Averaged in, this silently depresses the measured bandwidth | `g1_analyze.py` excludes rows whose elapsed time exceeds the median deadline by >25% and reports `n_invalid_process_descheduled`; RUNBOOK requires screen-on and Termux foregrounded |
| the Kaggle `traces_sparsity.py` crashed under `device_map="auto"` | the Fate lookahead applies layer l+1's router to layer l's input; sharded across 2 GPUs those are on different devices, so the real run died after downloading the model while the CPU self-test passed — a self-test that structurally could not reach the failing path | `router_logits` moves the activation to the weight's device; the self-test gained a cross-device case that RUNS on multi-GPU and prints an explicit SKIP otherwise |
| `--dtype auto` chose bfloat16 on a T4 | `torch.cuda.is_bf16_supported()` is True there because PyTorch *emulates* bf16 by upcasting to fp32 — several times slower, and it contradicted the RUNBOOK's documented "T4s have no bfloat16, so the reference runs in float16" | require compute capability >= 8.0 for native bf16; record `bf16_native` and the device count in the artifact |
| `cache_sim.py` simulated **one cache shared by all layers**, while `expert_policy.py`'s pinning policy was **per layer** | the LRU-vs-pinning comparison was between two different machines. A shared pool is a step function pinned at zero until it holds the entire `L x k` cycle, so it reported **no hits at all** at any phone-sized cache and that was published as a property of MoE routing | `--scope {per_layer,global}`, with `split_capacity()` preserving the total slot budget exactly so both scopes mean the same bytes; the old configuration is still runnable so the retracted number stays reproducible |
| the replay evicted after **every single expert access** | a miss early in a layer's top-k fetch could evict an expert the *same* fetch still needed, which was then counted as a miss. A real engine fetches a layer's misses as one event. Combined with the defect above, the published 10%-cache LRU hit rate was 0.000 where the correct value is **0.279** | `--replay {atomic,sequential}`; atomic decides residency for the whole event before any eviction, and a test asserts atomic can never score below sequential |
| `engine_target.py` printed G2 reference numbers **typed into a `print()`** | a hand-transcribed number (`CLAUDE.md` §7.1) that silently went stale the moment G2 was re-run — it survived the re-run still quoting the retracted values | the script now reads the curve from the artifact and interpolates it, and reports which models fall outside the measured range |
| two `str.replace()` edits to `engine_target.py` **silently did nothing** | a no-op replace leaves the old text in place and reports success, so the tool printed a correct data row under a stale header and kept the hand-transcribed block it was meant to delete | every scripted edit now asserts the pattern is present and unique before substituting; this is the §0 failure mode in miniature |
| `engine_target.py` credited a **fully resident** model with a cache-curve hit rate (OLMoE "27.1 tok/s on plain LRU"), derived tok/s from **clamped** out-of-range curve values while printing the extrapolation flag in only one verdict branch (DeepSeek-V2-Lite "DOUBLE DIGITS", unmarked), and charged **flash time only** | the "double digits on three candidate models" in `ARCHITECTURE.md` came from these | resident models are reported flash-free; out-of-range rows get no number; every row carries the serial flash + DRAM figure and the S9 alternative (2026-09-16) |
| an "independently reported 0.693" hit rate for a 512-expert model was cited as the one external check on S9, in five files, with no source recorded | an unsourced number was doing the work of evidence | deleted everywhere; S9 is pre-registered instead (2026-09-16) |
| the phone's lock screen came up during an unattended run although `stayon` was set | Termux left the foreground cgroup; runs were descheduled and llama-bench thrashed at 15 MB resident | the campaign is kept as contaminated evidence, not analysed; a laptop-side monitor logs wakefulness and focus every 30 s |
| the Kaggle notebook claimed to be "generated" but had no generator, and contained U+FFFD | an edited-by-hand copy drifts from the gate scripts, and then the Kaggle numbers are not this repository's numbers | `kaggle/build_notebook.py` plus tests asserting the notebook matches the repo scripts byte for byte |
