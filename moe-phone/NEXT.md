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
