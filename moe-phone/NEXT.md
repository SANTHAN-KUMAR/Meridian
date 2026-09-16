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
2. **S9 trace** — the control half is DONE (2026-09-16): tokens, the OLMoE Q4_0 trace through
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

## Local setup this session used (not committed)

- Linux venv in the session scratchpad; `numpy pytest nbformat transformers==4.56.2 datasets`.
- `moe-work/` beside the repo: llama.cpp clone (`build-cpu`, `build-android` via NDK r30),
  `route_trace` binary, GGUFs. Phone: Termux sshd on 8022 over `adb forward`.
