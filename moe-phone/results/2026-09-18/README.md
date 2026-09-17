# `results/2026-09-18/` — in-app device measurements, and the ceiling they imply

Everything here was measured **inside an app process** (`com.moephone.npu2` for upstream llama.cpp,
`com.moephone.bmoe3` for our engine), because that is the only SELinux domain on this phone where the
Hexagon DSP opens a FastRPC session. The app is built by
[`../../host/build_npu_app.sh`](../../host/build_npu_app.sh); its whole source is in
[`../../host/app/`](../../host/app/).

Do not quote a number from these files by hand. Every number this tree supports is a claim in
[`../../gates/claims_check.py`](../../gates/claims_check.py); run it and cite the claim id.

## What is here

| path | producer | analysed by |
|---|---|---|
| `app_engine/` | `host/overnight_npu.sh` — upstream llama.cpp (PR #25294 expert-streaming branch) in-app: Qwen3-30B-A3B streamed on HTP0 (24 and 32 slots), on the Adreno, on the CPU; then OLMoE **resident** on all three devices. 2 repeats each. | `gates/app_engine_analyze.py` → `app_engine.json` |
| `app_engine.json` | per-arm medians, min/max, failure counts | claims `inapp_olmoe_*`, `upstream_inapp_qwen3_best` |
| `device_bandwidth.json` | `gates/device_bandwidth.py` — resident rate → effective weight-byte throughput → the ceiling that implies for Qwen3-30B-A3B | claims `device_ceiling_qwen3_*` |
| `gguf_active_qwen_olmoe.json` | `gates/gguf_active.py` — the bytes a token of each model reads, from the GGUF's own tensor table | claims `qwen3_active_mb_per_token`, `olmoe_active_mb_per_token` |
| `PREREG_zram_and_aggregate.md` | the three questions of this night, with arms, decision rules and predicted signs, committed before any of them had a row | — |
| `bmoe_attn_ab/` | `host/bmoe_attn_ab.sh` — OUR engine in-app, `--attn-device` CPU vs HTP0 vs GPUOpenCL | `gates/app_engine_analyze.py --producer bmoe --reference attn_cpu --text-compare` |
| `bmoe_zram/` | `device/bmoe_zram.sh` — expert cache deliberately larger than RAM, with `/proc/vmstat` swap counters per row | `gates/bmoe_analyze.py --swap-log` |
| `agg_bandwidth/` | `host/agg_bandwidth.sh` — two devices decoding at once vs each alone | — |
| `matmul_sweep/` | `host/matmul_sweep.sh` — every Qwen3 matmul shape, per device, compute and upload | `gates/matmul_sweep_analyze.py` |

## The one thing to read this tree for

A decoded token reads a fixed number of weight bytes, so a decode rate is a weight-byte throughput and
each device's own throughput bounds what any engine can reach on that device. Measured on a resident
model in the same process, the three devices rank **Adreno > CPU > Hexagon** (claims
`inapp_olmoe_gpu_tok_s`, `inapp_olmoe_cpu_tok_s`, `inapp_olmoe_htp_tok_s`) — the NPU is the *slowest*
of the three on this workload, which is the opposite of what "NPU" suggests and the opposite of what
this project assumed for two days.

The ceilings those throughputs imply for Qwen3-30B-A3B (claims `device_ceiling_qwen3_*`) are all above
the project's 10 tok/s goal, so the goal is not excluded by any single device — but it needs a large
fraction of what the best device can deliver, and the streaming engine currently achieves well under
that. The gap is where the remaining work is, and the two questions that decide whether it can be
closed (does an oversized ZRAM cache help; do two devices add bandwidth) are pre-registered in
`PREREG_zram_and_aggregate.md`.

## Two caveats that belong with these numbers, not in a footnote

1. **The ceilings are derived from OLMoE and applied to Qwen3.** OLMoE's per-token working set is under
   a gigabyte; Qwen3's is nearly two. The larger model gets fewer system-cache hits, so its achievable
   throughput is likely *below* the OLMoE-derived figure, making these ceilings optimistic. The per-op
   sweep (`matmul_sweep/`, with rotated weight copies to defeat cache reuse) is the instrument that does
   not have this problem, and it is the one to believe when the two disagree.
2. **Upstream's streamed Qwen3 rows are not comparable to our engine's.** PR #25294 has no read/compute
   overlap; its numbers here (claim `upstream_inapp_qwen3_best`) are a *lower* bound on what the phone
   can do and are included to show what the device comparison looks like within one engine, not to
   flatter ours. Our engine's own in-app rows are in `bmoe_attn_ab/`.
