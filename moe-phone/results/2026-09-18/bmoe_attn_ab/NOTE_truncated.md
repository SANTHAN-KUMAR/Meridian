# Why this campaign has one repeat instead of three

`host/bmoe_attn_ab.sh` was stopped at 02:09 on 2026-09-18, after `attn_cpu_rep1` and `attn_htp_rep1`.
Both rows are kept; the GPU arm produced **no row at all** and that is a failure, not a gap:

**`attn_gpu_rep1` crashed at model load.** SIGSEGV, `SEGV_MAPERR`, fault address `0x60`, inside
`ggml_backend_opencl_buffer_type_get_alignment` (logcat crash buffer, 01:52:10). Cause found and fixed
the same night: our `--attn-device` path took a device's buffer type with
`ggml_backend_dev_buffer_type()` **without first initialising that device's backend**. ggml-hexagon
tolerates this; ggml-opencl builds the context its buffer type reads only at device init, so the first
weight allocation dereferenced null. Patch 0009 now calls `ggml_backend_dev_init` first, keeps the
backend for the session (the weight buffers belong to it) and frees it after the model. The campaign was
stopped rather than left running because the driver polls for 25 minutes before giving up on a dead run,
and two more GPU repeats would have cost ~50 minutes of the night for nothing.

**What the two surviving rows say, and what they do not.**

| arm | decode | compute residual | cache mgmt | stall | cache budget granted | hit |
|---|---|---|---|---|---|---|
| `attn_cpu_rep1` | 4.587 tok/s | 0.102 s/tok | 0.033 | — | 4127 MiB | 78.8% |
| `attn_htp_rep1` | 3.822 tok/s | 0.177 s/tok | 0.022 | — | 4617 MiB | 84.3% |

Attention on the NPU **lost 17%**, and it lost while holding the *better* cache of the two (a larger
granted budget and a higher hit rate, because its dense weights sat in the DSP's buffer instead of our
anon arena). The whole of the loss is in the compute term, which went from 0.102 to 0.177 s/token. That
is the cost of splitting the graph across a backend boundary four times per layer — 192 crossings per
token, each copying activations in and out of DSP memory — and it is not a statement about the DSP's
arithmetic, which `host/matmul_sweep.sh` measures separately and per op.

Neither row is comparable to the 6.20 tok/s shell baseline (`results/2026-09-17/bmoe_cache.json`,
`ceil5000`): both ran with a smaller granted cache after a 17 GB model copy and with the cores
thermally capped at 1.6516 GHz of 3.8016. `host/app_vs_shell.sh` measures that separately, after idling.

**The `435 weight tensors live in a non-host buffer` line in these logs is wrong and must not be
quoted.** The counter tested the buffer before testing whether the tensor was file-backed, so it also
counted the graph's activations. llama.cpp's own log in the same file gives the true figure:
`'blk.0.attn_q.weight' (q4_0) (and 191 others) cannot be used with preferred buffer type CPU, using
OpenCL instead` — 192 tensors, exactly the four patterns over 48 layers. Fixed in patch 0009.
