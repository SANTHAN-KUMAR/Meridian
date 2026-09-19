# E7 verdict: KILL (by bounds; the pre-registered fit was invalid). The GPU as the main engine cannot reach the target

Run `bmoe_e7/bmoe_e7_20260919_2126/` (phone unplugged on Wi-Fi adb, held Awake; llama.cpp 9f31776 with the OpenCL/Adreno backend,
the whole graph on the Adreno 829 via `-ngl 99`; specimens = Qwen3-30B-A3B-Q4_0 truncated to its first 4 and 8 blocks, byte-identical
layers). Scored by `gates/e7_score.py` into `e7_summary.json`.

| specimen | GPU ms/token | CPU ms/token (llama.cpp, -t 4, unpinned, noisy) |
|---|---|---|
| 4 layers | 59.1 (16.9 tok/s, sd 0.4) | 14.3 (sd large) |
| 8 layers | 48.4; **51.5 sustained** (median of the last half of 19 x 256-token runs over 10 min) | 17.5 |

- **The pre-registered fit t(L) = a + bL is invalid:** the 8-layer GPU specimen ran faster than the 4-layer one, which gives b < 0, a
  physically impossible negative cost per layer. The most likely cause is a one-time warm-up in the first GPU row (a fresh OpenCL
  kernel cache, GPU DVFS ramp, or the Adreno weight conversion); it was not isolated. The first scoring printed "POSITIVE" from the
  negative projection. That was a scorer bug (no guard against b <= 0); the guard was added and this note records it.
- **The verdict comes from two bounds that do not need the fit:**
  1. monotonicity: 48 layers cannot run faster than 8, so C_gpu48 >= 51.5 ms (sustained). That alone excludes POSITIVE (<= 28.5).
  2. bandwidth: each further layer reads >= ~37 MB (8 experts x 2.65 MB + attention); at the GPU's measured peak read rate (31 GB/s,
     MEMPROBE) that is >= 1.2 ms/layer, so C_gpu48 >= 51.5 + 40 x 1.2 = **~99 ms > 50.5 ms: KILL**.
- In the same session the CPU (plain llama.cpp, repacked kernels by default, resident) ran the same 8 layers ~3x faster than the GPU.
  On Qwen3's shapes the Adreno is not a faster engine than the CPU, even with the whole graph resident.
