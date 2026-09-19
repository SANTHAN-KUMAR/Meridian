# E4 verdict: KILL (research spec §6 E4 v2; pre-registered thresholds). The GPU branch is closed; S_gpu = 0 for the E1 closure.

Run: `results/2026-09-19/gx_e4_phone_201740` (log `e4_run.log`). The phone was unplugged on Wi-Fi adb, 20:17 IST. The kernel is gx variant 5
(commit 8dbcfb9: bit-exactness replaced by a declared tolerance; fp32 x and h, hardware fp16, fp32 fma).
- Correctness on the Adreno: TOLSUMMARY 27 cases, 0 fails, worst rel vs fp64 5.5e-7 (gate 1e-5); vs ggml ARM 3.0e-2 (ggml's own Q8 error).
- Criterion (spin=1, down Q4_0, host-visible median): **k=1 1.128 ms (positive needs <= 0.15); k=2 1.468 ms (kill at >= 0.40)**. KILL.
  Device time: k=1 0.805 ms, k=2 1.125 ms, 2.95x v4's at both k (v4: 0.273, 0.381 ms), and kernel 1 dominates. In the same session the
  bit-exact v4 took 0.58 ms host at k=1 (spin=1). **Why v5 is slower is NOT measured.** Candidates (gx NOTE §11.2): fp32 x in local
  memory, float16 register pressure, the device exp/div. The verdict does not depend on it.
  Caveat: the phone dozed before the run, the CPU caps were below the hardware maximum, and the GPU clock was not logged. That cannot
  bridge a 3.7x gap from the positive threshold, and v4 ran in the same session.
- Consequence (spec E4 §8-10): the Adreno cannot beat the repacked CPU at the engine's dispatch size (~2 experts). Every GPU lever is
  closed, including the dispatch-overhead ones. In the closing formula S_gpu = 0, so 10 tok/s lossless requires C_R(capped) <= 43.4 ms.
