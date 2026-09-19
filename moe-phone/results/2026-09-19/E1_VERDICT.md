# E1 verdict (research spec E1; closing formula pre-registered in device/bmoe_e1.sh): 10 tok/s lossless is NOT reachable on the CPU path

Run `bmoe_e1/bmoe_e1_20260919_2022/` (phone unplugged, Wi-Fi adb, held Awake), scored by `gates/e1_score.py` into `e1_summary.json`.
Replay `--fixed-routing`: stall 0.000 and 99.6% hit in every valid row (the ~5.7 MiB/token is the one-time load of the 384 fixed experts).

| arm | throttled, short context: rows (ms/token) | median | ~3,000-token context |
|---|---|---|---|
| G generic Q4_0 kernels | 82.0 91.0 90.4 96.4 101.6 97.2 | **93.7** | 180.8 |
| R repacked i8mm (pre-repacked experts + repacked dense) | 79.6 71.1 69.5 72.7 71.0 70.2 | **71.0** | 144.3 |
- Repacked gain per ABBA repeat: 11.2, 22.3, 28.8 ms (mean 20.7). R is faster in 3/3 repeats: the largest lossless compute gain measured
  in this project. Caps during the rows: policy0 2.19 -> 1.75 GHz, policy6 1.65 GHz (hardware 3.32 / 3.80).
- **Closing formula:** C_R + stall 34.6 + mgmt 22.0 - S_gpu (0, E4 killed) = 71.0 + 56.6 = **127.6 ms/token -> 7.8 tok/s at best**. 10 tok/s
  lossless needs C_R <= 43.4 ms. Not reachable on this phone's sustained clock with the CPU-centric architecture.
- At long context (~3,000 tokens) compute alone is 144 ms/token (repacked): below 7 tok/s before any flash cost.
- Not measured (recorded, not rerun): the full-clock "cold" rows. The phone did not cool below 34.6 C in 25 min while held awake;
  cold_G ran NOT COLD (93.9 ms, the same as the capped rows) and E1 was stopped before cold_R. The instruction-counter rows failed
  (a wrong simpleperf event name, `cycles:u` instead of `cpu-cycles:u`); they were diagnostic only.
- E7 (the GPU as the main engine) is the last open direction; it runs next.
