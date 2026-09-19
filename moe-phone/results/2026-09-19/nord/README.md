# OnePlus Nord (AC2001, Snapdragon 765G / SM7250, 12 GB, UFS, Android 12): first second-device data point (2026-09-19)
Pre-registered prediction: `PREDICTION.md` (~2.6 tok/s, range 1.8-3.5). Device probe: `g0_nord.txt`.
Engine: our current tree for armv8.2-a+dotprod+fp16 (the i8mm build would not run: no i8mm on this SoC). Placement adapted to the
topology: compute on the 2 big cores (cpu6-7, A76), 4 I/O lanes on cpu0-3 (A55). Qwen3-30B-A3B Q4_0, cache 5000 MiB. On USB power.
Kept Awake by a host keep-alive (a wake keypress every 20 s): this build refuses settings writes from adb (WRITE_SECURE_SETTINGS), so
neither `svc power stayon` nor `stay_on_while_plugged_in` applied. A few boundary samples still read Dozing; the run times show no
bimodality.

| run (`bmoe_sustain_20260919_2223/`) | tok/s | compute | mgmt | stall | hit | skin |
|---|---|---|---|---|---|---|
| 1 | 1.768 | 350 | 52 | 164 | 89.3% | 45.2 C |
| 2 | 1.702 | 365 | 52 | 170 | 89.1% | 46.5 |
| 3 | 1.685 | 376 | 51 | 166 | 89.1% | 47.1 |
| 4 | 1.670 | 376 | 51 | 171 | 89.1% | 47.7 |
| 5 | 1.667 | 378 | 51 | 171 | 89.0% | 47.4 |
**Steady state (median of the second half): 1.67 tok/s.** Big-core caps fell 2.21/2.40 -> 1.73/1.77 GHz.
- **The prediction was falsified** (below its range):
  - compute 376 vs ~290 ms: two A76 cores deliver less than the capacity scaling assumed;
  - stall 171 vs ~50 ms: the overlap did not hide the older storage's reads behind the long compute.
  Both are inputs for the cross-device performance model (paper plan §4).
- An earlier attempt (`bmoe_sustain_20260919_2218/`) ran while the phone dozed (1.50 tok/s, compute 463 ms). It is excluded; this is
  the autosuspend confound again.
