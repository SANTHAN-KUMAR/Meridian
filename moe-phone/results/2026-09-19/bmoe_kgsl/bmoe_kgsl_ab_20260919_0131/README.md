# kgsl-arena A/B — STOPPED after 4 complete rows (no verdict; pre-registration in device/bmoe_kgsl.sh)

| row | decode tok/s | compute ms | mgmt ms | stall ms | hit % | MiB/token |
|---|---|---|---|---|---|---|
| base_rep1a | 5.46 | 97 | 36 | 50 | 80.8 | 191.8 |
| stack_rep1a (kgsl) | 3.74 | 175 | 19 | 73 | 85.9 | 214.5 |
| stack_rep1b (kgsl) | 3.72 | 175 | 19 | 75 | 86.0 | 214.3 |
| base_rep1b | 3.60 | 148 | 55 | 75 | 80.8 | 192.3 |
(from log.txt; the caps were identical in every row: 2.02 / 2.48 GHz)

Why it was stopped, measured on the phone during stack_rep2a (02:03, a 30 s window):
- engine major faults: 104,680 in 30 s; the engine's VmSwap was 212 MB (its remaining anonymous memory: dense weights, KV, buffers)
- system pswpout: 212,454 pages in 30 s; MemAvailable 1.5 GB
Reading:
- The kgsl arena (4.6 GB that cannot be reclaimed) moves reclaim onto everything else, including the engine's hot
  anonymous dense weights, which are read every token. So cache management falls as predicted (36 -> 19 ms), but compute
  rises +78 ms.
- The swapped-out state carries over into the next base row (compute 97 -> 148 ms). ABBA cannot cancel carryover
  between rows, so this design cannot give a valid verdict. Continuing would have spent ~2 h of battery for nothing.
- The mechanism behind R1 (the anonymous arena's +14 ms compute) fits the same picture: memory pressure lands on hot pages.
- Lesson: the pinned target should be the hot set (dense weights, `--dense-weights ahwb`, which exists but has never
  been measured here), not the cold expert cache. Its ceiling is small: major faults explained ~3 ms/token in the node
  trace. Parked behind the GPU-tier A/B.
