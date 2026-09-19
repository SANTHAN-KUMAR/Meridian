# OnePlus Nord (AC2001, Snapdragon 765G / SM7250, 12 GB, UFS 2.x): prediction written BEFORE any engine row (2026-09-19 22:15)
Model: Qwen3-30B-A3B Q4_0 (the same file as the 15R). Engine: our current tree built for armv8.2-a+dotprod+fp16 (build-android-ref).
Config adapted to the topology: compute on the 2 big cores (cpu6-7: A76 2.2 / 2.4 GHz; -t 2 --cpu-mask c0); 4 I/O lanes on cpu0-3
(A55); cache auto (ceiling 5000 MiB); SLRU + predictive prefetch with selective adoption. On USB power (logged). 10-minute sustained.
Prediction (crude scaling from the 15R, stated so it can be falsified):
- compute ~290 ms/token (the 15R's generic 93.7 ms from 4 fast cores; the Nord has 2 A76 at ~2.3 GHz, ~1/3 of the capacity)
- stall ~50 ms (misses ~120 MiB/token at an assumed ~0.7 GB/s = ~180 ms of reads, mostly hidden behind the long compute)
- cache management ~40 ms
=> ~380 ms/token, **~2.6 tok/s (range 1.8-3.5)**. Outside that range, the scaling assumptions (core capacity, UFS rate) are wrong,
and the measurement says which.
