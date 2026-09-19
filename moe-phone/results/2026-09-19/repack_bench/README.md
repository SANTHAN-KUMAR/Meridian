# repack_bench (pre-registered in host/repack_bench/repack_bench.cpp): VERDICT BUILD on the phone
Phone (15R, awake, 4 threads on cpu4-7, strict; libggml-cpu from bmoe-i8mm-0024), the same Q4_0 bytes, MUL_MAT_ID of 8 experts, 1 token:
| shape | generic ms | repacked ms | ratio | generic GB/s | repacked GB/s | max abs diff |
|---|---|---|---|---|---|---|
| gate/up (2048 -> 768) | 0.354 | 0.211 | **0.596** | 20.0 | 33.5 | 1.2e-6 |
| down (768 -> 2048) | 0.266 | 0.209 | **0.786** | 26.6 | 33.9 | 4.8e-7 |
The laptop (x86, q4_0_8x8) gave 0.76 / 0.72 (repack_bench_laptop_x86.out).
Reading: an isolated kernel measurement; the engine A/B (device/bmoe_repack.sh) decides the in-engine effect. Gate/up are
2/3 of expert bytes, so the expert matmuls would run ~0.66x as long if the kernel rate carried over (~-19 ms/token of ~55).
The generic gate/up rate here (20 GB/s) is below down's (26.6), the per-byte anomaly of the ceiling handoff section 1.3. The repacked kernel
removes it (33.5 vs 33.9).
