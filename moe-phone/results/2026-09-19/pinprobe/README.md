# pinprobe on the 15R (2026-09-19 01:25), pre-registered in host/pinned_arena/pinprobe.cpp. VERDICT BUILD.
- (a) CPU read bandwidth, 4 threads on cpu4-7, 768 MiB each: kgsl 62.93 vs malloc 61.74 GB/s (medians of 5), ratio 1.019.
  One kgsl sample was 4.25 GB/s (run 5), an outlier the median absorbs. It is unexplained (possibly a clock step or
  preemption). The engine A/B's compute term is the real test.
- (b) MADV_PAGEOUT then re-read: kgsl 0 major faults (0.033 s); malloc 196,608 major faults (= 768 MiB / 4 KiB, 0.919 s).
  VmSwap went 0 -> 786,048 kB for the malloc region and stayed 0 for kgsl. smaps: kgsl blocks are /dev/kgsl-3d0 shared
  device mappings (Anonymous 0, Swap 0, VmFlags dc de dd).
- Consequence: memory in the kgsl arena is outside the reach of zram, and the CPU reads it at full speed. It is also
  memory the system cannot reclaim, so bmoe_kgsl.sh caps the cache at 4000 MiB.
- Not shown here: whether the engine gains. That is bmoe_kgsl.sh (ABBA x6, pre-registered).
