# zcbench run 1 (2026-09-18 ~19:57): kept as evidence, NOT the verdict

The harness chose the concurrent-phase GPU mechanism as "the first zero-copy-family mechanism
present", which was `use_host_ptr`, a mechanism that had just FAILED the zero-copy test on Adreno
(its sync runs at copy speed, and without a sync the kernel sees stale bytes: err_initial = 1).
analyze.py therefore reported "NOT viable: failed 2 correct" about the wrong arm. The selection now
takes a mechanism that PASSED (alloc_host_ptr here) and prints `phase=concurrent_arm` (host/gpu_zerocopy
commit that follows this run). No threshold or rule was changed; the analyzer is unchanged.

Everything else in this run is valid and agrees with run 2: alloc_host_ptr zero-copy (0.114 ms sync for
132 MB against a 5.4 ms memcpy), GPU kernels correct to ~1.4e-7, and the concurrent aggregate of the
use_host_ptr arm 35.53 GB/s = 1.565x the CPU alone, fully overlapped, caps 1.90/1.65 GHz throughout.
