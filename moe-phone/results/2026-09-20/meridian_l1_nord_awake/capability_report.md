# Capability report — OnePlus AC2001

Conformance: **L1 (profiled runner)**. State: `partial`. Probe suite `meridian-l1-0.1.0`. Profile id `47c5146a1dc539ad`.

SoC: Qualcomm SM7250 (lito), Android 12 (API 31). Rooted: False.

## What was actually measured on this device

- **DRAM read bandwidth** (sequential, per unprivileged process): 13.31 GB/s [measured]
- **Memory this device will grant one quiesced process and let it keep**: 8592031744 bytes [measured]
- **Memory grantable with a real app in the foreground**: not measured (PL-S2:never_measured_no_app_exists) — this is the number an agent actually lives on and this build has not measured it anywhere (registered stub PL-S2). Do not plan agent memory against the quiesced figure above.
- **Storage** (f2fs at `/data/user/0`): efficient request size 262144 bytes [measured], direct I/O supported: True
    - 4 KiB @ 1 threads: 11.5 MB/s [measured] (p50 336 us, p99 823 us)
    - 4 KiB @ 4 threads: 88.5 MB/s [measured] (p50 156 us, p99 490 us)
    - 4 KiB @ 8 threads: 130.2 MB/s [measured] (p50 230 us, p99 494 us)
    - 64 KiB @ 1 threads: 127.6 MB/s [measured] (p50 446 us, p99 1203 us)
    - 64 KiB @ 4 threads: 587.2 MB/s [measured] (p50 409 us, p99 941 us)
    - 64 KiB @ 8 threads: 812.7 MB/s [measured] (p50 617 us, p99 1098 us)
    - 256 KiB @ 1 threads: 257.3 MB/s [measured] (p50 967 us, p99 2059 us)
    - 256 KiB @ 4 threads: 867.8 MB/s [measured] (p50 1086 us, p99 3769 us)
    - 256 KiB @ 8 threads: 916.2 MB/s [measured] (p50 2241 us, p99 3791 us)
    - 1024 KiB @ 1 threads: 288.8 MB/s [measured] (p50 3133 us, p99 7017 us)
    - 1024 KiB @ 4 threads: 955.0 MB/s [measured] (p50 4332 us, p99 5838 us)
    - 1024 KiB @ 8 threads: 943.1 MB/s [measured] (p50 8565 us, p99 25938 us)
    - 4096 KiB @ 1 threads: 297.6 MB/s [measured] (p50 12924 us, p99 20249 us)
    - 4096 KiB @ 4 threads: 930.6 MB/s [measured] (p50 17170 us, p99 35736 us)
    - 4096 KiB @ 8 threads: 966.1 MB/s [measured] (p50 34408 us, p99 41593 us)

## CPU topology (identity only — no throughput measured at L1)
- **efficiency**: cores [0, 1, 2, 3, 4, 5], max 1.805 GHz, matmul throughput: not measured (compute_probe:not_implemented_L1)
- **performance**: cores [6], max 2.208 GHz, matmul throughput: not measured (compute_probe:not_implemented_L1)
- **prime**: cores [7], max 2.400 GHz, matmul throughput: not measured (compute_probe:not_implemented_L1)

## Accelerators
- **cpu** (`cpu-generic`): available=True, reachable_unprivileged=True, fidelity_class=bit-exact, decode=not measured (engine_backend_probe:no_engine_built_L1), prefill=not measured (engine_backend_probe:no_engine_built_L1)
- **gpu** (`opencl-adreno`): available=True, reachable_unprivileged=True, fidelity_class=unverified, decode=not measured (engine_backend_probe:no_engine_built_L1), prefill=not measured (engine_backend_probe:no_engine_built_L1)
- **npu** (`qnn-htp`): available=True, reachable_unprivileged=False, fidelity_class=unverified, decode=not measured (engine_backend_probe:no_engine_built_L1), prefill=not measured (engine_backend_probe:no_engine_built_L1)

## Conditions these measurements were taken under
- power: **unplugged**, wakefulness: awake, foreground: **none**, battery: 97%
- 6 run(s) rejected as contaminated: ['descheduled: min_cpu_over_wall=0.895 < 0.9', 'descheduled: min_cpu_over_wall=0.377 < 0.9', 'descheduled: min_cpu_over_wall=0.832 < 0.9', 'descheduled: min_cpu_over_wall=0.741 < 0.9', 'descheduled: min_cpu_over_wall=0.789 < 0.9', 'descheduled: min_cpu_over_wall=0.560 < 0.9']

## What this report is not
- No tokens-per-second prediction for any model. That requires the PerformanceModel (05_PERFORMANCE_MODEL.md), which is not built in this pass.
- No thermal derate curve (T3), no per-backend decode/prefill rate, no foreground-coexistence memory grant. See STATUS.md for the full stub list.