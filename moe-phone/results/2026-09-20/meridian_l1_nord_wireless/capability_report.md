# Capability report — OnePlus AC2001

Conformance: **L1 (profiled runner)**. State: `partial`. Probe suite `meridian-l1-0.1.0`. Profile id `47c5146a1dc539ad`.

SoC: Qualcomm SM7250 (lito), Android 12 (API 31). Rooted: False.

## What was actually measured on this device

- **DRAM read bandwidth** (sequential, per unprivileged process): 13.23 GB/s [prior, interval 12.66-13.35 GB/s]
- **Memory this device will grant one quiesced process and let it keep**: 8860467200 bytes [prior, interval 8860467200-8860467200 bytes]
- **Memory grantable with a real app in the foreground**: not measured (PL-S2:never_measured_no_app_exists) — this is the number an agent actually lives on and this build has not measured it anywhere (registered stub PL-S2). Do not plan agent memory against the quiesced figure above.
- **Storage** (f2fs at `/data/user/0`): efficient request size 262144 bytes [prior, interval 262144-4194304 bytes], direct I/O supported: True
    - 4 KiB @ 1 threads: 6.2 MB/s [prior, interval 5.3-6.6 MB/s] (p50 267 us, p99 756 us)
    - 4 KiB @ 4 threads: 30.9 MB/s [prior, interval 28.7-31.6 MB/s] (p50 185 us, p99 768 us)
    - 4 KiB @ 8 threads: 54.7 MB/s [prior, interval 41.9-71.2 MB/s] (p50 253 us, p99 988 us)
    - 64 KiB @ 1 threads: 72.1 MB/s [prior, interval 68.7-75.1 MB/s] (p50 454 us, p99 911 us)
    - 64 KiB @ 4 threads: 250.3 MB/s [prior, interval 247.4-267.5 MB/s] (p50 443 us, p99 2102 us)
    - 64 KiB @ 8 threads: 413.2 MB/s [prior, interval 289.2-424.6 MB/s] (p50 636 us, p99 2512 us)
    - 256 KiB @ 1 threads: 107.8 MB/s [prior, interval 65.1-131.1 MB/s] (p50 1008 us, p99 4581 us)
    - 256 KiB @ 4 threads: 386.6 MB/s [prior, interval 357.5-393.5 MB/s] (p50 1225 us, p99 6885 us)
    - 256 KiB @ 8 threads: 384.5 MB/s [prior, interval 322.0-892.8 MB/s] (p50 2442 us, p99 137899 us)
    - 1024 KiB @ 1 threads: 121.8 MB/s [prior, interval 118.3-185.6 MB/s] (p50 3710 us, p99 159015 us)
    - 1024 KiB @ 4 threads: 474.8 MB/s [prior, interval 465.9-615.4 MB/s] (p50 4427 us, p99 153828 us)
    - 1024 KiB @ 8 threads: 555.5 MB/s [prior, interval 363.9-746.2 MB/s] (p50 9258 us, p99 167500 us)
    - 4096 KiB @ 1 threads: 117.6 MB/s [prior, interval 68.5-153.0 MB/s] (p50 15360 us, p99 184767 us)
    - 4096 KiB @ 4 threads: 686.3 MB/s [prior, interval 498.4-840.6 MB/s] (p50 19535 us, p99 187877 us)
    - 4096 KiB @ 8 threads: 663.7 MB/s [prior, interval 573.4-743.7 MB/s] (p50 38708 us, p99 213387 us)

## CPU topology (identity only — no throughput measured at L1)
- **efficiency**: cores [0, 1, 2, 3, 4, 5], max 1.805 GHz, matmul throughput: not measured (compute_probe:not_implemented_L1)
- **performance**: cores [6], max 2.208 GHz, matmul throughput: not measured (compute_probe:not_implemented_L1)
- **prime**: cores [7], max 2.400 GHz, matmul throughput: not measured (compute_probe:not_implemented_L1)

## Accelerators
- **cpu** (`cpu-generic`): available=True, reachable_unprivileged=True, fidelity_class=bit-exact, decode=not measured (engine_backend_probe:no_engine_built_L1), prefill=not measured (engine_backend_probe:no_engine_built_L1)
- **gpu** (`opencl-adreno`): available=True, reachable_unprivileged=True, fidelity_class=unverified, decode=not measured (engine_backend_probe:no_engine_built_L1), prefill=not measured (engine_backend_probe:no_engine_built_L1)
- **npu** (`qnn-htp`): available=True, reachable_unprivileged=False, fidelity_class=unverified, decode=not measured (engine_backend_probe:no_engine_built_L1), prefill=not measured (engine_backend_probe:no_engine_built_L1)

## Conditions these measurements were taken under
- power: **unplugged**, wakefulness: dozing, foreground: **none**, battery: 98%
  **Wakefulness was 'dozing', not 'awake'.** The screen-off state is a listed contaminant (04_DEVICE_PROFILING.md section 4); this run's storage rates were 40-60% below an earlier awake run on the same device with p99 latency ~150 ms, so values are downgraded to `prior`, not `measured`. Re-run with the screen on and idle.
- 10 run(s) rejected as contaminated: ['descheduled: min_cpu_over_wall=0.608 < 0.9', 'descheduled: min_cpu_over_wall=0.540 < 0.9', 'descheduled: min_cpu_over_wall=0.569 < 0.9', 'descheduled: min_cpu_over_wall=0.658 < 0.9', 'descheduled: min_cpu_over_wall=0.288 < 0.9', 'descheduled: min_cpu_over_wall=0.462 < 0.9', 'descheduled: min_cpu_over_wall=0.469 < 0.9', 'descheduled: min_cpu_over_wall=0.483 < 0.9', 'descheduled: min_cpu_over_wall=0.391 < 0.9', 'descheduled: min_cpu_over_wall=0.288 < 0.9']

## What this report is not
- No tokens-per-second prediction for any model. That requires the PerformanceModel (05_PERFORMANCE_MODEL.md), which is not built in this pass.
- No thermal derate curve (T3), no per-backend decode/prefill rate, no foreground-coexistence memory grant. See STATUS.md for the full stub list.