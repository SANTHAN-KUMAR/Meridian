# Capability report — OnePlus AC2001

Conformance: **L1 (profiled runner)**. State: `partial`. Probe suite `meridian-l1-0.1.0`. Profile id `47c5146a1dc539ad`.

SoC: Qualcomm SM7250 (lito), Android 12 (API 31). Rooted: False.

## What was actually measured on this device

- **DRAM read bandwidth** (sequential, per unprivileged process): 12.40 GB/s [prior, interval 9.87-13.13 GB/s]
- **Memory this device will grant one quiesced process and let it keep**: 8860467200 bytes [prior, interval 0-8860467200 bytes]
- **Memory grantable with a real app in the foreground**: not measured (PL-S2:never_measured_no_app_exists) — this is the number an agent actually lives on and this build has not measured it anywhere (registered stub PL-S2). Do not plan agent memory against the quiesced figure above.
- **Storage** (f2fs at `/data/user/0`): efficient request size 262144 bytes [prior, interval 262144-262144 bytes], direct I/O supported: True
    - 4 KiB @ 1 threads: 16.8 MB/s [prior, interval 11.7-21.8 MB/s] (p50 185 us, p99 1191 us)
    - 4 KiB @ 4 threads: 68.9 MB/s [prior, interval 48.2-89.5 MB/s] (p50 174 us, p99 1379 us)
    - 4 KiB @ 8 threads: 101.7 MB/s [prior, interval 71.2-132.3 MB/s] (p50 253 us, p99 1690 us)
    - 64 KiB @ 1 threads: 111.8 MB/s [prior, interval 78.3-145.4 MB/s] (p50 445 us, p99 2956 us)
    - 64 KiB @ 4 threads: 513.6 MB/s [prior, interval 359.5-667.7 MB/s] (p50 440 us, p99 1935 us)
    - 64 KiB @ 8 threads: 725.0 MB/s [prior, interval 507.5-942.5 MB/s] (p50 650 us, p99 2322 us)
    - 256 KiB @ 1 threads: 265.3 MB/s [prior, interval 185.7-344.9 MB/s] (p50 950 us, p99 1594 us)
    - 256 KiB @ 4 threads: 831.4 MB/s [prior, interval 582.0-1080.8 MB/s] (p50 1191 us, p99 2609 us)
    - 256 KiB @ 8 threads: 839.2 MB/s [prior, interval 587.5-1091.0 MB/s] (p50 2395 us, p99 4361 us)
    - 1024 KiB @ 1 threads: 295.8 MB/s [prior, interval 207.1-384.5 MB/s] (p50 3278 us, p99 6384 us)
    - 1024 KiB @ 4 threads: 858.3 MB/s [prior, interval 600.8-1115.8 MB/s] (p50 4620 us, p99 10966 us)
    - 1024 KiB @ 8 threads: 859.2 MB/s [prior, interval 601.5-1117.0 MB/s] (p50 9382 us, p99 15725 us)
    - 4096 KiB @ 1 threads: 313.0 MB/s [prior, interval 219.1-406.9 MB/s] (p50 12589 us, p99 18828 us)
    - 4096 KiB @ 4 threads: 845.1 MB/s [prior, interval 591.6-1098.7 MB/s] (p50 18847 us, p99 35102 us)
    - 4096 KiB @ 8 threads: 850.8 MB/s [prior, interval 595.6-1106.1 MB/s] (p50 38164 us, p99 55277 us)

## CPU topology (identity only — no throughput measured at L1)
- **efficiency**: cores [0, 1, 2, 3, 4, 5], max 1.805 GHz, matmul throughput: not measured (compute_probe:not_implemented_L1)
- **performance**: cores [6], max 2.208 GHz, matmul throughput: not measured (compute_probe:not_implemented_L1)
- **prime**: cores [7], max 2.400 GHz, matmul throughput: not measured (compute_probe:not_implemented_L1)

## Accelerators
- **cpu** (`cpu-generic`): available=True, reachable_unprivileged=True, fidelity_class=bit-exact, decode=not measured (engine_backend_probe:no_engine_built_L1), prefill=not measured (engine_backend_probe:no_engine_built_L1)
- **gpu** (`opencl-adreno`): available=True, reachable_unprivileged=True, fidelity_class=unverified, decode=not measured (engine_backend_probe:no_engine_built_L1), prefill=not measured (engine_backend_probe:no_engine_built_L1)
- **npu** (`qnn-htp`): available=True, reachable_unprivileged=False, fidelity_class=unverified, decode=not measured (engine_backend_probe:no_engine_built_L1), prefill=not measured (engine_backend_probe:no_engine_built_L1)

## Conditions these measurements were taken under
- power: **usb**, wakefulness: awake, foreground: none, battery: 93%
  **This is not the deployment regime (D4 requires unplugged).** Every `measured` value above is downgraded to `prior` with a widened interval for exactly this reason — it is a real number taken under charging, not a sustained-unplugged operating point, and must not be presented as one.
- 8 run(s) rejected as contaminated: ['descheduled: min_cpu_over_wall=0.888 < 0.9', 'descheduled: min_cpu_over_wall=0.760 < 0.9', 'descheduled: min_cpu_over_wall=0.622 < 0.9', 'descheduled: min_cpu_over_wall=0.889 < 0.9', 'descheduled: min_cpu_over_wall=0.804 < 0.9', 'descheduled: min_cpu_over_wall=0.462 < 0.9', 'descheduled: min_cpu_over_wall=0.825 < 0.9', 'descheduled: min_cpu_over_wall=0.395 < 0.9']

## What this report is not
- No tokens-per-second prediction for any model. That requires the PerformanceModel (05_PERFORMANCE_MODEL.md), which is not built in this pass.
- No thermal derate curve (T3), no per-backend decode/prefill rate, no foreground-coexistence memory grant. See STATUS.md for the full stub list.