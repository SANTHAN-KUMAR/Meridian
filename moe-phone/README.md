# moe-phone — beyond-DRAM MoE inference on a 12 GB phone, no retraining

**Start here:** [`POSITION.md`](POSITION.md) — the claim, the baselines, the plan, and what each
failure would mean. **Definitions:** [`ESTIMAND.md`](ESTIMAND.md).
This file is the **gate record and script index**. No number is transcribed into it.

Devices: **OnePlus 15R** (Snapdragon 8 Gen 5, SM8845) primary; **Galaxy S25** (SM8750) fallback
only if the 15R fails bring-up (G0).

## Gates

| gate | question | verdict | script |
|---|---|---|---|
| **G-ROOF** | best decode rate any engine could reach, from flash/DRAM traffic alone | **RUN on ASSUMED phone inputs** (stub S1) — upper bound only | `gates/byte_budget.py` |
| G0 | 15R bring-up: device facts, memory one app can hold, NPU presence, power rails | **READY, NOT RUN** — needs the phone | `device/g0_probe.sh`, `device/memprobe.c` |
| G1 | storage read throughput vs request size, concurrency, I/O mode; energy/GB | **READY, NOT RUN** — needs the phone; mechanics self-tested under WSL | `device/ufsbench.c` → `gates/g1_analyze.py` |
| G2 | routing traces; LRU and Belady hit rate vs cache size; training-free lookahead recall | **READY, NOT RUN** — needs a GPU; self-test passes | `gates/traces_sparsity.py` → `gates/cache_sim.py` |
| G3 | fidelity (KL, flip, ECE) vs density for gate-first sparsity; Q4_0 floor | **READY, NOT RUN** — needs a GPU; self-test passes | `gates/traces_sparsity.py` |
| G4 | co-activation layout → request-size distribution | NOT RUN | *not written* |
| G5 | engine, built step by step with ablations | NOT STARTED | — |
| G6 | head-to-heads, scale frontier, energy, sustained thermals, non-root | NOT RUN | — |

## Scripts

```bash
# G-ROOF — needs network on first run (HF config/API cache in moe-phone/cache/, not committed)
python moe-phone/gates/byte_budget.py --ram-gb 8 --flash-gbps 3.5 --flash-gbps-scattered 0.45 \
    --dram-gbps 45 --assumed flash,dram,ram --threshold-tps 5
# the granularity-wall comparison: scattered reads charged at bulk bandwidth
python moe-phone/gates/byte_budget.py --ram-gb 8 --flash-gbps 3.5 --flash-gbps-scattered 3.5 \
    --dram-gbps 45 --assumed flash,dram,ram --threshold-tps 5 --tag bundled_ideal
```

```bash
python -m pytest moe-phone/tests -q      # cache-sim closed forms, G1 checks, GPU-script self-test
python moe-phone/gates/traces_sparsity.py --selftest   # needs torch + transformers==4.56.2
```

**How to run the tests on the phone and GPU:** [`protocols/RUNBOOK.md`](protocols/RUNBOOK.md).

Artifacts go to `results/<date>/` via `gates/_paths.py` (dated, write-once; tests never write there).
