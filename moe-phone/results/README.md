# `results/` — measurement artifacts

Every claim in this repository is read from a file under here by a named script, per
[`../../CLAUDE.md`](../../CLAUDE.md) §7.1: no number is hand-transcribed into prose. Directories
are **dated and write-once** — a run lands under `results/<date>/`, via `gates/_paths.py`, and is
never edited afterwards. Tests never write here (`gates/_paths.py` refuses a results path from a
test context); they use a temp directory instead.

Superseded or contaminated runs are **kept, not deleted**, and are named or documented as such
(e.g. `..._contaminated`, `..._FAILED_argsplit`) so the record of what was tried is complete — see
[`../README.md`](../README.md)'s "Evidence kept on purpose" and "Superseded artifacts" notes.

## How to read this directory

- Start from a **verdict file** (`*_VERDICT.md`, `*_SUMMARY.md`, or a dated `README.md` inside a
  run directory) rather than a raw CSV/JSON — the verdict states what was measured, under what
  conditions, and which claim it supports.
- The gate table in [`../README.md`](../README.md) is the index from gate name to the artifact(s)
  under here that satisfy it.
- [`../research/2026-09-19_CLOSURE.md`](../research/2026-09-19_CLOSURE.md) and
  [`../HANDOFF.md`](../HANDOFF.md) cite the specific artifacts behind the headline research claims.

## Dated runs

| date | what's here (highlights) |
|---|---|
| [`2026-09-14/`](2026-09-14/) | device bring-up (G0), storage/DRAM probes (G1), the byte-budget roofline (G-ROOF v2), the first cache-simulation trace (G2), fidelity gate (G3), the RAM-grant sweep (G-RAM-SWEEP) |
| [`2026-09-16/`](2026-09-16/) | DRAM bandwidth (S1), reproducibility check (G-REPRO), external-validation against colibri (G-VALID-3), the S9 hit-rate-transfer pre-registration and confound control, baseline decode campaigns (G6-baseline) |
| [`2026-09-17/`](2026-09-17/) | BigMoeOnEdge reproduction and lever campaigns (arena, boost, pin, verify-cost, GPU expert cache), the thread-cliff fix, second-device (Nord) prediction |
| [`2026-09-18/`](2026-09-18/) | ceiling-ledger analysis, GPU expert-path design experiments, app-engine integration runs |
| [`2026-09-19/`](2026-09-19/) | the closing campaign: E1 (CPU compute floor), E4 (GPU helper), E6 (lossy-routing quality frontier), E7 (GPU as main engine), the **head-to-head vs BigMoeOnEdge** (`H2H_VERDICT.md`, the repo's headline result), the Nord sustained run, `CLOSURE.md`'s source data |
| [`2026-09-20/`](2026-09-20/) | first Meridian device-profiler runs on the Nord (`meridian_l1_nord*`, `meridian_app_nord`) |
| [`2026-09-22/`](2026-09-22/) | app-level measurements: compute-probe fits, pre-download predictions checked against real chat turns, agent eval-suite runs, on the 15R and Nord |

Each subdirectory's own files are documented inline in the gate tables and verdict documents
linked above; this index intentionally does not duplicate them.
