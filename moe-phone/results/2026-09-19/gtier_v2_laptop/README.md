# gx variant 2 in the engine: laptop plumbing check (OLMoE-1B-7B Q4_0, NVIDIA OpenCL; 2026-09-19)

Purpose: this checks the engine's variant-2 plumbing, not speed. Slot writes go through gx_repack_expert (promotion and
warm start) and the CPU paths (overflow, risk recompute) copy out with gx_unpack_expert. The laptop timings are not a
result: this is an x86 host and an NVIDIA GPU. Flags are in CMD.txt, plus the per-row flags below.

| row | flags | text vs off |
|---|---|---|
| off | no tier | reference |
| v1 / v2 | --gpu-tier-mb 500 --gpu-backend gx --gpu-variant 1 / 2 | identical / identical |
| v1w / v2w | the same, plus --gpu-tier-prior test_prior_synthetic.txt (148 slots warm) | differ from off at word 35; **v1w == v2w** byte for byte |
| sw | standin backend, the same synthetic prior | **identical** |

Reading:
- v2 == v1 everywhere, including 973 and 644 copy-outs through gx_unpack_expert. So repack and unpack are exact inside
  the engine.
- The warm-start rows differ from off with gx, but not with the standin, which is ggml-cpu's own x86 arithmetic, given
  the same prior and the same code path. So the engine logic is exact. The difference comes from gx's arithmetic, which is
  bit-matched to ggml-cpu's **ARM** path (M3/M6 acceptance), not to this laptop's x86 AVX2 path. More device experts
  (2437 vs 1336) made a divergence more likely. The no-prior gx rows matched off, but that says nothing about exactness.
- The deciding identity test is on the phone (ARM CPU plus Adreno): device/bmoe_gtier.sh smoke, compared by text.
- Counters for all rows: 0 failures, 0 risk recomputes, 0 map errors.
- test_prior_synthetic.txt is a test list (every 6th expert of layers 0-15), not a popularity prior.

## Variant 3 (gx 2783e0c, tiled SoA), same check, 00:50
| row | experts on device | risk-flagged, recomputed on the CPU |
|---|---|---|
| v1 / v2 | 1336 / 1336 | 0 / 0 |
| v3 | 1347 | **1347** |
| v1w / v2w | 2437 / 2437 | 0 / 0 |
| v3w | 2469 | **2469** |

**Defect: under v3, gx's risk mask flags every slot.** The engine therefore recomputes every v3 expert on the CPU. So v3's
text match (v3 == v1) says nothing about v3's arithmetic, and v3 cannot be used in the engine until this is fixed.
Reported to the gx session. v3w's first attempt (exit 1, "unknown arg") was a quoting error in the command, not the
engine; the row was rerun.
