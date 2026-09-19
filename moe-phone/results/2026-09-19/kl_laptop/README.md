# E6 fidelity plumbing, laptop (OLMoE-1B-7B Q4_0, x86), 2026-09-19. These are checks of the KL tool, not E6 results
Expected values were fixed by the maths before the runs:
- identity (top-8 scored against its own dump): KL 0.000000, 0/561 flips (batch mode) and 0/3256 flips (step mode, 4096 context, e6 corpus). As expected.
- ordering top-7 < top-6 < top-4: mean KL 0.0106 < 0.0286 < 0.170 (K=256). As expected.
- misaligned text: refused at position 336, the first differing token. As expected.
- tightness of the bound: at K=256 the worst reference tail mass was 0.42 and top-6's mean KL 0.0286; at K=4096 the tail is 0.073 and KL 0.0307
  (+7%; p99 +16%). **E6 uses K=8192** (tail 0.055 on the e6 corpus).
- drop-cold 1.0 at a 2000 MiB cache on OLMoE: 16.7% of experts dropped, KL 0.28, 21.7% flips. Heavily lossy at that cache size; E6
  measures it on Qwen3 at the deployed 5 GB cache.
The .bkl dumps were deleted (up to 213 MB each); the .out/.err files are the record.
