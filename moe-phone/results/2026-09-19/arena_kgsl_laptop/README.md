# --arena-kgsl plumbing check (laptop, OLMoE-1B-7B Q4_0, NVIDIA OpenCL pinned host memory; 2026-09-19)
This checks correctness only. Timings from a single laptop run are not a result. Flags are in CMD.txt.
- off, --slot-arena and --arena-kgsl give identical text (96 tokens).
- The kgsl arena allocated 2196 MiB against a 2000 MiB budget: the arena's per-slice-size free lists overshoot the budget by
  about 10%, and the anonymous arena does the same. On the phone this memory cannot be swapped, so a phone arm must use a
  lower ceiling (--cache-ceil-mb 4000) to keep a margin.
