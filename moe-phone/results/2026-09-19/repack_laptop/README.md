# Repack path on the laptop (OLMoE-1B-7B Q4_0, x86 q4_0_8x8), plumbing and fidelity checks, 2026-09-19 08:40-09:00
- --repack-experts (repack at run time on the I/O lanes), stack flags, 128 tokens: text identical to generic (rep.out vs gen.out).
- --repack-dense: 64 dense tensors (144 MiB) repacked, 0 failed. Alone, its text diverges at word 35 (a near-tie that also flipped
  in the GPU warm-start rows); together with --repack-experts it matches generic. Floating-point reordering, not a bug.
- repack_gguf on a copy: 48 tensors / 3072 slices rewritten, marker written. --experts-prerepacked on it == runtime repack
  byte for byte (pre.out). Negative controls: the flag on the ORIGINAL file is refused (marker mismatch), and the repacked
  file WITHOUT the flag is now refused by the engine (before the guard it produced garbage: raw_on_repacked.out).
- Teacher-forced perplexity on defer_ppl_text.txt (561 scored tokens): generic 8.2675, pre-repacked 8.2637 (-0.046%);
  next-token hits 318 vs 316. These set the declared reordering tolerance in device/bmoe_prerepack.sh (0.2%, 1%).
