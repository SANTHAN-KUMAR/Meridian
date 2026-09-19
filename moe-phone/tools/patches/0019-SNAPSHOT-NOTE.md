# 0019-SNAPSHOT — the exact engine tree the gx builds were made from

`0019-SNAPSHOT-bigmoeonedge-engine-tree-vs-74ba18f.patch` is **cumulative** (updated 2026-09-19 00:30 for gx variant 2: write_expert/read_expert, repack/unpack; Android bmoe-cli md5 752d281d4ad9f82e0ffafb177b91ad31 = bmoe-i8mm-0019v2): `git apply` on a clean BigMoeOnEdge at
74ba18f reproduces cli/ and core/ byte for byte, as they were when the gx engine (bmoe-i8mm-0019gx) was built. This was verified on
2026-09-19 with `git archive 74ba18f | tar x; git apply; diff -r`, and it showed no differences. The ggml-cpu split hooks
live in the llama.cpp submodule and are applied separately by 0017.

Why a snapshot and not an incremental patch: the incremental series 0010–0018 was generated against a moving tree and
no longer replays cleanly onto 74ba18f (hunks fail in main.cpp, config.h, expert_stream_source.*). The older
`0019-bigmoeonedge-gpu-expert-tier-standin.patch` predates the gx backend and --gpu-spin-wait. It is superseded by this
snapshot and kept only as history.

Update 2026-09-19 00:45: the snapshot also carries `--arena-kgsl` (slot arena in OpenCL ALLOC_HOST_PTR memory, CPU-only,
`gpu_pinned_alloc` in gpu_tier.cpp). The Android build is bmoe-i8mm-0020kgsl, bmoe-cli md5 0596bb92787f70a2e441dedec1c91717.
It is a superset of 0019v2.

Update 2026-09-19 01:00: repacked layouts for every gx variant >= 2 (the v2-only check wrote v3 slots unrepacked). Android bmoe-i8mm-0021, bmoe-cli md5 49e304df484689b27307d00a33926e44, supersedes 0020kgsl and 0019v2.

Update 2026-09-19 01:35: engine unchanged; bmoe-i8mm-0022 = this tree + libgx 58b13dd (unrolled kernels), bmoe-cli md5 4263e767a5e1c552820e91683fc71d52.

Update 2026-09-19 06:05: the repack decision comes from libgx (gx_variant_uses_repack: v2, v3), not from variant >= 2; v4 is native layout. Also carries the host-time counters (overflow map/unmap, promotion) of 0023.

Update 2026-09-19 09:00: --repack-dense (DenseWeights::repack_matmul_weights), --experts-prerepacked (marker-checked, no run-time repack) and a guard that refuses a marker-carrying model without that flag. Android bmoe-i8mm-0025, bmoe-cli md5 b3a67ba93503d70aac9575f5158fed5c.

Update 2026-09-19 16:00: --ppl-dump / --ppl-ref / --ppl-kl-out / --ppl-dump-k (E6 fidelity: KL(ref || arm) over the reference's top-k + tail, top-1 flips, alignment-checked). Android bmoe-i8mm-0026, bmoe-cli md5 be978e3969ce1e363b7c5cf902c4d28a (ggml libs unchanged).
