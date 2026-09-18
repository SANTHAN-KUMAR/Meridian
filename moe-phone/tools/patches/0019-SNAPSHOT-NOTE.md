# 0019-SNAPSHOT — the exact engine tree the gx builds were made from

`0019-SNAPSHOT-bigmoeonedge-engine-tree-vs-74ba18f.patch` is **cumulative** (updated 2026-09-19 00:30 for gx variant 2: write_expert/read_expert, repack/unpack; Android bmoe-cli md5 752d281d4ad9f82e0ffafb177b91ad31 = bmoe-i8mm-0019v2): `git apply` on a clean BigMoeOnEdge at
74ba18f reproduces cli/ and core/ byte for byte, as they were when the gx engine (bmoe-i8mm-0019gx) was built. This was verified on
2026-09-19 with `git archive 74ba18f | tar x; git apply; diff -r`, and it showed no differences. The ggml-cpu split hooks
live in the llama.cpp submodule and are applied separately by 0017.

Why a snapshot and not an incremental patch: the incremental series 0010–0018 was generated against a moving tree and
no longer replays cleanly onto 74ba18f (hunks fail in main.cpp, config.h, expert_stream_source.*). The older
`0019-bigmoeonedge-gpu-expert-tier-standin.patch` predates the gx backend and --gpu-spin-wait. It is superseded by this
snapshot and kept only as history.
