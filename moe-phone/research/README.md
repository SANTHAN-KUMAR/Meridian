# moe-phone/research — literature memos behind HEADROOM.md §4

Three research memos produced on 2026-09-17 by AI research subagents (Claude Opus, WebSearch /
WebFetch) at the project owner's direction, one per time term of the measured phone token
(compute residual, I/O stall + cache management, bytes per token / multi-token). Each was briefed
with the project's settled measurements and its closed negatives, and told not to re-propose them.

Provenance, in `LEADS.md`'s marks: everything in these files is **[U]** — fetched and read by a
subagent, numbers **not re-checked** by a human or by the main session, except where
`HEADROOM.md` says a claim was verified locally (build flags, artifact values). Each memo marks
its own "not verified" items inline. Treat the memos as a lead list, exactly like `LEADS.md`:
nothing here is a claim of this project until it is measured on the 15R and enters `CLAIMS.md`.

| file | question | the memo's own top item |
|---|---|---|
| `2026-09-17_compute_memo.md` | what cuts the ~100 ms/token CPU compute residual | instrument first (barrier ledger + counters); the frequency cap is an OEM governor, not thermal (a sibling-device hypothesis; the 15R's caps covary with temperature, mechanism unresolved: `host/gpu_ffn/NOTE.md` §10) |
| `2026-09-17_io_memo.md` | what removes the I/O stall and cache-management terms losslessly | a pre-committed never-decommitted slot arena; a split-phase (`gate|up` then `down`) expert read |
| `2026-09-17_bytes_memo.md` | what cuts bytes/token or yields >1 token per fetch without training | MTP-head checkpoints with the verify on the GPU; KL-instrument dynamic top-k |

## 2026-09-18 — analysis memos (not literature)

| file | what it is |
|---|---|
| `2026-09-18_why_levers_flip.md` | why the speed levers came back neutral: A/B power, self-cancelling prefetch, arena regression |
| `2026-09-18_ceiling_handoff.md` | **handoff and execution plan**: the 10 tok/s ceiling is capped-clock arithmetic, not I/O; two lossless compute levers, a best case near 8 tok/s, and the five decision experiments. Numbers from `gates/ceiling_ledger.py` -> `results/2026-09-18/ceiling_ledger.json` |
