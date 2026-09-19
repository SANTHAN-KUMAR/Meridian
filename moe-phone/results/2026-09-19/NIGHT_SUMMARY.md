# Night of 2026-09-18/19: engine session summary (final, 05:45)

Levels: L3 (what can each lever deliver?), L5 (engine plumbing) and L6 (verdicts). Every number below comes from the
artifact named next to it.

## Verdicts
| experiment | result | artifact |
|---|---|---|
| GPU expert tier, phone held Awake (gx v1 + spin, cap 8, 1000 MiB) vs the stack, ABBA x6 | **net-neutral**: decode 6.70 vs 6.76 tok/s, not resolved (~0); compute +6.9 ms (decisive); stall -5.3 ms (decisive); text 24/24 identical | bmoe_gtier/bmoe_gtier_ab_20260919_0447/README.md |
| GPU tier, cap 3, ran Dozing | **not resolved**: SoC autosuspend made rows bimodal in both arms | bmoe_gtier/bmoe_gtier_ab_20260919_0235/README.md |
| pinprobe: kgsl memory as the CPU cache | BUILD: reads at malloc speed (62.9 vs 61.7 GB/s); immune to reclaim (0 vs 196,608 major faults after PAGEOUT) | pinprobe/README.md |
| kgsl arena (4.6 GB pinned cache) vs the stack | **stopped after 4 rows**: pinning the cold cache pushes the system's reclaim onto the engine's hot anonymous data (104,680 majflt/30 s); compute +78 ms; the effect carries over between rows | bmoe_kgsl/bmoe_kgsl_ab_20260919_0131/README.md |
| gx kernels (gx session, M6 runs 3-5) | page-locality hypothesis refuted (memory-only reads reach 31-33 GB/s in every layout); kernels are compute-bound; unrolled accumulators give v3 13.6x and v1 0.90 -> 0.70 ms device at k=3 | gx_m6_001107, gx_m6_012604, gx_m6_020545 |

## Found and fixed tonight (mine unless noted)
- The Doze confound: unplugged, `svc power stayon true` does not hold, so the phone dozed and the SoC autosuspended during rows.
  Fix: a per-row waker in device/bmoe_gtier.sh and a pre-registered `--require-awake` keep rule in gates/stack_summary.py.
  The old Dozing run keeps 0/24 rows under it (negative control).
- A wrong mechanism, retracted: I proposed that overflow read-maps cost ~40 ms/token. The engine counters measure ~0.09 ms
  per overflow expert, so the claim was deleted.
- Engine bugs: v3 slots were written unrepacked (found by the gx session); a bare `--gpu-spin-wait` swallowed the next flag
  (smoke 0223 invalid); `LD_LIBRARY_PATH` included /vendor/lib64 (smoke 2344 invalid).
- adb was lost at 00:38. The phone did **not** reboot (uptime 13064 s at 01:25). The user restored adb.

## What this says about 10 tok/s (L3)
- Awake and unplugged, the stack decoded 6.76 tok/s at steady state (the base arm above, mostly capped clocks). On the
  charger last night the same arm gave 5.3-6.3. That comparison is not controlled; X3 (off-charger equilibrium) deserves its own run.
- The GPU tier pays only when a dispatch finishes, as seen from the host, faster than the CPU would compute the same ~2
  experts (~0.2-0.25 ms). Today it takes ~0.6 ms. The gx session's fixed-cost work (fusion, staging, doorbell dispatch) is
  that lever. Even a perfect device is worth ~14 ms/token on this tier.
- Pinning memory helps only for the hot set. The dense weights (`--dense-weights ahwb`, never measured here) are the
  candidate, and their ceiling is small (~3 ms/token of faults in the node trace).
- None of tonight's levers reaches 10 tok/s. The honest position is the one in research/2026-09-18_ceiling_handoff.md: the
  capped CPU's arithmetic is the wall.
