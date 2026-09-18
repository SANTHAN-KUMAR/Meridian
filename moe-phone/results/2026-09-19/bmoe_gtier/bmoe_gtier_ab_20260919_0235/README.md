# GPU tier A/B (gx v1 + spin, cap 3, bmoe-i8mm-0022): NOT RESOLVED, confounded by device suspend

Pre-registered rule (device/bmoe_gtier.sh; scored with gates/stack_summary.py --any-budget-prereg, which writes
results/2026-09-19/gtier_ab_summary.json):
- decode: tier 4.76 vs base 5.72 tok/s. Per-repeat differences are -2.85, +0.67, -2.22, +1.24, +0.46, -3.07. SE/|mean| is 0.83 and the sign holds
  in only 3 of 6 repeats, so the result is **not decisive**.
- compute: +35.5 ms (SE/|mean| 0.57, sign 3/6), not decisive. read MiB/token +5.5 (decisive: the CPU cache is 1 GB smaller).
- Text was identical in 24/24 rows. The tier had 0 failures and 0 risk recomputes.

Why it is not interpretable: every row ran with wakefulness=Dozing (the phone was unplugged, and `svc power stayon true`
keeps the screen on only while the phone is plugged in). With no wakelock, Android autosuspends the SoC even while the
engine runs. android.system.suspend-service used ~34% CPU in most pre-row snapshots. The rows are **bimodal in both arms**:
- base: compute ~80-86 ms (about 7 tok/s) or 134-180 ms (3.3-4.4 tok/s)
- tier: 106-108 ms or 137-203 ms
The two arms' difference is smaller than this hidden two-state factor, which is not controlled. All campaigns before
tonight ran on the charger with wakefulness=Awake. The kgsl-arena A/B (stopped early) ran under the same condition.

The only thing both arms' FAST modes say (a labelled observation, not a verdict): base-fast is 80-86 ms compute and
tier-fast is 106-108 ms. That points to a tier cost when the device is awake, which matches the overflow read-map
mechanism (bmoe_gtier.sh FOLLOW-UP note).
Next: rerun with the device held Awake (a screen-on waker before each row, wakefulness logged at row start and end).
