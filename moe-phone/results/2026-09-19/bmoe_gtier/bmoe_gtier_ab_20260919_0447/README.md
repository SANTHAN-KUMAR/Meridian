# GPU tier A/B, phone held Awake: gx v1 + spin, cap 8, 1000 MiB tier (bmoe-i8mm-0023) vs the stack

Scored with the pre-registered rule plus the pre-registered awake keep rule (gates/stack_summary.py --any-budget-prereg
--require-awake). The output is results/2026-09-19/gtier_cap8_awake_summary.json. 23 of 24 rows were kept: stack_rep4b ended Dozing (4.07 tok/s)
and was excluded by that rule.

| term (ms/token unless noted) | base | tier | diff | SE/|diff| | sign | verdict |
|---|---|---|---|---|---|---|
| decode tok/s | 6.755 | 6.700 | -0.07 | 1.04 | 4/6 | not resolved (~0) |
| compute | 91.6 | 97.9 | **+6.9** | 0.13 | 6/6 | decisive, worse |
| stall | 34.6 | 29.5 | **-5.3** | 0.16 | 6/6 | decisive, better |
| stall + mgmt | 56.6 | 51.6 | -5.2 | 0.19 | 6/6 | decisive, better |
| read MiB/token | 121.2 | 127.2 | +5.8 | 0.16 | 6/6 | decisive (the CPU cache is 1 GB smaller) |

Text was identical in 24/24 rows. Device: 0 failures, 0 risk recomputes, ~2.1 experts per dispatch. Host time per row:
overflow map+unmap ~34 ms, promotions ~61 ms, GPU wait ~3.2 s (~12 ms/token).

**Verdict: at the current kernel speed the GPU tier is net-neutral on decode.** The CPU waits on the device (~12 ms/token)
about as long as the CPU work the device saves. So compute rises +7 ms and the stall falls -5 ms. A plausible reading of the
stall drop, not tested here: the longer compute phase hides more of the flash reads. For the tier to pay, the device's
per-dispatch fixed cost (~0.25-0.3 ms device + ~0.2 ms host at k~2; gx_m6_020545 fits) has to fall below the CPU time of
the ~2 experts it takes (~0.2-0.25 ms).

Observation, not a controlled comparison: the base arm, unplugged and Awake with the screen on at brightness 2, decoded 6.76
tok/s at steady state. Its caps were mostly policy0 1.9-2.19 / policy6 1.65 GHz. The stack's base arm on the charger
(results/2026-09-18) was 5.3-6.3. X3 (off-charger equilibrium) deserves its own run.
