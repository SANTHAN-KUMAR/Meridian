# moe-phone / agentic — side track

**An on-device agent that performs automated tasks on the phone**, built on the same streamed-MoE
engine as the main line. Started 2026-09-18. **Side track: the main `moe-phone` line has priority,
and this track has no phone-time allocation.**

Nothing here has been measured on a device. What exists is the definition of what would be
measured, one pre-registered prediction derived from artifacts the main line already produced, and
a written record of which assumptions are load-bearing.

| file | what it is |
|---|---|
| [`ESTIMAND.md`](ESTIMAND.md) | **start here** — L1/L2/L3: the target quantity, what the decomposition assumes, what is estimable, the gates, the baselines, and the registered stubs |
| [`predict_context_tax.py`](predict_context_tax.py) | the pre-registered prediction of decode rate vs context length; separates the eviction and KV-traffic mechanisms |
| `prereg/context_tax_prediction.json` | its output, with calibration, recovery check, limitations and falsification rule |
| [`../host/chain_agentic.sh`](../host/chain_agentic.sh) | the phone queue for this track. **Disarmed** — refuses to run without an explicit approval variable |

```bash
python moe-phone/agentic/predict_context_tax.py   # regenerates the prediction; laptop only, ~1 s
```

## The three things that decide whether this is viable

All three are unmeasured. They are listed in the order that kills the project soonest.

1. **Can the engine hold a useful expert cache while the target app is foreground?** Every
   measurement this project holds was taken on a quiesced phone; a phone-task agent by definition
   runs alongside the app it is driving. `ESTIMAND.md` §3.3.
2. **What does prefill cost as a function of prompt length?** A phone-task turn is a large-ish
   prompt and a short output, so prefill dominates — and it has never been measured at any prompt
   length. `ESTIMAND.md` §3.1.
3. **Does the agent complete tasks at all, and better than a model small enough to fit in RAM?**
   The streamed 30B is slower by construction, so it has to be more capable by enough to pay for
   it. `ESTIMAND.md` §5.

## What was already true and did not have to be built

Read from the engine source, not assumed: the session protocol keeps the model loaded and the
expert cache warm across turns, and a turn re-ingests only the tokens not already in the KV.
The serving primitives an agent needs exist. `ESTIMAND.md` §2, A1–A3.

## A claim withdrawn here

An earlier conversation argued that KV-cache traffic was the dominant cost of context for an agent,
and that KV quantisation was therefore the decisive lever. The registered prediction separates that
mechanism from cache eviction and does not support it at the context lengths a phone-task agent
uses. The claim is withdrawn rather than annotated (`CLAUDE.md` §7.6); the artifact carries the
per-cell split so the reader can check the reasoning rather than take the correction on trust.
