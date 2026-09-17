# Resident-first expert order: it fires, it is lossless, and it is NOT YET MEASURED

Patch 0010 (`--expert-resident-first`). Three campaigns; the third is the one to believe and it was
still running when this was written. Claims: `order_idorder_decode`, `order_residentfirst_decode`,
`order_cover_bound_pct`, `order_probe_deferred_pct`.

## What was predicted, and why

`gates/decode_budget.py` splits the best measured cell into **93.0 ms compute residual + 20.0 ms cache
management + 48.3 ms stall** per token against a 59.3 ms CPU floor. That stall is almost exactly the
whole of the cell's miss traffic, i.e. `--overlap` was hiding essentially none of it. The mechanism was
in ggml, not in the engine: `mul_mat_id` consumes a layer's selected experts in ascending expert id and
the expert-ready hook blocks on each in turn, so a layer waits on its lowest-id miss while its resident
experts queue behind it. Patch 0010 computes the resident ones first.

`gates/order_cover.py` bounds what that can buy, from measured inputs only: per layer the resident
expert bytes are ~18.0 MB and the missing ones ~3.3 MB, so the cover is **0.58 ms of WALL time** (at the
CPU's measured 31.01 GB/s — four threads eat a layer's resident work four times faster than a
single-threaded intuition suggests) against **1.17 ms** of miss reads (at the measured 2.806 GB/s bulk
flash rate). Upper bound: **17.2% of the token** (claim `order_cover_bound_pct`).

## It fires

The instrumented arm counts the probe's answers: **9.68% of expert consumptions were deferred to the
second pass** (claim `order_probe_deferred_pct`; the three rows agree at 9.32 / 9.68 / 9.87%). So this
is not a switch that does nothing — it defers roughly the share the miss rate implies.

## The two first campaigns disagree, and both are confounded

| campaign | ascending id | resident first | separation |
|---|---|---|---|
| `bmoe_order` (3 repeats) | 5.837 tok/s median | 5.835 | none (0.06%) |
| `bmoe_order2` (3 repeats, instrumented) | 5.804 | **6.151 (+6.0%)** | every resident-first row beat every ascending-id row |

A perfect 3-vs-3 separation has a one-sided permutation probability of 1/20, so campaign 2 looks like a
result. **It is not, and the reason is my own design error.** With two cells and an alternating
rotation, three repeats give `A,B / B,A / A,B` — so one arm sits in position 1 twice and the other in
position 2 twice. Position is not neutral on this phone: `gates/position_effect.py` finds within-repeat
drift of up to 61% across past campaigns, in either direction depending on that campaign's own thermal
and cache history. In campaign 2 resident-first held position 2 twice, and the within-repeat
position-2 change was +4.3%, −2.6%, +9.6%. That is enough to manufacture the whole 6%.

Campaign 1 had the same unbalanced design and produced nothing, which is the other half of the reason
not to believe either: two campaigns, same confound, opposite answers.

## What is being done about it

`device/bmoe_order3.sh`: **ABBA within every repeat** (`A,B,B,A`, with the leading arm swapped on
alternate repeats), 3 repeats = 12 runs = 6 per arm, instrumented binary. Each arm then takes one early
and one late slot per repeat, so a monotone drift cancels inside the repeat instead of loading onto one
arm. On the pooled within-arm spread measured here (3.9% of the mean), 6 per arm resolves about a 6%
effect at 80% power — which is exactly the size in dispute.

## What is settled either way

* **Lossless.** All twelve rows across the two campaigns produced byte-identical generated text, checked
  by the campaigns themselves rather than assumed. Only the order of a sum changes.
* **Bounded.** Even at its ceiling the lever is worth 17.2% of a token, so it cannot reach the project's
  goal by itself — 6.2 tok/s + 17% is 7.3, against a target of 10.
* **The big cover does not exist inside a layer.** A whole layer's compute (~2.8 ms) would cover any
  miss, but it only exists *across* layers, and that needs the next layer's routing before the current
  layer's FFN — prediction, which is measured and negative for this design (`gates/predictor.py`).
* So the remaining levers are the other two terms: the **hit rate** (the ZRAM oversized-cache campaign)
  and the **compute residual** — 93 ms against a 59.3 ms floor (the thread sweep and the per-op device
  sweep).
