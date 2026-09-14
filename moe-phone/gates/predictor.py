"""
S12 — the cross-token, same-layer expert predictor, measured rather than swept.

RESULT: NEGATIVE, and it retired a design. An earlier ARCHITECTURE.md §3 made a
cheap predictor the PRIMARY lever, on the strength of an abstract accuracy
sweep. Measured with real predictors, the lever is not there.

WHY THE SWEEP LOOKED BETTER THAN THIS, AND IT IS NOT THE NOISE MODEL.

Two explanations were considered and one was tested and REJECTED.

  Rejected: that the sweep flatters a predictor because corrupt_routing()
  replaces a wrong guess with a UNIFORMLY RANDOM expert, which is rarely
  resident, whereas a real predictor's errors are PLAUSIBLE experts that are
  often resident and so misfire the veto. Tested by corrupting from the trace's
  own routing at matched accuracy instead of uniformly: the two score the same
  to within 0.2%. The noise model is not the problem, and an earlier version of
  this docstring asserted that it was.

  Supported: HORIZON. The favourable sweep numbers were taken at horizon 4, and
  a one-step predictor can only honestly supply horizon 1. At horizon 4 even a
  35%-accurate predictor beats LRU (0.324 vs 0.287); at horizon 1 accuracy
  ~0.5 is needed merely to break even (0.286 vs 0.287). Uniform corruption at
  accuracy 0.35 and horizon 1 scores 0.257, against this script's measured
  Markov predictor at slot accuracy 0.338 scoring 0.262 -- so the abstract
  sweep predicts the real predictor correctly once accuracy AND horizon are
  matched. tests/test_gates.py asserts that agreement.

So the lookahead lever is real but is not reachable by prediction: it needs a
horizon of about 4, and a horizon of 4 needs the next four tokens' routing,
which only a multi-token verification pass actually has.

What this script measures: real predictors, fed to the real eviction rule, hit
rate read directly. No accuracy abstraction is involved in the headline.

WHAT IS BEING PREDICTED, AND WHY IT IS NOT WHAT G2 MEASURED

  The cache is per layer, so the useful question is CROSS-TOKEN, SAME-LAYER:
  which experts will token t+1 want at layer l? G2's 0.88 recall is a different
  quantity -- CROSS-LAYER lookahead, layer l+1 from layer l's input (the Fate /
  Mixtral-offloading trick). A per-layer cache cannot use that at all, because
  layer l's cache is never asked about layer l+1's experts.

CAUSALITY

  A prediction for token j must use only information available before the
  decision that consumes it. cache_sim.lookahead_hits() takes the prediction as
  one array indexed by absolute position, which is exactly right for a ONE-STEP
  predictor at horizon 1: standing in token t, the rule reads the predicted set
  for token t+1, and a one-step predictor formed that set from token t.

  At horizon >= 2 the same array would hand a persistence predictor token t+1's
  TRUE routing while standing at token t, which it cannot know. So the
  history-based predictors are evaluated at horizon 1 only. Predictors that are
  time-invariant (fitted once on a warmup slice and then frozen) carry no such
  restriction and are also reported at longer horizons.

Run:
  python moe-phone/gates/predictor.py results/<date>/traces_OLMoE-1B-7B-0924.npz
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_sim  # noqa: E402

FRACTIONS = [0.05, 0.10, 0.20, 0.30]


def _offset(layer_rank, num_experts):
    return layer_rank * num_experts


def predict_persistence(ev, num_experts, split, lag=1):
    """Token t+1 will want what token t wanted, at the same layer.

    The cheapest predictor there is: no state, no training, no parameters. Its
    quality is bounded by the adjacent-token expert overlap, which on this trace
    is 0.366 against 0.125 under independence -- nearly 3x chance, which is why
    it looks promising.

    It is nonetheless INFORMATIONALLY EMPTY for this cache, and that is the
    finding rather than the low score. The rule already knows the current token's
    own expert set exactly, because the engine is executing that layer right now
    (the `vt < tok_end` branch in cache_sim.lookahead_hits). Persistence says the
    next token will want that same set, so it contributes nothing the rule did
    not already have. Measured: it reproduces the horizon-0 column to three
    decimals at every cache fraction, which
    tests/test_gates.py::test_a_persistence_predictor_adds_nothing_to_what_the_engine_knows
    asserts.
    """
    hat = ev.copy()
    hat[lag:] = ev[:-lag]
    return hat


def predict_popularity(ev, num_experts, split):
    """The k most popular experts at that layer, fitted on the warmup and frozen.

    Time-invariant, so it is causal at any horizon. It is also the predictor a
    static-pinning cache would imply, which is why it is worth having here: G2
    found pinning loses to LRU, and this asks whether the same signal is useful
    when it may only VETO rather than decide.
    """
    T, L, k = ev.shape
    hat = np.empty_like(ev)
    for l in range(L):
        base = _offset(l, num_experts)
        c = np.bincount((ev[:split, l, :] - base).ravel().astype(np.int64),
                        minlength=num_experts)
        top = np.argsort(c)[::-1][:k] + base
        hat[:, l, :] = top
    return hat


def predict_markov(ev, num_experts, split):
    """First-order co-activation: from the warmup, count how often expert e' is
    used at token t+1 given e at token t, per layer. At run time, score every
    candidate by the summed transition counts from the current token's k experts
    and take the top k.

    Still training-free -- it is a count table built from the model's own
    routing, needing no gradients and no labels -- but it does carry per-layer
    state of size num_experts^2, which is 4096 counters per layer for OLMoE.
    """
    T, L, k = ev.shape
    hat = np.empty_like(ev)
    for l in range(L):
        base = _offset(l, num_experts)
        cur = (ev[:split, l, :] - base).astype(np.int64)
        M = np.zeros((num_experts, num_experts), dtype=np.float64)
        for t in range(cur.shape[0] - 1):
            M[np.ix_(cur[t], cur[t + 1])] += 1.0
        rows = M.sum(axis=1, keepdims=True)
        M /= np.maximum(rows, 1.0)                      # P(next | this)
        allc = (ev[:, l, :] - base).astype(np.int64)
        scores = M[allc].sum(axis=1)                    # [T, num_experts]
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k] + base
        hat[1:, l, :] = top[:-1]                        # prediction FOR t+1, made at t
        hat[0, l, :] = ev[0, l, :]
    return hat


def predict_oracle(ev, num_experts, split):
    """The exact future. Not deployable; it is the ceiling the others are read
    against, and it must reproduce cache_sim's exact-lookahead numbers."""
    return ev.copy()


PREDICTORS = {
    "persistence": (predict_persistence, True),    # True = history-based, horizon 1 only
    "markov": (predict_markov, True),
    "popularity": (predict_popularity, False),
    "oracle": (predict_oracle, True),
}


def slot_accuracy(ev, hat, split):
    """Fraction of predicted slots that are correct, averaged over (token, layer).

    Defined to line up with cache_sim.corrupt_routing()'s `accuracy` parameter so
    the measured predictors can be placed on the existing sweep -- but note the
    two are not interchangeable: corruption's errors are uniform, a predictor's
    are plausible experts. That is exactly why the hit rate below is measured
    directly instead of being read off the sweep at this number.
    """
    T, L, k = ev.shape
    tot = 0.0
    n = 0
    for t in range(split, T):
        for l in range(L):
            tot += len(set(ev[t, l].tolist()) & set(hat[t, l].tolist())) / k
            n += 1
    return tot / max(1, n)


def horizon_recall(ev, hat, split, horizon):
    """Of the experts actually needed in the next `horizon` tokens at that layer,
    what fraction does the prediction name? This is the quantity the PROTECT rule
    consumes: it vetoes eviction for experts the prediction expects, so a miss
    here is a needed expert left unprotected."""
    T, L, k = ev.shape
    hit = 0
    need = 0
    for t in range(split, T - horizon):
        for l in range(L):
            fut = set(ev[t + 1:t + 1 + horizon, l, :].ravel().tolist())
            pred = set(hat[t + 1:t + 1 + horizon, l, :].ravel().tolist())
            hit += len(fut & pred)
            need += len(fut)
    return hit / max(1, need)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("npz")
    p.add_argument("--fractions", default=",".join(str(f) for f in FRACTIONS))
    p.add_argument("--warmup-frac", type=float, default=0.25,
                   help="fraction of tokens used to FIT the fitted predictors; every number is "
                        "scored on the remainder only")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--out-name", default=None)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    z = np.load(a.npz)
    traces = {int(kk[1:]): z[kk] for kk in z.files if kk.startswith("L")}
    E = int(z["num_experts"])
    ev_all, T_all, L, k = cache_sim.event_stream(traces, E, a.max_tokens)
    split = max(1, int(T_all * a.warmup_frac))

    print(f"{T_all} tokens x {L} layers x top-{k} of {E}")
    print(f"fitted on the first {split}, scored on the remaining {T_all - split}\n")

    hats = {}
    for name, (fn, _) in PREDICTORS.items():
        hats[name] = fn(ev_all, E, split)

    # --- predictive quality, before any cache is involved -------------------
    print("predictive quality on held-out tokens:")
    print(f"  {'predictor':<14}{'slot acc':>10}{'recall@1':>10}{'recall@4':>10}")
    quality = {}
    for name in PREDICTORS:
        sa = slot_accuracy(ev_all, hats[name], split)
        r1 = horizon_recall(ev_all, hats[name], split, 1)
        r4 = horizon_recall(ev_all, hats[name], split, 4)
        quality[name] = {"slot_accuracy": sa, "recall_h1": r1, "recall_h4": r4}
        print(f"  {name:<14}{sa:10.3f}{r1:10.3f}{r4:10.3f}")
    print(f"  (independence baseline for slot accuracy is k/E = {k / E:.3f})")

    # --- what the cache actually gets ---------------------------------------
    # Scored on the held-out tail only, so a fitted predictor is never measured
    # on the tokens it was fitted on.
    ev = ev_all[split:]
    n = ev.shape[0] * L * k
    print(f"\nhit rate at horizon 1 (the horizon a one-step predictor can honestly supply):")
    hdr = f"{'cache':>6} {'LRU':>7}{'H=0':>7}"
    for name in PREDICTORS:
        hdr += f"{name[:9] + '/pro':>14}"
    hdr += f"{'Belady':>8}"
    print(hdr)
    rows = []
    for f in [float(x) for x in a.fractions.split(",")]:
        cap = max(1, int(round(f * L * E)))
        lru = cache_sim.lru_hits_events(ev, cap, "per_layer", "atomic") / n
        bel = cache_sim.belady_hits_scoped(ev, cap, "per_layer") / n
        # horizon 0 = the engine knows ONLY the token it is executing, which it
        # gets for free. A persistence predictor says "t+1 wants what t wants",
        # so it carries no information this column does not already have; if the
        # two agree, persistence is informationally empty rather than merely weak.
        h0 = cache_sim.lookahead_hits(ev, cap, 0) / n
        r = {"cache_fraction": f, "cache_experts": cap, "lru": lru, "belady": bel,
             "h0_current_token_only": h0,
             "horizon": 1, "predictors": {}}
        line = f"{f:6.3f} {lru:7.3f}{h0:7.3f}"
        for name in PREDICTORS:
            hv = hats[name][split:]
            pro = cache_sim.lookahead_hits(ev, cap, 1, ev_hat=hv, mode="protect") / n
            rnk = cache_sim.lookahead_hits(ev, cap, 1, ev_hat=hv, mode="rank") / n
            r["predictors"][name] = {"protect": pro, "rank": rnk,
                                     "protect_gain_vs_lru": (pro - lru),
                                     "protect_tok_s_ratio": (1 - lru) / (1 - pro)}
            line += f"{pro:14.3f}"
        line += f"{bel:8.3f}"
        print(line)
        rows.append(r)

    print("\nsame table as a THROUGHPUT ratio against plain LRU "
          "(bytes/token falls as (1-h), so this is the speedup):")
    print(f"{'cache':>6}" + "".join(f"{nm[:9] + '/pro':>14}" for nm in PREDICTORS))
    for r in rows:
        print(f"{r['cache_fraction']:6.3f}"
              + "".join(f"{r['predictors'][nm]['protect_tok_s_ratio']:13.2f}x"
                        for nm in PREDICTORS))

    print("\nrank vs protect for the SAME predictors (rank uses the predicted ordering):")
    print(f"{'cache':>6} {'LRU':>7}" + "".join(f"{nm[:9] + '/rnk':>14}" for nm in PREDICTORS))
    for r in rows:
        print(f"{r['cache_fraction']:6.3f} {r['lru']:7.3f}"
              + "".join(f"{r['predictors'][nm]['rank']:14.3f}" for nm in PREDICTORS))

    out = {"source": os.path.abspath(a.npz), "inputs": vars(a),
           "tokens": T_all, "layers": L, "top_k": k, "num_experts": E,
           "warmup_tokens": split, "independence_slot_accuracy": k / E,
           "quality": quality, "rows": rows}
    name = a.out_name or ("predictor_" + os.path.basename(a.npz)
                          .replace("traces_", "").replace(".npz", ""))
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        path = os.path.join(a.out_dir, name + ".json")
    else:
        from _paths import write_path
        path = write_path(name + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
