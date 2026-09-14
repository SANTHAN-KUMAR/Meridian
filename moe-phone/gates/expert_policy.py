"""
Expert-cache POLICY comparison — what a phone engine should actually do.

WHY THIS MODULE EXISTS, AND WHY ITS ORIGINAL PREMISE WAS WRONG.

It was written to answer "if evicting is what breaks LRU, what does a cache
that never evicts achieve?" — because G2 appeared to show LRU getting no hits
at all at phone-sized caches. That premise did not survive: the figure it
rested on came from simulating ONE cache shared by all layers while this
module's own pinning policy was per layer, so the comparison was between two
different machines (cache_sim.py's docstring has both defects and the fix, and
POSITION.md §3c has the corrected numbers).

Corrected, per layer and event-atomic, on OLMoE-1B-7B: LRU reaches 0.279 at a
10% cache against a 0.100 no-locality floor. It is not failing. It beats every
policy compared here at every cache size from 10% up, including the static
oracle that cheats by knowing the whole trace's frequencies, and cache warming
is indistinguishable from plain LRU once past the warmup because LRU re-reaches
the same state within a few hundred tokens.

So this module's answer is now a NEGATIVE one, and that is its value: never
evicting is worse than evicting by recency, on this trace, at every cache size
that matters. The remaining gap to Belady is not an eviction-rule problem and
cannot be closed by choosing a better rule — it needs lookahead, which
cache_sim.lookahead_hits() measures.

Experts are not used uniformly — per layer, the most popular quarter of OLMoE's
experts take ~41-52% of selections against 25% under uniform routing — which is
why pinning them looks promising and why it still loses.

Policies compared, all at the same per-layer cache budget:

  floor          cache_fraction. What uniform independent routing gives any
                 policy; the number G-ROOF assumes.
  LRU            what a naive engine does. Reported by gates/cache_sim.py.
  static-oracle  pin the globally most popular experts PER LAYER, never evict.
                 Frequencies taken from the whole trace, so this is an upper
                 bound on any static policy — it cheats by knowing the future
                 distribution, exactly as Belady cheats by knowing the future
                 sequence. NOT a deployable number.
  static-warmup  the deployable version: frequencies estimated from the first
                 --warmup-frac of the trace, then frozen and evaluated on the
                 REMAINDER only. No peeking. This is the number an engine can
                 actually obtain.
  Belady         the offline optimum, an upper bound for ANY policy.

Also reported: **working-set drift**. If the popular set is stable over time, a
static cache holds up and no task classifier is needed. If it drifts, that drift
is the headroom a classifier (or any adaptive policy) could recover, and it is
measured here rather than assumed. Drift is the overlap between the top-N sets
of consecutive trace segments, per layer.

Run:
  python moe-phone/gates/expert_policy.py traces_OLMoE-1B-7B-0924.npz
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_sim  # noqa: E402

FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.50]


def per_layer_capacity(cap_total, n_layers):
    """Split a total expert budget evenly across layers.

    Even splitting, not proportional-to-traffic: every layer runs on every
    token, so a layer starved of cache stalls the whole token. A proportional
    split is a tuning knob and would need its own justification.
    """
    return max(1, cap_total // n_layers)


def static_hits(traces, caps, warmup_frac=None):
    """Hit rate of a never-evicting cache holding the most popular experts.

    warmup_frac=None  frequencies from the whole trace (oracle, upper bound).
    warmup_frac=f     frequencies from the first f of tokens; hit rate measured
                      on the rest only, so the evaluation never sees the data
                      the policy was chosen on.
    """
    hits = total = 0
    per_layer = {}
    if isinstance(caps, int):        # one budget for every layer
        caps = {l: caps for l in traces}
    elif not isinstance(caps, dict):  # a list in sorted-layer order
        caps = dict(zip(sorted(traces), caps))
    for l, t in traces.items():
        cap_per_layer = caps[l]
        T = t.shape[0]
        if warmup_frac:
            split = max(1, int(T * warmup_frac))
            fit, evaluate = t[:split], t[split:]
        else:
            fit, evaluate = t, t
        if evaluate.size == 0:
            continue
        counts = np.bincount(fit.ravel().astype(np.int64))
        pinned = set(np.argsort(counts)[::-1][:cap_per_layer].tolist())
        h = int(np.isin(evaluate, list(pinned)).sum())
        hits += h
        total += evaluate.size
        per_layer[l] = h / evaluate.size
    return (hits / total if total else 0.0), per_layer


def working_set_drift(traces, top_n, n_segments=8):
    """Jaccard overlap of the top-N expert set between consecutive segments.

    1.0 = the popular set never changes, so a static cache is optimal and an
    adaptive policy or task classifier has nothing to recover. Lower values are
    the headroom such a policy could address.
    """
    overlaps = []
    for t in traces.values():
        seg = np.array_split(t, n_segments)
        tops = []
        for s in seg:
            c = np.bincount(s.ravel().astype(np.int64))
            tops.append(set(np.argsort(c)[::-1][:top_n].tolist()))
        for a, b in zip(tops, tops[1:]):
            overlaps.append(len(a & b) / len(a | b))
    return float(np.mean(overlaps)), float(np.min(overlaps))


def main():
    p = argparse.ArgumentParser(description="expert-cache policy comparison")
    p.add_argument("npz")
    p.add_argument("--fractions", default=",".join(str(f) for f in FRACTIONS))
    p.add_argument("--warmup-frac", type=float, default=0.25,
                   help="fraction of the trace used to FIT the static policy; the hit rate is "
                        "then measured on the remainder only")
    p.add_argument("--scope", default="per_layer", choices=["per_layer", "global"])
    p.add_argument("--replay", default="atomic", choices=["atomic", "sequential"])
    p.add_argument("--out-name", default=None)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    z = np.load(a.npz)
    E = int(z["num_experts"])
    traces = {int(k[1:]): z[k] for k in z.files if k.startswith("L")}
    L = len(traces)
    T, k = next(iter(traces.values())).shape
    ev, T, L, k = cache_sim.event_stream(traces, E)
    print(f"{T} tokens x {L} MoE layers x top-{k} of {E} experts")
    print(f"static policies fitted on the first {100 * a.warmup_frac:.0f}% of tokens, "
          f"scored on the remaining {100 * (1 - a.warmup_frac):.0f}%\n")

    res = {"npz": os.path.abspath(a.npz), "tokens": T, "layers": L, "top_k": k,
           "num_experts": E, "warmup_frac": a.warmup_frac,
           "scope": a.scope, "replay": a.replay, "f_crit": k / E, "rows": []}
    print(f"scope={a.scope} replay={a.replay}   f_crit = k/E = {k / E:.4f}")
    print()
    print(f"{'cache':>6} {'rho':>5} {'floor':>7} {'LRU':>7} {'static-warmup':>14} "
          f"{'static-oracle':>14} {'Belady':>8}")
    for f in [float(x) for x in a.fractions.split(",")]:
        cap_total = max(1, int(round(f * E * L)))
        caps = cache_sim.split_capacity(cap_total, L)
        cap_layer = min(caps)
        n_req = ev.shape[0] * ev.shape[1] * ev.shape[2]
        lru = cache_sim.lru_hits_events(ev, cap_total, scope=a.scope,
                                        replay=a.replay) / n_req
        bel = cache_sim.belady_hits_scoped(ev, cap_total, scope=a.scope) / n_req
        s_or, _ = static_hits(traces, caps)
        s_wu, _ = static_hits(traces, caps, warmup_frac=a.warmup_frac)
        rho = f * E / k
        print(f"{f:>6.3f} {rho:>5.2f} {f:>7.3f} {lru:>7.3f} {s_wu:>14.3f} "
              f"{s_or:>14.3f} {bel:>8.3f}")
        res["rows"].append({"cache_fraction": f, "cap_total": cap_total,
                            "cap_per_layer_min": cap_layer, "rho_per_layer": rho,
                            "floor": f, "lru": lru,
                            "static_warmup": s_wu, "static_oracle": s_or, "belady": bel})

    # ------------------------------------------------------------------
    # Online policies, all scored on the SAME held-out tail so none of them is
    # reporting its own training fit. This is the table ARCHITECTURE.md 2.3
    # cites for "no classical policy beats LRU", and the claim is only as good
    # as this being reproducible, so it runs by default.
    # ------------------------------------------------------------------
    warm, split = cache_sim.warm_sets(traces, E, a.warmup_frac)
    names = ["LRU", "LFU", "warm-LRU", "pin25", "pin50", "pin100"]
    res["online_policies"] = {"fitted_on_first_tokens": split, "rows": []}
    print()
    print(f"online policies, all scored on tokens {split}..{T} only")
    print(f"{'cache':>6} {'floor':>6} " + " ".join(f"{x:>8}" for x in names)
          + f" {'Belady':>8}   winner")
    for f in [float(x) for x in a.fractions.split(",")]:
        cap = max(1, int(round(f * E * L)))
        got = {}
        got["LRU"] = cache_sim.policy_hits(ev, cap, "lru", eval_from=split)
        got["LFU"] = cache_sim.policy_hits(ev, cap, "lfu", eval_from=split)
        got["warm-LRU"] = cache_sim.policy_hits(ev, cap, "lru", warm=warm, eval_from=split)
        for pf, nm in ((0.25, "pin25"), (0.5, "pin50"), (1.0, "pin100")):
            got[nm] = cache_sim.policy_hits(ev, cap, "lru", warm=warm,
                                            pin_frac=pf, eval_from=split)
        rates = {kk: (h / seen if seen else 0.0) for kk, (h, seen) in got.items()}
        bel = cache_sim.belady_hits_scoped(ev[split:], cap, a.scope)
        bel /= max(1, (T - split) * L * k)
        best = max(rates, key=rates.get)
        print(f"{f:6.3f} {f:6.3f} " + " ".join(f"{rates[x]:8.3f}" for x in names)
              + f" {bel:8.3f}   {best} {rates[best]:.3f}"
              + (f" = {100 * rates[best] / bel:.0f}% of Belady" if bel > 0 else ""))
        res["online_policies"]["rows"].append(
            {"cache_fraction": f, "belady": bel, "winner": best, **rates})
    print("  pin100 never evicts; pin25/50 reserve that share of each layer's slots for the")
    print("  warm set and run LRU in the rest. warm-LRU pre-fills the cache and then evicts")
    print("  normally, which is cache warming as a serving stack does it.")

    top_n = min(cache_sim.split_capacity(max(1, int(round(0.10 * E * L))), L))
    mean_ov, min_ov = working_set_drift(traces, top_n)
    res["working_set_drift"] = {"top_n_per_layer": top_n, "n_segments": 8,
                                "mean_jaccard": mean_ov, "min_jaccard": min_ov}
    print(f"\nworking-set drift (top-{top_n}/layer, 8 segments): "
          f"mean Jaccard {mean_ov:.3f}, worst {min_ov:.3f}")
    print("  1.0 would mean the popular set never moves, so a static cache is already optimal")
    print("  and an adaptive policy or task classifier has nothing left to recover.")

    name = a.out_name or ("expert_policy_" + os.path.basename(a.npz)
                          .replace("traces_", "").replace(".npz", ""))
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        out = os.path.join(a.out_dir, name + ".json")
    else:
        from _paths import write_path
        out = write_path(name + ".json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
