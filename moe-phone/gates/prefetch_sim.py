"""
WITHIN-TOKEN PREFETCH — can layer l's expert reads hide behind layer l-1's compute?

This is a LATENCY lever, not a bandwidth one, and it is a different mechanism from
the cross-token prefetch the literature reports as net-negative (LEADS.md §8: those
predict the NEXT TOKEN's experts and spend bandwidth on wrong guesses in a
bandwidth-bound regime). Here the prediction is for the NEXT LAYER of the SAME
token, made by applying layer l's router to layer l-1's output (Fate, arXiv
2502.12224), and its value exists only when compute is not negligible — exactly
the regime G-VALID-3 says real engines are in.

Inputs, all measured or swept, none tuned:
  - the committed OLMoE routing trace (who needs which expert, when)
  - per-layer recall of the next-layer prediction, MEASURED by traces_sparsity.py
    on the same model (g3_*.json `lookahead_recall`, keyed by the predicted layer)
  - bulk bandwidth (G1) and expert size
  - compute + DRAM time per token `c`, SWEPT (S11 is not measured yet), split
    evenly across layers

The replay (per token, per layer l, per-layer LRU cache, event-atomic, as the engine):
  1. While layer l-1 computes (c/L seconds), the engine fetches the predicted set
     for layer l minus what is cached. The predictor names k experts; a fraction
     r_l of layer l's true top-k is among them (the measured recall), and the other
     k*(1-r_l) names are WRONG — drawn uniformly from the experts layer l does not
     need. Wrong prefetches cost bandwidth AND cache slots (they are inserted and
     can evict useful experts), so the simulation charges both.
  2. Layer l then needs its true top-k: hits are free, remaining misses are demand
     reads, serial.
  3. Layer time = c/L + max(0, prefetch_time - c/L) + demand_time.
  Layer 0 has no predecessor in the token and is never prefetched.

Assumed, and stated: wrong guesses are uniform over the non-needed experts (a real
predictor errs toward plausible experts, which are more often already cached, so
this over-charges wrong guesses a little); recall comes from the fp16 model.

Run:
  python moe-phone/gates/prefetch_sim.py results/2026-09-14/traces_OLMoE-1B-7B-0924.npz \
      --g3 results/2026-09-14/g3_OLMoE-1B-7B-0924.json --bulk-gbps 2.806 --expert-mb 3.54
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_sim  # noqa: E402


def replay(ev, E, cap_total, recall, c_s, t_exp, prefetch, seed=0, max_tokens=None):
    """Returns dict with seconds/token, hit rate, experts read/token (demand + prefetch)."""
    rng = np.random.default_rng(seed)
    T, L, k = ev.shape
    if max_tokens:
        T = min(T, max_tokens)
    caps = cache_sim.split_capacity(cap_total, L)
    caches = [OrderedDict() for _ in range(L)]
    tc = c_s / L
    total_t = 0.0
    hits = demand = pref_reads = wasted = 0

    def insert(l, e):
        c = caches[l]
        if e in c:
            c.move_to_end(e)
            return False
        while len(c) >= caps[l]:
            c.popitem(last=False)
        c[e] = None
        return True

    for t in range(T):
        for l in range(L):
            true = ev[t, l].tolist()
            pf_time = 0.0
            if prefetch and l > 0 and caps[l] > 0:
                r = recall.get(l, 0.0)
                n_right = int(rng.binomial(k, r))
                right = rng.choice(true, size=n_right, replace=False).tolist() if n_right else []
                base = l * E
                others = [x for x in range(base, base + E) if x not in set(true)]
                wrong = rng.choice(others, size=k - n_right, replace=False).tolist()
                for e in right + wrong:
                    if insert(l, e):
                        pref_reads += 1
                        pf_time += t_exp
                        if e in wrong:
                            wasted += 1
            c = caches[l]
            miss = []
            for e in true:
                if caps[l] > 0 and e in c:
                    hits += 1
                    c.move_to_end(e)
                else:
                    miss.append(e)
            if caps[l] > 0:
                for e in miss:
                    insert(l, e)
            demand += len(miss)
            total_t += tc + max(0.0, pf_time - tc) + len(miss) * t_exp
    n = T * L * k
    return {"s_per_tok": total_t / T, "tok_s": T / total_t, "hit_rate": hits / n,
            "demand_reads_per_tok": demand / T, "prefetch_reads_per_tok": pref_reads / T,
            "wasted_prefetch_per_tok": wasted / T,
            "flash_reads_per_tok": (demand + pref_reads) / T}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("npz")
    p.add_argument("--g3", required=True, help="g3_<model>.json carrying lookahead_recall")
    p.add_argument("--bulk-gbps", type=float, required=True)
    p.add_argument("--expert-mb", type=float, required=True)
    p.add_argument("--fractions", default="0.10,0.20")
    p.add_argument("--fwd-ms", default="0,10,20,40,80,160")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()
    z = np.load(a.npz)
    traces = {int(kk[1:]): z[kk] for kk in z.files if kk.startswith("L")}
    E = int(z["num_experts"])
    ev, T, L, k = cache_sim.event_stream(traces, E, a.max_tokens)
    g3 = json.load(open(a.g3, encoding="utf-8"))
    recall = {int(l): float(v) for l, v in g3["lookahead_recall"].items()}
    t_exp = a.expert_mb * 1e6 / (a.bulk_gbps * 1e9)
    rows = []
    print(f"{T} tokens x {L} layers x top-{k} of {E}; expert read {t_exp * 1e3:.2f} ms; "
          f"recall by layer {min(recall.values()):.2f}-{max(recall.values()):.2f}")
    print(f"{'cache':>6} {'c ms':>5} {'tok/s no-pf':>11} {'tok/s pf':>9} {'speed-up':>9} "
          f"{'hit no-pf':>9} {'hit pf':>7} {'reads/tok no-pf':>15} {'pf':>6} {'wasted':>7}")
    for f in (float(x) for x in a.fractions.split(",")):
        cap = max(1, int(round(f * L * E)))
        for cms in (float(x) for x in a.fwd_ms.split(",")):
            base = replay(ev, E, cap, recall, cms / 1e3, t_exp, prefetch=False)
            pf = replay(ev, E, cap, recall, cms / 1e3, t_exp, prefetch=True)
            row = {"cache_fraction": f, "fwd_ms": cms, "no_prefetch": base, "prefetch": pf,
                   "speedup": pf["tok_s"] / base["tok_s"]}
            rows.append(row)
            print(f"{f:6.2f} {cms:5.0f} {base['tok_s']:11.2f} {pf['tok_s']:9.2f} {row['speedup']:9.3f} "
                  f"{base['hit_rate']:9.3f} {pf['hit_rate']:7.3f} {base['flash_reads_per_tok']:15.1f} "
                  f"{pf['flash_reads_per_tok']:6.1f} {pf['wasted_prefetch_per_tok']:7.1f}")
    out = {"source": os.path.abspath(a.npz), "g3": os.path.abspath(a.g3), "inputs": vars(a),
           "recall_by_layer": recall, "tokens": T, "rows": rows}
    name = "prefetch_sim_" + os.path.basename(a.npz).replace("traces_", "").replace(".npz", "") + ".json"
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        path = os.path.join(a.out_dir, name)
    else:
        from _paths import write_path
        path = write_path(name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
