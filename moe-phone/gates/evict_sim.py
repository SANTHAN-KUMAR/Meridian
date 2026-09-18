"""
Eviction policy for the GLOBAL expert cache — can anything deployable beat LRU, and by how much?

The engine's cache is one shared budget keyed by (layer, expert), so the policy question is settled in
the global scope, not the per-layer one. `gates/expert_policy.py` already answered a different question
and answered it NEGATIVELY: per layer, on OLMoE, at a 10% cache, LRU beat every policy tried including a
static oracle that knew the whole trace's frequencies. This file does not contradict that; it asks the
remaining question — global scope, Qwen3-30B-A3B's geometry (128 experts, 48 layers, top-8), and the
cache fraction the phone actually runs at — and it restricts itself to policies that could be shipped.

Policies:
  lru        what the engine does today.
  belady     the offline optimum: the bound, not a candidate.
  cycle      evict from the layer whose next visit is furthest away. The layer order in a decode is
             fixed, so this needs no prediction. Included because LRU is textbook-pathological on a
             cyclic scan -- it evicts the layer touched longest ago, which is the layer about to be
             touched next.
  freq       evict the least popular entry (online frequency, Laplace-smoothed). Not deployable as
             written -- it scans candidates per eviction -- but it upper-bounds what popularity is worth.
  cyclefreq  both terms, to separate them.
  slru       segmented LRU: probation for entries seen once, protected for entries seen twice, with the
             protected tail demoting rather than dropping. O(1), two lists and a flag -- the same cost
             as the LRU the engine already runs. This is the deployable candidate.

`--protected-frac` is swept and every value reported, because a protected fraction chosen by looking at
the hit rate it produces would be a constant tuned against its own metric (CLAUDE.md §6.4). The
literature value for SLRU is 60-80%.

What the output means for the engine: a policy's value is the MISS traffic it removes, since misses are
what the stall is made of. The table therefore reports miss rate and its reduction against LRU, not just
hit rate.

Run:
  python moe-phone/gates/evict_sim.py results/2026-09-16/traces_Qwen3-30B-A3B-q4_0-llamacpp.npz \\
      --fractions 0.25,0.307,0.40 --tokens 4000 --out-name evict_policies_qwen3.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_sim as cs  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("npz")
    p.add_argument("--fractions", default="0.25,0.307,0.40")
    p.add_argument("--tokens", type=int, default=4000)
    p.add_argument("--protected-frac", default="0.6,0.8")
    p.add_argument("--slow-policies", action="store_true",
                   help="also run freq/cyclefreq, which scan candidates per eviction and are slow")
    p.add_argument("--out-name", required=True)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    z = np.load(a.npz)
    n_expert = int(z["num_experts"])
    traces = {int(k[1:]): z[k] for k in z.files if k.startswith("L")}
    ev, T, L, k = cs.event_stream(traces, n_expert, max_tokens=a.tokens)
    total_slots = L * n_expert
    n_req = T * L * k

    out = {"source": os.path.abspath(a.npz), "tokens": T, "layers": L, "top_k": k,
           "num_experts": n_expert, "total_slots": total_slots, "scope": "global",
           "replay": "atomic", "rows": []}

    for frac in [float(x) for x in a.fractions.split(",")]:
        cap = int(round(frac * total_slots))
        row = {"cache_fraction": frac, "slots": cap}
        t0 = time.time()
        row["lru_hit"] = cs.lru_hits_events(ev, cap, scope="global", replay="atomic") / n_req
        row["belady_hit"] = cs.belady_hits_scoped(ev, cap, scope="global") / n_req
        h, s = cs.cycle_hits_global(ev, cap, n_expert, policy="cycle")
        row["cycle_hit"] = h / s
        for pf in [float(x) for x in a.protected_frac.split(",")]:
            h, s = cs.slru_hits_global(ev, cap, protected_frac=pf)
            row[f"slru{int(pf * 100)}_hit"] = h / s
        if a.slow_policies:
            for pol in ("freq", "cyclefreq"):
                h, s = cs.cycle_hits_global(ev, cap, n_expert, policy=pol)
                row[f"{pol}_hit"] = h / s
        row["seconds"] = time.time() - t0
        # what the engine cares about: miss traffic removed, relative to LRU
        for key in [key for key in list(row) if key.endswith("_hit") and key != "lru_hit"]:
            name = key[:-4]
            row[f"{name}_miss_reduction_vs_lru"] = 1.0 - (1.0 - row[key]) / (1.0 - row["lru_hit"])
            row[f"{name}_gap_closed"] = ((row[key] - row["lru_hit"]) /
                                         (row["belady_hit"] - row["lru_hit"])
                                         if row["belady_hit"] > row["lru_hit"] else None)
        out["rows"].append(row)

    for r in out["rows"]:
        print(f"cache {r['cache_fraction']:.3f} ({r['slots']} slots): LRU {r['lru_hit']*100:5.2f}  "
              f"Belady {r['belady_hit']*100:5.2f}")
        for key in sorted(kk for kk in r if kk.endswith("_hit") and kk not in ("lru_hit", "belady_hit")):
            name = key[:-4]
            print(f"    {name:<10} {r[key]*100:5.2f}  miss traffic {r[name+'_miss_reduction_vs_lru']*100:+5.1f}%"
                  f"  gap closed {r[name+'_gap_closed']*100:4.0f}%")

    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        path = os.path.join(a.out_dir, a.out_name)
    else:
        from _paths import write_path
        path = write_path(a.out_name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
