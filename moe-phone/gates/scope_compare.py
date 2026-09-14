"""
SCOPE COMPARE — per-layer quotas against one shared pool, at equal total bytes.

This produces the table in ARCHITECTURE.md §2.1, which decides the single most
consequential structural choice in the engine. It exists as its own script
because the comparison is easy to get wrong in exactly the way this project
already got it wrong once: `cache_sim.py` originally simulated a shared pool
while `expert_policy.py`'s policy was per-layer, and the two were reported side
by side as if they were the same machine (POSITION.md §3c).

`split_capacity()` preserves the total slot budget exactly, so a given cache
fraction means the SAME BYTES under both scopes. Without that the comparison
would be between a cache and a bigger cache.

The horizon means different things under the two scopes, and that is the point
rather than an inconsistency:

  per_layer  a horizon of H is H TOKENS of future routing -- what a multi-token
             verification pass supplies over its window.
  global     a horizon of H is H EVENTS (layer-steps). A within-token
             cross-layer predictor supplies a few of these with no drafter at
             all, which is the only reason the shared pool is worth testing:
             if it could reach its (higher) optimum from cheap lookahead it
             would win.

Run:
  python moe-phone/gates/scope_compare.py results/<date>/traces_OLMoE-1B-7B-0924.npz \\
      --max-tokens 8000
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_sim  # noqa: E402

HORIZONS = [1, 2, 4, 8, 16]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("npz")
    p.add_argument("--fractions", default="0.05,0.10,0.20,0.30")
    p.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    p.add_argument("--max-tokens", type=int, default=8000,
                   help="sub-sample. The shared-pool lookahead scans a cache of cap_total "
                        "entries on every eviction, so the full trace is expensive; the "
                        "sub-sample size is recorded in the artifact and in CLAIMS.md so no "
                        "number from here is mistaken for a full-trace figure.")
    p.add_argument("--out-name", default=None)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    z = np.load(a.npz)
    traces = {int(k[1:]): z[k] for k in z.files if k.startswith("L")}
    E = int(z["num_experts"])
    ev, T, L, k = cache_sim.event_stream(traces, E, a.max_tokens)
    n = T * L * k
    fr = [float(x) for x in a.fractions.split(",")]
    hs = [int(x) for x in a.horizons.split(",")]

    print(f"{T} tokens x {L} layers x top-{k} of {E}"
          + (f"   (sub-sampled from {min(t.shape[0] for t in traces.values())})"
             if a.max_tokens else ""))
    print("equal TOTAL slot budget under both scopes; horizon is TOKENS per-layer, "
          "EVENTS global\n")
    rows = []
    for f in fr:
        cap = max(1, int(round(f * L * E)))
        r = {"cache_fraction": f, "cache_experts": cap, "per_layer": {}, "global": {}}
        for scope in ("per_layer", "global"):
            d = r[scope]
            d["lru"] = cache_sim.lru_hits_events(ev, cap, scope, "atomic") / n
            d["belady"] = cache_sim.belady_hits_scoped(ev, cap, scope) / n
            d["by_horizon"] = {str(h): cache_sim.lookahead_hits(ev, cap, h, scope=scope) / n
                               for h in hs}
        rows.append(r)
        print(f"cache {f:.0%}  ({cap} experts)")
        print(f"  {'scope':<10}{'LRU':>8}" + "".join(f"{'H=' + str(h):>8}" for h in hs)
              + f"{'Belady':>9}")
        for scope in ("per_layer", "global"):
            d = r[scope]
            print(f"  {scope:<10}{d['lru']:8.3f}"
                  + "".join(f"{d['by_horizon'][str(h)]:8.3f}" for h in hs)
                  + f"{d['belady']:9.3f}")
        pl, gl = r["per_layer"], r["global"]
        verdict = ("per-layer" if max(pl["by_horizon"].values()) >= max(gl["by_horizon"].values())
                   else "GLOBAL")
        print(f"  -> best reachable online: {verdict}"
              f"   (global's offline optimum is {gl['belady']:.3f} vs {pl['belady']:.3f},"
              f" so the pool has more headroom it cannot reach)\n")

    out = {"source": os.path.abspath(a.npz), "inputs": vars(a),
           "tokens_simulated": T, "layers": L, "top_k": k, "num_experts": E,
           "horizons": hs, "rows": rows}
    name = a.out_name or ("scope_compare_" + os.path.basename(a.npz)
                          .replace("traces_", "").replace(".npz", ""))
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        path = os.path.join(a.out_dir, name + ".json")
    else:
        from _paths import write_path
        path = write_path(name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
