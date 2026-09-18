"""
What would a LARGER expert cache be worth? A projection, built only from measured inputs, and labelled
as a projection so it is never mistaken for a row.

The engine has been capped at --cache-ceil-mb 5000 since a day when the phone had about 6 GB available.
After a reboot and a quiesce it reports ~7.6 GB. Whether raising the cap helps is two questions, and
only the first can be answered without a thermally clean phone:

  1. Does a bigger cache cut MISS TRAFFIC?  Hit rate and bytes-per-token are determined by the routing,
     the budget and the policy -- not by the clock. The simulator answers this from the committed trace
     (gates/evict_sim.py), and the engine's own counters confirm it on device at whatever clock.
  2. Does less miss traffic become SPEED?  That needs a clean rate measurement, and this script only
     PREDICTS it, using the measured decomposition at the current budget.

THE PROJECTION, and its one honest adjustment. The simulator's hit rate and the engine's disagree by a
fixed offset at the budget where both are known (simulated LRU at the current cache fraction vs the
engine's measured hit rate). That offset is applied to the simulated hit rate at every other budget
rather than pretending the simulator is exact; it is reported, so a reader can see how large a
correction is being carried. Stall is then scaled by the ratio of miss bytes, because the stall IS
unhidden miss traffic (gates/decode_budget.py shows it equals essentially all of it), while compute and
cache management are held fixed -- which makes the projection OPTIMISTIC in a stated way, since a larger
cache means more entries to manage and more memory pressure, both of which push the other two terms up.

Run:
  python moe-phone/gates/cache_projection.py --sim results/2026-09-18/evict_zram_budgets.json \\
      --engine results/2026-09-17/bmoe_cache.json --cell ceil5000 --current-fraction 0.320 \\
      --budget-mib 5000,7000,8500 --fractions 0.320,0.448,0.544 --out-name cache_projection.json
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def sim_row(sim, frac):
    for r in sim["rows"]:
        if abs(r["cache_fraction"] - frac) < 1e-9:
            return r
    raise ValueError(f"simulation has no cache_fraction {frac}: {[r['cache_fraction'] for r in sim['rows']]}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--sim", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--cell", required=True)
    p.add_argument("--current-fraction", type=float, required=True)
    p.add_argument("--budget-mib", required=True)
    p.add_argument("--fractions", required=True)
    p.add_argument("--policy", default="lru_hit", choices=["lru_hit", "slru80_hit"])
    p.add_argument("--out-name", required=True)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    sim = json.load(open(a.sim, encoding="utf-8"))
    eng = json.load(open(a.engine, encoding="utf-8"))
    cell = next(c for c in eng["cells"] if c["cell"] == a.cell)

    measured_hit = cell["cache_hit_pct_median"] / 100.0
    measured_read = cell["read_MiB_per_token_median"]
    decode_ms = 1000.0 / cell["decode_tok_s_median"]
    compute_ms = cell["compute_s_per_token_median"] * 1000.0
    mgmt = [r["cache_mgmt_s_per_token"] for r in eng["runs"]
            if r.get("cell") == a.cell and "cache_mgmt_s_per_token" in r]
    mgmt_ms = statistics.median(mgmt) * 1000.0
    stall_ms = decode_ms - compute_ms - mgmt_ms

    sim_here = sim_row(sim, a.current_fraction)[a.policy]
    offset = sim_here - measured_hit        # how optimistic the simulator is at the known point

    out = {"sim_source": os.path.abspath(a.sim), "engine_source": os.path.abspath(a.engine),
           "cell": a.cell, "policy": a.policy,
           "measured": {"hit": measured_hit, "read_MiB_per_token": measured_read,
                        "decode_ms": decode_ms, "compute_ms": compute_ms,
                        "cache_mgmt_ms": mgmt_ms, "stall_ms": stall_ms,
                        "decode_tok_s": cell["decode_tok_s_median"]},
           "simulator_offset_at_current_fraction": offset,
           "projection_is_optimistic_because": (
               "compute and cache management are held fixed, but a larger cache has more entries to "
               "manage and leaves less room for everything else, so both should rise"),
           "rows": []}

    budgets = [int(x) for x in a.budget_mib.split(",")]
    fracs = [float(x) for x in a.fractions.split(",")]
    if len(budgets) != len(fracs):
        raise ValueError("--budget-mib and --fractions must have the same length")

    for mib, frac in zip(budgets, fracs):
        sh = sim_row(sim, frac)[a.policy]
        adj = min(0.999, max(0.0, sh - offset))          # the simulator, corrected by the known offset
        miss_ratio = (1.0 - adj) / (1.0 - measured_hit)  # miss bytes relative to today's
        read = measured_read * miss_ratio
        stall = stall_ms * miss_ratio
        ms = compute_ms + mgmt_ms + stall
        out["rows"].append({"budget_MiB": mib, "cache_fraction": frac,
                            "simulated_hit": sh, "projected_measured_hit": adj,
                            "projected_read_MiB_per_token": read,
                            "projected_stall_ms": stall, "projected_decode_ms": ms,
                            "projected_decode_tok_s": 1000.0 / ms,
                            "projected_gain_vs_measured": (1000.0 / ms) / cell["decode_tok_s_median"] - 1.0})

    m = out["measured"]
    print(f"measured at {a.cell}: {m['decode_tok_s']:.3f} tok/s = {m['decode_ms']:.1f} ms "
          f"(compute {m['compute_ms']:.1f} + mgmt {m['cache_mgmt_ms']:.1f} + stall {m['stall_ms']:.1f}), "
          f"hit {m['hit']*100:.1f}%, read {m['read_MiB_per_token']:.1f} MiB/token")
    print(f"simulator is {offset*100:+.1f} points optimistic at the known budget; that offset is carried\n")
    for r in out["rows"]:
        print(f"  {r['budget_MiB']:5d} MiB -> hit {r['projected_measured_hit']*100:5.1f}% "
              f"read {r['projected_read_MiB_per_token']:6.1f} MiB/token  stall {r['projected_stall_ms']:5.1f} ms "
              f"-> PROJECTED {r['projected_decode_tok_s']:5.2f} tok/s ({r['projected_gain_vs_measured']*100:+5.1f}%)")
    print("\nPROJECTION, not a measurement: the speed column needs a thermally clean run to confirm.")

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
