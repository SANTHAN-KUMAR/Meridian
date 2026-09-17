"""
Where a streamed token's milliseconds go, against the floor the hardware sets — and therefore exactly
what would have to change for the project's goal rate.

Everything here is read from committed artifacts; nothing is fitted and nothing is assumed:

  active bytes per token      gates/gguf_active.py       (the model's own tensor table)
  device weight throughput    gates/device_bandwidth.py  (a resident model's measured decode rate)
  the engine's own split      gates/bmoe_analyze.py      (compute residual + cache mgmt + flash I/O,
                                                          as BigMoeOnEdge prints it per run)

THE IDENTITY BEING USED, and its one subtlety. The engine reports

    decode_s_per_token = compute + cache_mgmt + stall

where `compute` is a RESIDUAL (wall minus stall minus mgmt; docs/telemetry.md), so it is not pure
arithmetic: it contains the time to move every one of the token's active bytes through the compute
device, whether those bytes came from the cache or from a flash read that already finished. That byte
movement has a hardware floor,

    floor_ms = active_bytes_per_token / device_weight_throughput

measured on the same device with a resident model, and the difference

    compute_overhead = compute - floor

is the part that is barriers, scheduling, dispatch and arithmetic inefficiency rather than bytes. The
floor is subtracted from `compute` and from nothing else, because `stall` is flash time the overlap
failed to hide and `cache_mgmt` is page-table work — neither moves weights through the ALUs.

WHAT IS REPORTED, per (configuration, device):
    floor_ms, compute_overhead_ms, cache_mgmt_ms, stall_ms, measured_ms, and each as a share
    efficiency            = floor_ms / measured_ms (what fraction of the token is irreducible)
    goal_ms               = 1000 / goal_tok_s
    overhead_budget_ms    = goal_ms - floor_ms  (negative = the goal is below the floor: impossible)
    overhead_now_ms       = compute_overhead + cache_mgmt + stall
    required_cut          = 1 - overhead_budget_ms / overhead_now_ms, the fraction of ALL non-floor
                            time that would have to disappear to reach the goal
    per-term sensitivity  d(tok/s)/d(term) at the measured point, so the terms can be ranked by what a
                          millisecond saved in each is worth

The floor is optimistic in a way that is stated rather than hidden: the device throughput it uses was
measured on a model with a smaller per-token working set, which gets more system-cache hits than the
target model can. A pessimistic floor makes `required_cut` smaller, so this direction is the
conservative one for a "still far" conclusion and the unsafe one for a "nearly there" conclusion.

Run:
  python moe-phone/gates/decode_budget.py --active results/2026-09-18/gguf_active_qwen_olmoe.json \\
      --bandwidth results/2026-09-18/device_bandwidth.json --engine results/2026-09-17/bmoe_cache.json \\
      --cell ceil5000 --target Qwen3-30B-A3B-Q4_0.gguf --goal-tok-s 10 --out-name decode_budget.json
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--active", required=True)
    p.add_argument("--bandwidth", required=True)
    p.add_argument("--engine", required=True, help="a bmoe_analyze.py artifact")
    p.add_argument("--cell", required=True, help="which cell of that artifact is the configuration")
    p.add_argument("--target", required=True, help="model file name in the active-bytes artifact")
    p.add_argument("--goal-tok-s", type=float, required=True)
    p.add_argument("--out-name", required=True)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    active = json.load(open(a.active, encoding="utf-8"))
    bw = json.load(open(a.bandwidth, encoding="utf-8"))
    eng = json.load(open(a.engine, encoding="utf-8"))

    bytes_tok = next(r["active_bytes_per_token"] for r in active
                     if os.path.basename(r["file"]) == a.target)
    cell = next(c for c in eng["cells"] if c["cell"] == a.cell)

    # The engine's own three terms, in ms. `compute` is the residual; see the docstring.
    compute_ms = cell["compute_s_per_token_median"] * 1000.0
    # The cache-mgmt term entered bmoe_analyze's cell summary later than these artifacts were written,
    # so fall back to the median over that cell's own runs rather than requiring a regenerated artifact.
    if "cache_mgmt_s_per_token_median" in cell:
        mgmt_ms = cell["cache_mgmt_s_per_token_median"] * 1000.0
    else:
        vals = [r["cache_mgmt_s_per_token"] for r in eng["runs"]
                if r.get("cell") == a.cell and "cache_mgmt_s_per_token" in r]
        mgmt_ms = (statistics.median(vals) * 1000.0) if vals else None
    decode_tok_s = cell["decode_tok_s_median"]
    measured_ms = 1000.0 / decode_tok_s
    if mgmt_ms is None:
        raise ValueError(f"cell {a.cell!r} has no cache-mgmt term: keys are {sorted(cell)}")
    # The stall is taken as the remainder of the identity rather than from the printed flash-I/O figure,
    # which is SUMMED over lanes and therefore larger than the wall time it cost (that is why the
    # printed compute + mgmt + flash does not add up to the decode time).
    stall_ms = measured_ms - compute_ms - mgmt_ms

    goal_ms = 1000.0 / a.goal_tok_s
    out = {"target": a.target, "cell": a.cell, "goal_tok_s": a.goal_tok_s,
           "active_bytes_per_token": bytes_tok, "measured_decode_tok_s": decode_tok_s,
           "measured_ms_per_token": measured_ms, "goal_ms_per_token": goal_ms,
           "engine_source": os.path.abspath(a.engine), "bandwidth_source": os.path.abspath(a.bandwidth),
           "terms_ms": {"compute_residual": compute_ms, "cache_mgmt": mgmt_ms, "stall": stall_ms},
           "devices": []}
    if stall_ms < 0:
        out["warning_identity"] = ("compute + mgmt exceeds the decode time: the identity does not hold "
                                   "for this cell and the split cannot be trusted")

    for d in bw["devices"]:
        if "effective_GB_s" not in d:
            continue
        floor_ms = bytes_tok / 1e9 / d["effective_GB_s"] * 1000.0
        overhead_ms = compute_ms - floor_ms
        overhead_now = overhead_ms + mgmt_ms + stall_ms
        budget = goal_ms - floor_ms
        e = {"device_arm": d["arm"], "effective_GB_s": d["effective_GB_s"], "floor_ms": floor_ms,
             "compute_overhead_ms": overhead_ms, "cache_mgmt_ms": mgmt_ms, "stall_ms": stall_ms,
             "efficiency_floor_over_measured": floor_ms / measured_ms,
             "overhead_now_ms": overhead_now, "overhead_budget_ms": budget,
             "goal_below_floor": budget < 0}
        if overhead_ms < 0:
            e["warning"] = ("the measured compute residual is BELOW this device's floor, so either the "
                            "throughput is overstated (resident-model confound) or some bytes never "
                            "reached the ALUs; the overhead split is not usable for this device")
        if budget > 0 and overhead_now > 0:
            e["required_cut_fraction"] = 1.0 - budget / overhead_now
        # what one millisecond saved in each term is worth, at the measured point
        e["tok_s_per_ms_saved"] = decode_tok_s / measured_ms  # identical for every term, by the identity
        e["terms_ranked_by_size_ms"] = sorted(
            [("stall", stall_ms), ("compute_overhead", overhead_ms), ("cache_mgmt", mgmt_ms)],
            key=lambda kv: -kv[1])
        out["devices"].append(e)

    print(f"{a.target} cell {a.cell}: measured {decode_tok_s:.3f} tok/s = {measured_ms:.1f} ms/token "
          f"(compute residual {compute_ms:.1f} + mgmt {mgmt_ms:.1f} + stall {stall_ms:.1f})")
    print(f"goal {a.goal_tok_s} tok/s = {goal_ms:.1f} ms/token")
    for e in out["devices"]:
        print(f"  vs {e['device_arm']:<12} ({e['effective_GB_s']:5.2f} GB/s): floor {e['floor_ms']:6.1f} ms "
              f"({e['efficiency_floor_over_measured']*100:4.1f}% of the token is irreducible), "
              f"non-floor now {e['overhead_now_ms']:6.1f} ms, budget {e['overhead_budget_ms']:6.1f} ms"
              + (f" -> {e['required_cut_fraction']*100:.0f}% of all non-floor time must go"
                 if "required_cut_fraction" in e else "  -> GOAL IS BELOW THE FLOOR"))
        if "warning" in e:
            print(f"     WARNING: {e['warning']}")

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
