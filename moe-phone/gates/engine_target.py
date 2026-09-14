"""
ENGINE TARGET — what the engine must achieve, stated as one number per model.

This is the constructive counterpart to G-ROOF. G-ROOF answers "what is the best
any engine could do"; this answers "what must OUR engine do to clear the bar",
so the build has a target rather than a bound.

The reduction that makes it one number:

  G3 showed gate-first neuron sparsity fails the pre-registered fidelity margin
  on a stock model, so the engine reads WHOLE experts. And every expert in every
  candidate model is 1.77-99 MB (see the table this prints), comfortably above
  the 512 KB threshold at which G1 measured full bulk bandwidth. So every read
  the engine issues is already a bulk read, the 8.9x small-read penalty does not
  apply, and the co-activation layout work (G4) is NOT on the critical path --
  it only ever mattered for neuron-granularity reads.

  What remains is therefore:

      seconds / token  =  E_tok_bytes * (1 - h) / bulk_bandwidth

  and the only free variable is h, the expert-cache hit rate. Everything the
  engine does -- pinning, prefetching, eviction policy, lookahead -- exists to
  raise h. So:

      h_required = 1 - (bulk_bandwidth / target_tok_s) / E_tok_bytes

  Compare h_required against three measured reference points from G2:
    h_floor         what a cache with NO locality gets (cache_bytes / E_all)
    static-warmup   deployable, no training: pin the popular experts per layer
    Belady          the offline optimum; an upper bound for ANY policy

  A model whose h_required is below h_floor needs no policy at all. One whose
  h_required is above Belady at the available cache fraction is not reachable
  by any eviction policy and must be dropped or re-quantized.

Inputs are the MEASURED 15R values by default; pass your own to explore.

Run:
  python moe-phone/gates/engine_target.py --bulk-gbps 2.806 --ram-gb 4.85 \
      --target-tps 5
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import read_path, write_path  # noqa: E402

BULK_THRESHOLD_MB = 0.5  # G1: reads at/above this run at bulk bandwidth


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--byte-budget-json", default=None,
                   help="a byte_budget artifact to read model geometry from "
                        "(default: newest byte_budget_measured.json)")
    p.add_argument("--bulk-gbps", type=float, required=True,
                   help="MEASURED bulk read bandwidth, GB/s (G1)")
    p.add_argument("--ram-gb", type=float, required=True,
                   help="MEASURED weight budget, GB (G0 memprobe --file). Note this is "
                        "bimodal on a real phone; name which regime you mean.")
    p.add_argument("--target-tps", type=float, required=True,
                   help="decode rate to clear; ESTIMAND.md §4 derives it from reading speed")
    p.add_argument("--bpw", type=float, default=4.5, help="bits per weight (Q4_0 = 4.5)")
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    src = a.byte_budget_json or read_path("byte_budget_measured.json")
    with open(src, encoding="utf-8") as f:
        bb = json.load(f)

    rows = []
    for m in bb["models"]:
        g = m["geometry"]
        e_tok_b = g["E_tok_params"] * a.bpw / 8
        e_all_b = g["E_all_params"] * a.bpw / 8
        res_b = m["resident_params"] * a.bpw / 8
        cache_b = a.ram_gb * 1e9 - res_b
        fits = cache_b > 0
        h_floor = max(0.0, min(1.0, cache_b / e_all_b)) if fits else 0.0
        per_expert_mb = e_all_b / (g["moe_layers"] * g["num_experts"]) / 1e6
        # h such that E_tok_bytes*(1-h)/bulk == 1/target_tps
        need = 1.0 - (a.bulk_gbps * 1e9 / a.target_tps) / e_tok_b
        rows.append({
            "repo": m["repo"], "total_params": m["total_params"],
            "E_tok_GB": e_tok_b / 1e9, "resident_GB": res_b / 1e9,
            "cache_GB": cache_b / 1e9 if fits else 0.0,
            "resident_fits": fits,
            "cache_fraction_of_experts": h_floor,
            "per_expert_MB": per_expert_mb,
            "every_read_is_bulk": per_expert_mb >= BULK_THRESHOLD_MB,
            "h_required": need,
            "h_floor": h_floor,
            "clears_with_no_policy": fits and need <= h_floor,
            "unreachable_by_any_policy": need >= 1.0,
        })

    rows.sort(key=lambda r: (not r["resident_fits"], r["h_required"]))
    print(f"target {a.target_tps} tok/s | bulk {a.bulk_gbps} GB/s | budget {a.ram_gb} GB | "
          f"{a.bpw} bpw")
    print(f"\n{'model':<34}{'total':>7}{'MB/exp':>8}{'GB/tok':>8}{'cache%':>8}"
          f"{'h_floor':>9}{'h_needed':>10}  verdict")
    for r in rows:
        if not r["resident_fits"]:
            v = "RESIDENT WEIGHTS ALONE EXCEED THE BUDGET"
        elif r["unreachable_by_any_policy"]:
            v = "unreachable: needs h >= 1"
        elif r["clears_with_no_policy"]:
            v = "CLEARS WITH NO POLICY (floor is enough)"
        else:
            v = f"needs a policy: +{r['h_required'] - r['h_floor']:.3f} over floor"
        print(f"{r['repo']:<34}{r['total_params']/1e9:>6.0f}B{r['per_expert_MB']:>8.1f}"
              f"{r['E_tok_GB']:>8.2f}{100*r['cache_fraction_of_experts']:>7.1f}%"
              f"{r['h_floor']:>9.3f}{r['h_required']:>10.3f}  {v}")

    n_bulk = sum(1 for r in rows if r["every_read_is_bulk"])
    print(f"\nevery read is bulk for {n_bulk}/{len(rows)} models "
          f"(smallest expert {min(r['per_expert_MB'] for r in rows):.2f} MB, "
          f"bulk threshold {BULK_THRESHOLD_MB} MB) -- so the 8.9x small-read penalty")
    print("does not apply to whole-expert reads, and raising h is the ONLY lever.")
    print("\nG2 reference points (OLMoE-1B-7B, measured): Belady 0.323 @5% cache, 0.501 @10%;")
    print("deployable static-pin 0.094 @5%, 0.171 @10%; lookahead recall 0.88.")

    out = {"inputs": vars(a), "source": os.path.abspath(src),
           "bulk_threshold_MB": BULK_THRESHOLD_MB, "models": rows}
    name = "engine_target.json"
    path = os.path.join(a.out_dir, name) if a.out_dir else write_path(name)
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
