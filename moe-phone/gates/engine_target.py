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



def load_lru_curve(path):
    """Read the MEASURED LRU/Belady curve from a cache_sim artifact.

    CLAUDE.md §7.1: these numbers are read from a generated artifact, never
    typed. The curve is indexed by rho = per-layer slots / top_k, not by cache
    fraction, because rho is the dimensionless quantity: a 10% cache means
    something different on a top-8-of-64 model than on a top-10-of-512 one.
    """
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    scope = "per_layer"
    lru_key = f"lru_hit__{scope}_atomic"
    pts = []
    for r in c["rows"]:
        if lru_key in r:
            pts.append((r["rho_per_layer"], r[lru_key],
                        r.get(f"belady_hit__{scope}")))
    pts.sort()
    return {"path": os.path.abspath(path), "source_trace": c.get("source"),
            "num_experts": c["num_experts"], "top_k": c["top_k"],
            "tokens": c["tokens"], "points": pts}


def at_rho(curve, rho):
    """Linear interpolation of the measured curve in rho. Returns (lru, belady,
    extrapolated_flag). Outside the measured range the endpoint is clamped and
    the flag is set -- an extrapolated value is NOT a measurement."""
    pts = curve["points"]
    if not pts:
        return None, None, True
    if rho <= pts[0][0]:
        return pts[0][1], pts[0][2], rho < pts[0][0]
    if rho >= pts[-1][0]:
        return pts[-1][1], pts[-1][2], rho > pts[-1][0]
    for (r0, l0, b0), (r1, l1, b1) in zip(pts, pts[1:]):
        if r0 <= rho <= r1:
            w = 0.0 if r1 == r0 else (rho - r0) / (r1 - r0)
            bb = None if (b0 is None or b1 is None) else b0 + w * (b1 - b0)
            return l0 + w * (l1 - l0), bb, False
    return pts[-1][1], pts[-1][2], True


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
    p.add_argument("--dram-gbps", type=float, required=True,
                   help="MEASURED DRAM read bandwidth, GB/s (S1: dram_15r.json). Every resident "
                        "weight and every cache HIT is read at this rate; the flash-only formula "
                        "leaves it out, so rows report both.")
    p.add_argument("--target-tps", type=float, required=True,
                   help="decode rate to clear; ESTIMAND.md §4 derives it from reading speed")
    p.add_argument("--bpw", type=float, default=4.5, help="bits per weight (Q4_0 = 4.5)")
    p.add_argument("--cache-json", default=None,
                   help="a cache_sim artifact giving the MEASURED LRU curve "
                        "(default: newest cache_fcrit_OLMoE-1B-7B-0924.json)")
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    src = a.byte_budget_json or read_path("byte_budget_measured.json")
    with open(src, encoding="utf-8") as f:
        bb = json.load(f)
    curve = load_lru_curve(a.cache_json or read_path("cache_fcrit_OLMoE-1B-7B-0924.json"))

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
        rho = h_floor * g["num_experts"] / g["top_k"] if fits else 0.0
        fully_resident = fits and cache_b >= e_all_b
        h_lru, h_bel, extrap = at_rho(curve, rho)
        h_lru_floor_add = None
        if fully_resident:
            # Every expert fits: the hit rate is 1 by construction and there is no
            # flash traffic. Reading a cache curve here would credit a streaming
            # engine for a model that is never streamed (2026-09-16 defect: OLMoE
            # was reported at 27.1 tok/s "on plain LRU").
            h_lru, h_bel = 1.0, 1.0
        elif extrap:
            # Outside the measured rho range the curve was CLAMPED to an endpoint.
            # A clamped value is not a measurement, so no tok/s is derived from it
            # (2026-09-16 defect: the flag existed but only one verdict branch
            # printed it, so clamped rows were reported unmarked).
            h_lru, h_bel = None, None
        else:
            # S9's competing hypothesis (gates/s9_prereg.py): transfer only
            # OLMoE's excess over its own LRU null f = rho * k / E.
            f0 = rho * curve["top_k"] / curve["num_experts"]
            h_lru_floor_add = h_floor + (h_lru - f0)

        def rate(h):
            """(flash-only, serial flash+DRAM, overlapped max(flash, DRAM)) tok/s at hit rate h."""
            if not fits or h is None:
                return None, None, None
            t_f = e_tok_b * (1 - h) / (a.bulk_gbps * 1e9)
            t_d = (res_b + e_tok_b * h) / (a.dram_gbps * 1e9)
            return ((1 / t_f) if t_f > 0 else None, 1 / (t_f + t_d), 1 / max(t_f, t_d))

        fo_lru, se_lru, ov_lru = rate(h_lru)
        _, se_fa, _ = rate(h_lru_floor_add)
        rows.append({
            "rho": rho, "h_lru_measured": h_lru, "h_belady_measured": h_bel,
            "curve_extrapolated": extrap,
            # The CONSTRUCTIVE number: what the measured policy actually delivers.
            # tok/s = bandwidth / (E_tok_bytes * (1 - h)).  target_tps only decides
            # the verdict string; it does not enter this.
            "tok_s_at_floor": (a.bulk_gbps * 1e9 / (e_tok_b * (1 - h_floor))
                               if fits and h_floor < 1 else None),
            "tok_s_at_lru": fo_lru,
            "tok_s_at_belady": (a.bulk_gbps * 1e9 / (e_tok_b * (1 - h_bel))
                                if fits and h_bel is not None and h_bel < 1 else None),
            # The flash-only figure omits DRAM reads of resident weights and cache
            # hits. The serial figure charges both with no overlap; the overlap
            # figure assumes perfect overlap. Compute is still NOT charged (S11).
            "tok_s_at_lru_serial_dram": se_lru,
            "tok_s_at_lru_overlap_dram": ov_lru,
            "fully_resident": fully_resident,
            "h_lru_floor_additive": h_lru_floor_add,
            "tok_s_at_lru_floor_additive_serial_dram": se_fa,
            "clears_with_plain_lru": bool(fits and not extrap and h_lru is not None and need <= h_lru),
            "unreachable_even_with_belady": bool(
                fits and h_bel is not None and need > h_bel),
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

    rows.sort(key=lambda r: (not r["resident_fits"], -(r["tok_s_at_lru_serial_dram"] or -1)))
    print(f"target {a.target_tps} tok/s | bulk {a.bulk_gbps} GB/s | DRAM {a.dram_gbps} GB/s | "
          f"budget {a.ram_gb} GB | {a.bpw} bpw   (compute NOT charged: S11)")
    print(f"\nmeasured LRU curve: {os.path.basename(curve['path'])} "
          f"(top-{curve['top_k']} of {curve['num_experts']}, {curve['tokens']} tokens), "
          f"indexed by rho = per-layer slots / top_k")
    print(f"\n{'model':<30}{'total':>7}{'GB/tok':>8}{'cache%':>8}{'rho':>7}"
          f"{'h_LRU':>7}{'flash':>7}{'serial':>8}{'H_floor':>8}  verdict")
    nan = float("nan")
    for r in rows:
        t = r["tok_s_at_lru_serial_dram"]
        if not r["resident_fits"]:
            v = "RESIDENT WEIGHTS ALONE EXCEED THE BUDGET"
        elif r["fully_resident"]:
            v = "FULLY RESIDENT: no flash traffic; DRAM/compute set the rate"
        elif r["curve_extrapolated"]:
            v = "rho OUTSIDE the measured curve: no number (needs its own trace)"
        elif r["unreachable_by_any_policy"]:
            v = "unreachable: needs h >= 1"
        elif r["clears_with_no_policy"]:
            v = "CLEARS on the no-locality floor alone"
        elif r["clears_with_plain_lru"]:
            v = f"CLEARS with plain LRU (measured {r['h_lru_measured']:.3f})"
        elif r["unreachable_even_with_belady"] and r["h_belady_measured"] is not None:
            v = f"beyond Belady ({r['h_belady_measured']:.3f}): drop or re-quantise"
        else:
            v = (f"gap {r['h_required'] - r['h_lru_measured']:+.3f} over LRU"
                 f"{' [rho extrapolated]' if r['curve_extrapolated'] else ''}")
        print(f"{r['repo'][:29]:<30}{r['total_params']/1e9:>6.0f}B"
              f"{r['E_tok_GB']:>8.2f}{100*r['cache_fraction_of_experts']:>7.1f}%"
              f"{r['rho']:>7.2f}"
              f"{(r['h_lru_measured'] if r['h_lru_measured'] is not None else nan):>7.3f}"
              f"{(r['tok_s_at_lru'] or nan):>7.1f}"
              f"{(r['tok_s_at_lru_serial_dram'] or nan):>8.1f}"
              f"{(r['tok_s_at_lru_floor_additive_serial_dram'] or nan):>8.1f}  {v}")
    print("\nflash = flash-only (ARCHITECTURE §1); serial = flash + DRAM, no overlap; H_floor = "
          "serial under S9's competing hypothesis. All three ignore compute (S11).")

    n_bulk = sum(1 for r in rows if r["every_read_is_bulk"])
    print(f"\nevery read is bulk for {n_bulk}/{len(rows)} models "
          f"(smallest expert {min(r['per_expert_MB'] for r in rows):.2f} MB, "
          f"bulk threshold {BULK_THRESHOLD_MB} MB) -- so the 8.9x small-read penalty")
    print("does not apply to whole-expert reads, and raising h is the ONLY lever.")
    n_ex = sum(1 for r in rows if r["resident_fits"] and r["curve_extrapolated"])
    print(f"\nTRANSFER ASSUMPTION: h_LRU is read off a curve measured on "
          f"{curve['num_experts']}-expert top-{curve['top_k']} routing, applied at equal rho.")
    print("That the hit rate depends on geometry only through rho is an ASSUMPTION, not a")
    print(f"measurement -- it is untested across expert counts, and {n_ex}/{len(rows)} models")
    print("sit outside the measured rho range entirely. Registered as stub S9 in POSITION.md.")

    out = {"inputs": vars(a), "source": os.path.abspath(src),
           "lru_curve": {kk: vv for kk, vv in curve.items() if kk != "points"},
           "lru_curve_points": curve["points"],
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
