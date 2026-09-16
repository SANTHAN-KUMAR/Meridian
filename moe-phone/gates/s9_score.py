"""
S9 SCORING — apply the pre-registered decision rule to the measured curves.

Reads the pre-registration written by gates/s9_prereg.py BEFORE the target trace
existed, and scores it. Nothing here chooses a grid, a tolerance or a rule: all
three come from the pre-registration file, and this script refuses to run if the
target curve does not cover the registered grid.

Inputs:
  --prereg           s9_prereg_<target>.json
  --target-curve     cache_fcrit_<target>.json  (the second expert count)
  --confound-curve   cache_fcrit_<reference model, same engine and format as the
                     target>.json — the pre-registered control. The reference
                     curve came from an fp16 HF run; if the same MODEL measured
                     through the target's engine and format differs from it by
                     more than half the minimum separation between the two
                     hypotheses, the test is INCONCLUSIVE whatever the target shows.

Verdicts (from the pre-registration's decision_rule):
  H_rho supported    every grid point within tolerance of H_rho
  H_floor supported  every grid point within tolerance of H_floor
  otherwise          neither validated; the lower-RMSE hypothesis is PREFERRED and
                     every per-model tok/s derived through at_rho() stays provisional

Grid points the pre-registration marks `degenerate` (the target's cache would hold
every expert, so h = 1 for any policy) are excluded from the verdict and reported
separately, with their observed values, so the exclusion is visible rather than
silent.

Run:
  python moe-phone/gates/s9_score.py --prereg results/<d>/s9_prereg_Qwen3-30B-A3B.json \
      --target-curve results/<d>/cache_fcrit_Qwen3-30B-A3B.json \
      --confound-curve results/<d>/cache_fcrit_OLMoE-1B-7B-0924-q4_0-llamacpp.json
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import read_path, write_path  # noqa: E402


def curve(path):
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    return ({round(r["rho_per_layer"], 6): r["lru_hit__per_layer_atomic"] for r in c["rows"]},
            c["num_experts"], c["top_k"], c["tokens"])


def at(c, rho, what):
    for k, v in c.items():
        if abs(k - rho) < 1e-6:
            return v
    raise KeyError(f"{what} has no point at rho={rho}; the pre-registered grid is "
                   f"{sorted(c)} — rerun cache_sim with --fractions-around-crit")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--prereg", required=True)
    p.add_argument("--target-curve")
    p.add_argument("--confound-curve")
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()
    with open(a.prereg, encoding="utf-8") as f:
        pre = json.load(f)
    tol = pre["tolerance"]
    grid = pre["grid_rho"]
    out = {"prereg": os.path.abspath(a.prereg), "tolerance": tol, "grid_rho": grid}

    # ---- the confound control, scored first: it can void the whole test ----------
    if a.confound_curve:
        ref_path = read_path(pre["reference"]["curve"])
        ref, E0, k0, _ = curve(ref_path)
        con, Ec, kc, _ = curve(a.confound_curve)
        if (Ec, kc) != (E0, k0):
            raise ValueError(f"confound control must be the SAME geometry as the reference "
                             f"({E0}/{k0}), got {Ec}/{kc}")
        rows = [{"rho": r, "reference_fp16": at(ref, r, "reference curve"),
                 "same_model_target_format": at(con, r, "confound curve"),
                 "abs_diff": abs(at(con, r, "confound curve") - at(ref, r, "reference curve"))}
                for r in grid]
        worst = max(r["abs_diff"] for r in rows)
        limit = pre["confound_inconclusive_if_exceeds"]
        out["confound"] = {"curve": os.path.abspath(a.confound_curve), "rows": rows,
                           "max_abs_diff": worst, "limit": limit, "voids_test": worst > limit}
        print(f"confound control: max |Q4/engine - fp16/HF| = {worst:.4f} over the grid, "
              f"limit {limit:.4f} -> {'VOIDS THE TEST' if worst > limit else 'test can proceed'}")
        for r in rows:
            print(f"  rho {r['rho']:4.2f}  fp16 {r['reference_fp16']:.3f}  "
                  f"same-model-target-format {r['same_model_target_format']:.3f}  "
                  f"diff {r['abs_diff']:+.4f}")

    # ---- the test itself ----------------------------------------------------------
    if a.target_curve:
        tgt, Et, kt, _ = curve(a.target_curve)
        if (Et, kt) != (pre["target"]["E"], pre["target"]["k"]):
            raise ValueError(f"target curve is {Et}/{kt}, pre-registration is for "
                             f"{pre['target']['E']}/{pre['target']['k']}")
        rows = []
        for pred in pre["predictions"]:
            obs = at(tgt, pred["rho"], "target curve")
            rows.append({"rho": pred["rho"], "observed": obs,
                         "H_rho": pred["pred_H_rho"], "H_floor": pred["pred_H_floor"],
                         "degenerate": bool(pred.get("degenerate")),
                         "err_H_rho": obs - pred["pred_H_rho"],
                         "err_H_floor": obs - pred["pred_H_floor"]})
        scored = [r for r in rows if not r["degenerate"]]
        if not scored:
            raise ValueError("every pre-registered grid point is degenerate for this target")
        rmse = {h: math.sqrt(sum(r[f"err_{h}"] ** 2 for r in scored) / len(scored))
                for h in ("H_rho", "H_floor")}
        supported = {h: all(abs(r[f"err_{h}"]) <= tol for r in scored)
                     for h in ("H_rho", "H_floor")}
        preferred = min(rmse, key=rmse.get)
        verdict = ("H_rho supported" if supported["H_rho"] and not supported["H_floor"] else
                   "H_floor supported" if supported["H_floor"] and not supported["H_rho"] else
                   "both within tolerance — the grid does not separate them here"
                   if supported["H_rho"] and supported["H_floor"] else
                   f"neither validated; {preferred} preferred (lower RMSE)")
        if pre.get("control"):
            # both hypotheses are the same prediction here; the question is family, not geometry
            verdict = ("CONTROL PASSES: no family effect at equal E/k (within tolerance)"
                       if supported["H_rho"] else
                       f"CONTROL FAILS: a family effect at equal E/k (RMSE {rmse['H_rho']:.4f} "
                       f"against tolerance {tol:.4f})")
        if out.get("confound", {}).get("voids_test"):
            verdict = "INCONCLUSIVE — the confound control exceeded its pre-registered limit"
        out["test"] = {"curve": os.path.abspath(a.target_curve), "rows": rows, "rmse": rmse,
                       "n_scored": len(scored), "n_degenerate_excluded": len(rows) - len(scored),
                       "supported": supported, "preferred": preferred, "verdict": verdict}
        print(f"\n{'rho':>5} {'observed':>9} {'H_rho':>7} {'H_floor':>8} {'err_rho':>8} {'err_flr':>8}")
        for r in rows:
            print(f"{r['rho']:5.2f} {r['observed']:9.3f} {r['H_rho']:7.3f} {r['H_floor']:8.3f} "
                  f"{r['err_H_rho']:+8.3f} {r['err_H_floor']:+8.3f}"
                  f"{'   [degenerate: excluded]' if r['degenerate'] else ''}")
        print(f"RMSE H_rho {rmse['H_rho']:.4f}, H_floor {rmse['H_floor']:.4f}, tolerance {tol:.4f}")
        print(f"VERDICT: {verdict}")
    elif not a.confound_curve:
        p.error("nothing to score: pass --target-curve and/or --confound-curve")

    name = f"s9_result_{pre['target']['name']}.json"
    path = os.path.join(a.out_dir, name) if a.out_dir else write_path(name)
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
