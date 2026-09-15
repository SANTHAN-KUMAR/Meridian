"""
S9 PRE-REGISTRATION — the two predictions for a second expert count, written
down (and committed) BEFORE the second model's routing trace exists.

The assumption under test (POSITION.md §11, S9): engine_target.at_rho() reads the
LRU hit rate off OLMoE's curve at equal rho = per-layer slots / top_k and applies
it to every model. Two hypotheses make different predictions for a model with a
different E/k:

  H_rho    h_target(rho) = h_OLMoE(rho)
           the current assumption: the hit rate depends on geometry ONLY through rho.

  H_floor  h_target(rho) = f_target(rho) + [h_OLMoE(rho) - f_OLMoE(rho)]
           with f = rho * k / E, the cache fraction. Motivation: on locality-free
           routing LRU returns exactly f (asserted in tests/test_gates.py), so f is
           LRU's null, and it depends on E/k, not on rho alone. H_floor transfers
           only OLMoE's EXCESS over its own null.

For the same E/k both hypotheses coincide, so a model with OLMoE's E/k (e.g.
gpt-oss-20b, 32/4) is a control, not a test.

Decision rule, fixed here and not after seeing the trace:
  - Evaluate both at the pre-registered rho grid of cache_sim --fractions-around-crit
    (rho = multiples of 1: 0.5 ... 4, identical for every model because the grid
    is defined as multiples of f_crit = k/E), restricted to rho >= 1, where a layer
    can hold a whole token's working set.
  - Noise scale: OLMoE's own split-half disagreement (first vs second half of
    the committed trace), max over the grid. The tolerance is 2x that. It comes
    from the committed OLMoE trace alone, so it cannot be tuned against the
    target (CLAUDE.md §6.4).
  - "H_rho supported"   : |h_target - H_rho| <= tol at every grid point.
    "H_floor supported" : |h_target - H_floor| <= tol at every grid point.
    Otherwise the one with smaller RMS error is PREFERRED, and neither is
    validated: each model needs its own trace.
  - Engine/format confound: the target trace is collected through llama.cpp at
    Q4_0 while OLMoE's curve came from HF fp16. A second OLMoE trace collected
    through the SAME llama.cpp Q4_0 path measures that confound directly. If its
    max |difference| from the fp16 curve exceeds half the minimum separation
    between the two hypotheses on the grid, the S9 test is declared
    INCONCLUSIVE, whatever the target shows.

Run (writes s9_prereg_<target>.json; run it BEFORE collecting the target trace):
  python moe-phone/gates/s9_prereg.py --target-E 128 --target-k 8 \
      --target-name Qwen3-30B-A3B
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_sim  # noqa: E402
from _paths import read_path, write_path  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def curve(art):
    """rho -> LRU (per-layer, atomic) from a cache_sim --fractions-around-crit artifact."""
    return {round(r["rho_per_layer"], 6): r["lru_hit__per_layer_atomic"] for r in art["rows"]}


def split_half(npz, E, rhos, k):
    z = np.load(npz)
    tr = {int(kk[1:]): z[kk] for kk in z.files if kk.startswith("L")}
    T = min(t.shape[0] for t in tr.values())
    halves = [{l: t[:T // 2] for l, t in tr.items()}, {l: t[T // 2:T] for l, t in tr.items()}]
    out = []
    for rho in rhos:
        f = rho * k / E
        hs = []
        for h in halves:
            ev, Th, L, kk = cache_sim.event_stream(h, E)
            cap = max(1, int(round(f * L * E)))
            hs.append(cache_sim.lru_hits_events(ev, cap, "per_layer", "atomic") / (Th * L * kk))
        out.append({"rho": rho, "first_half": hs[0], "second_half": hs[1],
                    "abs_diff": abs(hs[0] - hs[1])})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--target-E", type=int, required=True)
    p.add_argument("--target-k", type=int, required=True)
    p.add_argument("--target-name", required=True)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    src_curve = read_path("cache_fcrit_OLMoE-1B-7B-0924.json")
    src_trace = read_path("traces_OLMoE-1B-7B-0924.npz")
    art = json.load(open(src_curve, encoding="utf-8"))
    E0, k0 = art["num_experts"], art["top_k"]
    c = curve(art)
    grid = sorted(r for r in c if r >= 1.0 - 1e-9)
    noise = split_half(src_trace, E0, grid, k0)
    tol = 2.0 * max(n["abs_diff"] for n in noise)

    rows = []
    for rho in grid:
        h0 = c[rho]
        f0 = rho * k0 / E0
        ft = rho * a.target_k / a.target_E
        rows.append({"rho": rho, "h_olmoe": h0, "f_olmoe": f0, "f_target": ft,
                     "pred_H_rho": h0, "pred_H_floor": ft + (h0 - f0),
                     "separation": abs(h0 - (ft + (h0 - f0)))})
    out = {
        "registered_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "target": {"name": a.target_name, "E": a.target_E, "k": a.target_k},
        "reference": {"E": E0, "k": k0, "curve": os.path.basename(src_curve),
                      "curve_sha256": sha256(src_curve),
                      "trace": os.path.basename(src_trace), "trace_sha256": sha256(src_trace)},
        "grid_rho": grid,
        "noise_split_half": noise,
        "tolerance": tol,
        "min_separation": min(r["separation"] for r in rows),
        "confound_inconclusive_if_exceeds": min(r["separation"] for r in rows) / 2.0,
        "predictions": rows,
        "decision_rule": "see module docstring of gates/s9_prereg.py",
    }
    print(f"S9 pre-registration for {a.target_name} (E={a.target_E}, k={a.target_k}); "
          f"reference OLMoE E={E0} k={k0}")
    print(f"tolerance (2x split-half) = {tol:.4f}; confound limit = "
          f"{out['confound_inconclusive_if_exceeds']:.4f}")
    print(f"{'rho':>5} {'H_rho':>7} {'H_floor':>8} {'sep':>7}")
    for r in rows:
        print(f"{r['rho']:5.2f} {r['pred_H_rho']:7.3f} {r['pred_H_floor']:8.3f} {r['separation']:7.3f}")
    name = f"s9_prereg_{a.target_name}.json"
    path = os.path.join(a.out_dir, name) if a.out_dir else write_path(name)
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
