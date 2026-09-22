"""
validation_report.py - predicted vs observed decode speed, from the app's own audit trail (no hand-typed numbers).

Every chat turn the app records writes one audit line with the plan's registered prediction ("predicted": value/lo/hi,
made BEFORE the model was ever run, from the compute probe and the GGUF header) and the observed decode rate. This script
groups those lines by model and reports, per model: the prediction, the observed turns, the median observed rate, the
point error of the prediction against that median, and how many observed turns fell inside the predicted range.
Only in-regime turns (screen on, unplugged, app in front) count; the others are listed as excluded.

  python3 validation_report.py AUDIT_JSONL OUT_JSON
"""
import json, statistics, sys


def main(src, out):
    by = {}
    excluded = 0
    for line in open(src):
        try:
            j = json.loads(line)
        except ValueError:
            continue
        p, o = j.get("predicted"), j.get("observed")
        if not p or not o or not str(j.get("turn_id", "")).startswith("chat."):
            continue
        if not j.get("in_regime"):
            excluded += 1
            continue
        by.setdefault(j["model_id"], {"pred": p, "obs": []})["obs"].append(o["tokens_per_s"])
    rows = []
    for m, d in sorted(by.items()):
        obs, p = d["obs"], d["pred"]
        med = statistics.median(obs)
        inside = sum(1 for v in obs if p["lo"] <= v <= p["hi"])
        rows.append({"model": m, "predicted": round(p["value"], 2), "predicted_lo": round(p["lo"], 2), "predicted_hi": round(p["hi"], 2),
                     "basis": p.get("provenance"), "observed": [round(v, 2) for v in obs], "observed_median": round(med, 2),
                     "point_error_frac": round((p["value"] - med) / med, 4), "inside_range": f"{inside}/{len(obs)}"})
    doc = {"source": src, "excluded_out_of_regime_turns": excluded, "rows": rows,
           "note": "predictions were registered in the plan before the first turn; one device, one session each"}
    json.dump(doc, open(out, "w"), indent=1)
    for r in rows:
        print(f"{r['model']:34s} predicted {r['predicted']:6.2f} [{r['predicted_lo']:.2f}-{r['predicted_hi']:.2f}] {r['basis']:10s} "
              f"observed median {r['observed_median']:6.2f} (n={len(r['observed'])})  error {100 * r['point_error_frac']:+.1f}%  inside {r['inside_range']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
