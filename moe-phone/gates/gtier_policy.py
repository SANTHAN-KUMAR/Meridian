"""
GPU expert tier policy on the LAPTOP (patch 0019, CPU stand-in; design §13): per run, text md5 and the tier's
counters. Rows (OLMoE, 96 tokens, stack config, 500 MiB tier unless noted):
  off | k3 (cap 3, no promotion budget) | k3h1 (800 MiB, promote after 1 hit) | k1 (cap 1: heavy overflow through
  read-mapped slots) | k8 (no effective cap) | k3b2 (cap 3, <= 2 promotions/token, the default) | k3b1 (<= 1/token)
Correctness: every row's text must equal `off`. Policy: experts per dispatch vs the cap; promotions (churn).
Run:
  python moe-phone/gates/gtier_policy.py results/2026-09-18/gtier_policy_laptop --out results/2026-09-18/gtier_policy.json
"""
import argparse, glob, hashlib, json, os, re
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out", required=True); a = ap.parse_args()
rows = {}
for f in sorted(glob.glob(os.path.join(a.root, "*.out"))):
    tag = os.path.basename(f)[:-4]
    t = open(f).read(); text = t[:t.index("\ngeneration:")] if "\ngeneration:" in t else t
    err = open(f[:-4] + ".err").read()
    r = dict(text_md5=hashlib.md5(text.encode()).hexdigest(), fatal="FATAL" in err)
    m = re.search(r"promotions (\d+), evictions (\d+), .*?dispatches (\d+), experts on device (\d+) \(([0-9.]+)/dispatch\).*?"
                  r"overflow experts CPU-computed from the tier (\d+)(?:, promotions denied by the per-token budget (\d+))?", err)
    if m:
        r.update(promotions=int(m.group(1)), tier_evictions=int(m.group(2)), dispatches=int(m.group(3)),
                 experts_on_device=int(m.group(4)), experts_per_dispatch=float(m.group(5)), overflow=int(m.group(6)),
                 budget_denied=int(m.group(7)) if m.group(7) else None)
    rows[tag] = r
ref = rows["off"]["text_md5"]
out = dict(source=os.path.abspath(a.root), rows=rows, n_runs=len(rows),
           all_text_identical=all(r["text_md5"] == ref for r in rows.values()),
           any_fatal=any(r["fatal"] for r in rows.values()),
           overflow_total=sum(r.get("overflow", 0) for r in rows.values()))
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "rows" and k != "source"}, indent=1))
for k, r in rows.items(): print(k, {kk: vv for kk, vv in r.items() if kk not in ("text_md5",)})
