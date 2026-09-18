"""
Which expert-cache budgets can this phone actually GRANT? From the engine's own sizing line in every run's
stderr (`bmoe: cache auto — N MiB available, leaving F MiB free → B MiB budget (cap C MiB)`), which the
engine prints AFTER the dense weights and compute buffers exist. The largest budget auto sizing could give
is (available - floor); a projection at a budget above that is a statement about a phone with more free
memory than this one has had on any logged run.
Run:
  python moe-phone/gates/budget_reach.py results --out results/2026-09-18/budget_reach.json
"""
import argparse, json, os, re, statistics as S
PAT = re.compile(r"cache auto — (\d+) MiB available, leaving (\d+) MiB free")
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out", required=True); a = ap.parse_args()
obs = []
for dp, _, fs in os.walk(a.root):
    for f in fs:
        if not f.endswith(".err"): continue
        p = os.path.join(dp, f)
        for m in PAT.finditer(open(p, errors="replace").read()):
            obs.append(dict(file=os.path.relpath(p, a.root), available_MiB=int(m.group(1)), floor_MiB=int(m.group(2))))
if not obs: raise SystemExit("no sizing lines found")
reach = [o["available_MiB"] - o["floor_MiB"] for o in obs]
out = dict(n_runs=len(obs), available_max_MiB=max(o["available_MiB"] for o in obs),
           available_median_MiB=S.median(o["available_MiB"] for o in obs),
           max_grantable_budget_MiB=max(reach), median_grantable_budget_MiB=S.median(reach),
           budget_7000_reachable_on_any_run=any(r >= 7000 for r in reach),
           budget_6000_reachable_on_any_run=any(r >= 6000 for r in reach),
           runs_reaching_5000=sum(r >= 5000 for r in reach), observations=obs)
json.dump(out, open(a.out, "w"), indent=1)
print({k: v for k, v in out.items() if k != "observations"})
