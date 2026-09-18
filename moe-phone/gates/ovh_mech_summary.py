"""
Mechanism of the engine's compute overhead (device/bmoe_ovh_mech.sh), as pre-registered in its header:
per STEADY decode token = (counts at -n 160 - counts at -n 32) / 128, per arm and repeat, for user-space
instructions, cycles, data-TLB walks and L2 data refills (simpleperf stat -o files). Also the steady wall ms/token
from the -n 160 CSV (tokens 33..160).
Run:
  python moe-phone/gates/ovh_mech_summary.py results/2026-09-18/bmoe_ovh_mech/bmoe_ovh_mech_<stamp> \
      --out results/2026-09-18/ovh_mech_summary.json
"""
import argparse, csv, glob, json, os, re, statistics as S
EV = ("instructions:u", "cpu-cycles:u", "raw-dtlb-walk:u", "raw-l2d-cache-refill:u")
def perf(path):
    out = {}
    for line in open(path):
        m = re.match(r"\s*([\d,]+)\s+(\S+)", line)
        if m and m.group(2) in EV: out[m.group(2)] = int(m.group(1).replace(",", ""))
    return out
def steady_wall(csvp, lo=33, hi=160):
    rows = [r for r in csv.DictReader(l for l in open(csvp) if not l.startswith("#"))]
    w = [float(r["wall_ms"]) for r in rows if lo <= int(r["step"]) <= hi]
    return S.median(w) if w else None
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out", required=True); a = ap.parse_args()
per = {}
for arm in ("plain", "stream", "streammap"):
    for rep in (1, 2):
        p32 = os.path.join(a.root, f"{arm}_n32_rep{rep}.perf"); p160 = os.path.join(a.root, f"{arm}_n160_rep{rep}.perf")
        if not (os.path.exists(p32) and os.path.exists(p160)): continue
        c32, c160 = perf(p32), perf(p160)
        d = {e: (c160[e] - c32[e]) / 128.0 for e in EV if e in c32 and e in c160}
        d["steady_wall_ms"] = steady_wall(os.path.join(a.root, f"{arm}_n160_rep{rep}.csv"))
        per.setdefault(arm, []).append(d)
summ = {arm: {k: S.mean(x[k] for x in v if x.get(k) is not None) for k in list(EV) + ["steady_wall_ms"]} for arm, v in per.items()}
for arm in summ:
    s = summ[arm]; s["IPC"] = s["instructions:u"] / s["cpu-cycles:u"]
rel = {arm: {k: summ[arm][k] / summ["plain"][k] for k in summ[arm]} for arm in summ if arm != "plain"}
out = dict(source=os.path.abspath(a.root), per_rep=per, per_token=summ, ratio_vs_plain=rel)
json.dump(out, open(a.out, "w"), indent=1)
for arm, s in summ.items():
    print(arm, {k: (f"{v/1e6:.2f}M" if v > 1e5 else round(v, 3)) for k, v in s.items()})
for arm, r in rel.items():
    print("ratio", arm, {k: round(v, 2) for k, v in r.items()})
