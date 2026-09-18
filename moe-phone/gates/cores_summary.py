"""
Core placement (device/bmoe_cores.sh), scored as pre-registered in its header. Written before any row.
Arms f0 (current, cpu4-7), 3c (cpu2-5), 3f (6 threads, cpu0-5). Per repeat r, d_r = mean(arm rows of r) -
mean(f0 rows of r); estimate mean(d_r), SE = sd(d_r)/sqrt(n). DECISIVE iff SE/|mean| < 0.5 and the sign holds
in >= n-1 repeats. PRIMARY compute_ms; GUARD stall_ms; SECONDARY decode; counters hit/MiB must not move.
Run:
  python moe-phone/gates/cores_summary.py results/2026-09-18/bmoe_cores/bmoe_cores_<stamp> --out results/2026-09-18/cores_summary.json
"""
import argparse, glob, json, math, os, re, statistics as S
def num(p, t, c=float):
    m = re.search(p, t); return c(m.group(1)) if m else None
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out", required=True); a = ap.parse_args()
log = open(os.path.join(a.root, "log.txt")).read()
foreign = dict(re.findall(r"=== (\w+) \S+ .*?foreign=\[([^\]]*)\]", log))
caps = dict((m[0], (int(m[1]), int(m[2]))) for m in re.findall(r"=== (\w+) .*?\n?exit=\d+ .*?cap0_median_run=(\d+) cap6_median_run=(\d+)", log, re.S))
rows = []
for f in sorted(glob.glob(os.path.join(a.root, "*.out"))):
    tag = os.path.basename(f)[:-4]
    m = re.match(r"(\w+)_rep(\d+)$", tag)
    if not m: continue
    t = open(f).read() + open(f[:-4] + ".err").read()
    r = dict(tag=tag, cell=m.group(1), rep=int(m.group(2)), decode_tok_s=num(r"\(([0-9.]+) tok/s\)", t),
             read_MiB_per_token=num(r"\(([0-9.]+) MiB/token\)", t), hit_pct=num(r"([0-9.]+)% hit", t),
             budget_MiB=num(r"budget (\d+) MiB", t, int), compute_ms=num(r"compute ([0-9.]+) \+", t),
             mgmt_ms=num(r"cache mgmt ([0-9.]+) \+", t), stall_ms=num(r"stall ([0-9.]+) s/token", t),
             foreign=foreign.get(tag), caps=caps.get(tag))
    for k in ("compute_ms", "mgmt_ms", "stall_ms"):
        if r[k] is not None: r[k] *= 1000.0
    r["kept"] = r["decode_tok_s"] is not None and r["budget_MiB"] == 5000 and r["foreign"] == ""
    rows.append(r)
kept = [r for r in rows if r["kept"]]
reps = sorted({r["rep"] for r in rows})
arms = sorted({r["cell"] for r in rows} - {"f0"})
def paired(arm, key):
    d = []
    for rp in reps:
        x = [r[key] for r in kept if r["rep"] == rp and r["cell"] == arm and r[key] is not None]
        b = [r[key] for r in kept if r["rep"] == rp and r["cell"] == "f0" and r[key] is not None]
        if x and b: d.append(S.mean(x) - S.mean(b))
    if len(d) < 2: return dict(n_repeats=len(d), verdict="insufficient repeats")
    m, se = S.mean(d), S.stdev(d) / math.sqrt(len(d))
    ratio = se / abs(m) if m else float("inf")
    sign = sum((x > 0) == (m > 0) for x in d)
    out = dict(n_repeats=len(d), per_repeat=d, mean_diff=m, se=se, se_over_abs_mean=ratio, sign_holds_in=sign,
               decisive=ratio < 0.5 and sign >= len(d) - 1)
    if not out["decisive"] and math.isfinite(ratio): out["n_repeats_needed_for_0_5"] = math.ceil(len(d) * (ratio / 0.5) ** 2)
    return out
res = {arm: {k: paired(arm, k) for k in ("compute_ms", "stall_ms", "mgmt_ms", "decode_tok_s", "read_MiB_per_token", "hit_pct")} for arm in arms}
means = {c: {k: S.mean(r[k] for r in kept if r["cell"] == c and r[k] is not None) for k in ("decode_tok_s", "compute_ms", "stall_ms", "mgmt_ms")} for c in ["f0"] + arms if any(r["cell"] == c for r in kept)}
out = dict(source=os.path.abspath(a.root), rows=rows, n_rows=len(rows), n_kept=len(kept), arm_means=means, vs_f0=res,
           text_match_ok=len(re.findall(r"text_match \S+ OK", log)), text_match_differs=len(re.findall(r"text_match \S+ DIFFERS", log)))
for arm in arms:
    if "f0" in means and arm in means: out.setdefault("decode_ratio_vs_f0", {})[arm] = means[arm]["decode_tok_s"] / means["f0"]["decode_tok_s"]
json.dump(out, open(a.out, "w"), indent=1)
for arm, rr in res.items():
    for k, v in rr.items():
        print(arm, k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "per_repeat"})
print({k: out[k] for k in ("n_rows", "n_kept", "arm_means", "text_match_ok", "text_match_differs")}, out.get("decode_ratio_vs_f0"))
