"""
Patch 0012 (--spec-adopt-selective) on the phone: device/bmoe_specadopt.sh, scored by its pre-registered
header. Arms base / predpf / predsel, Latin rotation over 3 repeats.
Per row, from the engine's own output: decode rate, stall (moe-overlap), compute and cache-mgmt terms,
MiB read per token, hit rate, speculative MiB and useful/integrated experts (moe-prefetch), adopted entries
(spec-adopt-selective line), and text identity against the first row.
Verdict rule (pre-registered): rate claim only if predsel beats BOTH other arms in EVERY repeat.
Run:
  python moe-phone/gates/specadopt_summary.py results/2026-09-18/bmoe_specadopt/bmoe_specadopt_20260918_1601 \
      --out results/2026-09-18/specadopt_summary.json
"""
import argparse, glob, json, os, re, statistics as S
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out", required=True); a = ap.parse_args()
def num(pat, text, cast=float):
    m = re.search(pat, text); return cast(m.group(1)) if m else None
rows = []
for f in sorted(glob.glob(os.path.join(a.root, "*.out"))):
    tag = os.path.basename(f)[:-4]
    cell, rep = re.match(r"(\w+)_rep(\d+)", tag).groups()
    out = open(f).read(); err = open(f[:-4] + ".err").read(); t = out + err
    r = dict(tag=tag, cell=cell, rep=int(rep),
             decode_tok_s=num(r"\(([0-9.]+) tok/s\)", t),
             read_MiB_per_token=num(r"\(([0-9.]+) MiB/token\)", t),
             compute_ms=num(r"compute ([0-9.]+) \+", t), cache_mgmt_ms=num(r"cache mgmt ([0-9.]+) \+", t),
             stall_ms=num(r"stall ([0-9.]+) s/token", t), hit_pct=num(r"([0-9.]+)% hit", t),
             spec_MiB=num(r"moe-prefetch: ([0-9.]+) MiB speculative", t),
             spec_useful=num(r"experts? useful|(\d+)/\d+ experts useful", t) if False else num(r"(\d+)/\d+ experts useful", t, int),
             spec_integrated=num(r"\d+/(\d+) experts useful", t, int),
             adopted=num(r"adopted (\d+)", t, int), cancelled=num(r"cancelled (\d+)", t, int),
             left_to_finish=num(r"left-to-finish (\d+)", t, int),
             n_generated=num(r"generation: (\d+) tokens", t, int))
    for k in ("compute_ms", "cache_mgmt_ms", "stall_ms"):
        if r[k] is not None: r[k] *= 1000.0
    rows.append(r)
log = open(os.path.join(a.root, "log.txt")).read()
cells = {}
for c in ("base", "predpf", "predsel"):
    rr = [r for r in rows if r["cell"] == c]
    def med(k):
        v = [r[k] for r in rr if r[k] is not None]; return S.median(v) if v else None
    cells[c] = dict(n=len(rr), **{k + "_median": med(k) for k in ("decode_tok_s", "stall_ms", "compute_ms",
                    "cache_mgmt_ms", "read_MiB_per_token", "hit_pct", "spec_MiB", "spec_useful", "adopted")})
    if cells[c]["spec_useful_median"] is not None:
        cells[c]["useful_spec_experts_per_token"] = cells[c]["spec_useful_median"] / 256.0
per_rep = []
for rp in sorted({r["rep"] for r in rows}):
    d = {r["cell"]: r["decode_tok_s"] for r in rows if r["rep"] == rp}
    per_rep.append(dict(rep=rp, **d, predsel_best=d["predsel"] > max(d["base"], d["predpf"])))
out = dict(source=os.path.abspath(a.root), rows=rows, cells=cells, per_repeat=per_rep,
           manipulation_check_adopted_min=min(r["adopted"] for r in rows if r["cell"] == "predsel"),
           predsel_best_in_repeats=sum(p["predsel_best"] for p in per_rep),
           verdict=("predsel faster" if all(p["predsel_best"] for p in per_rep) else "no rate effect resolved at n=3"),
           text_match_ok=len(re.findall(r"text_match \S+ OK", log)),
           text_match_differs=len(re.findall(r"text_match \S+ DIFFERS", log)),
           stall_delta_predsel_vs_base_ms=cells["predsel"]["stall_ms_median"] - (cells["base"]["stall_ms_median"] or 0) if cells["base"]["stall_ms_median"] is not None else None,
           read_delta_predsel_vs_base=cells["predsel"]["read_MiB_per_token_median"] - cells["base"]["read_MiB_per_token_median"],
           useful_ratio_predsel_over_predpf=cells["predsel"]["spec_useful_median"] / cells["predpf"]["spec_useful_median"])
json.dump(out, open(a.out, "w"), indent=1)
for c, d in cells.items(): print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()})
for p in per_rep: print(p)
print({k: out[k] for k in ("manipulation_check_adopted_min", "predsel_best_in_repeats", "verdict", "text_match_ok",
                           "text_match_differs", "read_delta_predsel_vs_base", "useful_ratio_predsel_over_predpf")})
