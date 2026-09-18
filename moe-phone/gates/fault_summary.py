"""
Major page faults and swap per steady decode token, by arm, for any campaign directory of engine rows
(<arm>_rep*.csv). Tests reopened failure R1 (research/2026-09-18_gpu_expert_path_design.md §10): is the slot
arena's +14 ms compute (claim arena2_compute_delta_ms) the cost of the process being swapped to zram?
Per row: majflt summed over decode tokens >= 33 / tokens; last-token swap_mib, mem_available_mib; median compute.
Run:
  python moe-phone/gates/fault_summary.py results/2026-09-18/bmoe_arena2/bmoe_arena2_20260918_1508 \
      --arms base arena --out results/2026-09-18/fault_arena2.json
"""
import argparse, csv, glob, json, os, statistics as S
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--arms", nargs="+", required=True)
ap.add_argument("--out", required=True); a = ap.parse_args()
out = dict(source=os.path.abspath(a.root), arms={})
for arm in a.arms:
    rows = []
    for f in sorted(glob.glob(os.path.join(a.root, f"{arm}_*.csv"))):
        st = [r for r in csv.DictReader(l for l in open(f) if not l.startswith("#")) if int(r["step"]) >= 33]
        if not st: continue
        rows.append(dict(file=os.path.basename(f), majflt_per_token=sum(float(r["majflt"]) for r in st) / len(st),
                         swap_mib=float(st[-1]["swap_mib"]), mem_available_mib=float(st[-1]["mem_available_mib"]),
                         compute_ms=S.median(float(r["compute_ms"]) for r in st)))
    out["arms"][arm] = dict(rows=rows, n=len(rows), **{k + "_median": S.median(x[k] for x in rows)
                            for k in ("majflt_per_token", "swap_mib", "mem_available_mib", "compute_ms")})
json.dump(out, open(a.out, "w"), indent=1)
for arm, d in out["arms"].items():
    print(arm, {k: round(v, 1) for k, v in d.items() if k.endswith("_median")}, "n", d["n"])
