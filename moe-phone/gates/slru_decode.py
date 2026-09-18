"""
SLRU vs LRU decode, clean re-run (bmoe_slru_20260918_1537), scored by the rule pre-registered in
host/chain_after.sh's header BEFORE the run:
  keep row  iff exit=0 AND granted budget == 5000 MiB AND foreign=[] in its log line
  estimate  median(slru kept) / median(lru kept) - 1, every row listed
  verdict   decode claim only if, in every ABBA repeat, every kept SLRU row is faster than every kept LRU
            row of that repeat; otherwise "no decode effect resolved at n=3"
Run:
  python moe-phone/gates/slru_decode.py results/2026-09-18/bmoe_slru_decode/bmoe_slru_20260918_1537 \
      --out results/2026-09-18/slru_decode.json
"""
import argparse, json, re, statistics as S
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out", required=True); a = ap.parse_args()
log = open(a.root + "/log.txt").read().splitlines()
rows, cur = [], None
for line in log:
    m = re.match(r"=== (\w+)_rep(\d)([ab]) .*foreign=\[([^\]]*)\]", line)
    if m:
        cur = dict(cell=m.group(1), rep=int(m.group(2)), slot=m.group(3), foreign=m.group(4)); continue
    if line.startswith("exit=") and cur:
        cur["exit"] = int(re.match(r"exit=(\d+)", line).group(1))
        r = re.search(r"\(([0-9.]+) tok/s\)", line); cur["decode_tok_s"] = float(r.group(1)) if r else None
        b = re.search(r"budget (\d+) MiB", line); cur["budget_MiB"] = int(b.group(1)) if b else None
        mt = re.search(r"\(([0-9.]+) MiB/token", line); cur["read_MiB_per_token"] = float(mt.group(1)) if mt else None
        cur["kept"] = cur["exit"] == 0 and cur["budget_MiB"] == 5000 and cur["foreign"] == "" and cur["decode_tok_s"] is not None
        rows.append(cur); cur = None
kept = [r for r in rows if r["kept"]]
med = {c: S.median(r["decode_tok_s"] for r in kept if r["cell"] == c) for c in ("lru", "slru")}
reps = sorted({r["rep"] for r in rows})
per_rep = []
for rp in reps:
    s = [r["decode_tok_s"] for r in kept if r["rep"] == rp and r["cell"] == "slru"]
    l = [r["decode_tok_s"] for r in kept if r["rep"] == rp and r["cell"] == "lru"]
    per_rep.append(dict(rep=rp, slru=s, lru=l, slru_all_faster=bool(s and l and min(s) > max(l))))
verdict_holds = all(p["slru_all_faster"] for p in per_rep)
out = dict(source=a.root, rows=rows, n_rows=len(rows), n_kept=len(kept), median_tok_s=med,
           ratio_minus_1=med["slru"] / med["lru"] - 1, per_repeat=per_rep,
           verdict="SLRU decode faster" if verdict_holds else "no decode effect resolved at n=3",
           n_repeats_slru_all_faster=sum(p["slru_all_faster"] for p in per_rep))
json.dump(out, open(a.out, "w"), indent=1)
for r in rows: print(r)
print({k: out[k] for k in ("n_rows", "n_kept", "median_tok_s", "ratio_minus_1", "n_repeats_slru_all_faster", "verdict")})
