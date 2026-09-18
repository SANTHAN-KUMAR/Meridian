"""
Slot-arena re-run (device/bmoe_arena2.sh, 2026-09-18 15:08): per-arm medians of the decode rate and of the
engine's own time terms, plus the memory samples the script took every 2 s during each row.

Why a separate script: bmoe_analyze.py summarises rates and cache counters, but the arena's question is
WHERE the time goes (cache management vs compute residual) and whether it costs memory, and the .mem
samples (MemAvailable, SwapFree in kB) are only in this campaign.

Per repeat r the ABBA design gives two rows per arm; `repeat_sign` compares the arms' within-repeat means,
so a drift across the campaign cannot produce the sign.

Run:
  python moe-phone/gates/arena2_summary.py results/2026-09-18/bmoe_arena2/bmoe_arena2_20260918_1508 \
      --runs results/2026-09-18/bmoe_arena2.json --out results/2026-09-18/arena2_summary.json
"""
import argparse
import json
import os
import re
import statistics as S


def mem_min_mib(path):
    avail, swap = [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                avail.append(int(p[0]))
                swap.append(int(p[1]))
    if not avail:
        raise ValueError(f"{path}: no memory samples")
    return min(avail) / 1024.0, min(swap) / 1024.0, len(avail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--runs", required=True, help="bmoe_analyze.py output for the same directory")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = json.load(open(a.runs))["runs"]
    rows = []
    for r in runs:
        tag = r["tag"]
        mm, sm, n = mem_min_mib(os.path.join(a.root, tag + ".mem"))
        rep = re.search(r"rep(\d+)", tag).group(1)
        rows.append(dict(tag=tag, cell=r["cell"], rep=int(rep), decode_tok_s=r["decode_tok_s"],
                         budget_MiB=r["budget_MiB"], cache_hit_pct=r["cache_hit_pct"],
                         read_MiB_per_token=r["read_MiB_per_token"],
                         compute_ms=r["compute_s_per_token"] * 1000.0,
                         cache_mgmt_ms=r["cache_mgmt_s_per_token"] * 1000.0,
                         memavail_min_MiB=mm, swapfree_min_MiB=sm, mem_samples=n))
    text = open(os.path.join(a.root, "log.txt")).read()
    n_match = len(re.findall(r"text_match \S+ OK", text))
    n_differs = len(re.findall(r"text_match \S+ DIFFERS", text))
    cells = {}
    for c in ("base", "arena"):
        rr = [x for x in rows if x["cell"] == c]
        cells[c] = dict(n=len(rr), **{k + "_median": S.median(x[k] for x in rr)
                                      for k in ("decode_tok_s", "compute_ms", "cache_mgmt_ms",
                                                "memavail_min_MiB", "read_MiB_per_token", "cache_hit_pct")},
                        cache_mgmt_ms_max=max(x["cache_mgmt_ms"] for x in rr),
                        cache_mgmt_ms_min=min(x["cache_mgmt_ms"] for x in rr))
    reps = sorted({x["rep"] for x in rows})
    repeat_sign = []
    for rp in reps:
        b = [x["decode_tok_s"] for x in rows if x["rep"] == rp and x["cell"] == "base"]
        ar = [x["decode_tok_s"] for x in rows if x["rep"] == rp and x["cell"] == "arena"]
        repeat_sign.append(dict(rep=rp, base_mean=S.mean(b), arena_mean=S.mean(ar),
                                arena_faster=S.mean(ar) > S.mean(b)))
    out = dict(source=os.path.abspath(a.root), rows=rows, cells=cells, repeat_sign=repeat_sign,
               arena_faster_in_repeats=sum(r["arena_faster"] for r in repeat_sign), n_repeats=len(reps),
               decode_ratio_arena_over_base=cells["arena"]["decode_tok_s_median"] / cells["base"]["decode_tok_s_median"],
               compute_delta_ms=cells["arena"]["compute_ms_median"] - cells["base"]["compute_ms_median"],
               memavail_min_delta_MiB=cells["arena"]["memavail_min_MiB_median"] - cells["base"]["memavail_min_MiB_median"],
               text_match_ok=n_match, text_match_differs=n_differs)
    json.dump(out, open(a.out, "w"), indent=1)
    for k in ("decode_ratio_arena_over_base", "compute_delta_ms", "memavail_min_delta_MiB",
              "arena_faster_in_repeats", "text_match_ok", "text_match_differs"):
        print(k, out[k])
    print({c: {k: round(v, 2) for k, v in d.items()} for c, d in cells.items()})


if __name__ == "__main__":
    main()
