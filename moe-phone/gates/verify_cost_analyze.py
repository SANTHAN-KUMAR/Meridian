"""
Verify cost c(N) of a streamed MoE on the phone, from device/bmoe_verify_cost.sh logs.

Each run is teacher-forced --ppl over the same text with --ppl-batch N (N tokens per llama_decode, logits at
every position): one decode is exactly a speculative verify of N positions with real routing and the real
expert cache. Per run:
  n_tokens      = scored + skip(8) + 1   (bmoe --ppl scores tokens[skip .. n-2]; "hits: a/b" gives b = scored)
  decodes       = ceil(n_tokens / N)
  s_per_decode  = wall seconds printed by bmoe / decodes
  c(N)          = median s_per_decode at N / median s_per_decode at N=1
A speculative decoder drafting N-1 tokens beats plain decode only if tokens accepted per verify exceeds c(N)
plus its drafting cost expressed in plain-decode units. Thermal state and wakefulness are carried per run.

Run:
  python moe-phone/gates/verify_cost_analyze.py results/2026-09-17/bmoe_verify_clean/log.txt --out-name verify_cost.json
"""
import argparse, collections, json, math, os, re, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SKIP = 8  # bmoe --ppl-skip default


def parse(path):
    runs = []
    for block in open(path, encoding="utf-8").read().split("=== ")[1:]:
        tag = block.split()[0]
        m = re.match(r"b(\d+)_rep(\d+)$", tag)
        if not m:
            raise ValueError(f"unexpected tag {tag!r}")
        t = re.search(r"\s([0-9.]+) s\s*$", block.strip())
        hits = re.search(r"hits: (\d+)/(\d+)", block)
        nll = re.search(r"nll: ([0-9.]+)", block)
        after = re.search(r"AFTER wake=(\w+) status=(\d+) cap0=(\d+) cap6=(\d+)", block)
        ok = re.search(r"exit=(\d+)", block)
        if not (t and hits and ok) or ok.group(1) != "0":
            runs.append({"tag": tag, "N": int(m.group(1)), "rep": int(m.group(2)), "failed": True})
            continue
        n_tok = int(hits.group(2)) + SKIP + 1
        N = int(m.group(1))
        dec = math.ceil(n_tok / N)
        runs.append({"tag": tag, "N": N, "rep": int(m.group(2)), "failed": False, "wall_s": float(t.group(1)),
                     "n_tokens": n_tok, "decodes": dec, "s_per_decode": float(t.group(1)) / dec,
                     "nll": float(nll.group(1)) if nll else None,
                     "wake": after.group(1) if after else None, "thermal_status": int(after.group(2)) if after else None,
                     "cap0_khz": int(after.group(3)) if after else None, "cap6_khz": int(after.group(4)) if after else None})
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("log")
    p.add_argument("--out-name", required=True)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()
    runs = parse(a.log)
    ok = [r for r in runs if not r["failed"]]
    by = collections.defaultdict(list)
    for r in ok:
        by[r["N"]].append(r)
    if 1 not in by:
        raise ValueError("no N=1 runs: c(N) needs the single-token reference")
    base = statistics.median(r["s_per_decode"] for r in by[1])
    cells = []
    for N in sorted(by):
        s = [r["s_per_decode"] for r in by[N]]
        cells.append({"N": N, "n": len(s), "s_per_decode_all": s, "s_per_decode_median": statistics.median(s),
                      "c_median": statistics.median(s) / base,
                      "nll_all": [r["nll"] for r in by[N]],
                      "all_awake": all(r["wake"] == "Awake" for r in by[N]),
                      "thermal_status_all": [r["thermal_status"] for r in by[N]]})
        print(f"N={N}: s/decode {[round(x, 3) for x in s]} median {statistics.median(s):.3f}  c(N)={statistics.median(s)/base:.2f}")
    out = {"source": os.path.abspath(a.log), "skip": SKIP, "n_failed": len(runs) - len(ok), "runs": runs, "cells": cells}
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        path = os.path.join(a.out_dir, a.out_name)
    else:
        from _paths import write_path
        path = write_path(a.out_name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
