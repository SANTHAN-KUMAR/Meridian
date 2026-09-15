"""
G6 (first slice) — on-device decode rate and flash traffic of an unmodified
engine, from device/llm_decode_probe.sh output.

ESTIMAND.md §1-2 quantities, each obtained two ways where possible (CLAUDE.md §6.5):

  tg_tok_s          llama-bench's own generation rate (avg over the n tokens,
                    excluding load and prompt)
  marginal s/token  (t_wall(n_long) - t_wall(n_short)) / (n_long - n_short), paired
                    by repeat. Load, fault-in and first-token costs cancel, so this
                    is the steady-state window of ESTIMAND §1 (tokens > n_short).
  marginal B_flash  same differencing on /proc/<pid>/io read_bytes: bytes that
                    reached the block layer per decoded token. Page-cache hits are
                    excluded by definition of read_bytes.
  file_fault_multiple  read_bytes / checkpoint size for a whole run: >1 means
                    pages were evicted and faulted back in (thrash).

Each run records the MemAvailable it started from, because on this phone the
file-backed budget is bimodal (1.6 vs 4.8 GB, G-RAM-SWEEP) and decides whether a
3.9 GB checkpoint is beyond DRAM at all. Rows are grouped by (model, threads, mode)
and NEVER pooled across groups.

Run:
  python moe-phone/gates/decode_analyze.py results/<date>/decode_15r/ --model-bytes MODEL=BYTES ...
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def load_rows(root):
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "**", "runs.csv"), recursive=True)):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["_group_dir"] = os.path.relpath(os.path.dirname(p), root)
                rows.append(r)
    return rows


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def analyse(rows, model_bytes):
    groups = {}
    for r in rows:
        key = (r["model"], int(r["threads"]), r["mode"], r["_group_dir"])
        groups.setdefault(key, []).append(r)
    out = []
    for (model, thr, mode, gdir), rs in sorted(groups.items()):
        ns = sorted({int(r["n_gen"]) for r in rs})
        by = {(int(r["rep"]), int(r["n_gen"])): r for r in rs}
        pairs = []
        if len(ns) >= 2:
            a, b = ns[0], ns[-1]
            for rep in sorted({int(r["rep"]) for r in rs}):
                if (rep, a) in by and (rep, b) in by:
                    ra, rb = by[(rep, a)], by[(rep, b)]
                    dt = fnum(rb["t_wall_s"]) - fnum(ra["t_wall_s"])
                    db = fnum(rb["read_bytes"]) - fnum(ra["read_bytes"])
                    pairs.append({"rep": rep, "marginal_s_per_tok": dt / (b - a),
                                  "marginal_tok_s": (b - a) / dt if dt > 0 else None,
                                  "marginal_flash_MB_per_tok": db / (b - a) / 1e6,
                                  "memavail_MB_start": [fnum(ra["memavail_kb_start"]) / 1024,
                                                        fnum(rb["memavail_kb_start"]) / 1024]})
        mb = model_bytes.get(model)
        runs = [{"n_gen": int(r["n_gen"]), "rep": int(r["rep"]),
                 "tg_tok_s": fnum(r["tg_tok_s"]), "t_wall_s": fnum(r["t_wall_s"]),
                 "read_GB": fnum(r["read_bytes"]) / 1e9,
                 "file_fault_multiple": (fnum(r["read_bytes"]) / mb) if mb else None,
                 "max_vmrss_MB": fnum(r["max_vmrss_kb"]) / 1024,
                 "max_rssfile_MB": fnum(r["max_rssfile_kb"]) / 1024,
                 "anon_MB_at_peak_est": (fnum(r["max_vmrss_kb"]) - fnum(r["max_rssfile_kb"])) / 1024,
                 "memavail_MB_start": fnum(r["memavail_kb_start"]) / 1024,
                 "resident_after_drop": fnum(r["resident_after_drop"]),
                 "exit": int(r["exit"])} for r in rs]
        out.append({"model": model, "threads": thr, "mode": mode, "dir": gdir,
                    "n_short_long": ns, "runs": runs, "pairs": pairs,
                    "tg_tok_s_median_by_n": {str(n): med([x["tg_tok_s"] for x in runs if x["n_gen"] == n])
                                              for n in ns},
                    "marginal_tok_s_median": med([p["marginal_tok_s"] for p in pairs]),
                    "marginal_flash_MB_per_tok_median": med([p["marginal_flash_MB_per_tok"] for p in pairs]),
                    "n_failed_runs": sum(1 for x in runs if x["exit"] != 0)})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("root")
    p.add_argument("--model-bytes", nargs="*", default=[],
                   help="basename=bytes of each checkpoint, for the fault multiple")
    p.add_argument("--out-name", default="decode_15r.json")
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()
    mb = {k: float(v) for k, v in (s.split("=") for s in a.model_bytes)}
    res = {"source": os.path.abspath(a.root), "groups": analyse(load_rows(a.root), mb)}
    for g in res["groups"]:
        print(f"{g['model']:<42} t{g['threads']} {g['mode']:<5} tg by n {g['tg_tok_s_median_by_n']} "
              f"marginal {g['marginal_tok_s_median']} tok/s, "
              f"{g['marginal_flash_MB_per_tok_median']} MB/tok flash, failed {g['n_failed_runs']}")
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        path = os.path.join(a.out_dir, a.out_name)
    else:
        from _paths import write_path
        path = write_path(a.out_name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
