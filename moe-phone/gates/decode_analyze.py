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
  cpu_share         (probe >= 2026-09-16 01:19) process CPU seconds / (wall x threads).
                    CAUTION: llama.cpp's worker threads busy-wait at barriers, so a
                    stalled run can still burn CPU; a LOW share means threads were
                    blocked or descheduled, a high share does not prove useful work.
  memavail_min_MB   the lowest MemAvailable sampled during the run.

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
        key = (r["model"], int(r["threads"]), r["mode"], r["_group_dir"],
               (r.get("extra") or "").strip().strip('"'))
        groups.setdefault(key, []).append(r)
    out = []
    for (model, thr, mode, gdir, extra), rs in sorted(groups.items()):
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
                 "exit": int(r["exit"]),
                 "cpu_share": ((fnum(r.get("cpu_s")) / (fnum(r["t_wall_s"]) * thr))
                               if fnum(r.get("cpu_s")) is not None else None),
                 "memavail_min_MB": (fnum(r.get("memavail_kb_min")) / 1024
                                     if fnum(r.get("memavail_kb_min")) is not None else None)}
                for r in rs]
        out.append({"model": model, "threads": thr, "mode": mode, "dir": gdir, "extra": extra,
                    "majflt_per_run": [fnum(r.get("majflt")) for r in rs],
                    "n_short_long": ns, "runs": runs, "pairs": pairs,
                    "tg_tok_s_median_by_n": {str(n): med([x["tg_tok_s"] for x in runs if x["n_gen"] == n])
                                              for n in ns},
                    "marginal_tok_s_median": med([p["marginal_tok_s"] for p in pairs]),
                    "marginal_tok_s_all": [p["marginal_tok_s"] for p in pairs],
                    "cpu_share_median": med([x["cpu_share"] for x in runs]),
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
    # Cells of an interleaved sweep (one repeat per directory) pooled per (model, mode,
    # threads). Pooling ACROSS those keys is never done.
    cells = {}
    for g in res["groups"]:
        c = cells.setdefault((g["model"], g["mode"], g["threads"], g["extra"]),
                             {"marg": [], "flash": [], "cpu": [], "majflt": [], "tg_long": []})
        # llama-bench's OWN generation rate on the long run. Under memory pressure the page cache is
        # evicted between runs, so a short run can absorb a re-fault the long run does not; the
        # marginal difference then inflates (negative flash bytes are the tell). tg over n_long
        # is the robust per-configuration rate and is reported beside it.
        n_long = max(g["n_short_long"]) if g["n_short_long"] else None
        c["tg_long"] += [x["tg_tok_s"] for x in g["runs"] if x["n_gen"] == n_long and x["tg_tok_s"]]
        c["majflt"] += [x for x in g["majflt_per_run"] if x is not None]
        c["marg"] += [x for x in g["marginal_tok_s_all"] if x]
        c["flash"] += [p_["marginal_flash_MB_per_tok"] for p_ in g["pairs"]]
        c["cpu"] += [x["cpu_share"] for x in g["runs"] if x["cpu_share"] is not None]
    res["by_threads"] = [{"model": m, "mode": mo, "threads": t, "extra": ex, "n_pairs": len(c["marg"]),
                          "majflt_median": med(c["majflt"]),
                          "tg_tok_s_long_all": c["tg_long"], "tg_tok_s_long_median": med(c["tg_long"]),
                          "marginal_suspect": any(f is not None and f < 0 for f in c["flash"]),
                          "marginal_tok_s_all": c["marg"], "marginal_tok_s_median": med(c["marg"]),
                          "marginal_flash_MB_per_tok_median": med(c["flash"]),
                          "cpu_share_median": med(c["cpu"])}
                         for (m, mo, t, ex), c in sorted(cells.items())]
    for c in res["by_threads"]:
        print(f"  BY THREADS {c['model'][:34]:<34} {c['mode']:<5} t{c['threads']} [{c['extra'] or 'default'}]: "
              f"tg@long {[round(x, 1) for x in c['tg_tok_s_long_all']]} median {c['tg_tok_s_long_median']} | "
              f"{c['marginal_tok_s_median'] if c['marginal_tok_s_median'] is None else round(c['marginal_tok_s_median'], 1)} tok/s "
              f"(n={c['n_pairs']}: {[round(x, 1) for x in c['marginal_tok_s_all']]}), "
              f"flash {c['marginal_flash_MB_per_tok_median']} MB/tok, cpu share {c['cpu_share_median']}")
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
