"""
S1 — turn device/dramprobe.c output into the MEASURED --dram-gbps input of
byte_budget.py.

Rules (the same ones g1_analyze.py applies to storage):
  - A row where any thread received < 90% of the wall clock was measuring the
    scheduler, not DRAM. dramprobe marks it `descheduled=1`; it is EXCLUDED and
    counted (`n_invalid_process_descheduled`), never averaged in.
  - Each file is one run in one SELinux domain (adb `shell` or the unprivileged
    app); the domain is named on the command line and kept separate, so an
    app-vs-shell difference is a measurement rather than an assumption.
  - The value handed to byte_budget.py is the MEDIAN of clean rows at the thread
    count with the highest clean median, and the artifact also reports how many
    clean rows each thread count has. A thread count with fewer than 3 clean rows
    is not eligible, because a single lucky row is not a sustained rate.

Run:
  python moe-phone/gates/dram_analyze.py app=results/<date>/dram_15r_app.csv \
      shell=results/<date>/dram_15r_shell.csv
"""
import argparse
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MIN_CLEAN = 3


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(line for line in f if not line.startswith("#")):
            rows.append({"repeat": int(r["repeat"]), "threads": int(r["threads"]),
                         "GBps": float(r["GBps"]), "descheduled": int(r["descheduled"]),
                         "min_cpu_over_wall": float(r["min_cpu_over_wall"]),
                         "resident_frac": float(r["resident_frac"]),
                         "buffer_mb": int(r["buffer_mb"])})
    return rows


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    i = (len(xs) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def analyse(rows):
    by_t = {}
    for r in rows:
        by_t.setdefault(r["threads"], []).append(r)
    table = []
    for t in sorted(by_t):
        clean = [r["GBps"] for r in by_t[t] if not r["descheduled"]]
        table.append({"threads": t, "n": len(by_t[t]), "n_clean": len(clean),
                      "GBps_median_clean": statistics.median(clean) if clean else None,
                      "GBps_q25_clean": q(clean, 0.25), "GBps_q75_clean": q(clean, 0.75),
                      "GBps_all_rows": [r["GBps"] for r in by_t[t]]})
    eligible = [t for t in table if t["n_clean"] >= MIN_CLEAN]
    best = max(eligible, key=lambda t: t["GBps_median_clean"]) if eligible else None
    return {"n_rows": len(rows),
            "n_invalid_process_descheduled": sum(r["descheduled"] for r in rows),
            "min_resident_frac": min(r["resident_frac"] for r in rows),
            "buffer_mb": sorted({r["buffer_mb"] for r in rows}),
            "table": table, "best": best,
            "min_clean_rows_for_eligibility": MIN_CLEAN}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("runs", nargs="+", help="domain=path pairs, e.g. app=dram_15r_app.csv")
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()
    out = {"runs": {}}
    for spec in a.runs:
        dom, _, path = spec.partition("=")
        if not path:
            p.error(f"expected domain=path, got {spec!r}")
        res = analyse(load(path))
        res["source"] = os.path.abspath(path)
        out["runs"][dom] = res
        b = res["best"]
        print(f"[{dom}] {res['n_rows']} rows, {res['n_invalid_process_descheduled']} descheduled "
              f"(excluded), min resident fraction {res['min_resident_frac']:.3f}")
        for t in res["table"]:
            med = t["GBps_median_clean"]
            print(f"  {t['threads']:>2} threads  clean {t['n_clean']}/{t['n']}  "
                  f"median {med if med is None else round(med, 2)} GB/s")
        print(f"  -> --dram-gbps {b['GBps_median_clean']:.2f} ({b['threads']} threads)" if b
              else "  -> no thread count has enough clean rows")
    path = (os.path.join(a.out_dir, "dram_15r.json") if a.out_dir else None)
    if path is None:
        from _paths import write_path
        path = write_path("dram_15r.json")
    else:
        os.makedirs(a.out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
