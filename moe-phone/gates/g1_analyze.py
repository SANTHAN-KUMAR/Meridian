"""
G1 analysis — turn ufsbench.csv (and optionally memprobe output) into MEASURED
inputs for G-ROOF, replacing stub S1's assumed values.

Checks built in (CLAUDE.md §8: validate against the world, not against yourself):
  - Physical ceiling. No read can exceed the storage interface's bandwidth.
    UFS 4.x (2 lanes x HS-G5) is listed at 5.8 GB/s for UFS 4.0 in ActiveFlow
    (arXiv 2504.08378, Table 2). A row above it was served by a cache between
    the benchmark and the flash (this happened in the WSL self-test: host page
    cache under the VM) and is marked INVALID, never averaged in.
  - Run-to-run noise across --repeats is reported, because ESTIMAND.md §7 sets
    the number of runs per condition from it.

Run:
  python moe-phone/gates/g1_analyze.py path/to/ufsbench.csv [--memprobe path/to/memprobe.csv]
"""
import argparse
import csv
import json
import os
import statistics
import sys

IFACE_CEILING_MBPS = 5800.0  # UFS 4.x interface ceiling; see module docstring
NUMERIC = ("repeat", "size_kb", "threads", "bytes", "seconds", "MBps", "iops",
           "lat_p50_us", "lat_p99_us", "errors", "first_errno", "power_w",
           "temp_start_c", "temp_end_c")


def load(path):
    rows, direct = [], None
    with open(path, encoding="utf-8") as f:
        lines = []
        for line in f:
            if line.startswith("#"):
                if "direct_io_supported=" in line:
                    direct = line.strip().endswith("=1")
                continue
            lines.append(line)
    for r in csv.DictReader(lines):
        for k in NUMERIC:
            try:
                r[k] = float(r[k])
            except (ValueError, KeyError):
                r[k] = float("nan")
        rows.append(r)
    return rows, direct


def load_memprobe(path):
    last, adj = None, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("oom_score_adj="):
                adj = line.split()[0].split("=")[1]
            parts = line.strip().split(",")
            if len(parts) == 5 and parts[0].isdigit():
                last = {"held_MB": int(parts[0]), "MemAvailable_MB": int(parts[2]),
                        "stop_reason": parts[4] or "none"}
    return {"last": last, "oom_score_adj": adj}


def analyse(rows, direct_supported):
    idle = [r["power_w"] for r in rows if r["mode"] == "idle" and r["power_w"] == r["power_w"]]
    idle_w = statistics.median(idle) if idle else None
    data = [r for r in rows if r["mode"] in ("direct", "buffered")]
    invalid = [r for r in data if r["MBps"] > IFACE_CEILING_MBPS]
    valid = [r for r in data if r["MBps"] <= IFACE_CEILING_MBPS and r["errors"] == 0]
    mode = "direct" if direct_supported else "buffered"

    # Median over repeats per (mode, pattern, size, threads).
    cells = {}
    for r in valid:
        key = (r["mode"], r["pattern"], int(r["size_kb"]), int(r["threads"]))
        cells.setdefault(key, []).append(r)
    table = []
    for (m, pat, size, thr), rs in sorted(cells.items()):
        mb = [x["MBps"] for x in rs]
        med = statistics.median(mb)
        pw = [x["power_w"] for x in rs if x["power_w"] == x["power_w"]]
        j_per_gb = None
        if pw and idle_w is not None and med > 0:
            j_per_gb = (statistics.median(pw) - idle_w) / (med / 1000.0)
        table.append({"mode": m, "pattern": pat, "size_kb": size, "threads": thr,
                      "MBps_median": med, "MBps_min": min(mb), "MBps_max": max(mb),
                      "rel_spread": (max(mb) - min(mb)) / med if med else None,
                      "lat_p50_us": statistics.median(x["lat_p50_us"] for x in rs),
                      "lat_p99_us": statistics.median(x["lat_p99_us"] for x in rs),
                      "J_per_GB_above_idle": j_per_gb, "n": len(rs)})

    def best(pattern, pred):
        c = [t for t in table if t["mode"] == mode and t["pattern"] == pattern and pred(t["size_kb"])]
        return max(c, key=lambda t: t["MBps_median"]) if c else None

    bulk = best("rand", lambda s: s >= 512)
    small = best("rand", lambda s: s == 4)
    spreads = [t["rel_spread"] for t in table if t["rel_spread"] is not None and t["n"] > 1]
    temps = [r["temp_end_c"] for r in rows if r["temp_end_c"] == r["temp_end_c"]]
    return {
        "mode_used": mode, "direct_io_supported": direct_supported,
        "n_rows": len(data), "n_invalid_above_interface_ceiling": len(invalid),
        "n_rows_with_errors": sum(1 for r in data if r["errors"] > 0),
        "idle_power_w": idle_w,
        "bulk_rand_best": bulk, "small_4k_rand_best": small,
        "bulk_to_small_ratio": (bulk["MBps_median"] / small["MBps_median"])
        if bulk and small and small["MBps_median"] else None,
        "median_run_to_run_rel_spread": statistics.median(spreads) if spreads else None,
        "max_temp_c": max(temps) if temps else None,
        "table": table,
    }


def main():
    p = argparse.ArgumentParser(description="G1 analysis of ufsbench output")
    p.add_argument("csv")
    p.add_argument("--memprobe", default=None)
    a = p.parse_args()
    rows, direct = load(a.csv)
    res = analyse(rows, direct)
    res["source_csv"] = os.path.abspath(a.csv)
    if a.memprobe:
        res["memprobe"] = load_memprobe(a.memprobe)

    print(f"I/O mode used: {res['mode_used']} (O_DIRECT supported: {direct})")
    print(f"rows: {res['n_rows']}, INVALID above interface ceiling: "
          f"{res['n_invalid_above_interface_ceiling']}, with read errors: {res['n_rows_with_errors']}")
    if res["n_invalid_above_interface_ceiling"]:
        print("  -> a cache sits between the benchmark and the flash; those rows are excluded.")
    b, s = res["bulk_rand_best"], res["small_4k_rand_best"]
    if b:
        print(f"bulk random (>=512 KB): {b['MBps_median']:.0f} MB/s at {b['size_kb']} KB x {b['threads']} threads")
    if s:
        print(f"4 KB random:            {s['MBps_median']:.0f} MB/s at {s['threads']} threads")
    if res["bulk_to_small_ratio"]:
        print(f"bulk / 4 KB ratio:      {res['bulk_to_small_ratio']:.1f}x  (the granularity wall)")
    if res["median_run_to_run_rel_spread"] is not None:
        print(f"median run-to-run spread: {100 * res['median_run_to_run_rel_spread']:.1f}%")
    ram_arg = "--ram-gb <ASSUMED>"
    assumed = ["dram"]
    mp = res.get("memprobe", {}).get("last") if a.memprobe else None
    if mp:
        gb = mp["held_MB"] * 1048576 / 1e9
        print(f"memprobe: one process held {mp['held_MB']} MB (stop: {mp['stop_reason']}) "
              f"= {gb:.2f} GB before pushback — an upper bound on the weight budget")
        ram_arg = f"--ram-gb {gb:.2f}"
    else:
        assumed.append("ram")
    if b and s:
        print("\nG-ROOF v2 command with measured storage inputs:")
        print(f"  python moe-phone/gates/byte_budget.py {ram_arg} "
              f"--flash-gbps {b['MBps_median'] / 1000:.3f} --flash-gbps-scattered {s['MBps_median'] / 1000:.3f} "
              f"--dram-gbps <ASSUMED> --assumed {','.join(assumed)} --threshold-tps 5 --tag measured")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _paths import write_path
    out = write_path("g1_storage.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
