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

# Temperature validity, checked here as well as in device/ufsbench.c, because a
# collector and its analysis must not share one assumption (CLAUDE.md section 8:
# validate against something outside the code). Linux thermal sysfs reports
# millidegrees Celsius, and a phone die outside 0-120 C is not a reading.
#
# This band alone is not enough, which is why CONSTANT_TEMP_MIN_ROWS exists: on
# the OnePlus 15R, thermal_zone85 `socd` reports a bare 75 (state-of-charge
# depletion, not a temperature) and the 2026-09-14 collector read it as a
# perfectly plausible 75.0 C in EVERY row. A temperature that never moves across
# a multi-minute sustained I/O benchmark is not a sensor, whatever its value.
TEMP_MIN_C, TEMP_MAX_C = 0.0, 120.0
CONSTANT_TEMP_MIN_ROWS = 10

# Each ufsbench configuration is a deadline-driven loop: workers stop at
# t0 + --seconds, so a row should overrun its deadline by at most one I/O
# latency (p99 is milliseconds against a 2-second deadline, i.e. well under 1%).
# A row that took materially longer was not RUNNING for part of its interval,
# and its bytes/elapsed is then a measure of how much CPU the process was given,
# not of how fast the storage is.
#
# Observed 2026-09-14: with the phone's screen off, a 2.0 s configuration took
# 4.88 s and reported 108 MB/s where the same configuration with the screen on
# gave 814 MB/s — while p50 latency was unchanged (158 vs 150 us). Latency
# unchanged with throughput collapsed is the signature: the device was fine, the
# process was descheduled. The tolerance is set from the deadline structure
# above, not fitted to any result; the real overrun was 144%.
SCHEDULING_OVERRUN_TOL = 0.25


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
    """Parse device/memprobe.c output.

    The quantity that feeds byte_budget.py's --ram-gb is `max_VmRSS_MB`: bytes
    the process kept RESIDENT in DRAM. It is NOT the last `held_MB`, which
    counts pages the kernel has already compressed into zram and which an
    engine cannot read at DRAM speed.

    Parsed by column NAME from the header row, not by field count. The previous
    version matched `len(parts) == 5` against memprobe's old 5-column format;
    when the collector grew to 8 columns that test stopped matching, and
    --memprobe silently contributed nothing while main() fell back to printing
    `--ram-gb <ASSUMED>`. A parse that finds nothing now says so.
    """
    header, rows, meta = None, [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "=" in line and "," not in line:
                for part in line.split():
                    if "=" in part:
                        k, _, v = part.partition("=")
                        meta[k] = v
            elif line.startswith("held_MB,"):
                header = line.split(",")
            elif header and line[0].isdigit():
                rows.append(dict(zip(header, line.split(","))))
    if not rows:
        return {"error": f"no data rows parsed from {path}", "meta": meta}
    last = rows[-1]
    res = {
        "meta": meta, "n_chunks": len(rows),
        # Which regime produced this ceiling. Anonymous and file-backed pages
        # are different quantities; an artifact that does not say which one it
        # holds cannot be used for either.
        "mode": meta.get("mode", "anonymous"),
        "last_held_MB": int(last["held_MB"]),
        "MemAvailable_MB_at_stop": int(last["MemAvailable_MB"]),
        "SwapFree_MB_at_stop": int(last["SwapFree_MB"]),
        "stop_reason": last["stop_reason"] or "none",
    }
    if "max_VmRSS_MB" in meta:
        mb = int(meta["max_VmRSS_MB"])
        res["max_VmRSS_MB"] = mb
        # Decimal GB, because byte_budget.py's --ram-gb is decimal. Converting
        # in exactly one place keeps the two from drifting apart.
        res["resident_GB_decimal"] = round(mb * 1048576 / 1e9, 3)
    else:
        res["error"] = ("no max_VmRSS_MB line — this CSV is from a memprobe "
                        "older than 2026-09-14, which reported allocated, not "
                        "resident, memory. Rebuild and re-run.")
    if "VmSwap_MB" in last:
        res["VmSwap_MB_at_stop"] = int(last["VmSwap_MB"])
    return res


def analyse(rows, direct_supported):
    idle = [r["power_w"] for r in rows if r["mode"] == "idle" and r["power_w"] == r["power_w"]]
    idle_w = statistics.median(idle) if idle else None
    data = [r for r in rows if r["mode"] in ("direct", "buffered")]
    invalid = [r for r in data if r["MBps"] > IFACE_CEILING_MBPS]

    # Rows where the process was not scheduled for part of its interval (see
    # SCHEDULING_OVERRUN_TOL). The nominal duration is the median across rows
    # rather than a command-line value the CSV does not carry.
    durations = sorted(r["seconds"] for r in data if r["seconds"] == r["seconds"])
    nominal_s = statistics.median(durations) if durations else None
    descheduled = [r for r in data
                   if nominal_s and r["seconds"] > nominal_s * (1 + SCHEDULING_OVERRUN_TOL)]
    bad = {id(r) for r in descheduled}

    valid = [r for r in data if r["MBps"] <= IFACE_CEILING_MBPS
             and r["errors"] == 0 and id(r) not in bad]

    # ufsbench writes "# direct_io_supported=N" only when it finishes, so a
    # truncated or still-running CSV has no trailer. Treating a missing trailer
    # as "direct unsupported" would silently report BUFFERED bandwidths under a
    # direct-I/O heading — a different quantity wearing the right name. Infer it
    # from the rows instead, and say which way it was decided.
    if direct_supported is None:
        mode = "direct" if any(r["mode"] == "direct" for r in data) else "buffered"
        mode_source = "inferred from rows; collector trailer absent (run truncated?)"
    else:
        mode = "direct" if direct_supported else "buffered"
        mode_source = "collector trailer"

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

    # --- temperature validity (see the TEMP_* constants above) ---------------
    finite = [r for r in data if r["temp_end_c"] == r["temp_end_c"]]
    plausible = [r for r in finite if TEMP_MIN_C <= r["temp_end_c"] <= TEMP_MAX_C]
    implausible = len(finite) - len(plausible)
    hottest = max(plausible, key=lambda r: r["temp_end_c"], default=None)
    distinct = {round(r["temp_end_c"], 3) for r in plausible}
    temp_suspect = len(plausible) >= CONSTANT_TEMP_MIN_ROWS and len(distinct) == 1
    temp_info = {
        "max_temp_c": hottest["temp_end_c"] if hottest else None,
        # Present only in CSVs from ufsbench built after 2026-09-14; older
        # artifacts get None rather than a guess.
        "max_temp_zone": hottest.get("temp_zone") if hottest else None,
        "n_rows_temp_implausible": implausible,
        "n_distinct_temps": len(distinct),
        "temp_column_suspect_constant": temp_suspect,
    }
    return {
        "mode_used": mode, "direct_io_supported": direct_supported,
        "mode_determined_by": mode_source,
        "n_rows": len(data), "n_invalid_above_interface_ceiling": len(invalid),
        "nominal_seconds": nominal_s,
        "n_invalid_process_descheduled": len(descheduled),
        "descheduled_rows": [{"repeat": r["repeat"], "mode": r["mode"], "pattern": r["pattern"],
                              "size_kb": r["size_kb"], "threads": r["threads"],
                              "seconds": r["seconds"], "MBps": r["MBps"]}
                             for r in descheduled[:20]],
        "n_rows_with_errors": sum(1 for r in data if r["errors"] > 0),
        "idle_power_w": idle_w,
        "bulk_rand_best": bulk, "small_4k_rand_best": small,
        "bulk_to_small_ratio": (bulk["MBps_median"] / small["MBps_median"])
        if bulk and small and small["MBps_median"] else None,
        "median_run_to_run_rel_spread": statistics.median(spreads) if spreads else None,
        "thermal": temp_info,
        "table": table,
    }


def main():
    p = argparse.ArgumentParser(description="G1 analysis of ufsbench output")
    p.add_argument("csv")
    p.add_argument("--memprobe", nargs="*", default=[],
                   help="one or more memprobe CSVs, anonymous and/or --file runs. Pass every "
                        "repeat: the resident ceiling is a pressure effect, so its spread is "
                        "part of the result.")
    a = p.parse_args()
    rows, direct = load(a.csv)
    res = analyse(rows, direct)
    res["source_csv"] = os.path.abspath(a.csv)
    if a.memprobe:
        # Summarised by the SAME function g0_summarize.py uses, so the two
        # scripts cannot print different --ram-gb values for one set of runs.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from g0_summarize import summarise_memprobe_runs
        res["memprobe_runs"] = [dict(load_memprobe(m), source=os.path.abspath(m))
                                for m in a.memprobe]
        res["memprobe"] = summarise_memprobe_runs(res["memprobe_runs"])

    print(f"I/O mode used: {res['mode_used']} (O_DIRECT supported: {direct}; "
          f"{res['mode_determined_by']})")
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
    th = res["thermal"]
    zone = f" (zone {th['max_temp_zone']})" if th["max_temp_zone"] else ""
    print(f"thermal: max {th['max_temp_c']} C{zone}, "
          f"{th['n_distinct_temps']} distinct values, "
          f"{th['n_rows_temp_implausible']} implausible rows excluded")
    if th["temp_column_suspect_constant"]:
        print("  -> SUSPECT: every row reports the same temperature under sustained load. "
              "That is a level/threshold pseudo-zone, not a sensor; treat the column as void.")
    if res["idle_power_w"] is None:
        print("energy: NOT MEASURED — no readable battery rail in this run "
              "(J/GB is absent, not zero).")
    ram_arg = "--ram-gb <ASSUMED>"
    assumed = ["dram"]
    mp = res.get("memprobe")
    if mp:
        for err in mp["errors"]:
            print(f"memprobe: NOT USABLE — {err}")
        for mode, st in mp["by_mode"].items():   # not `s`: that is the 4 KB row above
            print(f"memprobe [{mode}]: max resident median {st['max_VmRSS_MB_median']:.0f} MB "
                  f"= {st['resident_GB_decimal_median']:.2f} GB over n={st['n']} "
                  f"(range {st['max_VmRSS_MB_min']}-{st['max_VmRSS_MB_max']} MB)")
        # file_backed is the regime an mmap'd checkpoint lives in, so it is the
        # budget for model weights; anonymous is the engine's own buffers. Prefer
        # the former and SAY which was used, rather than printing a number whose
        # regime the reader has to guess.
        for mode in ("file_backed", "anonymous"):
            if mode in mp["by_mode"]:
                ram_arg = f"--ram-gb {mp['by_mode'][mode]['resident_GB_decimal_median']:.2f}"
                print(f"  using the {mode} median for --ram-gb (POSITION.md stub S5)")
                break
    if ram_arg.endswith("<ASSUMED>"):
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
