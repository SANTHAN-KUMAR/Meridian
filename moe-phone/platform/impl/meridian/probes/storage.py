"""Parser for ufsbench.csv -> Storage.random_read + efficient_request_size.

efficient_request_size is defined per 03_INTERFACES.md as "the knee: smallest
size reaching bulk rate". We take "bulk rate" as the max MBps observed at the
highest thread count in the sweep, and the knee as the smallest request size
whose MBps at that thread count is within `tolerance` of bulk -- never a
constant tuned by eye (CLAUDE.md section 6.4).

Earlier version of this file's caller (meridian/profiler.py) attached a
fabricated +/-30% interval to every downgraded (non-"measured") storage
point, and a degenerate [x, x] "interval" to efficient_request_size -- both
invented to satisfy contracts.Measured's "non-measured values need an
interval" check, not derived from anything. That is the exact failure this
project's CLAUDE.md exists to prevent: a plausible-looking substitute
standing in for a quantity nobody measured. The fix is here, not in the
caller: run ufsbench with real repeats (like dramprobe already does) and
report the actual min/max observed across them. Where a config was only run
once, this module refuses to invent a spread -- see NoRepeats below.
"""
from __future__ import annotations

import csv


class NoRepeats(Exception):
    """Raised when a caller asks for an interval but the data has only one
    repeat for that config. There is no honest interval to report; the
    caller must either re-run with repeats or accept provenance=unknown."""


def parse_rows(csv_path: str) -> list:
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("repeat", "0") in ("0",) or row.get("mode") == "idle":
                continue  # the idle power-baseline row, not a read measurement
            if row.get("repeat", "").startswith("#"):
                continue
            rows.append(row)
    return rows


def _bulk_rows(rows: list, pattern: str, mode: str, max_threads: int) -> list:
    return [r for r in rows if r["pattern"] == pattern and r["mode"] == mode
            and int(r["threads"]) == max_threads]


def efficient_request_size(rows: list, pattern: str = "rand", mode: str = "buffered",
                            tolerance: float = 0.90) -> dict:
    """Returns size_bytes plus size_bytes_range: the (min, max) knee found
    across whatever repeats are present, computed independently per repeat
    -- never a single-repeat point widened by a made-up factor. If there is
    only one repeat, size_bytes_range == (size_bytes, size_bytes) because
    that IS the honest statement: one observation, zero repeat-derived
    spread. It must not be silently upgraded to a false confidence interval,
    and the caller (profiler.py) is responsible for treating a range that
    equals a point as "no repeat-derived interval available" rather than
    "measured with zero uncertainty".
    """
    subset = [r for r in rows if r["pattern"] == pattern and r["mode"] == mode]
    if not subset:
        return {"size_bytes": None, "size_bytes_range": (None, None), "n_repeats": 0}
    max_threads = max(int(r["threads"]) for r in subset)
    repeats = sorted({r["repeat"] for r in subset})
    knees = []
    for rep in repeats:
        bulk_rows = [r for r in subset if int(r["threads"]) == max_threads and r["repeat"] == rep]
        if not bulk_rows:
            continue
        bulk_mbps = max(float(r["MBps"]) for r in bulk_rows)
        threshold = tolerance * bulk_mbps
        by_size = sorted({int(r["size_kb"]) for r in bulk_rows})
        knee_kb = by_size[-1]
        for size_kb in by_size:
            row = next(r for r in bulk_rows if int(r["size_kb"]) == size_kb)
            if float(row["MBps"]) >= threshold:
                knee_kb = size_kb
                break
        knees.append(knee_kb)
    if not knees:
        return {"size_bytes": None, "size_bytes_range": (None, None), "n_repeats": 0}
    knees_bytes = [k * 1024 for k in knees]
    return {
        "size_bytes": knees_bytes[len(knees_bytes) // 2],  # median repeat's knee
        "size_bytes_range": (min(knees_bytes), max(knees_bytes)),
        "n_repeats": len(knees),
        "at_threads": max_threads,
    }


def random_read_points(rows: list, mode: str = "buffered", pattern: str = "rand") -> list:
    """One point per (size, threads), aggregated across repeats: mbps_median
    plus mbps_range = (min, max) actually observed across repeats. Latency
    percentiles are reported from the median-rate repeat.
    """
    subset = [r for r in rows if r["mode"] == mode and r["pattern"] == pattern]
    configs = sorted({(int(r["size_kb"]), int(r["threads"])) for r in subset})
    points = []
    for size_kb, threads in configs:
        matching = [r for r in subset if int(r["size_kb"]) == size_kb and int(r["threads"]) == threads]
        mbps_values = sorted(float(r["MBps"]) for r in matching)
        median_row = sorted(matching, key=lambda r: float(r["MBps"]))[len(matching) // 2]
        points.append({
            "size_bytes": size_kb * 1024,
            "threads": threads,
            "mbps_median": mbps_values[len(mbps_values) // 2],
            "mbps_range": (mbps_values[0], mbps_values[-1]),
            "n_repeats": len(matching),
            "p50_us": float(median_row["lat_p50_us"]),
            "p99_us": float(median_row["lat_p99_us"]),
        })
    return points


def direct_io_supported(rows: list) -> bool:
    return any(r["mode"] == "direct" and int(r.get("errors", "0")) == 0 for r in rows)


def parse_write(csv_path: str):
    """wrbench.csv -> median/min/max MB/s over repeats with zero write errors.
    None if no valid repeat (never a default)."""
    vals = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                if int(row["errors"]) == 0:
                    vals.append(float(row["MBps"]))
            except (KeyError, ValueError):
                continue
    if not vals:
        return None
    vals.sort()
    return {"mbps_median": vals[len(vals) // 2], "mbps_range": (vals[0], vals[-1]), "n_repeats": len(vals)}
