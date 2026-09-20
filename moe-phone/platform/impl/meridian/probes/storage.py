"""Parser for ufsbench.csv -> Storage.random_read + efficient_request_size.

efficient_request_size is defined per 03_INTERFACES.md as "the knee: smallest
size reaching bulk rate". We take "bulk rate" as the max MBps observed at the
highest thread count in the sweep, and the knee as the smallest request size
whose MBps at that thread count is within `tolerance` of bulk -- never a
constant tuned by eye (CLAUDE.md section 6.4).
"""
from __future__ import annotations

import csv


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


def efficient_request_size(rows: list, pattern: str = "rand", mode: str = "buffered",
                            tolerance: float = 0.90) -> dict:
    subset = [r for r in rows if r["pattern"] == pattern and r["mode"] == mode]
    if not subset:
        return {"size_bytes": None, "basis": "no matching rows"}
    max_threads = max(int(r["threads"]) for r in subset)
    bulk_rows = [r for r in subset if int(r["threads"]) == max_threads]
    bulk_mbps = max(float(r["MBps"]) for r in bulk_rows)
    by_size = sorted({int(r["size_kb"]) for r in bulk_rows})
    threshold = tolerance * bulk_mbps
    for size_kb in by_size:
        row = next(r for r in bulk_rows if int(r["size_kb"]) == size_kb)
        if float(row["MBps"]) >= threshold:
            return {
                "size_bytes": size_kb * 1024,
                "bulk_mbps": bulk_mbps,
                "threshold_mbps": threshold,
                "at_threads": max_threads,
            }
    return {"size_bytes": by_size[-1] * 1024, "bulk_mbps": bulk_mbps,
            "threshold_mbps": threshold, "at_threads": max_threads}


def random_read_points(rows: list, mode: str = "buffered", pattern: str = "rand") -> list:
    points = []
    for r in rows:
        if r["mode"] != mode or r["pattern"] != pattern:
            continue
        points.append({
            "size_bytes": int(r["size_kb"]) * 1024,
            "threads": int(r["threads"]),
            "mbps": float(r["MBps"]),
            "p50_us": float(r["lat_p50_us"]),
            "p99_us": float(r["lat_p99_us"]),
        })
    return points


def direct_io_supported(rows: list) -> bool:
    return any(r["mode"] == "direct" and int(r.get("errors", "0")) == 0 for r in rows)
