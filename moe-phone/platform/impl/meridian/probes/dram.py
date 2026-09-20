"""Parser for dramprobe.csv -> Memory.dram_read_gbps.

dramprobe.c already enforces the two hard correctness rules itself (buffer
resident_frac >= 0.95, or it refuses with exit 3) and reports
min_cpu_over_wall per row so descheduled runs are visible. This module only
aggregates the rows that survive validity.reject_descheduled.
"""
from __future__ import annotations

import csv
import statistics
from ..validity import reject_descheduled


def parse(csv_path: str) -> dict:
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("repeat", "").startswith("#"):
                continue
            rows.append(row)
    clean, rejected, reasons = reject_descheduled(rows, "min_cpu_over_wall", threshold=0.90)
    if not clean:
        return {"gbps": None, "n_clean": 0, "n_rejected": len(rejected), "reasons": reasons}
    gbps_values = [float(r["GBps"]) for r in clean]
    resident_fracs = [float(r["resident_frac"]) for r in clean]
    return {
        "gbps_median": statistics.median(gbps_values),
        "gbps_min": min(gbps_values),
        "gbps_max": max(gbps_values),
        "n_clean": len(clean),
        "n_rejected": len(rejected),
        "reasons": reasons,
        "all_resident": all(f >= 0.95 for f in resident_fracs),
    }
