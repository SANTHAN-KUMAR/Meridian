"""Parser for memprobe's stdout -> Memory.grantable_quiesced.

This is GrantProbe's "quiesced" measurement (04_DEVICE_PROFILING.md section
3.4, item 2): allocate in increments, verify residency after each step, stop
on a plateau. memprobe.c's own stop rule already implements this; we read
its `max_VmRSS_MB=` line and never derive the number any other way (its
docstring explains why the ceiling must be the observed plateau, not the
last allocation attempted).

grantable_foreground (item 3) is NOT computed here: it requires a real
target app in the foreground, which does not exist without the Android app
this repository has not built yet. It is registered as unmeasured
(STATUS.md, extends PL-S2) rather than approximated.
"""
from __future__ import annotations

import re


def parse_multi(stdout_texts: list) -> dict:
    """Aggregate several independent memprobe runs into a real observed
    min/max, instead of one run plus an invented [0, value] band."""
    runs = [parse(t) for t in stdout_texts]
    values = [r["max_vmrss_mb"] for r in runs if r.get("max_vmrss_mb")]
    if not values:
        return {"max_vmrss_mb": None, "range_mb": (None, None), "n_runs": 0}
    return {
        "max_vmrss_mb": sorted(values)[len(values) // 2],  # median run
        "range_mb": (min(values), max(values)),
        "n_runs": len(values),
        "stop_reason": runs[0].get("stop_reason"),
    }


def parse(stdout_text: str) -> dict:
    m = re.search(r"max_VmRSS_MB=(\d+)\s+chunks=(\d+)\s+mode=(\w+)", stdout_text)
    if not m:
        return {"max_vmrss_mb": None, "mode": None}
    stop_reason = None
    for line in stdout_text.splitlines():
        parts = line.split(",")
        if len(parts) == 8 and parts[-1] and parts[-1] != "stop_reason":
            stop_reason = parts[-1]
    return {
        "max_vmrss_mb": int(m.group(1)),
        "chunks": int(m.group(2)),
        "mode": m.group(3),
        "stop_reason": stop_reason,
    }
