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
