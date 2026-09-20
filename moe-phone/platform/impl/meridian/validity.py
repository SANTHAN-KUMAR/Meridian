"""ValidityGuard, per platform/04_DEVICE_PROFILING.md section 5.

The guard never looks at the measured value, only at the conditions that
were true while it was taken -- so it cannot reject a number for being
inconvenient (04_DEVICE_PROFILING.md section 5).
"""
from __future__ import annotations

import re
from .contracts import ValidityConditions


def parse_battery_dumpsys(text: str) -> dict:
    """Extract charging state and battery percentage from `adb shell dumpsys battery`."""
    usb_powered = re.search(r"USB powered:\s*(\w+)", text)
    ac_powered = re.search(r"AC powered:\s*(\w+)", text)
    level = re.search(r"^\s*level:\s*(\d+)", text, re.M)
    if ac_powered and ac_powered.group(1) == "true":
        power = "ac"
    elif usb_powered and usb_powered.group(1) == "true":
        power = "usb"
    else:
        power = "unplugged"
    battery_pct = int(level.group(1)) if level else None
    return {"power": power, "battery_pct": battery_pct}


def parse_power_state(text: str) -> dict:
    """Extract wakefulness from `adb shell dumpsys power`."""
    m = re.search(r"mWakefulness=(\w+)", text)
    raw = m.group(1) if m else "unknown"
    mapping = {"Awake": "awake", "Dozing": "dozing", "Asleep": "dozing"}
    return {"wakefulness": mapping.get(raw, "unknown")}


# Packages that count as "no foreground app" for profiling purposes: the
# home screen and system UI are not a foreground workload competing for
# memory/CPU/thermal headroom the way a real app is.
_IDLE_PACKAGES = ("com.android.launcher", "com.android.systemui", "NA")


def classify_foreground(resumed_activity_text: str) -> tuple:
    """Parse `dumpsys activity activities | grep -i resumed` and classify
    what was actually in the foreground while a probe ran. This replaced a
    hardcoded foreground="none" that a real session caught: the user was
    using another app (YouTube) during part of an earlier profiling run, and
    the hardcoded value would have silently reported that run as quiesced
    when it was not (04_DEVICE_PROFILING.md section 4's "foreground loss"
    contaminant, just inverted -- foreground GAIN by a third party).

    Returns (foreground_class, package_name_or_None).
    """
    m = re.search(r"ResumedActivity:\s*ActivityRecord\{[^ ]+ u\d+ ([^ /]+)/", resumed_activity_text)
    if not m:
        return "unknown", None
    pkg = m.group(1)
    if any(pkg.startswith(p) for p in _IDLE_PACKAGES):
        return "none", pkg
    return "other-app", pkg


def build_conditions(battery_text: str, power_text: str, resumed_activity_text: str = "",
                      concurrent_load: list | None = None) -> ValidityConditions:
    """Assemble ValidityConditions from the raw dumpsys captures. foreground
    is measured from resumed_activity_text, not assumed -- see
    classify_foreground(). An empty resumed_activity_text yields "unknown",
    never a silent "none" (03_INTERFACES.md Rule 1: never substitute a
    default and call it a value).
    """
    b = parse_battery_dumpsys(battery_text)
    p = parse_power_state(power_text)
    fg, fg_pkg = classify_foreground(resumed_activity_text) if resumed_activity_text else ("unknown", None)
    load = list(concurrent_load or [])
    if fg == "other-app" and fg_pkg:
        load.append(fg_pkg)
    return ValidityConditions(
        wakefulness=p["wakefulness"],
        foreground=fg,
        power=b["power"],
        thermal_status=None,  # OS thermal-status API not queried at L1; see STATUS.md
        skin_c_start=None,
        skin_c_end=None,
        battery_pct=b["battery_pct"],
        descheduled=False,
        concurrent_load=load,
    )


def valid_thermal_zones(g0_text: str) -> list:
    """Parse g0_probe.sh's `== thermal zones ==` dump and apply the same
    filter ufsbench.c's max_thermal_c() applies in C: a Linux thermal-sysfs
    `temp` file is documented in millidegrees C, so |raw| < 1000 cannot be a
    temperature -- it is a level or index (e.g. this device's
    `pm7250b-bcl-lvl0` reads a bare 0, `pm7250b-ibat-lvl0` reads -468). Names
    containing "trip", "bcl" or "-lvl" are excluded outright since a
    plausible-looking value there (this device's `cpu-hw-trip-*` = 95000) is
    still not a live temperature. -273000 (absolute zero) marks a disabled
    sensor. Returns [{"name", "temp_c"}] for zones that pass, sorted hottest
    first -- never a single hand-picked zone name.
    """
    import re as _re
    zones = []
    for line in g0_text.splitlines():
        m = _re.match(r"(thermal_zone\d+)\s+(\S*)\s*(-?\d+)?\s*$", line.strip())
        if not m:
            continue
        zone_id, ztype, raw = m.groups()
        if raw is None or ztype in ("", None):
            continue
        raw_v = int(raw)
        if any(bad in ztype for bad in ("trip", "bcl", "-lvl")):
            continue
        if abs(raw_v) < 1000:
            continue  # a level or index, not millidegrees
        temp_c = raw_v / 1000.0
        if not (0.0 <= temp_c <= 120.0):
            continue  # disabled-sensor sentinel or out of plausible range
        zones.append({"name": ztype, "temp_c": temp_c})
    return sorted(zones, key=lambda z: -z["temp_c"])


def reject_descheduled(rows: list, ratio_key: str, threshold: float = 0.90) -> tuple:
    """Split probe rows into (clean, rejected, reasons), per the descheduling
    contamination rule in 04_DEVICE_PROFILING.md section 4: a process that did
    not get its expected CPU time during a fixed-duration probe reports a
    fraction of the true rate and must be discarded, never averaged in.
    """
    clean, rejected, reasons = [], [], []
    for row in rows:
        ratio = float(row.get(ratio_key, 1.0))
        if ratio < threshold:
            rejected.append(row)
            reasons.append(f"descheduled: {ratio_key}={ratio:.3f} < {threshold}")
        else:
            clean.append(row)
    return clean, rejected, reasons
