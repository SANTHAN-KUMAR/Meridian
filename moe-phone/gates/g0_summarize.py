"""
G0 summary — turn device/g0_probe.sh's text output into a machine-readable
artifact, so that no device fact is ever hand-transcribed into a document
(CLAUDE.md section 7.1).

This script PARSES; it does not judge. Where the probe could not read something
it records *that*, with the probe's own wording, because on an unrooted phone
the set of facts an ordinary app cannot see is itself a result (POSITION.md F6:
"beyond-DRAM inference as an unprivileged app").

Three outcomes are kept distinct throughout, matching the probe's `try`:
  DENIED_OR_ABSENT   a sysfs read returned nothing
  FAILED rc=N        the command exited non-zero
  EMPTY rc=0         the command ran and matched nothing

Run:
  python moe-phone/gates/g0_summarize.py moe-phone/results/<date>/g0_15r.txt         --memprobe moe-phone/results/<date>/memprobe_15r*.csv

Pass EVERY memprobe repeat. The resident ceiling is a memory-pressure effect,
not a device constant, so its run-to-run spread is part of the result.
"""
import argparse
import json
import os
import re
import statistics
import sys

# Which /sys/class/thermal entries are temperatures at all. This must stay
# identical to max_thermal_c() in device/ufsbench.c and to the band in
# gates/g1_analyze.py; if the three disagree, the "max temperature" of a run
# depends on which script you ask.
#
# Two things masquerade as temperatures on the OnePlus 15R:
#   - threshold and LEVEL pseudo-zones. `cpu-hw-trip-*` reads a constant 95000;
#     `*-bcl-*` and `*-lvl*` are battery current-limit levels (0, 1, 2, -843);
#     and `socd` reads a bare 75-77 (state-of-charge depletion), which is both
#     name-innocent and numerically plausible as degrees Celsius.
#   - disabled sensors, reporting the -273000 absolute-zero sentinel.
#
# The load-bearing rule is not the name list but the UNIT: Linux thermal sysfs
# defines `temp` in millidegrees Celsius, so any |raw| < 1000 is below 1 mC and
# is an index or a level, not a die temperature. That alone rejects `socd` and
# every *-lvl zone without knowing their names. The name filter is then needed
# only for `trip`, whose 95000 is a plausible millidegree value.
NON_SENSOR_ZONE = re.compile(r"trip|bcl|-lvl")
MILLIDEGREE_MIN_RAW = 1000      # |raw| below this is a level, not a temperature
TEMP_MIN_C, TEMP_MAX_C = 0.0, 120.0

PROP_KEYS = ("ro.product.manufacturer", "ro.product.model", "ro.soc.manufacturer",
             "ro.soc.model", "ro.board.platform", "ro.build.version.release",
             "ro.build.version.sdk", "ro.build.fingerprint",
             "ro.boot.verifiedbootstate", "ro.boot.flash.locked", "ro.secure",
             "dalvik.vm.heapgrowthlimit")
MEM_KEYS = ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree", "Cached")


def split_sections(text):
    """== name == delimited blocks, in order."""
    out, name, buf = {}, "preamble", []
    for line in text.splitlines():
        m = re.match(r"^== (.+) ==$", line)
        if m:
            out[name] = buf
            name, buf = m.group(1), []
        else:
            buf.append(line)
    out[name] = buf
    return out


def parse(text):
    sec = split_sections(text)
    flat = text.splitlines()
    res = {}

    props = {}
    for line in flat:
        for k in PROP_KEYS:
            if line.startswith(k + "="):
                props[k] = line.split("=", 1)[1].strip()
    res["props"] = props

    kern = [l for l in sec.get("identity", []) if l.startswith("Linux ")]
    res["kernel"] = kern[0] if kern else None
    ctx = [l for l in sec.get("identity", []) if l.startswith("context: ")]
    res["process_context"] = ctx[0][len("context: "):] if ctx else None
    m = re.search(r"context=(\S+)", res["process_context"] or "")
    res["selinux_domain"] = m.group(1) if m else None
    m = re.search(r"^probe_epoch_s: (\d+)", text, re.M)
    res["probe_epoch_s"] = int(m.group(1)) if m else None

    mem = {}
    for line in sec.get("memory", []):
        for k in MEM_KEYS:
            m = re.match(rf"^{k}:\s+(\d+) kB", line)
            if m:
                mem[k + "_kB"] = int(m.group(1))
        if line.startswith("oom_score_adj="):
            mem["oom_score_adj"] = line.split("=", 1)[1].strip()
        if line.startswith("zram0_disksize_bytes="):
            mem["zram0_disksize_bytes"] = line.split("=", 1)[1].strip()
    # Derived, and named so the name is checkable against the value
    # (CLAUDE.md section 6.5). GiB, because /proc/meminfo is binary kB.
    if "MemTotal_kB" in mem:
        mem["MemTotal_GiB"] = round(mem["MemTotal_kB"] / 1048576, 3)
    if "SwapTotal_kB" in mem:
        mem["SwapTotal_GiB"] = round(mem["SwapTotal_kB"] / 1048576, 3)
    res["memory"] = mem

    sto = {"queue_params": {}, "mounts": []}
    for line in sec.get("storage", []):
        if line.startswith("data_device="):
            sto["data_device"] = line.split("=", 1)[1].strip()
        elif line.startswith("data_mount="):
            sto["data_mount"] = line.split("=", 1)[1].strip()
        elif line.startswith("/sys/block/"):
            k, _, v = line.partition("=")
            sto["queue_params"][k] = v.strip()
        elif re.match(r"^/dev/\S+ /\S+ \S+ ", line):
            dev, mnt, fs, opts = line.split()[:4]
            sto["mounts"].append({"device": dev, "mount": mnt, "fs": fs,
                                  "options": opts.split(",")})
    sto["queue_params_readable"] = sum(
        1 for v in sto["queue_params"].values() if v != "DENIED_OR_ABSENT")
    sto["queue_params_denied"] = sum(
        1 for v in sto["queue_params"].values() if v == "DENIED_OR_ABSENT")
    data_fs = [m for m in sto["mounts"] if m["mount"] in ("/data", "/data/user/0")]
    sto["data_fs"] = data_fs[0]["fs"] if data_fs else None
    res["storage"] = sto

    npu = {"libs_present": [], "libs_absent": []}
    for line in sec.get("npu (Hexagon) presence", []):
        if line.startswith("present: "):
            npu["libs_present"].append(line[len("present: "):])
        elif line.startswith("absent: "):
            npu["libs_absent"].append(line[len("absent: "):])
    npu["fastrpc_dev_nodes_listable"] = not any(
        "fastrpc" in l and l.startswith("FAILED")
        for l in sec.get("npu (Hexagon) presence", []))
    res["npu"] = npu

    zones, excluded = [], []
    for line in sec.get("thermal zones", []):
        m = re.match(r"^(thermal_zone\d+) (\S+) (-?\d+)$", line)
        if not m:
            continue
        name, typ, raw = m.group(1), m.group(2), int(m.group(3))
        rec = {"zone": name, "type": typ, "raw": raw}
        if NON_SENSOR_ZONE.search(typ):
            excluded.append({**rec, "why": "threshold_or_level_name"})
            continue
        if abs(raw) < MILLIDEGREE_MIN_RAW:
            excluded.append({**rec, "why": "not_millidegrees"})
            continue
        c = raw / 1000.0
        if not TEMP_MIN_C <= c <= TEMP_MAX_C:
            excluded.append({**rec, "why": "outside_physical_band", "temp_c": c})
            continue
        zones.append({**rec, "temp_c": c})
    hottest = max(zones, key=lambda z: z["temp_c"], default=None)
    by_reason = {}
    for z in excluded:
        by_reason.setdefault(z["why"], []).append(z["type"])
    res["thermal"] = {
        "n_zones_total": len(zones) + len(excluded),
        "n_zones_sensor": len(zones),
        "n_zones_excluded": len(excluded),
        "excluded_by_reason": {k: sorted(set(v)) for k, v in sorted(by_reason.items())},
        "max_sensor_c": hottest["temp_c"] if hottest else None,
        "max_sensor_zone": hottest["type"] if hottest else None,
        "nsp_zones": sorted(z["type"] for z in zones if "nsph" in z["type"]),
        "zones": zones,
    }

    cpu = {"max_khz": {}}
    for line in sec.get("cpu", []):
        m = re.match(r"^(cpu\d+) max_khz=(\d+)$", line)
        if m:
            cpu["max_khz"][m.group(1)] = int(m.group(2))
    cpu["n_cores"] = len(cpu["max_khz"])
    # Clusters as distinct max frequencies, not as an assumed big.LITTLE shape.
    cpu["clusters"] = [{"max_khz": f, "n": list(cpu["max_khz"].values()).count(f)}
                       for f in sorted(set(cpu["max_khz"].values()), reverse=True)]
    res["cpu"] = cpu

    power = {}
    for line in sec.get("power measurement", []):
        if line.startswith("battery/"):
            k, _, v = line.partition("=")
            power[k] = v.strip()
    power["any_rail_readable"] = any(
        v != "DENIED_OR_ABSENT" for k, v in power.items() if k.startswith("battery/"))
    power["dumpsys_powerstats_available"] = not any(
        l.startswith("FAILED") and "dumpsys powerstats" in l
        for l in sec.get("power measurement", []))
    res["power"] = power

    gpu = [l.split("=", 1)[1] for l in sec.get("gpu", []) if l.startswith("gpu_model=")]
    res["gpu_model"] = gpu[0] if gpu else None

    root = {"rooted": None, "evidence": None}
    for line in sec.get("root", []):
        if line.startswith("ROOT:"):
            root = {"rooted": True, "evidence": line}
        elif line.startswith("NOT_ROOT:"):
            root = {"rooted": False, "evidence": line}
    root["verifiedbootstate"] = props.get("ro.boot.verifiedbootstate")
    root["bootloader_locked"] = props.get("ro.boot.flash.locked")
    res["root"] = root

    res["probe_outcomes"] = {
        "n_failed": sum(1 for l in flat if l.startswith("FAILED ")),
        "n_empty": sum(1 for l in flat if l.startswith("EMPTY ")),
        "n_denied_or_absent": sum(1 for l in flat if l.endswith("=DENIED_OR_ABSENT")),
        "failed_commands": [l for l in flat if l.startswith("FAILED ")],
        "denied_keys": [l.split("=")[0] for l in flat if l.endswith("=DENIED_OR_ABSENT")],
    }
    return res


def parse_memprobe(path):
    """device/memprobe.c output: the measured memory operating point.

    The operating point is `max_VmRSS_MB` — bytes the process kept RESIDENT in
    DRAM — not the last `held_MB`, which also counts pages the kernel has
    already compressed into zram. On a phone with a large zram swap the two
    differ by more than 40%, and only the resident figure is memory an engine
    can read at DRAM speed. See memprobe.c's header.
    """
    rows, header, meta = [], None, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "=" in line and "," not in line:
                meta.update(p.split("=", 1) for p in line.split() if "=" in p)
            elif line.startswith("held_MB,"):
                header = line.split(",")
            elif header and "," in line and line[0].isdigit():
                rows.append(dict(zip(header, line.split(","))))
    if not rows:
        return {"error": f"no data rows in {path}"}
    if "max_VmRSS_MB" not in meta:
        return {"error": f"{path} has no max_VmRSS_MB line — it is from a memprobe "
                         f"older than 2026-09-14, which reported allocated rather than "
                         f"resident memory. Rebuild memprobe and re-run.",
                "meta": meta, "n_chunks": len(rows)}
    last = rows[-1]
    rss_mb = int(meta["max_VmRSS_MB"])
    return {
        # Anonymous vs file-backed: different quantities, so the artifact says which.
        "mode": meta.get("mode", "anonymous"),
        "oom_score_adj": meta.get("oom_score_adj"),
        "MemTotal_MB": meta.get("MemTotal_MB"), "SwapTotal_MB": meta.get("SwapTotal_MB"),
        "n_chunks": len(rows),
        "max_VmRSS_MB": rss_mb,
        # GB is decimal, because byte_budget.py's --ram-gb is decimal.
        # Converting in one place stops the two drifting.
        "resident_GiB": round(rss_mb / 1024, 3),
        "resident_GB_decimal": round(rss_mb * 1048576 / 1e9, 3),
        "last_held_MB": int(last["held_MB"]),
        "VmSwap_MB_at_stop": int(last["VmSwap_MB"]) if "VmSwap_MB" in last else None,
        "MemAvailable_MB_at_stop": int(last["MemAvailable_MB"]),
        "SwapFree_MB_at_stop": int(last["SwapFree_MB"]),
        "stop_reason": last["stop_reason"] or "none",
        "first_chunk_s": float(rows[0]["chunk_s"]),
        "last_chunk_s": float(last["chunk_s"]),
        "rows": rows,
    }


def summarise_memprobe_runs(runs):
    """Median and full range over memprobe repeats, grouped by regime.

    The resident ceiling is not a device constant: it is where zram pressure
    stops the process growing, so it moves with whatever else is running. One
    run is an anecdote. The median is what feeds --ram-gb; min and max are
    reported beside it so nobody quotes the median as if it were exact.
    """
    usable = [r for r in runs if "error" not in r]
    out = {"n_runs": len(runs), "n_usable": len(usable),
           "errors": [r["error"] for r in runs if "error" in r], "by_mode": {}}
    for mode in sorted({r.get("mode", "anonymous") for r in usable}):
        rs = [r for r in usable if r.get("mode", "anonymous") == mode]
        vals = sorted(r["max_VmRSS_MB"] for r in rs)
        med = statistics.median(vals)
        out["by_mode"][mode] = {
            "n": len(vals), "max_VmRSS_MB_median": med,
            "max_VmRSS_MB_min": vals[0], "max_VmRSS_MB_max": vals[-1],
            "max_VmRSS_MB_all": vals,
            "rel_spread": (vals[-1] - vals[0]) / med if med else None,
            "resident_GB_decimal_median": round(med * 1048576 / 1e9, 3),
            "stop_reasons": sorted({r["stop_reason"] for r in rs}),
        }
    return out


def main():
    p = argparse.ArgumentParser(description="G0 device-facts summary")
    p.add_argument("probe_txt")
    p.add_argument("--memprobe", nargs="*", default=[],
                   help="one or more memprobe CSVs. Pass every repeat: the resident ceiling is "
                        "a memory-pressure effect, so its run-to-run spread is part of the result "
                        "(ESTIMAND.md section 1) and a single run is not one.")
    a = p.parse_args()

    with open(a.probe_txt, encoding="utf-8", errors="replace") as f:
        res = parse(f.read())
    res["source_probe_txt"] = os.path.abspath(a.probe_txt)
    if a.memprobe:
        res["memprobe_runs"] = [dict(parse_memprobe(m), source=os.path.abspath(m))
                                for m in a.memprobe]
        res["memprobe"] = summarise_memprobe_runs(res["memprobe_runs"])

    pr, mem, sto = res["props"], res["memory"], res["storage"]
    print(f"device : {pr.get('ro.product.manufacturer')} {pr.get('ro.product.model')} "
          f"({pr.get('ro.soc.model')}, {pr.get('ro.board.platform')}), "
          f"Android {pr.get('ro.build.version.release')} / SDK {pr.get('ro.build.version.sdk')}")
    print(f"context: {res['selinux_domain']}  oom_score_adj={mem.get('oom_score_adj')}")
    print(f"memory : MemTotal {mem.get('MemTotal_GiB')} GiB, "
          f"SwapTotal(zram) {mem.get('SwapTotal_GiB')} GiB")
    print(f"storage: {sto.get('data_device')} {sto.get('data_fs')} at {sto.get('data_mount')}; "
          f"queue params readable {sto['queue_params_readable']}/"
          f"{sto['queue_params_readable'] + sto['queue_params_denied']}")
    print(f"npu    : libs {len(res['npu']['libs_present'])} present; "
          f"NSP thermal zones {res['thermal']['nsp_zones'] or 'none'}")
    print(f"gpu    : {res['gpu_model']}")
    th = res["thermal"]
    print(f"thermal: {th['n_zones_sensor']} sensor zones, {th['n_zones_excluded']} excluded "
          f"({'; '.join(f'{k}: {len(v)}' for k, v in th['excluded_by_reason'].items()) or 'none'}); "
          f"max {th['max_sensor_c']} C at {th['max_sensor_zone']}")
    print(f"cpu    : {res['cpu']['n_cores']} cores, clusters "
          f"{[(c['n'], c['max_khz']) for c in res['cpu']['clusters']]}")
    print(f"power  : any battery rail readable = {res['power']['any_rail_readable']}; "
          f"dumpsys powerstats available = {res['power']['dumpsys_powerstats_available']}")
    print(f"root   : {res['root']['evidence']} "
          f"(verifiedboot={res['root']['verifiedbootstate']}, "
          f"bootloader_locked={res['root']['bootloader_locked']})")
    po = res["probe_outcomes"]
    print(f"probe  : {po['n_failed']} FAILED, {po['n_empty']} EMPTY, "
          f"{po['n_denied_or_absent']} DENIED_OR_ABSENT")
    if "memprobe" in res:
        mp = res["memprobe"]
        for err in mp["errors"]:
            print(f"memprobe: NOT USABLE — {err}")
        for mode, s in mp["by_mode"].items():
            print(f"memprobe [{mode}]: max RESIDENT median {s['max_VmRSS_MB_median']:.0f} MB "
                  f"= {s['resident_GB_decimal_median']} GB over n={s['n']} "
                  f"(range {s['max_VmRSS_MB_min']}-{s['max_VmRSS_MB_max']} MB, "
                  f"spread {100 * (s['rel_spread'] or 0):.0f}%); stop {', '.join(s['stop_reasons'])}")
        for mode in ("file_backed", "anonymous"):
            if mode in mp["by_mode"]:
                gb = mp["by_mode"][mode]["resident_GB_decimal_median"]
                print(f"\n  --ram-gb {gb:.2f}   <- measured ({mode} pages), for byte_budget.py")
                break

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _paths import write_path
    out = write_path("g0_device.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
