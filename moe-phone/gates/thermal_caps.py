#!/usr/bin/env python3
"""thermal_caps.py — how the 15R's CPU frequency caps covary with its thermal state, from every phone log.

Every phone run in this project logs one gate line per row/phase of the form
  wake=<Awake|Dozing> status=<Android thermal status 0-3> cap0=<kHz> cap6=<kHz> hw0=<kHz> hw6=<kHz> shell_front_mC=<m°C>
(device/thermal_gate.sh). cap0 / cap6 are scaling_max_freq of the 6-core (policy0) and 2-core prime (policy6)
clusters; hw0 / hw6 their hardware maxima. This script collects all such lines under results/, deduplicates
identical lines per file, and reports the cap distribution per thermal status and per front-temperature band.

What it can and cannot say: it is OBSERVATIONAL. Sustained load raises the temperature and also the time under
load; an OEM module that clamps after sustained load (cpufreq_bouncing, documented on the OnePlus 13) and a
temperature-driven thermal engine both produce this pattern. Separating them needs an intervention (cooling at
equal load; High Performance Mode at equal temperature), not more of these logs.

  python3 gates/thermal_caps.py  ->  results/2026-09-19/thermal_caps.json
"""
import collections
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAT = re.compile(r"wake=(\w+) status=(\d) cap0=(\d+) cap6=(\d+) hw0=(\d+) hw6=(\d+) shell_front_mC=(\d+)")


def main():
    rows = []
    files = 0
    for p in sorted((ROOT / "results").rglob("*")):
        if not p.is_file() or p.suffix not in (".txt", ".log", ".out", ".csv"):
            continue
        try:
            txt = p.read_text(errors="replace")
        except OSError:
            continue
        seen = set()
        hit = False
        for m in PAT.finditer(txt):
            if m.group(0) in seen:
                continue
            seen.add(m.group(0))
            hit = True
            wake, status, c0, c6, h0, h6, t = m.groups()
            rows.append(dict(file=str(p.relative_to(ROOT)), wake=wake, status=int(status), cap0=int(c0) / 1e6,
                             cap6=int(c6) / 1e6, hw0=int(h0) / 1e6, hw6=int(h6) / 1e6, front_c=int(t) / 1000))
        files += hit

    def summ(sel):
        if not sel:
            return None
        fr = [r["front_c"] for r in sel]
        full = sum(1 for r in sel if r["cap0"] >= r["hw0"] and r["cap6"] >= r["hw6"])
        return dict(n=len(sel), front_c_median=st.median(fr), front_c_min=min(fr), front_c_max=max(fr),
                    cap0_median_ghz=st.median(r["cap0"] for r in sel), cap6_median_ghz=st.median(r["cap6"] for r in sel),
                    fraction_at_hardware_max=full / len(sel))

    by_status = {s: summ([r for r in rows if r["status"] == s]) for s in range(4)}
    bands = [(0, 32), (32, 34), (34, 36), (36, 38), (38, 40), (40, 42), (42, 44), (44, 99)]
    by_band = {f"{a}-{b}C": summ([r for r in rows if a <= r["front_c"] < b]) for a, b in bands}
    by_wake = {w: summ([r for r in rows if r["wake"] == w]) for w in ("Awake", "Dozing")}
    full = [r for r in rows if r["cap0"] >= r["hw0"] and r["cap6"] >= r["hw6"]]
    out = dict(
        description=__doc__.strip().splitlines()[0],
        files_with_gate_lines=files, samples=len(rows),
        hardware_max_ghz=dict(policy0=max(r["hw0"] for r in rows), policy6=max(r["hw6"] for r in rows)),
        by_thermal_status=by_status, by_front_temperature=by_band, by_wake=by_wake,
        at_hardware_max=summ(full),
        status0_below_hardware_max=summ([r for r in rows if r["status"] == 0 and not (r["cap0"] >= r["hw0"] and r["cap6"] >= r["hw6"])]),
    )
    p = ROOT / "results/2026-09-19/thermal_caps.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"{len(rows)} gate samples from {files} files -> {p}")
    for s, v in by_status.items():
        if v:
            print(f"status {s}: n={v['n']:4d} front {v['front_c_median']:.1f} C  caps {v['cap0_median_ghz']:.2f}/{v['cap6_median_ghz']:.2f} GHz"
                  f"  at hw max {v['fraction_at_hardware_max']:.0%}")
    for k, v in by_band.items():
        if v:
            print(f"front {k:>7}: n={v['n']:4d} caps {v['cap0_median_ghz']:.2f}/{v['cap6_median_ghz']:.2f} GHz  at hw max {v['fraction_at_hardware_max']:.0%}")


if __name__ == "__main__":
    main()
