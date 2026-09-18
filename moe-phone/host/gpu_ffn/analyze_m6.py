"""
analyze_m6.py — summarise one gx phone run (run_phone_m6.sh) into m6_summary.json beside it.

  python3 host/gpu_ffn/analyze_m6.py results/<date>/gx_m6_<time>

Reports, all read from the run's own files (nothing tuned, nothing typed in):
  - VERDICT and the gx_test SUMMARY / DETECTOR / PREMISE / SWIGLU lines
  - SwiGLU mismatches classified from swiglu_mismatch.csv: denormal input (unreachable in a dispatch),
    |gate| > 80, reference result below FLT_MIN, other; and how many of each the detector flagged
  - the bench (if the gate allowed it): median / p90 ms and GB/s per variant, down type and k, and which
    variant is faster per (down type, k)
"""
import csv
import json
import os
import re
import struct
import sys

FLT_MIN = 1.1754943508222875e-38


def f32(x):
    return struct.unpack("<f", struct.pack("<f", float.fromhex(x) if isinstance(x, str) else x))[0]


def is_denormal(v):
    return v != 0.0 and abs(v) < FLT_MIN


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    R = sys.argv[1]
    out = {"run": os.path.basename(os.path.abspath(R))}
    vpath = os.path.join(R, "VERDICT")
    out["verdict"] = open(vpath).read().strip() if os.path.exists(vpath) else None
    txt = open(os.path.join(R, "gx_test.out"), errors="replace").read() if os.path.exists(os.path.join(R, "gx_test.out")) else ""
    for tag in ("DEVICE", "DIV", "QUANT", "SWIGLU", "PREMISE", "DETECTOR", "SUMMARY"):
        m = re.search(rf"^{tag} .*$", txt, re.M)
        out[tag.lower()] = m.group(0) if m else None
    # per-case failures
    fails = []
    for line in re.findall(r"^CASE .*$", txt, re.M):
        m = re.search(r"CASE (\S+)\s+v(\d) down=(\S+) k=(\d+) .*down_diff_bits=(\d+)/(\d+).*h_diff_bits=(\d+)/(\d+)", line)
        if m and (int(m.group(5)) or int(m.group(7))):
            fails.append(dict(case=m.group(1), variant=int(m.group(2)), down=m.group(3), k=int(m.group(4)),
                              down_diff=int(m.group(5)), h_diff=int(m.group(7))))
    out["case_failures"] = fails
    # SwiGLU mismatch classes
    cpath = os.path.join(R, "swiglu_mismatch.csv")
    if os.path.exists(cpath):
        cls = {}
        for r in csv.DictReader(open(cpath)):
            g, u, ref = (float.fromhex(r[k]) for k in ("gate", "up", "ref"))
            if is_denormal(g) or is_denormal(u):
                c = "denormal_input"
            elif abs(g) > 80:
                c = "abs_gate_gt_80"
            elif ref == 0.0 or abs(ref) < FLT_MIN:
                c = "ref_result_below_FLT_MIN"
            else:
                c = "other"
            d = cls.setdefault(c, {"n": 0, "flagged": 0})
            d["n"] += 1
            d["flagged"] += int(r["flagged"])
        out["swiglu_mismatch_classes"] = cls
    # bench
    bench = []
    bpath = os.path.join(R, "bench.out")
    if os.path.exists(bpath):
        for line in open(bpath):
            m = re.match(r"BENCH variant=(\d) down=(\S+) k=(\d+) n=(\d+) median_ms=([\d.]+) p10_ms=([\d.]+) p90_ms=([\d.]+) GBps_at_median=([\d.]+)", line)
            if m:
                bench.append(dict(variant=int(m.group(1)), down=m.group(2), k=int(m.group(3)), n=int(m.group(4)),
                                  median_ms=float(m.group(5)), p10_ms=float(m.group(6)), p90_ms=float(m.group(7)),
                                  GBps=float(m.group(8))))
            m = re.match(r"SLOTWRITE variant=(\d) n=(\d+) median_ms=([\d.]+) p90_ms=([\d.]+)", line)
            if m:
                out.setdefault("slotwrite", []).append(dict(variant=int(m.group(1)), median_ms=float(m.group(3)),
                                                            p90_ms=float(m.group(4))))
    out["bench"] = bench
    best = {}
    for b in bench:
        key = f"{b['down']}_k{b['k']}"
        if key not in best or b["median_ms"] < best[key]["median_ms"]:
            best[key] = {"variant": b["variant"], "median_ms": b["median_ms"], "GBps": b["GBps"]}
    out["fastest_variant"] = best
    p = os.path.join(R, "m6_summary.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("verdict", "detector", "premise", "swiglu")}, indent=1))
    for c, d in out.get("swiglu_mismatch_classes", {}).items():
        print(f"swiglu mismatch class {c:26s} n={d['n']:6d} flagged={d['flagged']}")
    for b in bench:
        print(f"v{b['variant']} {b['down']} k={b['k']}: median {b['median_ms']:.3f} ms, p90 {b['p90_ms']:.3f} ms, {b['GBps']:.2f} GB/s")
    print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
