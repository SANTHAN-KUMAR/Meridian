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
            if line.startswith("BENCH ") and "median_ms=" in line:
                kv = dict(t.split("=", 1) for t in line.split()[1:] if "=" in t)
                num = lambda k: float(kv[k]) if k in kv else None
                bench.append(dict(spin=int(kv.get("spin", 0)), variant=int(kv["variant"]), down=kv["down"], k=int(kv["k"]),
                                  n=int(kv["n"]), median_ms=num("median_ms"), p10_ms=num("p10_ms"), p90_ms=num("p90_ms"),
                                  GBps=num("GBps_at_median"), device_median_ms=num("device_median_ms"),
                                  device_p90_ms=num("device_p90_ms")))
            m = re.match(r"SLOTWRITE variant=(\d) n=(\d+) median_ms=([\d.]+) p90_ms=([\d.]+)", line)
            if m:
                out.setdefault("slotwrite", []).append(dict(variant=int(m.group(1)), median_ms=float(m.group(3)),
                                                            p90_ms=float(m.group(4))))
    out["bench"] = bench
    best = {}
    for b in bench:
        key = f"{b['down']}_k{b['k']}"
        if key not in best or b["median_ms"] < best[key]["median_ms"]:
            best[key] = {"variant": b["variant"], "spin": b["spin"], "median_ms": b["median_ms"], "GBps": b["GBps"]}
    out["fastest_variant"] = best
    # least-squares line median_ms = a + b*k per (variant, down, host|device): a = fixed per-dispatch overhead
    fits = {}
    for sp, v in sorted({(b["spin"], b["variant"]) for b in bench}):
        for dn in sorted({b["down"] for b in bench}):
            rows = [b for b in bench if b["variant"] == v and b["down"] == dn and b["spin"] == sp]
            for kind, key in (("host", "median_ms"), ("device", "device_median_ms")):
                pts = [(b["k"], b[key]) for b in rows if b[key] is not None]
                if len(pts) >= 3:
                    n = len(pts); sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
                    sxx = sum(p[0] ** 2 for p in pts); sxy = sum(p[0] * p[1] for p in pts)
                    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx); icpt = (sy - slope * sx) / n
                    fits[f"spin{sp}_v{v}_{dn}_{kind}"] = {"intercept_ms": icpt, "per_expert_ms": slope, "n_k": n}
    out["latency_fit"] = fits
    p = os.path.join(R, "m6_summary.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("verdict", "detector", "premise", "swiglu")}, indent=1))
    for c, d in out.get("swiglu_mismatch_classes", {}).items():
        print(f"swiglu mismatch class {c:26s} n={d['n']:6d} flagged={d['flagged']}")
    for b in bench:
        dv = f", device {b['device_median_ms']:.3f} ms" if b.get("device_median_ms") is not None else ""
        print(f"spin{b['spin']} v{b['variant']} {b['down']} k={b['k']}: host median {b['median_ms']:.3f} ms, p90 {b['p90_ms']:.3f} ms, "
              f"{b['GBps']:.2f} GB/s{dv}")
    for key, f in fits.items():
        print(f"fit {key}: intercept {f['intercept_ms']:.3f} ms + {f['per_expert_ms']:.3f} ms/expert")
    print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
