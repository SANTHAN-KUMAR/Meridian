"""
analyze.py — apply NOTE.md's pre-registered decision rule to one zcbench output.

Reads the `RESULT` and `CLOCKS` lines zcbench prints, writes zcbench.json beside the input, and prints
the verdict with the condition that decided it. Nothing here is tuned: the four conditions and their
thresholds are copied from NOTE.md, where they were fixed before any phone row existed.

  python3 host/gpu_zerocopy/analyze.py results/<date>/zcbench/zcbench.out
"""
import json
import os
import shlex
import sys

ZC_SPEED_OVER_MEMCPY = 10.0   # condition 1: whole-region sync >= 10x memcpy rate
GPU_ERR_MAX = 1e-3            # condition 2
AGG_OVER_CPU = 1.5            # condition 3
OVERLAP_MIN = 0.8             # condition 4
ZC_MECHS = ("use_host_ptr", "ion_dmabuf", "alloc_host_ptr")
SAMPLES = {}


def parse_samples(path):
    """CLKS lines (one per second, from zcbench's sampler) grouped by the phase they fell in."""
    by = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        if not line.startswith("CLKS "):
            continue
        kv = dict(t.split("=", 1) for t in line.split()[1:] if "=" in t)
        by.setdefault(kv.get("phase", "?"), []).append(kv)
    out = {}
    for ph, rows in by.items():
        def col(k):
            v = sorted(int(r[k]) for r in rows if r.get(k, "NA").isdigit())
            return {"min": v[0], "median": v[len(v) // 2], "max": v[-1]} if v else None
        out[ph] = {"n": len(rows), **{k: col(k) for k in ("cap0", "cur0", "cap6", "cur6", "gpu_cur")}}
    return out


def parse(path):
    results, clocks = [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if line.startswith("RESULT ") or line.startswith("CLOCKS "):
            kv = {}
            for tok in shlex.split(line.split(" ", 1)[1]):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try:
                        v = float(v) if v.lower() not in ("nan", "inf") else float(v)
                    except ValueError:
                        pass
                    kv[k] = v
            (results if line.startswith("RESULT") else clocks).append(kv)
    return results, clocks


def verdict(results, clocks):
    zc = [r for r in results if r.get("phase") == "zc" and r.get("status") == "ok"]
    unavailable = {r["mech"]: r.get("why") for r in results if r.get("phase") == "zc" and r.get("status") == "unavailable"}
    good_zc = [r for r in zc if r.get("mech") in ZC_MECHS and r.get("zero_copy") == 1 and r.get("correct") == 1
               and isinstance(r.get("sync_region_GBps"), float) and isinstance(r.get("memcpy_region_GBps"), float)
               and r["sync_region_GBps"] >= ZC_SPEED_OVER_MEMCPY * r["memcpy_region_GBps"]]
    conc = next((r for r in results if r.get("phase") == "concurrent"), None)
    lanes = [r for r in results if r.get("phase") == "lanes"]
    gpu_alone = [r for r in results if r.get("phase") == "gpu_alone"]
    caps = {c["tag"]: {k: v for k, v in c.items() if k.startswith("cap")} for c in clocks if "tag" in c}
    cmp_tags = ("cpu_alone_1", "concurrent", "cpu_alone_2")
    samples = SAMPLES
    if all(t in samples for t in cmp_tags):
        # the per-second samples decide: the median cap of each CPU policy must match across the phases
        caps_same = all(len({json.dumps(samples[t][k]["median"] if samples[t][k] else None) for t in cmp_tags}) == 1
                        for k in ("cap0", "cap6"))
    else:
        caps_same = len({json.dumps(caps.get(t), sort_keys=True) for t in cmp_tags if t in caps}) <= 1

    c1 = bool(good_zc)
    mech_c = conc.get("mech") if conc else None
    zc_row = next((r for r in zc if r.get("mech") == mech_c), None)
    c2 = bool(zc_row) and zc_row.get("err_initial", 1) < GPU_ERR_MAX
    c3 = bool(conc) and conc.get("ratio_vs_cpu_alone", 0) >= AGG_OVER_CPU
    c4 = bool(conc) and conc.get("overlap_frac_of_cpu_busy", 0) >= OVERLAP_MIN
    failed = [n for n, ok in (("1 zero-copy", c1), ("2 correct", c2), ("3 aggregate >= 1.5x CPU", c3),
                              ("4 proven overlap", c4)) if not ok]
    return {
        "viable": not failed and caps_same,
        "failed_conditions": failed + ([] if caps_same else ["caps changed across compared phases"]),
        "zero_copy_mechanisms_passing": [r["mech"] for r in good_zc],
        "zc_rows": zc, "unavailable": unavailable, "lanes": lanes, "gpu_alone": gpu_alone,
        "concurrent": conc, "clocks": caps, "clock_samples_by_phase": SAMPLES,
        "caps_same_across_compared_phases": caps_same,
        "thresholds": {"zc_speed_over_memcpy": ZC_SPEED_OVER_MEMCPY, "gpu_err_max": GPU_ERR_MAX,
                       "agg_over_cpu": AGG_OVER_CPU, "overlap_min": OVERLAP_MIN},
    }


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    res, clk = parse(sys.argv[1])
    global SAMPLES
    SAMPLES = parse_samples(sys.argv[1])
    v = verdict(res, clk)
    out = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "zcbench.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(v, f, indent=1)
    for r in v["zc_rows"]:
        print(f"zc {r['mech']:<15} zero_copy={r.get('zero_copy')} correct={r.get('correct')} "
              f"sync {r.get('sync_region_GBps')} GB/s vs memcpy {r.get('memcpy_region_GBps')} GB/s")
    for m, why in v["unavailable"].items():
        print(f"zc {m:<15} unavailable: {why}")
    for r in v["gpu_alone"]:
        print(f"gpu {r['mech']:<15} batch={int(r['batch'])} {r['GBps']} GB/s, {r['ms_per_layer']} ms/layer")
    c = v["concurrent"]
    if c:
        print(f"concurrent ({c['mech']}): cpu {c['cpu_GBps']} + gpu {c['gpu_GBps']} = {c['aggregate_GBps']} GB/s; "
              f"cpu alone {c['cpu_alone_GBps']}; ratio {c['ratio_vs_cpu_alone']}; overlap {c['overlap_frac_of_cpu_busy']}")
    print("VERDICT:", "viable" if v["viable"] else "NOT viable - failed: " + ", ".join(v["failed_conditions"]))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
