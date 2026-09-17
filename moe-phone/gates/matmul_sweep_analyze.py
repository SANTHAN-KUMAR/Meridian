"""
Per-op matmul throughput per device, and what it predicts for a placement decision.

Input: a `results/<date>/matmul_sweep/` tree of `<shape>_<device>_rep<k>.txt` files, each containing one
`RESULT key=value ...` line from host/app/ggml_matmul_bench.cpp (compute ms/GB/s with the weights already
in the device's buffer, upload ms/GB/s for putting them there, and the elementwise agreement with the CPU
backend).

What it computes, per shape:
  speedup_vs_cpu          median device compute time / median CPU compute time, inverted -- >1 means the
                          device computes that shape faster
  upload_over_compute     the device's upload time for the same weights divided by its compute time. For
                          a STREAMED weight this ratio decides everything: a cache miss pays the upload
                          every token, so a device only wins on streamed experts if
                          speedup_vs_cpu > 1 + (1 - hit_rate) * upload_over_compute -- and on
                          ggml-hexagon the upload includes a tiled repack, not a memcpy.
  break_even_hit_rate     the cache hit rate at which moving that shape to the device stops losing,
                          derived from the line above and reported as a fraction (>1 = never wins).
  correct                 false if the device disagreed with the CPU backend; such a shape is reported
                          and excluded from every ratio, never quietly dropped.

For a RESIDENT weight (the attention tensors: uploaded once at load) the upload column is irrelevant and
`speedup_vs_cpu` alone decides. The script therefore prints both readings and does not collapse them.

Predicted end-to-end effect: with --node-share-json (a compute_trace artifact) and --shape-share NAME=FRAC
mappings, the per-shape speedups are combined into the fraction of decode node time saved, which is the
honest bound on what an end-to-end A/B could show.

Run:
  python moe-phone/gates/matmul_sweep_analyze.py results/2026-09-18/matmul_sweep --out-name matmul_sweep.json
"""
import argparse
import collections
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RE_RESULT = re.compile(r"^RESULT (.*)$", re.M)
NUM = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")


def parse_result(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    m = RE_RESULT.search(txt)
    if not m:
        return None
    row = {}
    for kv in m.group(1).split():
        k, _, v = kv.partition("=")
        row[k] = float(v) if NUM.match(v) else v
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("dir")
    p.add_argument("--cpu-device", default="CPU")
    p.add_argument("--hit-rate", type=float, default=None,
                   help="measured expert-cache hit rate (fraction) to evaluate the streamed-weight "
                        "criterion at; without it only break_even_hit_rate is reported")
    p.add_argument("--out-name", required=True)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    rows, failed = [], []
    for fn in sorted(os.listdir(a.dir)):
        if not fn.endswith(".txt") or fn.endswith(".runner.txt"):
            continue
        r = parse_result(os.path.join(a.dir, fn))
        if r is None:
            failed.append(fn)
            continue
        stem = fn[:-4]
        m = re.match(r"(.+)_([A-Za-z0-9]+)_rep(\d+)$", stem)
        r["file"] = fn
        r["shape"] = m.group(1) if m else stem
        r["rep"] = int(m.group(3)) if m else None
        rows.append(r)
    if not rows:
        raise ValueError(f"no RESULT lines under {a.dir} ({len(failed)} files had none)")

    by = collections.defaultdict(list)
    for r in rows:
        by[(r["shape"], r["device"])].append(r)

    cells = []
    for (shape, dev), rs in sorted(by.items()):
        ok = [r for r in rs if r.get("correct") == 1.0]
        e = {"shape": shape, "device": dev, "n": len(rs), "n_incorrect": len(rs) - len(ok)}
        if ok:
            e.update(
                compute_ms_median=statistics.median(r["compute_ms"] for r in ok),
                compute_GBps_median=statistics.median(r["compute_GBps"] for r in ok),
                upload_ms_median=statistics.median(r["upload_ms"] for r in ok),
                upload_GBps_median=statistics.median(r["upload_GBps"] for r in ok),
                read_MB=ok[0]["read_MB"], weight_MB=ok[0]["weight_MB"],
                max_rel_err_max=max(r["max_rel_err"] for r in ok),
            )
        cells.append(e)

    cpu = {c["shape"]: c for c in cells if c["device"] == a.cpu_device and "compute_ms_median" in c}
    for c in cells:
        if c["device"] == a.cpu_device or "compute_ms_median" not in c:
            continue
        ref = cpu.get(c["shape"])
        if not ref:
            c["note"] = f"no {a.cpu_device} row for this shape: no ratio computable"
            continue
        c["speedup_vs_cpu"] = ref["compute_ms_median"] / c["compute_ms_median"]
        c["upload_over_compute"] = c["upload_ms_median"] / c["compute_ms_median"]
        # Streamed-weight criterion. Per token a fraction (1-h) of the weights must be uploaded, and the
        # compute is c.compute; the CPU alternative costs ref.compute and uploads nothing (the streaming
        # engine's own read is already counted in its stall). Device wins iff
        #   c.compute + (1-h)*c.upload < ref.compute
        # => h > 1 - (ref.compute - c.compute) / c.upload
        gain = ref["compute_ms_median"] - c["compute_ms_median"]
        c["break_even_hit_rate"] = 1.0 - gain / c["upload_ms_median"] if c["upload_ms_median"] > 0 else None
        if c["break_even_hit_rate"] is not None and c["break_even_hit_rate"] > 1.0:
            c["streamed_verdict"] = "never wins at any hit rate (upload alone exceeds the compute saved)"
        elif a.hit_rate is not None:
            wins = a.hit_rate > c["break_even_hit_rate"]
            c["streamed_verdict"] = (f"{'wins' if wins else 'loses'} at the measured hit rate "
                                     f"{a.hit_rate:.3f} (break-even {c['break_even_hit_rate']:.3f})")

    out = {"source": os.path.abspath(a.dir), "cpu_device": a.cpu_device, "hit_rate": a.hit_rate,
           "files_without_result": failed, "rows": rows, "cells": cells}

    for c in cells:
        if "compute_ms_median" not in c:
            print(f"{c['shape']:<14} {c['device']:<10} NO USABLE ROW "
                  f"(n={c['n']}, incorrect={c['n_incorrect']})")
            continue
        line = (f"{c['shape']:<14} {c['device']:<10} compute {c['compute_ms_median']:8.4f} ms "
                f"({c['compute_GBps_median']:6.2f} GB/s)  upload {c['upload_ms_median']:9.4f} ms "
                f"({c['upload_GBps_median']:6.2f} GB/s)")
        if "speedup_vs_cpu" in c:
            line += f"  {c['speedup_vs_cpu']:5.2f}x vs CPU"
            if c.get("break_even_hit_rate") is not None:
                line += f"  streamed break-even hit {c['break_even_hit_rate']:.3f}"
        if c["n_incorrect"]:
            line += f"  [{c['n_incorrect']}/{c['n']} INCORRECT, excluded]"
        print(line)
    if failed:
        print(f"{len(failed)} files carried no RESULT line: {failed[:6]}")

    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        path = os.path.join(a.out_dir, a.out_name)
    else:
        from _paths import write_path
        path = write_path(a.out_name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
