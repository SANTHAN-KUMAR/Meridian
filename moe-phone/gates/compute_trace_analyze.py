"""
Where a streamed decode's time goes, from BigMoeOnEdge's own graph tracing (device/bmoe_trace.sh).

simpleperf cannot run on this phone (the retail kernel exposes no perf events at all, hardware or software,
even with security.perf_harden=0 -- results/2026-09-17/bmoe_profile2), so the attribution comes from the
engine's tracing instead:

  --compute-trace-layers   one barrier per layer, so read/compute overlap and expert prefetch survive and
                           the total stays close to an untraced run. Rows: op=LAYER, name=blk.<il> or
                           pre/post. This is the trustworthy source for PER-LAYER and OUTPUT-HEAD cost.
  --compute-trace          every graph node isolated and timed. The graph is serialised, so absolute times
                           are inflated and only PROPORTIONS BETWEEN OPS may be read from it.

Both files carry `phase` (0 = prefill, 1 = decode); only decode rows are used. wall_ns is per row, and the
number of decoded tokens comes from the row count of the `post` (output head) node, which runs once per
token.

Reported:
  per_layer_ms_per_token        median over blk.* rows
  output_head_ms_per_token      the `post` row (the 151936-row output projection)
  total_traced_ms_per_token     sum of all decode rows / tokens
  node_op_share                 fraction of decode node time per op, from the serialised trace
  bytes_floor_ms_per_token      what the per-token active bytes alone would cost at the measured
                                non-flash rate (S11: 23.2 GB/s), for comparison with the traced cost --
                                the gap is barrier/scheduling/stall overhead, not arithmetic.

Run:
  python moe-phone/gates/compute_trace_analyze.py results/2026-09-17/bmoe_trace --out-name compute_trace.json
"""
import argparse
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NONFLASH_GBPS = 23.2  # S11, measured: resident bytes consumed per second by this SoC at 4 threads


def rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            break
        yield from csv.DictReader(f, fieldnames=["turn", "phase", "step", "seq", "layer", "op", "name",
                                                 "wall_ns", "majflt"])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("dir", help="a bmoe_trace_* directory (trace_layers.csv, trace_nodes.csv)")
    p.add_argument("--active-mb", type=float, default=1800.0,
                   help="active bytes per token in MiB (dense + top-k experts at Q4_0)")
    p.add_argument("--out-name", required=True)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    out = {"source": os.path.abspath(a.dir), "nonflash_gbps": NONFLASH_GBPS, "active_mb": a.active_mb}

    lay_path = os.path.join(a.dir, "trace_layers.csv")
    lay = [r for r in rows(lay_path) if r["phase"] == "1" and r["wall_ns"]]
    n_tok = sum(1 for r in lay if r["name"] == "post")
    if n_tok == 0:
        raise ValueError("no decode `post` rows: cannot count tokens")
    blk = {}
    for r in lay:
        blk.setdefault(r["name"], 0.0)
        blk[r["name"]] += int(r["wall_ns"]) / 1e6 / n_tok
    per_layer = sorted(v for k, v in blk.items() if k.startswith("blk."))
    out["tokens"] = n_tok
    out["layers"] = {"n": len(per_layer), "ms_per_token_median": statistics.median(per_layer),
                     "ms_per_token_min": per_layer[0], "ms_per_token_max": per_layer[-1],
                     "ms_per_token_sum": sum(per_layer)}
    out["output_head_ms_per_token"] = blk.get("post", 0.0)
    out["pre_ms_per_token"] = blk.get("pre", 0.0)
    out["total_traced_ms_per_token"] = sum(blk.values())

    node_path = os.path.join(a.dir, "trace_nodes.csv")
    if os.path.exists(node_path):
        ops = {}
        for r in rows(node_path):
            if r["phase"] != "1" or not r["wall_ns"]:
                continue
            ops[r["op"]] = ops.get(r["op"], 0) + int(r["wall_ns"])
        tot = sum(ops.values()) or 1
        out["node_op_share"] = {k: v / tot for k, v in sorted(ops.items(), key=lambda kv: -kv[1])}
        out["node_total_ms_per_token_serialised"] = tot / 1e6 / n_tok

    # what the bytes alone would cost, for comparison with the traced per-token cost
    out["bytes_floor_ms_per_token"] = a.active_mb / 1024.0 / NONFLASH_GBPS * 1000.0
    out["traced_over_bytes_floor"] = out["total_traced_ms_per_token"] / out["bytes_floor_ms_per_token"]

    print(f"tokens {n_tok}; per-layer median {out['layers']['ms_per_token_median']:.2f} ms/token over "
          f"{out['layers']['n']} layers (sum {out['layers']['ms_per_token_sum']:.1f}); output head "
          f"{out['output_head_ms_per_token']:.2f}; total traced {out['total_traced_ms_per_token']:.1f} ms/token")
    print(f"bytes-only floor at {NONFLASH_GBPS} GB/s: {out['bytes_floor_ms_per_token']:.1f} ms/token "
          f"(traced / floor = {out['traced_over_bytes_floor']:.2f})")
    for k, v in list(out.get("node_op_share", {}).items())[:6]:
        print(f"  node share {k:<16} {v*100:5.1f}%")

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
