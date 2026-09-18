#!/usr/bin/env python3
"""ceiling_ledger.py — where a decoded token's time goes, from artifacts that already exist.

Analysis only: no phone run. Produces results/2026-09-18/ceiling_ledger.json, the single source for
every number quoted in research/2026-09-18_ceiling_handoff.md.

Inputs (all committed):
  results/2026-09-17/bmoe_trace/trace_nodes.csv   per-node wall time, Qwen3-30B-A3B, 4 threads, serialised
  results/2026-09-17/bmoe_trace/log.txt           clock caps and engine terms of the traced and plain runs
  results/2026-09-17/ocl_split/runs.csv           resident OLMoE, CPU vs GPU, across a thermal drift
  results/2026-09-18/device_bandwidth.json        bytes a token reads (from gguf_active_qwen_olmoe.json)
  results/2026-09-18/stack_summary.json           the latest capped-clock engine terms (12 rows per arm)
  results/2026-09-18/clock_trace/clocks.csv       scaling_max_freq sampled every ~2 s for ~93 min
  results/2026-09-18/gguf_expert_types.json       per-layer expert quant types (gates/gguf_expert_types.py)

What it does NOT do: it does not measure arithmetic directly. The node trace is serialised (a sync per
node), so its absolute times are inflated (traced run vs plain run is reported); medians are used as
the 'typical' cost of a node and mean-minus-median as the 'tail' (waits, preemption, clock steps).
"""
import collections
import csv
import json
import re
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R17, R18 = ROOT / "results/2026-09-17", ROOT / "results/2026-09-18"
STEADY_SKIP = 64          # ESTIMAND.md §1: steady state = tokens 65+
# Expert bytes per call (8 experts). Gate and up are Q4_0 in every layer. Down is Q4_1 (20 bytes per 32
# weights) in some layers and Q4_0 (18) in the rest, read from the GGUF: 2026-09-18 correction, an earlier
# version counted every down as Q4_1 and so overstated down's GB/s by 1.11x on 42 of 48 layers.
TYPES = json.loads((R18 / "gguf_expert_types.json").read_text())
DOWN_MB_BY_LAYER = {r["layer"]: 8 * r["down_bytes_per_expert"] / 1e6 for r in TYPES["layers"]}
DOWN_TYPE_BY_LAYER = {r["layer"]: r["down"] for r in TYPES["layers"]}
EXPERT_MB = {"ffn_moe_gate": 8 * 2048 * 768 * 18 / 32 / 1e6,   # Q4_0: 18 bytes per 32 weights
             "ffn_moe_up": 8 * 2048 * 768 * 18 / 32 / 1e6,
             "ffn_moe_down": sum(DOWN_MB_BY_LAYER.values()) / len(DOWN_MB_BY_LAYER)}   # layer mean; rates use per-layer bytes
DENSE_MB = {"Qcur": 2048 * 4096 * 18 / 32 / 1e6, "Kcur": 2048 * 512 * 18 / 32 / 1e6,
            "Vcur": 2048 * 512 * 18 / 32 / 1e6}


def category(op, name):
    if op == "MUL_MAT_ID":
        return "expert_matmul"
    if (op, name) == ("GET_ROWS", "ffn_moe_weights"):
        return "moe_weights_get_rows"
    if name == "result_output":
        return "output_head"
    if op == "MUL_MAT":
        return "dense_matmul"
    if op == "FLASH_ATTN_EXT":
        return "flash_attn"
    return "small_ops"


def node_ledger():
    f = open(R17 / "bmoe_trace/trace_nodes.csv")
    f.readline(), f.readline()
    rows = [r for r in csv.DictReader(f) if r["phase"] == "1"]
    steps = sorted({int(r["step"]) for r in rows})
    rows = [r for r in rows if int(r["step"]) >= steps[0] + STEADY_SKIP]
    n = len({r["step"] for r in rows})
    by = collections.defaultdict(list)
    down_rate = collections.defaultdict(list)             # per call, per-layer bytes, keyed by down type
    up_rate = collections.defaultdict(list)               # up calls split by the same layer groups
    flt = collections.defaultdict(lambda: [0, 0, 0])      # faults, nodes with a fault, ns in those nodes
    for r in rows:
        k = (r["op"], re.sub(r"[-_ ]?\d+( \(view\))?$", "", r["name"]))
        by[k].append(int(r["wall_ns"]))
        if k == ("MUL_MAT_ID", "ffn_moe_down"):
            L = int(r["layer"])
            down_rate[DOWN_TYPE_BY_LAYER[L]].append(DOWN_MB_BY_LAYER[L] / (int(r["wall_ns"]) / 1e9) / 1e3)
            down_rate["all"].append(DOWN_MB_BY_LAYER[L] / (int(r["wall_ns"]) / 1e9) / 1e3)
        if k == ("MUL_MAT_ID", "ffn_moe_up"):
            up_rate["layers_down_" + DOWN_TYPE_BY_LAYER[int(r["layer"])]].append(
                EXPERT_MB["ffn_moe_up"] / (int(r["wall_ns"]) / 1e9) / 1e3)
        m = int(r["majflt"])
        flt[k][0] += m
        if m:
            flt[k][1] += 1
            flt[k][2] += int(r["wall_ns"])
    cats = collections.defaultdict(lambda: dict(calls_per_token=0.0, mean_sum_ms=0.0, median_sum_ms=0.0))
    for k, v in by.items():
        c = cats[category(*k)]
        c["calls_per_token"] += len(v) / n
        c["mean_sum_ms"] += sum(v) / n / 1e6
        c["median_sum_ms"] += st.median(v) * len(v) / n / 1e6
    for c in cats.values():
        c["tail_ms"] = c["mean_sum_ms"] - c["median_sum_ms"]
    med_us = {k[1]: st.median(v) / 1e3 for k, v in by.items()
              if k[0] in ("MUL_MAT", "MUL_MAT_ID") and k[1] in {**EXPERT_MB, **DENSE_MB}}
    gbps = {k: {**EXPERT_MB, **DENSE_MB}[k] / (us / 1e6) / 1e3 for k, us in med_us.items()}
    # down's bytes differ by layer: its rate is the median of per-call rates with each call's own bytes
    gbps["ffn_moe_down"] = st.median(down_rate["all"])

    def q(v, p):
        v = sorted(v)
        return v[min(len(v) - 1, int(p * len(v)))]
    rate_split = {f"down_{t}": dict(n=len(v), median=st.median(v), p90_rate=q(v, 0.90), p95_rate=q(v, 0.95))
                  for t, v in down_rate.items() if t != "all"}
    rate_split.update({f"up_{t}": dict(n=len(v), median=st.median(v), p90_rate=q(v, 0.90), p95_rate=q(v, 0.95))
                       for t, v in up_rate.items()})
    total_flt = sum(v[0] for v in flt.values())
    return dict(
        n_tokens=n, nodes_per_token=len(rows) / n, categories=cats,
        total_mean_sum_ms=sum(c["mean_sum_ms"] for c in cats.values()),
        total_median_sum_ms=sum(c["median_sum_ms"] for c in cats.values()),
        median_us=med_us, median_GB_s=gbps,
        # same type (Q4_0 up vs Q4_0 down, layers 6-47) and same shape across types (down Q4_1 vs Q4_0):
        # the natural experiment on "the up/down gap is the quant type" (it is not; see the handoff §1.3)
        rate_by_layer_type_GB_s=rate_split,
        up_vs_down_same_type_Q4_0_median_ratio=rate_split["down_Q4_0"]["median"] / rate_split["up_layers_down_Q4_0"]["median"],
        # gate's median carries the expert-ready wait under serialisation; up is the same shape and type
        expert_arith_ms_gate_as_up=48 * (2 * med_us["ffn_moe_up"] + med_us["ffn_moe_down"]) / 1e3,
        up_vs_down_per_byte_ratio=gbps["ffn_moe_down"] / gbps["ffn_moe_up"],
        expert_ms_if_gate_up_ran_at_down_rate=48 * (2 * EXPERT_MB["ffn_moe_up"] / gbps["ffn_moe_down"]
                                                    + med_us["ffn_moe_down"] / 1e3),
        majflt=dict(per_token=total_flt / n,
                    faulting_nodes_per_token=sum(v[1] for v in flt.values()) / n,
                    ms_per_token_in_faulting_nodes=sum(v[2] for v in flt.values()) / n / 1e6),
    )


def trace_runs():
    out = {}
    txt = (R17 / "bmoe_trace/log.txt").read_text()
    for tag, before, after in re.findall(r"=== (\w+) [^\n]*?(cap0=\d+ cap6=\d+)[^\n]*\n(exit=[^\n]*)", txt):
        m = re.search(r"\(([\d.]+) tok/s\).*?compute ([\d.]+) \+ cache mgmt ([\d.]+)", after)
        out[tag] = dict(caps_before=before, tok_s=float(m[1]), compute_residual_ms=float(m[2]) * 1e3,
                        mgmt_ms=float(m[3]) * 1e3)
    return out


def capped_floor():
    rows = list(csv.DictReader(open(R17 / "ocl_split/runs.csv")))
    cpu = [float(r["tg_tok_s"]) for r in rows if r["cell"] == "cpu_t4_pinned"]
    gpu = [float(r["tg_tok_s"]) for r in rows if r["cell"] == "gpu_all"]
    dev = json.load(open(R18 / "device_bandwidth.json"))
    olmoe_b, qwen_b = dev["resident_active_bytes_per_token"], dev["target_active_bytes_per_token"]
    thr = sorted(cpu)[:2]                                   # the two throttled repeats
    thr_gbs = st.mean(thr) * olmoe_b / 1e9
    return dict(olmoe_cpu_t4_tok_s=cpu, olmoe_gpu_all_tok_s=gpu,
                cpu_max_over_min=max(cpu) / min(cpu), gpu_max_over_min=max(gpu) / min(gpu),
                throttled_cpu_effective_GB_s=thr_gbs,
                qwen3_ms_per_token_at_throttled_cpu_rate=qwen_b / 1e9 / thr_gbs * 1e3,
                qwen3_ms_per_token_at_gpu_rate=qwen_b / (st.median(gpu) * olmoe_b) * 1e3,
                note="the OLMoE-derived rate embeds that model's own graph overhead and larger matrices; "
                     "README of results/2026-09-18 says it is optimistic for Qwen3")


def capped_terms():
    s = json.load(open(R18 / "stack_summary.json"))["results"]
    g = lambda k, a: s[k][a]
    base = {k: g(k, "base_mean") for k in ("compute_ms", "stall_ms", "mgmt_ms", "decode_tok_s")}
    stack = {k: g(k, "stack_mean") for k in ("compute_ms", "stall_ms", "mgmt_ms", "decode_tok_s")}
    need = 100.0 - stack["stall_ms"] - stack["mgmt_ms"]
    return dict(base=base, stack=stack, compute_ms_allowed_at_10_tok_s_with_stack_io=need,
                compute_cut_needed_fraction=1 - need / stack["compute_ms"])


def clock_state():
    rows = list(csv.DictReader(open(R18 / "clock_trace/clocks.csv")))
    p0 = [int(r["p0_max"]) for r in rows]
    p6 = [int(r["p6_max"]) for r in rows]
    return dict(n_samples=len(rows), span_s=int(rows[-1]["t"]) - int(rows[0]["t"]),
                first_sample=dict(p0_max=p0[0], p6_max=p6[0], thermal_status=int(rows[0]["thermal_status"]),
                                  batt_decideg=int(rows[0]["batt_decideg"])),
                p0_max_median_kHz=st.median(p0), p6_max_median_kHz=st.median(p6),
                frac_p6_at_or_below_1651200=sum(x <= 1651200 for x in p6) / len(p6),
                frac_p0_at_or_below_2265600=sum(x <= 2265600 for x in p0) / len(p0),
                p0_levels_kHz=sorted(set(p0)), p6_levels_kHz=sorted(set(p6)))


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else R18
    try:
        ver = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception as e:                                   # counted: recorded in the artifact, not hidden
        ver = f"unknown ({type(e).__name__})"
    nl, ct = node_ledger(), capped_terms()
    io = ct["stack"]["stall_ms"] + ct["stack"]["mgmt_ms"]
    w6 = ct["stack"]["compute_ms"] * 4 / 6                  # upper bound on what 6 threads can give: perfect scaling
    gu = nl["expert_arith_ms_gate_as_up"] - nl["expert_ms_if_gate_up_ran_at_down_rate"]   # at the trace's clock
    proj = dict(io_ms_stack=io, compute_ms_stack=ct["stack"]["compute_ms"],
                compute_ms_perfect_6_thread_scaling=w6, gate_up_saving_ms_at_trace_clock=gu,
                best_case_ms_per_token_width_plus_gate_up=w6 - gu + io,
                best_case_tok_s_width_plus_gate_up=1e3 / (w6 - gu + io),
                io_ms_allowed_at_10_tok_s_after_both=100.0 - (w6 - gu),
                note="every entry is a best case: perfect 4->6 scaling and the whole up/down gap recovered")
    A = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S%z"), code_version=ver, steady_skip_tokens=STEADY_SKIP,
             node_ledger=nl, trace_runs=trace_runs(), capped_floor=capped_floor(),
             capped_terms=ct, projection=proj, clock_state=clock_state())
    p = out_dir / "ceiling_ledger.json"
    p.write_text(json.dumps(A, indent=1, default=float))
    print(p)


if __name__ == "__main__":
    main()
