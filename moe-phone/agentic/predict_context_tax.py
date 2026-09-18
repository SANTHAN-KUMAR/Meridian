#!/usr/bin/env python3
"""Predict decode rate vs context length for a phone-automation agent, BEFORE measuring it.

This is a pre-registered prediction (CLAUDE.md §10.2). It is committed before any long-context cell
is run, so the measurement that follows can only confirm or refute it.

WHAT IT PREDICTS, AND WHY THE TWO TERMS ARE SEPARATE
----------------------------------------------------
A phone-automation agent re-sends a UI snapshot each turn, so it runs at a context of roughly 1k-8k
tokens, not the few dozen every committed campaign used. Growing the context costs the engine in two
DIFFERENT ways, which no existing campaign separated because none varied context at all:

  (1) EVICTION.  KV takes RAM from the expert cache, the hit rate falls, and more expert bytes are
      read from flash per token.  Fixable: quantise KV, shrink the snapshot, or shrink the cache.
  (2) TRAFFIC.   Attention re-reads the whole KV from DRAM every decoded token: L * kv_bytes_per_token.
      Independent of the cache, and NOT fixable by giving the cache more room.

They have different fixes, so a prediction that does not say which dominates is not actionable. Each
cell below reports both terms and the share of the total that term (2) accounts for.

CALIBRATION, AND WHAT IS EXTRAPOLATION
--------------------------------------
Term (1) is calibrated by ordinary least squares over EVERY run in `bmoe_cache.json`, on two
relations:  s_per_token ~ read_MiB_per_token,  and  read_MiB_per_token ~ (1 - hit).
Fitting all runs rather than two hand-grouped anchors matters here: the requested 6000 MiB cell was
refused by the phone and granted the same cache as the 5000 MiB cell, so the design has only TWO
distinct cache budgets, and any attempt to group runs into "operating points" needs a rounding width
chosen by looking at the answer -- the move CLAUDE.md §6.4 forbids. Least squares needs no grouping,
and it yields a residual, which two hand-picked anchors cannot.

The leverage limitation stays and is reported: with only two distinct budgets the slope is determined
by the gap between two clusters, so `n_distinct_budgets` is in the output and is 2. Cells whose
implied cache falls outside the measured span are marked `extrapolated`.

Term (2) is NOT calibrated at all. It is a pure roofline: KV bytes per token over the measured DRAM
bandwidth, charged at the FASTEST clean measured rate. Charging the fastest rate makes the predicted
tax a LOWER bound, the conservative direction for a claim of the form "long context still hurts": the
prediction cannot be rescued by arguing the memory system is faster than assumed.

Inputs, all committed artifacts:
  results/2026-09-17/bmoe_cache.json              measured decode at the two granted cache budgets
  results/2026-09-17/cache_qwen3_phone_sizes.json simulated LRU hit rate vs cache fraction
  results/2026-09-18/gguf_active_qwen_olmoe.json  expert / dense byte counts read from the GGUF
  results/2026-09-16/dram_15r.json                measured DRAM bandwidth
  cache/hf/Qwen__Qwen3-30B-A3B__config.json       KV geometry

Usage:  python moe-phone/agentic/predict_context_tax.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

MIB = 1024.0 * 1024.0

# The only free choices in this script, all made before any long-context measurement exists.
# ggml block sizes: q8_0 stores 32 int8 plus one f16 scale; q4_0 stores 32 nibbles plus one f16 scale.
KV_ELEM_BYTES = {"f16": 2.0, "q8_0": 34.0 / 32.0, "q4_0": 18.0 / 32.0}
CONTEXTS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
ENGINE_RAM_MIB = [3000.0, 4000.0, 5000.0]  # total engine budget: dense + KV + expert cache
FALSIFY_BAND = 0.25


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def kv_bytes_per_token(cfg: dict, elem_bytes: float) -> float:
    """K and V, every layer, at `elem_bytes` per stored element."""
    n_layer = cfg["num_hidden_layers"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    assert head_dim * cfg["num_attention_heads"] >= cfg["hidden_size"], "head_dim looks inconsistent"
    return n_layer * n_kv * head_dim * 2.0 * elem_bytes  # 2.0 = K and V


def ols(xs: list[float], ys: list[float]) -> dict:
    """Slope, intercept and fit quality. Reports the residual two hand-picked anchors cannot."""
    n = len(xs)
    if n < 3:
        raise ValueError("refusing to fit fewer than 3 points: there would be no residual to report")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("all x identical: the design has no variation in this regressor")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    ss_res = sum(r * r for r in resid)
    ss_tot = sum((y - my) ** 2 for y in ys)
    resid_sd = (ss_res / (n - 2)) ** 0.5
    se_slope = resid_sd / sxx**0.5
    # CLAUDE.md §4.1: the estimator is a ratio whose denominator is the variation in the regressor
    # that survives conditioning. Report SE/|slope| so a slope that is indistinguishable from zero
    # cannot be quoted as if it were a measurement.
    return {
        "slope": slope,
        "intercept": intercept,
        "n": n,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "residual_sd": resid_sd,
        "se_slope": se_slope,
        "se_over_slope": abs(se_slope / slope) if slope != 0 else float("inf"),
        "sd_x": (sxx / n) ** 0.5,
        "x_min": min(xs),
        "x_max": max(xs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    mp = repo_root() / "moe-phone"
    out_dir = Path(args.out_dir) if args.out_dir else mp / "agentic" / "prereg"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = json.loads((mp / "results/2026-09-17/bmoe_cache.json").read_text())["runs"]
    hit_curve = json.loads((mp / "results/2026-09-17/cache_qwen3_phone_sizes.json").read_text())
    gguf = json.loads((mp / "results/2026-09-18/gguf_active_qwen_olmoe.json").read_text())
    dram = json.loads((mp / "results/2026-09-16/dram_15r.json").read_text())
    cfg = json.loads((mp / "cache/hf/Qwen__Qwen3-30B-A3B__config.json").read_text())

    qwen = next(g for g in gguf if g["file"] == "Qwen3-30B-A3B-Q4_0.gguf")
    expert_bytes_all = float(qwen["expert_bytes_all"])
    dense_mib = (float(qwen["other_bytes"]) + float(qwen["embedding_bytes"])) / MIB

    # --- calibration: least squares over every run, no grouping ---------------------------------
    fit_time = ols([r["read_MiB_per_token"] for r in runs], [r["s_per_token"] for r in runs])
    fit_read = ols([1.0 - r["cache_hit_pct"] / 100.0 for r in runs], [r["read_MiB_per_token"] for r in runs])
    k_flash, c0 = fit_time["slope"], fit_time["intercept"]
    m_read, b_read = fit_read["slope"], fit_read["intercept"]

    # how much leverage the design really has: distinct GRANTED caches, to the nearest 500 MiB
    fit_time["n_distinct_budgets"] = len({round(r["resident_MiB"] / 500.0) for r in runs})

    # run-to-run noise: spread of s_per_token among runs sharing a requested ceiling
    by_cell_rows: dict[str, list[dict]] = {}
    for r in runs:
        by_cell_rows.setdefault(r["cell"], []).append(r)
    noise_frac = max(
        (max(s) - min(s)) / statistics.median(s)
        for s in ([r["s_per_token"] for r in rows] for rows in by_cell_rows.values())
        if len(s) > 1
    )

    # --- DRAM: the fastest clean measured rate, as the app ---------------------------------------
    app_rows = [r for r in dram["runs"]["app"]["table"] if r.get("GBps_median_clean") and r.get("n_clean", 0) >= 2]
    dram_gbps = max(r["GBps_median_clean"] for r in app_rows)
    dram_threads = next(r["threads"] for r in app_rows if r["GBps_median_clean"] == dram_gbps)

    # --- simulated hit rate vs cache fraction -----------------------------------------------------
    curve = sorted((r["cache_fraction"], r["lru_hit__per_layer_atomic"]) for r in hit_curve["rows"])
    xs = [c[0] for c in curve]
    ys = [c[1] for c in curve]

    def hit_at(fraction: float) -> tuple[float, bool]:
        """Linear interpolation on the simulated curve; True means the cell is off its measured span."""
        if fraction <= xs[0]:
            return ys[0] * (fraction / xs[0]), True  # below the curve: scale toward the origin
        if fraction >= xs[-1]:
            return ys[-1], True
        for i in range(1, len(xs)):
            if fraction <= xs[i]:
                t = (fraction - xs[i - 1]) / (xs[i] - xs[i - 1])
                return ys[i - 1] + t * (ys[i] - ys[i - 1]), False
        raise AssertionError("unreachable")

    cache_span = (min(r["resident_MiB"] for r in runs), max(r["resident_MiB"] for r in runs))

    # --- recovery check (CLAUDE.md §9.2, Recovery class) ------------------------------------------
    # At its own calibration points the model must return the measured value. Two separate things are
    # checked, because they can fail independently:
    #   (a) the cost path  hit -> read_MiB -> s_per_token, driven by the MEASURED hit rate;
    #   (b) the SIMULATED hit curve against the MEASURED hit rate at the same cache size, which is the
    #       one input the prediction takes on trust from a simulator rather than from the phone.
    # (b) failing does not invalidate (a); it bounds how far the predicted cells can be trusted, since
    # every predicted cell gets its hit rate from the simulator.
    recovery = []
    for cell, rows in sorted(by_cell_rows.items()):
        hit_meas = statistics.median(r["cache_hit_pct"] for r in rows) / 100.0
        s_meas = statistics.median(r["s_per_token"] for r in rows)
        resident = statistics.median(r["resident_MiB"] for r in rows)
        s_pred = c0 + k_flash * max(0.0, m_read * (1.0 - hit_meas) + b_read)
        hit_sim, off = hit_at(resident * MIB / expert_bytes_all)
        recovery.append(
            {
                "cell": cell,
                "n_rep": len(rows),
                "resident_mib": resident,
                "hit_measured": hit_meas,
                "hit_simulated_same_size": hit_sim,
                "hit_sim_minus_measured": hit_sim - hit_meas,
                "sim_curve_extrapolated": off,
                "s_per_token_measured": s_meas,
                "s_per_token_model": s_pred,
                "rel_error": (s_pred - s_meas) / s_meas,
                "within_replicate_noise": abs((s_pred - s_meas) / s_meas) <= noise_frac,
            }
        )

    predictions = []
    for kv_name, elem in KV_ELEM_BYTES.items():
        kv_bpt = kv_bytes_per_token(cfg, elem)
        for ram_mib in ENGINE_RAM_MIB:
            for L in CONTEXTS:
                kv_mib = kv_bpt * L / MIB
                cache_mib = ram_mib - dense_mib - kv_mib
                row = {
                    "kv_precision": kv_name,
                    "kv_bytes_per_token": kv_bpt,
                    "engine_ram_mib": ram_mib,
                    "context_tokens": L,
                    "kv_mib": kv_mib,
                    "dense_mib": dense_mib,
                    "expert_cache_mib": cache_mib,
                    "feasible": cache_mib > 0,
                }
                if cache_mib <= 0:
                    row["note"] = "KV plus dense weights exceed the engine budget; no cache left"
                    predictions.append(row)
                    continue
                frac = cache_mib * MIB / expert_bytes_all
                hit, off_curve = hit_at(frac)
                read_mib = max(0.0, m_read * (1.0 - hit) + b_read)
                s_evict = c0 + k_flash * read_mib             # term (1), calibrated
                s_traffic = (kv_bpt * L) / (dram_gbps * 1e9)  # term (2), roofline lower bound
                s_total = s_evict + s_traffic
                row.update(
                    {
                        "cache_fraction": frac,
                        "hit_rate_sim": hit,
                        "read_mib_per_token": read_mib,
                        "s_per_token_eviction_term": s_evict,
                        "s_per_token_kv_traffic_term": s_traffic,
                        "s_per_token_total": s_total,
                        "decode_tok_s_pred": 1.0 / s_total,
                        "kv_traffic_share": s_traffic / s_total,
                        "extrapolated": bool(off_curve or not (cache_span[0] <= cache_mib <= cache_span[1])),
                    }
                )
                predictions.append(row)

    out = {
        "what": "pre-registered prediction of decode rate vs context length, Qwen3-30B-A3B on the 15R",
        "status": "PRE-REGISTERED, UNTESTED. No long-context cell has been run on any device.",
        "model": "Qwen3-30B-A3B-Q4_0",
        "generated_by": "moe-phone/agentic/predict_context_tax.py",
        "inputs": {
            "decode_runs": "results/2026-09-17/bmoe_cache.json",
            "hit_curve": "results/2026-09-17/cache_qwen3_phone_sizes.json",
            "gguf_bytes": "results/2026-09-18/gguf_active_qwen_olmoe.json",
            "dram": "results/2026-09-16/dram_15r.json",
            "config": "cache/hf/Qwen__Qwen3-30B-A3B__config.json",
        },
        "calibration": {
            "s_per_token_vs_read_mib": fit_time,
            "read_mib_vs_miss_rate": fit_read,
            "granted_cache_mib_observed": sorted({round(r["resident_MiB"]) for r in runs}),
            "cache_span_mib": list(cache_span),
            "dram_gbps_used": dram_gbps,
            "dram_threads_used": dram_threads,
            "dram_choice": "fastest clean measured app rate, making the KV traffic term a lower bound",
            "expert_bytes_all": expert_bytes_all,
            "dense_mib": dense_mib,
            "replicate_noise_frac": noise_frac,
        },
        "recovery_check": recovery,
        "limitations": [
            "The design has only TWO distinct granted cache budgets (the requested 6000 MiB cell was "
            "refused and granted the same cache as the 5000 MiB one), so the slope rests on the gap "
            "between two clusters however many runs are fitted.",
            "The KV traffic term is an uncalibrated roofline: it assumes attention reads the whole KV "
            "once per decoded token, with no reuse and no overlap with expert streaming.",
            "The hit curve is simulated and was matched to the phone at one cache size only; it is "
            "not validated at the small caches the long-context cells imply.",
            "Every calibration run used a ~34-token prompt on a quiesced phone. A phone-automation "
            "agent runs with a target app resident and in the foreground, which is not represented "
            "here and is expected to reduce the grantable budget.",
            "No agentic turn structure is modelled: this predicts DECODE only, and a phone-automation "
            "turn is expected to be dominated by PREFILL, which is unmeasured at any prompt length.",
        ],
        "falsified_if": {
            "band_frac": FALSIFY_BAND,
            "rule": "measured decode at a given (context, engine RAM, KV precision) cell falls outside "
            f"+/-{FALSIFY_BAND:.0%} of decode_tok_s_pred, or the measured ordering of cells differs "
            "from the predicted ordering",
            "band_vs_noise": f"+/-{FALSIFY_BAND:.0%} against a measured replicate spread of {noise_frac:.1%}",
        },
        "predictions": predictions,
    }
    path = out_dir / "context_tax_prediction.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}\n")

    print(f"s/token = {c0:.4f} + {k_flash:.3e} * read_MiB    "
          f"(n={fit_time['n']}, R2={fit_time['r2']:.3f}, resid SD {fit_time['residual_sd']:.4f} s, "
          f"{fit_time['n_distinct_budgets']} distinct budgets)")
    print(f"read_MiB = {b_read:.1f} + {m_read:.1f} * (1-hit)    "
          f"(n={fit_read['n']}, R2={fit_read['r2']:.3f})")
    print(f"  SE/|slope| = {fit_time['se_slope']/abs(fit_time['slope']):.3f}  <- CLAUDE.md 4.1: "
          f"{'SLOPE IS NOT ESTIMABLE FROM THIS DESIGN' if fit_time['se_over_slope'] >= 1 else 'slope is resolved'}")
    print(f"DRAM {dram_gbps:.1f} GB/s @ {dram_threads} threads; dense {dense_mib:.0f} MiB; "
          f"replicate noise {noise_frac:.1%}\n")
    print("recovery check at the calibration points:")
    for rc in recovery:
        print(f"  {rc['cell']:>9}  cache {rc['resident_mib']:>6.0f} MiB   "
              f"s/tok meas {rc['s_per_token_measured']:.3f} model {rc['s_per_token_model']:.3f} "
              f"({rc['rel_error']:+.1%}, within noise: {rc['within_replicate_noise']})   "
              f"hit meas {rc['hit_measured']:.3f} sim {rc['hit_simulated_same_size']:.3f} "
              f"({rc['hit_sim_minus_measured']:+.3f})")
    print()

    for ram in ENGINE_RAM_MIB:
        print(f"--- engine budget {ram:.0f} MiB " + "-" * 40)
        print(f"{'KV':>5} {'ctx':>6} {'kvMiB':>7} {'cacheMiB':>9} {'hit':>6} {'tok/s':>7} {'KVshare':>8} {'extrap':>7}")
        for r in predictions:
            if r["engine_ram_mib"] != ram:
                continue
            if not r["feasible"]:
                print(f"{r['kv_precision']:>5} {r['context_tokens']:>6} {r['kv_mib']:>7.0f} "
                      f"{'--':>9} {'--':>6} {'--':>7} {'--':>8} {'--':>7}   no cache left")
                continue
            print(f"{r['kv_precision']:>5} {r['context_tokens']:>6} {r['kv_mib']:>7.0f} "
                  f"{r['expert_cache_mib']:>9.0f} {r['hit_rate_sim']:>6.3f} {r['decode_tok_s_pred']:>7.2f} "
                  f"{r['kv_traffic_share']:>8.2f} {str(r['extrapolated']):>7}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
