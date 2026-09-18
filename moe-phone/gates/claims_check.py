"""
CLAIMS CHECK — every number this project asserts in prose, verified against the
artifact that produced it.

`CLAUDE.md` §7.1: *"Never write a number into prose, a README, a table or a
paper by hand. Every reported figure is read from a generated artifact by a
named script. One claim, one script, one named cell, listed in a claims map."*

This is that claims map, and it is executable. Each entry names:

  id        a stable handle, so a document can cite the claim not the digits
  text      the sentence the number appears in
  artifact  the file under results/<date>/ it is read from
  value     a function of the loaded artifacts returning the number
  expected  what the documents currently say
  tol       how close counts as agreeing

Run it two ways:

  python moe-phone/gates/claims_check.py           verify; non-zero exit on drift
  python moe-phone/gates/claims_check.py --emit    regenerate CLAIMS.md

tests/test_gates.py runs the verification, so re-running a gate and forgetting
to update a document is a test failure rather than a stale number in a paper.

WHAT THIS DOES NOT DO. It checks that prose matches artifacts. It cannot check
that the artifact is *right* -- that is what the property tests in
tests/test_gates.py are for, and the two defects retracted in POSITION.md §3c
produced artifacts that were internally consistent and wrong.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")

# Roofline constants, measured. G1 bulk bandwidth and the OLMoE expert size at
# 4.5 bpw; both also appear in the engine_sim artifact's own `inputs` block and
# are cross-checked against it by CLAIM roofline_inputs_agree.
BULK_GBPS = 2.806
EXPERT_MB = 3.54


def newest(name):
    hits = sorted(glob.glob(os.path.join(RESULTS, "*", name)))
    if not hits:
        raise FileNotFoundError(f"no artifact named {name} under {RESULTS}")
    return hits[-1]


def load(name):
    with open(newest(name), encoding="utf-8") as f:
        return json.load(f)


def row(art, frac, key="rows"):
    for r in art[key]:
        if abs(r["cache_fraction"] - frac) < 1e-9:
            return r
    raise KeyError(f"no row at cache_fraction={frac}")


def spec_ratio(A, window, key):
    """engine_sim speed-up over W=1 at alpha=0.9, at a named compute-cost setting."""
    r = row(A["engine"], 0.10)
    for w in (window, "1"):
        if key not in r["windows"][w]["tok_s_with_compute"]:
            raise KeyError(f"engine_sim artifact has no compute setting {key!r} for W={w}; "
                           f"rerun with --fwd-ms including the measured value")
    return (r["windows"][window]["tok_s_with_compute"][key]["0.9"]
            / r["windows"]["1"]["tok_s_with_compute"][key]["0.9"])


def pf_speedup(A, frac):
    """prefetch_sim speed-up at the MEASURED forward-pass cost (the row whose fwd_ms is not
    one of the round sweep values)."""
    want = [r for r in A["pf"]["rows"]
            if abs(r["cache_fraction"] - frac) < 1e-9 and abs(r["fwd_ms"] - 30.07) < 0.01]
    if len(want) != 1:
        raise KeyError(f"expected exactly one prefetch row at fraction {frac} and the measured "
                       f"30.07 ms, found {len(want)}")
    return want[0]["speedup"]


def tok_s(h, e_tok_gb):
    """The roofline of ARCHITECTURE.md §1, applied to one hit rate."""
    return BULK_GBPS / (e_tok_gb * (1.0 - h))


# OLMoE active bytes per token at 4.5 bpw, from the byte-budget artifact rather
# than typed: E_tok_params x bpw / 8.
def olmoe_e_tok_gb():
    bb = load("byte_budget_measured.json")
    m = next(x for x in bb["models"] if "OLMoE" in x["repo"])
    return m["geometry"]["E_tok_params"] * 4.5 / 8 / 1e9


CLAIMS = [
    # ---------------------------------------------------------------- G2 core
    dict(id="lru_10pct",
         text="plain per-layer LRU reaches 0.279 at a 10% cache",
         artifact="cache_OLMoE-1B-7B-0924.json", expected=0.279, tol=0.001,
         value=lambda A: row(A["cache"], 0.10)["lru_hit__per_layer_atomic"]),
    dict(id="lru_10pct_global_seq",
         text="the retracted configuration (shared pool, per-access replay) reports 0.000",
         artifact="cache_OLMoE-1B-7B-0924.json", expected=0.000, tol=0.0005,
         value=lambda A: row(A["cache"], 0.10)["lru_hit__global_sequential"]),
    dict(id="lru_10pct_perlayer_seq",
         text="per-layer scope with per-access replay reports 0.123, so both defects contribute",
         artifact="cache_OLMoE-1B-7B-0924.json", expected=0.123, tol=0.001,
         value=lambda A: row(A["cache"], 0.10)["lru_hit__per_layer_sequential"]),
    dict(id="belady_10pct",
         text="the per-layer Belady optimum at a 10% cache is 0.447",
         artifact="cache_OLMoE-1B-7B-0924.json", expected=0.447, tol=0.001,
         value=lambda A: row(A["cache"], 0.10)["belady_hit__per_layer"]),
    # ---------------------------------------------------------- S12 predictor
    dict(id="persistence_is_informationally_empty",
         text="a persistence predictor reproduces the horizon-0 rule exactly: it re-states the "
              "current token's set, which the engine already knows",
         artifact="predictor_OLMoE-1B-7B-0924.json", expected=0.0, tol=0.0005,
         value=lambda A: abs(row(A["predictor"], 0.10)["predictors"]["persistence"]["protect"]
                             - row(A["predictor"], 0.10)["h0_current_token_only"])),
    dict(id="no_real_predictor_beats_lru",
         text="the best real predictor's throughput ratio over plain LRU at a 10% cache is "
              "1.00x -- the predictor lever does not exist",
         artifact="predictor_OLMoE-1B-7B-0924.json", expected=1.00, tol=0.02,
         value=lambda A: max(v["protect_tok_s_ratio"] for kk, v in
                             row(A["predictor"], 0.10)["predictors"].items() if kk != "oracle")),
    dict(id="statistical_predictors_are_worse_than_lru",
         text="a fitted first-order Markov predictor scores 0.262 against LRU's 0.287, because "
              "its errors are plausible experts that get wrongly protected",
         artifact="predictor_OLMoE-1B-7B-0924.json", expected=0.262, tol=0.004,
         value=lambda A: row(A["predictor"], 0.10)["predictors"]["markov"]["protect"]),
    dict(id="one_step_oracle_ceiling",
         text="even an EXACT one-step oracle reaches only 0.383 against LRU's 0.287 (1.16x), so "
              "the lookahead lever needs a horizon of 4, not 1",
         artifact="predictor_OLMoE-1B-7B-0924.json", expected=0.383, tol=0.004,
         value=lambda A: row(A["predictor"], 0.10)["predictors"]["oracle"]["protect"]),
    dict(id="persistence_slot_accuracy",
         text="persistence names 0.387 of the next token's experts, against 0.125 under "
              "independence -- good prediction, useless information",
         artifact="predictor_OLMoE-1B-7B-0924.json", expected=0.387, tol=0.005,
         value=lambda A: A["predictor"]["quality"]["persistence"]["slot_accuracy"]),
    # ------------------------------------------------------------ cache scope
    # These come from scope_compare.py on an 8000-token SUB-SAMPLE, declared in
    # that artifact's `inputs`, because the shared-pool lookahead scans the whole
    # pool on every eviction. They are not full-trace figures and the claim text
    # says so.
    dict(id="scope_global_lru_is_zero",
         text="a shared pool gets NO hits at a 10% cache, where per-layer gets 0.272 "
              "(8000-token sub-sample)",
         artifact="scope_compare_OLMoE-1B-7B-0924.json", expected=0.000, tol=0.0005,
         value=lambda A: row(A["scope"], 0.10)["global"]["lru"]),
    dict(id="scope_per_layer_lru",
         text="per-layer LRU on the same sub-sample is 0.272",
         artifact="scope_compare_OLMoE-1B-7B-0924.json", expected=0.272, tol=0.002,
         value=lambda A: row(A["scope"], 0.10)["per_layer"]["lru"]),
    dict(id="scope_global_belady_is_higher",
         text="the shared pool's OFFLINE optimum is higher, 0.490 vs per-layer's 0.437",
         artifact="scope_compare_OLMoE-1B-7B-0924.json", expected=0.490, tol=0.002,
         value=lambda A: row(A["scope"], 0.10)["global"]["belady"]),
    dict(id="scope_global_h16_still_loses",
         text="a shared pool with 16 EVENTS of lookahead reaches 0.363, still below what "
              "per-layer reaches with 4 TOKENS (0.437)",
         artifact="scope_compare_OLMoE-1B-7B-0924.json", expected=0.363, tol=0.003,
         value=lambda A: row(A["scope"], 0.10)["global"]["by_horizon"]["16"]),
    dict(id="scope_per_layer_h4",
         text="per-layer with 4 tokens of lookahead reaches 0.437 on the sub-sample",
         artifact="scope_compare_OLMoE-1B-7B-0924.json", expected=0.437, tol=0.003,
         value=lambda A: row(A["scope"], 0.10)["per_layer"]["by_horizon"]["4"]),
    # ------------------------------------------------------------- decomposition
    dict(id="uniform_control_equals_floor",
         text="LRU on locality-free routing returns the cache fraction (the null is correct for LRU)",
         artifact="cache_OLMoE-1B-7B-0924.json", expected=0.100, tol=0.003,
         value=lambda A: next(
             r["lru_hit__per_layer_atomic"] for r in A["cache"]["controls"]["uniform"]
             if abs(r["cache_fraction"] - 0.10) < 1e-9)),
    dict(id="skew_term",
         text="expert-popularity skew contributes +0.034 of LRU's hit rate at a 10% cache",
         artifact="cache_OLMoE-1B-7B-0924.json", expected=0.034, tol=0.003,
         value=lambda A: (
             next(r["lru_hit__per_layer_atomic"] for r in A["cache"]["controls"]["shuffled"]
                  if abs(r["cache_fraction"] - 0.10) < 1e-9)
             - next(r["lru_hit__per_layer_atomic"] for r in A["cache"]["controls"]["uniform"]
                    if abs(r["cache_fraction"] - 0.10) < 1e-9))),
    dict(id="recency_term",
         text="temporal recency contributes +0.146, the majority term",
         artifact="cache_OLMoE-1B-7B-0924.json", expected=0.146, tol=0.003,
         value=lambda A: (
             row(A["cache"], 0.10)["lru_hit__per_layer_atomic"]
             - next(r["lru_hit__per_layer_atomic"] for r in A["cache"]["controls"]["shuffled"]
                    if abs(r["cache_fraction"] - 0.10) < 1e-9))),
    dict(id="belady_beats_floor_on_noise",
         text="Belady beats the cache fraction even on locality-free routing (0.297 vs 0.100), "
              "so the fraction is not Belady's null",
         artifact="cache_OLMoE-1B-7B-0924.json", expected=0.297, tol=0.004,
         value=lambda A: next(r["belady_hit__per_layer"] for r in A["cache"]["controls"]["uniform"]
                              if abs(r["cache_fraction"] - 0.10) < 1e-9)),
    # ------------------------------------------------------- classical policies
    dict(id="policy_lru_wins_at_10pct",
         text="of LRU / LFU / warm-LRU / pin25 / pin50 / pin100 on a held-out split, LRU wins "
              "at a 10% cache",
         artifact="expert_policy_OLMoE-1B-7B-0924.json", expected=1.0, tol=1e-9,
         value=lambda A: float(row(A["policy"]["online_policies"], 0.10)["winner"] == "LRU")),
    dict(id="policy_warming_equals_lru",
         text="cache warming is indistinguishable from plain LRU once past the warmup",
         artifact="expert_policy_OLMoE-1B-7B-0924.json", expected=0.0, tol=0.0005,
         value=lambda A: abs(row(A["policy"]["online_policies"], 0.10)["warm-LRU"]
                             - row(A["policy"]["online_policies"], 0.10)["LRU"])),
    dict(id="policy_pinning_loses",
         text="never evicting (pin100) scores 0.179, far below LRU's 0.282",
         artifact="expert_policy_OLMoE-1B-7B-0924.json", expected=0.179, tol=0.002,
         value=lambda A: row(A["policy"]["online_policies"], 0.10)["pin100"]),
    # ------------------------------------------------------------------ lookahead
    dict(id="lookahead4_reaches_belady_10pct",
         text="a 4-token lookahead reaches Belady at a 10% cache (gap closed = 1.000)",
         artifact="cache_OLMoE-1B-7B-0924.json", expected=1.000, tol=0.01,
         value=lambda A: row(A["cache"]["lookahead"], 0.10)["fraction_of_belady_gap_closed"]),
    dict(id="lookahead4_20pct",
         text="at a 20% cache a 4-token lookahead reaches 0.635 against Belady's 0.654",
         artifact="cache_OLMoE-1B-7B-0924.json", expected=0.635, tol=0.002,
         value=lambda A: row(A["cache"]["lookahead"], 0.20)["by_horizon"]["4"]),
    # ----------------------------------------------------------------- predictor
    dict(id="protect_acc09",
         text="a predictor in protect mode at accuracy 0.9 reaches 0.383 at a 10% cache",
         artifact="cache_pred_OLMoE-1B-7B-0924.json", expected=0.383, tol=0.002,
         value=lambda A: row(A["pred"]["prediction"], 0.10)["protect"]["by_accuracy"]["0.9"]),
    dict(id="rank_collapses_at_zero_accuracy",
         text="rank mode at accuracy 0 scores 0.097, BELOW plain LRU",
         artifact="cache_pred_OLMoE-1B-7B-0924.json", expected=0.097, tol=0.003,
         value=lambda A: row(A["pred"]["prediction"], 0.10)["rank"]["by_accuracy"]["0.0"]),
    dict(id="rank_break_even",
         text="rank mode's break-even predictor accuracy is 0.70",
         artifact="cache_pred_OLMoE-1B-7B-0924.json", expected=0.70, tol=1e-9,
         value=lambda A: row(A["pred"]["prediction"], 0.10)["rank"]["break_even_accuracy"]),
    dict(id="protect_never_loses",
         text="protect mode never falls below LRU at any swept accuracy (break-even reported at 0.0)",
         artifact="cache_pred_OLMoE-1B-7B-0924.json", expected=0.0, tol=1e-9,
         value=lambda A: row(A["pred"]["prediction"], 0.10)["protect"]["break_even_accuracy"]),
    dict(id="protect_acc09_tok_s",
         text="protect mode at accuracy 0.9 gives 10.0 tok/s against plain LRU's 8.6",
         artifact="cache_pred_OLMoE-1B-7B-0924.json", expected=10.0, tol=0.1,
         value=lambda A: tok_s(
             row(A["pred"]["prediction"], 0.10)["protect"]["by_accuracy"]["0.9"],
             olmoe_e_tok_gb())),
    # -------------------------------------------------------------- engine sim
    dict(id="spec_w4_a08_is_a_regression",
         text="speculation at W=4, alpha=0.8 gives 8.3 tok/s -- BELOW the 8.6 baseline",
         artifact="engine_sim_OLMoE-1B-7B-0924.json", expected=8.3, tol=0.15,
         value=lambda A: row(A["engine"], 0.10)["windows"]["4"]["tok_s_by_alpha"]["0.8"]),
    dict(id="spec_w4_a09",
         text="speculation at W=4, alpha=0.9 gives 9.7 tok/s, which the predictor beats",
         artifact="engine_sim_OLMoE-1B-7B-0924.json", expected=9.7, tol=0.15,
         value=lambda A: row(A["engine"], 0.10)["windows"]["4"]["tok_s_by_alpha"]["0.9"]),
    dict(id="spec_baseline",
         text="the W=1 baseline at a 10% cache is 8.6 tok/s",
         artifact="engine_sim_OLMoE-1B-7B-0924.json", expected=8.6, tol=0.15,
         value=lambda A: row(A["engine"], 0.10)["windows"]["1"]["tok_s_by_alpha"]["0.9"]),
    # ------------------------------------------------------------ engine target
    dict(id="qwen3_30b_lru_tok_s",
         text="Qwen3-30B-A3B reaches 11.5 tok/s on plain LRU",
         artifact="engine_target.json", expected=11.5, tol=0.1,
         value=lambda A: next(m["tok_s_at_lru"] for m in A["target"]["models"]
                              if m["repo"] == "Qwen/Qwen3-30B-A3B")),
    dict(id="qwen3_next_80b_lru_tok_s",
         text="Qwen3-Next-80B-A3B reaches 10.0 tok/s on plain LRU",
         artifact="engine_target.json", expected=10.0, tol=0.1,
         value=lambda A: next(m["tok_s_at_lru"] for m in A["target"]["models"]
                              if "Qwen3-Next-80B" in m["repo"])),
    dict(id="qwen3_next_80b_h_lru",
         text="the curve interpolates h = 0.669 for Qwen3-Next-80B-A3B (transferred from OLMoE at "
              "equal rho, S9)",
         artifact="engine_target.json", expected=0.669, tol=0.002,
         value=lambda A: next(m["h_lru_measured"] for m in A["target"]["models"]
                              if "Qwen3-Next-80B" in m["repo"])),
    dict(id="qwen3_30b_serial_dram_tok_s",
         text="charging DRAM reads of resident weights and cache hits (no overlap), Qwen3-30B-A3B "
              "falls from 11.5 to 8.8 tok/s",
         artifact="engine_target.json", expected=8.8, tol=0.1,
         value=lambda A: next(m["tok_s_at_lru_serial_dram"] for m in A["target"]["models"]
                              if m["repo"] == "Qwen/Qwen3-30B-A3B")),
    dict(id="qwen3_30b_floor_additive_tok_s",
         text="under S9's competing floor-additive hypothesis the same serial figure is 5.0 tok/s",
         artifact="engine_target.json", expected=5.0, tol=0.1,
         value=lambda A: next(m["tok_s_at_lru_floor_additive_serial_dram"]
                              for m in A["target"]["models"] if m["repo"] == "Qwen/Qwen3-30B-A3B")),
    dict(id="olmoe_fully_resident",
         text="OLMoE-1B-7B fits entirely in the 4.85 GB budget, so it has no flash traffic and "
              "no cache-curve number (1 = fully resident)",
         artifact="engine_target.json", expected=1.0, tol=1e-9,
         value=lambda A: float(next(m["fully_resident"] for m in A["target"]["models"]
                                    if "OLMoE" in m["repo"]))),
    dict(id="n_models_outside_curve",
         text="3 of 10 candidate models sit outside the measured rho range and get no number",
         artifact="engine_target.json", expected=3.0, tol=1e-9,
         value=lambda A: float(sum(1 for m in A["target"]["models"]
                                   if m["resident_fits"] and m["curve_extrapolated"]
                                   and not m["fully_resident"]))),
    # -------------------------------------------------------------- S1 DRAM
    dict(id="dram_app_gbps",
         text="DRAM read bandwidth available to an unprivileged app is 59.7 GB/s (4 threads, "
              "median of clean rows)",
         artifact="dram_15r.json", expected=59.74, tol=0.01,
         value=lambda A: A["dram"]["runs"]["app"]["best"]["GBps_median_clean"]),
    dict(id="dram_8thread_never_clean",
         text="at 8 threads no DRAM row was free of descheduling in either domain",
         artifact="dram_15r.json", expected=0.0, tol=1e-9,
         value=lambda A: float(sum(t["n_clean"] for d in A["dram"]["runs"].values()
                                   for t in d["table"] if t["threads"] == 8))),
    # ----------------------------------------------------------------- S2
    dict(id="turbosparse_resident_ratio",
         text="TurboSparse-Mixtral's HF-derived resident count is 3.70x its config-derived count",
         artifact="byte_budget_measured.json", expected=3.70, tol=0.01,
         value=lambda A: 1.0 / (1.0 - next(m["resident_crosscheck_rel_diff"]
                                           for m in A["bb"]["models"]
                                           if "TurboSparse" in m["repo"]))),
    # -------------------------------------------------- G-VALID-3 (colibri)
    dict(id="colibri_flash_only_overpredicts",
         text="on colibri's disk-bound MTP-off rows the flash-only formula over-predicts tok/s "
              "by 2.8x (median)",
         artifact="external_validation_colibri.json", expected=2.79, tol=0.01,
         value=lambda A: A["ext"]["by_cold_bytes"]["benchmarks.md:86"]["summary"]
                          ["disk_bound_ratio_median"]),
    dict(id="colibri_high_hit_overpredicts",
         text="at a 98% hit rate (RAM-bandwidth + matmul bound) it over-predicts by 4.4x",
         artifact="external_validation_colibri.json", expected=4.39, tol=0.01,
         value=lambda A: A["ext"]["by_cold_bytes"]["benchmarks.md:86"]["summary"]
                          ["high_hit_ratio"][0]),
    dict(id="colibri_m1_eta_reproduces_their_93pct",
         text="the in-engine disk efficiency computed for colibri's M1 Ultra row is 0.93, the "
              "same '~93% of its iobench ceiling' colibri reports for that run",
         artifact="external_validation_colibri.json", expected=0.93, tol=0.005,
         value=lambda A: next(r["eta_in_engine"] for r in
                              A["ext"]["by_cold_bytes"]["benchmarks.md:86"]["rows"]
                              if r["line"] == 111)),
    # ------------------------------------------- speculation, compute charged
    dict(id="spec_w4_a09_compute10ms",
         text="charging 10 ms per forward pass (draft passes and verify, beta = 0), W=4 at "
              "alpha=0.9 gives 1.10x over plain decode, not 1.13x",
         artifact="engine_sim_OLMoE-1B-7B-0924.json", expected=1.10, tol=0.005,
         value=lambda A: (row(A["engine"], 0.10)["windows"]["4"]["tok_s_with_compute"]
                          ["c=10ms,beta=0"]["0.9"]
                          / row(A["engine"], 0.10)["windows"]["1"]["tok_s_with_compute"]
                          ["c=10ms,beta=0"]["0.9"])),
    dict(id="spec_w4_a09_compute10ms_beta1",
         text="if the batched verify is compute-bound (beta = 1) the same point is 1.02x",
         artifact="engine_sim_OLMoE-1B-7B-0924.json", expected=1.02, tol=0.005,
         value=lambda A: (row(A["engine"], 0.10)["windows"]["4"]["tok_s_with_compute"]
                          ["c=10ms,beta=1"]["0.9"]
                          / row(A["engine"], 0.10)["windows"]["1"]["tok_s_with_compute"]
                          ["c=10ms,beta=1"]["0.9"])),
    # ------------------------------------------------------- G1 / G-VALID-1
    dict(id="g1_4k_cell_spread",
         text="the 4 KB cell behind the 8.9x bulk/small ratio has a 39% run-to-run spread",
         artifact="g1_storage.json", expected=0.389, tol=0.001,
         value=lambda A: A["g1"]["small_4k_rand_best"]["rel_spread"]),
    dict(id="gvalid1_bound_below_published",
         text="the roofline bound for PowerInfer-2's own configuration is 1.90 tok/s, BELOW the "
              "2.13 tok/s PowerInfer-2 measured",
         artifact="byte_budget_validate_powerinfer2.json", expected=1.90, tol=0.005,
         value=lambda A: next(r["tps_overlap_bound"] for r in A["pi2"]["models"][0]["rows"]
                              if r["format"] == "Q4_0" and r["d"] == 0.03
                              and r["mode"] == "predicted" and r["h_label"] == "floor")),
    # --------------------------------------------- S11, on-device decode (G6-baseline)
    dict(id="s11_effective_nonflash_gbps",
         text="a resident MoE on the 15R consumes its per-token bytes at an effective 23.2 GB/s "
              "(granite-1b-a400m Q4_0, 4 threads, stock llama.cpp)",
         artifact="s11_nonflash_15r.json", expected=23.2, tol=0.05,
         value=lambda A: A["s11"]["effective_nonflash_GBps"]),
    dict(id="s11_fraction_of_dram",
         text="that is 39% of the DRAM probe's bandwidth: resident decode is compute/overhead-bound",
         artifact="s11_nonflash_15r.json", expected=0.39, tol=0.005,
         value=lambda A: A["s11"]["fraction_of_dram"]),
    dict(id="s11_olmoe_resident_prediction",
         text="predicted fully resident OLMoE-1B-7B Q4_0 decode: 33.3 tok/s (DRAM-only bound 85.7)",
         artifact="s11_nonflash_15r.json", expected=33.3, tol=0.05,
         value=lambda A: A["s11"]["prediction"]["resident_tok_s_predicted"]),
    dict(id="granite_resident_tok_s",
         text="granite-1b-a400m Q4_0 fully resident decodes at 107.2 tok/s steady state, 4 threads",
         artifact="decode_15r.json", expected=107.2, tol=0.1,
         value=lambda A: next(g["marginal_tok_s_median"] for g in A["dec"]["groups"]
                              if g["model"].startswith("granite") and g["threads"] == 4)),
    dict(id="olmoe_stock_steady_state_spread",
         text="stock llama.cpp OLMoE Q4_0 steady-state s/token varied 4.7x between cold repeats "
              "with no change in flash bytes",
         artifact="decode_15r.json", expected=4.7, tol=0.05,
         value=lambda A: (lambda ps: max(ps) / min(ps))(
             [p["marginal_s_per_tok"] for g in A["dec"]["groups"]
              if g["model"].startswith("olmoe") and g["mode"] == "cold" for p in g["pairs"]])),
    dict(id="qwen3_30b_serial_nonflash_tok_s",
         text="with the measured non-flash rate charged, Qwen3-30B-A3B is 6.3 tok/s under H_rho",
         artifact="engine_target.json", expected=6.3, tol=0.05,
         value=lambda A: next(m["tok_s_at_lru_serial_nonflash"] for m in A["target"]["models"]
                              if m["repo"] == "Qwen/Qwen3-30B-A3B")),
    dict(id="qwen3_30b_floor_additive_nonflash_tok_s",
         text="and 4.2 tok/s under H_floor — either side of the 5 tok/s target, so S9 decides it",
         artifact="engine_target.json", expected=4.2, tol=0.05,
         value=lambda A: next(m["tok_s_at_lru_floor_additive_serial_nonflash"]
                              for m in A["target"]["models"] if m["repo"] == "Qwen/Qwen3-30B-A3B")),
    dict(id="nonflash_input_agrees",
         text="the non-flash rate engine_target was run with is the one s11_nonflash derived",
         artifact="engine_target.json", expected=0.0, tol=0.001,
         value=lambda A: abs(A["target"]["inputs"]["nonflash_gbps"]
                             - A["s11"]["effective_nonflash_GBps"])),
    dict(id="queue_depth_plateau",
         text="at 1 MB direct reads, 16 or 32 threads reach 0.99x of 8 threads: no queue-depth lever",
         artifact="g1_storage_qd.json", expected=0.99, tol=0.005,
         value=lambda A: max(t["MBps_median"] for t in A["g1qd"]["table"]
                             if t["mode"] == "direct" and t["size_kb"] == 1024 and t["threads"] in (16, 32))
                        / next(t["MBps_median"] for t in A["g1qd"]["table"]
                               if t["mode"] == "direct" and t["size_kb"] == 1024 and t["threads"] == 8)),
    dict(id="olmoe_t2_warm_marginal",
         text="campaign 2: warm OLMoE Q4_0 decodes at 30.1 tok/s steady state with 2 threads",
         artifact="decode_15r_cpu.json", expected=30.1, tol=0.1,
         value=lambda A: next(g["marginal_tok_s_median"] for g in A["dec2"]["groups"]
                              if g["model"].startswith("olmoe") and g["threads"] == 2)),
    dict(id="olmoe_t4_warm_marginal",
         text="and at 8.0 tok/s with 4 threads, in the same memory state",
         artifact="decode_15r_cpu.json", expected=8.0, tol=0.05,
         value=lambda A: next(g["marginal_tok_s_median"] for g in A["dec2"]["groups"]
                              if g["model"].startswith("olmoe") and g["threads"] == 4
                              and g["mode"] == "warm")),
    dict(id="s11_matched_t2_ratio",
         text="at matched thread count (2) the linear-in-bytes rule under-predicts OLMoE by 1.33x",
         artifact="s11_nonflash_15r.json", expected=1.33, tol=0.01,
         value=lambda A: next(t["ratio_observed_over_predicted"]
                              for t in A["s11"]["matched_thread_tests"] if t["threads"] == 2)),
    dict(id="thr_olmoe_t1",
         text="thread sweep (interleaved): warm OLMoE 20.8 tok/s at 1 thread",
         artifact="decode_15r_threads.json", expected=20.8, tol=0.1,
         value=lambda A: next(c["marginal_tok_s_median"] for c in A["thr"]["by_threads"] if c["model"].startswith("olmoe") and c["threads"] == 1)),
    dict(id="thr_olmoe_t2",
         text="17.4 tok/s at 2 threads",
         artifact="decode_15r_threads.json", expected=17.4, tol=0.1,
         value=lambda A: next(c["marginal_tok_s_median"] for c in A["thr"]["by_threads"] if c["model"].startswith("olmoe") and c["threads"] == 2)),
    dict(id="thr_olmoe_t3",
         text="2.3 tok/s at 3 threads: a cliff between 2 and 3",
         artifact="decode_15r_threads.json", expected=2.3, tol=0.05,
         value=lambda A: next(c["marginal_tok_s_median"] for c in A["thr"]["by_threads"] if c["model"].startswith("olmoe") and c["threads"] == 3)),
    dict(id="thr_olmoe_t4",
         text="2.1 tok/s at 4 threads",
         artifact="decode_15r_threads.json", expected=2.1, tol=0.05,
         value=lambda A: next(c["marginal_tok_s_median"] for c in A["thr"]["by_threads"] if c["model"].startswith("olmoe") and c["threads"] == 4)),
    dict(id="thr_granite_t4",
         text="resident granite scales normally: 88.6 tok/s at 4 threads",
         artifact="decode_15r_threads.json", expected=88.6, tol=0.1,
         value=lambda A: next(c["marginal_tok_s_median"] for c in A["thr"]["by_threads"] if c["model"].startswith("granite") and c["threads"] == 4)),
    # ------------------------------------------- S9 confound control (pre-registered)
    dict(id="s9_confound_max_curve_diff",
         text="the same model measured through llama.cpp at Q4_0 differs from the fp16/HF "
              "reference curve by at most 0.0013 across the pre-registered rho grid",
         artifact="s9_result_Qwen3-30B-A3B.json", expected=0.00133, tol=0.00005,
         value=lambda A: A["s9r"]["confound"]["max_abs_diff"]),
    dict(id="s9_confound_under_limit",
         text="that is far below the pre-registered limit of 0.031, so the S9 test may proceed",
         artifact="s9_result_Qwen3-30B-A3B.json", expected=0.0, tol=1e-9,
         value=lambda A: float(A["s9r"]["confound"]["voids_test"])),
    dict(id="q4_token_slot_overlap",
         text="Q4_0 routing shares 94.8% of each token's expert slots with fp16",
         artifact="traces_confound_olmoe_q4_vs_fp16.json", expected=0.9478, tol=0.0005,
         value=lambda A: A["conf"]["mean_overlap"]),
    dict(id="q4_exact_set_match",
         text="but only 61.8% of tokens get an identical top-8 SET: quantisation moves routing "
              "per token while leaving the hit-rate curve intact",
         artifact="traces_confound_olmoe_q4_vs_fp16.json", expected=0.6179, tol=0.0005,
         value=lambda A: A["conf"]["exact_set_match"]),
    # ------------------------------------------------- S9 result (granite, E/k=4)
    dict(id="s9_granite_rmse_h_rho",
         text="the pre-registered S9 test on granite-3.1-1b-a400m (E/k=4): the equal-rho "
              "transfer misses by RMSE 0.167, against a tolerance of 0.029",
         artifact="s9_result_granite-3.1-1b-a400m.json", expected=0.1669, tol=0.0005,
         value=lambda A: A["s9g"]["test"]["rmse"]["H_rho"]),
    dict(id="s9_granite_rmse_h_floor",
         text="the floor-additive rule misses by RMSE 0.048 — closer, but also outside tolerance",
         artifact="s9_result_granite-3.1-1b-a400m.json", expected=0.0482, tol=0.0005,
         value=lambda A: A["s9g"]["test"]["rmse"]["H_floor"]),
    dict(id="s9_granite_h_rho_refuted",
         text="no grid point supports the equal-rho transfer (supported = false)",
         artifact="s9_result_granite-3.1-1b-a400m.json", expected=0.0, tol=1e-9,
         value=lambda A: float(A["s9g"]["test"]["supported"]["H_rho"])),
    dict(id="s9_granite_worst_h_rho_error",
         text="its largest miss is +0.243 in hit rate, at rho = 3",
         artifact="s9_result_granite-3.1-1b-a400m.json", expected=0.243, tol=0.001,
         value=lambda A: max(r["err_H_rho"] for r in A["s9g"]["test"]["rows"]
                             if not r["degenerate"])),
    dict(id="s9_granite_all_errors_same_sign",
         text="every scored point lies ABOVE the equal-rho prediction, so the error is a bias, "
              "not scatter",
         artifact="s9_result_granite-3.1-1b-a400m.json", expected=1.0, tol=1e-9,
         value=lambda A: float(all(r["err_H_rho"] > 0 for r in A["s9g"]["test"]["rows"]
                                   if not r["degenerate"]))),
    # -------------------------------- levers priced at the MEASURED compute cost
    dict(id="spec_w4_a09_measured_compute",
         text="at the phone's MEASURED forward-pass cost (30.07 ms for OLMoE), speculation at "
              "W=4, alpha=0.9 is worth 1.06x, not the 1.13x the flash-only pricing showed",
         artifact="engine_sim_OLMoE-1B-7B-0924.json", expected=1.061, tol=0.005,
         value=lambda A: spec_ratio(A, "4", "c=30.07ms,beta=0")),
    dict(id="spec_best_measured_compute",
         text="the best window at that cost is 1.07x (W=8), against 1.45x under flash-only pricing",
         artifact="engine_sim_OLMoE-1B-7B-0924.json", expected=1.068, tol=0.005,
         value=lambda A: max(spec_ratio(A, W, "c=30.07ms,beta=0") for W in ("2", "4", "8", "16"))),
    dict(id="spec_measured_compute_beta1_is_a_regression",
         text="if the batched verify is compute-bound the same W=4 point becomes 0.89x, a regression",
         artifact="engine_sim_OLMoE-1B-7B-0924.json", expected=0.891, tol=0.005,
         value=lambda A: spec_ratio(A, "4", "c=30.07ms,beta=1")),
    dict(id="prefetch_measured_compute_10pct",
         text="within-token prefetch at the measured compute cost LOSES at a 10% cache (0.76x): "
              "it needs about 80 ms of compute per token to pay, and the phone has 30",
         artifact="prefetch_sim_OLMoE-1B-7B-0924.json", expected=0.758, tol=0.005,
         value=lambda A: pf_speedup(A, 0.10)),
    dict(id="prefetch_measured_compute_20pct",
         text="and only breaks even at a 20% cache (0.99x)",
         artifact="prefetch_sim_OLMoE-1B-7B-0924.json", expected=0.989, tol=0.005,
         value=lambda A: pf_speedup(A, 0.20)),
    dict(id="cliff_unpinned_t4",
         text="thread-cliff sweep: stock llama.cpp OLMoE Q4_0, 4 UNPINNED threads, decodes at 4.0 tok/s",
         artifact="decode_15r_pin.json", expected=3.98, tol=0.05,
         value=lambda A: next(c["tg_tok_s_long_median"] for c in A["pin"]["by_threads"] if c["threads"] == 4 and c["extra"] == '')),
    dict(id="cliff_fix_pinned_t4",
         text="the same 4 threads pinned with --cpu-strict 1 decode at 30.1 tok/s: the cliff is thread placement",
         artifact="decode_15r_pin.json", expected=30.1, tol=0.05,
         value=lambda A: next(c["tg_tok_s_long_median"] for c in A["pin"]["by_threads"] if c["threads"] == 4 and c["extra"] == '-C 0xF0 --cpu-strict 1')),
    dict(id="cliff_fix_no_primes_needed",
         text="pinned to cores 0-3, with no prime cores, still 28.1 tok/s",
         artifact="decode_15r_pin.json", expected=28.11, tol=0.05,
         value=lambda A: next(c["tg_tok_s_long_median"] for c in A["pin"]["by_threads"] if c["threads"] == 4 and c["extra"] == '-C 0x0F --cpu-strict 1')),
    dict(id="cliff_t6_pinned",
         text="6 pinned threads: 29.0 tok/s, no gain over 4",
         artifact="decode_15r_pin.json", expected=28.96, tol=0.05,
         value=lambda A: next(c["tg_tok_s_long_median"] for c in A["pin"]["by_threads"] if c["threads"] == 6 and c["extra"] == '-C 0xFC --cpu-strict 1')),
    dict(id="cliff_t8_pinned_worse",
         text="8 pinned threads on every core: 16.1 tok/s, contending with the system",
         artifact="decode_15r_pin.json", expected=16.05, tol=0.05,
         value=lambda A: next(c["tg_tok_s_long_median"] for c in A["pin"]["by_threads"] if c["threads"] == 8 and c["extra"] == '-C 0xFF --cpu-strict 1')),
    dict(id="bmoe_reproduced_decode",
         text="BigMoeOnEdge's published Qwen3-30B-A3B run reproduces on our OnePlus 15R: 3.98 tok/s median decode (reference build, its documented command)",
         artifact="bmoe_repro.json", expected=3.977, tol=0.01,
         value=lambda A: next(c["decode_tok_s_median"] for c in A["bmoe"]["cells"] if c["cell"] == "reference_armv82")),
    dict(id="bmoe_reproduced_hit",
         text="at a 69.5% expert-cache hit rate (auto budget ~3.0-3.4 GB)",
         artifact="bmoe_repro.json", expected=69.5, tol=0.1,
         value=lambda A: next(c["cache_hit_pct_median"] for c in A["bmoe"]["cells"] if c["cell"] == "reference_armv82")),
    dict(id="bmoe_compute_share",
         text="decode spends 0.145 s/token in its compute residual (wall - stall - cache mgmt, BigMoeOnEdge docs/telemetry.md), the largest single term",
         artifact="bmoe_repro.json", expected=0.1445, tol=0.001,
         value=lambda A: next(c["compute_s_per_token_median"] for c in A["bmoe"]["cells"] if c["cell"] == "reference_armv82")),
    dict(id="bmoe_i8mm_no_gain",
         text="an i8mm build of the same engine gains nothing: 3.71 tok/s (its kernels are generic with repacking off)",
         artifact="bmoe_repro.json", expected=3.706, tol=0.01,
         value=lambda A: next(c["decode_tok_s_median"] for c in A["bmoe"]["cells"] if c["cell"] == "i8mm")),
    dict(id="bmoe_pinning_starves_io",
         text="pinning every thread to 4 cores collapses streaming to 0.23 tok/s: it starves the I/O lanes",
         artifact="bmoe_repro.json", expected=0.2265, tol=0.005,
         value=lambda A: next(c["decode_tok_s_median"] for c in A["bmoe"]["cells"] if c["cell"] == "i8mm_pinned")),
    # ------------------------------------------------------ 2026-09-17 phone campaigns (throttled CPU, n=2 per cell)
    dict(id="levers_reference",
         text="one-lever sweep, reference cell: 3.68 tok/s median over 2 interleaved repeats (throttled CPU)",
         artifact="bmoe_levers.json", expected=3.6765, tol=0.01,
         value=lambda A: next(c["decode_tok_s_median"] for c in A["levers"]["cells"] if c["cell"] == "reference")),
    dict(id="levers_cache_big",
         text="a larger expert cache (1 GB floor instead of 1.5 GB) is the only lever above reference: 3.98 tok/s median",
         artifact="bmoe_levers.json", expected=3.9805, tol=0.01,
         value=lambda A: next(c["decode_tok_s_median"] for c in A["levers"]["cells"] if c["cell"] == "cache_big")),
    dict(id="levers_t6_collapse",
         text="6 compute threads on top of 4 I/O lanes collapse decode to 1.10 tok/s median (oversubscribed cores)",
         artifact="bmoe_levers.json", expected=1.0985, tol=0.01,
         value=lambda A: next(c["decode_tok_s_median"] for c in A["levers"]["cells"] if c["cell"] == "t6")),
    dict(id="repack_plain",
         text="i8mm build, generic kernels: 4.11 tok/s median, the fastest cell of the repack A/B",
         artifact="bmoe_repack.json", expected=4.1145, tol=0.01,
         value=lambda A: next(c["decode_tok_s_median"] for c in A["repack"]["cells"] if c["cell"] == "i8mm_plain")),
    dict(id="repack_compute",
         text="repacked kernels on streamed experts: compute residual 0.170 s/token median vs 0.146 generic - no gain",
         artifact="bmoe_repack.json", expected=0.17, tol=0.002,
         value=lambda A: next(c["compute_s_per_token_median"] for c in A["repack"]["cells"] if c["cell"] == "i8mm_repack")),
    dict(id="pin2_pinned_best",
         text="rotated-order confirmation (4 repeats, awake, throttled SoC): pinned 4 compute threads on cores 4-7 with I/O on cores 0-3 decode Qwen3-30B-A3B at 5.36 tok/s median, the best cell",
         artifact="bmoe_pin2.json", expected=5.359, tol=0.01,
         value=lambda A: next(c["decode_tok_s_median"] for c in A["bpin2"]["cells"] if c["cell"] == "pin_t4c47_io4c03")),
    dict(id="pin2_unpinned",
         text="the unpinned baseline in the same confirmation: 4.75 tok/s median",
         artifact="bmoe_pin2.json", expected=4.747, tol=0.01,
         value=lambda A: next(c["decode_tok_s_median"] for c in A["bpin2"]["cells"] if c["cell"] == "unpinned_t4_io4")),
    dict(id="pin2_recycle_not_a_gain",
         text="page recycling on top of pinning: 5.10 tok/s median - cache management falls but compute rises, a net loss",
         artifact="bmoe_pin2.json", expected=5.101, tol=0.01,
         value=lambda A: next(c["decode_tok_s_median"] for c in A["bpin2"]["cells"] if c["cell"] == "pin_t4_recycle")),
    dict(id="pin2_pinned_compute",
         text="pinned compute residual (wall - stall - cache mgmt, not measured matmul time) 0.0995 s/token median vs 0.1335 s unpinned",
         artifact="bmoe_pin2.json", expected=0.0995, tol=0.002,
         value=lambda A: next(c["compute_s_per_token_median"] for c in A["bpin2"]["cells"] if c["cell"] == "pin_t4c47_io4c03")),
    dict(id="verify_cost_n2",
         text="a streamed verify of 2 positions costs 1.71x a single-token decode (Qwen3-30B-A3B on the 15R, 3 repeats, awake)",
         artifact="verify_cost.json", expected=1.7119, tol=0.01,
         value=lambda A: next(x["c_median"] for x in A["vcost"]["cells"] if x["N"] == 2)),
    dict(id="verify_cost_n3",
         text="a streamed verify of 3 positions costs 2.37x a single-token decode",
         artifact="verify_cost.json", expected=2.3716, tol=0.01,
         value=lambda A: next(x["c_median"] for x in A["vcost"]["cells"] if x["N"] == 3)),
    dict(id="verify_cost_n5",
         text="a streamed verify of 5 positions costs 3.71x a single-token decode: a 4-token draft must average more than 3.71 accepted tokens per verify to break even",
         artifact="verify_cost.json", expected=3.7098, tol=0.01,
         value=lambda A: next(x["c_median"] for x in A["vcost"]["cells"] if x["N"] == 5)),
    dict(id="lookahead4_gap_at_30pct",
         text="at a 30% cache a 4-token exact lookahead closes 70% of the LRU-to-Belady gap on OLMoE (all of it only up to ~12.5%)",
         artifact="cache_OLMoE-1B-7B-0924.json", expected=0.7, tol=0.01,
         value=lambda A: next(((r["by_horizon"]["4"] - r["lru_atomic"]) / (r["belady"] - r["lru_atomic"])) for r in A["cacheo"]["lookahead"]["rows"] if abs(r["cache_fraction"] - 0.3) < 1e-9)),
    dict(id="cache5000_tok_s",
         text="pinned t4 with a 5 GB expert cache (floor 1024 MiB): 6.20 tok/s median over 3 rotated repeats on Qwen3-30B-A3B, the best measured configuration (throttled caps, phone under AC from 21:48)",
         artifact="bmoe_cache.json", expected=6.199, tol=0.01,
         value=lambda A: next(c["decode_tok_s_median"] for c in A["bcache"]["cells"] if c["cell"] == "ceil5000")),
    dict(id="cache4000_tok_s",
         text="the same campaign's 4 GB cache: 5.76 tok/s median",
         artifact="bmoe_cache.json", expected=5.764, tol=0.01,
         value=lambda A: next(c["decode_tok_s_median"] for c in A["bcache"]["cells"] if c["cell"] == "ceil4000")),
    dict(id="cache5000_hit",
         text="raising the cache from 4 GB to 5 GB lifts the hit rate from 78.0% to 85.3%",
         artifact="bmoe_cache.json", expected=85.3, tol=0.05,
         value=lambda A: next(c["cache_hit_pct_median"] for c in A["bcache"]["cells"] if c["cell"] == "ceil5000")),
    dict(id="cache5000_read",
         text="and cuts flash reads from 193.8 to 119.8 MiB per token",
         artifact="bmoe_cache.json", expected=119.8, tol=0.5,
         value=lambda A: next(c["read_MiB_per_token_median"] for c in A["bcache"]["cells"] if c["cell"] == "ceil5000")),
    dict(id="qwen3_sim_lru_5gb",
         text="replayed on Qwen3-30B-A3B's own llama.cpp trace (wikitext, 8192 tokens), global LRU at the 5 GB cache's 30.7% of experts hits 86.4%, vs 85.3% measured on the phone (essay prompt)",
         artifact="cache_qwen3_phone_sizes.json", expected=0.864, tol=0.002,
         value=lambda A: next(r["lru_hit__global_atomic"] for r in A["qsizes"]["rows"] if abs(r["cache_fraction"] - 0.307) < 1e-6)),
    dict(id="qwen3_sim_belady_5gb",
         text="at the same cache size the offline optimum (Belady, global) hits 94.6%: eviction, not cache size, is the larger remaining I/O lever",
         artifact="cache_qwen3_phone_sizes.json", expected=0.946, tol=0.002,
         value=lambda A: next(r["belady_hit__global"] for r in A["qsizes"]["rows"] if abs(r["cache_fraction"] - 0.307) < 1e-6)),
    dict(id="trace_per_layer_ms",
         text="per-layer decode cost on the phone, traced at layer granularity: 2.82 ms/token median over 48 layers",
         artifact="compute_trace.json", expected=2.8184, tol=0.02,
         value=lambda A: A["ctrace"]["layers"]["ms_per_token_median"]),
    dict(id="trace_output_head_ms",
         text="the 151936-row output projection alone costs 10.64 ms/token, 7% of a traced token",
         artifact="compute_trace.json", expected=10.6383, tol=0.05,
         value=lambda A: A["ctrace"]["output_head_ms_per_token"]),
    dict(id="trace_over_bytes_floor",
         text="the traced token costs 1.94x what its active bytes alone would at the measured 23.2 GB/s non-flash rate: the gap is barrier/scheduling overhead, not arithmetic",
         artifact="compute_trace.json", expected=1.9425, tol=0.02,
         value=lambda A: A["ctrace"]["traced_over_bytes_floor"]),
    dict(id="trace_mul_mat_id_share",
         text="expert matmuls (MUL_MAT_ID, including the expert-ready wait) are 50.2% of decode node time in the serialised node trace",
         artifact="compute_trace.json", expected=0.502, tol=0.005,
         value=lambda A: A["ctrace"]["node_op_share"]["MUL_MAT_ID"]),
    # ---------------------------------------- ACTIVE BYTES (gates/gguf_active.py, 2026-09-18)
    # The denominator of every ceiling argument: a token's weight bytes are fixed by the file, so a
    # decode rate is equivalent to a weight-byte throughput and vice versa.
    dict(id="qwen3_active_mb_per_token",
         text="a decoded token of Qwen3-30B-A3B Q4_0 reads 1840.0 MB of weights (top-8 of 128 experts, plus every dense tensor and one embedding row)",
         artifact="gguf_active_qwen_olmoe.json", expected=1840.0, tol=0.1,
         value=lambda A: next(r["active_bytes_per_token"] for r in A["act18"]
                              if r["file"] == "Qwen3-30B-A3B-Q4_0.gguf") / 1e6),
    dict(id="olmoe_active_mb_per_token",
         text="a decoded token of OLMoE-1B-7B Q4_0 reads 697.4 MB of weights, which is what makes it the resident-model yardstick for a device's byte throughput",
         artifact="gguf_active_qwen_olmoe.json", expected=697.4, tol=0.1,
         value=lambda A: next(r["active_bytes_per_token"] for r in A["act18"]
                              if r["file"] == "olmoe-1b-7b-0924-q4_0.gguf") / 1e6),
    # ------------------------- IN-APP DEVICE THROUGHPUT (gates/app_engine_analyze.py +
    #                           gates/device_bandwidth.py, 2026-09-18)
    # Resident-model decode rates measured inside an app process, where the DSP session opens. These are
    # llama.cpp's own llama-bench tg numbers, so they are a property of the DEVICE, not of our engine.
    dict(id="inapp_olmoe_cpu_tok_s",
         text="OLMoE-1B-7B Q4_0 resident, decoded in an app process on 4 CPU threads: 44.47 tok/s",
         artifact="app_engine.json", expected=44.465, tol=0.01,
         value=lambda A: next(x["decode_tok_s_median"] for x in A["app18"]["arms"] if x["arm"] == "olmoe_cpu")),
    dict(id="inapp_olmoe_gpu_tok_s",
         text="the same model on the Adreno via OpenCL: 49.98 tok/s, the fastest single device on this phone",
         artifact="app_engine.json", expected=49.975, tol=0.01,
         value=lambda A: next(x["decode_tok_s_median"] for x in A["app18"]["arms"] if x["arm"] == "olmoe_gpu")),
    dict(id="inapp_olmoe_htp_tok_s",
         text="the same model on the Hexagon NPU: 40.73 tok/s at token generation -- last of the three, but only at batch-1 decode and only in the backend's DEFAULT configuration (GGML_HEXAGON_OPPOLL=0, HOSTBUF=0, one DSP session); at prefill the same rows put it first by 2.1x",
         artifact="app_engine.json", expected=40.73, tol=0.01,
         value=lambda A: next(x["decode_tok_s_median"] for x in A["app18"]["arms"] if x["arm"] == "olmoe_htp")),
    dict(id="inapp_olmoe_htp_prefill",
         text="the same NPU rows reach 284.07 tok/s at prefill, 2.1x the Adreno's 135.33 and 14x the CPU's 20.36: the device is compute-strong and the decode gap is a latency effect, not a bandwidth one",
         artifact="app_engine.json", expected=284.07, tol=0.01,
         value=lambda A: next(x["prefill_tok_s_median"] for x in A["app18"]["arms"] if x["arm"] == "olmoe_htp")),
    dict(id="inapp_olmoe_gpu_prefill",
         text="the Adreno's prefill on the same model and campaign: 135.33 tok/s",
         artifact="app_engine.json", expected=135.33, tol=0.01,
         value=lambda A: next(x["prefill_tok_s_median"] for x in A["app18"]["arms"] if x["arm"] == "olmoe_gpu")),
    dict(id="device_ceiling_qwen3_cpu",
         text="the CPU's measured weight-byte throughput (31.01 GB/s) puts a ceiling of 16.85 tok/s on Qwen3-30B-A3B, so the 10 tok/s goal needs 59% of what the CPU alone can deliver",
         artifact="device_bandwidth.json", expected=16.8538, tol=0.01,
         value=lambda A: next(x["implied_target_tok_s"] for x in A["devbw"]["devices"] if x["arm"] == "olmoe_cpu")),
    dict(id="device_ceiling_qwen3_gpu",
         text="the Adreno's 34.85 GB/s puts the highest single-device ceiling on Qwen3-30B-A3B: 18.94 tok/s",
         artifact="device_bandwidth.json", expected=18.9423, tol=0.01,
         value=lambda A: next(x["implied_target_tok_s"] for x in A["devbw"]["devices"] if x["arm"] == "olmoe_gpu")),
    dict(id="device_ceiling_qwen3_htp",
         text="the NPU's 28.41 GB/s puts its Qwen3-30B-A3B ceiling at 15.44 tok/s",
         artifact="device_bandwidth.json", expected=15.4381, tol=0.01,
         value=lambda A: next(x["implied_target_tok_s"] for x in A["devbw"]["devices"] if x["arm"] == "olmoe_htp")),
    dict(id="upstream_inapp_qwen3_best",
         text="upstream llama.cpp's expert-streaming branch, run in the same app process on the same model, reaches 2.36 tok/s at best (on the GPU); its CPU path reaches 0.12",
         artifact="app_engine.json", expected=2.36, tol=0.01,
         value=lambda A: max(x["decode_tok_s_median"] for x in A["app18"]["arms"] if x["arm"].startswith("qwen_"))),
    # ------------------- EXPERT CONSUMPTION ORDER (device/bmoe_order.sh, patch 0010, 2026-09-18)
    # A NEGATIVE, kept because the arithmetic that motivated it was right and the lever still was not
    # there: the two arms agree to within a fraction of their own spread.
    dict(id="order_idorder_decode",
         text="consuming a layer's experts in ascending expert id (the stock ggml order): 5.837 tok/s median over 3 rotated repeats",
         artifact="bmoe_order.json", expected=5.837, tol=0.01,
         value=lambda A: next(c["decode_tok_s_median"] for c in A["border"]["cells"] if c["cell"] == "idorder")),
    dict(id="order_residentfirst_decode",
         text="computing the already-resident experts first instead: 5.835 tok/s median, i.e. no measurable change (0.03%) against a within-arm spread of about 5%",
         artifact="bmoe_order.json", expected=5.835, tol=0.01,
         value=lambda A: next(c["decode_tok_s_median"] for c in A["border"]["cells"] if c["cell"] == "residentfirst")),
    dict(id="order2_residentfirst_decode",
         text="the instrumented repeat of the same A/B put resident-first ahead on every row, 6.151 tok/s against 5.804 (+6.0%) -- but that campaign's rotation was unbalanced and the effect is not believed; see bmoe_order/FINDING.md",
         artifact="bmoe_order2.json", expected=6.151, tol=0.01,
         value=lambda A: next(c["decode_tok_s_median"] for c in A["border2"]["cells"] if c["cell"] == "residentfirst")),
    dict(id="order_probe_deferred_pct",
         text="the reordering does fire: 9.68% of expert consumptions were deferred to the second pass (the three instrumented rows agree at 9.32 / 9.68 / 9.87%)",
         artifact="bmoe_order2.json", expected=9.68, tol=0.01,
         value=lambda A: next(c["probe_deferred_pct_median"] for c in A["border2"]["cells"] if c["cell"] == "residentfirst")),
    dict(id="order_cover_bound_pct",
         text="intra-layer reordering can buy at most 17.2% of a token: 0.58 ms of resident-expert compute per layer (wall time, at the CPU's measured 31.01 GB/s) against 1.17 ms of miss reads (at the measured 2.806 GB/s bulk flash rate)",
         artifact="order_cover.json", expected=0.1716, tol=0.002,
         value=lambda A: A["ocover"]["per_token"]["max_saving_fraction_of_decode"]),
    # ------------------------- EVICTION POLICY (gates/evict_sim.py, 2026-09-18)
    dict(id="evict_slru80_hit",
         text="segmented LRU (protected 80%) on the global expert cache at the phone's 30.7% cache fraction: 88.32% hit against LRU's 86.90%, removing 10.8% of the miss traffic and closing 18% of the LRU-to-Belady gap -- O(1), so it costs what LRU costs",
         artifact="evict_policies_qwen3.json", expected=0.8832, tol=0.0005,
         value=lambda A: next(r["slru80_hit"] for r in A["evict"]["rows"]
                              if abs(r["cache_fraction"] - 0.307) < 1e-9)),
    dict(id="evict_slru80_miss_reduction",
         text="that is 10.8% less flash traffic per token at the same cache size",
         artifact="evict_policies_qwen3.json", expected=0.1077, tol=0.001,
         value=lambda A: next(r["slru80_miss_reduction_vs_lru"] for r in A["evict"]["rows"]
                              if abs(r["cache_fraction"] - 0.307) < 1e-9)),
    dict(id="evict_freq_is_worse",
         text="online frequency eviction is WORSE than LRU on the same trace -- 83.89% against 86.90%, i.e. 23% MORE miss traffic -- which is the same verdict gates/expert_policy.py reached per-layer on a different model: stale popularity pollutes the cache",
         artifact="evict_policies_qwen3.json", expected=0.8389, tol=0.0005,
         value=lambda A: next(r["freq_hit"] for r in A["evict"]["rows"]
                              if abs(r["cache_fraction"] - 0.307) < 1e-9)),
    dict(id="evict_cycle_is_nothing",
         text="exploiting the fixed layer order in eviction is worth nothing measurable (86.99% against 86.90%): an entry's next use is dominated by when its layer next SELECTS it, which is many tokens away, not by the at-most-one-token distance to its layer's next visit",
         artifact="evict_policies_qwen3.json", expected=0.8699, tol=0.0005,
         value=lambda A: next(r["cycle_hit"] for r in A["evict"]["rows"]
                              if abs(r["cache_fraction"] - 0.307) < 1e-9)),
    # ------------------------- ORDERING DRIFT (gates/position_effect.py, diagnostic, 2026-09-18)
    dict(id="position_drift_median",
         text="across 12 rotated phone campaigns the median last-position/first-position decode ratio is 0.992, with 5 campaigns drifting up and 7 down: the drift is campaign-specific, not one shared bias",
         artifact="position_effect.json", expected=0.9922, tol=0.002,
         value=lambda A: A["posfx"]["median_last_over_first"]),
    # ------------------------------------------------------ instrument agreement
    # ------------------------------------------- HEADROOM (gates/headroom_sims.py, 2026-09-17)
    # The offline verdicts re-run at the phone's operating point (rho 2-4), plus the per-token
    # CSV analysis of the pinned confirmation. See HEADROOM.md.
    dict(id="hr_pinned_steady_tok_s",
         text="pinned t4/io4 decodes Qwen3-30B-A3B at 5.30 tok/s over tokens 65-256 (ESTIMAND §1 steady "
              "state; the 256-token mean the pin2 claims quote is 5.36)",
         artifact="headroom_sims.json", expected=5.30, tol=0.01,
         value=lambda A: A["hr"]["phone_csv"]["cells"]["pin_t4c47_io4c03"]["tok_s_steady_median"]),
    dict(id="hr_unpinned_steady_tok_s",
         text="unpinned t4/io4: 4.72 tok/s over tokens 65-256",
         artifact="headroom_sims.json", expected=4.72, tol=0.01,
         value=lambda A: A["hr"]["phone_csv"]["cells"]["unpinned_t4_io4"]["tok_s_steady_median"]),
    dict(id="hr_pinned_compute_ms",
         text="pinned steady state: 100 ms/token of 'compute' RESIDUAL (wall - stall - mgmt; BigMoeOnEdge "
              "telemetry.md: 'a residual, not a measured quantity')",
         artifact="headroom_sims.json", expected=100.2, tol=0.1,
         value=lambda A: A["hr"]["phone_csv"]["cells"]["pin_t4c47_io4c03"]["compute_ms_steady_median"]),
    dict(id="hr_pinned_stall_ms",
         text="pinned steady state: 54 ms/token of I/O stall",
         artifact="headroom_sims.json", expected=53.7, tol=0.1,
         value=lambda A: A["hr"]["phone_csv"]["cells"]["pin_t4c47_io4c03"]["stall_ms_steady_median"]),
    dict(id="hr_pinned_mgmt_ms",
         text="pinned steady state: 35 ms/token of cache management",
         artifact="headroom_sims.json", expected=35.0, tol=0.1,
         value=lambda A: A["hr"]["phone_csv"]["cells"]["pin_t4c47_io4c03"]["mgmt_ms_steady_median"]),
    dict(id="hr_pinned_stall_intercept_ms",
         text="regressing per-token stall on flash MiB (pinned): a fixed 25 ms/token that bytes do not "
              "explain -- the per-layer head-of-line latency",
         artifact="headroom_sims.json", expected=24.7, tol=0.1,
         value=lambda A: A["hr"]["phone_csv"]["cells"]["pin_t4c47_io4c03"]["stall_intercept_ms_median"]),
    dict(id="hr_pinned_stall_slope",
         text="and 0.151 ms per MiB on top (a marginal ~6.6 GB/s: overlap already hides part of each byte)",
         artifact="headroom_sims.json", expected=0.151, tol=0.001,
         value=lambda A: A["hr"]["phone_csv"]["cells"]["pin_t4c47_io4c03"]["stall_slope_ms_per_MiB_median"]),
    dict(id="hr_compute_drift_pinned",
         text="within a 256-token pinned run the compute residual rises 11% from the first 32 tokens to the "
              "last 32 (thermal drift inside the run)",
         artifact="headroom_sims.json", expected=1.107, tol=0.002,
         value=lambda A: A["hr"]["phone_csv"]["cells"]["pin_t4c47_io4c03"]["compute_drift_last32_over_first32_median"]),
    dict(id="hr_recycle_mgmt_ms",
         text="page recycling cuts cache management to 14 ms/token",
         artifact="headroom_sims.json", expected=14.0, tol=0.1,
         value=lambda A: A["hr"]["phone_csv"]["cells"]["pin_t4_recycle"]["mgmt_ms_steady_median"]),
    dict(id="hr_recycle_compute_ms",
         text="but the compute residual rises to 115 ms/token: the cost moved into the compute cores",
         artifact="headroom_sims.json", expected=115.2, tol=0.1,
         value=lambda A: A["hr"]["phone_csv"]["cells"]["pin_t4_recycle"]["compute_ms_steady_median"]),
    dict(id="hr_engine_rho",
         text="the BigMoeOnEdge run's 4000 MiB cache is rho 4.12 for Qwen3-30B-A3B (clamped to the "
              "OLMoE curve's last grid point, rho 4.0)",
         artifact="headroom_sims.json", expected=4.115, tol=0.001,
         value=lambda A: A["hr"]["engine_vs_curve"]["rho"]),
    dict(id="hr_engine_hit_vs_h_rho",
         text="its measured 0.780 hit rate is +0.009 from the equal-rho prediction (H_rho)",
         artifact="headroom_sims.json", expected=0.009, tol=0.001,
         value=lambda A: A["hr"]["engine_vs_curve"]["err_H_rho"]),
    dict(id="hr_engine_hit_vs_h_floor",
         text="and +0.266 from the floor-additive prediction (H_floor) the documents prefer",
         artifact="headroom_sims.json", expected=0.266, tol=0.001,
         value=lambda A: A["hr"]["engine_vs_curve"]["err_H_floor"]),
    dict(id="hr_union_c5_olmoe_rho4",
         text="at rho 4 the union of a 5-token window fetches 4.17x what one token fetches (OLMoE): bytes "
              "alone predict a near-linear verify cost",
         artifact="headroom_sims.json", expected=4.17, tol=0.01,
         value=lambda A: next(r for r in A["hr"]["union"]["OLMoE"]["rows"] if r["rho"] == 4.0)["c_io"]["5"]),
    dict(id="hr_union_c5_gptoss_rho4",
         text="4.29x on gpt-oss-20b at rho 4",
         artifact="headroom_sims.json", expected=4.29, tol=0.01,
         value=lambda A: next(r for r in A["hr"]["union"]["gpt-oss-20b"]["rows"] if r["rho"] == 4.0)["c_io"]["5"]),
    dict(id="hr_union_c2_olmoe_rho4",
         text="and a 2-token window fetches 1.90x (so even a free verify needs >1.9 accepted tokens per "
              "2-position pass on the I/O side)",
         artifact="headroom_sims.json", expected=1.90, tol=0.01,
         value=lambda A: next(r for r in A["hr"]["union"]["OLMoE"]["rows"] if r["rho"] == 4.0)["c_io"]["2"]),
    dict(id="hr_la_olmoe_rho4_gap4",
         text="at rho 4 (OLMoE) a 4-token lookahead closes only 47% of the LRU-to-Belady gap",
         artifact="headroom_sims.json", expected=0.472, tol=0.002,
         value=lambda A: next(r for r in A["hr"]["lookahead"]["OLMoE"]["rows"]
                              if abs(r["cache_fraction"] - 0.5) < 1e-9)["gap_closed_by_horizon"]["4"]),
    dict(id="hr_la_olmoe_rho4_gap16",
         text="and 16 tokens are needed to close 97% of it",
         artifact="headroom_sims.json", expected=0.970, tol=0.002,
         value=lambda A: next(r for r in A["hr"]["lookahead"]["OLMoE"]["rows"]
                              if abs(r["cache_fraction"] - 0.5) < 1e-9)["gap_closed_by_horizon"]["16"]),
    dict(id="hr_la_olmoe_rho24_gap4",
         text="at rho 2.4 a 4-token lookahead closes 69% of the gap (the 30% row POSITION §3c calls 'reaches Belady')",
         artifact="headroom_sims.json", expected=0.690, tol=0.002,
         value=lambda A: next(r for r in A["hr"]["lookahead"]["OLMoE"]["rows"]
                              if abs(r["cache_fraction"] - 0.3) < 1e-9)["gap_closed_by_horizon"]["4"]),
    dict(id="hr_la_gptoss_rho4_gap4",
         text="on gpt-oss-20b at rho 4, 45%",
         artifact="headroom_sims.json", expected=0.448, tol=0.002,
         value=lambda A: next(r for r in A["hr"]["lookahead"]["gpt-oss-20b"]["rows"]
                              if abs(r["cache_fraction"] - 0.5) < 1e-9)["gap_closed_by_horizon"]["4"]),
    dict(id="hr_gap_olmoe_rho4",
         text="the LRU-to-Belady gap at rho 4 is 12.5 pp on OLMoE",
         artifact="headroom_sims.json", expected=0.125, tol=0.001,
         value=lambda A: (lambda r: r["belady_hit"] - r["lru_hit"])(
             next(r for r in A["hr"]["miss_dist"]["OLMoE"]["rows"] if r["rho"] == 4.0))),
    dict(id="hr_gap_gptoss_rho4",
         text="4.8 pp on gpt-oss-20b",
         artifact="headroom_sims.json", expected=0.048, tol=0.001,
         value=lambda A: (lambda r: r["belady_hit"] - r["lru_hit"])(
             next(r for r in A["hr"]["miss_dist"]["gpt-oss-20b"]["rows"] if r["rho"] == 4.0))),
    dict(id="hr_gap_granite_rho4",
         text="and 3.2 pp on granite-3b: eviction policy is a small lever at phone cache sizes",
         artifact="headroom_sims.json", expected=0.032, tol=0.001,
         value=lambda A: (lambda r: r["belady_hit"] - r["lru_hit"])(
             next(r for r in A["hr"]["miss_dist"]["granite-3b"]["rows"] if r["rho"] == 4.0))),
    dict(id="hr_p_miss_olmoe_rho4",
         text="at rho 4 a layer-event has at least one miss with probability 0.745 (OLMoE)",
         artifact="headroom_sims.json", expected=0.745, tol=0.001,
         value=lambda A: next(r for r in A["hr"]["miss_dist"]["OLMoE"]["rows"] if r["rho"] == 4.0)["p_event_ge1_miss"]),
    dict(id="hr_p_miss_gptoss_rho4",
         text="0.279 on gpt-oss-20b",
         artifact="headroom_sims.json", expected=0.279, tol=0.001,
         value=lambda A: next(r for r in A["hr"]["miss_dist"]["gpt-oss-20b"]["rows"] if r["rho"] == 4.0)["p_event_ge1_miss"]),
    dict(id="hr_pf_rho24_fwd60",
         text="within-token prefetch at rho 2.4 with 60 ms of compute per token: 1.23x (the closed verdict was "
              "0.76x at rho 0.8 and 30 ms)",
         artifact="headroom_sims.json", expected=1.226, tol=0.002,
         value=lambda A: next(r for r in A["hr"]["prefetch"]["rows"]
                              if abs(r["cache_fraction"] - 0.3) < 1e-9 and r["fwd_ms"] == 60)["speedup"]),
    dict(id="hr_pf_rho24_fwd100",
         text="1.35x at 100 ms",
         artifact="headroom_sims.json", expected=1.348, tol=0.002,
         value=lambda A: next(r for r in A["hr"]["prefetch"]["rows"]
                              if abs(r["cache_fraction"] - 0.3) < 1e-9 and r["fwd_ms"] == 100)["speedup"]),
    dict(id="hr_pf_rho4_fwd60",
         text="at rho 4: 1.24x at 60 ms",
         artifact="headroom_sims.json", expected=1.241, tol=0.002,
         value=lambda A: next(r for r in A["hr"]["prefetch"]["rows"]
                              if abs(r["cache_fraction"] - 0.5) < 1e-9 and r["fwd_ms"] == 60)["speedup"]),
    dict(id="hr_pf_rho4_fwd100",
         text="and 1.24x at 100 ms",
         artifact="headroom_sims.json", expected=1.241, tol=0.002,
         value=lambda A: next(r for r in A["hr"]["prefetch"]["rows"]
                              if abs(r["cache_fraction"] - 0.5) < 1e-9 and r["fwd_ms"] == 100)["speedup"]),
    dict(id="hr_protect_rho4_acc05",
         text="a protect-mode veto at horizon 8 and 50% slot accuracy reaches 0.800 at rho 4 against LRU's 0.774",
         artifact="headroom_sims.json", expected=0.800, tol=0.002,
         value=lambda A: next(r for r in A["hr"]["lookahead"]["OLMoE"]["rows"]
                              if abs(r["cache_fraction"] - 0.5) < 1e-9)["prediction_at_horizon_8"]["protect"]["0.5"]),
    dict(id="hr_protect_rho24_acc05",
         text="and 0.649 at rho 2.4 against LRU's 0.593",
         artifact="headroom_sims.json", expected=0.649, tol=0.002,
         value=lambda A: next(r for r in A["hr"]["lookahead"]["OLMoE"]["rows"]
                              if abs(r["cache_fraction"] - 0.3) < 1e-9)["prediction_at_horizon_8"]["protect"]["0.5"]),
    dict(id="hr_lex_olmoe",
         text="the token id alone names 0.445 of OLMoE's expert slots (table fitted on half of wikitext, scored on "
              "the other half) against persistence 0.369 and independence 0.125",
         artifact="headroom_sims.json", expected=0.445, tol=0.002,
         value=lambda A: A["hr"]["lexical"]["OLMoE-q4-llamacpp"]["mean_lexical"]),
    dict(id="hr_lex_gptoss",
         text="0.479 on gpt-oss-20b (persistence 0.491: its late layers are contextual)",
         artifact="headroom_sims.json", expected=0.479, tol=0.002,
         value=lambda A: A["hr"]["lexical"]["gpt-oss-20b"]["mean_lexical"]),
    dict(id="hr_lex_granite",
         text="0.586 on granite-3b (persistence 0.473); its layer 0 is 0.803 lexical",
         artifact="headroom_sims.json", expected=0.586, tol=0.002,
         value=lambda A: A["hr"]["lexical"]["granite-3b"]["mean_lexical"]),
    dict(id="hr_q4_nibble_entropy_bits",
         text="the Q4_0 nibbles of Qwen3-30B-A3B's expert tensors carry 3.754 bits of zeroth-order entropy per "
              "4-bit code (6 tensors, 37748736 blocks)",
         artifact="headroom_sims.json", expected=3.754, tol=0.001,
         value=lambda A: A["hr"]["q4_entropy"]["nibble_entropy_bits_pooled"]),
    dict(id="hr_q4_scale_entropy_bits",
         text="and the fp16 block scales 11.52 bits of 16",
         artifact="headroom_sims.json", expected=11.52, tol=0.01,
         value=lambda A: A["hr"]["q4_entropy"]["scale_entropy_bits_mean"]),
    dict(id="hr_q4_max_lossless_saving",
         text="so an order-0 entropy coder can save at most 8.6% of expert bytes on flash (4.114 of 4.5 "
              "bits/weight): lossless expert compression is closed",
         artifact="headroom_sims.json", expected=0.086, tol=0.001,
         value=lambda A: A["hr"]["q4_entropy"]["max_lossless_saving_fraction"]),
    dict(id="roofline_inputs_agree",
         text="the bandwidth constant used here is the one the engine_sim artifact was run with",
         artifact="engine_sim_OLMoE-1B-7B-0924.json", expected=BULK_GBPS, tol=1e-9,
         value=lambda A: A["engine"]["inputs"]["bulk_gbps"]),
]


def load_all():
    return {
        "cache": load("cache_OLMoE-1B-7B-0924.json"),
        "pred": load("cache_pred_OLMoE-1B-7B-0924.json"),
        "engine": load("engine_sim_OLMoE-1B-7B-0924.json"),
        "target": load("engine_target.json"),
        "scope": load("scope_compare_OLMoE-1B-7B-0924.json"),
        "policy": load("expert_policy_OLMoE-1B-7B-0924.json"),
        "predictor": load("predictor_OLMoE-1B-7B-0924.json"),
        "dram": load("dram_15r.json"),
        "bb": load("byte_budget_measured.json"),
        "ext": load("external_validation_colibri.json"),
        "g1": load("g1_storage.json"),
        "pi2": load("byte_budget_validate_powerinfer2.json"),
        "s11": load("s11_nonflash_15r.json"),
        "dec": load("decode_15r.json"),
        "g1qd": load("g1_storage_qd.json"),
        "dec2": load("decode_15r_cpu.json"),
        "thr": load("decode_15r_threads.json"),
        "s9r": load("s9_result_Qwen3-30B-A3B.json"),
        "conf": load("traces_confound_olmoe_q4_vs_fp16.json"),
        "s9g": load("s9_result_granite-3.1-1b-a400m.json"),
        "pf": load("prefetch_sim_OLMoE-1B-7B-0924.json"),
        "pin": load("decode_15r_pin.json"),
        "bmoe": load("bmoe_repro.json"),
        "levers": load("bmoe_levers.json"),
        "repack": load("bmoe_repack.json"),
        "bpin2": load("bmoe_pin2.json"),
        "bcache": load("bmoe_cache.json"),
        "ctrace": load("compute_trace.json"),
        "qsizes": load("cache_qwen3_phone_sizes.json"),
        "cacheo": load("cache_OLMoE-1B-7B-0924.json"),
        "vcost": load("verify_cost.json"),
        "hr": load("headroom_sims.json"),
        "act18": load("gguf_active_qwen_olmoe.json"),
        "app18": load("app_engine.json"),
        "devbw": load("device_bandwidth.json"),
        "border": load("bmoe_order.json"),
        "border2": load("bmoe_order2.json"),
        "ocover": load("order_cover.json"),
        "posfx": load("position_effect.json"),
        "evict": load("evict_policies_qwen3.json"),
    }


def check():
    A = load_all()
    bad = []
    out = []
    for c in CLAIMS:
        try:
            got = c["value"](A)
        except Exception as exc:                      # counted and reported, never swallowed
            bad.append((c["id"], "ERROR", repr(exc)))
            out.append((c, None, "ERROR"))
            continue
        ok = got is not None and abs(got - c["expected"]) <= c["tol"]
        if not ok:
            bad.append((c["id"], c["expected"], got))
        out.append((c, got, "ok" if ok else "DRIFT"))
    return A, out, bad


def emit_markdown(out):
    lines = [
        "# moe-phone — claims map",
        "",
        "**Generated by `gates/claims_check.py --emit`. Do not edit by hand.**",
        "",
        "Every number asserted in prose across this project, the artifact it is read from, and",
        "the value in that artifact right now. `CLAUDE.md` §7.1 requires one claim, one script,",
        "one artifact; this is the map, and `tests/test_gates.py` fails if any row drifts.",
        "",
        "This checks that the **prose matches the artifacts**. It cannot check that an artifact",
        "is *right*: the two defects retracted in `POSITION.md` §3c produced artifacts that were",
        "internally consistent and wrong. That is what the property tests are for.",
        "",
        "| id | claim | artifact | value | status |",
        "|---|---|---|---|---|",
    ]
    for c, got, status in out:
        shown = "—" if got is None else f"{got:.4f}"
        lines.append(f"| `{c['id']}` | {c['text']} | `{c['artifact']}` | {shown} | {status} |")
    lines += ["", f"{len(out)} claims checked."]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--emit", action="store_true", help="write CLAIMS.md and exit")
    a = p.parse_args()
    A, out, bad = check()
    path = os.path.join(os.path.dirname(HERE), "CLAIMS.md")
    if a.emit:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(emit_markdown(out))
        print("wrote", path)
    for c, got, status in out:
        shown = "   ERROR" if got is None else f"{got:9.4f}"
        print(f"  [{status:5s}] {c['id']:<34} {shown}  (documents say {c['expected']})")
    print(f"\n{len(out) - len(bad)}/{len(out)} claims agree with their artifacts")
    if bad:
        print("\nDRIFT — a document asserts a number its artifact no longer produces:")
        for cid, exp, got in bad:
            print(f"  {cid}: documents say {exp}, artifact says {got}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
