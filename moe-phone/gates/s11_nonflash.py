"""
S11 — the non-flash cost of a decoded token on the 15R, measured on a resident MoE.

The flash-only formula (ARCHITECTURE §1) and the serial flash + DRAM figure both
assume the bytes a token touches in DRAM stream at DRAM bandwidth. A fully
resident MoE on the same phone measures what they actually cost:

  effective_nonflash_GBps = active_bytes_per_token x steady-state tok/s

with active_bytes_per_token read from the GGUF's own tensor table
(gates/gguf_active.py) and tok/s the MARGINAL steady-state rate of a resident run
with zero flash bytes (gates/decode_analyze.py). Its ratio to the measured DRAM
bandwidth (S1) is the fraction of DRAM speed the engine's matmuls achieve.

ASSUMPTION, stated and testable: the non-flash time scales linearly with active
bytes at fixed engine, format, threads and device. Under it, the same rate predicts
the RESIDENT decode rate of any other model in the same format — reported here for
OLMoE as a prediction for the next clean resident OLMoE run to confirm or refute.

TEST (added after the 2026-09-16 01:19 campaign, which produced the first
flash-free steady-state OLMoE runs): the linear-in-bytes rule is scored at
MATCHED thread count — the reference model's rate at T threads predicts the
target's rate at T threads — against the observed marginal rate. A comparison
across thread counts would test something else.

Run:
  python moe-phone/gates/s11_nonflash.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import read_path, write_path  # noqa: E402

REF_MODEL = "granite-3.1-1b-a400m-instruct-Q4_0.gguf"
PRED_MODEL = "olmoe-1b-7b-0924-q4_0.gguf"
THREADS = 4


def main():
    dec = json.load(open(read_path("decode_15r.json"), encoding="utf-8"))
    act = {x["file"]: x for x in json.load(open(read_path("gguf_active.json"), encoding="utf-8"))}
    dram = json.load(open(read_path("dram_15r.json"), encoding="utf-8"))
    g = next(x for x in dec["groups"]
             if x["model"] == REF_MODEL and x["threads"] == THREADS and x["mode"] == "warm")
    if g["marginal_flash_MB_per_tok_median"] is None or g["marginal_flash_MB_per_tok_median"] > 1.0:
        raise ValueError("reference run is not flash-free; it cannot measure the non-flash term")
    tps = g["marginal_tok_s_median"]
    ab = act[REF_MODEL]["active_bytes_per_token"]
    eff = ab * tps / 1e9
    dram_gbps = dram["runs"]["app"]["best"]["GBps_median_clean"]
    pa = act[PRED_MODEL]["active_bytes_per_token"]
    out = {
        "reference": {"model": REF_MODEL, "threads": THREADS, "marginal_tok_s": tps,
                      "active_bytes_per_token": ab, "n_pairs": len(g["pairs"])},
        "effective_nonflash_GBps": eff,
        "dram_GBps_measured": dram_gbps,
        "fraction_of_dram": eff / dram_gbps,
        "nonflash_s_per_GB": 1.0 / eff,
        "prediction": {"model": PRED_MODEL, "active_bytes_per_token": pa,
                       "resident_tok_s_predicted": eff * 1e9 / pa,
                       "resident_tok_s_dram_only_bound": dram_gbps * 1e9 / pa,
                       "note": "resident = every weight in DRAM; linear-in-active-bytes assumption"},
    }
    print(f"{REF_MODEL} resident, {THREADS} threads: {tps:.1f} tok/s x {ab / 1e6:.1f} MB/token "
          f"= {eff:.1f} GB/s effective ({100 * eff / dram_gbps:.0f}% of the {dram_gbps:.1f} GB/s DRAM probe)")
    print(f"predicted resident {PRED_MODEL}: {out['prediction']['resident_tok_s_predicted']:.1f} tok/s "
          f"(DRAM-only bound {out['prediction']['resident_tok_s_dram_only_bound']:.1f})")
    # ---- the test: matched-thread predictions against observed OLMoE runs ----------
    obs_groups = []
    for name in ("decode_15r.json", "decode_15r_cpu.json"):
        try:
            obs_groups += [dict(g, _artifact=name)
                           for g in json.load(open(read_path(name), encoding="utf-8"))["groups"]]
        except FileNotFoundError:
            pass
    ref_rate = {}
    for g in obs_groups:
        if g["model"] == REF_MODEL and g["marginal_tok_s_median"] and \
                (g["marginal_flash_MB_per_tok_median"] or 0) < 1.0:
            ref_rate.setdefault(g["threads"], []).extend(x for x in g["marginal_tok_s_all"] if x)
    tests = []
    for g in obs_groups:
        if g["model"] != PRED_MODEL or g["mode"] != "warm" or g["threads"] not in ref_rate:
            continue
        rr = sorted(ref_rate[g["threads"]])
        ref_med = rr[len(rr) // 2] if len(rr) % 2 else 0.5 * (rr[len(rr) // 2 - 1] + rr[len(rr) // 2])
        pred = ref_med * ab / pa
        obs = g["marginal_tok_s_median"]
        tests.append({"threads": g["threads"], "artifact": g["_artifact"],
                      "reference_marginal_tok_s": ref_med, "n_reference_pairs": len(rr),
                      "predicted_tok_s": pred, "observed_marginal_tok_s": obs,
                      "observed_all": g["marginal_tok_s_all"],
                      "observed_marginal_flash_MB_per_tok": g["marginal_flash_MB_per_tok_median"],
                      "ratio_observed_over_predicted": obs / pred if obs else None})
        print(f"TEST {g['threads']} threads: predicted {pred:.1f} tok/s from {REF_MODEL} at "
              f"{ref_med:.1f}; observed {obs:.1f} ({g['_artifact']}); ratio {obs / pred:.2f}")
    out["matched_thread_tests"] = tests
    path = write_path("s11_nonflash_15r.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
