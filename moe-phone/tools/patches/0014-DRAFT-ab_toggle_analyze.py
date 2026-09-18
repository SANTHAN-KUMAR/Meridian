"""
AB-TOGGLE ANALYZE — read an in-process A/B run (BigMoeOnEdge + moe-phone patch 0014, --ab-toggle N
--ab-lever NAME) and decide whether the lever moved the rate, or say it cannot.

Why this design exists (research/2026-09-18_why_levers_flip.md, CLAUDE.md §4.1): between separate
runs on the 15R the clock governor alone gives a 5.5% run-to-run sd, so n=3 per arm resolves only
~10% effects while every lever tested is 3-9%. With --ab-toggle the two arms alternate every N decode
tokens inside ONE process, sharing its cache, clocks and thermal state. The unit of replication is the
BLOCK (N consecutive tokens of one arm), not the run.

Input: the per-token CSVs written by `bmoe-cli --csv` (0014 adds the columns `arm`,
`tokens_since_switch`, `spec_issued`, `spec_useful`). Several CSVs (repeats) are analysed separately
and pooled; blocks are never paired across files.

What is computed, per file and pooled:

  exclusions   the first --cold decode tokens (default 32: the cache is filling and neither arm is at
               steady state), and the first W tokens after every switch (carryover: the cache and the
               speculation queue still reflect the other arm). W = --carryover (default 4); the whole
               analysis is ALSO run at W = 0 and W = 8 and reported, so a reader can see the verdict
               does not depend on the value chosen.
  per arm      per-token mean and median of wall_ms, stall_ms, compute_ms (a residual), mgmt_ms,
               flash MiB, spec_issued, spec_useful, over the kept tokens.
  paired       a block-level comparison. Each B block is compared with the mean of the A blocks on
               either side of it (both when present), which cancels a linear drift in clock or cache
               state across the pair; the adjacent-pair form (A_i then B_i) is reported beside it.
               diff = B - A per block (ms/token for wall and each term, MiB/token for flash), mean over
               pairs, paired SE = sd(diffs) / sqrt(n_pairs), and SE/|diff| (§4.1).
  verdict      on wall time (the rate): reported only if SE/|diff| < 0.5; otherwise "not resolved",
               with the number of block pairs the same effect would need to reach 0.5 (assuming the
               effect and the between-pair sd stay as measured, which is an assumption, not a result).

Nothing is fitted. The cold window, the carryover W and the 0.5 threshold are fixed here, before any
toggled row exists, and W's influence is shown rather than chosen.

Run:
  python moe-phone/gates/ab_toggle_analyze.py RUN1.csv [RUN2.csv ...] --tag specgate_r2 [--out-dir DIR]
  python moe-phone/gates/ab_toggle_analyze.py --selftest
"""
import argparse
import csv
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

METRICS = ("wall_ms", "stall_ms", "compute_ms", "mgmt_ms", "flash_mib", "spec_issued", "spec_useful")
SE_OVER_DIFF_MAX = 0.5
COLD_DEFAULT = 32
CARRYOVER_DEFAULT = 4
CARRYOVER_SHOWN = (0, 4, 8)


def load(path):
    """Rows of one --csv file as dicts of floats (arm kept as a string). Refuses a file without the
    0014 columns: a CSV from a non-toggled run has no arms to compare."""
    with open(path, encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#")]
    rows = list(csv.DictReader(lines))
    if not rows or "arm" not in rows[0] or "tokens_since_switch" not in rows[0]:
        raise ValueError(f"{path}: no 'arm'/'tokens_since_switch' columns - not an --ab-toggle run (patch 0014)")
    out = []
    for r in rows:
        if r["arm"] not in ("A", "B"):
            raise ValueError(f"{path}: row step={r.get('step')} has arm {r['arm']!r}; every row of a toggled "
                             "run must be A or B")
        d = {"step": int(float(r["step"])), "arm": r["arm"], "since": int(float(r["tokens_since_switch"]))}
        for k in ("wall_ms", "stall_ms", "compute_ms", "mgmt_ms", "spec_issued", "spec_useful"):
            d[k] = float(r[k])
        d["flash_mib"] = float(r["read_bytes"]) / 1048576.0
        d["mtp_batch"] = int(float(r.get("mtp_batch", 1) or 1))
        out.append(d)
    if any(d["mtp_batch"] != 1 for d in out):
        raise ValueError(f"{path}: a row has mtp_batch != 1; the toggle is defined for single-token decode only")
    return out


def blocks_of(rows):
    """Consecutive runs of one arm, in order: [(arm, [rows...]), ...]. A block starts where the arm
    changes; the since-counter is the engine's own and is checked against that."""
    blocks = []
    for r in rows:
        if blocks and blocks[-1][0] == r["arm"]:
            blocks[-1][1].append(r)
        else:
            if r["since"] != 0 and blocks:
                raise ValueError(f"step {r['step']}: arm changed but tokens_since_switch={r['since']} (expected 0)")
            blocks.append((r["arm"], [r]))
    return blocks


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def arm_stats(rows, cold, w):
    kept = [r for r in rows if r["step"] > cold and r["since"] >= w]
    res = {}
    for arm in ("A", "B"):
        rs = [r for r in kept if r["arm"] == arm]
        res[arm] = {"n_tokens": len(rs)}
        for m in METRICS:
            xs = [r[m] for r in rs]
            res[arm][m + "_mean"] = mean(xs)
            res[arm][m + "_median"] = statistics.median(xs) if xs else None
        res[arm]["tok_s_from_mean_wall"] = 1000.0 / res[arm]["wall_ms_mean"] if rs and res[arm]["wall_ms_mean"] else None
    return res


def paired(rows, cold, w):
    """Block-level paired differences B - A, sandwich and adjacent forms, per metric."""
    blocks = []
    for arm, rs in blocks_of(rows):
        keep = [r for r in rs if r["step"] > cold and r["since"] >= w]
        if keep:
            blocks.append((arm, {m: mean([r[m] for r in keep]) for m in METRICS}, len(keep)))
    sandwich, adjacent = [], []
    for i, (arm, vals, _) in enumerate(blocks):
        if arm != "B":
            continue
        nb = [blocks[j][1] for j in (i - 1, i + 1) if 0 <= j < len(blocks) and blocks[j][0] == "A"]
        if nb:
            sandwich.append({m: vals[m] - mean([a[m] for a in nb]) for m in METRICS})
        if i >= 1 and blocks[i - 1][0] == "A":
            adjacent.append({m: vals[m] - blocks[i - 1][1][m] for m in METRICS})
    return summarise(sandwich), summarise(adjacent), len(blocks)


def summarise(diffs):
    out = {"n_pairs": len(diffs)}
    for m in METRICS:
        xs = [d[m] for d in diffs]
        if len(xs) >= 2:
            mu = mean(xs)
            se = statistics.stdev(xs) / math.sqrt(len(xs))
            ratio = se / abs(mu) if mu != 0 else math.inf
        else:
            mu = mean(xs) if xs else None
            se, ratio = None, None
        out[m] = {"diff_B_minus_A": mu, "paired_se": se, "se_over_abs_diff": ratio}
    return out


def verdict(pairs, arms):
    w = pairs["wall_ms"]
    n = pairs["n_pairs"]
    if w["se_over_abs_diff"] is None:
        return {"resolved": False, "text": f"not resolved: {n} block pair(s), need at least 2 for an SE"}
    ratio = w["se_over_abs_diff"]
    a_ms = arms["A"]["wall_ms_mean"]
    rel = w["diff_B_minus_A"] / a_ms if a_ms else None
    if ratio < SE_OVER_DIFF_MAX:
        faster = "A" if w["diff_B_minus_A"] > 0 else "B"
        return {"resolved": True, "faster_arm": faster, "wall_diff_B_minus_A_ms": w["diff_B_minus_A"],
                "wall_diff_rel_to_A": rel, "se_over_abs_diff": ratio, "n_pairs": n,
                "text": f"arm {faster} faster: B-A = {w['diff_B_minus_A']:+.2f} ms/token "
                        f"({100 * rel:+.1f}% of A), SE/|diff| = {ratio:.2f} over {n} block pairs"}
    need = math.ceil(n * (ratio / SE_OVER_DIFF_MAX) ** 2) if math.isfinite(ratio) else None
    return {"resolved": False, "wall_diff_B_minus_A_ms": w["diff_B_minus_A"], "wall_diff_rel_to_A": rel,
            "se_over_abs_diff": ratio, "n_pairs": n, "n_pairs_needed_for_0_5": need,
            "text": f"not resolved: B-A = {w['diff_B_minus_A']:+.2f} ms/token ({100 * rel:+.1f}% of A) with "
                    f"SE/|diff| = {ratio:.2f} over {n} block pairs; ~{need} pairs needed at this effect"
                    if need is not None else "not resolved: zero mean difference"}


def analyse(rows, cold, w):
    arms = arm_stats(rows, cold, w)
    sw, adj, nb = paired(rows, cold, w)
    return {"cold_tokens": cold, "carryover": w, "n_blocks_kept": nb, "arms": arms,
            "paired_sandwich": sw, "paired_adjacent": adj, "verdict": verdict(sw, arms)}


def run(paths, cold, w_primary):
    per_file = {}
    pooled_rows_by_file = []
    for p in paths:
        rows = load(p)
        pooled_rows_by_file.append(rows)
        per_file[os.path.basename(p)] = {str(w): analyse(rows, cold, w) for w in sorted({w_primary, *CARRYOVER_SHOWN})}
    # Pooled: pairs from every file, never across files; arm stats over all kept tokens.
    pooled = {}
    for w in sorted({w_primary, *CARRYOVER_SHOWN}):
        sw_all, adj_all, all_kept = [], [], []
        for rows in pooled_rows_by_file:
            all_kept.extend(rows)
            blocks = []
            for arm, rs in blocks_of(rows):
                keep = [r for r in rs if r["step"] > cold and r["since"] >= w]
                if keep:
                    blocks.append((arm, {m: mean([r[m] for r in keep]) for m in METRICS}))
            for i, (arm, vals) in enumerate(blocks):
                if arm != "B":
                    continue
                nb = [blocks[j][1] for j in (i - 1, i + 1) if 0 <= j < len(blocks) and blocks[j][0] == "A"]
                if nb:
                    sw_all.append({m: vals[m] - mean([a[m] for a in nb]) for m in METRICS})
                if i >= 1 and blocks[i - 1][0] == "A":
                    adj_all.append({m: vals[m] - blocks[i - 1][1][m] for m in METRICS})
        arms = {}
        for arm in ("A", "B"):
            rs = [r for r in all_kept if r["arm"] == arm and r["step"] > cold and r["since"] >= w]
            arms[arm] = {"n_tokens": len(rs), **{m + "_mean": mean([r[m] for r in rs]) for m in METRICS},
                         **{m + "_median": (statistics.median([r[m] for r in rs]) if rs else None) for m in METRICS}}
        sw, adj = summarise(sw_all), summarise(adj_all)
        pooled[str(w)] = {"cold_tokens": cold, "carryover": w, "arms": arms, "paired_sandwich": sw,
                          "paired_adjacent": adj, "verdict": verdict(sw, arms)}
    return {"inputs": [os.path.abspath(p) for p in paths], "cold_tokens": cold, "carryover_primary": w_primary,
            "carryover_shown": list(CARRYOVER_SHOWN), "se_over_diff_max": SE_OVER_DIFF_MAX,
            "per_file": per_file, "pooled": pooled, "verdict": pooled[str(w_primary)]["verdict"]}


def print_report(res):
    for w in sorted(res["pooled"], key=int):
        p = res["pooled"][w]
        a, b = p["arms"]["A"], p["arms"]["B"]
        tag = " (primary)" if int(w) == res["carryover_primary"] else ""
        print(f"\nW = {w}{tag}: kept A {a['n_tokens']} tokens, B {b['n_tokens']} tokens; "
              f"{p['paired_sandwich']['n_pairs']} sandwich pairs")
        print(f"  {'metric':<12}{'A mean':>10}{'B mean':>10}{'B-A':>10}{'SE':>9}{'SE/|d|':>8}")
        for m in METRICS:
            s = p["paired_sandwich"][m]
            fmt = lambda v: f"{v:10.2f}" if v is not None else f"{'-':>10}"  # noqa: E731
            r = s["se_over_abs_diff"]
            se = f"{s['paired_se']:9.2f}" if s["paired_se"] is not None else f"{'-':>9}"
            rr = f"{r:8.2f}" if r is not None and math.isfinite(r) else f"{'-':>8}"
            print(f"  {m:<12}{fmt(a[m + '_mean'])}{fmt(b[m + '_mean'])}{fmt(s['diff_B_minus_A'])}{se}{rr}")
        print("  verdict:", p["verdict"]["text"])


# ---------------------------------------------------------------- self-test (math-fixed expectations)
def _synthetic(n_tokens, N, a_ms, b_ms, noise, drift=0.0, seed=0):
    import random
    rnd = random.Random(seed)
    rows = []
    for i in range(n_tokens):
        arm = "A" if (i // N) % 2 == 0 else "B"
        base = a_ms if arm == "A" else b_ms
        w = base + drift * i + rnd.gauss(0, noise)
        rows.append({"step": i + 1, "arm": arm, "since": i % N, "wall_ms": w, "stall_ms": 0.0,
                     "compute_ms": w, "mgmt_ms": 0.0, "flash_mib": 0.0, "spec_issued": 0.0,
                     "spec_useful": 0.0, "mtp_batch": 1})
    return rows


def selftest():
    fails = 0

    def check(cond, msg):
        nonlocal fails
        print(("ok   " if cond else "FAIL ") + msg)
        fails += 0 if cond else 1

    # 1. identical arms: the verdict must not claim a difference
    r = analyse(_synthetic(2048, 16, 150.0, 150.0, 10.0), 32, 4)
    check(not r["verdict"]["resolved"], "identical arms -> not resolved")
    # 2. a 10 ms effect with modest noise is recovered within 1 ms and resolved
    r = analyse(_synthetic(2048, 16, 150.0, 160.0, 5.0), 32, 4)
    d = r["paired_sandwich"]["wall_ms"]["diff_B_minus_A"]
    check(r["verdict"]["resolved"] and abs(d - 10.0) < 1.0, f"10 ms effect recovered ({d:.2f}) and resolved")
    # 3. a strong linear drift is cancelled by the sandwich pairing (adjacent pairing is biased by it)
    rows = _synthetic(2048, 16, 150.0, 150.0, 0.5, drift=0.05)
    r = analyse(rows, 32, 0)
    ds = r["paired_sandwich"]["wall_ms"]["diff_B_minus_A"]
    da = r["paired_adjacent"]["wall_ms"]["diff_B_minus_A"]
    check(abs(ds) < 0.3 and abs(da) > 0.5, f"drift: sandwich {ds:.2f} ~ 0, adjacent {da:.2f} biased")
    # 4. carryover exclusion removes exactly W tokens per block from each arm
    r0 = analyse(_synthetic(512, 16, 150.0, 150.0, 1.0), 0, 0)
    r4 = analyse(_synthetic(512, 16, 150.0, 150.0, 1.0), 0, 4)
    check(r0["arms"]["A"]["n_tokens"] - r4["arms"]["A"]["n_tokens"] == 16 * 4,
          "carryover W=4 drops 4 tokens from each of 16 A blocks")
    # 5. a CSV without the 0014 columns is refused
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("step,steps,wall_ms\n1,2,3\n")
    try:
        load(f.name)
        check(False, "a non-toggled CSV is refused")
    except ValueError:
        check(True, "a non-toggled CSV is refused")
    os.unlink(f.name)
    print("all passed" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("csv", nargs="*")
    p.add_argument("--tag", default=None, help="artifact name suffix: ab_toggle_<tag>.json")
    p.add_argument("--cold", type=int, default=COLD_DEFAULT)
    p.add_argument("--carryover", type=int, default=CARRYOVER_DEFAULT)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    if not a.csv or not a.tag:
        p.error("give one or more --csv files and --tag (or --selftest)")
    res = run(a.csv, a.cold, a.carryover)
    print_report(res)
    name = f"ab_toggle_{a.tag}.json"
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        path = os.path.join(a.out_dir, name)
    else:
        from _paths import write_path
        path = write_path(name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
