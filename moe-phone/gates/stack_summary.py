"""
The stacked-levers comparison (device/bmoe_stack.sh), scored exactly as pre-registered in its header.
Written BEFORE any stacked row existed.

Power, from 24 existing rows (arena2 base, specadopt, slru_decode; 2026-09-18): row sd of stall+mgmt is
~3.6 ms (base rows), of compute ~5.5 ms, and the decode rate's coefficient of variation ~4.7%. With 6 ABBA
repeats (12 rows per arm) the paired design's SE on stall+mgmt is ~1.5-2 ms against an expected ~10 ms
effect, which is decisive; on decode it is ~2% against an expected ~4.5%, which is borderline -- hence
stall+mgmt is PRIMARY and decode SECONDARY.

Per repeat r (rows <r>a and <r>b of each arm, kept rows only), the paired difference is
    d_r = mean(stack rows of r) - mean(base rows of r)
and the estimate is mean(d_r) with SE = sd(d_r)/sqrt(n_repeats). ABBA cancels a linear drift within each
repeat, so this SE reflects the noise the design actually leaves.
DECISIVE iff SE/|mean| < 0.5 AND sign(d_r) == sign(mean) in >= 5 of 6 repeats. Otherwise "not resolved",
with n_needed = n_repeats * (SE/|mean| / 0.5)**2 repeats for SE/|mean| = 0.5 at the same noise.
Run:
  python moe-phone/gates/stack_summary.py results/2026-09-18/bmoe_stack/bmoe_stack_<stamp> \
      --out results/2026-09-18/stack_summary.json
"""
import argparse, glob, json, math, os, re, statistics as S


def num(pat, text, cast=float):
    m = re.search(pat, text)
    return cast(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out", required=True)
    # SENSITIVITY ONLY: the pre-registered keep rule is budget == 5000 MiB (the default here). --min-budget relaxes it,
    # and any output produced with it is a labelled sensitivity analysis, never the pre-registered verdict.
    ap.add_argument("--min-budget", type=int, default=None)
    # PRE-REGISTERED ALTERNATIVE for campaigns whose arms differ in budget BY DESIGN (device/bmoe_gtier.sh: the GPU tier
    # takes its MiB out of the CPU cache): keep = exit 0 and foreign=[], whatever the granted budget.
    ap.add_argument("--any-budget-prereg", action="store_true")
    # PRE-REGISTERED 2026-09-19 04:55 for runs after the Dozing confound (device/bmoe_gtier.sh REVISION): a row is kept only
    # if the phone was Awake when the engine started (wake_start=Awake) and when it exited (AFTER wake=Awake)
    ap.add_argument("--require-awake", action="store_true")
    a = ap.parse_args()
    log = open(os.path.join(a.root, "log.txt")).read()
    foreign = dict(re.findall(r"=== (\w+) \S+ .*?foreign=\[([^\]]*)\]", log))
    wake_start = dict(re.findall(r"=== (\w+) \S+ .*?wake_start=(\S+)", log))
    # the AFTER state of each row: the exit line that follows its === line
    wake_end, cur = {}, None
    for line in log.splitlines():
        m = re.match(r"=== (\w+) ", line)
        if m: cur = m.group(1)
        m = re.match(r"exit=\S+ AFTER wake=(\S+)", line)
        if m and cur: wake_end[cur] = m.group(1)
    rows = []
    for f in sorted(glob.glob(os.path.join(a.root, "*.out"))):
        tag = os.path.basename(f)[:-4]
        cell, rep, slot = re.match(r"(base|stack)_rep(\d+)([ab])", tag).groups()
        t = open(f).read() + open(f[:-4] + ".err").read()
        r = dict(tag=tag, cell=cell, rep=int(rep), slot=slot,
                 decode_tok_s=num(r"\(([0-9.]+) tok/s\)", t),
                 read_MiB_per_token=num(r"\(([0-9.]+) MiB/token\)", t),
                 hit_pct=num(r"([0-9.]+)% hit", t),
                 budget_MiB=num(r"budget (\d+) MiB", t, int),
                 stall_ms=num(r"stall ([0-9.]+) s/token", t),
                 compute_ms=num(r"compute ([0-9.]+) \+", t),
                 mgmt_ms=num(r"cache mgmt ([0-9.]+) \+", t),
                 foreign=foreign.get(tag))
        for k in ("stall_ms", "compute_ms", "mgmt_ms"):
            if r[k] is not None: r[k] *= 1000.0
        if r["stall_ms"] is None and r["decode_tok_s"] is not None:
            r["stall_ms"] = 0.0  # a run without the overlap line reports no stall term; counted, not hidden
            r["stall_missing"] = True
        r["stall_plus_mgmt_ms"] = (r["stall_ms"] or 0.0) + (r["mgmt_ms"] or 0.0) if r["mgmt_ms"] is not None else None
        if a.any_budget_prereg:
            budget_ok = r["budget_MiB"] is not None
        elif a.min_budget is None:
            budget_ok = r["budget_MiB"] == 5000
        else:
            budget_ok = (r["budget_MiB"] or 0) >= a.min_budget
        r["wake_start"], r["wake_end"] = wake_start.get(tag), wake_end.get(tag)
        awake_ok = (not a.require_awake) or (r["wake_start"] == "Awake" and r["wake_end"] == "Awake")
        r["kept"] = r["decode_tok_s"] is not None and r["foreign"] == "" and budget_ok and awake_ok
        rows.append(r)
    kept = [r for r in rows if r["kept"]]
    reps = sorted({r["rep"] for r in rows})

    def paired(key):
        d = []
        for rp in reps:
            s = [r[key] for r in kept if r["rep"] == rp and r["cell"] == "stack" and r[key] is not None]
            b = [r[key] for r in kept if r["rep"] == rp and r["cell"] == "base" and r[key] is not None]
            if s and b: d.append(S.mean(s) - S.mean(b))
        if len(d) < 2: return dict(n_repeats=len(d), verdict="insufficient repeats")
        m, se = S.mean(d), S.stdev(d) / math.sqrt(len(d))
        ratio = se / abs(m) if m else float("inf")
        sign_ok = sum((x > 0) == (m > 0) for x in d)
        decisive = ratio < 0.5 and sign_ok >= len(d) - 1
        out = dict(n_repeats=len(d), per_repeat=d, mean_diff=m, se=se, se_over_abs_mean=ratio,
                   sign_holds_in=sign_ok, decisive=decisive,
                   base_mean=S.mean(r[key] for r in kept if r["cell"] == "base" and r[key] is not None),
                   stack_mean=S.mean(r[key] for r in kept if r["cell"] == "stack" and r[key] is not None))
        if not decisive:
            out["n_repeats_needed_for_0_5"] = math.ceil(len(d) * (ratio / 0.5) ** 2) if math.isfinite(ratio) else None
        return out

    res = {k: paired(k) for k in ("stall_plus_mgmt_ms", "compute_ms", "decode_tok_s", "read_MiB_per_token",
                                  "hit_pct", "stall_ms", "mgmt_ms")}
    dec = res["decode_tok_s"]
    out = dict(source=os.path.abspath(a.root), require_awake=a.require_awake, keep_rule=("exit 0 + foreign=[], any budget (pre-registered: budgets differ by design)" if a.any_budget_prereg else
                          "budget == 5000 (pre-registered)" if a.min_budget is None else f"SENSITIVITY: budget >= {a.min_budget}"), stack_flags=re.search(r"stack_flags=\[([^\]]*)\]", log).group(1),
               rows=rows, n_rows=len(rows), n_kept=len(kept),
               n_stall_missing=sum(1 for r in rows if r.get("stall_missing")), results=res,
               decode_ratio=(dec["stack_mean"] / dec["base_mean"]) if "stack_mean" in dec else None,
               text_match_ok=len(re.findall(r"text_match \S+ OK", log)),
               text_match_differs=len(re.findall(r"text_match \S+ DIFFERS", log)))
    json.dump(out, open(a.out, "w"), indent=1)
    for k, v in res.items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "per_repeat"})
    print({k: out[k] for k in ("stack_flags", "n_rows", "n_kept", "n_stall_missing", "decode_ratio",
                               "text_match_ok", "text_match_differs")})


if __name__ == "__main__":
    main()
