"""
Scores research-spec E1 as pre-registered in device/bmoe_e1.sh (written before any E1 row), and applies the closing formula.

  python moe-phone/gates/e1_score.py <bmoe_e1_dir> [--s-gpu MS] --out results/2026-09-19/e1_summary.json

Validity per row: the replay removed flash I/O from the critical path: stall <= 1 ms. (REVISION 2026-09-19 20:30, after the
first E1 row and BEFORE any scoring: the earlier additional "MiB/token <= 1" test was wrong. The run average includes the one-time
initial load of the 384 fixed experts (~1 GB, spread over 256 tokens = ~5.7 MiB/token), not steady-state traffic. MiB/token is
reported, not gated.)
C = decode ms/token = 1000 / tok/s. Capped: median per arm over the 6 ABBA rows; repacked gain = per-repeat paired difference.
Closing formula (spec section 0, bmoe_e1.sh DECISION): 10 tok/s lossless at the capped clock iff C_R(capped) + 56.6 - S_gpu <= 100,
where 56.6 ms = stall 34.6 + mgmt 22.0 measured awake/unplugged (gtier A/B 0447 base arm), S_gpu = E4's measured saving (0 if E4 kills).
"""
import argparse, glob, json, os, re, statistics as S


def row(f):
    t = open(f).read() + open(f[:-4] + ".err").read()
    g = lambda p, c=float: (lambda m: c(m.group(1)) if m else None)(re.search(p, t))
    r = dict(tag=os.path.basename(f)[:-4], tok_s=g(r"\(([0-9.]+) tok/s\)"), mib_tok=g(r"\(([0-9.]+) MiB/token\)"),
             stall_s=g(r"stall ([0-9.]+) s/token"), compute_s=g(r"compute ([0-9.]+) \+"), hit=g(r"([0-9.]+)% hit"),
             instr=g(r"([0-9,]+)\s+instructions:u", lambda s: int(s.replace(",", ""))), cycles=g(r"([0-9,]+)\s+cycles:u", lambda s: int(s.replace(",", ""))),
             n=g(r"generation: (\d+) tokens", int))
    r["ms_tok"] = 1000.0 / r["tok_s"] if r["tok_s"] else None
    r["valid"] = r["ms_tok"] is not None and ((r["stall_s"] or 0) * 1000) <= 1.0
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--s-gpu", type=float, default=0.0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [row(f) for f in sorted(glob.glob(os.path.join(a.root, "*.out")))]
    log = open(os.path.join(a.root, "log.txt")).read()
    cap = [r for r in rows if r["tag"].startswith("cap_") and r["valid"]]
    C = {arm: S.median([r["ms_tok"] for r in cap if r["tag"].startswith(f"cap_{arm}_")]) for arm in "GR" if any(r["tag"].startswith(f"cap_{arm}_") for r in cap)}
    pairs = []
    for rep in (1, 2, 3):
        g = [r["ms_tok"] for r in cap if r["tag"].startswith(f"cap_G_r{rep}_")]; rr = [r["ms_tok"] for r in cap if r["tag"].startswith(f"cap_R_r{rep}_")]
        if g and rr: pairs.append(S.mean(g) - S.mean(rr))
    by = {r["tag"]: r for r in rows}
    CR = C.get("R")
    budget = (CR + 56.6 - a.s_gpu) if CR is not None else None
    out = dict(source=os.path.abspath(a.root), rows=rows, n_invalid=sum(1 for r in rows if not r["valid"]),
               C_capped_ms=C, repacked_gain_ms_per_repeat=pairs, repacked_gain_ms_mean=S.mean(pairs) if pairs else None,
               long_ms={k: by.get(f"long_{k}", {}).get("ms_tok") for k in "GR"},
               cold_ms={k: by.get(f"cold_{k}", {}).get("ms_tok") for k in "GR"},
               cold_labels=re.findall(r"cold wait \S+ shell_mC=\d+[^\n]*", log),
               instr_per_token={k: (by[f"cnt_{k}"]["instr"] / by[f"cnt_{k}"]["n"]) if by.get(f"cnt_{k}", {}).get("instr") and by[f"cnt_{k}"].get("n") else None for k in "GR"},
               s_gpu_ms=a.s_gpu, closing_ms_per_token=budget, closing_tok_s=(1000.0 / budget) if budget else None,
               ten_tok_s_reachable=(budget is not None and budget <= 100.0),
               e1_positive=(CR is not None and CR <= 80.0), e1_kill=(CR is not None and CR >= 95.0))
    json.dump(out, open(a.out, "w"), indent=1)
    for k in ("C_capped_ms", "repacked_gain_ms_per_repeat", "repacked_gain_ms_mean", "long_ms", "cold_ms", "cold_labels", "instr_per_token", "n_invalid",
              "s_gpu_ms", "closing_ms_per_token", "closing_tok_s", "ten_tok_s_reachable", "e1_positive", "e1_kill"):
        print(k, out[k])


if __name__ == "__main__":
    main()
