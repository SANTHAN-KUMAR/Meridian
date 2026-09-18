"""
Patch 0013 confidence-gate sweep on the phone (device/bmoe_specgate.sh), scored by the rule pre-registered in
tools/patches/0013-NOTE.md and clarified in the script header before any row:
  PRIMARY (relative, against THIS sweep's own medians): a gated arm is preferred over `sel` iff
      read(arm)  <= read(sel)  - 0.5 * (read(sel)  - read(base))     [removes >= half of sel's excess bytes]
      stall(arm) <= stall(base) - 0.8 * (stall(base) - stall(sel))   [keeps >= 80% of sel's stall cut]
  SECONDARY (absolute, from the 0012 run at spec-max 2): read <= 125.3 MiB/token and stall <= 37.8 ms.
Among preferred arms, the one with the lowest read MiB/token is chosen; if none, the stack uses no gate.
The 0012 shutdown line prints twice per run; the first occurrence is used (re.search).
Run:
  python moe-phone/gates/specgate_summary.py results/2026-09-18/bmoe_specgate/bmoe_specgate_<stamp> \
      --out results/2026-09-18/specgate_summary.json
"""
import argparse, glob, json, os, re, statistics as S

GATE_FLAGS = {"r1": "--predict-spec-rank 1", "r2": "--predict-spec-rank 2",
              "m05": "--predict-spec-margin 0.5", "m10": "--predict-spec-margin 1.0"}


def num(pat, text, cast=float):
    m = re.search(pat, text)
    return cast(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.root, "*.out"))):
        tag = os.path.basename(f)[:-4]
        cell, rep = re.match(r"(\w+?)_rep(\d+)", tag).groups()
        t = open(f).read() + open(f[:-4] + ".err").read()
        r = dict(tag=tag, cell=cell, rep=int(rep),
                 decode_tok_s=num(r"\(([0-9.]+) tok/s\)", t),
                 read_MiB_per_token=num(r"\(([0-9.]+) MiB/token\)", t),
                 stall_ms=num(r"stall ([0-9.]+) s/token", t),
                 compute_ms=num(r"compute ([0-9.]+) \+", t),
                 cache_mgmt_ms=num(r"cache mgmt ([0-9.]+) \+", t),
                 hit_pct=num(r"([0-9.]+)% hit", t),
                 spec_useful=num(r"(\d+)/\d+ experts useful", t, int),
                 spec_integrated=num(r"\d+/(\d+) experts useful", t, int),
                 issued=num(r"issued (\d+)", t, int))
        for k in ("stall_ms", "compute_ms", "cache_mgmt_ms"):
            if r[k] is not None: r[k] *= 1000.0
        if r["decode_tok_s"] is None or r["read_MiB_per_token"] is None:
            r["failed"] = True
        rows.append(r)
    ok = [r for r in rows if not r.get("failed")]
    cells = {}
    for c in ("base", "sel", "r1", "r2", "m05", "m10"):
        rr = [r for r in ok if r["cell"] == c]
        if not rr: continue
        def med(k):
            v = [r[k] for r in rr if r[k] is not None]
            return S.median(v) if v else None
        cells[c] = dict(n=len(rr), **{k + "_median": med(k) for k in (
            "decode_tok_s", "read_MiB_per_token", "stall_ms", "compute_ms", "cache_mgmt_ms", "hit_pct",
            "spec_useful", "spec_integrated", "issued")})
        if cells[c]["spec_useful_median"] is not None and cells[c]["issued_median"]:
            cells[c]["precision_useful_over_issued"] = cells[c]["spec_useful_median"] / cells[c]["issued_median"]
    b, s = cells["base"], cells["sel"]
    read_thr = s["read_MiB_per_token_median"] - 0.5 * (s["read_MiB_per_token_median"] - b["read_MiB_per_token_median"])
    stall_thr = b["stall_ms_median"] - 0.8 * (b["stall_ms_median"] - s["stall_ms_median"])
    decisions = {}
    for g in GATE_FLAGS:
        if g not in cells: continue
        c = cells[g]
        decisions[g] = dict(
            primary_read_ok=c["read_MiB_per_token_median"] <= read_thr,
            primary_stall_ok=c["stall_ms_median"] <= stall_thr,
            secondary_read_ok=c["read_MiB_per_token_median"] <= 125.3,
            secondary_stall_ok=c["stall_ms_median"] <= 37.8)
        decisions[g]["preferred"] = decisions[g]["primary_read_ok"] and decisions[g]["primary_stall_ok"]
    pref = [g for g, d in decisions.items() if d["preferred"]]
    chosen = min(pref, key=lambda g: cells[g]["read_MiB_per_token_median"]) if pref else None
    log = open(os.path.join(a.root, "log.txt")).read()
    out = dict(source=os.path.abspath(a.root), rows=rows, n_failed=len(rows) - len(ok), cells=cells,
               thresholds=dict(read_MiB_per_token=read_thr, stall_ms=stall_thr,
                               secondary_read=125.3, secondary_stall=37.8),
               sel_excess_read_over_base=s["read_MiB_per_token_median"] - b["read_MiB_per_token_median"],
               sel_stall_cut_vs_base=b["stall_ms_median"] - s["stall_ms_median"],
               decisions=decisions, chosen_gate=chosen, chosen_flags=GATE_FLAGS.get(chosen, ""),
               text_match_ok=len(re.findall(r"text_match \S+ OK", log)),
               text_match_differs=len(re.findall(r"text_match \S+ DIFFERS", log)))
    json.dump(out, open(a.out, "w"), indent=1)
    for c, d in cells.items():
        print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()})
    print("thresholds", out["thresholds"]); print("decisions", decisions)
    print({k: out[k] for k in ("chosen_gate", "chosen_flags", "text_match_ok", "text_match_differs", "n_failed",
                               "sel_excess_read_over_base", "sel_stall_cut_vs_base")})


if __name__ == "__main__":
    main()
