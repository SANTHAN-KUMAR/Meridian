"""
Could a campaign's conclusion be an artifact of the ORDER its cells ran in?

Every phone campaign in this project rotates its cells (a Latin-square-ish rotation per repeat) so that
drift cannot line up with one arm. This checks whether that precaution was doing any work, and whether
any particular campaign is at risk: it reads each campaign's `log.txt`, recovers the EXECUTION order
from the `=== <cell>_rep<k>` lines, and reports the median decode rate by position within a repeat.

Reading the output:

  * A large `last_over_first` means that campaign drifted a lot during a repeat -- because the phone
    heated (rate falls) or because its page cache warmed (rate rises). Its per-cell differences are
    then only trustworthy because of the rotation, and a campaign of the same design WITHOUT rotation
    would be uninterpretable.
  * The SIGN differing between campaigns is the useful part: a single systematic direction would mean
    a bias every campaign shares (and that the rotation only half-cancels), whereas mixed signs mean
    drift whose direction depends on the campaign's own thermal and cache history.
  * This is a diagnostic, not a claim about any lever. It cannot say a lever is real; it can only say
    whether a reported difference is smaller than the drift that happened while it was measured.

`n_cells` is how many cells a repeat had, so `positions` can be read against it: a campaign with three
cells and three positions had each cell in each position exactly once per rotation cycle.

Run:
  python moe-phone/gates/position_effect.py --root results --out-name position_effect.json
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RE_CELL = re.compile(r"=== (\S+?)_rep(\d+)\s")
RE_GEN = re.compile(r"generation: \d+ tokens, [\d.]+ s/token \(([\d.]+) tok/s\)")


def parse_campaign(path):
    """(position within repeat, cell, decode_tok_s) in execution order."""
    rows = []
    pos = collections.Counter()
    cur = None
    cur_pos = None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = RE_CELL.match(line)
        if m:
            cur = m.group(1)
            rep = int(m.group(2))
            pos[rep] += 1
            cur_pos = pos[rep]
            continue
        g = RE_GEN.search(line)
        if g and cur is not None:
            rows.append((cur_pos, cur, float(g.group(1))))
            cur = None
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--root", default="results")
    p.add_argument("--glob", default="*/bmoe_*/log.txt")
    p.add_argument("--min-rows", type=int, default=4)
    p.add_argument("--out-name", required=True)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    out = {"root": os.path.abspath(a.root), "glob": a.glob, "campaigns": []}
    for lg in sorted(glob.glob(os.path.join(a.root, a.glob))):
        rows = parse_campaign(lg)
        if len(rows) < a.min_rows:
            continue
        by_pos = collections.defaultdict(list)
        by_cell = collections.defaultdict(list)
        for pos, cell, v in rows:
            by_pos[pos].append(v)
            by_cell[cell].append(v)
        med = {int(k): statistics.median(v) for k, v in sorted(by_pos.items())}
        if len(med) < 2:
            continue
        first, last = med[min(med)], med[max(med)]
        # A failed run (an engine that fell over mid-campaign) lands in these medians as an extreme
        # position value and is indistinguishable from drift unless it is pointed at. Rows are NOT
        # dropped -- that would be an exclusion rule tuned against the output -- they are counted, so a
        # large last/first can be checked against whether the campaign had a collapsed row at all.
        camp_med = statistics.median(v for _, _, v in rows)
        n_collapsed = sum(1 for _, _, v in rows if v < 0.25 * camp_med)
        e = {"log": os.path.relpath(lg, a.root), "campaign": lg.split(os.sep)[-2],
             "campaign_median_tok_s": camp_med, "n_rows_below_quarter_median": n_collapsed,
             "n_rows": len(rows), "n_cells": len(by_cell), "positions": med,
             "median_first_position": first, "median_last_position": last,
             "last_over_first": last / first if first else None,
             "spread_over_positions": (max(med.values()) - min(med.values())) / statistics.median(list(med.values()))}
        out["campaigns"].append(e)

    if not out["campaigns"]:
        raise ValueError(f"no campaign logs with >= {a.min_rows} rows under {a.root}/{a.glob}")

    ratios = [e["last_over_first"] for e in out["campaigns"] if e["last_over_first"]]
    out["n_campaigns"] = len(out["campaigns"])
    out["n_drifting_up"] = sum(1 for r in ratios if r > 1.0)
    out["n_drifting_down"] = sum(1 for r in ratios if r < 1.0)
    out["median_last_over_first"] = statistics.median(ratios)
    out["max_abs_drift_fraction"] = max(abs(r - 1.0) for r in ratios)
    clean = [e for e in out["campaigns"] if not e["n_rows_below_quarter_median"] and e["last_over_first"]]
    out["n_campaigns_without_collapsed_rows"] = len(clean)
    out["max_abs_drift_fraction_excluding_collapsed_campaigns"] = (
        max(abs(e["last_over_first"] - 1.0) for e in clean) if clean else None)

    for e in out["campaigns"]:
        print(f"{e['campaign']:<26} n={e['n_rows']:3d} cells={e['n_cells']} "
              + "  ".join(f"p{k}={v:.3f}" for k, v in e["positions"].items())
              + f"   last/first={e['last_over_first']:.3f}"
              + (f"  [{e['n_rows_below_quarter_median']} COLLAPSED row(s): this campaign's drift figure "
                 f"is a failure, not drift]" if e["n_rows_below_quarter_median"] else ""))
    print(f"\n{out['n_campaigns']} campaigns: {out['n_drifting_up']} drift up, "
          f"{out['n_drifting_down']} drift down, median last/first "
          f"{out['median_last_over_first']:.3f}, largest drift {out['max_abs_drift_fraction']*100:.1f}% "
          f"(largest among the {out['n_campaigns_without_collapsed_rows']} campaigns with no collapsed "
          f"row: {out['max_abs_drift_fraction_excluding_collapsed_campaigns']*100:.1f}%)")
    print("Mixed signs mean campaign-specific drift (heat vs cache warm-up), not one shared bias; the "
          "rotation is what makes per-cell differences readable either way.")

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
