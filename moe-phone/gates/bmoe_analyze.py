"""
Parse bmoe-cli (BigMoeOnEdge) run summaries into one artifact.

bmoe-cli prints its per-run summary on stdout; the lines used here, as the CLI writes them:
  generation: N tokens, X s/token (Y tok/s)
  prefill: N tokens, X s (Y tok/s) | model load X s | TTFT X s
  moe-stream: read X MiB (Y MiB/token), decode X s/token (compute A + cache mgmt B + flash I/O C s/token, D MiB/s)
  moe-cache: X% hit, resident X MiB, budget X MiB (...)
  moe-cache: N evictions, M re-reads (R/token) ...
  <src>: A/B drafts accepted (X%), T tokens per verify decode ...     (speculative runs only)
  <src>: drafting costs X s/token on top of decode -> Y tok/s effective  (speculative runs only)
Cells are named <cell>_rep<k> by the campaign scripts; repeats are grouped per cell, never pooled
across cells. A field the CLI did not print is recorded as null, not zero.

Run:
  python moe-phone/gates/bmoe_analyze.py results/<date>/bmoe_repro --out-name bmoe_repro.json
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PATTERNS = {
    "gen": re.compile(r"generation: (\d+) tokens, ([\d.]+) s/token \(([\d.]+) tok/s\)"),
    "prefill": re.compile(r"prefill: (\d+) tokens, ([\d.]+) s \(([\d.]+) tok/s\) \| model load ([\d.]+) s \| TTFT ([\d.]+) s"),
    "stream": re.compile(r"moe-stream: read ([\d.]+) MiB \(([\d.]+) MiB/token\), decode ([\d.]+) s/token "
                         r"\(compute ([\d.]+) \+ cache mgmt ([\d.]+) \+ flash I/O ([\d.]+) s/token, ([\d.]+) MiB/s\)"),
    "hit": re.compile(r"moe-cache: ([\d.]+)% hit, resident ([\d.]+) MiB, budget ([\d.]+) MiB"),
    "evict": re.compile(r"moe-cache: (\d+) evictions, (\d+) re-reads \(([\d.]+)/token\)"),
    "drafts": re.compile(r"(\w+): (\d+)/(\d+) drafts accepted \(([\d.]+)%\), ([\d.]+) tokens per verify decode"),
    "draftcost": re.compile(r"(\w+): drafting costs ([\d.]+) s/token on top of decode .{1,4} ([\d.]+) tok/s effective"),
}


def parse(text):
    r = {}
    m = PATTERNS["gen"].search(text)
    if m:
        r.update(n_generated=int(m[1]), s_per_token=float(m[2]), decode_tok_s=float(m[3]))
    m = PATTERNS["prefill"].search(text)
    if m:
        r.update(n_prompt=int(m[1]), prefill_s=float(m[2]), prefill_tok_s=float(m[3]),
                 load_s=float(m[4]), ttft_s=float(m[5]))
    m = PATTERNS["stream"].search(text)
    if m:
        r.update(read_MiB=float(m[1]), read_MiB_per_token=float(m[2]), compute_s_per_token=float(m[4]),
                 cache_mgmt_s_per_token=float(m[5]), flash_io_s_per_token_summed=float(m[6]),
                 io_lane_MiBps=float(m[7]))
    m = PATTERNS["hit"].search(text)
    if m:
        r.update(cache_hit_pct=float(m[1]), resident_MiB=float(m[2]), budget_MiB=float(m[3]))
    m = PATTERNS["evict"].search(text)
    if m:
        r.update(evictions=int(m[1]), rereads=int(m[2]), rereads_per_token=float(m[3]))
    m = PATTERNS["drafts"].search(text)
    if m:
        r.update(draft_source=m[1], drafts_accepted=int(m[2]), drafts_total=int(m[3]),
                 draft_accept_pct=float(m[4]), tokens_per_verify=float(m[5]))
    m = PATTERNS["draftcost"].search(text)
    if m:
        r.update(draft_cost_s_per_token=float(m[2]), effective_tok_s_with_drafting=float(m[3]))
    return r


# device/bmoe_zram.sh writes memory and swap counters around each run, so a row claiming to have swapped
# can be checked against /proc/vmstat rather than believed: "=== <tag> <time> MemAvailable=.. SwapFree=..
# pswpin=.. pswpout=.. BEFORE .." then "exit=N AFTER .. MemAvailable=.. SwapFree=.. pswpin=.. pswpout=..".
RE_LOG_BEFORE = re.compile(r"^=== (\S+_rep\d+)\s")
RE_KV = re.compile(r"(MemAvailable|SwapFree|SwapTotal|pswpin|pswpout)=(\d+)")


def parse_swap_log(path):
    """tag -> {before: {...}, after: {...}, delta_pswpin, delta_pswpout, exit}"""
    out = {}
    tag = None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = RE_LOG_BEFORE.match(line)
        if m:
            tag = m.group(1)
            out[tag] = {"before": {k: int(v) for k, v in RE_KV.findall(line)}}
            continue
        if tag and line.startswith("exit="):
            e = out[tag]
            e["exit"] = int(line.split("=", 1)[1].split()[0])
            e["after"] = {k: int(v) for k, v in RE_KV.findall(line)}
            for k in ("pswpin", "pswpout"):
                if k in e["before"] and k in e["after"]:
                    e["delta_" + k] = e["after"][k] - e["before"][k]
            if "MemAvailable" in e["before"]:
                e["memavail_MB_before"] = e["before"]["MemAvailable"] / 1024.0
            if "SwapFree" in e["before"] and "SwapFree" in e["after"]:
                e["swap_used_MB_during"] = (e["before"]["SwapFree"] - e["after"]["SwapFree"]) / 1024.0
            tag = None
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("root")
    p.add_argument("--out-name", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--swap-log", default=None,
                   help="a campaign log.txt carrying MemAvailable/SwapFree/pswp* around each run "
                        "(device/bmoe_zram.sh); attaches the measured swap activity per row")
    a = p.parse_args()
    swap = parse_swap_log(a.swap_log) if a.swap_log else {}
    runs = []
    for out in sorted(glob.glob(os.path.join(a.root, "*.out"))):
        tag = os.path.basename(out)[:-4]
        text = open(out, encoding="utf-8", errors="replace").read()
        err = out[:-4] + ".err"
        if os.path.exists(err):
            text += "\n" + open(err, encoding="utf-8", errors="replace").read()
        cell, _, rep = tag.rpartition("_rep")
        r = dict(parse(text), tag=tag, cell=cell or tag, rep=int(rep) if rep.isdigit() else None)
        if tag in swap:
            r.update({k: v for k, v in swap[tag].items() if k not in ("before", "after")})
            r["swap_counters"] = {"before": swap[tag].get("before"), "after": swap[tag].get("after")}
        runs.append(r)
    cells = {}
    for r in runs:
        cells.setdefault(r["cell"], []).append(r)
    summary = []
    keys = ("decode_tok_s", "effective_tok_s_with_drafting", "cache_hit_pct", "read_MiB_per_token",
            "compute_s_per_token", "cache_mgmt_s_per_token", "flash_io_s_per_token_summed",
            "rereads_per_token", "budget_MiB",
            "draft_accept_pct", "tokens_per_verify", "prefill_tok_s",
            "delta_pswpin", "delta_pswpout", "swap_used_MB_during", "memavail_MB_before")
    for cell, rs in sorted(cells.items()):
        s = {"cell": cell, "n": len(rs), "failed": sum(1 for r in rs if "decode_tok_s" not in r)}
        for k in keys:
            s[k + "_all"] = [r.get(k) for r in rs]
            s[k + "_median"] = med([r.get(k) for r in rs])
        summary.append(s)
        print(f"{cell:<22} n={s['n']} decode {s['decode_tok_s_all']} tok/s "
              f"(eff w/ drafting {s['effective_tok_s_with_drafting_all']}) hit {s['cache_hit_pct_all']}% "
              f"read {s['read_MiB_per_token_median']} MiB/tok compute {s['compute_s_per_token_median']} s "
              f"accept {s['draft_accept_pct_all']}")
    out = {"source": os.path.abspath(a.root), "runs": runs, "cells": summary}
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
