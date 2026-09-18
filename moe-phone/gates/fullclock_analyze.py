"""
What would cooling buy? Per-token engine terms at FULL clock vs capped, from device/bmoe_fullclock.sh.
Written before any run of it existed; the rule below is the one pre-registered in that script's header.

Timing: each run's .caps file starts with `launch_uptime T0`, then one line per second:
`uptime cap0 cap6 shell_mC`. The CSV gives every decode token's wall_ms, and its summary line gives load_s
and prefill_s, so a token's end time is T0 + load_s + prefill_s + cumsum(wall_ms). That anchor is approximate
(process start-up is not counted), so a token is classified only if EVERY cap sample within +-2 s of its time
agrees:
    full    cap0 == hw0 and cap6 == hw6 in all samples of the window
    capped  cap6 <  hw6 in all samples of the window
    (anything else is a transition and is excluded)
Tokens 1-15 are excluded (cache warm-up). Validity: >= 20 full-clock tokens pooled over runs; otherwise the
full-clock number is "not measurable this way" and none is printed.
Run:
  python moe-phone/gates/fullclock_analyze.py results/2026-09-18/bmoe_fullclock/bmoe_fullclock_<stamp> \
      --out results/2026-09-18/fullclock.json
"""
import argparse, glob, json, os, re, statistics as S


def parse_caps(path):
    t0, samples = None, []
    for line in open(path):
        p = line.split()
        if p and p[0] == "launch_uptime":
            t0 = float(p[1])
        elif len(p) >= 3:
            samples.append((float(p[0]), int(p[1]), int(p[2]), int(p[3]) if len(p) > 3 and p[3].isdigit() else None))
    return t0, samples


def parse_csv(path):
    rows, summ = [], {}
    hdr = None
    for line in open(path):
        if line.startswith("# summary"):
            summ = dict(kv.split("=", 1) for kv in line[len("# summary"):].split() if "=" in kv)
            continue
        if line.startswith("#"):
            continue
        if hdr is None:
            hdr = line.strip().split(","); continue
        v = line.strip().split(",")
        if len(v) == len(hdr):
            rows.append(dict(zip(hdr, v)))
    return rows, summ


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    log = open(os.path.join(a.root, "log.txt")).read()
    hw0 = int(re.findall(r"hw0=(\d+)", log)[-1])
    hw6 = int(re.findall(r"hw6=(\d+)", log)[-1])
    toks = {"full": [], "capped": []}
    per_run = []
    for caps in sorted(glob.glob(os.path.join(a.root, "run*.caps"))):
        run = os.path.basename(caps)[:-5]
        csvp = os.path.join(a.root, run + ".csv")
        if not os.path.exists(csvp):
            per_run.append(dict(run=run, error="no csv")); continue
        t0, samples = parse_caps(caps)
        rows, summ = parse_csv(csvp)
        if t0 is None or not samples or not rows or "load_s" not in summ:
            per_run.append(dict(run=run, error="missing timing")); continue
        t = t0 + float(summ["load_s"]) + float(summ.get("prefill_s", 0))
        counts = {"full": 0, "capped": 0, "transition": 0, "warmup": 0}
        for r in rows:
            t += float(r["wall_ms"]) / 1000.0
            step = int(r["step"])
            if step < 16:
                counts["warmup"] += 1; continue
            win = [s for s in samples if abs(s[0] - t) <= 2.0]
            if not win:
                counts["transition"] += 1; continue
            if all(s[1] == hw0 and s[2] == hw6 for s in win):
                cls = "full"
            elif all(s[2] < hw6 for s in win):
                cls = "capped"
            else:
                counts["transition"] += 1; continue
            counts[cls] += 1
            toks[cls].append(dict(run=run, step=step, wall_ms=float(r["wall_ms"]), compute_ms=float(r["compute_ms"]),
                                  stall_ms=float(r["stall_ms"]), mgmt_ms=float(r["mgmt_ms"]),
                                  cap0=S.median(s[1] for s in win), cap6=S.median(s[2] for s in win)))
        per_run.append(dict(run=run, **counts, shell_start_mC=samples[0][3] if samples else None,
                            shell_end_mC=samples[-1][3] if samples else None))
    def summ_cls(ts):
        if not ts: return None
        return {k + "_median": S.median(x[k] for x in ts) for k in ("wall_ms", "compute_ms", "stall_ms", "mgmt_ms", "cap0", "cap6")} | {"n_tokens": len(ts)}
    out = dict(source=os.path.abspath(a.root), hw0=hw0, hw6=hw6, per_run=per_run,
               full=summ_cls(toks["full"]), capped=summ_cls(toks["capped"]),
               valid=len(toks["full"]) >= 20)
    if out["valid"]:
        out["full_clock_decode_tok_s"] = 1000.0 / out["full"]["wall_ms_median"]
        if out["capped"]:
            out["capped_decode_tok_s"] = 1000.0 / out["capped"]["wall_ms_median"]
            out["compute_ratio_capped_over_full"] = out["capped"]["compute_ms_median"] / out["full"]["compute_ms_median"]
    else:
        out["verdict"] = "not measurable this way: fewer than 20 full-clock decode tokens"
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "source"}, indent=1))


if __name__ == "__main__":
    main()
