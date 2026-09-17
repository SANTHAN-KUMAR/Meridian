"""
What decode rate each compute device on this phone can support, from the only quantity that bounds it:
the rate at which it consumes WEIGHT BYTES.

A decoded token of a MoE reads a fixed set of weights (gates/gguf_active.py reads the set from the GGUF's
own tensor table), so for any engine and any device

    decode_tok_s  =  effective_weight_GB_s / active_bytes_per_token

This script inverts that. It takes a measured decode rate for a model that is FULLY RESIDENT (no flash in
the loop, so the measurement is of the device, not of the streaming engine) and reports

    effective_GB_s          = active_bytes_per_token * decode_tok_s
    implied_tok_s(target)   = effective_GB_s / active_bytes_per_token(target)
    n/a                     if the row failed

`implied_tok_s` is an UPPER BOUND for that device on the target model, and a loose one: it assumes the
target streams weights at the same effective rate the resident model achieved, which ignores flash reads,
cache misses, a larger KV cache and the fact that the target's experts do not fit in any device buffer.
A device whose bound is already below the goal cannot reach the goal by any amount of engine work, which
is the only inference this file is for (CLAUDE.md 4.1: report the ceiling the design permits before
tuning an estimator against it).

Inputs are artifacts, never numbers: an app_engine_analyze.py JSON for the rates and a gguf_active.py JSON
for the byte counts. The resident model and the target model are named on the command line and must both
appear in the active-bytes artifact.

Run:
  python moe-phone/gates/device_bandwidth.py \
      --rates results/2026-09-18/app_engine.json \
      --active results/2026-09-18/gguf_active_qwen_olmoe.json \
      --resident olmoe-1b-7b-0924-q4_0.gguf --target Qwen3-30B-A3B-Q4_0.gguf \
      --arm-prefix olmoe_ --out-name device_bandwidth.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def active_of(active, fname):
    for row in active:
        if row["file"] == fname or os.path.basename(row["file"]) == fname:
            return float(row["active_bytes_per_token"])
    raise ValueError(f"{fname!r} not in the active-bytes artifact: {[r['file'] for r in active]}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--rates", required=True, help="app_engine_analyze.py JSON")
    p.add_argument("--active", required=True, help="gguf_active.py JSON")
    p.add_argument("--resident", required=True, help="file name of the model the rates were measured on")
    p.add_argument("--target", required=True, help="file name of the model the bound is projected onto")
    p.add_argument("--arm-prefix", default="", help="only arms whose name starts with this")
    p.add_argument("--goal-tok-s", type=float, required=True,
                   help="the project's decode-rate goal for the target model, for the verdict column")
    p.add_argument("--out-name", required=True)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    rates = json.load(open(a.rates, encoding="utf-8"))
    active = json.load(open(a.active, encoding="utf-8"))
    b_res = active_of(active, a.resident)
    b_tgt = active_of(active, a.target)

    out = {"rates_source": os.path.abspath(a.rates), "active_source": os.path.abspath(a.active),
           "resident_model": a.resident, "target_model": a.target,
           "resident_active_bytes_per_token": b_res, "target_active_bytes_per_token": b_tgt,
           "goal_tok_s": a.goal_tok_s, "devices": []}

    for arm in rates["arms"]:
        if not arm["arm"].startswith(a.arm_prefix):
            continue
        e = {"arm": arm["arm"], "n": arm["n"], "n_failed": arm["n_failed"]}
        r = arm.get("decode_tok_s_median")
        if r is None:
            e["status"] = "no rate: every repeat failed"
        else:
            gbs = b_res * r / 1e9
            e.update(resident_decode_tok_s_median=r, effective_GB_s=gbs,
                     implied_target_tok_s=gbs * 1e9 / b_tgt,
                     meets_goal=bool(gbs * 1e9 / b_tgt >= a.goal_tok_s))
        out["devices"].append(e)

    if not out["devices"]:
        raise ValueError(f"no arms start with {a.arm_prefix!r}: {[x['arm'] for x in rates['arms']]}")

    print(f"{a.resident}: {b_res/1e6:.1f} MB/token   {a.target}: {b_tgt/1e6:.1f} MB/token   "
          f"goal {a.goal_tok_s} tok/s needs {b_tgt * a.goal_tok_s / 1e9:.1f} GB/s")
    for e in out["devices"]:
        if "effective_GB_s" not in e:
            print(f"  {e['arm']:<18} {e['status']}")
            continue
        print(f"  {e['arm']:<18} {e['resident_decode_tok_s_median']:7.2f} tok/s resident -> "
              f"{e['effective_GB_s']:6.2f} GB/s -> at most {e['implied_target_tok_s']:6.2f} tok/s on the "
              f"target ({'goal reachable' if e['meets_goal'] else 'BELOW GOAL: no engine work can close it'})")

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
