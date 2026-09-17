"""
Decode rates measured INSIDE an app process (SELinux untrusted_app), which is the only domain on this phone
where the Hexagon DSP session opens (results/2026-09-17/npu_inprocess). Two producers write into the same
results tree and both are parsed here:

  upstream    llama.cpp's llama-completion / llama-bench, run in-process through the JNI shim
              (moe-work/overnight_npu.sh -> results/.../app_engine). Rate comes from the
              `common_perf_print: eval time = ... ( X ms per token, Y tokens per second)` line.
  bmoe        our own engine (BigMoeOnEdge + patch 0009) run the same way
              (moe-work/bmoe_attn_ab.sh -> results/.../bmoe_attn_ab). Rate comes from the engine's own
              `generation: N tokens, X s/token (Y tok/s)` line, i.e. the same number every other bmoe
              row in this project is quoted from -- decode only, prefill and model load excluded.

Both numbers are decode (token-generation) throughput, and neither includes load or prefill, so arms are
comparable within a producer. They are NOT comparable across producers without saying so: upstream's
streaming path has no read/compute overlap.

A row is a failure unless a rate was printed; failures are counted and listed, never dropped (CLAUDE.md
7.2). Arms are grouped by their file name (`<model>_<device>_<config>_rep<k>` / `attn_<arm>_rep<k>`) and
reported with median, min, max and n, plus the ratio of each arm's median to the named reference arm.

Run:
  python moe-phone/gates/app_engine_analyze.py results/2026-09-18/app_engine --out-name app_engine.json
  python moe-phone/gates/app_engine_analyze.py results/2026-09-18/bmoe_attn_ab --producer bmoe \
      --reference attn_cpu --out-name attn_ab.json
"""
import argparse
import collections
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# upstream: llama.cpp's own perf print. Group 1 = ms/token, group 2 = tok/s.
RE_UPSTREAM = re.compile(r"eval time =\s*[0-9.]+ ms /\s*(\d+) runs?\s*\(\s*([0-9.]+) ms per token,\s*([0-9.]+) tokens per second\)")
RE_UP_PP = re.compile(r"prompt eval time =\s*[0-9.]+ ms /\s*(\d+) tokens?\s*\(\s*[0-9.]+ ms per token,\s*([0-9.]+) tokens per second\)")
RE_UP_LOAD = re.compile(r"load time =\s*([0-9.]+) ms")
# llama-bench prints a markdown table instead of a perf block: "| ... | tg64 | 40.70 ± 0.35 |"
RE_BENCH_TG = re.compile(r"\|\s*tg(\d+)\s*\|\s*([0-9.]+)\s*±\s*([0-9.]+)\s*\|")
RE_BENCH_PP = re.compile(r"\|\s*pp(\d+)\s*\|\s*([0-9.]+)\s*±\s*([0-9.]+)\s*\|")
# bmoe: the engine's own decode report (core/src/engine/session.cpp print_predict_report)
RE_BMOE = re.compile(r"generation: (\d+) tokens, ([0-9.]+) s/token \(([0-9.]+) tok/s\)")
RE_BMOE_PP = re.compile(r"prefill: (\d+) tokens, ([0-9.]+) s \(([0-9.]+) tok/s\) \| model load ([0-9.]+) s")
RE_BMOE_STREAM = re.compile(r"read ([0-9.]+) MiB \(([0-9.]+) MiB/token\), decode ([0-9.]+) s/token "
                            r"\(compute ([0-9.]+) \+ cache mgmt ([0-9.]+) \+ flash I/O ([0-9.]+) s/token")
RE_BMOE_ATTN = re.compile(r"bmoe: attention weights -> (\S+)")
RE_BMOE_DEVRES = re.compile(r"bmoe: (\d+) weight tensors live in a non-host buffer")
RE_HIT = re.compile(r"cache: ([0-9.]+)% hit")


def parse_row(path, producer):
    txt = open(path, encoding="utf-8", errors="replace").read()
    row = {"file": os.path.basename(path)}
    if producer == "upstream":
        m = RE_UPSTREAM.search(txt)
        if not m:
            # llama-bench (the resident-model rows) rather than llama-completion
            b = RE_BENCH_TG.search(txt)
            if not b:
                row["failed"] = True
                return row
            row.update(failed=False, tokens=int(b.group(1)), decode_tok_s=float(b.group(2)),
                       decode_tok_s_sd=float(b.group(3)), ms_per_token=1000.0 / float(b.group(2)),
                       source_tool="llama-bench")
            pp = RE_BENCH_PP.search(txt)
            if pp:
                row["prefill_tok_s"] = float(pp.group(2))
            return row
        row["source_tool"] = "llama-completion"
        row.update(failed=False, tokens=int(m.group(1)), ms_per_token=float(m.group(2)),
                   decode_tok_s=float(m.group(3)))
        pp = RE_UP_PP.search(txt)
        if pp:
            row["prefill_tok_s"] = float(pp.group(2))
        ld = RE_UP_LOAD.search(txt)
        if ld:
            row["load_s"] = float(ld.group(1)) / 1000.0
    else:
        m = RE_BMOE.search(txt)
        if not m:
            row["failed"] = True
            return row
        row.update(failed=False, tokens=int(m.group(1)), s_per_token=float(m.group(2)),
                   decode_tok_s=float(m.group(3)))
        pp = RE_BMOE_PP.search(txt)
        if pp:
            row["prefill_tok_s"] = float(pp.group(3))
            row["load_s"] = float(pp.group(4))
        st = RE_BMOE_STREAM.search(txt)
        if st:
            row.update(read_MiB_per_token=float(st.group(2)), compute_s_per_token=float(st.group(4)),
                       mgmt_s_per_token=float(st.group(5)), stall_s_per_token=float(st.group(6)))
        at = RE_BMOE_ATTN.search(txt)
        row["attn_device"] = at.group(1) if at else "CPU(unset)"
        dr = RE_BMOE_DEVRES.search(txt)
        row["device_resident_tensors"] = int(dr.group(1)) if dr else 0
        hit = RE_HIT.search(txt)
        if hit:
            row["cache_hit_pct"] = float(hit.group(1))
    return row


def arm_of(name):
    stem = name[:-4] if name.endswith(".txt") else name
    return re.sub(r"_rep\d+$", "", stem)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("dir", help="a results directory of <arm>_rep<k>.txt files")
    p.add_argument("--producer", choices=["upstream", "bmoe"], default="upstream")
    p.add_argument("--reference", default=None, help="arm whose median other arms are divided by")
    p.add_argument("--out-name", required=True)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    rows = []
    for fn in sorted(os.listdir(a.dir)):
        if not fn.endswith(".txt") or fn.endswith(".runner.txt") or ".state" in fn:
            continue
        rows.append(parse_row(os.path.join(a.dir, fn), a.producer))
    if not rows:
        raise ValueError(f"no result files in {a.dir}")

    by = collections.defaultdict(list)
    for r in rows:
        by[arm_of(r["file"])].append(r)

    arms = []
    for arm, rs in sorted(by.items()):
        ok = [r for r in rs if not r.get("failed")]
        e = {"arm": arm, "n": len(rs), "n_failed": len(rs) - len(ok),
             "failure_rate": (len(rs) - len(ok)) / len(rs)}
        if ok:
            rates = sorted(r["decode_tok_s"] for r in ok)
            e.update(decode_tok_s_median=statistics.median(rates), decode_tok_s_min=rates[0],
                     decode_tok_s_max=rates[-1], ms_per_token_median=1000.0 / statistics.median(rates))
            for k in ("prefill_tok_s", "load_s", "read_MiB_per_token", "compute_s_per_token",
                      "mgmt_s_per_token", "stall_s_per_token", "cache_hit_pct"):
                vals = [r[k] for r in ok if k in r]
                if vals:
                    e[k + "_median"] = statistics.median(vals)
            devs = sorted({r.get("attn_device") for r in ok if r.get("attn_device")})
            if devs:
                e["attn_device"] = devs[0] if len(devs) == 1 else devs
            drs = sorted({r.get("device_resident_tensors") for r in ok if "device_resident_tensors" in r})
            if drs:
                e["device_resident_tensors"] = drs[0] if len(drs) == 1 else drs
        arms.append(e)

    out = {"source": os.path.abspath(a.dir), "producer": a.producer, "rows": rows, "arms": arms}
    if a.reference:
        ref = next((e for e in arms if e["arm"] == a.reference), None)
        if ref is None or "decode_tok_s_median" not in ref:
            raise ValueError(f"reference arm {a.reference!r} has no rate: "
                             f"{[e['arm'] for e in arms]}")
        out["reference"] = a.reference
        for e in arms:
            if "decode_tok_s_median" in e:
                e["vs_reference"] = e["decode_tok_s_median"] / ref["decode_tok_s_median"]

    for e in arms:
        r = e.get("decode_tok_s_median")
        print(f"{e['arm']:<22} n={e['n']} fail={e['n_failed']} "
              + (f"median {r:6.3f} tok/s ({e['ms_per_token_median']:7.1f} ms/token)" if r else "NO RATE")
              + (f"  vs {a.reference}: {e['vs_reference']:.3f}x" if e.get("vs_reference") else ""))

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
