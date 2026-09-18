"""
Engine overhead on ONE model (device/bmoe_overhead.sh), scored as pre-registered in its header. Written before
any row existed.
  per row    steady-state ms/token = median CSV wall_ms over decode tokens 33..128
  per repeat d_r = mean(stream rows of r) - mean(plain rows of r)   (rows <r> and <r>b)
  estimate   mean(d_r), SE = sd(d_r)/sqrt(n), SE/|mean|; DECISIVE iff SE/|mean| < 0.5 and sign holds in >= n-1
Also reported: weight throughput in GB/s = active bytes per token / steady ms (OLMoE active bytes from
results/2026-09-18/gguf_active_qwen_olmoe.json), llama-bench's tg128 (reference), text identity.
Run:
  python moe-phone/gates/overhead_summary.py results/2026-09-18/bmoe_overhead/bmoe_overhead_<stamp> \
      --active results/2026-09-18/gguf_active_qwen_olmoe.json --out results/2026-09-18/overhead_summary.json
"""
import argparse, glob, json, math, os, re, statistics as S


def steady_ms(csv_path, lo=33, hi=128):
    hdr, walls = None, []
    for line in open(csv_path):
        if line.startswith("#"):
            continue
        if hdr is None:
            hdr = line.strip().split(","); continue
        v = dict(zip(hdr, line.strip().split(",")))
        if "step" in v and lo <= int(v["step"]) <= hi:
            walls.append(float(v["wall_ms"]))
    return (S.median(walls), len(walls)) if walls else (None, 0)


def olmoe_active_bytes(path):
    d = json.load(open(path))
    for m in (d if isinstance(d, list) else d.get("models", [])):
        name = json.dumps(m).lower()
        if "olmoe" in name:
            for k in ("active_bytes_per_token", "active_bytes"):
                if k in m: return float(m[k])
    raise SystemExit("OLMoE active bytes not found in " + path)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--active", required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    act = olmoe_active_bytes(a.active)
    log = open(os.path.join(a.root, "log.txt")).read()
    foreign = dict(re.findall(r"=== (\w+) \S+ .*?foreign=\[([^\]]*)\]", log))
    rows = []
    for f in sorted(glob.glob(os.path.join(a.root, "*.out"))):
        tag = os.path.basename(f)[:-4]
        m = re.match(r"(plain|stream|lb)_rep(\d+)(b?)$", tag)
        if not m: continue
        r = dict(tag=tag, cell=m.group(1), rep=int(m.group(2)), foreign=foreign.get(tag))
        if r["cell"] == "lb":
            t = open(f).read()
            mm = re.search(r"tg128 \|\s+([0-9.]+)", t)
            r["tg128_tok_s"] = float(mm.group(1)) if mm else None
        else:
            ms, n = steady_ms(f[:-4] + ".csv") if os.path.exists(f[:-4] + ".csv") else (None, 0)
            r.update(steady_ms_per_token=ms, n_steady_tokens=n,
                     steady_GBps=(act / 1e9) / (ms / 1000.0) if ms else None)
            t = open(f).read() + open(f[:-4] + ".err").read()
            hit = re.search(r"([0-9.]+)% hit", t); r["hit_pct"] = float(hit.group(1)) if hit else None
        r["kept"] = r["foreign"] == "" and (r.get("steady_ms_per_token") is not None or r.get("tg128_tok_s") is not None)
        rows.append(r)
    kept = [r for r in rows if r["kept"]]
    reps = sorted({r["rep"] for r in rows})
    d = []
    for rp in reps:
        s = [r["steady_ms_per_token"] for r in kept if r["rep"] == rp and r["cell"] == "stream"]
        p = [r["steady_ms_per_token"] for r in kept if r["rep"] == rp and r["cell"] == "plain"]
        if s and p: d.append(S.mean(s) - S.mean(p))
    res = dict(n_repeats=len(d), per_repeat=d)
    if len(d) >= 2:
        mdiff, se = S.mean(d), S.stdev(d) / math.sqrt(len(d))
        ratio = se / abs(mdiff) if mdiff else float("inf")
        sign = sum((x > 0) == (mdiff > 0) for x in d)
        res.update(mean_diff_ms=mdiff, se=se, se_over_abs_mean=ratio, sign_holds_in=sign,
                   decisive=ratio < 0.5 and sign >= len(d) - 1)
    cell = {c: dict(steady_ms_median=S.median([r["steady_ms_per_token"] for r in kept if r["cell"] == c]),
                    GBps_median=S.median([r["steady_GBps"] for r in kept if r["cell"] == c]))
            for c in ("plain", "stream") if any(r["cell"] == c for r in kept)}
    lb = [r["tg128_tok_s"] for r in kept if r["cell"] == "lb" and r.get("tg128_tok_s")]
    out = dict(source=os.path.abspath(a.root), olmoe_active_bytes=act, rows=rows, cells=cell, stream_minus_plain=res,
               overhead_ratio=(cell["stream"]["steady_ms_median"] / cell["plain"]["steady_ms_median"]) if len(cell) == 2 else None,
               llama_bench_tg128_median=S.median(lb) if lb else None,
               llama_bench_GBps_median=(act / 1e9) * S.median(lb) if lb else None,
               text_match_ok=len(re.findall(r"text_match \S+ OK", log)),
               text_match_differs=len(re.findall(r"text_match \S+ DIFFERS", log)))
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "source")}, indent=1))


if __name__ == "__main__":
    main()
