"""
Scores research-spec E7 as pre-registered in device/bmoe_e7.sh (written before any E7 row).

  python moe-phone/gates/e7_score.py <bmoe_e7_dir> --out results/2026-09-19/e7_summary.json

Per row: llama-bench csv, generation rows (n_prompt 0), avg_ts = tok/s -> ms/token = 1000/avg_ts.
Fit t(L) = a + b*L from L=4 and L=8 (GPU and CPU separately); projection to Qwen3-30B-A3B's 48 layers: t48 = a + 48*b.
GPU projection uses the median of the sustained run's last 5 minutes (sus_L8_*) for the L=8 point if it is > 5% slower than the
-r 5 row (throttling), else the -r 5 row.
Decision: POSITIVE C_gpu48 <= 28.5 ms; KILL C_gpu48 > 50.5 ms; MIDDLE otherwise ("not supported").
"""
import argparse, csv, glob, json, os, re, statistics as S


def ts(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                if int(r.get("n_prompt", 0)) == 0 and int(r.get("n_gen", 0)) > 0:
                    rows.append(float(r["avg_ts"]))
            except (ValueError, KeyError):
                pass
    return rows[-1] if rows else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ms = {}
    for tag in ("gpu_L4", "gpu_L8", "cpu_L4", "cpu_L8"):
        p = os.path.join(a.root, tag + ".csv")
        t = ts(p) if os.path.exists(p) else None
        ms[tag] = 1000.0 / t if t else None
    log = open(os.path.join(a.root, "log.txt")).read()
    sus = []
    for p in sorted(glob.glob(os.path.join(a.root, "sus_L8_*.csv")), key=lambda x: int(re.search(r"_(\d+)\.csv$", x).group(1))):
        t = ts(p)
        if t: sus.append(1000.0 / t)
    # last 5 minutes ~ last half of the 10-minute run
    sus_tail = sus[len(sus) // 2:] if sus else []
    sus_med = S.median(sus_tail) if sus_tail else None
    gpu_L8 = ms["gpu_L8"]
    used_sustained = bool(sus_med and gpu_L8 and sus_med > 1.05 * gpu_L8)
    if used_sustained: gpu_L8 = sus_med

    def proj(t4, t8):
        if t4 is None or t8 is None: return None, None, None
        b = (t8 - t4) / 4.0; a0 = t4 - 4 * b
        return a0, b, a0 + 48 * b

    ga, gb, g48 = proj(ms["gpu_L4"], gpu_L8)
    ca, cb, c48 = proj(ms["cpu_L4"], ms["cpu_L8"])
    verdict = None
    if g48 is not None:
        verdict = "POSITIVE" if g48 <= 28.5 else ("KILL" if g48 > 50.5 else "MIDDLE (not supported)")
    out = dict(source=os.path.abspath(a.root), ms_per_token=ms, sustained_L8_ms=sus, sustained_tail_median_ms=sus_med,
               used_sustained_for_gpu_L8=used_sustained, gpu_fit=dict(a=ga, b=gb, t48=g48), cpu_fit=dict(a=ca, b=cb, t48=c48),
               gpu_full_model_projection_ms=g48, gpu_projection_plus_io_ms=(g48 + 14.9 + 56.6) if g48 else None, verdict=verdict)
    json.dump(out, open(a.out, "w"), indent=1)
    for k in ("ms_per_token", "sustained_tail_median_ms", "used_sustained_for_gpu_L8", "gpu_fit", "cpu_fit", "gpu_projection_plus_io_ms", "verdict"):
        print(k, out[k])


if __name__ == "__main__":
    main()
