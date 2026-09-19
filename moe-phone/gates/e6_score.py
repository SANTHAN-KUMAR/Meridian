"""
Scores research-spec E6 exactly as pre-registered in device/bmoe_e6.sh (written before any Qwen3 E6 row).

  python moe-phone/gates/e6_score.py --kl <bmoe_e6_kl_dir> [--mc <bmoe_e6_mc_dir> ...] --answers <e6_corpus/mmlu_answers.json>
      --out results/2026-09-19/e6_summary.json

KL criteria (all three, per arm, against FLOOR = Q4_0 top-8):
  KL_mean(arm) <= 2 x KL_mean(FLOOR);  KL_p99(arm) <= 2 x KL_p99(FLOOR);  flips(arm) <= 2 x flips(FLOOR)
Knowledge criterion: MMLU accuracy(arm) >= lower end of FLOOR's 95% Wilson interval (100 questions).
An arm PASSES only if all four hold. Adoption additionally needs >= 10.0 tok/s in the separate speed A/B (spec section 0).
"""
import argparse, glob, json, math, os, re


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def parse_kl(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.out")):
        tag = os.path.basename(f)[:-4]
        t = open(f).read()
        m = re.search(r"ppl-kl: mean ([0-9.]+)\s+p99 ([0-9.]+)\s+max ([0-9.]+)\s+flips (\d+)/(\d+).*?ref_tail_max ([0-9.eE+-]+)", t)
        p = re.search(r"^ppl: ([0-9.]+)", t, re.M)
        pol = re.search(r"ppl-policy: (\d+)/(\d+) routed experts dropped, (\d+)/(\d+) reranked slots substituted", t)
        r = dict(ppl=float(p.group(1)) if p else None)
        if m:
            r.update(kl_mean=float(m.group(1)), kl_p99=float(m.group(2)), kl_max=float(m.group(3)), flips=int(m.group(4)),
                     n=int(m.group(5)), ref_tail_max=float(m.group(6)))
        if pol:
            r.update(dropped=int(pol.group(1)), routed=int(pol.group(2)), substituted=int(pol.group(3)), reranked=int(pol.group(4)))
        out[tag] = r
    return out


def parse_mc(d, answers):
    out = {}
    for f in glob.glob(os.path.join(d, "mc_*.out")):
        arm = os.path.basename(f)[3:-4]
        picks = []
        for line in open(f):
            if line.startswith("ppl-choices:"):
                lp = dict(re.findall(r"(\S)=(-?[0-9.]+)", line))
                picks.append(max("ABCD", key=lambda k: float(lp[k])))
        k = sum(p == a for p, a in zip(picks, answers))
        out[arm] = dict(n=len(picks), correct=k, acc=k / len(picks) if picks else None, picks=picks)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kl", required=True); ap.add_argument("--mc", nargs="*", default=[])
    ap.add_argument("--answers", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    answers = json.load(open(a.answers))
    kl = parse_kl(a.kl)
    mc = {}
    for d in a.mc: mc.update(parse_mc(d, answers))
    fl = kl.get("FLOOR")
    if not fl or "kl_mean" not in fl:
        raise SystemExit("FLOOR arm missing: the margin is undefined, nothing can be scored")
    res = {}
    for arm, r in kl.items():
        if arm in ("REF", "FLOOR") or "kl_mean" not in r: continue
        c1 = r["kl_mean"] <= 2 * fl["kl_mean"]; c2 = r["kl_p99"] <= 2 * fl["kl_p99"]; c3 = r["flips"] <= 2 * fl["flips"]
        c4 = None
        if arm in mc and "FLOOR" in mc:
            lo, _ = wilson(mc["FLOOR"]["correct"], mc["FLOOR"]["n"]); c4 = mc[arm]["acc"] >= lo
        res[arm] = dict(kl_mean_ok=c1, kl_p99_ok=c2, flips_ok=c3, mmlu_ok=c4,
                        passes_kl=c1 and c2 and c3, passes=bool(c1 and c2 and c3 and c4))
    out = dict(source_kl=os.path.abspath(a.kl), source_mc=[os.path.abspath(x) for x in a.mc], floor=fl, arms=kl, mmlu=mc, verdict=res,
               floor_mmlu_wilson95=wilson(mc["FLOOR"]["correct"], mc["FLOOR"]["n"]) if "FLOOR" in mc else None)
    json.dump(out, open(a.out, "w"), indent=1)
    print("FLOOR", {k: fl.get(k) for k in ("kl_mean", "kl_p99", "flips", "n", "ppl", "ref_tail_max")})
    for arm, v in res.items():
        r = kl[arm]
        print(arm, {k: r.get(k) for k in ("kl_mean", "kl_p99", "flips", "ppl", "dropped", "routed")}, v, mc.get(arm, {}).get("acc"))
    for arm in ("REF", "FLOOR"):
        if arm in mc: print("MMLU", arm, mc[arm]["correct"], "/", mc[arm]["n"])


if __name__ == "__main__":
    main()
