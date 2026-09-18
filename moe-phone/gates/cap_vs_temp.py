"""
CPU frequency cap vs phone shell temperature, from every thermal_state line logged by the device campaigns
(device/thermal_gate.sh thermal_state: `status=S cap0=F cap6=F hw0=F hw6=F shell_front_mC=T`).

The caps are scaling_max_freq of policy0 (cpu0-5) and policy6 (cpu6-7); hw* are cpuinfo_max_freq. The
kernel's thermal cooling devices for these CPUs sat at cur_state 0 while caps were applied
(2026-09-18 15:40), so the cap comes from a userspace controller (horae / oplus performance HAL), and this
table is its observed behaviour -- a description, not a model of it.

Output: per whole-degree shell temperature, the number of samples and the share at full policy6 clock.
Run:
  python moe-phone/gates/cap_vs_temp.py results --out results/2026-09-18/cap_vs_temp.json
"""
import argparse, collections, json, os, re
PAT = re.compile(r"status=(\d+) cap0=(\d+) cap6=(\d+) hw0=(\d+) hw6=(\d+) shell_front_mC=(\d+)")
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out", required=True); a = ap.parse_args()
samples = []
for dp, _, fs in os.walk(a.root):
    for f in fs:
        if not (f.endswith(".txt") or f.endswith(".log")): continue
        with open(os.path.join(dp, f), errors="replace") as fh:
            for m in PAT.finditer(fh.read()):
                s, c0, c6, h0, h6, t = map(int, m.groups())
                samples.append(dict(status=s, cap0=c0, cap6=c6, hw0=h0, hw6=h6, shell_C=t / 1000.0))
by = collections.defaultdict(list)
for s in samples: by[int(s["shell_C"])].append(s)
rows = []
for t in sorted(by):
    ss = by[t]
    rows.append(dict(shell_C=t, n=len(ss),
                     frac_cap6_full=sum(s["cap6"] >= s["hw6"] for s in ss) / len(ss),
                     frac_cap6_at_or_below_1651200=sum(s["cap6"] <= 1651200 for s in ss) / len(ss),
                     cap6_median_GHz=sorted(s["cap6"] for s in ss)[len(ss) // 2] / 1e6))
def frac(cond, pred):
    xs = [s for s in samples if cond(s)]
    return (sum(pred(s) for s in xs) / len(xs), len(xs)) if xs else (None, 0)
full_le30, n_le30 = frac(lambda s: s["shell_C"] < 31, lambda s: s["cap6"] >= s["hw6"])
capped_ge33, n_ge33 = frac(lambda s: s["shell_C"] >= 33, lambda s: s["cap6"] <= 2476800)
st0_capped, n_st0 = frac(lambda s: s["status"] == 0, lambda s: s["cap6"] < s["hw6"])
out = dict(n_samples=len(samples), rows=rows,
           below31C_frac_full_clock=full_le30, below31C_n=n_le30,
           from33C_frac_cap6_le_2_48GHz=capped_ge33, from33C_n=n_ge33,
           thermal_status0_frac_capped=st0_capped, thermal_status0_n=n_st0)
json.dump(out, open(a.out, "w"), indent=1)
print({k: v for k, v in out.items() if k != "rows"})
