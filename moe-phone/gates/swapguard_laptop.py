"""
Swap guard (patch 0018) on the LAPTOP under forced swap pressure: OLMoE, 2000 MiB forced cache, 64 tokens, in a
memory cgroup (MemoryMax=2G, MemorySwapMax=1536M; the laptop's swap is zram) so the kernel compresses part of the
engine's anonymous cache. One run per guard mode (0 off, 1 sample, 2 full): a mechanism check, NOT a rate claim --
laptop rates do not transfer to the phone. Per mode: text md5 (decode text before "generation:"), major faults per
token and median compute/stall/wall over tokens >= 17, flash MiB read per token, the guard's own counters.
Run:
  python moe-phone/gates/swapguard_laptop.py results/2026-09-18/swapguard_laptop --out results/2026-09-18/swapguard_laptop.json
"""
import argparse, csv, hashlib, json, os, re, statistics as S
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out", required=True); a = ap.parse_args()
out = dict(source=os.path.abspath(a.root), modes={})
for g in (0, 1, 2):
    t = open(os.path.join(a.root, f"g{g}.out")).read()
    text = t[:t.index("\ngeneration:")] if "\ngeneration:" in t else t
    rows = [r for r in csv.DictReader(l for l in open(os.path.join(a.root, f"g{g}.csv")) if not l.startswith("#"))]
    st = [r for r in rows if int(r["step"]) >= 17]
    err = open(os.path.join(a.root, f"g{g}.err")).read()
    m = re.search(r"swap-guard: checks (\d+) swapped-hits (\d+)", err)
    out["modes"][str(g)] = dict(text_md5=hashlib.md5(text.encode()).hexdigest(),
        majflt_per_token=sum(float(r["majflt"]) for r in st) / len(st),
        compute_ms=S.median(float(r["compute_ms"]) for r in st), stall_ms=S.median(float(r["stall_ms"]) for r in st),
        wall_ms=S.median(float(r["wall_ms"]) for r in st),
        read_MiB_per_token=sum(float(r["read_bytes"]) for r in st) / len(st) / 1048576.0,
        guard_checks=int(m.group(1)) if m else 0, guard_swapped_hits=int(m.group(2)) if m else 0)
md = out["modes"]
out["text_identical_all_modes"] = len({v["text_md5"] for v in md.values()}) == 1
out["majflt_reduction_full_vs_off"] = 1 - md["2"]["majflt_per_token"] / md["0"]["majflt_per_token"]
out["compute_reduction_full_vs_off"] = 1 - md["2"]["compute_ms"] / md["0"]["compute_ms"]
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "source"}, indent=1))
