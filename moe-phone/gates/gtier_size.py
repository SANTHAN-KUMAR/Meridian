"""
GPU-tier sizing for Qwen3-30B-A3B from the committed routing trace (results/2026-09-16/traces_...npz, 48 layers x
8 of 128 experts per token). A static stand-in for the tier's steady state: the tier holds the N most frequently
selected (layer, expert) pairs. Frequencies are counted on the FIRST half of the trace and coverage is measured on
the SECOND half (out of sample, so the choice is not scored on the data that picked it).
Per N: share of all selections the tier covers, and the mean device experts per layer per token after the
per-layer cap K (the device's actual share of work). Slot size 2,752,512 B (Q4_1-down layout, the pool's size).
Run:
  python moe-phone/gates/gtier_size.py --npz results/2026-09-16/traces_Qwen3-30B-A3B-q4_0-llamacpp.npz --cap 3 \
      --out results/2026-09-18/gtier_size.json
"""
import argparse, json
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--cap", type=int, default=3)
ap.add_argument("--out", required=True); a = ap.parse_args()
z = np.load(a.npz); E = int(z["num_experts"])
L = sorted(int(k[1:]) for k in z.files if k.startswith("L"))
T = z["L0"].shape[0]; h = T // 2
cnt = np.zeros((len(L), E), np.int64)
for li, l in enumerate(L):
    np.add.at(cnt[li], z[f"L{l}"][:h].ravel(), 1)
order = np.argsort(-cnt.ravel(), kind="stable")
SLOT = 2752512
rows = []
for mib in (500, 1000, 1500, 2000, 2500):
    n = int(mib * 1048576 // SLOT)
    owned = np.zeros(len(L) * E, bool); owned[order[:n]] = True; owned = owned.reshape(len(L), E)
    sel = cover = dev = 0
    for li, l in enumerate(L):
        ids = z[f"L{l}"][h:]
        o = owned[li][ids]              # [tokens, 8]
        per = o.sum(1)
        sel += ids.size; cover += int(o.sum()); dev += int(np.minimum(per, a.cap).sum())
    tok = (T - h) * len(L)
    rows.append(dict(tier_mib=mib, slots=n, coverage=cover / sel, owned_per_layer=cover / tok,
                     device_per_layer_capped=dev / tok))
out = dict(source=a.npz, tokens=int(T), train_tokens=int(h), cap=a.cap, rows=rows)
json.dump(out, open(a.out, "w"), indent=1)
for r in rows: print(r)
