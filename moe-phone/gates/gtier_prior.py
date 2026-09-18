"""
Popularity prior for the GPU tier's warm start (patch 0019, --gpu-tier-prior): every (layer, expert) pair of
Qwen3-30B-A3B ordered by how often the router selected it in the committed trace (results/2026-09-16/traces_...npz,
all tokens). The engine fills the tier at load with the first N pairs that fit. The trace is llama.cpp's routing on
its own prompts, not the benchmark prompt, so this is a prior, not a fit to the measured run.
Output: one "layer expert count" line per pair, most frequent first.
Run:
  python moe-phone/gates/gtier_prior.py --npz results/2026-09-16/traces_Qwen3-30B-A3B-q4_0-llamacpp.npz \
      --out results/2026-09-18/qwen3_gtier_prior.txt
"""
import argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
z = np.load(a.npz); E = int(z["num_experts"])
L = sorted(int(k[1:]) for k in z.files if k.startswith("L"))
cnt = np.zeros((len(L), E), np.int64)
for li, l in enumerate(L):
    np.add.at(cnt[li], z[f"L{l}"].ravel(), 1)
order = np.argsort(-cnt.ravel(), kind="stable")
with open(a.out, "w") as f:
    for idx in order:
        li, e = divmod(int(idx), E)
        f.write(f"{L[li]} {e} {int(cnt[li, e])}\n")
print("pairs", len(order), "top", [divmod(int(i), E) for i in order[:3]])
