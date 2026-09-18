#!/usr/bin/env python3
"""gguf_expert_types.py — the quant type of every expert tensor, per layer, read from the GGUF tensor table.

Why it exists: several documents assumed Qwen3-30B-A3B-Q4_0.gguf stores every ffn_down_exps as Q4_1. The
file says otherwise (Q4_1 in layers 0-5, Q4_0 in 6-47). Byte counts per call (ceiling_ledger.py) and kernel
choices (host/gpu_ffn) depend on it, so it is read from the file, never typed in.

  python3 gates/gguf_expert_types.py [gguf] -> results/2026-09-18/gguf_expert_types.json
Only the header and tensor table are parsed (gguf-py, memory-mapped); no weights are read.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGUF = sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parents[1] / "moe-work/models/Qwen3-30B-A3B-Q4_0.gguf")
sys.path.insert(0, os.environ.get("GGUF_PY", "/tmp/claude-1000/bmoe-prefetch/third_party/llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402

r = GGUFReader(GGUF)
t = {x.name: x for x in r.tensors}
n_layer = 1 + max(int(n.split(".")[1]) for n in t if n.startswith("blk."))
layers = []
for L in range(n_layer):
    row = {"layer": L}
    for p in ("gate", "up", "down"):
        x = t[f"blk.{L}.ffn_{p}_exps.weight"]
        row[p] = x.tensor_type.name
        row[f"{p}_bytes_per_expert"] = int(x.n_bytes) // int(x.shape[-1])
    layers.append(row)
out = {"gguf": os.path.basename(GGUF), "gguf_size": os.path.getsize(GGUF), "n_layer": n_layer, "layers": layers,
       "down_q4_1_layers": [r["layer"] for r in layers if r["down"] == "Q4_1"],
       "down_q4_0_layers": [r["layer"] for r in layers if r["down"] == "Q4_0"]}
p = ROOT / "results/2026-09-18/gguf_expert_types.json"
p.write_text(json.dumps(out, indent=1))
print(f"down Q4_1 in {len(out['down_q4_1_layers'])} layers {out['down_q4_1_layers']}, Q4_0 in {len(out['down_q4_0_layers'])}; wrote {p}")
