"""
G2 (offline half) — expert-cache hit rates from routing traces.

Input: the traces_<model>.npz written by traces_sparsity.py — one int16 array
per MoE layer, shape [tokens, top_k], of the experts the model's own router
selected. Replays the request stream (token-major, then layer, then expert)
through an expert cache shared by all layers and reports, per cache size:

  - LRU hit rate            (what a simple engine gets)
  - Belady-optimal hit rate (the offline optimum: an upper bound for ANY
                             eviction policy, used as the oracle, CLAUDE.md §4.3)
  - no-locality floor       (= cache fraction; what uniform independent routing
                             gives any policy — the value G-ROOF assumed)
  - misses per token        (x expert bytes = flash bytes per token, feeding G-ROOF)

If Belady barely beats the floor, routing has no exploitable locality at that
cache size — finding F4 in POSITION.md, not a failure to hide.

Run:
  python moe-phone/gates/cache_sim.py traces_Qwen3-30B-A3B.npz --fractions 0.05,0.1,0.2,0.3,0.5
"""
import argparse
import heapq
import json
import os
import sys
from collections import OrderedDict

import numpy as np


def request_stream(traces, num_experts, max_tokens=None):
    """traces: dict layer_index -> int array [T, k]. Returns int64 keys in
    token-major order: key = layer_rank * num_experts + expert."""
    layers = sorted(traces)
    T = min(traces[l].shape[0] for l in layers)
    if max_tokens:
        T = min(T, max_tokens)
    k = traces[layers[0]].shape[1]
    stacked = np.stack([traces[l][:T].astype(np.int64) + r * num_experts
                        for r, l in enumerate(layers)], axis=1)  # [T, L, k]
    return stacked.reshape(-1), T, len(layers), k


def lru_hits(seq, cap):
    cache = OrderedDict()
    hits = 0
    for key in seq.tolist():
        if key in cache:
            hits += 1
            cache.move_to_end(key)
        else:
            if len(cache) >= cap:
                cache.popitem(last=False)
            cache[key] = None
    return hits


def belady_hits(seq, cap):
    """Belady's MIN: on a miss with a full cache, evict the key whose next use
    is farthest in the future. Optimal for any sequence (Belady 1966)."""
    keys = seq.tolist()
    n = len(keys)
    nxt = [0] * n
    last = {}
    for i in range(n - 1, -1, -1):
        nxt[i] = last.get(keys[i], n + i)  # never used again: distinct far-future sentinel
        last[keys[i]] = i
    cache, cur, heap, hits = set(), {}, [], 0
    for i, key in enumerate(keys):
        if key in cache:
            hits += 1
        else:
            if len(cache) >= cap:
                while True:
                    neg, victim = heapq.heappop(heap)
                    if victim in cache and cur[victim] == -neg:
                        cache.remove(victim)
                        break
            cache.add(key)
        cur[key] = nxt[i]
        heapq.heappush(heap, (-nxt[i], key))
    return hits


def simulate(traces, num_experts, fractions, max_tokens=None, belady=True):
    seq, T, L, k = request_stream(traces, num_experts, max_tokens)
    total_experts = L * num_experts
    out = []
    for f in fractions:
        cap = max(1, int(f * total_experts))
        lh = lru_hits(seq, cap)
        row = {"cache_fraction": f, "cache_experts": cap, "floor_hit": cap / total_experts,
               "lru_hit": lh / len(seq), "lru_misses_per_token": (len(seq) - lh) / T}
        if belady:
            bh = belady_hits(seq, cap)
            row.update({"belady_hit": bh / len(seq), "belady_misses_per_token": (len(seq) - bh) / T})
        out.append(row)
    return {"tokens": T, "moe_layers": L, "top_k": k, "num_experts": num_experts,
            "requests": int(len(seq)), "rows": out}


def main():
    p = argparse.ArgumentParser(description="expert-cache simulation from routing traces")
    p.add_argument("npz")
    p.add_argument("--fractions", default="0.05,0.1,0.2,0.3,0.5")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--no-belady", action="store_true")
    p.add_argument("--out-dir", default=None,
                   help="write here instead of results/<date>/ (use for tests and dry runs)")
    a = p.parse_args()
    z = np.load(a.npz)
    traces = {int(k[1:]): z[k] for k in z.files if k.startswith("L")}
    E = int(z["num_experts"])
    res = simulate(traces, E, [float(x) for x in a.fractions.split(",")], a.max_tokens,
                   belady=not a.no_belady)
    res["source"] = os.path.abspath(a.npz)
    print(f"{res['tokens']} tokens x {res['moe_layers']} MoE layers x top-{res['top_k']} of {E} experts")
    print(f"{'cache':>6s} {'floor':>7s} {'LRU':>7s} {'Belady':>7s} {'LRU miss/tok':>13s}")
    for r in res["rows"]:
        print(f"{r['cache_fraction']:6.2f} {r['floor_hit']:7.3f} {r['lru_hit']:7.3f} "
              f"{r.get('belady_hit', float('nan')):7.3f} {r['lru_misses_per_token']:13.1f}")
    name = os.path.splitext(os.path.basename(a.npz))[0].replace("traces_", "cache_")
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        out = os.path.join(a.out_dir, name + ".json")
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from _paths import write_path
        out = write_path(name + ".json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
