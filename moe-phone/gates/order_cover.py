"""
How much cover can INTRA-LAYER expert reordering possibly buy? The bound, from measured inputs only.

Patch 0010 computes a layer's resident experts before the ones still being read, so that their compute
overlaps the outstanding reads. Whether that can matter is arithmetic, and the arithmetic needs no new
measurement -- every input is already a committed artifact:

  active expert bytes per token   gates/gguf_active.py        (the model's own tensor table)
  expert-cache hit rate           gates/bmoe_analyze.py       (the engine's own counter, per run)
  weight-byte throughput          gates/device_bandwidth.py   (a resident model on the same device)
  flash read bandwidth            gates/g1_analyze.py         (measured bulk read rate, aggregate)
  layers, projections, top_k      the model's config, passed explicitly

Per layer and token:

  expert_bytes_layer   = expert_bytes_per_token / n_layer
  resident_bytes       = hit * expert_bytes_layer          (already in memory: pure compute)
  miss_bytes           = (1 - hit) * expert_bytes_layer    (must be read before use)
  cover_ms             = resident_bytes / device_GB_s      WALL time, because the device throughput was
                                                           measured with the same thread count -- this
                                                           is the mistake worth naming: a layer's
                                                           resident expert work is ~0.6 ms of WALL time,
                                                           not the ~2.4 ms of CPU time it represents.
  miss_read_ms         = miss_bytes / flash_GB_s
  per_projection       = both divided by n_proj, because a miss is waited on inside ONE projection and
                         only that projection's resident compute is available to cover it.

The bound on what reordering can save is then, per layer,

  saving_ms <= n_proj * min(cover_ms/n_proj, miss_read_ms/n_proj) = min(cover_ms, miss_read_ms)

and over the token, `saving_ms * n_layer`, expressed as a fraction of the measured decode time. It is an
UPPER bound in two ways that are stated rather than hidden: ascending-id order already collects part of
this cover for free (a miss at a random position among top_k experts is usually preceded by resident
ones whose compute overlaps its read), and the reads of a layer may already be partly complete when the
layer's consumption starts.

Run:
  python moe-phone/gates/order_cover.py --active results/2026-09-18/gguf_active_qwen_olmoe.json \\
      --engine results/2026-09-18/bmoe_order.json --cell residentfirst \\
      --bandwidth results/2026-09-18/device_bandwidth.json --device-arm olmoe_cpu \\
      --flash-gbps 2.806 --n-layer 48 --n-proj 3 --out-name order_cover.json
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--active", required=True)
    p.add_argument("--target", default="Qwen3-30B-A3B-Q4_0.gguf")
    p.add_argument("--engine", required=True, help="a bmoe_analyze.py artifact with the measured cell")
    p.add_argument("--cell", required=True)
    p.add_argument("--bandwidth", required=True)
    p.add_argument("--device-arm", required=True)
    p.add_argument("--flash-gbps", type=float, required=True,
                   help="measured aggregate flash read bandwidth (results/*/g1_storage.json bulk rate)")
    p.add_argument("--n-layer", type=int, required=True)
    p.add_argument("--n-proj", type=int, required=True, help="expert projections per layer (SwiGLU: 3)")
    p.add_argument("--out-name", required=True)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    act = json.load(open(a.active, encoding="utf-8"))
    eng = json.load(open(a.engine, encoding="utf-8"))
    bw = json.load(open(a.bandwidth, encoding="utf-8"))

    row = next(r for r in act if os.path.basename(r["file"]) == a.target)
    expert_bytes_tok = float(row["expert_bytes_per_token"])
    cell = next(c for c in eng["cells"] if c["cell"] == a.cell)
    hit = cell["cache_hit_pct_median"] / 100.0
    decode_ms = 1000.0 / cell["decode_tok_s_median"]
    dev = next(d for d in bw["devices"] if d["arm"] == a.device_arm)
    dev_gbps = dev["effective_GB_s"]

    per_layer = expert_bytes_tok / a.n_layer
    resident = hit * per_layer
    miss = (1.0 - hit) * per_layer
    cover_ms = resident / 1e9 / dev_gbps * 1000.0
    miss_ms = miss / 1e9 / a.flash_gbps * 1000.0
    saving_layer = min(cover_ms, miss_ms)
    saving_tok = saving_layer * a.n_layer

    # what the engine actually deferred, if the artifact carries the counter
    deferred = [r.get("probe_deferred_pct") for r in eng["runs"]
                if r.get("cell") == a.cell and r.get("probe_deferred_pct") is not None]

    out = {"target": a.target, "cell": a.cell, "device_arm": a.device_arm,
           "inputs": {"expert_bytes_per_token": expert_bytes_tok, "hit_rate": hit,
                      "device_GB_s": dev_gbps, "flash_GB_s": a.flash_gbps,
                      "n_layer": a.n_layer, "n_proj": a.n_proj, "decode_ms_per_token": decode_ms},
           "per_layer": {"expert_bytes": per_layer, "resident_bytes": resident, "miss_bytes": miss,
                         "resident_compute_wall_ms": cover_ms, "miss_read_ms": miss_ms,
                         "resident_compute_wall_ms_per_projection": cover_ms / a.n_proj,
                         "miss_read_ms_per_projection": miss_ms / a.n_proj,
                         "max_saving_ms": saving_layer},
           "per_token": {"max_saving_ms": saving_tok,
                         "max_saving_fraction_of_decode": saving_tok / decode_ms},
           "measured_deferred_pct": statistics.median(deferred) if deferred else None,
           "note": ("an upper bound: ascending-id order already collects part of this cover, because the "
                    "ready hook returns immediately for a resident expert, so a miss at a random "
                    "position among the selected experts is usually preceded by resident ones")}

    print(f"{a.target} cell {a.cell}: hit {hit*100:.1f}%, decode {decode_ms:.1f} ms/token")
    print(f"per layer: {per_layer/1e6:.1f} MB of experts -> resident {resident/1e6:.1f} MB "
          f"(compute {cover_ms:.2f} ms WALL at {dev_gbps:.2f} GB/s), miss {miss/1e6:.1f} MB "
          f"(read {miss_ms:.2f} ms at {a.flash_gbps:.3f} GB/s)")
    print(f"per projection: cover {cover_ms/a.n_proj:.2f} ms vs wait {miss_ms/a.n_proj:.2f} ms")
    print(f"UPPER BOUND on reordering: {saving_layer:.2f} ms/layer = {saving_tok:.1f} ms/token = "
          f"{saving_tok/decode_ms*100:.1f}% of the token")
    if out["measured_deferred_pct"] is not None:
        print(f"the engine actually deferred {out['measured_deferred_pct']:.2f}% of expert consumptions "
              f"to the second pass, so the reordering did fire")

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
