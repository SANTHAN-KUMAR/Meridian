"""
G-ROOF — flash-read roofline and model-scale frontier for MoE decode on a phone.

Level: L3 (estimability of the systems claim), per CLAUDE.md §4.3: this is the
"oracle first" step. Before building an engine, compute the best decode rate
ANY engine could reach if the only costs were (a) reading non-resident routed-
expert weights from flash and (b) reading resident weights from DRAM. If this
upper bound cannot clear a target, no engine can, and the target is dropped.

What it computes, per model, from the model's own config.json and HF parameter
count (no hand-transcribed architecture numbers):

  E_tok   routed-expert parameters touched per token
          = n_moe_layers * top_k * 3 * hidden * expert_ffn        (SwiGLU: gate, up, down)
  E_all   routed-expert parameters in total
          = n_moe_layers * n_experts * 3 * hidden * expert_ffn
  R       resident (non-routed) parameters = total_params - E_all
          (attention, embeddings, lm_head, routers, shared experts, dense layers)

  flash bytes / token = E_tok * bpw/8 * (1 - h) * rho(d)
  DRAM bytes  / token = (R + E_tok * h * rho(d)) * bpw/8   (resident weights + cache hits)

  h    expert-cache hit rate. h_floor = cache_bytes / E_all_bytes is the hit
       rate of a cache with no locality at all (uniform independent routing);
       measured traces must beat it. Other h values are SCENARIOS, not claims.
  d    fraction of an expert's FFN neurons actually needed (1.0 = dense).
  rho  read fraction given d:
         'predicted'  : rho = d               (neurons known before any read)
         'gate_first' : rho = 1/3 + 2/3 * d   (read all gate rows, compute the
                        gate exactly, then read only the needed up/down rows —
                        needs no predictor)

  Flash time charges BULK reads (whole expert tensors, >=512 KB) at the bulk
  bandwidth and SCATTERED reads (per-neuron rows, ~1-4 KB) at the small-read
  bandwidth. Phone UFS delivers several times less per byte on small random
  reads (PowerInfer-2, arXiv 2406.06282 §I/O characterisation: ~3.5 GB/s at
  512 KB vs 0.45-1 GB/s at 4 KB on a OnePlus 12), so a sparse read that saves
  bytes can still lose time. Both bandwidths are required arguments.

  tok/s upper bound  = 1 / max(flash_time, dram_time)   (perfect overlap)
  tok/s serial bound = 1 / (flash_time + dram_time)      (no overlap)

Everything that is a property of the PHONE (flash bandwidth, DRAM bandwidth,
RAM available for weights) is a required argument — never a default — and the
artifact records whether each value was measured or assumed. Bits per weight
are exact properties of the storage formats (ggml block layouts), not tuned:
  Q4_0  = 18 bytes / 32 weights = 4.5 bpw
  Q8_0  = 34 bytes / 32 weights = 8.5 bpw
  MXFP4 = 17 bytes / 32 weights = 4.25 bpw (OCP MX spec: 32 x FP4 + 1 x E8M0 scale)

Ignored, and therefore making this an UPPER bound: compute time, KV-cache
reads, flash read-size inefficiency (small random reads run far below peak —
measured separately by the UFS read-size gate), prediction misses, OS page-
cache effects, thermal throttling. Every one of those only lowers the rate.

Run (until the UFS read-size gate has measured the 15R, the flash values are
PowerInfer-2's OnePlus 12 measurements and must be declared --assumed):
  python moe-phone/gates/byte_budget.py --ram-gb 8 --flash-gbps 3.5 \
      --flash-gbps-scattered 0.45 --dram-gbps 45 --assumed flash,dram,ram \
      --threshold-tps 5
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import PROJECT, write_path  # noqa: E402

CACHE = os.path.join(PROJECT, "cache", "hf")  # not committed (see .gitignore)

# Repository names only; every architecture number is read from the repo.
DEFAULT_MODELS = [
    "allenai/OLMoE-1B-7B-0924",
    "deepseek-ai/DeepSeek-V2-Lite",
    "Qwen/Qwen3-30B-A3B",
    "openai/gpt-oss-20b",
    "mistralai/Mixtral-8x7B-v0.1",
    "PowerInfer/TurboSparse-Mixtral",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "openai/gpt-oss-120b",
    "zai-org/GLM-4.5-Air",
    "Qwen/Qwen3-235B-A22B",
]

BPW = {"Q4_0": 4.5, "Q8_0": 8.5, "MXFP4": 4.25}
DENSITIES = [1.0, 0.5, 0.3, 0.1]
HIT_SCENARIOS = [0.5, 0.7, 0.9]


def _fetch_json(url, cache_name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, cache_name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    req = urllib.request.Request(url, headers={"User-Agent": "moe-phone-gate"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def _first(cfg, *keys):
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k], k
    return None, None


def moe_geometry(cfg):
    """Routed-expert geometry from a config.json. Raises on anything it cannot
    derive unambiguously — a guessed architecture is worse than no number."""
    # Some repos nest the text model config (multimodal wrappers).
    if "text_config" in cfg and "num_hidden_layers" not in cfg:
        cfg = cfg["text_config"]
    L = cfg["num_hidden_layers"]
    H = cfg["hidden_size"]
    n_exp, k_exp = _first(cfg, "num_experts", "num_local_experts", "n_routed_experts",
                          "moe_num_experts")
    top_k, k_top = _first(cfg, "num_experts_per_tok", "experts_per_token", "moe_k")
    ffn, k_ffn = _first(cfg, "moe_intermediate_size", "intermediate_size")
    if n_exp is None or top_k is None or ffn is None:
        raise ValueError("config has no recognisable routed-expert fields")
    for name, v in (("num_experts", n_exp), ("top_k", top_k), ("expert_ffn", ffn)):
        if not isinstance(v, int):
            raise ValueError(f"{name} is {v!r}; non-scalar expert geometry is not supported")

    # Which layers are MoE layers. Each rule is recorded so the artifact is auditable.
    rule = "all layers"
    moe_layers = list(range(L))
    if cfg.get("first_k_dense_replace"):
        k = cfg["first_k_dense_replace"]
        moe_layers = [i for i in moe_layers if i >= k]
        rule = f"first_k_dense_replace={k}"
    if cfg.get("mlp_only_layers"):
        dense = set(cfg["mlp_only_layers"])
        moe_layers = [i for i in moe_layers if i not in dense]
        rule += f"; mlp_only_layers={sorted(dense)}"
    step = cfg.get("decoder_sparse_step")
    if step and step > 1:
        moe_layers = [i for i in moe_layers if (i + 1) % step == 0]
        rule += f"; decoder_sparse_step={step}"
    if "moe_layer_start_index" in cfg:
        s = cfg["moe_layer_start_index"]
        iv = cfg.get("moe_layer_interval", 1)
        moe_layers = [i for i in moe_layers if i >= s and (i - s) % iv == 0]
        rule += f"; moe_layer_start_index={s}, interval={iv}"
    if cfg.get("interleave_moe_layer_step", 1) > 1:
        iv = cfg["interleave_moe_layer_step"]
        moe_layers = [i for i in moe_layers if (i + 1) % iv == 0]
        rule += f"; interleave_moe_layer_step={iv}"
    if "num_dense_layers" in cfg:
        k = cfg["num_dense_layers"]
        moe_layers = [i for i in moe_layers if i >= k]
        rule += f"; num_dense_layers={k}"

    per_expert = 3 * H * ffn
    return {
        "layers": L, "moe_layers": len(moe_layers), "moe_layer_rule": rule,
        "hidden": H, "expert_ffn": ffn, "num_experts": n_exp, "top_k": top_k,
        "keys_used": [k_exp, k_top, k_ffn],
        "E_tok_params": len(moe_layers) * top_k * per_expert,
        "E_all_params": len(moe_layers) * n_exp * per_expert,
    }


def total_params(repo):
    info = _fetch_json(f"https://huggingface.co/api/models/{repo}",
                       repo.replace("/", "__") + "__api.json")
    st = info.get("safetensors") or {}
    if "total" not in st:
        raise ValueError("HF API reports no safetensors parameter total")
    return st["total"], st.get("parameters", {})


def resident_from_config(cfg, g):
    """Non-routed parameters derived from config alone, for plain GQA
    attention models. Returns None for architectures this formula does not
    describe (MLA, linear/conv hybrids, shared experts, dense MLP layers),
    rather than guessing. Norms and biases are omitted (<0.1% of weights)."""
    if "text_config" in cfg and "num_hidden_layers" not in cfg:
        cfg = cfg["text_config"]
    exotic = ("kv_lora_rank", "linear_num_key_heads", "layer_types_conv",
              "shared_expert_intermediate_size", "n_shared_experts",
              "moe_num_shared_experts", "num_dense_layers", "first_k_dense_replace")
    if any(cfg.get(k) for k in exotic) or g["moe_layers"] != g["layers"]:
        return None
    lt = cfg.get("layer_types") or []
    if any(t not in ("full_attention", "sliding_attention") for t in lt):
        return None
    H, L = cfg["hidden_size"], cfg["num_hidden_layers"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim") or H // nh
    attn = L * (H * nh * hd + 2 * H * nkv * hd + nh * hd * H)
    router = L * H * g["num_experts"]
    emb = cfg["vocab_size"] * H * (1 if cfg.get("tie_word_embeddings") else 2)
    return attn + router + emb


def analyse(repo, a):
    cfg = _fetch_json(f"https://huggingface.co/{repo}/resolve/main/config.json",
                      repo.replace("/", "__") + "__config.json")
    g = moe_geometry(cfg)
    total, by_dtype = total_params(repo)
    R_cfg = resident_from_config(cfg, g)
    # Packed low-bit checkpoints (e.g. MXFP4 stored as U8 blocks) make the HF
    # parameter total count storage elements, not weights. Use the config-
    # derived resident count if the architecture allows it; otherwise refuse
    # rather than silently produce a wrong resident size.
    packed = [d for d in by_dtype if d.upper() in ("U8", "I8", "U4", "I4")]
    if packed:
        if R_cfg is None:
            raise ValueError(f"checkpoint stores packed dtypes {packed} and the architecture "
                             f"is outside resident_from_config — resolve by hand")
        R, R_method = R_cfg, "config-derived (HF total counts packed storage)"
        total = R + g["E_all_params"]
    else:
        R, R_method = total - g["E_all_params"], "HF total minus routed experts"
    if not (0 < R < total):
        raise ValueError(f"resident params {R} outside (0, total={total}); "
                         f"expert geometry and parameter total disagree")
    # CLAUDE.md §6.5: a derived quantity is recomputed a second way where possible.
    cross = None
    if R_cfg is not None and not packed:
        cross = (R - R_cfg) / R

    rows = []
    for fmt, bpw in BPW.items():
        B = bpw / 8.0
        res_bytes = R * B
        cache_bytes = a.ram_gb * 1e9 - res_bytes
        fits = cache_bytes >= 0
        E_all_b = g["E_all_params"] * B
        h_floor = max(0.0, min(1.0, cache_bytes / E_all_b)) if fits else 0.0
        for h_label, h in [("floor", h_floor)] + [(f"{x:.1f}", x) for x in HIT_SCENARIOS]:
            if h < h_floor:
                continue  # a scenario below the no-locality floor is not informative
            for mode in ("gate_first", "predicted"):
                for d in DENSITIES:
                    miss = g["E_tok_params"] * B * (1 - h)
                    if d == 1.0:                      # dense: whole tensors, bulk reads
                        bulk, scat = miss, 0.0
                    elif mode == "gate_first":        # gate rows bulk, up/down rows scattered
                        bulk, scat = miss / 3, miss * 2 / 3 * d
                    else:                             # predicted: every read is per-neuron
                        bulk, scat = 0.0, miss * d
                    # Cache hits are read from DRAM, and sparsity trims those reads too
                    # (DRAM has no first-order small-read penalty at these row sizes).
                    rho = 1.0 if d == 1.0 else (d if mode == "predicted" else 1 / 3 + 2 / 3 * d)
                    dram_b = res_bytes + g["E_tok_params"] * B * h * rho
                    t_f = bulk / (a.flash_gbps * 1e9) + scat / (a.flash_gbps_scattered * 1e9)
                    t_d = dram_b / (a.dram_gbps * 1e9)
                    rows.append({
                        "format": fmt, "bpw": bpw, "resident_fits_ram": fits,
                        "h_label": h_label, "h": round(h, 4), "mode": mode, "d": d,
                        "flash_bulk_GB_per_tok": bulk / 1e9,
                        "flash_scattered_GB_per_tok": scat / 1e9,
                        "dram_GB_per_tok": dram_b / 1e9,
                        "tps_overlap_bound": (1 / max(t_f, t_d)) if fits else 0.0,
                        "tps_serial_bound": (1 / (t_f + t_d)) if fits else 0.0,
                    })
    return {"repo": repo, "geometry": g, "total_params": total,
            "resident_params": R, "resident_method": R_method,
            "resident_crosscheck_rel_diff": cross, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--ram-gb", type=float, required=True,
                   help="RAM available for model weights (resident + expert cache), GB")
    p.add_argument("--flash-gbps", type=float, required=True,
                   help="flash bandwidth for BULK reads (whole expert tensors), GB/s")
    p.add_argument("--flash-gbps-scattered", type=float, required=True,
                   help="flash bandwidth for SCATTERED per-neuron reads (~1-4 KB), GB/s")
    p.add_argument("--dram-gbps", type=float, required=True,
                   help="achievable DRAM bandwidth for decode, GB/s")
    p.add_argument("--assumed", default="",
                   help="comma list of inputs that are ASSUMED rather than measured: "
                        "flash,dram,ram — recorded in the artifact")
    p.add_argument("--threshold-tps", type=float, required=True,
                   help="decode rate counted as 'comfortable'; its source must be cited "
                        "in ESTIMAND.md §4")
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    p.add_argument("--tag", default="",
                   help="suffix for the artifact name, so scenario runs do not overwrite each other")
    a = p.parse_args()

    out = {"inputs": vars(a), "assumed": [x for x in a.assumed.split(",") if x],
           "status": "UPPER BOUND — see module docstring for every ignored cost",
           "models": [], "failures": []}
    for repo in a.models:
        try:
            out["models"].append(analyse(repo, a))
        except (ValueError, KeyError, urllib.error.URLError, urllib.error.HTTPError) as e:
            # Counted and reported, never silently dropped (CLAUDE.md §6.3).
            out["failures"].append({"repo": repo, "error": f"{type(e).__name__}: {e}"})

    hdr = f"{'model':34s} {'E_tok':>7s} {'E_all':>7s} {'resid':>7s}  " \
          f"{'Q4_0 dense h=floor':>18s}  {'best Q4_0 gate_first d=0.1 h=floor':>34s}"
    print(out["status"])
    if out["assumed"]:
        print("ASSUMED (unmeasured) inputs:", ", ".join(out["assumed"]))
    print(hdr)
    for m in out["models"]:
        g = m["geometry"]

        def pick(mode, d):
            for r in m["rows"]:
                if r["format"] == "Q4_0" and r["h_label"] == "floor" and r["mode"] == mode and r["d"] == d:
                    return r
        dense, sparse = pick("gate_first", 1.0), pick("gate_first", 0.1)
        fmt = lambda r: ("does not fit" if not r["resident_fits_ram"]
                         else f"{r['tps_overlap_bound']:6.2f} tok/s")
        print(f"{m['repo']:34s} {g['E_tok_params']/1e9:6.2f}B {g['E_all_params']/1e9:6.1f}B "
              f"{m['resident_params']/1e9:6.2f}B  {fmt(dense):>18s}  {fmt(sparse):>34s}")
    for f in out["failures"]:
        print(f"FAILED {f['repo']}: {f['error']}")
    print(f"{len(out['failures'])} of {len(a.models)} models could not be analysed.")

    path = write_path(f"byte_budget{'_' + a.tag if a.tag else ''}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
