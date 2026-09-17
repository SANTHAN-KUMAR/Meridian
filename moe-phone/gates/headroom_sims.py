"""
HEADROOM SIMS — the load-bearing offline results, re-evaluated at the PHONE's operating point.

Every cache-policy conclusion in ARCHITECTURE.md was drawn on OLMoE at cache fractions of
5-20% (rho = per-layer slots / top_k of 0.4-1.6, i.e. a cache that cannot hold one token's
working set). The engine that actually runs on the 15R (BigMoeOnEdge, Qwen3-30B-A3B, 4 GB
cache) sits at rho ~ 4.1 and is CPU-compute-bound (~100 ms/token) with a ~50 ms/token I/O
stall. This script re-runs the same simulators, unchanged, at rho 2-4, and adds the two
quantities the phone breakdown says matter and no gate measured:

  union        fetches per verify window / fetches per single token, at rho 2-4: the I/O
               side of the verify cost c(N) (gates/verify_cost_analyze.py measured the
               whole c(N) on the phone; this separates the part bytes alone explain)
  miss_dist    per layer-event: P(>=1 miss) and the miss-count histogram under per-layer
               LRU. The head-of-line stall a one-expert prefetch could remove scales with
               P(>=1 miss), not with the mean miss count; the LRU-to-Belady gap bounds any
               eviction policy at that rho
  lookahead    hit rate vs horizon (tokens) at rho 2.4-4, plus protect/rank eviction under
               an imperfect predictor at the largest horizon (cache_sim.lookahead_hits)
  prefetch     within-token cross-layer prefetch (gates/prefetch_sim.replay) at rho 2.4 and
               4 with the forward-pass cost swept through the phone's measured ~100 ms
  lexical      how much of each layer's routing the TOKEN ID alone predicts (a table fitted
               on the first half of the corpus, scored on the second), against persistence
               and independence. Needs the token-id files written by
               gates/llamacpp_traces.py tokens (--tokens-dir); skipped, and said so, if absent
  phone_csv    the per-token CSVs of results/2026-09-17/bmoe_pin2: steady-state rate over
               tokens 65-256 (ESTIMAND.md §1) against the 256-token mean the claims quote,
               the three time terms at steady state, within-run drift of the compute
               residual, and a regression of per-token stall on flash bytes (a fixed
               intercept = per-layer head-of-line latency; a slope = bandwidth-limited part)
  engine_vs_curve  the engine's measured Qwen3-30B-A3B hit rate against BOTH S9 hypotheses
               evaluated at the engine's own rho — an on-device data point for S9
  q4_entropy   zeroth-order entropy of the Q4_0 nibbles and fp16 scales of the deployed
               Qwen3-30B-A3B expert tensors: the bound on lossless expert compression on flash
               (--gguf; skipped, and said so, if the file is absent)

Nothing here tunes anything. Trace sims use --max-tokens 16384 (recorded) because the
lookahead scan is O(cache) per eviction; the 32768-token artifacts of cache_sim differ from
these by < 0.005 where both exist (compare cache_fcrit_OLMoE-1B-7B-0924.json at rho 4).

Run (about 40 minutes, single core):
  python moe-phone/gates/headroom_sims.py [--tokens-dir ../moe-work] [--max-tokens 16384]
"""
import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import struct
import sys
from collections import Counter, OrderedDict, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_sim  # noqa: E402
import prefetch_sim  # noqa: E402
from _paths import PROJECT, REPO, read_path, write_path  # noqa: E402

TRACES = {
    "OLMoE": ("traces_OLMoE-1B-7B-0924.npz", None),
    "OLMoE-q4-llamacpp": ("traces_OLMoE-1B-7B-0924-q4_0-llamacpp.npz", "tok_olmoe.bin"),
    "gpt-oss-20b": ("traces_gpt-oss-20b-mxfp4-llamacpp.npz", "tok_gpt-oss-20b.bin"),
    "granite-3b": ("traces_granite-3.1-3b-a800m-q4_0-llamacpp.npz", "tok_granite-3.1-3b-a800m.bin"),
}
RHOS = {"OLMoE": [1.6, 2.4, 3.2, 4.0], "gpt-oss-20b": [2.0, 2.86, 4.0], "granite-3b": [2.0, 3.0, 4.0]}
WINDOWS = [1, 2, 3, 4, 5, 8]
HORIZONS = [0, 1, 2, 4, 8, 16, 32]
ACCS = [1.0, 0.7, 0.5, 0.35]
PREFETCH_FRACS = [0.3, 0.5]
PREFETCH_FWD_MS = [30, 60, 100, 160]
# G1 bulk bandwidth and the OLMoE Q4_0 expert size, the same constants engine_sim was run with
# (cross-checked by claims_check.roofline_inputs_agree).
BULK_GBPS, EXPERT_MB = 2.806, 3.54
STEADY_FROM = 64  # ESTIMAND.md §1: steady state is tokens 65..; the first 64 are cold start


def load_trace(name, max_tokens):
    z = np.load(read_path(TRACES[name][0]))
    tr = {int(k[1:]): z[k] for k in z.files if k.startswith("L")}
    E = int(z["num_experts"])
    ev, T, L, k = cache_sim.event_stream(tr, E, max_tokens)
    return ev, T, L, k, E, tr


def cap_for(rho, k, E, L):
    f = rho * k / E
    return f, max(1, int(round(f * L * E)))


def sec_union(max_tokens):
    out = {}
    for name, rhos in RHOS.items():
        ev, T, L, k, E, _ = load_trace(name, max_tokens)
        rows = []
        for rho in rhos:
            f, cap = cap_for(rho, k, E, L)
            lru = cache_sim.lru_hits_events(ev, cap, "per_layer", "atomic") / (T * L * k)
            fw = {W: cache_sim.window_fetches(ev, cap, W, "per_layer") for W in WINDOWS}
            base = fw[1][0] / fw[1][1]
            rows.append({"rho": rho, "frac": f, "lru": lru, "fetch_per_token_W1": base,
                         "c_io": {str(W): (fe / nw) / base for W, (fe, nw) in fw.items()}})
            print(f"union {name} rho {rho}: lru {lru:.3f} c_io {rows[-1]['c_io']}", flush=True)
        out[name] = {"E": E, "k": k, "L": L, "T": T, "rows": rows}
    return out


def sec_miss_dist(max_tokens):
    out = {}
    for name, rhos in RHOS.items():
        ev, T, L, k, E, _ = load_trace(name, max_tokens)
        rows = []
        for rho in rhos:
            f, cap = cap_for(rho, k, E, L)
            caps = cache_sim.split_capacity(cap, L)
            caches = [OrderedDict() for _ in range(L)]
            hist = np.zeros(k + 1, int)
            misses = 0
            for t in range(T):
                for l in range(L):
                    c, cp, miss = caches[l], caps[l], []
                    for key in ev[t, l].tolist():
                        if key in c:
                            c.move_to_end(key)
                        else:
                            miss.append(key)
                    for key in miss:
                        while len(c) >= cp:
                            c.popitem(last=False)
                        c[key] = None
                    hist[len(miss)] += 1
                    misses += len(miss)
            bel = cache_sim.belady_hits_scoped(ev, cap, "per_layer") / (T * L * k)
            rows.append({"rho": rho, "frac": f, "lru_hit": 1 - misses / (T * L * k), "belady_hit": bel,
                         "p_event_ge1_miss": float(1 - hist[0] / hist.sum()),
                         "mean_miss_per_event": misses / (T * L),
                         "miss_hist": (hist / hist.sum()).tolist()})
            print(f"miss {name} rho {rho}: {rows[-1]}", flush=True)
        out[name] = {"E": E, "k": k, "rows": rows}
    return out


def sec_lookahead(max_tokens, seed=0):
    out = {}
    for name, fracs in (("OLMoE", [0.3, 0.5]), ("gpt-oss-20b", [0.25, 0.5])):
        ev, T, L, k, E, _ = load_trace(name, max_tokens)
        n = T * L * k
        rows = []
        for f in fracs:
            cap = max(1, int(round(f * L * E)))
            lru = cache_sim.lru_hits_events(ev, cap, "per_layer", "atomic") / n
            bel = cache_sim.belady_hits_scoped(ev, cap, "per_layer") / n
            byh = {str(h): cache_sim.lookahead_hits(ev, cap, h) / n for h in HORIZONS}
            H = max(HORIZONS)
            pred = {}
            for mode in ("rank", "protect"):
                ys = {}
                for acc in ACCS:
                    hat = None if acc >= 1.0 else cache_sim.corrupt_routing(ev, acc, E, seed)
                    ys[str(acc)] = cache_sim.lookahead_hits(ev, cap, 8, ev_hat=hat, mode=mode) / n
                pred[mode] = ys
            row = {"cache_fraction": f, "rho": f * E / k, "lru_atomic": lru, "belady": bel,
                   "by_horizon": byh,
                   "gap_closed_by_horizon": {h: (v - byh["0"]) / (bel - byh["0"]) if bel > byh["0"] else None
                                             for h, v in byh.items()},
                   "prediction_at_horizon_8": pred}
            rows.append(row)
            print(f"lookahead {name} frac {f} rho {row['rho']:.2f}: {byh} belady {bel:.3f} "
                  f"gap4 {row['gap_closed_by_horizon']['4']:.2f}", flush=True)
        out[name] = {"E": E, "k": k, "horizons": HORIZONS, "accuracies": ACCS, "rows": rows}
    return out


def sec_prefetch(max_tokens):
    ev, T, L, k, E, _ = load_trace("OLMoE", min(max_tokens, 8192))
    g3 = json.load(open(read_path("g3_OLMoE-1B-7B-0924.json"), encoding="utf-8"))
    recall = {int(l): float(v) for l, v in g3["lookahead_recall"].items()}
    t_exp = EXPERT_MB * 1e6 / (BULK_GBPS * 1e9)
    rows = []
    for f in PREFETCH_FRACS:
        cap = max(1, int(round(f * L * E)))
        for cms in PREFETCH_FWD_MS:
            base = prefetch_sim.replay(ev, E, cap, recall, cms / 1e3, t_exp, prefetch=False)
            pf = prefetch_sim.replay(ev, E, cap, recall, cms / 1e3, t_exp, prefetch=True)
            rows.append({"cache_fraction": f, "rho": f * E / k, "fwd_ms": cms, "no_prefetch": base,
                         "prefetch": pf, "speedup": pf["tok_s"] / base["tok_s"]})
            print(f"prefetch frac {f} fwd {cms}: {rows[-1]['speedup']:.3f}", flush=True)
    return {"tokens": T, "recall_by_layer": recall, "rows": rows}


def sec_lexical(tokens_dir, max_tokens):
    out = {}
    for name in ("OLMoE-q4-llamacpp", "gpt-oss-20b", "granite-3b"):
        tokf = os.path.join(tokens_dir, TRACES[name][1])
        if not os.path.exists(tokf):
            out[name] = {"skipped": f"token file absent: {tokf} (gates/llamacpp_traces.py tokens)"}
            print(f"lexical {name}: SKIPPED, {tokf} absent", flush=True)
            continue
        z = np.load(read_path(TRACES[name][0]))
        Ls = sorted(int(kk[1:]) for kk in z.files if kk.startswith("L"))
        E = int(z["num_experts"])
        ev = np.stack([z[f"L{l}"] for l in Ls], axis=1)
        with open(tokf, "rb") as fh:
            nwin, wl = struct.unpack("<ii", fh.read(8))
            ids = np.frombuffer(fh.read(), dtype=np.int32)[:nwin * wl]
        T = min(ev.shape[0], ids.size, max_tokens if max_tokens else ev.shape[0])
        ev, ids = ev[:T], ids[:T]
        k, nL, half = ev.shape[2], ev.shape[1], T // 2
        layers = []
        for l in range(nL):
            a, b = ev[half:, l, :], ev[half - 1:-1, l, :]
            pers = float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a.tolist(), b.tolist())]))
            uni, bi = defaultdict(Counter), defaultdict(Counter)
            for t in range(half):
                for e in ev[t, l].tolist():
                    uni[int(ids[t])][e] += 1
                    if t > 0:
                        bi[(int(ids[t - 1]), int(ids[t]))][e] += 1
            hit_u = hit_b = seen_u = seen_b = 0
            for t in range(half, T):
                true, tid = set(ev[t, l].tolist()), int(ids[t])
                if tid in uni:
                    seen_u += 1
                    hit_u += len(true & set(e for e, _ in uni[tid].most_common(k)))
                key = (int(ids[t - 1]), tid)
                if key in bi and sum(bi[key].values()) >= 2 * k:
                    pred, seen_b = [e for e, _ in bi[key].most_common(k)], seen_b + 1
                elif tid in uni:
                    pred = [e for e, _ in uni[tid].most_common(k)]
                else:
                    pred = []
                hit_b += len(true & set(pred))
            n = T - half
            layers.append({"layer": l, "lexical_unigram_slot_acc": hit_u / (n * k),
                           "unigram_coverage": seen_u / n, "bigram_backoff_slot_acc": hit_b / (n * k),
                           "bigram_coverage": seen_b / n, "persistence_slot_acc": pers,
                           "independence": k / E})
        res = {"E": E, "k": k, "T": T, "token_file": tokf, "token_file_sha256": sha256(tokf),
               "layers": layers,
               "mean_lexical": float(np.mean([r["lexical_unigram_slot_acc"] for r in layers])),
               "mean_bigram": float(np.mean([r["bigram_backoff_slot_acc"] for r in layers])),
               "mean_persistence": float(np.mean([r["persistence_slot_acc"] for r in layers])),
               "independence": k / E}
        print(f"lexical {name}: mean lexical {res['mean_lexical']:.3f} bigram {res['mean_bigram']:.3f} "
              f"persistence {res['mean_persistence']:.3f} indep {k / E:.3f}", flush=True)
        out[name] = res
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sec_q4_entropy(gguf_path, gguf_py, n_tensors=6):
    """Shannon entropy of the Q4_0 nibble stream and of the fp16 scale stream of routed-expert
    tensors, straight from the deployment file. Bounds any lossless compression of experts on
    flash: bits/weight achievable ~ H(nibble) + H(scale)/32 against the 4.5 stored. One pass over
    a sample of expert tensors (evenly spaced layers, all three projections); no model run."""
    if not os.path.exists(gguf_path):
        return {"skipped": f"GGUF absent: {gguf_path}"}
    if gguf_py and gguf_py not in sys.path:
        sys.path.insert(0, gguf_py)
    from gguf import GGUFReader
    r = GGUFReader(gguf_path)
    exps = [t for t in r.tensors if "_exps" in t.name and t.tensor_type.name == "Q4_0"]
    if not exps:
        return {"skipped": "no Q4_0 expert tensors in file"}
    step = max(1, len(exps) // n_tensors)
    picked = exps[::step][:n_tensors]
    nib_hist = np.zeros(16, np.int64)
    scale_hist = Counter()
    n_blocks = 0
    rows = []
    for t in picked:
        raw = np.asarray(t.data).reshape(-1).view(np.uint8)
        blocks = raw.reshape(-1, 18)                     # Q4_0: fp16 scale + 16 bytes of nibbles
        q = blocks[:, 2:]
        h = np.bincount(np.concatenate([q & 0x0F, q >> 4]).reshape(-1), minlength=16)
        nib_hist += h
        sc = blocks[:, :2].copy().view(np.uint16).reshape(-1)
        scale_hist.update(np.bincount(sc, minlength=65536).nonzero()[0].tolist())  # distinct only
        sh = np.bincount(sc, minlength=65536)
        ps = sh[sh > 0] / sh.sum()
        n_blocks += blocks.shape[0]
        ph = h / h.sum()
        rows.append({"tensor": t.name, "blocks": int(blocks.shape[0]),
                     "nibble_entropy_bits": float(-(ph[ph > 0] * np.log2(ph[ph > 0])).sum()),
                     "scale_entropy_bits": float(-(ps * np.log2(ps)).sum())})
        print(f"q4_entropy {t.name}: nibble H {rows[-1]['nibble_entropy_bits']:.3f} bits, "
              f"scale H {rows[-1]['scale_entropy_bits']:.2f} bits", flush=True)
    p = nib_hist / nib_hist.sum()
    h_nib = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    h_scale = float(np.mean([x["scale_entropy_bits"] for x in rows]))
    bpw_bound = h_nib + h_scale / 32
    return {"gguf": os.path.basename(gguf_path), "gguf_sha256_first_1MiB": sha256_prefix(gguf_path),
            "tensors": rows, "n_blocks": int(n_blocks), "nibble_hist": (p).tolist(),
            "nibble_entropy_bits_pooled": h_nib, "scale_entropy_bits_mean": h_scale,
            "stored_bits_per_weight": 4.5, "entropy_bound_bits_per_weight": bpw_bound,
            "max_lossless_saving_fraction": 1 - bpw_bound / 4.5,
            "note": "zeroth-order (memoryless) entropy: an upper bound on what any order-0 entropy coder "
                    "saves; context models could do a little better, never worse than this bound says"}


def sha256_prefix(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()


def sec_phone_csv():
    root = os.path.join(PROJECT, "results", "2026-09-17", "bmoe_pin2")
    cells = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(root, "*.csv"))):
        rows = [{kk: float(v) for kk, v in r.items()}
                for r in csv.DictReader(l for l in open(p, encoding="utf-8") if not l.startswith("#"))]
        if len(rows) < 100:
            continue
        tag = os.path.basename(p)[:-4]
        cell = tag.rpartition("_rep")[0]
        ss = rows[STEADY_FROM:]
        x = np.array([r["read_bytes"] / 2 ** 20 for r in ss])
        y = np.array([r["stall_ms"] for r in ss])
        slope, icpt = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
        cells[cell].append({
            "tag": tag,
            "tok_s_all_256": 1000 * len(rows) / sum(r["wall_ms"] for r in rows),
            "tok_s_first_64": 1000 * STEADY_FROM / sum(r["wall_ms"] for r in rows[:STEADY_FROM]),
            "tok_s_steady_65_256": 1000 * len(ss) / sum(r["wall_ms"] for r in ss),
            "compute_ms_steady": float(np.mean([r["compute_ms"] for r in ss])),
            "stall_ms_steady": float(np.mean([r["stall_ms"] for r in ss])),
            "mgmt_ms_steady": float(np.mean([r["mgmt_ms"] for r in ss])),
            "majflt_per_tok_steady": float(np.mean([r["majflt"] for r in ss])),
            "read_MiB_per_tok_steady": float(x.mean()),
            "compute_ms_first32": float(np.mean([r["compute_ms"] for r in rows[:32]])),
            "compute_ms_last32": float(np.mean([r["compute_ms"] for r in rows[-32:]])),
            "stall_regression": {"intercept_ms": float(icpt), "slope_ms_per_MiB": float(slope),
                                 "marginal_GBps": float(1000 / slope / 1024) if slope > 0 else None,
                                 "r": float(np.corrcoef(x, y)[0, 1])},
            "cache_hit_pct_cumulative_end": rows[-1]["cache_hit_pct"],
        })
    summary = {}
    for cell, rs in cells.items():
        med = lambda key: float(np.median([r[key] for r in rs]))  # noqa: E731
        summary[cell] = {"n": len(rs), "runs": rs,
                         "tok_s_all_256_median": med("tok_s_all_256"),
                         "tok_s_steady_median": med("tok_s_steady_65_256"),
                         "tok_s_first_64_median": med("tok_s_first_64"),
                         "compute_ms_steady_median": med("compute_ms_steady"),
                         "stall_ms_steady_median": med("stall_ms_steady"),
                         "mgmt_ms_steady_median": med("mgmt_ms_steady"),
                         "stall_intercept_ms_median": float(np.median([r["stall_regression"]["intercept_ms"] for r in rs])),
                         "stall_slope_ms_per_MiB_median": float(np.median([r["stall_regression"]["slope_ms_per_MiB"] for r in rs])),
                         "compute_drift_last32_over_first32_median": float(np.median(
                             [r["compute_ms_last32"] / r["compute_ms_first32"] for r in rs]))}
        print(f"phone_csv {cell}: 256-mean {summary[cell]['tok_s_all_256_median']:.2f} steady "
              f"{summary[cell]['tok_s_steady_median']:.2f} compute {summary[cell]['compute_ms_steady_median']:.1f} "
              f"stall {summary[cell]['stall_ms_steady_median']:.1f} mgmt {summary[cell]['mgmt_ms_steady_median']:.1f}",
              flush=True)
    return {"source": root, "steady_from_token": STEADY_FROM + 1, "cells": summary}


def sec_engine_vs_curve():
    """The engine's measured hit rate on Qwen3-30B-A3B against H_rho and H_floor at ITS rho."""
    bb = json.load(open(read_path("byte_budget_measured.json"), encoding="utf-8"))
    m = next(x for x in bb["models"] if x["repo"] == "Qwen/Qwen3-30B-A3B")
    g = m["geometry"]
    e_all_b = g["E_all_params"] * 4.5 / 8
    pin2 = json.load(open(read_path("bmoe_pin2.json"), encoding="utf-8"))
    runs = [r for r in pin2["runs"] if r["cell"] == "pin_t4c47_io4c03"]
    budget_b = float(np.median([r["budget_MiB"] for r in runs])) * 2 ** 20
    hit = float(np.median([r["cache_hit_pct"] for r in runs])) / 100
    f = budget_b / e_all_b
    rho = f * g["num_experts"] / g["top_k"]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import engine_target
    curve = engine_target.load_lru_curve(read_path("cache_fcrit_OLMoE-1B-7B-0924.json"))
    h_rho, _, extrap = engine_target.at_rho(curve, rho)
    f0 = rho * curve["top_k"] / curve["num_experts"]
    h_floor = min(1.0, f + (h_rho - f0))
    out = {"cache_budget_MiB": budget_b / 2 ** 20, "expert_bytes_all_GB": e_all_b / 1e9,
           "cache_fraction": f, "rho": rho, "curve_extrapolated": extrap,
           "engine_hit_measured": hit, "pred_H_rho": h_rho, "pred_H_floor": h_floor,
           "err_H_rho": hit - h_rho, "err_H_floor": hit - h_floor,
           "note": "engine hit is cumulative over prefill + 256 tokens on one prompt (BigMoeOnEdge "
                   "moe-cache line); the curve is wikitext, steady state. A coarse check, not S9."}
    print(f"engine_vs_curve: rho {rho:.2f} measured {hit:.3f} H_rho {h_rho:.3f} H_floor {h_floor:.3f}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--tokens-dir", default=os.path.join(REPO, "..", "moe-work"))
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--only", default=None, help="comma list of sections to run (debug)")
    p.add_argument("--gguf", default=os.path.join(REPO, "..", "moe-work", "models", "Qwen3-30B-A3B-Q4_0.gguf"))
    p.add_argument("--gguf-py", default=os.path.join(REPO, "..", "moe-work", "llama.cpp", "gguf-py"))
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()
    secs = {"phone_csv": sec_phone_csv, "engine_vs_curve": sec_engine_vs_curve,
            "union": lambda: sec_union(a.max_tokens), "miss_dist": lambda: sec_miss_dist(a.max_tokens),
            # lexical fits a table on half the corpus, so it uses the WHOLE trace (cheap, O(T)).
            "lexical": lambda: sec_lexical(os.path.abspath(a.tokens_dir), None),
            "prefetch": lambda: sec_prefetch(a.max_tokens), "lookahead": lambda: sec_lookahead(a.max_tokens),
            "q4_entropy": lambda: sec_q4_entropy(os.path.abspath(a.gguf), os.path.abspath(a.gguf_py))}
    want = a.only.split(",") if a.only else list(secs)
    path = os.path.join(a.out_dir, "headroom_sims.json") if a.out_dir else write_path("headroom_sims.json")
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
    # --only re-runs named sections INTO an existing artifact (merge), so a section can be
    # regenerated without silently dropping the others; a full run always starts fresh.
    out = json.load(open(path, encoding="utf-8")) if (a.only and os.path.exists(path)) else {}
    out.update({"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "inputs": {"max_tokens": a.max_tokens, "bulk_gbps": BULK_GBPS, "expert_mb": EXPERT_MB,
                      "windows": WINDOWS, "horizons": HORIZONS, "accuracies": ACCS,
                      "prefetch_fracs": PREFETCH_FRACS, "prefetch_fwd_ms": PREFETCH_FWD_MS,
                      "steady_from_token": STEADY_FROM + 1, "lexical_uses_full_trace": True}})
    for name in want:
        out[name] = secs[name]()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
