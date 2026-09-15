"""
ENGINE SIM — the decode loop this project proposes, replayed against a measured
routing trace and measured device numbers.

This is the constructive counterpart to cache_sim.py. cache_sim asks "what hit
rate does policy X get"; this asks "how many seconds does a token take", which
is the number the project is actually about.

THE LOOP

  Draft W tokens with a zero-byte self-draft (the same model restricted to
  cache-resident experts), then verify all W in one forward pass. Per layer that
  pass needs the UNION of the W tokens' top-k sets, so each distinct expert is
  fetched once, not W times. That is llama.cpp PR #25294's wave-partitional
  prefill, pointed at decode instead.

  The draft pass has already evaluated the router for those W tokens. So at
  verification time the engine KNOWS which experts the next W tokens need. It
  does not predict them: there is no accuracy term, no trained predictor, and
  nothing to calibrate. Eviction can therefore use Belady's farthest-next-use
  rule over that known window.

THE POINT, AND WHY IT IS ONE MECHANISM AND NOT TWO

  Union fetching and lookahead eviction both exploit the SAME expert reuse, so
  their gains are not independent and must not be multiplied. They are measured
  here TOGETHER, in one replay, against the same cache. What the draft buys is a
  window of known-future routing; union fetching and Belady eviction are two
  ways of spending it, and spending it twice does not pay twice.

WHAT IS ASSUMED, NOT MEASURED

  - acceptance rate alpha. A draft token that is rejected still cost its share
    of the fetches, so bytes are charged per VERIFIED token and credited per
    ACCEPTED token. E[accepted per pass] = (1 - alpha^W) / (1 - alpha) for a
    chain draft. alpha is a property of the drafter and is NOT measured here;
    the sweep reports the whole alpha range so the reader can place their own.
  - that the draft's routing matches the target's routing for accepted tokens.
    True by construction for accepted tokens (verification re-runs the real
    router); for rejected ones the engine fetched experts it did not need,
    which is already charged.
  - compute is free -- in the DEFAULT output only. --fwd-ms charges it
    (2026-09-16): a self-draft runs W-1 extra forward passes per window, each
    flash-free but not free (DRAM + compute, c seconds), and the verify pass
    costs c * (1 + beta * (W - 1)): beta = 0 if the batched pass is purely
    weight-read-bound (extra positions ride along), 1 if it is compute-bound
    (W positions cost W passes). Plain decode (W = 1) pays c too, so the
    baseline is charged the same way. c and beta are SWEPT, not measured,
    until an on-device forward-pass time exists (S11).

Run:
  python moe-phone/gates/engine_sim.py results/<date>/traces_OLMoE-1B-7B-0924.npz \\
      --bulk-gbps 2.806 --expert-mb 3.54
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cache_sim  # noqa: E402

WINDOWS = [1, 2, 3, 4, 6, 8, 12, 16]
ALPHAS = [0.6, 0.7, 0.8, 0.9, 0.95]


def accepted_per_pass(window, alpha):
    """Expected tokens emitted by verifying a chain draft of `window` positions.

    Leviathan et al. 2023 (Fast Inference from Transformers via Speculative
    Decoding), eq. 1:  E[#tokens] = (1 - alpha^(gamma+1)) / (1 - alpha), with
    gamma draft tokens plus the one the target always produces. Here `window`
    counts the positions VERIFIED in the pass, i.e. gamma+1, so the exponent is
    `window`. At alpha -> 1 this tends to `window`, and at window = 1 it is 1
    for every alpha, which is the no-speculation case.
    """
    if window <= 1:
        return 1.0
    if alpha >= 1.0:
        return float(window)
    return (1.0 - alpha ** window) / (1.0 - alpha)


def engine_fetches(ev, cap_total, window, lookahead_beyond=0):
    """Replay the loop above. Returns (fetches, windows, verified_tokens).

    Per window and per layer:
      1. need = union of the window's top-k sets at that layer  (ONE fetch each)
      2. hits are decided against the cache state at window start
      3. misses are fetched; eviction prefers the expert whose next use inside
         the KNOWN horizon is farthest, and anything not needed in the horizon
         at all is evicted first, ties broken by LRU.

    The horizon is the rest of the current window plus `lookahead_beyond`
    further tokens. lookahead_beyond=0 is the honest default: at a window
    boundary the engine has not drafted past it yet, so it knows nothing beyond.
    """
    T, L, k = ev.shape
    caps = cache_sim.split_capacity(cap_total, L)
    caches = [OrderedDict() for _ in range(L)]   # local: no state leaks between runs
    INF = float("inf")
    fetches = 0
    n_win = 0
    verified = 0
    for t0 in range(0, T - window + 1, window):
        n_win += 1
        verified += window
        hz_end = min(T, t0 + window + lookahead_beyond)
        for l in range(L):
            cp = caps[l]
            if cp <= 0:
                continue
            cache = caches[l]
            need = []
            seen = set()
            for e in ev[t0:t0 + window, l, :].reshape(-1).tolist():
                if e not in seen:
                    seen.add(e)
                    need.append(e)
            # next-use distance, in tokens, for everything visible in the horizon
            nxt = {}
            for dt in range(t0, hz_end):
                for e in ev[dt, l, :].tolist():
                    if e not in nxt:
                        nxt[e] = dt
            miss = []
            for e in need:
                if e in cache:
                    cache.move_to_end(e)
                else:
                    miss.append(e)
            for e in miss:
                while len(cache) >= cp:
                    victim, best = None, -1.0
                    for c in cache:
                        d = INF if c not in nxt else float(nxt[c] - t0)
                        if d > best:
                            victim, best = c, d
                            if d == INF:
                                break
                    del cache[victim]
                cache[e] = None
            fetches += len(miss)
    return fetches, n_win, verified


def tok_s_with_compute(bytes_per_window, window, alpha, bulk_bps, c, beta):
    """Accepted tokens per second when flash I/O and compute are serial."""
    t = bytes_per_window / bulk_bps + (window - 1) * c + c * (1 + beta * (window - 1))
    return accepted_per_pass(window, alpha) / t


def run(ev, num_experts, fractions, windows, expert_bytes, bulk_bps,
        lookahead_beyond=0, fwd_s=(), betas=(0.0,)):
    T, L, k = ev.shape
    rows = []
    for f in fractions:
        cap = max(1, int(round(f * L * num_experts)))
        per_f = {"cache_fraction": f, "cache_experts": cap,
                 "rho": f * num_experts / k, "windows": {}}
        for w in windows:
            fetch, n_win, verified = engine_fetches(ev, cap, w, lookahead_beyond)
            bytes_per_window = fetch * expert_bytes
            per_f["windows"][str(w)] = {
                "fetches_per_verified_token": fetch / verified,
                "hit_rate": 1.0 - fetch / (verified * L * k),
                "bytes_per_window": bytes_per_window / n_win,
                "tok_s_by_alpha": {
                    str(a): bulk_bps * accepted_per_pass(w, a) * n_win / bytes_per_window
                    for a in ALPHAS},
            }
            if fwd_s:
                per_f["windows"][str(w)]["tok_s_with_compute"] = {
                    f"c={c * 1e3:g}ms,beta={b:g}": {
                        str(a): tok_s_with_compute(bytes_per_window / n_win, w, a, bulk_bps, c, b)
                        for a in ALPHAS}
                    for c in fwd_s for b in betas}
        rows.append(per_f)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("npz")
    p.add_argument("--bulk-gbps", type=float, required=True,
                   help="MEASURED bulk read bandwidth, GB/s (G1)")
    p.add_argument("--expert-mb", type=float, required=True,
                   help="bytes per expert at the target quantisation, MB")
    p.add_argument("--fractions", default="0.05,0.10,0.20,0.30")
    p.add_argument("--windows", default=",".join(str(w) for w in WINDOWS))
    p.add_argument("--lookahead-beyond", type=int, default=0,
                   help="extra tokens of known routing past the window end. 0 is honest "
                        "for a chain draft; >0 models a drafter kept one window ahead.")
    p.add_argument("--fwd-ms", default=None,
                   help="comma list of forward-pass compute+DRAM times in ms to charge "
                        "(draft passes and verify). Omit for the original flash-only pricing.")
    p.add_argument("--verify-beta", default="0,0.25,1",
                   help="marginal verify cost per extra position, as a fraction of one pass")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()

    z = np.load(a.npz)
    traces = {int(kk[1:]): z[kk] for kk in z.files if kk.startswith("L")}
    E = int(z["num_experts"])
    ev, T, L, k = cache_sim.event_stream(traces, E, a.max_tokens)
    fr = [float(x) for x in a.fractions.split(",")]
    wins = [int(x) for x in a.windows.split(",")]
    eb = a.expert_mb * 1e6
    bps = a.bulk_gbps * 1e9

    fwd = [float(x) / 1e3 for x in a.fwd_ms.split(",")] if a.fwd_ms else []
    betas = [float(x) for x in a.verify_beta.split(",")]
    rows = run(ev, E, fr, wins, eb, bps, a.lookahead_beyond, fwd, betas)
    print(f"{T} tokens x {L} layers x top-{k} of {E}   "
          f"{a.expert_mb} MB/expert   {a.bulk_gbps} GB/s bulk")
    print("W=1 is plain autoregressive decode with the same cache: the baseline.\n")
    for r in rows:
        print(f"cache {r['cache_fraction']:.0%}  (rho {r['rho']:.2f}, "
              f"{r['cache_experts']} experts)")
        print(f"  {'W':>3} {'fetch/tok':>10} {'hit':>7}" +
              "".join(f"{'a=' + format(x, '.2f'):>9}" for x in ALPHAS) + "   tok/s")
        base = r["windows"]["1"]["tok_s_by_alpha"][str(ALPHAS[0])]
        for w in wins:
            d = r["windows"][str(w)]
            print(f"  {w:>3} {d['fetches_per_verified_token']:10.2f} {d['hit_rate']:7.3f}" +
                  "".join(f"{d['tok_s_by_alpha'][str(x)]:9.1f}" for x in ALPHAS))
        best = max((d["tok_s_by_alpha"][str(x)], w, x)
                   for w, d in r["windows"].items() for x in ALPHAS)
        print(f"  baseline W=1: {base:.1f} tok/s   "
              f"best in sweep: {best[0]:.1f} tok/s at W={best[1]}, alpha={best[2]}")
        for key in (r["windows"]["1"].get("tok_s_with_compute") or {}):
            b1 = r["windows"]["1"]["tok_s_with_compute"][key]["0.9"]
            cells = "  ".join(f"W={w}:{r['windows'][str(w)]['tok_s_with_compute'][key]['0.9'] / b1:.2f}x"
                              for w in wins if w in (2, 4, 8, 16))
            print(f"  {key:<20} alpha=0.9 speed-up over W=1 with compute charged: {cells}")
        print()

    out = {"source": os.path.abspath(a.npz), "inputs": vars(a),
           "tokens": T, "layers": L, "top_k": k, "num_experts": E,
           "alphas": ALPHAS, "rows": rows}
    name = "engine_sim_" + os.path.basename(a.npz).replace("traces_", "").replace(".npz", "")
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        path = os.path.join(a.out_dir, name + ".json")
    else:
        from _paths import write_path
        path = write_path(name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
