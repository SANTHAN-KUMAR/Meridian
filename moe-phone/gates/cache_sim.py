"""
G2 (offline half) — expert-cache hit rates from routing traces.

Input: the traces_<model>.npz written by traces_sparsity.py — one int16 array
per MoE layer, shape [tokens, top_k], of the experts the model's own router
selected.

TWO AXES, both of which changed a reported number (see the retraction in
POSITION.md F4). Neither is a tuning knob; each names a different machine.

  scope     "global"     one cache shared by all layers.
            "per_layer"  each layer gets its own quota out of the same TOTAL
                         slot budget (split_capacity below). This is what every
                         published expert-streaming engine does, because layer
                         l's experts are never candidates for layer l+1.
            The first release of this file simulated ONLY "global" while
            gates/expert_policy.py's pinning policy was per-layer, so the
            LRU-vs-pinning comparison it printed was between two different
            cache architectures. That comparison is retracted.

  replay    "atomic"     the engine fetches a layer's top-k misses as ONE
                         event, so residency is decided for all k before any
                         eviction. This is what a real engine does.
            "sequential" evict after every single expert access. This lets a
                         miss early in an event evict an expert the SAME event
                         still needs, which is then counted as a miss. It is a
                         simulator artifact, and it inflates LRU's miss count.

Reports, per cache size:

  - LRU hit rate            (what a simple engine gets)
  - Belady-optimal hit rate (the offline optimum: an upper bound for ANY
                             eviction policy, used as the oracle, CLAUDE.md §4.3)
  - no-locality floor       (= cache fraction; what uniform independent routing
                             gives any policy — the value G-ROOF assumed)
  - misses per token        (x expert bytes = flash bytes per token, feeding G-ROOF)

The geometry that decides whether any of this can work is

      rho = (cache slots per layer) / top_k

with a critical threshold at rho = 1, i.e. cache_fraction = k/E: below it a
per-layer cache cannot hold even one token's working set, so LRU degenerates to
the cyclic-scan worst case regardless of how much locality the trace has. Pass
--fractions-around-crit to sweep across it. OLMoE (k/E = 0.125) and
Qwen3-Next-80B-A3B (10/512 = 0.0195) sit on opposite sides at any realistic
cache size, so an OLMoE trace is NOT a conservative proxy for the target model.

Run:
  python moe-phone/gates/cache_sim.py traces_OLMoE-1B-7B-0924.npz --both-scopes
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


def event_stream(traces, num_experts, max_tokens=None):
    """Same keys as request_stream, but shaped [T, L, k] so the (token, layer)
    EVENT survives as a unit. One event is one layer's top-k fetch."""
    layers = sorted(traces)
    T = min(traces[l].shape[0] for l in layers)
    if max_tokens:
        T = min(T, max_tokens)
    ev = np.stack([traces[l][:T].astype(np.int64) + r * num_experts
                   for r, l in enumerate(layers)], axis=1)  # [T, L, k]
    return ev, T, len(layers), ev.shape[2]


def split_capacity(cap_total, n_layers):
    """Split a TOTAL slot budget across layers, preserving the total exactly.

    base = cap//L each; the first cap%L layers get one more. Even splitting,
    not proportional-to-traffic: every layer runs on every token, so a layer
    starved of cache stalls the whole token. A proportional split is a tuning
    knob and would need its own justification (CLAUDE.md §6.4).

    The total is preserved so that "10% cache" means the same BYTES under both
    scopes and the two are comparable.
    """
    base, rem = divmod(cap_total, n_layers)
    return [base + (1 if i < rem else 0) for i in range(n_layers)]


def lru_hits_events(ev, cap_total, scope="per_layer", replay="atomic"):
    """LRU over the [T, L, k] event stream. See the module docstring for scope
    and replay; both change the answer and neither has a default that is right
    for every question."""
    T, L, k = ev.shape
    if scope == "per_layer":
        caps = split_capacity(cap_total, L)
        caches = [OrderedDict() for _ in range(L)]
    elif scope == "global":
        caps = [cap_total] * L
        shared = OrderedDict()
        caches = [shared] * L
    else:
        raise ValueError(f"scope must be 'global' or 'per_layer', got {scope!r}")
    if replay not in ("atomic", "sequential"):
        raise ValueError(f"replay must be 'atomic' or 'sequential', got {replay!r}")

    hits = 0
    rows = ev.reshape(T * L, k).tolist()
    layer_of = list(range(L)) * T
    for keys, l in zip(rows, layer_of):
        c, cp = caches[l], caps[l]
        if cp <= 0:
            continue  # a layer with no slots misses everything
        if replay == "atomic":
            # Residency is decided against the cache state at EVENT START, and
            # every hit is promoted before any miss is inserted -- so a miss can
            # never evict an expert this same event still needs.
            miss = []
            for key in keys:
                if key in c:
                    hits += 1
                    c.move_to_end(key)
                else:
                    miss.append(key)
            for key in miss:
                while len(c) >= cp:
                    c.popitem(last=False)
                c[key] = None
        else:
            for key in keys:
                if key in c:
                    hits += 1
                    c.move_to_end(key)
                else:
                    while len(c) >= cp:
                        c.popitem(last=False)
                    c[key] = None
    return hits


def cycle_hits_global(ev, cap_total, num_experts, policy="cycle", eval_from=0):
    """Eviction policies for a GLOBAL cache that exploit the one thing about this access pattern that
    needs no prediction: the LAYER ORDER IS FIXED.

    A decode token walks layers 0..L-1 in order, every token, forever. So standing at layer l, an entry
    belonging to layer j will not be touched again for

        d(j) = (j - l) mod L      layers,    with d(l) = 0 meaning "just used, next use is L layers away"

    and that quantity is known exactly, not predicted. Global LRU ignores it, and in doing so picks
    close to the WORST victim available: the least recently used entry is the one from the layer touched
    longest ago, which in a cyclic scan is the layer about to be touched NEXT. This is the classic
    LRU-on-a-cyclic-scan pathology, and our engine's cache is exactly that shape.

      policy="cycle"      evict from the layer whose next visit is furthest away -- i.e. the layer just
                          completed -- breaking ties by least-recently-used inside that layer. Uses no
                          information the engine does not already have at eviction time.
      policy="cyclefreq"  the same, but scored by EXPECTED next use rather than layer distance alone:
                          an expert is only re-read when its layer both comes round AND selects it, so
                              score = d(j) + L / p_hat(e)
                          with p_hat estimated online from the stream seen so far (Laplace-smoothed).
                          An unpopular expert one layer ahead is then correctly ranked as a worse keep
                          than a popular expert in the layer just finished.

    Both are online and deployable: no future routing, no draft model, no second pass. Belady remains
    the bound; these are attempts to get closer to it than recency does.
    """
    T, L, k = ev.shape
    # "freq" drops the cyclic term entirely (score = L/p_hat), which is the control that says how much
    # of cyclefreq's gain is the cycle and how much is just popularity.
    if policy not in ("cycle", "cyclefreq", "freq"):
        raise ValueError(f"policy must be 'cycle', 'cyclefreq' or 'freq', got {policy!r}")
    buckets = [OrderedDict() for _ in range(L)]   # layer -> its resident keys, LRU order
    n_res = 0
    freq = {}
    seen_events = 0
    hits = seen = 0
    for t in range(T):
        for l in range(L):
            counting = t >= eval_from
            keys = ev[t, l, :].tolist()
            b = buckets[l]
            miss = []
            for key in keys:
                if counting:
                    seen += 1
                freq[key] = freq.get(key, 0) + 1
                if key in b:
                    if counting:
                        hits += 1
                    b.move_to_end(key)
                else:
                    miss.append(key)
            seen_events += 1
            for key in miss:
                while n_res >= cap_total:
                    # choose the victim layer: furthest next visit, i.e. largest (j - l) mod L
                    victim_layer, victim_key, best = -1, None, None
                    for j in range(L):
                        if not buckets[j]:
                            continue
                        d = (j - l) % L
                        if policy == "cycle":
                            score = (d, )
                            cand = next(iter(buckets[j]))          # LRU within the layer
                        else:
                            # least likely to come back soonest inside this layer
                            cand = min(buckets[j], key=lambda e: freq.get(e, 0))
                            p = (freq.get(cand, 0) + 1.0) / (seen_events + 2.0)
                            score = ((d if policy == "cyclefreq" else 0.0) + L / p, )
                        if best is None or score > best:
                            best, victim_layer, victim_key = score, j, cand
                    if victim_key is None:
                        break
                    del buckets[victim_layer][victim_key]
                    n_res -= 1
                if n_res < cap_total:
                    b[key] = None
                    n_res += 1
    return hits, seen


def slru_hits_global(ev, cap_total, protected_frac=0.8, eval_from=0):
    """Segmented LRU on the GLOBAL cache: the O(1) way to be frequency-aware.

    Exact frequency scoring (cycle_hits_global's "freq") needs a scan over candidates at every
    eviction, which would land on the very cache-management term this project is trying to shrink
    (20 ms/token, gates/decode_budget.py). SLRU gets most of the same signal for the same cost as LRU:

      probation  entries seen once. Evictions come from here, LRU order.
      protected  entries promoted on their SECOND hit. When it overflows, its LRU tail demotes to
                 probation rather than being dropped.

    So a one-off expert cannot push out one that keeps coming back, and every operation is O(1) --
    two linked lists and a flag per entry, which is what the engine already maintains for its LRU.

    `protected_frac` is a structural parameter, not a fitted one; Karedla et al. (1994) use 60-80% and
    the caller is expected to report a sweep rather than a chosen value.
    """
    T, L, k = ev.shape
    n_prot_max = int(round(protected_frac * cap_total))
    prot = OrderedDict()
    prob = OrderedDict()
    hits = seen = 0
    for t in range(T):
        counting = t >= eval_from
        for l in range(L):
            keys = ev[t, l, :].tolist()
            miss = []
            for key in keys:
                if counting:
                    seen += 1
                if key in prot:
                    if counting:
                        hits += 1
                    prot.move_to_end(key)
                elif key in prob:
                    if counting:
                        hits += 1
                    del prob[key]                       # second sighting: promote
                    prot[key] = None
                    while len(prot) > n_prot_max:       # demote the protected tail
                        dk, _ = prot.popitem(last=False)
                        prob[dk] = None
                else:
                    miss.append(key)
            for key in miss:
                while len(prot) + len(prob) >= cap_total:
                    if prob:
                        prob.popitem(last=False)
                    else:
                        prot.popitem(last=False)
                prob[key] = None
    return hits, seen


def belady_hits_scoped(ev, cap_total, scope="per_layer"):
    """Belady's MIN under the same scope. Per-layer, each layer's subsequence is
    solved independently, which is exactly right: with private quotas the layers
    are independent caching problems.

    Belady is run sequentially in both cases. Unlike LRU it is insensitive to
    the atomic/sequential distinction to first order, because the victim it
    picks is the farthest-next-use and an expert needed later in the SAME event
    has a next use only a few accesses away, so it is never the victim while any
    alternative exists.
    """
    T, L, k = ev.shape
    if scope == "global":
        return belady_hits(ev.reshape(-1), cap_total)
    caps = split_capacity(cap_total, L)
    return sum(belady_hits(ev[:, l, :].reshape(-1), caps[l])
               for l in range(L) if caps[l] > 0)


def shuffle_control(traces, seed=0):
    """Permute the TOKEN order within each layer, independently.

    Preserves each layer's expert-popularity distribution exactly and destroys
    temporal order. The gap between the real trace and this control is the part
    of the hit rate attributable to RECENCY; the gap between this control and
    the uniform control is the part attributable to POPULARITY SKEW.
    """
    rng = np.random.default_rng(seed)
    return {l: t[rng.permutation(t.shape[0])] for l, t in traces.items()}


def uniform_control(traces, num_experts, seed=0):
    """iid uniform top-k per token: no skew and no order. The analytic null --
    LRU on this must equal the cache fraction, which is the recovery gate in
    tests/test_gates.py::test_lru_on_locality_free_routing_equals_the_floor."""
    rng = np.random.default_rng(seed)
    out = {}
    for l, t in traces.items():
        T, k = t.shape
        out[l] = np.stack([rng.choice(num_experts, size=k, replace=False)
                           for _ in range(T)]).astype(t.dtype)
    return out


def window_fetches(ev, cap_total, window, scope="per_layer"):
    """Multi-token verification over the SAME LRU cache.

    A speculative engine runs `window` tokens through one forward pass, so for
    each layer it needs the UNION of those tokens' top-k sets and fetches each
    distinct expert once. This is exactly llama.cpp PR #25294's wave-partitional
    prefill, pointed at decode.

    Returns (fetches, windows): `fetches` counts expert LOADS from flash, which
    is the quantity that costs time. Bytes/token = fetches * expert_bytes /
    tokens_emitted, and tokens_emitted depends on the acceptance rate, so it is
    applied by the caller.

    This composes with the cache rather than multiplying against it: the union
    and the cache exploit the SAME expert reuse, so the saving from widening the
    window is measured here on top of whatever the cache already supplies, not
    assumed to be independent of it.
    """
    T, L, k = ev.shape
    if scope == "per_layer":
        caps = split_capacity(cap_total, L)
        caches = [OrderedDict() for _ in range(L)]
    else:
        caps = [cap_total] * L
        shared = OrderedDict()
        caches = [shared] * L
    fetches = 0
    n_win = 0
    for t0 in range(0, T - window + 1, window):
        n_win += 1
        for l in range(L):
            c, cp = caches[l], caps[l]
            if cp <= 0:
                continue
            need = set(ev[t0:t0 + window, l, :].reshape(-1).tolist())
            miss = [key for key in need if key not in c]
            for key in need:
                if key in c:
                    c.move_to_end(key)
            for key in miss:
                while len(c) >= cp:
                    c.popitem(last=False)
                c[key] = None
            fetches += len(miss)
    return fetches, n_win


def policy_hits(ev, cap_total, policy="lru", scope="per_layer",
                warm=None, pin_frac=0.0, eval_from=0):
    """Deployable online policies, all event-atomic and per-layer by default.

      lru   recency. What a naive engine does.
      lfu   frequency, counted online from the replayed stream only.
      warm  LRU whose cache is PRE-FILLED at load time with the most popular
            experts (from `warm`, a dict layer -> ranked expert ids fitted on a
            disjoint warmup slice). This is cache warming as a serving stack
            does it, and it costs one sequential read at startup.
      pin   `pin_frac` of each layer's slots are RESERVED for the warm set and
            never evicted; the remainder runs LRU. pin_frac=0 is warm-start LRU,
            pin_frac=1 is pure static pinning.

    `eval_from` skips the first N tokens when counting, so a policy fitted on a
    warmup slice is never scored on it.
    """
    T, L, k = ev.shape
    caps = split_capacity(cap_total, L) if scope == "per_layer" else [cap_total] * L
    hits = seen = 0
    for l in range(L):
        cp = caps[l]
        if cp <= 0:
            seen += (T - eval_from) * k
            continue
        n_pin = min(cp, int(round(pin_frac * cp)))
        pinned = set(warm[l][:n_pin]) if (warm is not None and n_pin) else set()
        cache = OrderedDict()
        if warm is not None:                      # pre-fill the free slots too
            for e in warm[l][:cp]:
                if e not in pinned:
                    cache[e] = None
        freq = {}
        col = ev[:, l, :]
        for t in range(T):
            keys = col[t].tolist()
            counting = t >= eval_from
            miss = []
            for key in keys:
                if counting:
                    seen += 1
                if key in pinned:
                    if counting:
                        hits += 1
                elif key in cache:
                    if counting:
                        hits += 1
                    cache.move_to_end(key)
                else:
                    miss.append(key)
                freq[key] = freq.get(key, 0) + 1
            free = cp - len(pinned)
            for key in miss:
                # free == 0 means every slot is pinned: nothing to evict into.
                while free > 0 and len(cache) >= free:
                    if policy == "lfu":
                        victim = min(cache, key=lambda e: (freq.get(e, 0),))
                        del cache[victim]
                    else:
                        cache.popitem(last=False)
                if free > 0:
                    cache[key] = None
    return hits, seen


def warm_sets(traces, num_experts, warmup_frac):
    """Expert ids per layer, ranked by frequency over the FIRST warmup_frac of
    tokens only, offset into the global key space. Fitted on a slice that the
    evaluation then skips, so nothing is scored on the data it was chosen on."""
    layers = sorted(traces)
    T = min(traces[l].shape[0] for l in layers)
    split = max(1, int(T * warmup_frac))
    out = {}
    for r, l in enumerate(layers):
        c = np.bincount(traces[l][:split].ravel().astype(np.int64),
                        minlength=num_experts)
        out[r] = (np.argsort(c)[::-1] + r * num_experts).tolist()
    return out, split


def corrupt_routing(ev, accuracy, num_experts, seed=0):
    """A PREDICTED view of the routing: each expert is right with probability
    `accuracy`, else replaced by a uniformly random expert of the same layer.

    accuracy=1.0 is a perfect oracle (or, equivalently, an accepted speculative
    draft, whose routing is exact by construction). accuracy=0.0 is no
    information at all. Layer offsets are preserved so a corrupted id never
    escapes its own layer's expert range.
    """
    rng = np.random.default_rng(seed)
    T, L, k = ev.shape
    out = ev.copy()
    flip = rng.random((T, L, k)) >= accuracy
    for l in range(L):
        m = flip[:, l, :]
        if m.any():
            col = out[:, l, :]
            col[m] = l * num_experts + rng.integers(0, num_experts, size=int(m.sum()))
            out[:, l, :] = col
    return out


def lookahead_hits(ev, cap_total, horizon, scope="per_layer", ev_hat=None,
                   mode="rank"):
    """LRU eviction improved by a BOUNDED lookahead of `horizon` tokens.

    horizon=None is exactly Belady, which tests/test_gates.py asserts as an
    identity. horizon=0 is NOT "no information": the engine is executing the
    current token, so it always knows that token's own expert set, and H=0 is
    therefore event-atomic (measured within ~1% of atomic LRU, far above
    sequential LRU). The tests assert the bracketing -- sequential LRU <= any
    horizon <= Belady -- and monotonicity in the horizon, rather than an
    identity that does not hold.

    WHERE THE LOOKAHEAD COMES FROM, and what happens when it is wrong:

      Multi-token verification supplies it EXACTLY, and it is worth being
      precise about why, because the obvious explanation is wrong. It is not
      that the draft model routed those tokens: a self-draft restricted to
      cache-resident experts is a different model, so its routing is a
      prediction of the target's. The exactness comes from the shape of the
      verification pass. Verifying W tokens is ONE batched forward pass, so
      layer l routes all W positions in a single matmul -- and that happens
      before layer l's expert weights are touched. The engine therefore reads
      the true top-k sets for W tokens off the router it has just run. The
      horizon is exactly the window and not one token more: nothing is yet
      known about the next window's layer-l needs.

      `ev_hat` models the cheaper alternative -- a routing PREDICTOR -- by
      letting eviction see a corrupted view of the future while hits are still
      counted against the true one.

      A wrong eviction spends no bandwidth on bytes nobody wanted, which is the
      direct cost that sinks a wrong PREFETCH (Budgeting Bytes 2609.04238; WiSP
      2606.21868; llama.cpp discussion #24528). It does NOT follow that a bad
      predictor is harmless here, and an earlier draft of this docstring claimed
      it did. Measured: under mode="rank" a predictor at accuracy 0 scores 0.095
      against LRU's 0.273 at a 10% cache, because ranking by a wrong prediction
      discards recency, which is itself real information. Break-even is accuracy
      ~0.70.

      `mode` is the fix, and which one to use depends on where the lookahead
      came from:

        "rank"     evict the farthest predicted next use. Optimal with an EXACT
                   lookahead -- which is what a batched verification pass
                   supplies over its own window -- and it reaches Belady at a
                   horizon of ~4 tokens.
        "protect"  the predictor may only VETO a candidate, never propose one;
                   recency still does the ranking. Strictly worse at accuracy
                   1.0 (0.379 vs 0.439) and far more robust: it still beats LRU
                   at accuracy 0.3. Use it when the lookahead is a PREDICTION.

    The engine always knows the CURRENT token's routing exactly, because it is
    executing that layer, so next-use distances inside the current token are
    taken from `ev` and only distances beyond it come from `ev_hat`.
    """
    T, L, k = ev.shape
    if ev_hat is None:
        ev_hat = ev
    INF = float("inf")
    hits = 0
    if scope == "global":
        # ONE pool over the token-major stream. The horizon is then counted in
        # EVENTS (layer-steps), not tokens, because that is the unit a global
        # cache advances by: a within-token, cross-layer predictor supplies a
        # horizon of a few events without any drafter, which a per-layer cache
        # cannot use at all.
        streams = [(ev.reshape(-1).tolist(), ev_hat.reshape(-1).tolist(), cap_total)]
    elif scope == "per_layer":
        caps = split_capacity(cap_total, L)
        streams = [(ev[:, l, :].reshape(-1).tolist(),
                    ev_hat[:, l, :].reshape(-1).tolist(), caps[l])
                   for l in range(L)]
    else:
        raise ValueError(f"scope must be 'global' or 'per_layer', got {scope!r}")

    for seq, hat, cp in streams:
        if cp <= 0:
            continue
        n = len(seq)
        # next occurrence of the TRUE key seq[i], strictly after i, in each of
        # the true and the predicted stream
        nxt_t, nxt_h = [0] * n, [0] * n
        last_t, last_h = {}, {}
        for i in range(n - 1, -1, -1):
            nxt_t[i] = last_t.get(seq[i], n + i)   # distinct far-future sentinel
            nxt_h[i] = last_h.get(seq[i], n + i)
            last_t[seq[i]] = i
            last_h[hat[i]] = i
        # per_layer: one event per token, so horizon*k accesses == horizon tokens.
        # global: k accesses per event, so horizon*k accesses == horizon EVENTS.
        span = n if horizon is None else horizon * k
        cache = OrderedDict()          # key -> (next true use, next predicted use)
        for i, key in enumerate(seq):
            tok_end = (i // k + 1) * k
            if key in cache:
                hits += 1
                cache.move_to_end(key)
            elif len(cache) >= cp:
                victim, best = None, -1.0
                for e, (vt, vh) in cache.items():   # iterates LRU-most first
                    if vt < tok_end:
                        d = float(vt - i)   # inside the token being executed: known
                    else:
                        d = INF if vh - i > span else float(vh - i)
                    if mode == "protect":
                        if d == INF:
                            victim = e     # LRU-most entry the horizon does not want
                            break
                        if victim is None and vt >= tok_end:
                            victim = e     # fallback: LRU-most that is not needed now
                    elif d > best:         # strictly greater keeps the LRU-most entry
                        victim, best = e, d
                        if d == INF:
                            break          # nothing beats "not needed in the horizon"
                if victim is None:
                    victim = next(iter(cache))
                del cache[victim]
            cache[key] = (nxt_t[i], nxt_h[i])
            cache.move_to_end(key)
    return hits


def simulate(traces, num_experts, fractions, max_tokens=None, belady=True,
             scopes=("per_layer",), replays=("atomic",)):
    ev, T, L, k = event_stream(traces, num_experts, max_tokens)
    total_experts = L * num_experts
    n_req = T * L * k
    out = []
    for f in fractions:
        cap = max(1, int(round(f * total_experts)))
        caps = split_capacity(cap, L)
        row = {"cache_fraction": f, "cache_experts": cap,
               "cap_per_layer_min": min(caps), "cap_per_layer_max": max(caps),
               # rho = per-layer slots / top_k = f * E / k, exact (not the
               # integer-floored min, which aliases adjacent fractions).
               "rho_per_layer": f * num_experts / k,
               "rho_per_layer_min_int": min(caps) / k,
               "floor_hit": cap / total_experts}
        for scope in scopes:
            for replay in replays:
                tag = f"{scope}_{replay}"
                h = lru_hits_events(ev, cap, scope=scope, replay=replay)
                row[f"lru_hit__{tag}"] = h / n_req
                row[f"lru_misses_per_token__{tag}"] = (n_req - h) / T
            if belady:
                b = belady_hits_scoped(ev, cap, scope=scope)
                row[f"belady_hit__{scope}"] = b / n_req
                row[f"belady_misses_per_token__{scope}"] = (n_req - b) / T
        out.append(row)
    return {"tokens": T, "moe_layers": L, "top_k": k, "num_experts": num_experts,
            "requests": int(n_req), "scopes": list(scopes), "replays": list(replays),
            "f_crit": k / num_experts, "rows": out}


def _crit_fractions(k, E, n_layers):
    """Fractions bracketing f_crit = k/E, where a per-layer cache first holds a
    whole token's working set. Pre-registered as multiples of f_crit so the grid
    is fixed by the geometry, not chosen after seeing the curve (CLAUDE.md §6.4)."""
    fc = k / E
    mult = [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0]
    return sorted({round(min(1.0, m * fc), 6) for m in mult})


def main():
    p = argparse.ArgumentParser(description="expert-cache simulation from routing traces")
    p.add_argument("npz")
    p.add_argument("--fractions", default="0.05,0.1,0.2,0.3,0.5")
    p.add_argument("--fractions-around-crit", action="store_true",
                   help="sweep multiples of f_crit = k/E instead (pre-registered grid)")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--no-belady", action="store_true")
    p.add_argument("--scope", default="per_layer", choices=["per_layer", "global"])
    p.add_argument("--both-scopes", action="store_true",
                   help="report per_layer AND global, so the retracted global number stays visible")
    p.add_argument("--replay", default="atomic", choices=["atomic", "sequential"])
    p.add_argument("--both-replays", action="store_true")
    p.add_argument("--lookahead", default=None,
                   help="comma-separated lookahead horizons in TOKENS to sweep, e.g. "
                        "0,1,2,4,8,16. 0 is sequential LRU and an empty horizon is "
                        "Belady, so the curve says how many tokens of future routing "
                        "an engine needs to reach the offline optimum.")
    p.add_argument("--pred-accuracy", default=None,
                   help="comma-separated predictor accuracies to sweep alongside --lookahead, "
                        "e.g. 1.0,0.9,0.8,0.7. Eviction sees the corrupted routing; hits are "
                        "counted against the true one. Reports the break-even accuracy at "
                        "which a predictor stops beating plain LRU.")
    p.add_argument("--controls", action="store_true",
                   help="also simulate the shuffled and uniform controls, decomposing the hit "
                        "rate into floor + popularity skew + recency")
    p.add_argument("--seed", type=int, default=0, help="seed for the controls")
    p.add_argument("--out-name", default=None, help="basename override for the artifact")
    p.add_argument("--out-dir", default=None,
                   help="write here instead of results/<date>/ (use for tests and dry runs)")
    a = p.parse_args()
    z = np.load(a.npz)
    traces = {int(kk[1:]): z[kk] for kk in z.files if kk.startswith("L")}
    E = int(z["num_experts"])
    k = next(iter(traces.values())).shape[1]
    if a.fractions_around_crit:
        fracs = _crit_fractions(k, E, len(traces))
    else:
        fracs = [float(x) for x in a.fractions.split(",")]
    scopes = ("per_layer", "global") if a.both_scopes else (a.scope,)
    replays = ("atomic", "sequential") if a.both_replays else (a.replay,)
    res = simulate(traces, E, fracs, a.max_tokens, belady=not a.no_belady,
                   scopes=scopes, replays=replays)
    res["source"] = os.path.abspath(a.npz)
    if a.controls:
        res["controls"] = {"seed": a.seed}
        for cname, ctrace in (("shuffled", shuffle_control(traces, a.seed)),
                              ("uniform", uniform_control(traces, E, a.seed))):
            res["controls"][cname] = simulate(ctrace, E, fracs, a.max_tokens,
                                              belady=not a.no_belady,
                                              scopes=scopes, replays=replays)["rows"]
    print(f"{res['tokens']} tokens x {res['moe_layers']} MoE layers x "
          f"top-{res['top_k']} of {E} experts    f_crit = k/E = {res['f_crit']:.4f}")
    cols = [(f"lru_hit__{s}_{r}", f"LRU/{s[:3]}/{r[:3]}") for s in scopes for r in replays]
    cols += [(f"belady_hit__{s}", f"Belady/{s[:3]}") for s in scopes if not a.no_belady]
    hdr = f"{'cache':>6s} {'rho':>5s} {'floor':>7s}" + "".join(f" {t:>15s}" for _, t in cols)
    print(hdr)
    for r in res["rows"]:
        line = f"{r['cache_fraction']:6.3f} {r['rho_per_layer']:5.2f} {r['floor_hit']:7.3f}"
        line += "".join(f" {r.get(c, float('nan')):15.3f}" for c, _ in cols)
        print(line)
    if a.controls:
        print()
        print("decomposition of the hit rate (scope=per_layer, replay=atomic):")
        print(f"{'cache':>6} {'floor':>7} {'=uniform':>9} {'+skew':>8} {'+recency':>10} "
              f"{'=real':>8}   {'Belady real':>12} {'Belady unif':>12}")
        for i, r in enumerate(res["rows"]):
            u = res["controls"]["uniform"][i].get("lru_hit__per_layer_atomic")
            sh = res["controls"]["shuffled"][i].get("lru_hit__per_layer_atomic")
            rl = r.get("lru_hit__per_layer_atomic")
            if None in (u, sh, rl):
                continue
            bu = res["controls"]["uniform"][i].get("belady_hit__per_layer")
            br = r.get("belady_hit__per_layer")
            nan = float("nan")
            print(f"{r['cache_fraction']:6.3f} {r['floor_hit']:7.3f} {u:9.3f} {sh - u:+8.3f} "
                  f"{rl - sh:+10.3f} {rl:8.3f}   "
                  f"{(br if br is not None else nan):12.3f} "
                  f"{(bu if bu is not None else nan):12.3f}")
        print("  The cache fraction is the correct null for LRU -- the uniform column")
        print("  reproduces it -- but NOT for Belady: an offline optimum beats the fraction")
        print("  even on locality-free routing, so a Belady-minus-fraction gap is not a")
        print("  measure of exploitable locality and overstates it by the uniform column.")
    if a.lookahead:
        hs = [int(x) for x in a.lookahead.split(",")]
        ev_la, T_la, L_la, k_la = event_stream(traces, E, a.max_tokens)
        n_la = T_la * L_la * k_la
        res["lookahead"] = {"horizons_tokens": hs, "rows": []}
        print()
        print("hit rate vs LOOKAHEAD HORIZON (tokens of future routing already known).")
        print("Verifying W tokens in one batched pass supplies this exactly: layer l")
        print("routes all W positions before it touches layer l's expert weights.")
        hdr = f"{'cache':>6} {'LRU':>7}"
        hdr += "".join(f"{'H=' + str(h):>8}" for h in hs)
        hdr += f"{'Belady':>9}   gap closed by the largest H"
        print(hdr)
        for f in fracs:
            cap = max(1, int(round(f * L_la * E)))
            lru = lru_hits_events(ev_la, cap, "per_layer", "atomic") / n_la
            bel = belady_hits_scoped(ev_la, cap, "per_layer") / n_la
            xs = [lookahead_hits(ev_la, cap, h) / n_la for h in hs]
            closed = ((xs[-1] - lru) / (bel - lru)) if bel > lru else float("nan")
            line = f"{f:6.3f} {lru:7.3f}"
            line += "".join(f"{x:8.3f}" for x in xs)
            line += f"{bel:9.3f}   {100 * closed:5.0f}%"
            print(line)
            res["lookahead"]["rows"].append(
                {"cache_fraction": f, "lru_atomic": lru, "belady": bel,
                 "by_horizon": dict(zip(map(str, hs), xs)),
                 "fraction_of_belady_gap_closed": closed})
        if a.pred_accuracy:
            accs = [float(x) for x in a.pred_accuracy.split(",")]
            H = max(hs) if hs else 4
            res["prediction"] = {"horizon_tokens": H, "accuracies": accs, "rows": []}
            print()
            print(f"the same eviction rule at horizon {H}, driven by an IMPERFECT predictor.")
            print("Eviction sees the corrupted routing; hits are counted on the true trace.")
            print("rank uses the predicted ordering; protect lets the predictor only veto,")
            print("leaving recency to rank. rank wins with an exact lookahead (a draft);")
            print("protect is the one that survives a wrong predictor.")
            hdr = f"{'cache':>6} {'mode':>8} {'LRU':>7}"
            hdr += "".join(f"{format(x, '.2f'):>7}" for x in accs)
            hdr += f"{'Belady':>8}   break-even"
            print(hdr)
            for f in fracs:
                cap = max(1, int(round(f * L_la * E)))
                lru = lru_hits_events(ev_la, cap, "per_layer", "atomic") / n_la
                bel = belady_hits_scoped(ev_la, cap, "per_layer") / n_la
                row = {"cache_fraction": f, "lru_atomic": lru, "belady": bel}
                for mode in ("rank", "protect"):
                    ys = []
                    for acc in accs:
                        hat = None if acc >= 1.0 else corrupt_routing(ev_la, acc, E, a.seed)
                        ys.append(lookahead_hits(ev_la, cap, H, ev_hat=hat,
                                                 mode=mode) / n_la)
                    be = next((acc for acc, y in zip(accs, ys) if y < lru), None)
                    line = f"{f:6.3f} {mode:>8} {lru:7.3f}"
                    line += "".join(f"{y:7.3f}" for y in ys)
                    line += f"{bel:8.3f}   {(format(be, '.2f') if be is not None else 'never'):>9}"
                    print(line)
                    row[mode] = {"by_accuracy": dict(zip(map(str, accs), ys)),
                                 "break_even_accuracy": be}
                res["prediction"]["rows"].append(row)
    name = a.out_name or os.path.splitext(os.path.basename(a.npz))[0].replace("traces_", "cache_")
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
