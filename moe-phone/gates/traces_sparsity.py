"""
G2 + G3 — routing traces, training-free lookahead, and gate-first sparsity
fidelity for STOCK MoE checkpoints (no weights changed, nothing trained).

Runs on a GPU: Kaggle (T4 x2) for models up to ~16B; a rented H100 (80 GB)
for 30B-class 16-bit references. `--selftest` runs on CPU with tiny random
models and must pass before any paid GPU time is spent.

What it measures, per model:
  G2  traces   — the experts the model's OWN router selects, per MoE layer and
                 token (saved to traces_<model>.npz for gates/cache_sim.py).
      lookahead — training-free next-layer expert prediction (Fate, arXiv
                 2502.12224): apply layer l+1's router to layer l's router
                 input; report recall of the true top-k.
  G3  sparsity — gate-first sparsity: for each routed expert, compute the gate
                 activation a = act(gate(x)) exactly, zero neurons with
                 |a| < t_layer, and skip their up/down rows:
                     out = down( (a * [|a| >= t]) * up(x) )
                 This is the read pattern a phone engine can execute without a
                 predictor (read gate rows, then only the needed up/down rows).
                 t_layer is calibrated per layer to a target density on a
                 CALIBRATION corpus (wikitext-2 train) and evaluated on a
                 DISJOINT corpus (wikitext-2 test) — never on a task
                 (CLAUDE.md §6.4).
      fidelity — per token, under teacher forcing, vs the unmodified model:
                 KL(p_ref || p_sparse) mean and p99, top-1 flip rate, and the
                 change in top-1 calibration (ECE, 15 equal-width bins as in
                 Guo et al., ICML 2017) — "does it still know when it is wrong".
      floor   — the Tier-A margin (ESTIMAND.md §3): the KL that Q4_0, the
                 published 4-bit format, adds relative to the 16-bit model.
                 Q4_0 is transcribed from ggml's reference quantizer (see
                 q4_0_fake_quant). Routers, embeddings and lm_head are left
                 unquantized; that choice is recorded in the artifact.

Supported layouts: MoE blocks exposing `.gate` (router) and `.experts` as a
ModuleList of MLPs with (gate_proj, up_proj, down_proj) or (w1, w3, w2) and
`.act_fn` — Qwen2/3-MoE, OLMoE, Mixtral, DeepSeek-V2 (remote code). Fused-
expert implementations are refused, not approximated. Pinned:
transformers==4.56.2 (the version the self-test was run with).

Run:
  python traces_sparsity.py --selftest
  python traces_sparsity.py --model allenai/OLMoE-1B-7B-0924 --windows 64 --seq-len 512
  python traces_sparsity.py --model Qwen/Qwen3-30B-A3B --windows 64 --seq-len 512
"""
import argparse
import datetime
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

DENSITIES = [1.0, 0.5, 0.3, 0.2, 0.1]
ECE_BINS = 15


# --------------------------------------------------------------------------
# Q4_0 — transcribed from ggml/src/ggml-quants.c, quantize_row_q4_0_ref:
#   for each block of QK4_0 = 32 consecutive weights:
#     amax = max_j |x_j|, max = the x_j attaining it (signed)
#     d  = max / -8          (stored as fp16)
#     id = d ? 1/d : 0
#     q_j = MIN(15, (int8_t)(x_j * id + 8.5f))       (truncation toward zero)
#   dequantized value = (q_j - 8) * d
# Symbol map: x -> w (blocks along the input dimension), d -> scale, q -> q.
# Verify against the ggml commit you compare with before publication.
# --------------------------------------------------------------------------
def q4_0_fake_quant(w):
    shape = w.shape
    x = w.detach().float().reshape(-1, 32)
    idx = x.abs().argmax(dim=1, keepdim=True)
    mx = torch.gather(x, 1, idx)
    d = (mx / -8.0).half().float()
    inv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
    q = torch.clamp(torch.trunc(x * inv + 8.5), max=15)
    return ((q - 8.0) * d).reshape(shape).to(w.dtype)


def q4_0_scalar_reference(block):
    """Direct scalar transcription of the ggml loop, for the self-test."""
    amax, mx = 0.0, 0.0
    for v in block:
        if amax < abs(v):
            amax, mx = abs(v), v
    d = float(np.float16(mx / -8.0))
    inv = 1.0 / d if d else 0.0
    out = []
    for v in block:
        q = min(15, int(np.float32(v) * np.float32(inv) + np.float32(8.5)))
        out.append((q - 8) * d)
    return out


# --------------------------------------------------------------------------
# Model structure
# --------------------------------------------------------------------------
def moe_blocks(model):
    blocks = []
    for name, mod in model.named_modules():
        experts, gate = getattr(mod, "experts", None), getattr(mod, "gate", None)
        if isinstance(experts, torch.nn.ModuleList) and gate is not None:
            m = re.search(r"layers\.(\d+)\.", name + ".")
            if m:
                blocks.append((int(m.group(1)), name, mod))
    if not blocks:
        raise SystemExit("No MoE block with `.gate` and a ModuleList `.experts` found. This is "
                         "probably a fused-expert implementation; pin transformers==4.56.2.")
    return sorted(blocks, key=lambda b: b[0])


def mlp_parts(mlp):
    for g, u, d in (("gate_proj", "up_proj", "down_proj"), ("w1", "w3", "w2")):
        if all(hasattr(mlp, x) for x in (g, u, d)) and hasattr(mlp, "act_fn"):
            return getattr(mlp, g), getattr(mlp, u), getattr(mlp, d), mlp.act_fn
    raise SystemExit(f"Unsupported expert MLP layout: {type(mlp).__name__}")


def top_k_of(model):
    c = model.config
    for k in ("num_experts_per_tok", "moe_k", "experts_per_token"):
        if getattr(c, k, None):
            return int(getattr(c, k))
    raise SystemExit("config has no top-k field")


def num_experts_of(block):
    return len(block.experts)


# --------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------
class State:
    def __init__(self, layers):
        self.mode = "off"          # off | calib | apply
        self.trace = False
        self.thr = {}              # layer -> threshold (current density)
        self.samples = {l: [] for l in layers}
        self.kept = {l: 0 for l in layers}
        self.total = {l: 0 for l in layers}
        self.cur = {}              # layer -> selected expert ids [T, k] (cpu int16)
        self.inp = {}              # layer -> router input [T, H] (device)
        self.per_call = 4096


def instrument(model, S, top_k):
    blocks = moe_blocks(model)
    for layer, _, block in blocks:
        for mlp in block.experts:
            G, U, D, act = mlp_parts(mlp)
            orig = mlp.forward

            def fwd(x, *args, _orig=orig, _G=G, _U=U, _D=D, _act=act, _l=layer, **kw):
                if S.mode == "off":
                    return _orig(x, *args, **kw)
                a = _act(_G(x))
                if S.mode == "calib":
                    flat = a.detach().abs().float().flatten()
                    if flat.numel() > S.per_call:
                        flat = flat[torch.randint(0, flat.numel(), (S.per_call,), device=flat.device)]
                    S.samples[_l].append(flat.cpu())
                    return _D(a * _U(x))
                m = a.abs() >= S.thr[_l]
                S.kept[_l] += int(m.sum())
                S.total[_l] += m.numel()
                return _D((a * m) * _U(x))

            mlp.forward = fwd

        def pre(mod, args, _l=layer):
            if S.trace:
                x = args[0]
                S.inp[_l] = x.detach().reshape(-1, x.shape[-1])

        def post(mod, args, out, _l=layer):
            if S.trace:
                ids = out[0] if isinstance(out, tuple) else torch.topk(out.float(), top_k, dim=-1).indices
                S.cur[_l] = ids.detach().reshape(-1, ids.shape[-1]).to(torch.int16).cpu()

        block.gate.register_forward_pre_hook(pre)
        block.gate.register_forward_hook(post)
    return blocks


def router_logits(gate, x):
    """Apply a router's weight matrix to an arbitrary activation.

    `x` is moved to the weight's DEVICE as well as its dtype. The Fate-style
    lookahead (arXiv 2502.12224) deliberately applies layer l+1's gate to layer
    l's router input, and under `device_map="auto"` those two layers can sit on
    different GPUs. At the shard boundary this raised, on Kaggle's 2x T4
    (2026-09-14):

        RuntimeError: Expected all tensors to be on the same device, but got
        mat2 is on cuda:1, different from other tensors on cuda:0

    The CPU self-test cannot reach this path, because it has one device. That
    is a gap in the self-test, not a reason the check is unnecessary: see
    selftest()'s cross-device case, which runs only when a second device exists.
    """
    W = gate.weight
    return F.linear(x.to(device=W.device, dtype=W.dtype), W).float()


def lookahead_supported(model):
    m = getattr(model.config, "topk_method", "greedy")
    return m in (None, "greedy")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def token_kl(logp_ref, logp):
    return (logp_ref.exp() * (logp_ref - logp)).sum(-1)


def nll_sum(logp, labels):
    """Summed teacher-forced negative log-likelihood of the true next token.

    Perplexity = exp(sum(nll) / n_tokens). Reported because the activation-
    sparsity literature justifies its sparsity levels with PERPLEXITY curves
    (TEAL, arXiv 2408.14690, reports perplexity vs sparsity; it never reports
    KL divergence or top-1 flip rate). ESTIMAND.md §3 sets our Tier-A margin in
    KL and flip rate instead, which is a far stricter bar: a model can change
    its argmax on a quarter of tokens while perplexity moves very little,
    because perplexity only scores the probability mass on the TRUE token.
    Without perplexity in our own artifact we cannot tell whether we DISAGREE
    with the published results or merely MEASURE something else — so both are
    reported, and any comparison to published sparsity levels must use this one.
    """
    return float(-logp[:-1].gather(-1, labels.unsqueeze(-1)).sum()), labels.numel()


class Calib:
    def __init__(self):
        self.conf, self.correct = [], []

    def add(self, logp, labels):
        c, pred = logp[:-1].exp().max(-1)
        self.conf.append(c.float().cpu())
        self.correct.append((pred == labels).float().cpu())

    def ece(self):
        c, k = torch.cat(self.conf), torch.cat(self.correct)
        e = torch.zeros(())
        for b in range(ECE_BINS):
            lo, hi = b / ECE_BINS, (b + 1) / ECE_BINS
            m = (c > lo) & (c <= hi)
            if m.any():
                e += m.float().mean() * (c[m].mean() - k[m].mean()).abs()
        return float(e)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def windows_from_text(tok, split, n, seq_len):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    n = min(n, ids.numel() // seq_len)
    return [ids[i * seq_len:(i + 1) * seq_len] for i in range(n)]


# --------------------------------------------------------------------------
# Core run
# --------------------------------------------------------------------------
@torch.no_grad()
def run(model, calib_windows, eval_windows, densities, floor_tokens, device, log=print):
    top_k = top_k_of(model)
    blocks = moe_blocks(model)
    layers = [b[0] for b in blocks]
    S = State(layers)
    instrument(model, S, top_k)
    E = num_experts_of(blocks[0][2])

    def fwd(ids):
        out = model(ids.unsqueeze(0).to(device)).logits[0].float()
        # float16 on GPUs without bfloat16 (Kaggle T4) can overflow; a KL computed
        # from inf/nan logits would be silently meaningless, so stop instead.
        if not torch.isfinite(out).all():
            raise FloatingPointError(f"non-finite logits (mode={S.mode}); rerun with --dtype bfloat16 "
                                     f"on a GPU that supports it, or on an H100")
        return out

    # 1. Calibrate thresholds (calibration corpus only).
    S.mode = "calib"
    for w in calib_windows:
        fwd(w)
    thresholds = {}
    for d in densities:
        thresholds[d] = {}
        for l in layers:
            v = torch.cat(S.samples[l]).numpy()
            thresholds[d][l] = 0.0 if d >= 1.0 else float(np.quantile(v, 1.0 - d))
    S.samples = {l: [] for l in layers}

    traces = {l: [] for l in layers}
    look_recall, look_n = {}, {}
    acc = {d: {"kl": [], "flip": 0, "n": 0, "calib": Calib(), "nll": 0.0, "nll_n": 0}
           for d in densities}
    ref_nll, ref_nll_n = 0.0, 0
    ref_calib = Calib()
    floor_store = []
    stored = 0
    for wi, w in enumerate(eval_windows):
        # 2. Reference pass with tracing.
        S.mode, S.trace = "off", True
        ref = fwd(w)
        S.trace = False
        logp_ref = F.log_softmax(ref, -1)
        labels = w[1:].to(device)
        ref_calib.add(logp_ref, labels)
        _s, _n = nll_sum(logp_ref, labels)
        ref_nll += _s; ref_nll_n += _n
        for l in layers:
            traces[l].append(S.cur[l].numpy())
        if lookahead_supported(model):
            for (l, _, b), (l2, _, b2) in zip(blocks[:-1], blocks[1:]):
                pred = torch.topk(router_logits(b2.gate, S.inp[l]), top_k, -1).indices.cpu()
                act = S.cur[l2].long()
                hit = (pred.unsqueeze(-1) == act.unsqueeze(-2)).any(-1).float().mean(-1)
                look_recall[l2] = look_recall.get(l2, 0.0) + float(hit.sum())
                look_n[l2] = look_n.get(l2, 0) + hit.numel()
        if stored < floor_tokens:
            floor_store.append((w, logp_ref.half().cpu()))
            stored += w.numel()
        # 3. Sparse passes.
        for d in densities:
            S.mode, S.thr = "apply", thresholds[d]
            lp = F.log_softmax(fwd(w), -1)
            kl = token_kl(logp_ref, lp)
            a = acc[d]
            a["kl"].append(kl.cpu())
            a["flip"] += int((lp.argmax(-1) != logp_ref.argmax(-1)).sum())
            a["n"] += kl.numel()
            a["calib"].add(lp, labels)
            _s, _n = nll_sum(lp, labels)
            a["nll"] += _s; a["nll_n"] += _n
        S.mode = "off"
        log(f"  window {wi + 1}/{len(eval_windows)}")

    realized = {}
    # kept/total were accumulated across all densities; recompute per density cleanly.
    results = {"top_k": top_k, "num_experts": E, "moe_layers": layers,
               "ref_ece": ref_calib.ece(),
               "ref_perplexity": math.exp(ref_nll / ref_nll_n), "densities": {}}
    for d in densities:
        a = acc[d]
        kl = torch.cat(a["kl"])
        results["densities"][str(d)] = {
            "kl_mean": float(kl.mean()), "kl_p99": float(torch.quantile(kl, 0.99)),
            "flip_rate": a["flip"] / a["n"], "ece": a["calib"].ece(), "tokens": a["n"],
            "perplexity": math.exp(a["nll"] / a["nll_n"]),
            "perplexity_ratio_vs_dense": math.exp(a["nll"] / a["nll_n"]) / math.exp(ref_nll / ref_nll_n),
            "thresholds": {str(l): thresholds[d][l] for l in layers}}
    # Realized density on eval data, one clean pass per density on the first window.
    for d in densities:
        S.kept = {l: 0 for l in layers}
        S.total = {l: 0 for l in layers}
        S.mode, S.thr = "apply", thresholds[d]
        fwd(eval_windows[0])
        realized[str(d)] = {str(l): S.kept[l] / max(1, S.total[l]) for l in layers}
    S.mode = "off"
    results["realized_density_first_window"] = realized
    results["lookahead_recall"] = ({str(l): look_recall[l] / look_n[l] for l in look_recall}
                                   if look_recall else "unsupported for this router")
    return results, {l: np.concatenate(traces[l]) for l in layers}, floor_store, S


@torch.no_grad()
def format_floor(model, floor_store, device):
    """Quantize linear weights in place to Q4_0 (irreversible — run last)."""
    skipped = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            if name.endswith("lm_head") or name.endswith(".gate") or mod.in_features % 32:
                skipped.append(name)
                continue
            mod.weight.data = q4_0_fake_quant(mod.weight.data)
    kls, flips, n = [], 0, 0
    for w, lr in floor_store:
        lp = F.log_softmax(model(w.unsqueeze(0).to(device)).logits[0].float(), -1)
        lr = lr.to(device).float()
        kl = token_kl(lr, lp)
        kls.append(kl.cpu())
        flips += int((lp.argmax(-1) != lr.argmax(-1)).sum())
        n += kl.numel()
    kl = torch.cat(kls)
    return {"format": "Q4_0", "kl_mean": float(kl.mean()), "kl_p99": float(torch.quantile(kl, 0.99)),
            "flip_rate": flips / n, "tokens": n,
            "left_unquantized": "lm_head, routers (*.gate), layers with in_features % 32 != 0",
            "n_linear_skipped": len(skipped)}


# --------------------------------------------------------------------------
# Self-test — every assertion is fixed by definition, not by a result
# --------------------------------------------------------------------------
def selftest():
    import transformers
    from transformers import (MixtralConfig, MixtralForCausalLM, OlmoeConfig, OlmoeForCausalLM,
                              Qwen3MoeConfig, Qwen3MoeForCausalLM)
    print("transformers", transformers.__version__, "torch", torch.__version__)
    failures = []

    def check(cond, msg):
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            failures.append(msg)

    rng = np.random.default_rng(0)
    for _ in range(50):
        blk = (rng.standard_normal(32) * rng.uniform(0.01, 3)).astype(np.float32)
        ref = np.array(q4_0_scalar_reference(blk.tolist()), dtype=np.float32)
        got = q4_0_fake_quant(torch.from_numpy(blk)).numpy()
        if not np.allclose(ref, got, atol=0, rtol=0):
            check(False, "Q4_0 vectorized == scalar ggml transcription")
            break
    else:
        check(True, "Q4_0 vectorized == scalar ggml transcription (50 random blocks)")
    blk = np.zeros(32, dtype=np.float32)
    blk[0], blk[1], blk[2] = -4.0, 1.0, 0.3
    got = q4_0_fake_quant(torch.from_numpy(blk)).numpy()
    check(got[0] == -4.0 and got[1] == 1.0 and got[2] == 0.5,
          "Q4_0 hand block: d=0.5 -> [-4, 1, 0.3] dequantize to [-4, 1, 0.5]")

    base = dict(vocab_size=96, hidden_size=64, num_hidden_layers=3, num_attention_heads=4,
                max_position_embeddings=128)
    models = [
        ("qwen3_moe", Qwen3MoeForCausalLM, Qwen3MoeConfig(**base, intermediate_size=128, moe_intermediate_size=32,
                                                         num_key_value_heads=2, head_dim=16, num_experts=8,
                                                         num_experts_per_tok=2)),
        ("olmoe", OlmoeForCausalLM, OlmoeConfig(**base, intermediate_size=32, num_key_value_heads=4,
                                                num_experts=8, num_experts_per_tok=2)),
        ("mixtral", MixtralForCausalLM, MixtralConfig(**base, intermediate_size=32, num_key_value_heads=2,
                                                      num_local_experts=4, num_experts_per_tok=2)),
    ]
    for tag, cls, cfg in models:
        torch.manual_seed(0)
        model = cls(cfg).eval()
        g = torch.Generator().manual_seed(1)
        wins = [torch.randint(0, cfg.vocab_size, (32,), generator=g) for _ in range(4)]
        with torch.no_grad():
            plain = model(wins[0].unsqueeze(0)).logits[0].float()
            out = model(wins[0].unsqueeze(0), output_router_logits=True)
            router = [r.float() for r in out.router_logits]
        res, traces, floor_store, S = run(model, wins[:2], wins[2:], DENSITIES, 64, "cpu", log=lambda *_: None)
        with torch.no_grad():
            S.mode, S.thr = "apply", {l: 0.0 for l in S.thr}
            patched = model(wins[0].unsqueeze(0)).logits[0].float()
            S.mode = "off"
        check(torch.equal(plain, patched), f"[{tag}] density 1.0 (threshold 0) is bit-identical to the unmodified model")
        check(res["densities"]["1.0"]["kl_mean"] == 0.0 and res["densities"]["1.0"]["flip_rate"] == 0.0,
              f"[{tag}] KL and flip rate are exactly 0 at density 1.0")
        k, E = res["top_k"], res["num_experts"]
        ok = all(t.shape[1] == k and t.min() >= 0 and t.max() < E for t in traces.values())
        check(ok, f"[{tag}] traces have shape [T,{k}] and ids in [0,{E})")
        # Traced ids must equal the model's own router decisions (external reference).
        S.trace = True
        with torch.no_grad():
            model(wins[0].unsqueeze(0))
        S.trace = False
        same = all(torch.equal(torch.sort(S.cur[l].long(), -1).values,
                               torch.sort(torch.topk(router[i], k, -1).indices, -1).values)
                   for i, l in enumerate(sorted(S.cur)))
        check(same, f"[{tag}] traced experts == top-k of the model's own router_logits")
        # Lookahead machinery: predicting layer l with ITS OWN router must give recall 1.
        blocks = moe_blocks(model)
        l0, _, b0 = blocks[0]
        own = torch.topk(router_logits(b0.gate, S.inp[l0]), k, -1).indices
        check(torch.equal(torch.sort(own, -1).values, torch.sort(S.cur[l0].long(), -1).values),
              f"[{tag}] router applied to its own captured input reproduces its selection")
        lr = res["lookahead_recall"]
        check(isinstance(lr, dict) and all(0.0 <= v <= 1.0 for v in lr.values()),
              f"[{tag}] lookahead recall is a proportion")
        check(res["densities"]["0.1"]["kl_mean"] > 0.0, f"[{tag}] density 0.1 changes the output (KL > 0)")
        fl = format_floor(model, floor_store, "cpu")
        check(fl["kl_mean"] > 0.0 and fl["tokens"] > 0, f"[{tag}] Q4_0 floor is measurable (KL > 0)")

    # Cross-device lookahead. The Fate lookahead applies layer l+1's gate to
    # layer l's input, and under device_map="auto" those layers can be on
    # different GPUs; that combination raised a RuntimeError on Kaggle's 2x T4
    # on 2026-09-14 while this self-test passed, because one device cannot
    # exercise it. Run it whenever a second device exists, and SAY SO when it
    # cannot be run rather than reporting a pass that did not happen.
    n_dev = torch.cuda.device_count()
    if n_dev >= 2:
        gate = torch.nn.Linear(8, 4, bias=False).to("cuda:1")
        x = torch.randn(3, 8, device="cuda:0")
        try:
            out = router_logits(gate, x)
            check(out.shape == (3, 4), "cross-device lookahead: gate on cuda:1, input on cuda:0")
        except RuntimeError as e:
            check(False, f"cross-device lookahead raised {type(e).__name__}: {e}")
    else:
        print(f"SKIP  cross-device lookahead: needs 2 CUDA devices, found {n_dev}. "
              f"This is the path that failed on Kaggle 2x T4 and it is NOT covered here.")

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


# --------------------------------------------------------------------------
def out_dir(arg):
    if arg:
        os.makedirs(arg, exist_ok=True)
        return arg
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from _paths import write_path
        return os.path.dirname(write_path("_dir_probe"))
    except ImportError:
        d = os.path.join("moe_phone_out", datetime.date.today().isoformat())
        print(f"_paths not importable (running outside the repo); writing to {d}")
        os.makedirs(d, exist_ok=True)
        return d


def main():
    p = argparse.ArgumentParser(description="G2+G3 on a stock MoE checkpoint")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--model")
    p.add_argument("--windows", type=int, default=64, help="evaluation windows (wikitext-2 test)")
    p.add_argument("--calib-windows", type=int, default=16, help="calibration windows (wikitext-2 train)")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--floor-tokens", type=int, default=8192)
    p.add_argument("--densities", default=",".join(str(d) for d in DENSITIES))
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16"])
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--no-floor", action="store_true")
    p.add_argument("--out-dir", default=None)
    a = p.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.model:
        p.error("--model is required unless --selftest")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    cuda = torch.cuda.is_available()
    # torch.cuda.is_bf16_supported() returns True on a Tesla T4, because PyTorch
    # EMULATES bfloat16 there by upcasting to fp32. That is numerically fine but
    # several times slower, and it silently contradicts protocols/RUNBOOK.md,
    # which states "T4s have no bfloat16, so the reference runs in float16" — a
    # documented behaviour that the code did not assert. Compute capability >= 8.0
    # (Ampere) is the test for NATIVE bfloat16. Taken as the minimum over all
    # visible devices, since the model is sharded across them.
    native_bf16 = cuda and min(torch.cuda.get_device_capability(i)[0]
                               for i in range(torch.cuda.device_count())) >= 8
    if a.dtype == "auto":
        dtype = torch.bfloat16 if native_bf16 else torch.float16
    else:
        dtype = getattr(torch, a.dtype)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=dtype, device_map="auto" if cuda else None,
        trust_remote_code=a.trust_remote_code).eval()
    device = next(model.parameters()).device
    calib = windows_from_text(tok, "train", a.calib_windows, a.seq_len)
    evalw = windows_from_text(tok, "test", a.windows, a.seq_len)
    dens = [float(x) for x in a.densities.split(",")]
    if 1.0 not in dens:
        dens = [1.0] + dens
    print(f"{a.model}: {len(calib)} calibration + {len(evalw)} evaluation windows of {a.seq_len}, dtype {dtype}")
    res, traces, floor_store, _ = run(model, calib, evalw, dens, a.floor_tokens, device)
    res["floor"] = None if a.no_floor else format_floor(model, floor_store, device)
    res.update({"model": a.model, "dtype": str(dtype), "bf16_native": native_bf16,
                "n_cuda_devices": torch.cuda.device_count() if cuda else 0,
                "seq_len": a.seq_len,
                "calib_windows": len(calib), "eval_windows": len(evalw),
                "corpus": "wikitext-2-raw-v1 (calibration: train, evaluation: test)",
                "transformers": __import__("transformers").__version__, "torch": torch.__version__,
                "runtime_s": time.time() - t0})
    od = out_dir(a.out_dir)
    tag = a.model.split("/")[-1]
    with open(os.path.join(od, f"g3_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(os.path.join(od, f"traces_{tag}.npz"),
                        num_experts=res["num_experts"], **{f"L{l}": t for l, t in traces.items()})
    print("\n density   KL mean    KL p99   flip    ECE")
    for d, r in res["densities"].items():
        print(f"  {d:>5s}  {r['kl_mean']:.5f}  {r['kl_p99']:.5f}  {r['flip_rate']:.4f}  {r['ece']:.4f}")
    if res["floor"]:
        f = res["floor"]
        print(f"  Q4_0 floor  {f['kl_mean']:.5f}  {f['kl_p99']:.5f}  {f['flip_rate']:.4f}   (Tier-A margin)")
    print(f"reference ECE {res['ref_ece']:.4f}; wrote {od}")


if __name__ == "__main__":
    main()
