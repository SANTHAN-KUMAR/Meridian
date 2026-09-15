"""
Routing traces through llama.cpp, in the SAME format as traces_sparsity.py.

Two subcommands:

  tokens   write the evaluation token ids for a model: wikitext-2-raw-v1 TEST,
           joined with "\\n\\n", tokenized once with the model's own HF tokenizer,
           cut into `--windows` windows of `--seq-len` — character for character
           the procedure of traces_sparsity.windows_from_text(), so a llama.cpp
           trace and an HF trace of the same model see identical token ids.

  convert  turn tools/route_trace's binary output into traces_<tag>.npz with the
           keys traces_sparsity.py writes (num_experts, L<layer> int16 [T, k]),
           so cache_sim.py and every downstream gate read it unchanged. Also
           writes a JSON sidecar recording the engine, GGUF file and its sha256,
           because a Q4_0 llama.cpp trace and an fp16 HF trace are different
           measurements and must never be silently pooled.

  compare  per-token agreement between two traces of the SAME model and tokens
           (e.g. llama.cpp Q4_0 vs the committed HF fp16 trace): the fraction of
           each token's top-k set shared, per layer. This is the direct size of
           the engine/format confound pre-registered in s9_prereg.py.

Run:
  python llamacpp_traces.py tokens --model allenai/OLMoE-1B-7B-0924 --out tok_olmoe.bin
  route_trace MODEL.gguf tok_olmoe.bin olmoe.trc 16
  python llamacpp_traces.py convert olmoe.trc --gguf MODEL.gguf --tag OLMoE-1B-7B-0924-q4_0-llamacpp
  python llamacpp_traces.py compare A.npz B.npz
"""
import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_tokens(a):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=a.split)
    ids = tok("\n\n".join(ds["text"])).input_ids          # same call as traces_sparsity
    n = min(a.windows, len(ids) // a.seq_len)
    ids = np.asarray(ids[:n * a.seq_len], dtype=np.int32)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<ii", n, a.seq_len))
        f.write(ids.tobytes())
    print(f"{a.model}: {n} windows x {a.seq_len} tokens -> {a.out}")


def read_trc(path):
    with open(path, "rb") as f:
        if f.read(8) != b"MOETRC01":
            raise ValueError(f"{path} is not a route_trace file")
        L, k, nw, wl, E = struct.unpack("<5i", f.read(20))
        layers = list(struct.unpack(f"<{L}i", f.read(4 * L)))
        data = np.frombuffer(f.read(), dtype=np.int16)
    T = data.size // (L * k)
    if T != nw * wl:
        # a run that died mid-way leaves whole windows only (route_trace flushes
        # per window); say so rather than pretend the trace is complete
        print(f"WARNING: {T} tokens present, header promised {nw * wl}")
    return layers, k, E, data[:T * L * k].reshape(T, L, k)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_convert(a):
    layers, k, E, ev = read_trc(a.trc)
    if a.num_experts:
        E = a.num_experts
    if E <= 0:
        raise ValueError("expert count unknown: pass --num-experts")
    if ev.max() >= E or ev.min() < 0:
        raise ValueError(f"expert id outside [0, {E}): min {ev.min()} max {ev.max()}")
    # every token's k experts must be distinct, or this is not a top-k set
    s = np.sort(ev, axis=2)
    n_dup = int((s[:, :, 1:] == s[:, :, :-1]).any(axis=2).sum())
    if n_dup:
        raise ValueError(f"{n_dup} (token, layer) events repeat an expert")
    od = a.out_dir or os.path.dirname(os.path.abspath(a.trc))
    os.makedirs(od, exist_ok=True)
    out = os.path.join(od, f"traces_{a.tag}.npz")
    np.savez_compressed(out, num_experts=E,
                        **{f"L{r}": ev[:, r, :].astype(np.int16) for r in range(len(layers))})
    meta = {"engine": "llama.cpp (tools/route_trace.cpp, ffn_moe_topk tensors)",
            "gguf": os.path.basename(a.gguf) if a.gguf else None,
            "gguf_sha256": sha256(a.gguf) if a.gguf and a.sha else None,
            "llama_cpp_commit": a.llama_commit, "tokens": int(ev.shape[0]),
            "moe_layers_in_graph": layers, "top_k": k, "num_experts": E,
            "corpus": "wikitext-2-raw-v1 test, traces_sparsity.windows_from_text procedure",
            "note": "keys L<i> are the i-th MoE layer in graph order, like traces_sparsity"}
    with open(out.replace(".npz", ".json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1)
    print(f"{ev.shape[0]} tokens x {len(layers)} layers x top-{k} of {E} -> {out}")


def load_npz(p):
    z = np.load(p)
    keys = sorted((kk for kk in z.files if kk.startswith("L")), key=lambda s: int(s[1:]))
    return int(z["num_experts"]), np.stack([z[kk] for kk in keys], axis=1)


def cmd_compare(a):
    Ea, A = load_npz(a.a)
    Eb, B = load_npz(a.b)
    if Ea != Eb or A.shape[1:] != B.shape[1:]:
        raise ValueError(f"geometry differs: {A.shape} E={Ea} vs {B.shape} E={Eb}")
    T = min(A.shape[0], B.shape[0])
    A, B = A[:T], B[:T]
    k = A.shape[2]
    shared = np.zeros(A.shape[:2])
    for j in range(k):
        shared += (B == A[:, :, j:j + 1]).any(axis=2)
    frac = shared / k
    res = {"tokens": T, "mean_overlap": float(frac.mean()),
           "exact_set_match": float((frac == 1.0).mean()),
           "per_layer_mean_overlap": frac.mean(axis=0).round(4).tolist()}
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(dict(res, a=os.path.basename(a.a), b=os.path.basename(a.b)), f, indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sp = p.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("tokens")
    t.add_argument("--model", required=True)
    t.add_argument("--split", default="test")
    t.add_argument("--windows", type=int, default=64)
    t.add_argument("--seq-len", type=int, default=512)
    t.add_argument("--out", required=True)
    c = sp.add_parser("convert")
    c.add_argument("trc")
    c.add_argument("--tag", required=True)
    c.add_argument("--gguf", default=None)
    c.add_argument("--sha", action="store_true", help="hash the GGUF (slow for 17 GB)")
    c.add_argument("--num-experts", type=int, default=0)
    c.add_argument("--llama-commit", default=None)
    c.add_argument("--out-dir", default=None)
    m = sp.add_parser("compare")
    m.add_argument("a")
    m.add_argument("b")
    m.add_argument("--out", default=None)
    a = p.parse_args()
    {"tokens": cmd_tokens, "convert": cmd_convert, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
