"""
Bytes one decoded token touches, read from a GGUF's own tensor table.

For a resident model the non-flash cost of a token is the weights it streams from
DRAM through the matmuls. That quantity is fixed by the file, so it is read from
the file rather than estimated from parameter counts:

  routed-expert tensors (name contains "_exps")  bytes * top_k / n_expert
  token embedding (token_embd)                   one row per token: bytes / n_vocab
  every other tensor                             all of it

Tensor byte sizes are the GGUF's stored sizes (quantised blocks included), so the
answer is in the deployment format, not in parameters x an assumed bpw.

Uses gguf-py from a llama.cpp checkout (pass --gguf-py or set GGUF_PY).

Run:
  python moe-phone/gates/gguf_active.py MODEL.gguf [MODEL2.gguf ...] --gguf-py <llama.cpp>/gguf-py
"""
import argparse
import json
import os
import sys


def active_bytes(path):
    from gguf import GGUFReader
    r = GGUFReader(path)
    kv = {}
    for f in r.fields.values():
        if f.name.endswith((".expert_count", ".expert_used_count")) or f.name == "general.architecture":
            v = f.parts[f.data[0]]
            kv[f.name] = (bytes(v).decode() if f.name == "general.architecture" else int(v[0]))
    arch = kv["general.architecture"]
    E = kv.get(f"{arch}.expert_count", 0)
    k = kv.get(f"{arch}.expert_used_count", 0)
    tot = exp = emb = other = 0
    n_vocab = None
    for t in r.tensors:
        b = int(t.n_bytes)
        tot += b
        if "_exps" in t.name:
            exp += b
        elif t.name == "token_embd.weight":
            emb = b
            n_vocab = int(t.shape[-1]) if len(t.shape) > 1 else None
        else:
            other += b
    if not (E and k):
        raise ValueError(f"{path}: no expert_count / expert_used_count metadata")
    emb_row = emb / n_vocab if n_vocab else 0.0
    act = other + exp * k / E + emb_row
    return {"file": os.path.basename(path), "arch": arch, "n_expert": E, "top_k": k,
            "file_tensor_bytes": tot, "expert_bytes_all": exp, "embedding_bytes": emb,
            "other_bytes": other, "active_bytes_per_token": act,
            "expert_bytes_per_token": exp * k / E,
            "per_expert_bytes_all_layers": exp / E}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("gguf", nargs="+")
    p.add_argument("--gguf-py", default=os.environ.get("GGUF_PY"))
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.gguf_py:
        sys.path.insert(0, a.gguf_py)
    res = [active_bytes(g) for g in a.gguf]
    for x in res:
        print(f"{x['file']}: {x['arch']} top-{x['top_k']} of {x['n_expert']}; "
              f"active {x['active_bytes_per_token'] / 1e6:.1f} MB/token "
              f"(experts {x['expert_bytes_per_token'] / 1e6:.1f} MB), file {x['file_tensor_bytes'] / 1e9:.3f} GB")
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
