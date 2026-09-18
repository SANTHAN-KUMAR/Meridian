"""
make_cases.py — test cases for gx (fused expert FFN): random ones that stress the numerics, and real
Qwen3-30B-A3B expert slices read from the GGUF by byte offset (the model is never loaded or run).

  python3 host/gpu_ffn/make_cases.py <out_dir> [--gguf PATH] [--random N]

Case file (little-endian): b"GXC2", int32 n_embd, n_ff, k, down_type (0 = Q4_0, 1 = Q4_1); int32 ids[k]
(slot s uses expert ids[s], a permutation of 0..k-1); float32 x[n_embd]; then for expert e = 0..k-1: gate
(Q4_0), up (Q4_0), down (Q4_0 or Q4_1, per down_type) bytes in GGUF block layout. In Qwen3-30B-A3B-Q4_0.gguf the down type is Q4_1 in
layers 0-5 and Q4_0 in layers 6-47, so real cases cover both. A manifest.json lists every case with what it contains.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

NE, NF = 2048, 768
GU = NF * (NE // 32) * 18
DN = {0: NE * (NF // 32) * 18, 1: NE * (NF // 32) * 20}


def f16(a):
    return np.asarray(a, dtype=np.float32).astype(np.float16)


def rand_q4_0(rng, rows, k, dmag):
    nb = rows * (k // 32)
    b = np.zeros((nb, 18), np.uint8)
    d = f16(rng.choice([-1, 1], nb) * 10 ** rng.uniform(np.log10(dmag) - 1, np.log10(dmag) + 0.5, nb))
    b[:, 0:2] = d.view(np.uint8).reshape(nb, 2)
    b[:, 2:] = rng.integers(0, 256, (nb, 16), dtype=np.uint8)
    return b.tobytes()


def rand_q4_1(rng, rows, k, dmag):
    nb = rows * (k // 32)
    b = np.zeros((nb, 20), np.uint8)
    d = f16(10 ** rng.uniform(np.log10(dmag) - 1, np.log10(dmag) + 0.5, nb))
    m = f16(-d.astype(np.float32) * rng.uniform(6, 9, nb))
    b[:, 0:2] = d.view(np.uint8).reshape(nb, 2)
    b[:, 2:4] = m.view(np.uint8).reshape(nb, 2)
    b[:, 4:] = rng.integers(0, 256, (nb, 16), dtype=np.uint8)
    return b.tobytes()


def write_case(path, x, ids, experts, dt):
    k = len(ids)
    with open(path, "wb") as f:
        f.write(b"GXC2" + struct.pack("<4i", NE, NF, k, dt))
        f.write(np.asarray(ids, np.int32).tobytes())
        f.write(np.asarray(x, np.float32).tobytes())
        for g, u, d in experts:
            assert len(g) == GU and len(u) == GU and len(d) == DN[dt]
            f.write(g + u + d)


def random_x(rng, kind):
    x = rng.standard_normal(NE).astype(np.float32)
    if kind == "normal":
        return x
    if kind == "wide":        # per-block scales over 8 decades: fp16 subnormal d, large d
        return (x * np.repeat(10 ** rng.uniform(-6, 2, NE // 32), 32)).astype(np.float32)
    if kind == "zeros":       # whole zero blocks (amax 0: d = 0, id = 0) and isolated zeros
        x[rng.integers(0, NE // 32, 8)[:, None] * 32 + np.arange(32)] = 0
        x[rng.integers(0, NE, 64)] = 0
        return x
    if kind == "ties":        # values that land exactly on .5 after scaling: x = (q + 0.5) * amax / 127
        blocks = x.reshape(-1, 32)
        amax = np.abs(blocks).max(1, keepdims=True)
        q = rng.integers(-126, 126, blocks.shape) + 0.5
        t = (q * amax / 127).astype(np.float32)
        t[:, 0] = amax[:, 0]
        return t.reshape(-1)
    if kind == "outliers":    # a few channels 100x the rest, like real hidden states
        x[rng.integers(0, NE, 6)] *= 100
        return x
    raise ValueError(kind)


def quant_blocks(rng, n):
    """Blocks aimed at the quantizer's rounding: elements within 2 ulps of a rounding half-way point
    (k + 0.5) / id, half of the blocks with an amax where 127/amax != 1/(amax/127) in fp32, zero blocks,
    blocks whose d is an fp16 subnormal, and fp32-subnormal elements."""
    f32 = np.float32
    out = np.zeros((n, 32), f32)
    diff = []
    while len(diff) < n // 2:
        a = (10 ** rng.uniform(-6, 3, 4096)).astype(f32)
        d = a / f32(127)
        m = (f32(127) / a) != (f32(1) / d)
        diff.extend(a[m].tolist())
    for i in range(n):
        kind = i % 8
        if kind == 6:
            continue                                   # all-zero block
        if i % 2 == 0:
            amax = f32(diff[i // 2])
        elif kind == 7:
            amax = f32(10 ** rng.uniform(-9, -5.5))    # d = amax/127 below fp16's normal range
        else:
            amax = f32(10 ** rng.uniform(-6, 3))
        d = f32(amax / f32(127))
        idv = f32(f32(1) / d)
        k = rng.integers(-127, 127, 32)
        t = ((k + 0.5) / np.float64(idv)).astype(f32)
        for _ in range(2):
            step = rng.integers(-1, 2, 32)
            t = np.where(step > 0, np.nextafter(t, f32(np.inf)), np.where(step < 0, np.nextafter(t, f32(-np.inf)), t)).astype(f32)
        t = np.clip(t, -amax, amax)
        if kind == 5:
            t[rng.integers(1, 32, 4)] = np.array([1e-40, -3e-39, 1e-45, -1e-44], f32)   # fp32 subnormals
        t[0] = amax if rng.integers(2) else -amax
        out[i] = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--random", type=int, default=20)
    ap.add_argument("--layers", default="0,5,6,12,24,36,47")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(20260918)
    man = []
    kinds = ["normal", "wide", "zeros", "ties", "outliers"]
    for i in range(a.random):
        k = 1 + i % 8
        kind = kinds[i % len(kinds)]
        dmag = [3e-3, 3e-2, 3e-4][i % 3]
        x = random_x(rng, kind)
        dt = (i // 2) % 2
        down = rand_q4_1 if dt else rand_q4_0
        ex = [(rand_q4_0(rng, NF, NE, dmag), rand_q4_0(rng, NF, NE, dmag), down(rng, NE, NF, dmag)) for _ in range(k)]
        ids = rng.permutation(k).tolist()
        name = f"rand_{i:02d}.bin"
        write_case(os.path.join(a.out, name), x, ids, ex, dt)
        man.append({"file": name, "source": "random", "k": k, "down_type": ["Q4_0", "Q4_1"][dt], "x": kind,
                    "d_scale": dmag, "ids": ids})
    qb = quant_blocks(rng, 65536)
    qb.tofile(os.path.join(a.out, "quant_blocks.f32"))
    man.append({"file": "quant_blocks.f32", "source": "crafted quantizer blocks", "blocks": int(qb.shape[0])})
    if a.gguf:
        gp = os.environ.get("GGUF_PY", "/tmp/claude-1000/bmoe-prefetch/third_party/llama.cpp/gguf-py")
        sys.path.insert(0, gp)
        from gguf import GGUFReader   # memory-maps the file; only the bytes read below are touched
        r = GGUFReader(a.gguf)
        t = {x.name: x for x in r.tensors}
        with open(a.gguf, "rb") as f:
            for li, L in enumerate(int(v) for v in a.layers.split(",")):
                tg, tu, td = (t[f"blk.{L}.ffn_{p}_exps.weight"] for p in ("gate", "up", "down"))
                tn = t[f"blk.{L}.ffn_norm.weight"]
                assert tg.tensor_type.name == "Q4_0" and tu.tensor_type.name == "Q4_0", (tg.tensor_type.name, tu.tensor_type.name)
                dt = {"Q4_0": 0, "Q4_1": 1}[td.tensor_type.name]
                n_exp = int(tg.shape[-1])
                assert tg.n_bytes == GU * n_exp and td.n_bytes == DN[dt] * n_exp
                k = 8 if li % 2 == 0 else 1 + (3 * li) % 8
                chosen = sorted(rng.choice(n_exp, k, replace=False).tolist())
                ex = []
                for e in chosen:
                    parts = []
                    for tt, sz in ((tg, GU), (tu, GU), (td, DN[dt])):
                        f.seek(int(tt.data_offset) + e * sz)
                        parts.append(f.read(sz))
                    ex.append(tuple(parts))
                gain = np.asarray(tn.data, np.float32)
                # the MoE input is RMSNorm(hidden) * ffn_norm.weight; RMSNorm output has unit RMS
                x = (rng.standard_normal(NE).astype(np.float32) * gain).astype(np.float32)
                ids = rng.permutation(k).tolist()
                name = f"qwen3_L{L:02d}.bin"
                write_case(os.path.join(a.out, name), x, ids, ex, dt)
                man.append({"file": name, "source": "qwen3-30b-a3b gguf", "layer": L, "k": k, "experts": chosen,
                            "down_type": td.tensor_type.name,
                            "x": "randn * ffn_norm.weight", "ids": ids})
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(f"wrote {len(man)} cases to {a.out}")


if __name__ == "__main__":
    main()
