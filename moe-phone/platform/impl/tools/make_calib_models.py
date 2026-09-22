"""
make_calib_models.py - synthetic calibration models for Meridian's on-device compute probe (closes stub PL-E11).

Why synthetic: the app must predict a model's speed BEFORE the user downloads it, so the kernel throughput has to be
measured with something the APK already carries. These files are real GGUFs (llama architecture) that the real engine
loads and decodes through its normal path. Their weights are one constant block repeated, so the APK's zip deflate
compresses them ~1000x, but at run time the engine still reads and multiplies every byte: dot-product kernels have no
data-dependent timing, so the measured rate is the kernel's rate on this device for that ggml type.

The tokenizer is byte-level BPE with one merge (every byte is one token), so the vocabulary is 259 entries and the
embedding/output tables are negligible: per-token bytes are the transformer matrices, which is what we want to time.

Outputs, per ggml type T in {Q4_0, Q8_0, Q4_K, Q6_K, MXFP4}:
  calib_<T>_L1.gguf  (1 layer)  and  calib_<T>_L3.gguf (3 layers)
Q4_0 also gets a narrow 3-layer model (calib_Q4_0_L3N.gguf, ffn 2048). The three Q4_0 models identify a fixed per-token
cost t0, a per-layer cost t_layer and a per-byte cost 1/W:
  t_token = t0 + n_layer * t_layer + active_bytes / W      (05_PERFORMANCE_MODEL.md section 3.2, compute term)
Usage: python3 make_calib_models.py OUT_DIR
"""
import os, struct, sys
import numpy as np

sys.path.insert(0, os.environ.get("GGUF_PY", os.path.expanduser("~/meridian-engine/src/third_party/llama.cpp/gguf-py")))
import gguf  # noqa: E402
from gguf import GGMLQuantizationType as Q  # noqa: E402

N_EMBD, N_FF, N_HEAD, N_KV, HD = 2048, 8192, 16, 8, 128


def f16(x):
    return struct.pack("<e", x)


# One block per type (bytes), chosen to give small finite weights. Layouts from ggml-common.h.
BLOCKS = {
    "Q4_0": (Q.Q4_0, 32, f16(0.001) + bytes([0x98]) * 16),                                   # d, qs[16]
    "Q8_0": (Q.Q8_0, 32, f16(0.001) + bytes([1]) * 32),                                      # d, qs[32]
    "Q4_K": (Q.Q4_K, 256, f16(0.001) + f16(0.0) + bytes([1]) * 12 + bytes([0x11]) * 128),     # d, dmin, scales[12], qs[128]
    "Q6_K": (Q.Q6_K, 256, bytes([0x11]) * 128 + bytes([0]) * 64 + bytes([1]) * 16 + f16(0.001)),  # ql, qh, scales, d
    "MXFP4": (Q.MXFP4, 32, bytes([120]) + bytes([0x11]) * 16),                               # e8m0, qs[16]
}


def byte_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    m = dict(zip(bs, cs))
    return [chr(m[b]) for b in range(256)]


def tensor_data(tname, shape, qname):
    """shape is ggml order (ne0, ne1). Returns (bytes-array, raw_shape_numpy_order, raw_dtype)."""
    qt, blk, block = BLOCKS[qname]
    ne0, ne1 = shape
    assert ne0 % blk == 0, (tname, ne0, blk)
    nblocks = ne0 // blk * ne1
    data = np.frombuffer(block * nblocks, dtype=np.uint8)
    return data, qt


def write(path, qname, n_layer, n_ff=N_FF):
    w = gguf.GGUFWriter(path, "llama")
    w.add_name(f"meridian-calib-{qname}-L{n_layer}")
    w.add_context_length(2048); w.add_embedding_length(N_EMBD); w.add_block_count(n_layer)
    w.add_feed_forward_length(n_ff); w.add_head_count(N_HEAD); w.add_head_count_kv(N_KV)
    w.add_rope_dimension_count(HD); w.add_layer_norm_rms_eps(1e-5); w.add_rope_freq_base(10000.0)
    # llama.cpp requires the merges key and gguf-py drops an empty list, so there is exactly one merge ("!" "!" -> "!!")
    toks = byte_unicode() + ["!!", "<|bos|>", "<|eos|>"]
    w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("default")
    w.add_token_list(toks); w.add_token_types([1] * 257 + [3, 3]); w.add_token_merges(["! !"])
    w.add_bos_token_id(257); w.add_eos_token_id(258); w.add_add_bos_token(False)
    n_vocab = len(toks)
    ones = np.ones(N_EMBD, dtype=np.float32)
    # 259 rows is not a multiple of any block row constraint (blocks run along ne0 = n_embd), so every type works.
    mats = [("token_embd.weight", (N_EMBD, n_vocab)), ("output.weight", (N_EMBD, n_vocab))]
    w.add_tensor("output_norm.weight", ones)
    for il in range(n_layer):
        w.add_tensor(f"blk.{il}.attn_norm.weight", ones)
        w.add_tensor(f"blk.{il}.ffn_norm.weight", ones)
        mats += [(f"blk.{il}.attn_q.weight", (N_EMBD, N_HEAD * HD)), (f"blk.{il}.attn_k.weight", (N_EMBD, N_KV * HD)),
                 (f"blk.{il}.attn_v.weight", (N_EMBD, N_KV * HD)), (f"blk.{il}.attn_output.weight", (N_HEAD * HD, N_EMBD)),
                 (f"blk.{il}.ffn_gate.weight", (N_EMBD, n_ff)), (f"blk.{il}.ffn_up.weight", (N_EMBD, n_ff)),
                 (f"blk.{il}.ffn_down.weight", (n_ff, N_EMBD))]
    for name, shape in mats:
        data, qt = tensor_data(name, shape, qname)
        # for a quantized raw_dtype the writer takes the BYTE shape in numpy order: (ne1, bytes per row)
        _, blk, block = BLOCKS[qname]
        w.add_tensor(name, data, raw_shape=(shape[1], shape[0] // blk * len(block)), raw_dtype=qt)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    types = sys.argv[2].split(",") if len(sys.argv) > 2 else list(BLOCKS)
    for q in types:
        for nl in (1, 3):
            p = os.path.join(out, f"calib_{q}_L{nl}.gguf"); write(p, q, nl)
            print(p, os.path.getsize(p))
    # the NARROW 3-layer Q4_0 model (ffn 2048 instead of 8192): same layer count as L3, fewer bytes per layer, so a
    # per-layer fixed cost (thread barriers per op) separates from the per-byte cost:
    #   t = t0 + n_layer * t_layer + bytes / W
    if "Q4_0" in types:
        p = os.path.join(out, "calib_Q4_0_L3N.gguf"); write(p, "Q4_0", 3, n_ff=2048); print(p, os.path.getsize(p))
