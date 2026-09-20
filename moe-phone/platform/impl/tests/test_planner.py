"""Planner/GGUF tests. Expected values are fixed by arithmetic on the GGUF
format and ggml block sizes, independent of the code under test."""
import os, struct, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from meridian import gguf, planner

def _s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
def _kv_u32(k, v): return _s(k) + struct.pack("<II", 4, v)
def _kv_str(k, v): return _s(k) + struct.pack("<I", 8) + _s(v)

def build_gguf(path, arch="olmoe", n_layer=2, n_exp=4, used=2, hidden=64, ff=32, tied=False):
    """Tensors are 'Q8_0-like': bytes = elems*34/32. Alignment 32."""
    kvs = [_kv_str("general.architecture", arch), _kv_u32(f"{arch}.block_count", n_layer),
           _kv_u32(f"{arch}.expert_count", n_exp), _kv_u32(f"{arch}.expert_used_count", used),
           _kv_u32(f"{arch}.context_length", 512), _kv_u32(f"{arch}.embedding_length", hidden),
           _kv_u32(f"{arch}.attention.head_count", 4), _kv_u32(f"{arch}.attention.head_count_kv", 2)]
    specs = [("token_embd.weight", [hidden, 100])]
    if not tied: specs.append(("output.weight", [hidden, 100]))
    for l in range(n_layer):
        specs.append((f"blk.{l}.attn_q.weight", [hidden, hidden]))
        for k in ("gate", "up", "down"): specs.append((f"blk.{l}.ffn_{k}_exps.weight", [hidden, ff, n_exp]))
    sizes, off, infos = {}, 0, b""
    for name, dims in specs:
        n = 1
        for d in dims: n *= d
        nb = n * 34 // 32
        pad = (-nb) % 32
        infos += _s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", 8, off)
        sizes[name] = nb; off += nb + pad
    head = b"GGUF" + struct.pack("<IQQ", 3, len(specs), len(kvs)) + b"".join(kvs) + infos
    head += b"\0" * ((-len(head)) % 32)
    with open(path, "wb") as f:
        f.write(head)
        for name, _ in specs:
            f.write(b"\1" * sizes[name]); f.write(b"\0" * ((-sizes[name]) % 32))
    return sizes

def test_card_arithmetic_hermetic():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.gguf"); sizes = build_gguf(p)
        c = planner.derive_card(p)
        exp = sum(v for k, v in sizes.items() if "_exps" in k)
        assert c.total_bytes == sum(sizes.values())
        assert c.expert_bytes == exp
        assert c.resident_bytes == sum(v for k, v in sizes.items() if "_exps" not in k)
        assert c.expert_slice_bytes["ffn_gate_exps.weight"] == [64 * 32 * 34 // 32] * 2
        # untied head: embedding is a lookup, excluded from active bytes
        assert c.active_bytes_per_token == c.resident_bytes - sizes["token_embd.weight"] + exp * 2 // 4
        # kv f16: layers 2 * kv_heads 2 * (hd 16 + 16) * 2 bytes
        assert c.kv_bytes_per_token["f16"] == 2 * 2 * 32 * 2

def test_tied_head_counts_embedding_as_active():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.gguf"); sizes = build_gguf(p, tied=True)
        c = planner.derive_card(p)
        assert c.active_bytes_per_token == c.resident_bytes + c.expert_bytes * 2 // 4

def test_unknown_architecture_is_refused_not_guessed():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.gguf"); build_gguf(p, arch="mystery-moe")
        try: planner.derive_card(p); assert False
        except planner.Refusal as r: assert r.reason == "ArchitectureUnsupported"

def test_truncated_file_refused():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.gguf"); build_gguf(p)
        data = open(p, "rb").read(); open(p, "wb").write(data[:60])
        try: gguf.read(p); assert False
        except gguf.GgufError: pass

def _profile(grant, prov="measured"):
    return {"memory": {"grantable_quiesced": {"value": grant, "provenance": prov}}}

def test_plan_never_says_feasible_and_names_missing():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.gguf"); build_gguf(p); c = planner.derive_card(p)
        r = planner.plan(_profile(10**9), c, 256)
        for v in r["tiers"].values():
            assert v["verdict"] == "NotCalibrated"
            assert any("grantable_foreground" in m for m in v["missing"])

def test_infeasible_is_a_sound_lower_bound_refusal():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.gguf"); build_gguf(p); c = planner.derive_card(p)
        try: planner.plan(_profile(1000), c, 256); assert False
        except planner.Refusal as r: assert r.reason == "Infeasible"
        # resident infeasible but streamed passes -> only resident refused
        kv = 2 * 2 * 32 * 2 * 256                    # layers*kv_heads*(hd+hdv)*2B * ctx
        streamed_need = c.resident_bytes + kv + 3 * (64 * 32 * 34 // 32) * 2
        assert streamed_need < c.total_bytes + kv
        grant = streamed_need
        r = planner.plan(_profile(grant), c, 256)
        assert r["tiers"]["resident"]["verdict"] == "Infeasible" and r["tiers"]["streamed"]["verdict"] == "NotCalibrated"

def test_out_of_range_context_and_unmeasured_grant_refused():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.gguf"); build_gguf(p); c = planner.derive_card(p)
        for prof, ctx, code in [(_profile(10**9), 10**6, "OutOfRange"), (_profile(None, "unknown"), 8, "NotCalibrated")]:
            try: planner.plan(prof, c, ctx); assert False
            except planner.Refusal as r: assert r.reason == code

REAL = "/run/media/santhankumar/New Volume/moe-work/models/olmoe-1b-7b-0924-q4_0.gguf"
def test_real_olmoe_slice_matches_format_arithmetic():
    if not os.path.exists(REAL): return  # host-specific fixture; hermetic tests above cover the logic
    c = planner.derive_card(REAL)
    assert c.expert_slice_bytes["ffn_gate_exps.weight"] == [2048 * 1024 * 18 // 32] * 2  # Q4_0: 32 elems / 18 B
    assert c.kv_bytes_per_token["f16"] == 16 * 16 * 256 * 2

if __name__ == "__main__":
    fails = 0
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            try: v(); print("PASS", k)
            except Exception as e: fails += 1; print("FAIL", k, repr(e))
    print("failed:", fails); raise SystemExit(bool(fails))
