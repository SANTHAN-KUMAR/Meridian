"""
truncate_gguf.py — research spec E7: a copy of a GGUF model keeping only its first L transformer blocks.

The embedding table, the final norm and the output head are kept, and block_count is set to L. Every kept tensor is copied
byte for byte (same type, same shape, same quantised bytes), so a kept layer computes exactly what it computes in the full
model. The output is meaningless as a language model; it is a timing specimen of Qwen3-30B-A3B's own layers, small enough to
be fully resident on the phone's GPU. No inference runs here (it only copies bytes; safe on the laptop).
  PYTHONPATH=<llama.cpp>/gguf-py python truncate_gguf.py IN.gguf OUT.gguf L
"""
import sys

import gguf


def main():
    src, dst, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
    r = gguf.GGUFReader(src)
    arch = r.fields["general.architecture"].contents()
    w = gguf.GGUFWriter(dst, arch=arch)
    n_blk_key = f"{arch}.block_count"
    for name, f in r.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        if name == n_blk_key:
            w.add_uint32(name, L)
            continue
        t = f.types[0]
        if t == gguf.GGUFValueType.ARRAY:
            sub = f.types[-1]
            w.add_key_value(name, f.contents(), t, sub_type=sub)
        else:
            w.add_key_value(name, f.contents(), t)
    kept = 0
    for t in r.tensors:
        nm = t.name
        if nm.startswith("blk."):
            if int(nm.split(".")[1]) >= L:
                continue
        # raw bytes with the reader's array shape (bytes per row for quantised types), as llama.cpp's gguf_new_metadata.py does
        w.add_tensor(nm, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
        kept += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    print(f"kept {kept} tensors, {L} blocks, arch {arch} -> {dst}")


if __name__ == "__main__":
    main()
