"""Minimal GGUF v2/v3 header reader (format: llama.cpp docs/gguf.md).

Reads metadata KVs and tensor infos only, never tensor data. Tensor byte
sizes are computed from dims and the ggml block table, then cross-checked
against the file layout (offset gaps differ from true size only by alignment
padding). Refuses (GgufError) on anything it does not understand rather than
guessing.
"""
from __future__ import annotations

import os
import struct

MAGIC = b"GGUF"
# value type ids per the GGUF spec
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9


class GgufError(ValueError):
    pass


# ggml type id -> (elements per block, bytes per block), from ggml-common.h /
# ggml.h. Unknown ids are refused: a wrong size here would silently corrupt
# every byte budget derived from it.
BLOCK = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
         9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
         15: (256, 292), 30: (1, 2), 39: (32, 17)}


class _R:
    def __init__(self, f):
        self.f = f

    def take(self, fmt):
        n = struct.calcsize(fmt)
        b = self.f.read(n)
        if len(b) != n:
            raise GgufError("truncated header")
        return struct.unpack(fmt, b)[0]

    def string(self):
        n = self.take("<Q")
        if n > 1 << 24:
            raise GgufError(f"implausible string length {n}")
        b = self.f.read(n)
        if len(b) != n:
            raise GgufError("truncated string")
        return b.decode("utf-8", "replace")

    def value(self, t, keep_array=False):
        if t in _SCALAR:
            return self.take(_SCALAR[t])
        if t == _STRING:
            return self.string()
        if t == _ARRAY:
            et, n = self.take("<I"), self.take("<Q")
            if et == _ARRAY:
                raise GgufError("nested arrays unsupported")
            items = [self.value(et) for _ in range(n)]  # tokenizer arrays are large but finite
            return items if keep_array else {"array_len": n}
        raise GgufError(f"unknown value type {t}")


def read(path: str) -> dict:
    """Returns {"version", "alignment", "kv": {...}, "tensors": [{name, dims, type, offset, bytes}], "file_bytes"}."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise GgufError("not a GGUF file")
        r = _R(f)
        version = r.take("<I")
        if version not in (2, 3):
            raise GgufError(f"unsupported GGUF version {version}")
        n_tensors, n_kv = r.take("<Q"), r.take("<Q")
        kv = {}
        for _ in range(n_kv):
            k = r.string()
            kv[k] = r.value(r.take("<I"))
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.take("<I")
            dims = [r.take("<Q") for _ in range(nd)]
            ttype, off = r.take("<I"), r.take("<Q")
            tensors.append({"name": name, "dims": dims, "type": ttype, "offset": off})
        align = int(kv.get("general.alignment", 32))
        data_start = (f.tell() + align - 1) // align * align
    order = sorted(range(len(tensors)), key=lambda i: tensors[i]["offset"])
    for pos, i in enumerate(order):
        t = tensors[i]
        if t["type"] not in BLOCK:
            raise GgufError(f"tensor {t['name']}: unknown ggml type {t['type']}")
        be, bb = BLOCK[t["type"]]
        n = 1
        for d in t["dims"]:
            n *= d
        if n % be:
            raise GgufError(f"tensor {t['name']}: {n} elements not a multiple of block size {be}")
        t["bytes"] = n // be * bb
        start = data_start + t["offset"]
        end = data_start + tensors[order[pos + 1]]["offset"] if pos + 1 < len(order) else size
        gap = end - start
        # independent cross-check: file layout must leave room for the tensor,
        # with slack of less than one alignment unit (padding).
        if not (0 <= gap - t["bytes"] < align):
            raise GgufError(f"tensor {t['name']}: computed {t['bytes']} B disagrees with file layout gap {gap} B "
                            f"(truncated/corrupt file or wrong block table)")
    return {"version": version, "alignment": align, "kv": kv, "tensors": tensors, "file_bytes": size}
