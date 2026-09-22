"""
elf_rename.py - give a second engine build distinct library names so two builds fit in ONE APK (closes stub PL-E23).

Android extracts every lib/<abi>/lib*.so into one flat directory, so the i8mm build's libggml*.so / libllama*.so would
overwrite the portable build's. This rewrites DT_NEEDED and DT_SONAME entries in place with SAME-LENGTH names (so no
section moves and no offsets change). It refuses, rather than guesses, if a name is not found exactly at the dynamic
entry's string offset or if that string is a shared suffix of another string (the byte before it is not NUL).

  python3 elf_rename.py IN OUT   (applies MAP; verify with llvm-readelf -d OUT)
"""
import struct, sys

MAP = {"libggml-base.so": "libg8ml-base.so", "libggml-cpu.so": "libg8ml-cpu.so", "libggml.so": "libg8ml.so",
       "libllama.so": "libl8ama.so", "libllama-common.so": "libl8ama-common.so"}
for k, v in MAP.items():
    assert len(k) == len(v), (k, v)
DT_NEEDED, DT_STRTAB, DT_SONAME = 1, 5, 14


def rename(src, dst):
    b = bytearray(open(src, "rb").read())
    assert b[:4] == b"\x7fELF" and b[4] == 2 and b[5] == 1, "ELF64 little-endian only"
    e_phoff, = struct.unpack_from("<Q", b, 0x20); e_phentsize, e_phnum = struct.unpack_from("<HH", b, 0x36)
    phdrs = [struct.unpack_from("<IIQQQQQQ", b, e_phoff + i * e_phentsize) for i in range(e_phnum)]
    dyn = [p for p in phdrs if p[0] == 2]  # PT_DYNAMIC
    assert len(dyn) == 1, "no single PT_DYNAMIC"
    _, _, d_off, d_vaddr, _, d_filesz, _, _ = dyn[0]
    ents = [struct.unpack_from("<qQ", b, d_off + i * 16) for i in range(d_filesz // 16)]
    strtab_vaddr = next(v for t, v in ents if t == DT_STRTAB)
    # vaddr -> file offset through the PT_LOAD that contains it
    load = next(p for p in phdrs if p[0] == 1 and p[3] <= strtab_vaddr < p[3] + p[5])
    strtab_off = strtab_vaddr - load[3] + load[2]
    changed = []
    for t, v in ents:
        if t not in (DT_NEEDED, DT_SONAME):
            continue
        o = strtab_off + v; end = b.index(0, o); name = b[o:end].decode()
        if name not in MAP:
            continue
        if b[o - 1] != 0:
            raise SystemExit(f"{src}: '{name}' is a shared suffix at 0x{o:x}; refusing to rewrite")
        b[o:end] = MAP[name].encode(); changed.append((t, name))
    open(dst, "wb").write(b)
    return changed


if __name__ == "__main__":
    ch = rename(sys.argv[1], sys.argv[2])
    print(sys.argv[2], ch)
