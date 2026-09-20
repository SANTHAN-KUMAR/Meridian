package com.meridian.app;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.*;

/** GGUF v2/v3 header reader. Port of platform/impl/meridian/gguf.py: tensor sizes come from dims and the ggml
 *  block table and are cross-checked against the file layout; unknown types or inconsistent layout are refused. */
public final class Gguf {
    public static final class GgufError extends Exception { GgufError(String m) { super(m); } }
    public static final class Tensor { public String name; public long[] dims; public int type; public long offset, bytes; }
    public long version, alignment, fileBytes;
    public Map<String, Object> kv = new LinkedHashMap<>();
    public List<Tensor> tensors = new ArrayList<>();

    private static final long[][] BLOCK = new long[64][];
    static {
        long[][] b = {{0,1,4},{1,1,2},{2,32,18},{3,32,20},{6,32,22},{7,32,24},{8,32,34},{9,32,36},{10,256,84},
                {11,256,110},{12,256,144},{13,256,176},{14,256,210},{15,256,292},{30,1,2},{39,32,17}};
        for (long[] r : b) BLOCK[(int) r[0]] = new long[]{r[1], r[2]};
    }

    private final DataInputStream in;
    private Gguf(DataInputStream in) { this.in = in; }
    private long pos = 0;

    private byte[] take(int n) throws IOException, GgufError {
        byte[] b = new byte[n];
        try { in.readFully(b); } catch (EOFException e) { throw new GgufError("truncated header"); }
        pos += n; return b;
    }
    private ByteBuffer bb(int n) throws IOException, GgufError { return ByteBuffer.wrap(take(n)).order(ByteOrder.LITTLE_ENDIAN); }
    private long u32() throws IOException, GgufError { return bb(4).getInt() & 0xFFFFFFFFL; }
    private long u64() throws IOException, GgufError { return bb(8).getLong(); }
    private String str() throws IOException, GgufError {
        long n = u64(); if (n < 0 || n > (1 << 24)) throw new GgufError("implausible string length " + n);
        return new String(take((int) n), "UTF-8");
    }
    private Object value(int t) throws IOException, GgufError {
        switch (t) {
            case 0: return (long) (take(1)[0] & 0xFF);
            case 1: return (long) take(1)[0];
            case 2: return (long) (bb(2).getShort() & 0xFFFF);
            case 3: return (long) bb(2).getShort();
            case 4: return u32();
            case 5: return (long) bb(4).getInt();
            case 6: return (double) bb(4).getFloat();
            case 7: return take(1)[0] != 0;
            case 10: return u64();
            case 11: return u64();
            case 12: return bb(8).getDouble();
            case 8: return str();
            case 9: {
                int et = (int) u32(); long n = u64();
                if (et == 9) throw new GgufError("nested arrays unsupported");
                for (long i = 0; i < n; i++) value(et);   // tokenizer arrays: skipped, only length kept
                return "array[" + n + "]";
            }
            default: throw new GgufError("unknown value type " + t);
        }
    }

    public static Gguf read(File f) throws IOException, GgufError {
        long size = f.length();
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(f), 1 << 16))) {
            Gguf g = new Gguf(in); g.fileBytes = size;
            byte[] magic = g.take(4);
            if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F') throw new GgufError("not a GGUF file");
            g.version = g.u32();
            if (g.version != 2 && g.version != 3) throw new GgufError("unsupported GGUF version " + g.version);
            long nT = g.u64(), nKv = g.u64();
            for (long i = 0; i < nKv; i++) { String k = g.str(); g.kv.put(k, g.value((int) g.u32())); }
            for (long i = 0; i < nT; i++) {
                Tensor t = new Tensor(); t.name = g.str();
                int nd = (int) g.u32(); t.dims = new long[nd];
                for (int d = 0; d < nd; d++) t.dims[d] = g.u64();
                t.type = (int) g.u32(); t.offset = g.u64(); g.tensors.add(t);
            }
            Object a = g.kv.get("general.alignment");
            g.alignment = a instanceof Long ? (Long) a : 32;
            long dataStart = (g.pos + g.alignment - 1) / g.alignment * g.alignment;
            List<Tensor> order = new ArrayList<>(g.tensors);
            Collections.sort(order, (x, y) -> Long.compare(x.offset, y.offset));
            for (int i = 0; i < order.size(); i++) {
                Tensor t = order.get(i);
                if (t.type < 0 || t.type >= BLOCK.length || BLOCK[t.type] == null) throw new GgufError("tensor " + t.name + ": unknown ggml type " + t.type);
                long be = BLOCK[t.type][0], bbytes = BLOCK[t.type][1], n = 1;
                for (long d : t.dims) n *= d;
                if (n % be != 0) throw new GgufError("tensor " + t.name + ": elements not a multiple of block size");
                t.bytes = n / be * bbytes;
                long start = dataStart + t.offset;
                long end = i + 1 < order.size() ? dataStart + order.get(i + 1).offset : size;
                long gap = end - start;
                if (gap - t.bytes < 0 || gap - t.bytes >= g.alignment)
                    throw new GgufError("tensor " + t.name + ": computed " + t.bytes + " B disagrees with file layout gap " + gap + " B");
            }
            return g;
        }
    }

    public long kvLong(String k, long dflt) { Object o = kv.get(k); return o instanceof Long ? (Long) o : dflt; }
    public String kvStr(String k) { Object o = kv.get(k); return o instanceof String ? (String) o : null; }
}
