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
    public static final String[] TYPE_NAMES = new String[64];
    static { String[] n = {"F32","F16","Q4_0","Q4_1",null,null,"Q5_0","Q5_1","Q8_0","Q8_1","Q2_K","Q3_K","Q4_K","Q5_K","Q6_K","Q8_K","IQ2_XXS","IQ2_XS","IQ3_XXS","IQ1_S","IQ4_NL","IQ3_S","IQ2_S","IQ4_XS","I8","I16","I32","I64","F64","IQ1_M","BF16",null,null,null,"TQ1_0","TQ2_0",null,null,null,"MXFP4","NVFP4","Q1_0","Q2_0"};
        for (int i = 0; i < n.length; i++) TYPE_NAMES[i] = n[i]; }
    public static String typeName(int t) { return t >= 0 && t < TYPE_NAMES.length && TYPE_NAMES[t] != null ? TYPE_NAMES[t] : "type" + t; }
    public long version, alignment, fileBytes;
    public Map<String, Object> kv = new LinkedHashMap<>();
    public List<Tensor> tensors = new ArrayList<>();

    private static final long[][] BLOCK = new long[64][];
    static {
        // {type id, elements per block, bytes per block}: ggml.h type ids, sizes from gguf-py GGML_QUANT_SIZES (same source tree).
        // Q8_1 was listed as 36 B before 2026-09-22; ggml says 40 B (never seen in a model file, so no verdict changed).
        long[][] b = {{0,1,4},{1,1,2},{2,32,18},{3,32,20},{6,32,22},{7,32,24},{8,32,34},{9,32,40},{10,256,84},
                {11,256,110},{12,256,144},{13,256,176},{14,256,210},{15,256,292},{16,256,66},{17,256,74},{18,256,98},
                {19,256,50},{20,32,18},{21,256,110},{22,256,82},{23,256,136},{24,1,1},{25,1,2},{26,1,4},{27,1,8},{28,1,8},
                {29,256,56},{30,1,2},{34,256,54},{35,256,66},{39,32,17},{40,64,36},{41,128,18},{42,64,18}};
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
                boolean ints = et <= 5 || et == 10 || et == 11;
                if (ints && n <= 1024) {   // small integer arrays are per-layer metadata (e.g. head_count_kv on hybrid stacks): kept
                    long[] a = new long[(int) n]; for (int i = 0; i < n; i++) a[i] = (Long) value(et); return a;
                }
                for (long i = 0; i < n; i++) value(et);   // tokenizer arrays: skipped, only length kept
                return "array[" + n + "]";
            }
            default: throw new GgufError("unknown value type " + t);
        }
    }

    public static Gguf read(File f) throws IOException, GgufError {
        try (InputStream in = new FileInputStream(f)) { return parse(in, f.length()); }
    }

    /** Parse a GGUF header from a stream positioned at byte 0 of a file of `size` bytes. Only the header is read, so a
     *  remote file can be evaluated from its first few MiB (RemoteGguf); tensor data is never touched. The layout
     *  cross-check needs only the offsets and the total size, so it runs identically on a partial download. */
    public static Gguf parse(InputStream raw, long size) throws IOException, GgufError {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(raw, 1 << 16))) {
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

    /** In-memory construction for tests (no file, no layout check): lets a gate feed the planner a malformed header. */
    public static final class Builder { private final Gguf g = new Gguf(null);
        public Builder kv(String k, Object v) { g.kv.put(k, v); return this; }
        public Builder tensor(String name, long[] dims, int type, long bytes) { Tensor t = new Tensor(); t.name = name; t.dims = dims; t.type = type; t.bytes = bytes; g.tensors.add(t); return this; }
        public Gguf build() { return g; } }

    public long kvLong(String k, long dflt) { Object o = kv.get(k); return o instanceof Long ? (Long) o : dflt; }
    /** Scalar, or the maximum of a per-layer array (an upper bound, used for KV sizing), or dflt. */
    public long kvLongMax(String k, long dflt) { Object o = kv.get(k); if (o instanceof Long) return (Long) o;
        if (o instanceof long[] && ((long[]) o).length > 0) { long m = Long.MIN_VALUE; for (long v : (long[]) o) m = Math.max(m, v); return m; } return dflt; }
    public String kvStr(String k) { Object o = kv.get(k); return o instanceof String ? (String) o : null; }
}
