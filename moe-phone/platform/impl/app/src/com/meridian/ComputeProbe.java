package com.meridian.app;

import android.content.Context;
import org.json.*;
import java.io.*;
import java.util.*;

/** T2 compute probe (04_DEVICE_PROFILING.md section 3.3; closes stub PL-E11): measures the REAL engine's decode and prefill
 *  cost on this device, per ggml weight type, with the synthetic calibration models bundled in the APK
 *  (tools/make_calib_models.py). Model: t_token = t0 + sum_T bytes_T / W_T (05_PERFORMANCE_MODEL.md section 3.2).
 *   - Q4_0 runs three models: 1 wide layer (A), 3 wide layers (B), 3 narrow layers (C). With t = t0 + n*t_layer + B/W:
 *       W = (B_B - B_C) / (t_B - t_C)          (same layer count: only bytes differ)
 *       t_layer = ((t_B - t_A) - (B_B - B_A)/W) / 2,   t0 = t_A - t_layer - B_A/W
 *     The per-layer term exists because every op in a layer ends in a thread barrier; without it, a model with many
 *     small layers would be predicted too fast (found 2026-09-22 before any user-facing claim).
 *   - W_T for the other types comes from their 3-layer model with t0 and t_layer held fixed.
 *   - every figure is the median over repeats; lo/hi are the observed min/max over repeats (never a multiplier).
 *   - repeats are ABBA-ordered across the arms so slow drift (heat) does not masquerade as a type difference.
 *  If both engine builds ship and the CPU has i8mm, the Q4_0 3-layer arm runs on both and the build with the shorter median
 *  representative turn becomes the device's engine variant: the kernel is chosen by measurement on the target device. */
public final class ComputeProbe {
    public interface Progress { void step(String s); }
    static final String[] TYPES = {"Q4_0", "Q8_0", "Q4_K", "Q6_K", "MXFP4"};
    static final String PROMPT = "Measure this device: the quick brown fox jumps over the lazy dog, again and again, near the quiet river bank at dawn.";   // 118 bytes = 118 tokens (byte-level vocabulary)
    static final int N_DECODE = 48, REPS = 2;
    static final double TURN_PROMPT = 200, TURN_OUT = 64;   // the representative agent turn (design parameter, stated in 00_PROBLEM.md section 2)

    static final class Arm { String type, variant; int layers; boolean narrow; long bytes; List<Double> dec = new ArrayList<>(), pre = new ArrayList<>();
        String file() { return "calib_" + type + "_L" + layers + (narrow ? "N" : "") + ".gguf"; } String key() { return type + "_L" + layers + (narrow ? "N" : "") + "@" + variant; } }

    static File extract(Context ctx, String name) throws IOException {
        File d = new File(ctx.getFilesDir(), "calib_models"); d.mkdirs(); File f = new File(d, name);
        if (f.exists()) return f;
        File tmp = new File(d, name + ".part");
        try (InputStream in = ctx.getAssets().open("calib/" + name); OutputStream o = new BufferedOutputStream(new FileOutputStream(tmp), 1 << 20)) {
            byte[] b = new byte[1 << 20]; int n; while ((n = in.read(b)) > 0) o.write(b, 0, n); }
        if (!tmp.renameTo(f)) throw new IOException("could not finalise " + name);
        return f;
    }

    public static boolean i8mmEngineBundled(Context ctx) { return Native.present(ctx, "bmoe_cli_i8"); }

    public static JSONObject run(Context ctx, JSONObject profile, Progress prog) throws Exception {
        Topo.Placement pl = Topo.effective(profile);
        long free = ctx.getFilesDir().getUsableSpace(); if (free < (1500L << 20)) throw new IOException("NotCalibrated: the compute probe needs ~1.5 GiB free for its calibration files (" + (free >> 20) + " MiB free)");
        boolean tryI8 = i8mmEngineBundled(ctx) && Topo.hasIsa(profile, "i8mm");
        Regime regime = new Regime(ctx); regime.begin(); int[] failures = {0}; JSONArray failLog = new JSONArray();
        String variant = "dot"; JSONObject variantBasis = new JSONObject().put("i8mm_engine_bundled", i8mmEngineBundled(ctx)).put("cpu_has_i8mm", Topo.hasIsa(profile, "i8mm"));
        List<Arm> arms = new ArrayList<>();
        try {
            if (tryI8) {   // phase 1: which engine build is faster on THIS device (same model, same placement, ABBA)
                Arm d3 = arm("Q4_0", 3, "dot"), i3 = arm("Q4_0", 3, "i8");
                measure(ctx, pl, Arrays.asList(d3, i3), REPS, prog, failures, failLog);
                if (!d3.dec.isEmpty() && !i3.dec.isEmpty()) {
                    // objective: the representative turn T_turn = prefill(200) + decode(64) (05_PERFORMANCE_MODEL.md section 3.1),
                    // not decode alone: on the 15R (2026-09-22) the i8mm build decoded 1% faster but prefilled 16% slower.
                    double turnDot = TURN_PROMPT / med(d3.pre) + TURN_OUT / med(d3.dec), turnI8 = TURN_PROMPT / med(i3.pre) + TURN_OUT / med(i3.dec);
                    variant = turnI8 < turnDot ? "i8" : "dot";
                    variantBasis.put("dot_decode_tok_s", new JSONArray(d3.dec)).put("i8_decode_tok_s", new JSONArray(i3.dec)).put("dot_prefill_tok_s", new JSONArray(d3.pre)).put("i8_prefill_tok_s", new JSONArray(i3.pre))
                        .put("turn_s_dot", turnDot).put("turn_s_i8", turnI8).put("rule", "shorter median turn (" + (int) TURN_PROMPT + "-token prompt + " + (int) TURN_OUT + " output tokens) on the calibration model");
                } else variantBasis.put("rule", "a variant arm failed: kept the portable build");
            }
            // phase 2: every type on the chosen build
            arms.add(arm("Q4_0", 1, variant)); arms.add(arm("Q4_0", 3, variant)); Arm nar = arm("Q4_0", 3, variant); nar.narrow = true; arms.add(nar);
            for (String t : TYPES) if (!t.equals("Q4_0")) arms.add(arm(t, 3, variant));
            measure(ctx, pl, arms, REPS, prog, failures, failLog);
        } finally { File d = new File(ctx.getFilesDir(), "calib_models"); File[] fs = d.listFiles(); if (fs != null) for (File f : fs) f.delete(); }
        JSONObject cond = regime.end(); boolean in = cond.getBoolean("in_regime");
        Arm s = find(arms, "Q4_0", 1, variant), l = find(arms, "Q4_0", 3, variant), n = findNarrow(arms, variant);
        if (s.dec.isEmpty() || l.dec.isEmpty() || n.dec.isEmpty()) throw new IllegalStateException("compute probe failed: a Q4_0 arm produced no rows (" + failLog + ")");
        double tA = 1 / med(s.dec), tB = 1 / med(l.dec), tC = 1 / med(n.dec);
        if (!(tB > tC) || !(tB > tA)) throw new IllegalStateException(String.format(Locale.ROOT, "compute probe: per-byte cost not identifiable (1-layer %.1f, 3-layer %.1f, 3-narrow %.1f tok/s): the bigger model was not slower", 1 / tA, 1 / tB, 1 / tC));
        double[] fit = fit(s.bytes, l.bytes, n.bytes, tA, tB, tC); double w = fit[0], tLayer = fit[1], t0 = fit[2];
        boolean t0Neg = t0 < 0, tlNeg = tLayer < 0; double t0c = Math.max(0, t0), tlc = Math.max(0, tLayer);
        JSONObject types = new JSONObject();
        for (String t : TYPES) {
            Arm a = t.equals("Q4_0") ? l : find(arms, t, 3, variant); if (a == null || a.dec.isEmpty()) continue;
            double fixed = t0c + 3 * tlc, tt = 1 / med(a.dec), tFast = 1 / Collections.max(a.dec), tSlow = 1 / Collections.min(a.dec);
            if (!(tt > fixed)) continue;
            types.put(t, new JSONObject().put("w_gbps", a.bytes / (tt - fixed) / 1e9).put("lo", a.bytes / (tSlow - fixed) / 1e9).put("hi", tFast > fixed ? a.bytes / (tFast - fixed) / 1e9 : JSONObject.NULL)
                .put("bytes_per_token", a.bytes).put("decode_tok_s", new JSONArray(a.dec)).put("prefill_tok_s", new JSONArray(a.pre)).put("n", a.dec.size())
                .put("note", t.equals("Q4_0") ? "fitted with the 1-layer and narrow arms" : "t0 and t_layer from the Q4_0 fit"));
        }
        double preTps = med(l.pre);
        JSONObject v = new JSONObject().put("t0_ms", t0c * 1000).put("t0_raw_ms", t0 * 1000).put("t0_clamped", t0Neg).put("t_layer_ms", tlc * 1000).put("t_layer_raw_ms", tLayer * 1000).put("t_layer_clamped", tlNeg)
            .put("q4_0_arms_tok_s", new JSONObject().put("L1", new JSONArray(s.dec)).put("L3", new JSONArray(l.dec)).put("L3N", new JSONArray(n.dec))).put("types", types)
            .put("prefill_gbps_q4_0", preTps * l.bytes / 1e9).put("prefill_tok_s_q4_0_L3", new JSONArray(l.pre))
            .put("engine_variant", variant).put("variant_basis", variantBasis).put("threads", pl.threads).put("cpu_mask", Long.toHexString(pl.computeMask)).put("placement_basis", pl.basis)
            .put("failures", failures[0]).put("failure_log", failLog);
        return new JSONObject().put("value", v).put("provenance", in ? "measured" : "prior").put("confidence", in ? 0.7 : 0.4)
            .put("source", "ComputeProbe: synthetic calibration GGUFs through the real engine, " + REPS + " ABBA repeats, " + N_DECODE + " decoded tokens").put("conditions", cond);
    }
    /** Solve t = t0 + n*t_layer + B/W from arms A (1 wide layer, bytes bA), B (3 wide, bB), C (3 narrow, bC).
     *  Returns {W bytes/s, t_layer s, t0 s}. Pure arithmetic, so the host test can check exact recovery. */
    public static double[] fit(double bA, double bB, double bC, double tA, double tB, double tC) {
        double w = (bB - bC) / (tB - tC), tLayer = ((tB - tA) - (bB - bA) / w) / 2, t0 = tA - tLayer - bA / w;
        return new double[]{w, tLayer, t0};
    }
    static void measure(Context ctx, Topo.Placement pl, List<Arm> arms, int reps, Progress prog, int[] failures, JSONArray failLog) throws Exception {
        for (int r = 0; r < reps; r++) {
            List<Arm> order = new ArrayList<>(arms); if (r % 2 == 1) Collections.reverse(order);   // ABBA
            for (Arm a : order) {
                prog.step("compute probe " + a.key() + " (rep " + (r + 1) + "/" + reps + ")");
                File f = extract(ctx, a.file());
                if (a.bytes == 0) a.bytes = Planner.derive(f).activeBytesPerToken;
                Engine.Config cfg = new Engine.Config(); cfg.model = f; cfg.threads = pl.threads; cfg.cpuMask = pl.computeMask == 0 ? null : Long.toHexString(pl.computeMask);
                cfg.ctx = 512; cfg.ubatch = 256; cfg.chatml = false; cfg.variant = a.variant;
                Chat chat = new Chat(ctx, cfg); android.os.PowerManager pm = (android.os.PowerManager) ctx.getSystemService(Context.POWER_SERVICE);
                try { chat.start(120000); chat.ask("warm", 4, true, null);
                    boolean awakeBefore = pm.isInteractive();
                    chat.ask(PROMPT, N_DECODE, true, null); JSONObject d = chat.lastResult;
                    if (d.optInt("tokens") < N_DECODE / 2) throw new IllegalStateException("only " + d.optInt("tokens") + " tokens decoded");
                    // a row that overlapped screen-off is excluded and counted, never averaged in: on the Nord (2026-09-22) a dozing
                    // repeat ran at 1/5 of the awake rate for the same model (CLAUDE.md section 6.3; the research's autosuspend finding)
                    if (!awakeBefore || !pm.isInteractive()) throw new IllegalStateException("row excluded: the screen was off during the run (Doze throttles the CPU)");
                    a.dec.add(d.getDouble("tok_s")); a.pre.add(d.getDouble("prefill_tps")); }
                catch (Exception e) { failures[0]++; failLog.put(a.key() + ": " + e.getMessage()); }
                finally { chat.close(); }
                Thread.sleep(1500);
            }
        }
    }
    static Arm arm(String t, int layers, String v) { Arm a = new Arm(); a.type = t; a.layers = layers; a.variant = v; return a; }
    static Arm find(List<Arm> as, String t, int layers, String v) { for (Arm a : as) if (a.type.equals(t) && a.layers == layers && a.variant.equals(v) && !a.narrow) return a; return null; }
    static Arm findNarrow(List<Arm> as, String v) { for (Arm a : as) if (a.narrow && a.variant.equals(v)) return a; return null; }
    static double med(List<Double> v) { List<Double> s = new ArrayList<>(v); Collections.sort(s); int n = s.size(); return n % 2 == 1 ? s.get(n / 2) : (s.get(n / 2 - 1) + s.get(n / 2)) / 2; }
}
