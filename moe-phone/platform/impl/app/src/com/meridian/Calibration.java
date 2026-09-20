package com.meridian.app;

import android.content.Context;
import org.json.*;
import java.io.*;
import java.util.*;

/** Runs the calibration cells for one (model, engine config) through the real engine, in the sampled regime, and stores them.
 *  Each run reports prefill rate at its prompt length AND decode rate at its resulting context, so one pass yields both curves.
 *  Closes the measurement half of stub PL-S3 (prefill vs prompt length). */
public final class Calibration {
    public interface Progress { void step(String s); }
    static final String UNIT = "The quick brown fox jumps over the lazy dog near the quiet river bank. ";

    public static String key(Engine.Config c) { return c.model.getName() + "__" + (c.moeStream ? "streamed" : "resident") + "_t" + c.threads + "_m" + c.cpuMask + "_c" + c.ctx; }
    public static File file(Context ctx, Engine.Config c) { File d = new File(ctx.getFilesDir(), "calib"); d.mkdirs(); return new File(d, key(c).replaceAll("[^A-Za-z0-9._-]", "_") + ".json"); }
    public static Cells load(Context ctx, Engine.Config c) { File f = file(ctx, c); if (!f.exists()) return null; try { return Cells.fromJson(new JSONObject(Native.readFile(f.getAbsolutePath()))); } catch (Exception e) { return null; } }

    static String prompt(int approxTokens) { StringBuilder sb = new StringBuilder("Read this text and answer with one word: "); for (int i = 0; i < Math.max(1, approxTokens / 16); i++) sb.append(UNIT); return sb.append("Word:").toString(); }

    /** targets: approximate prompt-token counts to measure; reps: repeats per target. */
    public static Cells run(Context ctx, Engine.Config cfg, int[] targets, int reps, Progress prog) throws Exception {
        Regime regime = new Regime(ctx); regime.begin();
        Chat chat = new Chat(ctx, cfg); List<double[]> pre = new ArrayList<>(), dec = new ArrayList<>(), tmap = new ArrayList<>(); double loadS;
        try {
            prog.step("loading " + cfg.describe()); chat.start(600000); loadS = chat.engine().readyInfo().optDouble("load_s", -1);
            prog.step("warm-up (not recorded)"); chat.ask("Hello", 8, true, null);
            for (int r = 0; r < reps; r++) for (int t : targets) {
                prog.step("cell prompt~" + t + " tokens, rep " + (r + 1) + "/" + reps);
                chat.ask(prompt(t), 24, true, null); JSONObject d = chat.lastResult;
                double n = d.getDouble("n_prompt"); tmap.add(new double[]{t, n}); pre.add(new double[]{n, d.getDouble("prefill_tps")}); dec.add(new double[]{n + d.getDouble("tokens") / 2, d.getDouble("tok_s")});
            }
        } finally { chat.close(); }
        JSONObject cond = regime.end();
        Cells c = new Cells(); c.prefill.addAll(Cells.group(pre)); c.decode.addAll(Cells.group(dec)); c.inRegime = cond.getBoolean("in_regime"); c.loadS = loadS;
        { double sx = 0, sy = 0, sxx = 0, sxy = 0; int m = tmap.size(); for (double[] p : tmap) { sx += p[0]; sy += p[1]; sxx += p[0] * p[0]; sxy += p[0] * p[1]; } double den = m * sxx - sx * sx; if (den != 0) { c.mapB = (m * sxy - sx * sy) / den; c.mapA = (sy - c.mapB * sx) / m; } }
        JSONObject j = c.toJson().put("conditions", cond).put("config", cfg.describe()).put("created_at", System.currentTimeMillis() / 1000.0).put("raw_prefill", new JSONArray(toJ(pre))).put("raw_decode", new JSONArray(toJ(dec)));
        try (FileWriter w = new FileWriter(file(ctx, cfg))) { w.write(j.toString(2)); }
        return c;
    }
    static String toJ(List<double[]> l) { StringBuilder s = new StringBuilder("["); for (int i = 0; i < l.size(); i++) { if (i > 0) s.append(','); s.append('[').append(l.get(i)[0]).append(',').append(l.get(i)[1]).append(']'); } return s.append(']').toString(); }
}
