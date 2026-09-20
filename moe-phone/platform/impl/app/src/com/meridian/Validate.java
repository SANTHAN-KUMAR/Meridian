package com.meridian.app;

import android.content.Context;
import org.json.*;
import java.io.*;
import java.security.MessageDigest;
import java.util.*;

/** Planner validation protocol (05_PERFORMANCE_MODEL.md section 7.1): (1) calibrate on some cells, (2) PRE-REGISTER predictions for held-out cells
 *  (written and hashed before those cells run), (3) run the held-out cells, (4) report error per predictor. Predictors: the measured/interpolating
 *  model, and the three mandatory floors of 01_RESEARCH.md 4.1 (spec-sheet calculator, best constant, one-parameter shrinkage of the calculator).
 *  Scope limit, stated in the output: one device and one model, so this tests held-out CONTEXT LENGTHS, not held-out devices (RQ1 needs >= 3 devices). */
public final class Validate {
    public interface Progress { void step(String s); }
    // External figure, NOT measured here: SoC memory nameplate bandwidth for the SM7250 (LPDDR4X, 2133 MHz clock, 2 x 16-bit channels:
    // 2133e6 x 2 (DDR) x 4 B = 17.06 GB/s). Recalled from the vendor spec sheet, unverified on 2026-09-20; it only feeds the baseline calculator.
    static final double NAMEPLATE_GBPS = 17.06;

    static double spec(Planner.Card c) { return NAMEPLATE_GBPS * 1e9 / c.activeBytesPerToken; }

    public static JSONObject run(Context ctx, Engine.Config cfg, Planner.Card card, JSONObject profile, Progress prog) throws Exception {
        int[] calT = {32, 128, 512}, heldT = {64, 256, 800};
        prog.step("calibration on prompt~" + Arrays.toString(calT)); Cells cells = Calibration.run(ctx, cfg, calT, 2, prog::step);
        if (!cells.inRegime) throw new IllegalStateException("calibration was not taken in the deployment regime; a validation against out-of-regime cells is not reportable");
        // held-out predictions. Each held-out prompt's actual token count is unknown until run, so predict at the target and keep the target's token estimate.
        double s = spec(card); List<Double> allPre = new ArrayList<>(), allDec = new ArrayList<>(), ratios = new ArrayList<>();
        for (Cells.Pt p : cells.prefill) { allPre.add(p.v); ratios.add(p.v / s); } for (Cells.Pt p : cells.decode) { allDec.add(p.v); ratios.add(p.v / s); }
        Collections.sort(allPre); Collections.sort(allDec); Collections.sort(ratios); double k = ratios.get(ratios.size() / 2), cPre = allPre.get(allPre.size() / 2), cDec = allDec.get(allDec.size() / 2);
        JSONArray pre = new JSONArray();
        for (int t : heldT) {
            double np = cells.promptTokens(t); JSONObject pp = Cells.predict(cells.prefill, np, true), dp = Cells.predict(cells.decode, np + 12, true);
            pre.put(new JSONObject().put("target_prompt_tokens", t)
                .put("prefill", new JSONObject().put("ours", pp).put("spec_sheet", s).put("best_constant", cPre).put("shrinkage", k * s))
                .put("decode", new JSONObject().put("ours", dp).put("spec_sheet", s).put("best_constant", cDec).put("shrinkage", k * s)));
        }
        JSONObject prereg = new JSONObject().put("written_at", System.currentTimeMillis() / 1000.0).put("model", card.modelId).put("config", cfg.describe()).put("calibration_targets", new JSONArray(Arrays.toString(calT)))
            .put("shrinkage_k_fitted_on_calibration_cells_only", k).put("prompt_token_map_measured", new JSONObject().put("a", cells.mapA).put("b", cells.mapB)).put("nameplate_gbps_external_unverified", NAMEPLATE_GBPS).put("predictions", pre).put("scope_limit", "one device, one model: held-out context lengths only, not held-out devices");
        String txt = prereg.toString(2); File pf = new File(ctx.getFilesDir(), "prereg.json"); try (FileWriter w = new FileWriter(pf)) { w.write(txt); }
        String sha; { MessageDigest md = MessageDigest.getInstance("SHA-256"); StringBuilder h = new StringBuilder(); for (byte b : md.digest(txt.getBytes("UTF-8"))) h.append(String.format("%02x", b)); sha = h.toString(); }
        try (FileWriter w = new FileWriter(new File(ctx.getFilesDir(), "prereg.sha256"))) { w.write(sha); }
        prog.step("PRE-REGISTERED (sha256 " + sha.substring(0, 12) + "); now running held-out cells");
        Regime regime = new Regime(ctx); regime.begin(); Chat chat = new Chat(ctx, cfg); JSONArray res = new JSONArray();
        try {
            chat.start(600000); chat.ask("Hello", 8, true, null);
            for (int i = 0; i < heldT.length; i++) for (int r = 0; r < 2; r++) {
                prog.step("held-out prompt~" + heldT[i] + " rep " + (r + 1)); chat.ask(Calibration.prompt(heldT[i]), 24, true, null); JSONObject d = chat.lastResult;
                res.put(new JSONObject().put("target", heldT[i]).put("n_prompt", d.getDouble("n_prompt")).put("prefill_tps", d.getDouble("prefill_tps")).put("decode_tok_s", d.getDouble("tok_s")));
            }
        } finally { chat.close(); }
        JSONObject cond = regime.end();
        // errors
        String[] names = {"ours", "spec_sheet", "best_constant", "shrinkage"}; JSONObject err = new JSONObject();
        for (String metric : new String[]{"prefill", "decode"}) { JSONObject m = new JSONObject();
            for (String nm : names) { List<Double> e = new ArrayList<>(); int inside = 0, tot = 0;
                for (int j = 0; j < res.length(); j++) { JSONObject o = res.getJSONObject(j); int ti = Arrays.binarySearch(heldT, o.getInt("target")); JSONObject pj = pre.getJSONObject(ti).getJSONObject(metric);
                    double obs = o.getDouble(metric.equals("prefill") ? "prefill_tps" : "decode_tok_s"), pred = nm.equals("ours") ? pj.getJSONObject("ours").getDouble("value") : pj.getDouble(nm);
                    e.add(Math.abs(pred - obs) / obs); if (nm.equals("ours")) { tot++; JSONObject po = pj.getJSONObject("ours"); if (obs >= po.getDouble("lo") && obs <= po.getDouble("hi")) inside++; } }
                Collections.sort(e); JSONObject r = new JSONObject().put("median_abs_rel_error", e.get(e.size() / 2)).put("max_abs_rel_error", e.get(e.size() - 1)); if (nm.equals("ours")) r.put("interval_coverage", inside + "/" + tot); m.put(nm, r); }
            err.put(metric, m); }
        // oracle check (01_RESEARCH.md 4.2): does the measured placement table distinguish planners at all?
        JSONObject oracle = new JSONObject(); JSONObject cm = profile.getJSONObject("cpu").getJSONObject("recommended_compute_mask");
        if (!cm.isNull("value") && cm.has("arms")) { JSONArray arms = cm.getJSONArray("arms"); double best = 0, dflt = -1; for (int i = 0; i < arms.length(); i++) { JSONObject a = arms.getJSONObject(i); best = Math.max(best, a.getDouble("tok_s_median")); if (a.getString("name").startsWith("unpinned")) dflt = a.getDouble("tok_s_median"); }
            oracle.put("oracle_best_arm_tok_s", best).put("best_constant_arm_tok_s (unpinned default)", dflt).put("oracle_advantage", dflt > 0 ? best / dflt : JSONObject.NULL).put("note", "an oracle advantage near 1.0 would mean this benchmark cannot distinguish planners"); }
        JSONObject out = new JSONObject().put("prereg_sha256", sha).put("results", res).put("errors", err).put("oracle_check", oracle).put("conditions", cond).put("model", card.modelId).put("config", cfg.describe());
        try (FileWriter w = new FileWriter(new File(ctx.getFilesDir(), "validation.json"))) { w.write(out.toString(2)); }
        return out;
    }
}
