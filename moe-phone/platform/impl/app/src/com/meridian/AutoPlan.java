package com.meridian.app;

import android.content.Context;
import org.json.*;
import java.io.File;

/** ConfigurationPlanner for the one-tap path: model file -> ExecutionPlan the Governor and Lease already understand
 *  (same schema as PlanV2.plan). Differences from PlanV2: it does not require per-model calibration cells, because the
 *  prediction comes from the device's compute probe (Predictor); and it may choose the streamed tier (closing stub PL-E25)
 *  when the resident tier does not fit. If calibration cells exist for the chosen configuration they take precedence
 *  (measured > calibrated > prior, 05_PERFORMANCE_MODEL.md section 2). The plan's falsification interval is the
 *  prediction's own interval, so the Governor flags a wrong prediction after 3 of 5 turns fall outside it. */
public final class AutoPlan {
    public static JSONObject plan(Context ctx, JSONObject profile, Planner.Card card, File model) throws JSONException {
        String fs = Profile.filesystemOf(model.getAbsolutePath());
        JSONObject ev = Predictor.evaluate(profile, card, model.getParentFile().getUsableSpace(), true, fs);
        JSONObject out = new JSONObject().put("model_id", card.modelId).put("evaluation", ev);
        if (!"Runs".equals(ev.getString("verdict"))) return out.put("refusal", new JSONObject().put("reason", ev.getString("verdict")).put("detail", ev.optString("detail")));
        String tier = ev.getString("tier"); int cctx = ev.getInt("ctx");
        Engine.Config cfg = config(profile, card, model, tier, cctx, ev.getLong("cache_bytes"));
        JSONObject dec = ev.getJSONObject("decode_tok_s"); String basis = dec.getString("provenance");
        Cells cells = Calibration.load(ctx, cfg);
        if (cells != null && cells.inRegime && !cells.decode.isEmpty()) {   // a measured cell for this exact configuration beats the model
            JSONObject m = Cells.predict(cells.decode, 512, true); dec = new JSONObject().put("value", m.getDouble("value")).put("lo", m.getDouble("lo")).put("hi", m.getDouble("hi")).put("provenance", "measured"); basis = "measured";
        }
        long need = tier.equals("resident") ? ev.getLong("need_resident") : ev.getLong("need_streamed_min");
        out.put("plan_id", Integer.toHexString((card.modelId + tier + System.nanoTime()).hashCode())).put("config", PlanV2.cfgJson(cfg).put("variant", cfg.variant))
            .put("chosen", new JSONObject().put("tier", tier).put("need_lower_bound", need).put("decode_tok_s", dec).put("basis", basis));
        out.put("lease", new JSONObject().put("floor", need).put("target", tier.equals("streamed") ? ev.getLong("grant_bytes") : need).put("purpose", "engine").put("priority", "interactive"));
        double half = (dec.getDouble("hi") - dec.getDouble("lo")) / 2;
        out.put("falsification", new JSONObject().put("metric", "decode_tok_s").put("lo", dec.getDouble("lo")).put("hi", dec.getDouble("hi")).put("tolerance_frac", half / Math.max(1e-9, dec.getDouble("value"))).put("min_observations", 3).put("window", 5));
        out.put("valid_while", new JSONObject().put("thermal_status_max", 1).put("power_state", "unplugged").put("profile_id", profile.getString("profile_id")));
        out.put("ladder", PlanV2.ladder(profile, card, cfg, tier, out.getJSONObject("chosen")));
        return out;
    }

    public static Engine.Config config(JSONObject profile, Planner.Card card, File model, String tier, int ctx, long cacheBytes) throws JSONException {
        Topo.Placement pl = Topo.effective(profile); Engine.Config c = new Engine.Config();
        c.model = model; c.ctx = ctx; c.ubatch = 512; c.threads = pl.threads; c.cpuMask = pl.computeMask == 0 ? null : Long.toHexString(pl.computeMask);
        JSONObject cp = Predictor.compute(profile); if (cp != null) c.variant = cp.getJSONObject("value").optString("engine_variant", "dot");
        if (tier.equals("streamed")) {
            // --cache-floor-mb is "RAM to leave free" when the engine sizes the cache (bmoe-cli --help), NOT a minimum cache size:
            // 1024 MiB is the value the research adopted on both phones (results/2026-09-19 configs). The minimum useful cache
            // (one full token cycle) is enforced at plan time instead (Predictor.evaluate needs it to fit).
            c.moeStream = true; c.cacheFloorMb = 1024; c.cacheCeilMb = (int) Math.max(PlanV2.cycleBytes(card) / Predictor.MIB + 1, cacheBytes / Predictor.MIB);
            c.ioThreads = PlanV2.ioThreads(profile); c.ioMask = pl.ioMask == 0 ? null : Long.toHexString(pl.ioMask);
        }
        return c;
    }
}
