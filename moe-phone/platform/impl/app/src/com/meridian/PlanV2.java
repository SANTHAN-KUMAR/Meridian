package com.meridian.app;

import android.content.Context;
import org.json.*;
import java.io.File;
import java.util.*;

/** ConfigurationPlanner (05_PERFORMANCE_MODEL.md section 6): enumerate {resident, streamed}, prune by measured feasibility,
 *  predict from calibration cells with intervals, decide lexicographically, and emit an ExecutionPlan carrying its ladder
 *  (computed now, from measured data) and its own falsification rule. Refusal is a first-class outcome. */
public final class PlanV2 {
    public static final long MIB = 1048576L;

    /** Bytes of the worst-case token cycle: every layer's routed-expert entries (all projections) at the applied top-k
     *  (docs/cache-sizing.md: a cache budget below this gives exactly 0% hit rate, not a low one). */
    public static long cycleBytes(Planner.Card c) { return c.sliceSumMax * c.nUsed * c.nLayer; }

    static String isaRefusal(JSONObject profile) throws JSONException {
        JSONArray cl = profile.getJSONObject("cpu").getJSONArray("clusters"); boolean dot = false, fp16 = false;
        for (int i = 0; i < cl.length(); i++) { String f = cl.getJSONObject(i).getJSONArray("isa_features").toString(); dot |= f.contains("asimddp"); fp16 |= f.contains("asimdhp") || f.contains("fphp"); }
        return dot && fp16 ? null : "EngineUnsupported: bundled engine needs armv8.2-a+dotprod+fp16 (dotprod=" + dot + ", fp16=" + fp16 + ")";
    }

    /** Base engine config from measured placement (or an explicitly-labelled default). */
    public static Engine.Config baseConfig(JSONObject profile, File model, int ctx) throws JSONException {
        Engine.Config cfg = new Engine.Config(); cfg.model = model; cfg.ctx = ctx; cfg.ubatch = 512;
        JSONObject cm = profile.getJSONObject("cpu").getJSONObject("recommended_compute_mask");
        if (!cm.isNull("value")) { JSONObject v = cm.getJSONObject("value"); cfg.threads = v.getInt("threads"); if (!v.isNull("mask_hex")) cfg.cpuMask = v.getString("mask_hex"); }
        return cfg;
    }
    static long allMask(JSONObject profile) throws JSONException { long m = 0; JSONArray cl = profile.getJSONObject("cpu").getJSONArray("clusters");
        for (int i = 0; i < cl.length(); i++) { JSONArray ids = cl.getJSONObject(i).getJSONArray("core_ids"); for (int k = 0; k < ids.length(); k++) m |= 1L << ids.getInt(k); } return m; }

    /** I/O lanes: the smallest thread count whose measured MB/s at the efficient request size is within 90% of the best measured count. */
    static int ioThreads(JSONObject profile) throws JSONException {
        JSONObject st = profile.getJSONObject("storage"); JSONObject eff = st.getJSONObject("efficient_request_size"); if (eff.isNull("value")) return 4;
        long size = eff.getLong("value"); JSONArray rr = st.getJSONArray("random_read"); TreeMap<Integer, Double> byT = new TreeMap<>();
        for (int i = 0; i < rr.length(); i++) { JSONObject r = rr.getJSONObject(i); if (r.getLong("size_bytes") == size) byT.put(r.getInt("threads"), r.getJSONObject("mbps").getDouble("value")); }
        if (byT.isEmpty()) return 4; double best = Collections.max(byT.values()); for (Map.Entry<Integer, Double> e : byT.entrySet()) if (e.getValue() >= 0.9 * best) return e.getKey(); return 4;
    }

    public static Engine.Config configFor(JSONObject profile, Planner.Card card, File model, String tier, int ctx, long grantBytes) throws JSONException {
        Engine.Config cfg = baseConfig(profile, model, ctx);
        if (tier.equals("streamed")) {
            long kv = card.kvF16PerToken * ctx, floor = cycleBytes(card), ceil = grantBytes - card.residentBytes - kv;
            cfg.moeStream = true; cfg.cacheFloorMb = (int) ((floor + MIB - 1) / MIB); cfg.cacheCeilMb = (int) Math.max(cfg.cacheFloorMb, ceil / MIB); cfg.ioThreads = ioThreads(profile);
            if (cfg.cpuMask != null) cfg.ioMask = Long.toHexString(allMask(profile) & ~Long.parseLong(cfg.cpuMask, 16));
        }
        return cfg;
    }

    /** request: contextTokens (prompt estimate), outTokens, latencyTargetMs (to first action). */
    public static JSONObject plan(Context ctx, JSONObject profile, Planner.Card card, File model, long contextTokens, long outTokens, long latencyTargetMs) throws JSONException {
        JSONObject out = new JSONObject().put("model_id", card.modelId).put("request", new JSONObject().put("context_tokens", contextTokens).put("output_tokens", outTokens).put("latency_target_ms", latencyTargetMs));
        String isa = isaRefusal(profile); if (isa != null) return out.put("refusal", new JSONObject().put("reason", "EngineUnsupported").put("detail", isa));
        JSONObject fg = profile.getJSONObject("memory").getJSONObject("grantable_foreground"), gq = profile.getJSONObject("memory").getJSONObject("grantable_quiesced");
        boolean haveFg = !fg.isNull("value") && "measured".equals(fg.getString("provenance"));
        JSONObject gsrc = haveFg ? fg : gq;
        if (gsrc.isNull("value")) return out.put("refusal", new JSONObject().put("reason", "NotCalibrated").put("detail", "no memory grant was measured").put("missing", new JSONArray().put("memory.grantable_quiesced")));
        long grant = gsrc.getLong("value"); String grantBasis = haveFg ? "grantable_foreground [measured with a target app in front]" : "grantable_quiesced [" + gsrc.getString("provenance") + "; a foreground app can only lower it (assumption A1)]";
        int cctx = (int) Math.min(card.contextLimit, Math.max(1024, contextTokens + outTokens + 256));
        String fs = Profile.filesystemOf(model.getAbsolutePath()); boolean fuse = fs.contains("fuse") || fs.contains("sdcardfs") || fs.equals("DENIED");
        JSONArray cands = new JSONArray(); JSONObject best = null; long kv = card.kvF16PerToken * cctx;
        for (String tier : new String[]{"resident", "streamed"}) {
            JSONObject c = new JSONObject().put("tier", tier); long need = tier.equals("resident") ? card.totalBytes + kv : card.residentBytes + kv + cycleBytes(card);
            c.put("need_lower_bound", need).put("grant_upper_bound", grant);
            if (need > grant) { cands.put(c.put("verdict", "Infeasible").put("detail", "needs >= " + (need / MIB) + " MiB, at most " + (grant / MIB) + " MiB can be kept (" + grantBasis + ")")); continue; }
            if (tier.equals("streamed") && fuse) { cands.put(c.put("verdict", "StorageTooSlow").put("detail", "the model's path is a userspace filesystem (" + fs + "); measured 14x slower than native on a PC, and it degrades with concurrency (00_PROBLEM.md 8.7)")); continue; }
            Engine.Config cfg = configFor(profile, card, model, tier, cctx, grant); Cells cells = Calibration.load(ctx, cfg);
            c.put("config", cfgJson(cfg));
            if (cells == null || !cells.inRegime) {
                c.put("verdict", "NotCalibrated").put("detail", cells == null ? "no calibration cells for this configuration" : "calibration cells were not taken in the deployment regime")
                    .put("offer", "run Calibrate for this model and tier (loads the engine and runs 3 prompt lengths x 2 repeats)");
                cands.put(c); continue;
            }
            JSONObject ttft = cells.ttftMs(contextTokens, contextTokens), dec = Cells.predict(cells.decode, contextTokens + outTokens / 2.0, true);
            double totLo = ttft.getDouble("lo_ms") + outTokens / Math.max(1e-9, dec.getDouble("hi")) * 1000, totHi = ttft.getDouble("hi_ms") + outTokens / Math.max(1e-9, dec.getDouble("lo")) * 1000;
            String verdict = ttft.getDouble("hi_ms") <= latencyTargetMs ? "Meets" : (ttft.getDouble("lo_ms") > latencyTargetMs ? "Misses" : "InsufficientInformation");
            c.put("verdict", verdict).put("ttft_ms", ttft).put("decode_tok_s", dec).put("total_ms", new JSONObject().put("lo", totLo).put("hi", totHi))
                .put("calibration_state", ttft.getString("state")).put("basis", new JSONArray().put("calibration cells for " + Calibration.key(cfg)).put("profile.storage/cpu placement"));
            if (verdict.equals("InsufficientInformation")) c.put("detail", "the TTFT interval [" + Math.round(ttft.getDouble("lo_ms")) + ", " + Math.round(ttft.getDouble("hi_ms")) + "] ms spans the " + latencyTargetMs + " ms target: I do not know yet; more calibration repeats would narrow it");
            cands.put(c);
            if (verdict.equals("Meets") && (best == null || need < best.getLong("need_lower_bound"))) best = c;   // objective 4: smallest footprint among configurations that meet the target
        }
        out.put("candidates", cands).put("grant_basis", grantBasis);
        if (best == null) {
            boolean anyCal = false, anyNot = false; for (int i = 0; i < cands.length(); i++) { String v = cands.getJSONObject(i).getString("verdict"); anyCal |= v.equals("Meets") || v.equals("Misses") || v.equals("InsufficientInformation"); anyNot |= v.equals("NotCalibrated"); }
            String reason = anyNot ? "NotCalibrated" : (anyCal ? "Infeasible" : "Infeasible");
            return out.put("refusal", new JSONObject().put("reason", reason).put("detail", anyNot ? "no candidate has calibration cells; nothing can be promised" : "no configuration is known to meet the latency target on this device").put("nearest", "relax the latency target or calibrate more"));
        }
        String tier = best.getString("tier"); Engine.Config cfg = configFor(profile, card, model, tier, cctx, grant);
        out.put("plan_id", Integer.toHexString((card.modelId + tier + System.nanoTime()).hashCode())).put("chosen", best).put("config", cfgJson(cfg));
        out.put("lease", new JSONObject().put("floor", best.getLong("need_lower_bound")).put("target", tier.equals("streamed") ? grant : best.getLong("need_lower_bound")).put("purpose", "engine").put("priority", "interactive"));
        JSONObject dec = best.getJSONObject("decode_tok_s"); double half = (dec.getDouble("hi") - dec.getDouble("lo")) / 2;
        out.put("falsification", new JSONObject().put("metric", "decode_tok_s").put("lo", dec.getDouble("lo")).put("hi", dec.getDouble("hi")).put("tolerance_frac", half / Math.max(1e-9, dec.getDouble("value"))).put("min_observations", 3).put("window", 5));
        out.put("valid_while", new JSONObject().put("thermal_status_max", 1).put("power_state", "unplugged").put("profile_id", profile.getString("profile_id")));
        out.put("ladder", ladder(profile, card, cfg, tier, best));
        return out;
    }

    static JSONObject cfgJson(Engine.Config c) throws JSONException {
        return new JSONObject().put("tier", c.moeStream ? "streamed" : "resident").put("threads", c.threads).put("cpu_mask", c.cpuMask == null ? JSONObject.NULL : c.cpuMask).put("ctx", c.ctx).put("ubatch", c.ubatch)
            .put("cache_floor_mb", c.cacheFloorMb).put("cache_ceil_mb", c.cacheCeilMb).put("io_threads", c.ioThreads).put("io_mask", c.ioMask == null ? JSONObject.NULL : c.ioMask).put("variant", c.variant);
    }
    public static Engine.Config configFromJson(JSONObject j, File model) throws JSONException {
        Engine.Config c = new Engine.Config(); c.model = model; c.threads = j.getInt("threads"); c.cpuMask = j.isNull("cpu_mask") ? null : j.getString("cpu_mask"); c.ctx = j.getInt("ctx"); c.ubatch = j.getInt("ubatch");
        c.moeStream = j.getString("tier").equals("streamed"); c.cacheFloorMb = j.optInt("cache_floor_mb"); c.cacheCeilMb = j.optInt("cache_ceil_mb"); c.ioThreads = j.optInt("io_threads", 4); c.ioMask = j.isNull("io_mask") ? null : j.getString("io_mask"); c.variant = j.optString("variant", "dot"); return c;
    }

    /** Degradation ladder, ordered by quality preserved per unit of pressure relieved; every cost is measured or stated unknown. */
    static JSONArray ladder(JSONObject profile, Planner.Card card, Engine.Config cfg, String tier, JSONObject chosen) throws JSONException {
        JSONArray l = new JSONArray(); final int HYST = 60000;   // design parameter, no empirical claim (experiment X6 measures oscillation)
        JSONObject cm = profile.getJSONObject("cpu").getJSONObject("recommended_compute_mask");
        if (!cm.isNull("value") && cm.has("arms")) {
            JSONArray arms = cm.getJSONArray("arms"); double cur = -1; for (int i = 0; i < arms.length(); i++) if (arms.getJSONObject(i).getInt("threads") == cfg.threads) cur = arms.getJSONObject(i).getDouble("tok_s_median");
            JSONObject alt = null; for (int i = 0; i < arms.length(); i++) { JSONObject a = arms.getJSONObject(i); if (a.getInt("threads") < cfg.threads && (alt == null || a.getDouble("tok_s_median") > alt.getDouble("tok_s_median"))) alt = a; }
            if (alt != null && cur > 0) l.put(new JSONObject().put("trigger", "thermal").put("action", "change_placement").put("params", new JSONObject().put("threads", alt.getInt("threads")).put("mask_hex", alt.getString("mask_hex")))
                .put("expected_cost", new JSONObject().put("rate_frac", alt.getDouble("tok_s_median") / cur).put("quality", "none")).put("reversible", true).put("hysteresis_ms", HYST));
        }
        if (tier.equals("streamed")) l.put(new JSONObject().put("trigger", "memory").put("action", "shrink_cache").put("params", new JSONObject().put("to_mb", cfg.cacheFloorMb))
            .put("expected_cost", new JSONObject().put("rate_frac", JSONObject.NULL).put("quality", "none")).put("reversible", true).put("hysteresis_ms", HYST));
        // never below 2048 tokens: the agent's tool list plus a few steps needs it (a 1536-token context broke the agent)
        if (cfg.ctx / 2 >= 2048) l.put(new JSONObject().put("trigger", "memory").put("action", "reduce_context").put("params", new JSONObject().put("ctx", cfg.ctx / 2))
            .put("expected_cost", new JSONObject().put("rate_frac", JSONObject.NULL).put("quality", "bounded")).put("reversible", true).put("hysteresis_ms", HYST));
        l.put(new JSONObject().put("trigger", "memory").put("action", "suspend").put("params", new JSONObject()).put("expected_cost", new JSONObject().put("rate_frac", 0).put("quality", "none")).put("reversible", true).put("hysteresis_ms", HYST));
        l.put(new JSONObject().put("trigger", "memory").put("action", "refuse").put("params", new JSONObject().put("code", "MemoryUnavailable")).put("expected_cost", new JSONObject().put("rate_frac", 0).put("quality", "none")).put("reversible", true).put("hysteresis_ms", 0));
        return l;
    }
}
