package com.meridian.app;

import org.json.*;
import java.util.*;

/** Pre-download performance model (05_PERFORMANCE_MODEL.md section 3.2), evaluated from THIS device's measured probes and a
 *  model's GGUF header (local or remote), so the app can say what a model will do before the user downloads it:
 *
 *    ms_per_token = compute_ms + attention_ms + stall_ms + mgmt_ms
 *    compute_ms   = t0 + n_layer * t_layer + sum_T active_bytes_T / W_T   (t0, t_layer, W_T measured by ComputeProbe here)
 *    attention_ms = kv_bytes_per_token * context / dram_read     (the KV cache is re-read every token; dramprobe)
 *    stall_ms     = (1 - hit) * expert_active_bytes / read_rate  (streamed tier only; read_rate from ufsbench at slice size)
 *    mgmt_ms      = streamed tier only; prior from the two devices the research measured (22.0 ms 15R E1, 51 ms Nord)
 *
 *  Basis labels (05 section 2): "calibrated" when every input was measured in-regime on this device and >= 90% of the active
 *  bytes are in a measured weight type; otherwise "prior", with the interval bracketing the unmeasured inputs by measured
 *  alternatives (never by a tuned constant). With no compute probe the answer is a refusal (NotCalibrated), not a number.
 *  Known gaps, stated rather than hidden: thermal derate (T3) is not applied, so figures are for the first minutes;
 *  the working set is a design margin (PL-E21); the hit-rate curve is transferred from a Qwen3-30B-A3B trace. */
public final class Predictor {
    public static final long MIB = 1048576L;
    /** Engine working set beyond weights+KV (compute buffers at ubatch 512, allocator slack). Design margin, not fitted:
     *  the engine reported 80 MiB of compute buffer at ubatch 512 on Qwen3 (cli --help text); doubled for other shapes. */
    public static final long WORKING_SET = 256 * MIB;
    /** Qwen3-30B-A3B expert-cache hit rate vs cache fraction, simulator offset to the engine's measured hit
     *  (results/2026-09-18/cache_projection.json, "projected_measured_hit"), with the physical ends (0,0) and (1,1). */
    static final double[][] HIT_CURVE = {{0, 0}, {0.32, 0.853}, {0.448, 0.928}, {0.544, 0.954}, {1, 1}};
    static final double MGMT_LO_MS = 22.0, MGMT_HI_MS = 51.0;
    /** E1 (results/2026-09-19/E1_VERDICT.md): Qwen3-30B-A3B replay compute, generic 93.7 ms vs repacked 71.0 ms, same phone. */
    static final double GENERIC_OVER_REPACK = 93.7 / 71.0;
    static final Map<String, double[]> BPW = new HashMap<>();   // type -> {elements per block, bytes per block}
    static { Object[][] t = {{"F32",1,4},{"F16",1,2},{"BF16",1,2},{"Q4_0",32,18},{"Q4_1",32,20},{"Q5_0",32,22},{"Q5_1",32,24},{"Q8_0",32,34},
            {"Q2_K",256,84},{"Q3_K",256,110},{"Q4_K",256,144},{"Q5_K",256,176},{"Q6_K",256,210},{"IQ2_XXS",256,66},{"IQ2_XS",256,74},{"IQ3_XXS",256,98},
            {"IQ1_S",256,50},{"IQ4_NL",32,18},{"IQ3_S",256,110},{"IQ2_S",256,82},{"IQ4_XS",256,136},{"IQ1_M",256,56},{"TQ1_0",256,54},{"TQ2_0",256,66},{"MXFP4",32,17},{"NVFP4",64,36}};
        for (Object[] r : t) BPW.put((String) r[0], new double[]{((Number) r[1]).doubleValue(), ((Number) r[2]).doubleValue()}); }
    static double bytesPerWeight(String t) { double[] b = BPW.get(t); return b == null ? Double.NaN : b[1] / b[0]; }

    public static final class Pred { public double value, lo, hi; public String basis; public List<String> notes = new ArrayList<>();
        JSONObject json() throws JSONException { return new JSONObject().put("value", value).put("lo", lo).put("hi", hi).put("provenance", basis).put("notes", new JSONArray(notes)); } }

    /** Seconds per byte for weight type t: {mid, lo, hi, measuredFlag}. Unmeasured types are bracketed by the measured types
     *  of the same kernel family, converted at equal time per WEIGHT (dot kernels do work per weight, not per byte). */
    static double[] secPerByte(JSONObject compute, JSONObject profile, String t) throws JSONException {
        JSONObject types = compute.getJSONObject("types");
        if (types.has(t)) { JSONObject w = types.getJSONObject(t); double mid = 1 / (w.getDouble("w_gbps") * 1e9);
            double lo = w.isNull("hi") ? mid : 1 / (w.getDouble("hi") * 1e9), hi = 1 / (w.getDouble("lo") * 1e9); return new double[]{mid, lo, hi, 1}; }
        if (t.equals("F32") || t.equals("F16") || t.equals("BF16")) {   // float matvec is memory-bound: priced at the measured DRAM read rate
            JSONObject d = profile.getJSONObject("memory").getJSONObject("dram_read_gbps");
            if (!d.isNull("value")) { double mid = 1 / (d.getDouble("value") * 1e9); JSONArray iv = d.optJSONArray("interval");
                return new double[]{mid, iv == null ? mid : 1 / (iv.getDouble(1) * 1e9), iv == null ? mid : 1 / (iv.getDouble(0) * 1e9), 0}; }
        }
        String[] fam = t.endsWith("_K") || t.startsWith("IQ") && !t.equals("IQ4_NL") || t.startsWith("TQ") ? new String[]{"Q4_K", "Q6_K"}
                     : t.equals("MXFP4") || t.equals("NVFP4") ? new String[]{"MXFP4", "Q4_0"} : new String[]{"Q4_0", "Q8_0"};
        double bu = bytesPerWeight(t); if (Double.isNaN(bu)) bu = 0.5625;
        double lo = Double.MAX_VALUE, hi = 0, sum = 0; int n = 0;
        for (String c : fam) { if (!types.has(c)) continue; double spw = 1 / (types.getJSONObject(c).getDouble("w_gbps") * 1e9) * bytesPerWeight(c); double spb = spw / bu;
            lo = Math.min(lo, spb); hi = Math.max(hi, spb); sum += spb; n++; }
        if (n == 0) throw new IllegalStateException("no measured kernel rate covers weight type " + t);
        return new double[]{sum / n, lo, hi, 0};
    }

    public static JSONObject compute(JSONObject profile) throws JSONException {
        JSONObject c = profile.getJSONObject("cpu").optJSONObject("compute"); return c == null || c.isNull("value") ? null : c;
    }

    /** Decode-rate prediction in tokens/s for `tier` at `context` tokens already in the KV cache. cacheBytes only for streamed. */
    public static Pred decode(JSONObject profile, Planner.Card card, String tier, long context, long cacheBytes) throws JSONException {
        JSONObject cp = compute(profile); if (cp == null) throw new IllegalStateException("NotCalibrated: run the compute probe (Device tab)");
        JSONObject cv = cp.getJSONObject("value"); boolean calibrated = "measured".equals(cp.getString("provenance"));
        Pred p = new Pred(); double t0 = cv.getDouble("t0_ms") / 1000 + card.nLayer * cv.optDouble("t_layer_ms", 0) / 1000, mid = t0, lo = t0, hi = t0; long measuredBytes = 0, all = 0;
        for (Map.Entry<String, Long> e : card.activeByType.entrySet()) {
            double[] s = secPerByte(cv, profile, e.getKey()); mid += e.getValue() * s[0]; lo += e.getValue() * s[1]; hi += e.getValue() * s[2];
            all += e.getValue(); if (s[3] == 1) measuredBytes += e.getValue(); else p.notes.add(e.getKey() + " (" + (e.getValue() / MIB) + " MiB/token) priced from measured kernels of the same family");
        }
        if (all > 0 && measuredBytes < 0.9 * all) calibrated = false;
        // attention: the KV cache is read once per decoded token
        JSONObject dr = profile.getJSONObject("memory").getJSONObject("dram_read_gbps");
        if (!dr.isNull("value") && context > 0) { double kvb = (double) card.kvF16PerToken * context, g = dr.getDouble("value") * 1e9; JSONArray iv = dr.optJSONArray("interval");
            mid += kvb / g; lo += kvb / (iv == null ? g : iv.getDouble(1) * 1e9); hi += kvb / (iv == null ? g : iv.getDouble(0) * 1e9); }
        else if (context > 0) { calibrated = false; p.notes.add("attention cost not priced: DRAM bandwidth was not measured"); }
        if (tier.equals("streamed")) {
            calibrated = false;
            double f = Math.min(1, (double) cacheBytes / Math.max(1, card.expertBytes)), cycle = (double) PlanV2.cycleBytes(card) / Math.max(1, card.expertBytes);
            double hitHi = curve(f), hitLo = f >= cycle ? f : 0, hitMid = hitHi;   // lower end: LRU with no routing locality hits exactly f
            double[] rr = readRate(profile, card); if (rr == null) throw new IllegalStateException("NotCalibrated: storage random-read rate was not measured on the model path");
            double eb = card.expertActiveBytesPerToken;
            // Streamed experts are computed from the cache's slots with the generic kernels (the engine's --repack-experts is
            // off), while ComputeProbe's resident calibration runs on llama.cpp's repacked kernels. The research measured the
            // gap on the 15R with routing replayed: generic 93.7 vs repacked 71.0 ms/token (results/2026-09-19/E1_VERDICT.md).
            // Expert bytes are therefore charged GENERIC_OVER_REPACK times more at mid/hi; lo keeps the repacked rate.
            double[] sx = secPerByte(cv, profile, card.dominantType == null ? "Q4_0" : card.dominantType);
            mid += eb * sx[0] * (GENERIC_OVER_REPACK - 1); hi += eb * sx[2] * (GENERIC_OVER_REPACK - 1);
            p.notes.add(String.format(Locale.ROOT, "streamed experts use the generic kernels: charged %.2fx the repacked rate (measured on the 15R, E1)", GENERIC_OVER_REPACK));
            mid += (1 - hitMid) * eb / rr[0] + (MGMT_LO_MS + MGMT_HI_MS) / 2000; lo += (1 - hitHi) * eb / rr[2] + MGMT_LO_MS / 1000; hi += (1 - hitLo) * eb / rr[1] + MGMT_HI_MS / 1000;
            p.notes.add(String.format(Locale.ROOT, "expert cache %.0f%% of the experts: hit %.0f-%.0f%% (curve from a Qwen3-30B-A3B trace; low end = no locality)", f * 100, hitLo * 100, hitHi * 100));
            p.notes.add("cache management 22-51 ms/token: prior from the two phones the research measured");
        }
        p.value = 1 / mid; p.lo = 1 / hi; p.hi = 1 / lo; p.basis = calibrated ? "calibrated" : "prior";
        if (!"measured".equals(cp.getString("provenance"))) p.notes.add("the compute probe ran out of regime (plugged in, screen off or app in background)");
        p.notes.add("sustained heat is not modelled yet (T3): expect lower after several minutes of continuous use");
        return p;
    }

    /** Prefill rate (prompt tokens/s) from the Q4_0 prefill calibration, scaled by active bytes. */
    public static Pred prefill(JSONObject profile, Planner.Card card) throws JSONException {
        JSONObject cp = compute(profile); if (cp == null) throw new IllegalStateException("NotCalibrated: run the compute probe");
        JSONObject cv = cp.getJSONObject("value"); JSONArray pts = cv.getJSONArray("prefill_tok_s_q4_0_L3"); double gb = cv.getDouble("prefill_gbps_q4_0") * 1e9;
        long bytesL3 = cv.getJSONObject("types").getJSONObject("Q4_0").getLong("bytes_per_token"); double mn = Double.MAX_VALUE, mx = 0;
        for (int i = 0; i < pts.length(); i++) { mn = Math.min(mn, pts.getDouble(i)); mx = Math.max(mx, pts.getDouble(i)); }
        Pred p = new Pred(); p.value = gb / card.activeBytesPerToken; p.lo = mn * bytesL3 / card.activeBytesPerToken; p.hi = mx * bytesL3 / card.activeBytesPerToken;
        p.basis = "prior"; p.notes.add("scaled from a Q4_0 prefill at a 118-token prompt; longer prompts and other weight types differ"); return p;
    }

    static double curve(double f) { for (int i = 0; i + 1 < HIT_CURVE.length; i++) if (f <= HIT_CURVE[i + 1][0]) { double[] a = HIT_CURVE[i], b = HIT_CURVE[i + 1]; return a[1] + (f - a[0]) / (b[0] - a[0]) * (b[1] - a[1]); } return 1; }

    /** {mid, lo, hi} bytes/s at the request size nearest the model's per-projection expert slice, at 4 lanes (else the nearest). */
    static double[] readRate(JSONObject profile, Planner.Card card) throws JSONException {
        JSONArray rr = profile.getJSONObject("storage").optJSONArray("random_read"); if (rr == null || rr.length() == 0) return null;
        double slice = card.sliceSumMax / 3.0; JSONObject best = null; double bs = Double.MAX_VALUE;
        for (int i = 0; i < rr.length(); i++) { JSONObject r = rr.getJSONObject(i); double d = Math.abs(Math.log(r.getLong("size_bytes") / slice)) + Math.abs(r.getInt("threads") - 4) * 0.01; if (d < bs) { bs = d; best = r; } }
        JSONObject m = best.getJSONObject("mbps"); JSONArray iv = m.optJSONArray("interval"); double v = m.getDouble("value") * 1e6;
        return new double[]{v, iv == null ? v : iv.getDouble(0) * 1e6, iv == null ? v : iv.getDouble(1) * 1e6};
    }

    /** The memory this app may keep: measured grant if present, else the MemAvailable reading (a prior, stated). */
    public static long[] grant(JSONObject profile) throws JSONException {   // {bytes, measuredFlag}
        JSONObject m = profile.getJSONObject("memory"); JSONObject fg = m.getJSONObject("grantable_foreground"), gq = m.getJSONObject("grantable_quiesced");
        if (!fg.isNull("value") && "measured".equals(fg.optString("provenance"))) return new long[]{fg.getLong("value"), 1};
        if (!gq.isNull("value")) return new long[]{gq.getLong("value"), "measured".equals(gq.optString("provenance")) ? 1 : 0};
        return new long[]{m.getJSONObject("available_at_probe").getLong("value"), 0};
    }

    /** Evaluate one model on this device: tier, config sizes, predictions and the verdict a user sees. */
    public static JSONObject evaluate(JSONObject profile, Planner.Card card, long freeDisk, boolean onDisk, String modelPathFs) throws JSONException {
        JSONObject o = new JSONObject().put("model_id", card.modelId);
        long[] g = grant(profile); long grant = g[0]; int ctx = (int) Math.min(card.contextLimit, 4096);
        // the largest context (4096, 3072, 2048 tokens) that lets the model stay resident: the KV cache is the only part that
        // shrinks, and 2048 tokens still holds the agent's tool list plus a few steps. Below 2048 the answer is Infeasible.
        for (int c : new int[]{4096, 3072, 2048}) { if (c > card.contextLimit) continue; if (card.totalBytes + card.kvF16PerToken * c + WORKING_SET <= grant) { ctx = c; break; } ctx = Math.min(c, (int) card.contextLimit); }
        long kv = card.kvF16PerToken * ctx, needRes = card.totalBytes + kv + WORKING_SET, floor = PlanV2.cycleBytes(card), needStr = card.residentBytes + kv + WORKING_SET + floor;
        o.put("grant_bytes", grant).put("grant_basis", g[1] == 1 ? "measured" : "prior").put("need_resident", needRes).put("need_streamed_min", card.streamable ? needStr : JSONObject.NULL).put("ctx", ctx);
        if (!onDisk && card.fileBytes + (512 * MIB) > freeDisk) return o.put("verdict", "NoSpace").put("detail", "needs " + (card.fileBytes / MIB) + " MiB of storage, " + (freeDisk / MIB) + " MiB free");
        String tier; long cache = 0;
        if (needRes <= grant) tier = "resident";
        else if (card.streamable && needStr <= grant) {
            boolean fuse = modelPathFs != null && (modelPathFs.contains("fuse") || modelPathFs.contains("sdcardfs"));
            if (fuse) return o.put("verdict", "StorageTooSlow").put("detail", "streaming needs the model on a native filesystem (model path is " + modelPathFs + ")");
            tier = "streamed"; cache = Math.min(card.expertBytes, grant - card.residentBytes - kv - WORKING_SET);
        } else {
            // shrink the context before giving up: a smaller KV cache may fit
            return o.put("verdict", "Infeasible").put("detail", "needs >= " + (Math.min(needRes, card.streamable ? needStr : needRes) / MIB) + " MiB of RAM; this phone can keep about " + (grant / MIB) + " MiB"
                + (card.moe && !card.streamable ? " (the engine cannot stream architecture '" + card.arch + "' yet)" : ""));
        }
        o.put("tier", tier).put("cache_bytes", cache);
        try {
            Pred d = decode(profile, card, tier, 512, cache), pf = prefill(profile, card);
            double ttft = 200 / pf.value + 1 / d.value, ttftHi = 200 / pf.lo + 1 / d.lo, ttftLo = 200 / pf.hi + 1 / d.hi;
            o.put("decode_tok_s", d.json()).put("prefill_tok_s", pf.json()).put("ttft_200_s", new JSONObject().put("value", ttft).put("lo", ttftLo).put("hi", ttftHi));
            // reading-speed classes (design thresholds, not measurements), judged on the LOWER bound: a promise uses the
            // lower end of the interval (05_PERFORMANCE_MODEL.md section 6.3)
            String cls = d.lo >= 10 ? "fluent" : d.lo >= 4 ? "usable" : "slow";
            o.put("verdict", "Runs").put("speed_class", cls);
        } catch (IllegalStateException e) { o.put("verdict", "NotCalibrated").put("detail", e.getMessage()); }
        return o;
    }

    /** Parameter estimate for ranking: bytes over bytes-per-weight of the dominant type. */
    public static double paramsB(Planner.Card c) { double b = bytesPerWeight(c.dominantType); if (Double.isNaN(b)) b = 0.5625; return c.totalBytes / b / 1e9; }

    /** Rank evaluated catalog entries: the recommendation is the largest model that is fluent, then the largest usable. */
    public static JSONArray rank(List<JSONObject> evals) throws JSONException {
        List<JSONObject> l = new ArrayList<>(evals);
        Collections.sort(l, (a, b) -> {
            int ra = rankClass(a), rb = rankClass(b); if (ra != rb) return Integer.compare(ra, rb);
            return Double.compare(b.optDouble("params_b"), a.optDouble("params_b"));
        });
        JSONArray out = new JSONArray(); boolean picked = false;
        for (JSONObject e : l) { if (!picked && rankClass(e) <= 1) { e.put("recommended", true); picked = true; } out.put(e); }
        return out;
    }
    static int rankClass(JSONObject e) { if (!"Runs".equals(e.optString("verdict"))) return 4; String c = e.optString("speed_class"); return c.equals("fluent") ? 0 : c.equals("usable") ? 1 : 2; }
}
