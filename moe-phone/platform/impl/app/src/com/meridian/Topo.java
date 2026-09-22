package com.meridian.app;

import org.json.*;
import java.util.*;

/** Default thread placement from the measured CPU topology, used until the engine A/B (Placement.calibrate) has run.
 *  Rule (06_EXECUTION_ENGINE.md section 3.4, measured on two devices): compute threads go on the fastest cores, never on
 *  in-order "little" cores, at most 4; the expert-read lanes of the streamed tier go on the remaining cores.
 *   - OnePlus Nord (SM7250, A76 x2 + A55 x6): cores 6-7 gave 17.3 tok/s vs 8.3 unpinned-4 (Placement A/B, OLMoE).
 *   - OnePlus 15R (SM8845, no little cores): compute cpu4-7 + I/O cpu0-3 was the adopted research configuration.
 *  Little cores are identified by ARM's own part numbers, not by frequency, because the Nord's A55s run at 75% of the prime
 *  core's clock yet deliver far less per clock (in-order). */
public final class Topo {
    /** ARM part numbers (MIDR) of in-order efficiency cores: Cortex-A53, A55, A510, A520, A35, A32. */
    static final Set<String> LITTLE = new HashSet<>(Arrays.asList("0xd03", "0xd05", "0xd46", "0xd80", "0xd04", "0xd01"));
    public static final class Placement { public int threads; public long computeMask, ioMask; public String basis; }

    public static Placement defaults(JSONObject profile) throws JSONException {
        JSONArray cl = profile.getJSONObject("cpu").getJSONArray("clusters");
        List<JSONObject> cs = new ArrayList<>(); for (int i = 0; i < cl.length(); i++) cs.add(cl.getJSONObject(i));
        Collections.sort(cs, (a, b) -> Long.compare(b.optLong("max_khz"), a.optLong("max_khz")));   // fastest first
        long all = 0, comp = 0; int n = 0;
        for (JSONObject c : cs) { JSONArray ids = c.getJSONArray("core_ids"); for (int k = 0; k < ids.length(); k++) all |= 1L << ids.getInt(k); }
        for (JSONObject c : cs) {
            if (LITTLE.contains(c.optString("cpu_part").toLowerCase(Locale.ROOT)) && n > 0) continue;   // never mix in little cores (unless there is nothing else)
            JSONArray ids = c.getJSONArray("core_ids");
            for (int k = ids.length() - 1; k >= 0 && n < 4; k--) { comp |= 1L << ids.getInt(k); n++; }
        }
        Placement p = new Placement(); p.threads = Math.max(1, n); p.computeMask = comp; p.ioMask = all & ~comp;
        if (p.ioMask == 0) p.ioMask = comp;   // single-cluster device with <= 4 cores: lanes share the compute cores (recorded in basis)
        p.basis = "topology rule: fastest non-little cores, max 4 (Topo.java); not yet A/B-confirmed on this device" + (p.ioMask == comp ? "; I/O lanes share compute cores" : "");
        return p;
    }

    /** Every core, when none is an in-order little core (e.g. the 15R: 6+2 big cores); null otherwise. */
    public static Placement allBigCores(JSONObject profile) throws JSONException {
        JSONArray cl = profile.getJSONObject("cpu").getJSONArray("clusters"); long all = 0; int n = 0;
        for (int i = 0; i < cl.length(); i++) { JSONObject c = cl.getJSONObject(i); if (LITTLE.contains(c.optString("cpu_part").toLowerCase(Locale.ROOT))) return null;
            JSONArray ids = c.getJSONArray("core_ids"); for (int k = 0; k < ids.length(); k++) { all |= 1L << ids.getInt(k); n++; } }
        if (n == 0) return null; Placement p = new Placement(); p.threads = n; p.computeMask = all; p.ioMask = all; p.basis = "all cores (no little cores; resident tier)"; return p;
    }
    /** Threads and mask to use: the measured placement if the A/B has run, else the topology rule. */
    public static Placement effective(JSONObject profile) throws JSONException {
        Placement d = defaults(profile);
        JSONObject cm = profile.getJSONObject("cpu").optJSONObject("recommended_compute_mask");
        if (cm != null && !cm.isNull("value")) {
            JSONObject v = cm.getJSONObject("value"); Placement p = new Placement(); p.threads = v.getInt("threads");
            p.computeMask = v.isNull("mask_hex") ? 0 : Long.parseLong(v.getString("mask_hex"), 16);
            long all = d.computeMask | d.ioMask; p.ioMask = p.computeMask == 0 ? d.ioMask : (all & ~p.computeMask); if (p.ioMask == 0) p.ioMask = d.ioMask;
            p.basis = "measured engine A/B: " + cm.optString("source"); return p;
        }
        return d;
    }

    public static boolean hasIsa(JSONObject profile, String feature) throws JSONException {
        JSONArray cl = profile.getJSONObject("cpu").getJSONArray("clusters"); boolean any = false;
        for (int i = 0; i < cl.length(); i++) { JSONArray f = cl.getJSONObject(i).getJSONArray("isa_features"); boolean has = false;
            for (int k = 0; k < f.length(); k++) if (f.getString(k).equals(feature)) has = true;
            if (!has) return false; any = true; }
        return any;   // a feature counts only if EVERY cluster has it: threads may migrate across clusters
    }
}
