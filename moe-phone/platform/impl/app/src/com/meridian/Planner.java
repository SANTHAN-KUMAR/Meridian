package com.meridian.app;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import java.io.File;
import java.util.*;

/** ModelCard derivation and the feasibility half of the ConfigurationPlanner (port of planner.py).
 *  Never predicts a rate and never returns "feasible": Infeasible only through a sound lower-bound argument
 *  (assumption A1: a foreground app cannot raise what one process may keep), otherwise NotCalibrated naming what is missing. */
public final class Planner {
    public static final class Refusal extends Exception {
        public final String reason; public final List<String> missing;
        public Refusal(String reason, String detail, List<String> missing) { super(reason + ": " + detail); this.reason = reason; this.missing = missing; }
        public Refusal(String reason, String detail) { this(reason, detail, new ArrayList<String>()); }
    }
    // Registry rows only after the pattern was observed in a real checkpoint (olmoe, gpt-oss, granitemoe).
    static final String[] EXPERT_SUFFIXES = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
            "ffn_gate_exps.bias", "ffn_up_exps.bias", "ffn_down_exps.bias"};
    static final Set<String> REGISTRY = new HashSet<>(Arrays.asList("olmoe", "gpt-oss", "granitemoe"));

    public static final class Card {
        public String modelId, arch; public int nLayer, nExpert, nUsed; public long contextLimit;
        public long totalBytes, residentBytes, expertBytes, activeBytesPerToken, kvF16PerToken, kvQ8PerToken, sliceSumMax;
        public String kvNote; public boolean tiedHead;
        public JSONObject toJson() throws JSONException {
            return new JSONObject().put("model_id", modelId).put("architecture", arch).put("n_layer", nLayer)
                .put("n_expert", nExpert).put("n_expert_used", nUsed).put("context_limit", contextLimit)
                .put("total_bytes", totalBytes).put("resident_bytes", residentBytes).put("expert_bytes", expertBytes)
                .put("active_bytes_per_token", activeBytesPerToken).put("kv_bytes_per_token_f16", kvF16PerToken)
                .put("expert_slice_bytes_sum_max", sliceSumMax).put("kv_note", kvNote);
        }
    }

    public static Card derive(File path) throws Refusal {
        Gguf g;
        try { g = Gguf.read(path); } catch (Gguf.GgufError e) { throw new Refusal("IntegrityFailed", e.getMessage()); }
        catch (java.io.IOException e) { throw new Refusal("IntegrityFailed", "unreadable: " + e.getMessage()); }
        String arch = g.kvStr("general.architecture");
        if (arch == null || !REGISTRY.contains(arch))
            throw new Refusal("ArchitectureUnsupported", "architecture '" + arch + "' has no registry row (known: " + REGISTRY + ")");
        Card c = new Card(); c.modelId = path.getName(); c.arch = arch;
        c.nLayer = (int) g.kvLong(arch + ".block_count", 0); c.nExpert = (int) g.kvLong(arch + ".expert_count", 0);
        c.nUsed = (int) g.kvLong(arch + ".expert_used_count", 0); c.contextLimit = g.kvLong(arch + ".context_length", 0);
        if (c.nLayer == 0 || c.nExpert == 0 || c.nUsed == 0) throw new Refusal("IntegrityFailed", "missing block/expert counts");
        long total = 0, exp = 0, embd = 0; boolean hasOut = false; Map<String, Long> kindMax = new HashMap<>();
        for (Gguf.Tensor t : g.tensors) {
            total += t.bytes;
            boolean isExp = false;
            for (String s : EXPERT_SUFFIXES) if (t.name.endsWith(s)) isExp = true;
            if (isExp) {
                if (t.dims[t.dims.length - 1] != c.nExpert) throw new Refusal("IntegrityFailed", t.name + " last dim != expert_count");
                if (t.bytes % c.nExpert != 0) throw new Refusal("IntegrityFailed", t.name + " bytes not divisible by expert_count");
                exp += t.bytes;
                String kind = t.name.substring(t.name.indexOf('.', 4) + 1);
                Long prev = kindMax.get(kind); long sl = t.bytes / c.nExpert;
                kindMax.put(kind, prev == null ? sl : Math.max(prev, sl));
            }
            if (t.name.equals("token_embd.weight")) embd = t.bytes;
            if (t.name.equals("output.weight")) hasOut = true;
        }
        if (exp == 0) throw new Refusal("ArchitectureUnsupported", arch + ": no tensors match the registry expert pattern");
        for (long v : kindMax.values()) c.sliceSumMax += v;
        c.totalBytes = total; c.expertBytes = exp; c.residentBytes = total - exp; c.tiedHead = !hasOut;
        long saving = hasOut ? embd : 0;   // an embedding lookup touches one row; a tied head reads the table in full
        c.activeBytesPerToken = c.residentBytes - saving + exp * c.nUsed / c.nExpert;
        long nHead = g.kvLong(arch + ".attention.head_count", 1), nKv = g.kvLong(arch + ".attention.head_count_kv", nHead);
        long hd = g.kvLong(arch + ".attention.key_length", g.kvLong(arch + ".embedding_length", 0) / nHead);
        long hdv = g.kvLong(arch + ".attention.value_length", hd);
        long elems = c.nLayer * nKv * (hd + hdv);
        c.kvF16PerToken = elems * 2; c.kvQ8PerToken = elems * 34 / 32;
        c.kvNote = g.kv.containsKey(arch + ".attention.sliding_window") ? "upper bound: sliding-window layers keep fewer tokens" : "exact for full attention";
        return c;
    }

    /** Per-tier verdicts. profile is the DeviceProfile JSON. Throws Refusal for OutOfRange / NotCalibrated / Infeasible(all tiers). */
    public static JSONObject plan(JSONObject profile, Card c, long context) throws Refusal, JSONException {
        JSONObject gq = profile.getJSONObject("memory").getJSONObject("grantable_quiesced");
        if (gq.isNull("value")) throw new Refusal("NotCalibrated", "grantable_quiesced was not measured", Arrays.asList("memory.grantable_quiesced"));
        if (context > c.contextLimit) throw new Refusal("OutOfRange", "context " + context + " exceeds model limit " + c.contextLimit);
        long upper = gq.getLong("value");
        List<String> missing = new ArrayList<>(Arrays.asList("memory.grantable_foreground (PL-S2)", "cpu.clusters[].matmul_gbps (PL-E11)", "thermal.sustained_derate (PL-E14)"));
        if (!"measured".equals(gq.getString("provenance"))) missing.add(0, "memory.grantable_quiesced measured in-regime (is " + gq.getString("provenance") + ")");
        JSONObject tiers = new JSONObject(); int infeasible = 0; StringBuilder why = new StringBuilder();
        for (String tier : new String[]{"resident", "streamed"}) {
            long kv = c.kvF16PerToken * context;
            long need = tier.equals("resident") ? c.totalBytes + kv : c.residentBytes + kv + c.sliceSumMax * c.nUsed;
            JSONObject v = new JSONObject().put("need_lower_bound", need).put("grant_upper_bound", upper);
            if (need > upper) { v.put("verdict", "Infeasible"); infeasible++; why.append(tier).append(" needs >= ").append(need).append(" B vs grant <= ").append(upper).append(" B; "); }
            else { v.put("verdict", "NotCalibrated"); JSONArray m = new JSONArray(missing); if (tier.equals("streamed")) m.put("expert-cache warm hit-rate sample"); v.put("missing", m); }
            tiers.put(tier, v);
        }
        if (infeasible == 2) throw new Refusal("Infeasible", "no tier fits context=" + context + ": " + why);
        return new JSONObject().put("context", context).put("tiers", tiers);
    }
}
