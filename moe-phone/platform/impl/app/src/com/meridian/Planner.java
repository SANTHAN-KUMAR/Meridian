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
    /** Expert tensors are recognised by llama.cpp's MoE naming (every build_moe_ffn architecture names its routed experts
     *  `ffn_*_exps`, including fused `ffn_gate_up_exps`, and their `.bias`/`.scale` companions); shared experts (`_shexp`),
     *  routers (`ffn_gate_inp`) and leading dense blocks match nothing and are resident. Integrity is checked, not assumed:
     *  every expert tensor's last dimension must equal expert_count and its bytes must divide by it. Until 2026-09-22 this
     *  was gated on a 4-row registry and every dense model was refused; the checks below are what the registry stood for. */
    static boolean isExpertTensor(String name) { int i = name.indexOf("_exps."); return i > 0 && name.substring(0, i).contains(".ffn_"); }
    /** Architectures the bundled engine can STREAM (core/src/moe/arch_registry.cpp, patched tree 2026-09-22). Any other
     *  architecture llama.cpp loads runs in the resident tier only. */
    public static final Set<String> STREAMABLE = new HashSet<>(Arrays.asList("qwen3moe", "qwen2moe", "olmoe", "qwen35moe", "gemma4",
            "gpt-oss", "lfm2moe", "deepseek4", "bailingmoe3", "qwen4exp"));

    public static final class Card {
        public String modelId, arch, dominantType, chatTemplateHint; public int nLayer, nExpert, nUsed; public long contextLimit, fileBytes;
        public long totalBytes, residentBytes, expertBytes, activeBytesPerToken, expertActiveBytesPerToken, kvF16PerToken, kvQ8PerToken, sliceSumMax;
        public String kvNote; public boolean tiedHead, moe, streamable;
        /** Bytes read per decoded token, by ggml type name (dense weights in full, routed experts at top-k/n_expert). */
        public final Map<String, Long> activeByType = new TreeMap<>();
        public JSONObject toJson() throws JSONException {
            return new JSONObject().put("model_id", modelId).put("architecture", arch).put("n_layer", nLayer)
                .put("n_expert", nExpert).put("n_expert_used", nUsed).put("context_limit", contextLimit).put("file_bytes", fileBytes)
                .put("total_bytes", totalBytes).put("resident_bytes", residentBytes).put("expert_bytes", expertBytes)
                .put("active_bytes_per_token", activeBytesPerToken).put("expert_active_bytes_per_token", expertActiveBytesPerToken)
                .put("kv_bytes_per_token_f16", kvF16PerToken).put("expert_slice_bytes_sum_max", sliceSumMax).put("kv_note", kvNote)
                .put("moe", moe).put("streamable", streamable).put("dominant_type", dominantType).put("tied_head", tiedHead)
                .put("active_by_type", new JSONObject(activeByType));
        }
        public static Card fromJson(JSONObject j) throws JSONException {
            Card c = new Card(); c.modelId = j.getString("model_id"); c.arch = j.getString("architecture"); c.nLayer = j.getInt("n_layer");
            c.nExpert = j.getInt("n_expert"); c.nUsed = j.getInt("n_expert_used"); c.contextLimit = j.getLong("context_limit"); c.fileBytes = j.optLong("file_bytes");
            c.totalBytes = j.getLong("total_bytes"); c.residentBytes = j.getLong("resident_bytes"); c.expertBytes = j.getLong("expert_bytes");
            c.activeBytesPerToken = j.getLong("active_bytes_per_token"); c.expertActiveBytesPerToken = j.optLong("expert_active_bytes_per_token");
            c.kvF16PerToken = j.getLong("kv_bytes_per_token_f16"); c.kvQ8PerToken = c.kvF16PerToken * 17 / 32; c.sliceSumMax = j.getLong("expert_slice_bytes_sum_max");
            c.kvNote = j.optString("kv_note"); c.moe = j.getBoolean("moe"); c.streamable = j.getBoolean("streamable"); c.dominantType = j.optString("dominant_type"); c.tiedHead = j.optBoolean("tied_head");
            JSONObject t = j.getJSONObject("active_by_type"); for (Iterator<String> it = t.keys(); it.hasNext(); ) { String k = it.next(); c.activeByType.put(k, t.getLong(k)); }
            return c;
        }
    }

    public static Card derive(File path) throws Refusal {
        Gguf g;
        try { g = Gguf.read(path); } catch (Gguf.GgufError e) { throw new Refusal("IntegrityFailed", e.getMessage()); }
        catch (java.io.IOException e) { throw new Refusal("IntegrityFailed", "unreadable: " + e.getMessage()); }
        return derive(g, path.getName());
    }

    public static Card derive(Gguf g, String modelId) throws Refusal {
        String arch = g.kvStr("general.architecture");
        if (arch == null) throw new Refusal("IntegrityFailed", "general.architecture is missing");
        Card c = new Card(); c.modelId = modelId; c.arch = arch; c.fileBytes = g.fileBytes;
        c.nLayer = (int) g.kvLong(arch + ".block_count", 0); c.nExpert = (int) g.kvLong(arch + ".expert_count", 0);
        c.nUsed = (int) g.kvLong(arch + ".expert_used_count", 0); c.contextLimit = g.kvLong(arch + ".context_length", 0);
        if (c.nLayer == 0) throw new Refusal("IntegrityFailed", arch + ".block_count is missing");
        long total = 0, exp = 0, embd = 0; boolean hasOut = false; Map<String, Long> kindMax = new HashMap<>();
        Map<String, Long> denseByType = new HashMap<>(), expByType = new HashMap<>(); int embdType = -1;
        for (Gguf.Tensor t : g.tensors) {
            total += t.bytes;
            if (isExpertTensor(t.name)) {
                if (c.nExpert == 0) throw new Refusal("IntegrityFailed", t.name + " is an expert tensor but " + arch + ".expert_count is missing");
                if (t.dims[t.dims.length - 1] != c.nExpert) throw new Refusal("IntegrityFailed", t.name + " last dim != expert_count");
                if (t.bytes % c.nExpert != 0) throw new Refusal("IntegrityFailed", t.name + " bytes not divisible by expert_count");
                exp += t.bytes; expByType.merge(Gguf.typeName(t.type), t.bytes, Long::sum);
                String kind = t.name.substring(t.name.indexOf('.', 4) + 1);
                Long prev = kindMax.get(kind); long sl = t.bytes / c.nExpert;
                kindMax.put(kind, prev == null ? sl : Math.max(prev, sl));
            } else denseByType.merge(Gguf.typeName(t.type), t.bytes, Long::sum);
            if (t.name.equals("token_embd.weight")) { embd = t.bytes; embdType = t.type; }
            if (t.name.equals("output.weight")) hasOut = true;
        }
        c.moe = exp > 0;
        if (c.moe && (c.nUsed == 0 || c.nUsed > c.nExpert)) throw new Refusal("IntegrityFailed", "expert_used_count " + c.nUsed + " is not in [1, " + c.nExpert + "]");
        if (!c.moe) { c.nExpert = 0; c.nUsed = 0; }
        c.streamable = c.moe && STREAMABLE.contains(arch);
        for (long v : kindMax.values()) c.sliceSumMax += v;
        c.totalBytes = total; c.expertBytes = exp; c.residentBytes = total - exp; c.tiedHead = !hasOut;
        long saving = hasOut ? embd : 0;   // an embedding lookup touches one row; a tied head reads the table in full
        c.expertActiveBytesPerToken = c.moe ? exp * c.nUsed / c.nExpert : 0;
        c.activeBytesPerToken = c.residentBytes - saving + c.expertActiveBytesPerToken;
        for (Map.Entry<String, Long> e : denseByType.entrySet()) c.activeByType.merge(e.getKey(), e.getValue(), Long::sum);
        if (saving > 0) c.activeByType.merge(Gguf.typeName(embdType), -saving, Long::sum);
        for (Map.Entry<String, Long> e : expByType.entrySet()) c.activeByType.merge(e.getKey(), e.getValue() * c.nUsed / c.nExpert, Long::sum);
        c.activeByType.values().removeIf(v -> v <= 0);
        String dom = null; long best = -1; for (Map.Entry<String, Long> e : c.activeByType.entrySet()) if (e.getValue() > best) { best = e.getValue(); dom = e.getKey(); } c.dominantType = dom;
        long nHead = g.kvLongMax(arch + ".attention.head_count", 1), nKv = g.kvLongMax(arch + ".attention.head_count_kv", nHead);
        long hd = g.kvLong(arch + ".attention.key_length", g.kvLong(arch + ".embedding_length", 0) / Math.max(1, nHead));
        long hdv = g.kvLong(arch + ".attention.value_length", hd);
        long elems = c.nLayer * nKv * (hd + hdv);
        c.kvF16PerToken = elems * 2; c.kvQ8PerToken = elems * 34 / 32;
        c.kvNote = g.kv.containsKey(arch + ".attention.sliding_window") ? "upper bound: sliding-window layers keep fewer tokens"
                 : (g.kv.get(arch + ".attention.head_count_kv") instanceof long[] || g.kv.containsKey(arch + ".full_attention_interval") ? "upper bound: hybrid stack, not every layer keeps a KV cache" : "exact for full attention");
        String tmpl = g.kvStr("tokenizer.chat_template"); c.chatTemplateHint = tmpl == null ? "none (raw prompts)" : tmpl.contains("<|im_start|>") ? "chatml" : tmpl.contains("<start_of_turn>") ? "gemma" : tmpl.contains("<|start_header_id|>") ? "llama3" : "model-specific";
        if (c.contextLimit == 0) throw new Refusal("IntegrityFailed", arch + ".context_length is missing");
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
