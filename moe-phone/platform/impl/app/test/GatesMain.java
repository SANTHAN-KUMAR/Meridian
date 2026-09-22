package com.meridian.app;

import org.json.*;
import java.io.File;
import java.util.*;

/** Host-JVM gates for the generalization layer (CLAUDE.md section 9.2). Every expected value is fixed by arithmetic, by the
 *  GGUF format, or by a measured topology, never by the method's own output. Run: test/run_gates.sh. Exit 1 on any failure.
 *  Would each gate still pass if the component were replaced by a constant? Answered next to each one. */
public class GatesMain {
    static int fails = 0, passes = 0;
    static void check(String name, boolean ok, String detail) { System.out.println((ok ? "PASS " : "FAIL ") + name + (ok ? "" : "  " + detail)); if (ok) passes++; else fails++; }
    static boolean close(double a, double b, double rel) { return Math.abs(a - b) <= rel * Math.max(Math.abs(a), Math.abs(b)); }

    static JSONObject profile(double t0ms, double tlms, double wQ40, double dram) throws JSONException {
        JSONObject types = new JSONObject();
        for (String t : new String[]{"Q4_0", "Q8_0", "Q4_K", "Q6_K", "MXFP4"}) types.put(t, new JSONObject().put("w_gbps", wQ40).put("lo", wQ40).put("hi", wQ40).put("bytes_per_token", 100000000L));
        JSONObject cv = new JSONObject().put("t0_ms", t0ms).put("t_layer_ms", tlms).put("types", types).put("prefill_gbps_q4_0", 20.0).put("prefill_tok_s_q4_0_L3", new JSONArray().put(200.0).put(200.0)).put("engine_variant", "dot");
        JSONObject cpu = new JSONObject().put("compute", new JSONObject().put("value", cv).put("provenance", "measured")).put("clusters", new JSONArray()).put("recommended_compute_mask", new JSONObject().put("value", JSONObject.NULL));
        JSONObject mem = new JSONObject().put("dram_read_gbps", new JSONObject().put("value", dram).put("interval", new JSONArray().put(dram).put(dram)))
            .put("grantable_quiesced", new JSONObject().put("value", 8L << 30).put("provenance", "measured")).put("grantable_foreground", new JSONObject().put("value", JSONObject.NULL))
            .put("available_at_probe", new JSONObject().put("value", 6L << 30));
        JSONArray rr = new JSONArray().put(new JSONObject().put("size_bytes", 1048576).put("threads", 4).put("mbps", new JSONObject().put("value", 2000.0).put("interval", new JSONArray().put(2000.0).put(2000.0))));
        return new JSONObject().put("cpu", cpu).put("memory", mem).put("storage", new JSONObject().put("random_read", rr)).put("profile_id", "test");
    }
    static Planner.Card dense(long bytes, int layers) { Planner.Card c = new Planner.Card(); c.modelId = "t"; c.arch = "llama"; c.nLayer = layers; c.contextLimit = 4096;
        c.totalBytes = c.residentBytes = c.activeBytesPerToken = bytes; c.activeByType.put("Q4_0", bytes); c.dominantType = "Q4_0"; c.kvF16PerToken = 0; return c; }

    public static void main(String[] a) throws Exception {
        // 1. Recovery: the probe fit returns the parameters that generated the timings (exact arithmetic).
        //    A constant fit could not return three different planted values.
        double W = 9.9e9, TL = 0.00021, T0 = 0.0004, bA = 3.7e7, bB = 1.1e8, bC = 4.3e7;
        double[] f = ComputeProbe.fit(bA, bB, bC, T0 + TL + bA / W, T0 + 3 * TL + bB / W, T0 + 3 * TL + bC / W);
        check("probe fit recovers W, t_layer, t0", close(f[0], W, 1e-9) && close(f[1], TL, 1e-9) && close(f[2], T0, 1e-9), Arrays.toString(f));

        // 2. Invariance: doubling every kernel rate halves the byte term exactly (t0 = t_layer = 0, no KV).
        Planner.Card c = dense(2_000_000_000L, 28);
        double r1 = Predictor.decode(profile(0, 0, 10, 50), c, "resident", 0, 0).value, r2 = Predictor.decode(profile(0, 0, 20, 50), c, "resident", 0, 0).value;
        check("decode rate scales 1:1 with kernel rate", close(r2 / r1, 2.0, 1e-9), r1 + " -> " + r2);
        //    ...and the absolute value is bytes / W: 2 GB at 10 GB/s = 0.2 s = 5 tok/s.
        check("decode = W / active bytes", close(r1, 5.0, 1e-9), String.valueOf(r1));

        // 3. The fixed terms add, per token and per layer: 5 tok/s model + 1 ms + 28 x 0.1 ms.
        double r3 = Predictor.decode(profile(1.0, 0.1, 10, 50), c, "resident", 0, 0).value;
        check("fixed + per-layer terms add", close(1 / r3, 0.2 + 0.001 + 0.0028, 1e-9), String.valueOf(1 / r3));

        // 4. Attention: the KV cache is re-read every token. 100 KiB/token x 1000 tokens at 50 GB/s adds 102400000/50e9 s.
        c.kvF16PerToken = 102400; double r4 = Predictor.decode(profile(0, 0, 10, 50), c, "resident", 1000, 0).value;
        check("KV re-read priced at DRAM rate", close(1 / r4, 0.2 + 102400.0 * 1000 / 50e9, 1e-9), String.valueOf(1 / r4)); c.kvF16PerToken = 0;

        // 5. Degeneracy: without a compute probe there is no number, only a refusal.
        JSONObject noProbe = profile(0, 0, 10, 50); noProbe.getJSONObject("cpu").remove("compute");
        boolean refused = false; try { Predictor.decode(noProbe, c, "resident", 0, 0); } catch (IllegalStateException e) { refused = e.getMessage().startsWith("NotCalibrated"); }
        check("no compute probe -> NotCalibrated refusal", refused, "returned a number");

        // 6. Monotonicity (streamed): a bigger expert cache never predicts a slower model, and the no-locality lower end
        //    of the hit rate equals the cache fraction. A constant predictor would fail the strict increase.
        Planner.Card m = dense(1_000_000_000L, 48); m.moe = true; m.streamable = true; m.nExpert = 128; m.nUsed = 8; m.expertBytes = 16_000_000_000L; m.totalBytes = 17_000_000_000L;
        m.residentBytes = 1_000_000_000L; m.expertActiveBytesPerToken = 1_000_000_000L; m.sliceSumMax = 3_000_000L; m.activeByType.put("Q4_0", 2_000_000_000L);
        double s1 = Predictor.decode(profile(0, 0, 10, 50), m, "streamed", 0, 4_000_000_000L).value, s2 = Predictor.decode(profile(0, 0, 10, 50), m, "streamed", 0, 8_000_000_000L).value;
        check("bigger cache -> faster streamed prediction", s2 > s1, s1 + " vs " + s2);
        check("hit lower bound = cache fraction", close(Predictor.curve(0.0), 0, 0) && Predictor.curve(0.25) >= 0.25 && close(Predictor.curve(1.0), 1, 0), "curve ends");

        // 7. Topology rule on measured topologies (values from the research repo's profiles).
        JSONObject nord = new JSONObject().put("cpu", new JSONObject().put("clusters", new JSONArray()
            .put(new JSONObject().put("core_ids", new JSONArray("[0,1,2,3,4,5]")).put("max_khz", 1804800).put("cpu_part", "0xd05"))
            .put(new JSONObject().put("core_ids", new JSONArray("[6]")).put("max_khz", 2208000).put("cpu_part", "0xd0d"))
            .put(new JSONObject().put("core_ids", new JSONArray("[7]")).put("max_khz", 2400000).put("cpu_part", "0xd0d"))));
        Topo.Placement pn = Topo.defaults(nord);
        check("Nord: compute on cores 6-7, never the A55s (A/B winner 17.3 vs 8.3 tok/s)", pn.threads == 2 && pn.computeMask == 0xc0 && pn.ioMask == 0x3f, pn.threads + " 0x" + Long.toHexString(pn.computeMask));
        JSONObject r15 = new JSONObject().put("cpu", new JSONObject().put("clusters", new JSONArray()
            .put(new JSONObject().put("core_ids", new JSONArray("[0,1,2,3,4,5]")).put("max_khz", 3321600).put("cpu_part", "0x002"))
            .put(new JSONObject().put("core_ids", new JSONArray("[6,7]")).put("max_khz", 3801600).put("cpu_part", "0x002"))));
        Topo.Placement p15 = Topo.defaults(r15);
        check("15R: compute cpu4-7 = mask f0 (the research's adopted placement)", p15.threads == 4 && p15.computeMask == 0xf0 && p15.ioMask == 0x0f, p15.threads + " 0x" + Long.toHexString(p15.computeMask));

        // 8. Cards from real files (skipped when absent): a tied-head dense model reads every byte per token; a MoE card's
        //    active bytes equal resident - untied embedding + experts x top-k / n_expert (the definition, recomputed here).
        File q06 = new File("/run/media/santhankumar/New Volume/moe-work/models/Qwen3-0.6B-Q8_0.gguf");
        if (q06.exists()) { Planner.Card d = Planner.derive(q06);
            check("dense tied-head card: active = total, not MoE", !d.moe && d.tiedHead && d.activeBytesPerToken == d.totalBytes && d.fileBytes == q06.length(), d.toJson().toString()); }
        File qm = new File("/run/media/santhankumar/New Volume/moe-work/models/Qwen3-30B-A3B-Q4_0.gguf");
        if (qm.exists()) { Gguf g = Gguf.read(qm); Planner.Card d = Planner.derive(g, qm.getName()); long embd = 0; boolean out = false;
            for (Gguf.Tensor t : g.tensors) { if (t.name.equals("token_embd.weight")) embd = t.bytes; if (t.name.equals("output.weight")) out = true; }
            long want = d.residentBytes - (out ? embd : 0) + d.expertBytes * d.nUsed / d.nExpert; long byType = 0; for (long v : d.activeByType.values()) byType += v;
            check("MoE card: active bytes by definition, and per-type split sums to it", d.moe && d.streamable && d.activeBytesPerToken == want && Math.abs(byType - want) <= d.activeByType.size() * 1L, want + " vs " + d.activeBytesPerToken + " / " + byType); }

        // 9. Integrity: an expert tensor whose last dim disagrees with expert_count is refused, not guessed.
        //    (Constructed in memory: the reader must never turn a malformed file into a card.)
        Gguf g = new Gguf.Builder().kv("general.architecture", "qwen3moe").kv("qwen3moe.block_count", 1L).kv("qwen3moe.expert_count", 8L).kv("qwen3moe.expert_used_count", 2L).kv("qwen3moe.context_length", 4096L)
            .tensor("blk.0.ffn_up_exps.weight", new long[]{64, 32, 7}, 2, 64 * 32 * 7 / 32 * 18).build();
        boolean ref = false; try { Planner.derive(g, "bad"); } catch (Planner.Refusal r) { ref = r.reason.equals("IntegrityFailed"); }
        check("malformed expert tensor -> IntegrityFailed", ref, "a card was produced");

        System.out.println(passes + " passed, " + fails + " failed");
        System.exit(fails == 0 ? 0 : 1);
    }
}
