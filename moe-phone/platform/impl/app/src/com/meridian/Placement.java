package com.meridian.app;

import android.content.Context;
import org.json.*;
import java.io.File;
import java.util.*;

/** Thread-placement A/B on the REAL workload (04_DEVICE_PROFILING.md section 3.2): candidate CPU masks are derived from the
 *  measured cluster topology, each is run through the actual engine on the actual model, ABBA-ordered with a cool-down, and
 *  the winner is chosen by median decode rate. Overlapping ranges are reported as a tie and resolved toward fewer threads. */
public final class Placement {
    public interface Progress { void step(String s); }
    static final class Cand { String name; long mask; int threads; List<Double> rates = new ArrayList<>(); }

    public static JSONObject calibrate(Context ctx, File model, JSONArray clusters, Progress prog) throws Exception {
        List<Cand> cands = new ArrayList<>(); long all = 0, big = 0, prime = 0; int nAll = 0, nBig = 0;
        for (int i = 0; i < clusters.length(); i++) { JSONObject c = clusters.getJSONObject(i); JSONArray ids = c.getJSONArray("core_ids");
            for (int k = 0; k < ids.length(); k++) { long bit = 1L << ids.getInt(k); all |= bit; nAll++; if (i > 0) { big |= bit; nBig++; } if (i == clusters.length() - 1) prime |= bit; } }
        if (clusters.length() == 0) throw new IllegalStateException("no CPU topology available");
        cands.add(mk("all-cores", all, nAll));
        if (nBig > 0 && big != all) cands.add(mk("non-efficiency", big, nBig));
        if (prime != 0 && prime != big && prime != all) cands.add(mk("prime-only", prime, Long.bitCount(prime)));
        Cand un = mk("unpinned-4", 0, 4); cands.add(un);
        int reps = 2;
        for (int r = 0; r < reps; r++) {
            List<Cand> order = new ArrayList<>(cands); if (r % 2 == 1) Collections.reverse(order);   // ABBA: order must not be the effect
            for (Cand c : order) {
                prog.step("placement " + c.name + " (rep " + (r + 1) + "/" + reps + ")");
                Engine.Config cfg = new Engine.Config(); cfg.model = model; cfg.threads = c.threads; cfg.ctx = 512; cfg.ubatch = 256; cfg.cpuMask = c.mask == 0 ? null : Long.toHexString(c.mask);
                Chat chat = new Chat(ctx, cfg);
                try { chat.start(120000); chat.ask("Count from one to thirty in words.", 24, true, null); c.rates.add(chat.lastResult.getDouble("tok_s")); }
                finally { chat.close(); }
                Thread.sleep(6000);   // cool-down between arms
            }
        }
        Cand best = null; JSONArray table = new JSONArray();
        for (Cand c : cands) { Collections.sort(c.rates); double med = c.rates.get(c.rates.size() / 2);
            table.put(new JSONObject().put("name", c.name).put("mask_hex", Long.toHexString(c.mask)).put("threads", c.threads).put("tok_s_median", med).put("tok_s_min", c.rates.get(0)).put("tok_s_max", c.rates.get(c.rates.size() - 1)));
            if (best == null || med > best.rates.get(best.rates.size() / 2)) best = c; }
        // tie rule: any arm whose max reaches the winner's min is indistinguishable; take the fewest threads among them
        boolean tie = false; Cand pick = best;
        for (Cand c : cands) if (c != best && c.rates.get(c.rates.size() - 1) >= best.rates.get(0)) { tie = true; if (c.threads < pick.threads) pick = c; }
        return new JSONObject().put("recommended", new JSONObject().put("name", pick.name).put("mask_hex", pick.mask == 0 ? JSONObject.NULL : Long.toHexString(pick.mask)).put("threads", pick.threads))
            .put("tie", tie).put("arms", table).put("basis", "engine A/B, ABBA, " + reps + " reps, 24 decoded tokens, model=" + model.getName());
    }
    private static Cand mk(String n, long m, int t) { Cand c = new Cand(); c.name = n; c.mask = m; c.threads = t; return c; }
}
