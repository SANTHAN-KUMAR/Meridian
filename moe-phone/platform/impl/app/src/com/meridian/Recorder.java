package com.meridian.app;

import android.content.Context;
import org.json.JSONObject;
import java.io.*;
import java.util.*;

/** Audit trail + evaluation data (03_INTERFACES.md section 9). One JSON line per turn/event in filesDir/audit.jsonl.
 *  Never records prompt or screen content, only structural fields and tool names. */
public final class Recorder {
    private final File f;
    public Recorder(Context c) { f = new File(c.getFilesDir(), "audit.jsonl"); }
    public synchronized void write(JSONObject j) {
        try (FileWriter w = new FileWriter(f, true)) { w.write(j.toString()); w.write('\n'); } catch (IOException ignored) { }
    }
    /** Observed decode rates from recorded in-regime turns, most recent last. */
    public synchronized List<Double> observedTokS(String modelId) {
        List<Double> v = new ArrayList<>();
        try (BufferedReader r = new BufferedReader(new FileReader(f))) {
            String l; while ((l = r.readLine()) != null) { JSONObject j = new JSONObject(l);
                if (modelId.equals(j.optString("model_id")) && j.optJSONObject("observed") != null && j.optBoolean("in_regime", false)) v.add(j.getJSONObject("observed").getDouble("tokens_per_s")); }
        } catch (Exception ignored) { }
        return v;
    }
    public synchronized String tail(int n) {
        Deque<String> d = new ArrayDeque<>();
        try (BufferedReader r = new BufferedReader(new FileReader(f))) { String l; while ((l = r.readLine()) != null) { d.addLast(l); if (d.size() > n) d.removeFirst(); } } catch (IOException e) { return "(no audit trail yet)"; }
        return String.join("\n", d);
    }
}
