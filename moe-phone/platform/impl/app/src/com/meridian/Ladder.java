package com.meridian.app;

import org.json.*;
import java.util.*;

/** Pure ladder logic (no Android APIs, testable on the host JVM): which rung to enter next for a trigger, honouring hysteresis and the
 *  rule that a rung with unbounded quality cost is never entered automatically (03_INTERFACES.md section 4.2). */
public final class Ladder {
    public final JSONArray rungs; private final Set<Integer> applied = new LinkedHashSet<>(); private long lastAt = Long.MIN_VALUE;
    public Ladder(JSONArray rungs) { this.rungs = rungs; }
    public Set<Integer> applied() { return applied; }

    /** Next rung for `trigger` at time nowMs, or null if none is allowed yet (hysteresis) or none remain. */
    public JSONObject next(String trigger, long nowMs) throws JSONException {
        for (int i = 0; i < rungs.length(); i++) {
            JSONObject r = rungs.getJSONObject(i); if (!r.getString("trigger").equals(trigger) || applied.contains(i)) continue;
            if (r.getJSONObject("expected_cost").getString("quality").equals("unbounded")) continue;
            if (lastAt != Long.MIN_VALUE && nowMs - lastAt < r.getLong("hysteresis_ms")) return null;
            applied.add(i); lastAt = nowMs; return r;
        }
        return null;
    }
    /** Pressure has cleared: re-arm rungs of this trigger, but not before the hysteresis window since the last action. */
    public boolean rearm(String trigger, long nowMs, long clearSinceMs) throws JSONException {
        long h = 0; for (int i = 0; i < rungs.length(); i++) if (rungs.getJSONObject(i).getString("trigger").equals(trigger)) h = Math.max(h, rungs.getJSONObject(i).getLong("hysteresis_ms"));
        if (nowMs - clearSinceMs < h) return false;
        Iterator<Integer> it = applied.iterator(); while (it.hasNext()) if (rungs.getJSONObject(it.next()).getString("trigger").equals(trigger)) it.remove(); return true;
    }
}
