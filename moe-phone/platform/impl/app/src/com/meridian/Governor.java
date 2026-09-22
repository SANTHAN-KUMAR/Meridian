package com.meridian.app;

import android.content.Context;
import android.os.*;
import org.json.*;
import java.util.*;
import java.util.concurrent.Executor;

/** RuntimeGovernor (02_ARCHITECTURE.md section 3.7): compares every turn with the plan's registered prediction, invalidates a falsified plan,
 *  and steps down the plan's own ladder on thermal / memory / latency pressure (with hysteresis) instead of improvising. It never blocks decode:
 *  it acts on turn boundaries and reacts to OS callbacks. A falsification is data: it is written to the audit trail and upgrades the profile. */
public final class Governor {
    public interface Host { void applyRung(JSONObject rung, String why); void invalidated(String why); }
    private final Context ctx; private final Host host; private final Recorder rec; private final JSONObject plan; private final Ladder ladder;
    private final Deque<Double> recent = new ArrayDeque<>(); private volatile int thermal = 0; private long thermalClearSince = -1;
    private PowerManager.OnThermalStatusChangedListener tl; public boolean falsified;

    public Governor(Context c, Host h, Recorder r, JSONObject plan) throws JSONException { ctx = c.getApplicationContext(); host = h; rec = r; this.plan = plan; ladder = new Ladder(plan.getJSONArray("ladder")); }
    public JSONObject plan() { return plan; }

    private final Deque<Long> pressure = new ArrayDeque<>(); private long started = System.currentTimeMillis();
    boolean nextIsDestructive() throws JSONException { for (int i = 0; i < ladder.rungs.length(); i++) { JSONObject r = ladder.rungs.getJSONObject(i);
            if (r.getString("trigger").equals("memory") && !ladder.applied().contains(i)) return r.getString("action").matches("suspend|refuse"); } return false; }
    public void start() { started = System.currentTimeMillis();
        if (Build.VERSION.SDK_INT < 29) return;
        PowerManager pm = (PowerManager) ctx.getSystemService(Context.POWER_SERVICE);
        tl = status -> onThermal(status, System.currentTimeMillis());
        Executor ex = r -> new Handler(Looper.getMainLooper()).post(r); pm.addThermalStatusListener(ex, tl);
    }
    public void stop() { if (Build.VERSION.SDK_INT >= 29 && tl != null) ((PowerManager) ctx.getSystemService(Context.POWER_SERVICE)).removeThermalStatusListener(tl); }

    /** Thermal status per android.os.PowerManager: 0 none, 1 light, 2 moderate, 3 severe... The plan is valid while status <= valid_while.thermal_status_max. */
    void onThermal(int status, long now) {
        thermal = status;
        try {
            int max = plan.getJSONObject("valid_while").getInt("thermal_status_max");
            rec.write(new JSONObject().put("event", "thermal_status").put("status", status).put("t", now / 1000.0));
            if (status > max) { thermalClearSince = -1; JSONObject r = ladder.next("thermal", now); if (r != null) host.applyRung(r, "thermal status " + status + " > " + max); }
            else { if (thermalClearSince < 0) thermalClearSince = now; if (ladder.rearm("thermal", now, thermalClearSince)) rec.write(new JSONObject().put("event", "ladder_rearmed").put("trigger", "thermal")); }
        } catch (JSONException ignored) { }
    }

    /** Memory pressure from the lease heartbeat or onTrimMemory. */
    public void onMemoryPressure(String why) { try {
        // 2026-09-22 (D-10): with the screen locked the OS trims the engine's file-backed (mmap'ed model) pages; they fault back in
        // from flash, so the right response is none. The resident plan's memory ladder is a single "refuse" rung, and entering it
        // unloaded the engine in the middle of an agent task. A reclaim, or any pressure while the phone is not interactive, is now
        // recorded without entering a rung.
        boolean interactive = ((PowerManager) ctx.getSystemService(Context.POWER_SERVICE)).isInteractive();
        if (why.contains("shrank") || !interactive) { rec.write(new JSONObject().put("event", "memory_pressure").put("why", why).put("rung", JSONObject.NULL)
                .put("deferred", why.contains("shrank") ? "reclaimed file-backed pages fault back in" : "screen off: recorded only")); return; }
        // destructive rungs (suspend, refuse) unload the engine: they need SUSTAINED pressure (>= 3 events in 120 s) and never fire in
        // the first 90 s after the plan starts, when the model load itself pushes other apps' pages to zram (15R: +91 MiB, then suspend)
        long now = System.currentTimeMillis(); pressure.addLast(now); while (!pressure.isEmpty() && now - pressure.peekFirst() > 120_000) pressure.removeFirst();
        if (nextIsDestructive() && Agent.running.get() > 0) { rec.write(new JSONObject().put("event", "memory_pressure").put("why", why).put("rung", JSONObject.NULL)
                .put("deferred", "an agent task is running; unloading the engine now would kill it")); return; }   // D-11
        if (nextIsDestructive() && (now - started < 90_000 || pressure.size() < 3)) { rec.write(new JSONObject().put("event", "memory_pressure").put("why", why).put("rung", JSONObject.NULL)
                .put("deferred", "destructive rung needs sustained pressure (" + pressure.size() + " of 3 events in 120 s" + (now - started < 90_000 ? ", within 90 s of load" : "") + ")")); return; }
        JSONObject r = ladder.next("memory", now); rec.write(new JSONObject().put("event", "memory_pressure").put("why", why).put("rung", r == null ? JSONObject.NULL : r.getString("action")));
        if (r != null) host.applyRung(r, why); } catch (JSONException ignored) { } }

    /** Observed decode rate for one turn. Falsified when >= min_observations of the last `window` fall outside the registered interval. */
    public void observe(double decodeTokS) {
        try {
            JSONObject f = plan.getJSONObject("falsification"); recent.addLast(decodeTokS); while (recent.size() > f.getInt("window")) recent.removeFirst();
            int out = 0; for (double v : recent) if (v < f.getDouble("lo") || v > f.getDouble("hi")) out++;
            if (!falsified && out >= f.getInt("min_observations")) {
                falsified = true; String why = out + " of the last " + recent.size() + " turns outside the predicted decode interval [" + String.format("%.2f", f.getDouble("lo")) + ", " + String.format("%.2f", f.getDouble("hi")) + "] tok/s";
                rec.write(new JSONObject().put("event", "plan_falsified").put("plan_id", plan.optString("plan_id")).put("why", why).put("observed", new JSONArray(recent)).put("t", System.currentTimeMillis() / 1000.0));
                host.invalidated(why);
            }
        } catch (JSONException ignored) { }
    }
}
