package com.meridian.app;

import android.content.*;
import android.os.*;
import org.json.JSONException;
import org.json.JSONObject;
import java.util.*;

/** ValidityGuard (04_DEVICE_PROFILING.md section 5): samples the conditions a measurement was taken under.
 *  It never looks at measured values. In-regime = unplugged, awake, this app in the foreground and idle (T1 "quiesced"). */
public final class Regime {
    public static volatile boolean appForeground = false;   // set from Activity.onResume/onPause
    private final Context ctx; private volatile boolean stop;
    private final List<Boolean> interactive = Collections.synchronizedList(new ArrayList<Boolean>());
    private final List<Boolean> foreground = Collections.synchronizedList(new ArrayList<Boolean>());
    private int maxThermal = -1; private boolean everPlugged; private int battery = -1; private Thread t;

    public Regime(Context c) { ctx = c.getApplicationContext(); }

    private void sample() {
        PowerManager pm = (PowerManager) ctx.getSystemService(Context.POWER_SERVICE);
        interactive.add(pm.isInteractive());
        foreground.add(appForeground);
        if (Build.VERSION.SDK_INT >= 29) maxThermal = Math.max(maxThermal, pm.getCurrentThermalStatus());
        Intent b = ctx.registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
        if (b != null) {
            if (b.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0) everPlugged = true;
            int lvl = b.getIntExtra(BatteryManager.EXTRA_LEVEL, -1), sc = b.getIntExtra(BatteryManager.EXTRA_SCALE, 100);
            if (lvl >= 0) battery = lvl * 100 / sc;
        }
    }

    /** Instantaneous conditions (no sampling thread), for per-turn audit records. */
    public static JSONObject snapshot(Context c) throws JSONException {
        Regime r = new Regime(c); r.sample();
        boolean awake = r.interactive.get(0), fg = r.foreground.get(0);
        return new JSONObject().put("wakefulness", awake ? "awake" : "dozing").put("foreground", fg ? "self" : "other-app")
            .put("power", r.everPlugged ? "plugged" : "unplugged").put("thermal_status", r.maxThermal < 0 ? JSONObject.NULL : r.maxThermal)
            .put("in_regime", awake && fg && !r.everPlugged);
    }

    public void begin() {
        stop = false; sample();
        t = new Thread(() -> { while (!stop) { try { Thread.sleep(1000); } catch (InterruptedException e) { return; } sample(); } }, "regime");
        t.setDaemon(true); t.start();
    }

    /** Stops sampling; the WORST sample decides (one dozing/plugged/backgrounded sample contaminates the run). */
    public JSONObject end() throws JSONException {
        stop = true; if (t != null) t.interrupt(); sample();
        boolean awake = !interactive.contains(Boolean.FALSE), fg = !foreground.contains(Boolean.FALSE);
        JSONObject c = new JSONObject().put("wakefulness", awake ? "awake" : "dozing").put("foreground", fg ? "self" : "other-app")
            .put("power", everPlugged ? "plugged" : "unplugged").put("thermal_status", maxThermal < 0 ? JSONObject.NULL : maxThermal)
            .put("battery_pct", battery).put("n_samples", interactive.size());
        c.put("in_regime", awake && fg && !everPlugged);
        return c;
    }
}
