package com.meridian.app;

import android.content.*;
import android.os.*;
import org.json.*;
import java.io.*;
import java.util.*;

/** T3 sustained-load probe (04_DEVICE_PROFILING.md section 3.5): the real engine decodes a fixed prompt repeatedly for `seconds`, logging per run the rate,
 *  battery temperature (a PROXY for skin temperature, labelled as such), OS thermal status and headroom; then recovery is polled with short decodes.
 *  Outputs the derate curve (throughput fraction vs temperature), time-to-first-throttle and recovery time, all from measurement. */
public final class Thermal {
    public interface Progress { void step(String s); }
    static double battTempC(Context c) { Intent b = c.registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED)); return b == null ? Double.NaN : b.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Integer.MIN_VALUE) / 10.0; }
    static int status(Context c) { return Build.VERSION.SDK_INT >= 29 ? ((PowerManager) c.getSystemService(Context.POWER_SERVICE)).getCurrentThermalStatus() : -1; }
    static double headroom(Context c) { return Build.VERSION.SDK_INT >= 30 ? ((PowerManager) c.getSystemService(Context.POWER_SERVICE)).getThermalHeadroom(10) : Double.NaN; }
    static double median(List<Double> v) { List<Double> s = new ArrayList<>(v); Collections.sort(s); return s.get(s.size() / 2); }

    public static JSONObject run(Context ctx, Engine.Config cfg, int seconds, Progress prog) throws Exception {
        Regime regime = new Regime(ctx); regime.begin(); Chat chat = new Chat(ctx, cfg); JSONArray series = new JSONArray(); List<Double> cool = new ArrayList<>();
        long t0 = System.currentTimeMillis(); Double coolMin = null; Double throttleAt = null; int below = 0; double endTemp;
        try {
            prog.step("loading " + cfg.describe()); chat.start(600000); chat.ask("Hello", 8, true, null);
            int i = 0;
            while ((System.currentTimeMillis() - t0) / 1000 < seconds) {
                double tempBefore = battTempC(ctx); chat.ask("Count from one to forty in words.", 48, true, null); double ts = chat.lastResult.getDouble("tok_s"); double el = (System.currentTimeMillis() - t0) / 1000.0;
                series.put(new JSONObject().put("t_s", el).put("tok_s", ts).put("batt_c", battTempC(ctx)).put("batt_c_before", tempBefore).put("thermal_status", status(ctx)).put("headroom", Double.isNaN(headroom(ctx)) ? JSONObject.NULL : headroom(ctx)));
                if (i < 3) cool.add(ts); else { if (coolMin == null) coolMin = Collections.min(cool); if (ts < coolMin) { if (++below >= 2 && throttleAt == null) throttleAt = el - 0; } else below = 0; }
                prog.step(String.format("sustained %.0f/%d s: %.2f tok/s at %.1f C (status %d)", el, seconds, ts, battTempC(ctx), status(ctx))); i++;
            }
            endTemp = battTempC(ctx);
            // recovery: short decodes every 30 s until back inside the cool range (or 5 min)
            Double recovery = null; long r0 = System.currentTimeMillis(); double coolLo = Collections.min(cool);
            while ((System.currentTimeMillis() - r0) < 300000) { Thread.sleep(30000); chat.ask("Count from one to twenty in words.", 24, true, null); double ts = chat.lastResult.getDouble("tok_s");
                series.put(new JSONObject().put("t_s", (System.currentTimeMillis() - t0) / 1000.0).put("tok_s", ts).put("batt_c", battTempC(ctx)).put("thermal_status", status(ctx)).put("phase", "recovery"));
                prog.step(String.format("recovery %.0f s: %.2f tok/s", (System.currentTimeMillis() - r0) / 1000.0, ts)); if (ts >= coolLo) { recovery = (System.currentTimeMillis() - r0) / 1000.0; break; } }
            JSONObject cond = regime.end(); double coolMed = median(cool);
            TreeMap<Integer, List<Double>> bins = new TreeMap<>(); for (int k = 0; k < series.length(); k++) { JSONObject o = series.getJSONObject(k); if (o.has("phase")) continue; bins.computeIfAbsent((int) Math.floor(o.getDouble("batt_c")), x -> new ArrayList<>()).add(o.getDouble("tok_s") / coolMed); }
            JSONArray derate = new JSONArray(); for (Map.Entry<Integer, List<Double>> e : bins.entrySet()) derate.put(new JSONObject().put("batt_c", e.getKey()).put("throughput_frac", median(e.getValue())).put("n", e.getValue().size()));
            JSONObject res = new JSONObject().put("config", cfg.describe()).put("seconds", seconds).put("cool_tok_s_median", coolMed).put("cool_tok_s_min", Collections.min(cool)).put("cool_tok_s_max", Collections.max(cool))
                .put("time_to_throttle_s", throttleAt == null ? JSONObject.NULL : throttleAt).put("recovery_s", recovery == null ? JSONObject.NULL : recovery).put("temp_end_c", endTemp).put("derate", derate)
                .put("temperature_source", "battery temperature (proxy for skin; the skin thermistor is not readable by an app)").put("conditions", cond).put("series", series);
            try (FileWriter w = new FileWriter(new File(ctx.getFilesDir(), "thermal_t3.json"))) { w.write(res.toString(2)); }
            return res;
        } finally { chat.close(); }
    }
}
