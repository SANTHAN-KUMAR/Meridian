package com.meridian.app;

import android.app.AppOpsManager;
import android.app.usage.UsageEvents;
import android.app.usage.UsageStatsManager;
import android.content.*;
import android.os.*;
import org.json.*;
import java.util.*;

/** Experiment X1 / stub PL-S2 (04_DEVICE_PROFILING.md 3.4 item 3): how much memory can this app keep resident while a REAL target app is in the
 *  foreground? The memory-grant probe (memprobe) runs as a child of this app's foreground service while the target app is verified to be the
 *  foreground app through UsageStats events before, between and after the runs. The result is `measured` only if every sample was awake,
 *  unplugged and showed the target in front; without Usage Access the foreground cannot be verified and the result is `prior`. */
public final class ForegroundGrant {
    public interface Progress { void step(String s); }
    public static boolean usageAccess(Context c) { AppOpsManager a = (AppOpsManager) c.getSystemService(Context.APP_OPS_SERVICE); return a.checkOpNoThrow(AppOpsManager.OPSTR_GET_USAGE_STATS, android.os.Process.myUid(), c.getPackageName()) == AppOpsManager.MODE_ALLOWED; }

    /** Package of the most recent MOVE_TO_FOREGROUND event, or null if unknown. */
    static String foregroundPackage(Context c) {
        if (!usageAccess(c)) return null; UsageStatsManager u = (UsageStatsManager) c.getSystemService(Context.USAGE_STATS_SERVICE);
        long now = System.currentTimeMillis(); UsageEvents ev = u.queryEvents(now - 5 * 60000, now + 1000); UsageEvents.Event e = new UsageEvents.Event(); String pkg = null;
        while (ev.hasNextEvent()) { ev.getNextEvent(e); if (e.getEventType() == UsageEvents.Event.MOVE_TO_FOREGROUND) pkg = e.getPackageName(); else if (e.getEventType() == UsageEvents.Event.MOVE_TO_BACKGROUND && e.getPackageName().equals(pkg)) pkg = null; }
        return pkg;
    }

    public static JSONObject run(Context ctx, String target, Progress prog) throws Exception {
        boolean ua = usageAccess(ctx); Intent li = ctx.getPackageManager().getLaunchIntentForPackage(target);
        if (li == null) throw new IllegalArgumentException("target app not installed / not launchable: " + target);
        prog.step("launching " + target); ctx.startActivity(li.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)); Thread.sleep(8000);
        List<Double> grants = new ArrayList<>(); List<String> fg = new ArrayList<>(); boolean awake = true, plugged = false; PowerManager pm = (PowerManager) ctx.getSystemService(Context.POWER_SERVICE);
        for (int i = 0; i < 2; i++) {
            fg.add(String.valueOf(foregroundPackage(ctx))); awake &= pm.isInteractive();
            Intent b = ctx.registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED)); plugged |= b != null && b.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0;
            prog.step("memory grant run " + (i + 1) + "/2 with " + target + " in front"); String o = Native.run(ctx, "memprobe", Arrays.asList("--oom-adj", "0"), 120000, null, false);
            java.util.regex.Matcher m = java.util.regex.Pattern.compile("max_VmRSS_MB=(\\d+)").matcher(o); if (m.find()) grants.add(Double.parseDouble(m.group(1)) * 1048576.0);
            awake &= pm.isInteractive();
        }
        fg.add(String.valueOf(foregroundPackage(ctx)));
        boolean verified = ua, allTarget = true; for (String f : fg) allTarget &= f.equals(target);
        boolean inRegime = verified && allTarget && awake && !plugged && !grants.isEmpty();
        Collections.sort(grants); JSONObject cond = new JSONObject().put("foreground", verified ? (allTarget ? target : "other-app:" + fg) : "unverified (Usage Access not granted)").put("foreground_samples", new JSONArray(fg))
            .put("wakefulness", awake ? "awake" : "dozing").put("power", plugged ? "plugged" : "unplugged").put("in_regime", inRegime);
        JSONObject m = new JSONObject().put("value", grants.isEmpty() ? JSONObject.NULL : (long) (double) grants.get(grants.size() / 2)).put("provenance", grants.isEmpty() ? "unknown" : (inRegime ? "measured" : "prior")).put("confidence", inRegime ? 0.7 : 0.3)
            .put("source", "memprobe x" + grants.size() + " with " + target + " in the foreground (UsageStats-verified: " + verified + ")").put("conditions", cond);
        if (!grants.isEmpty()) m.put("interval", new JSONArray().put(grants.get(0)).put(grants.get(grants.size() - 1)));
        return m;
    }
}
