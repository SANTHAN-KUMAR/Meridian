package com.meridian.app;

import android.content.*;
import org.json.*;
import java.io.*;

/** Debug-build tool smoke test (ignored unless the APK is debuggable): runs ONE tool with given arguments, no model involved, and
 *  appends {tool, args, result, verified, ms} to files/tool_probe.jsonl.
 *    adb shell am broadcast -n com.meridian.app/.ToolProbe --es tool weather --es args '{"place":"Chennai"}'
 *  Tools whose consent class is not "none" are refused here: this path has no consent gate, so it may not run them. */
public final class ToolProbe extends BroadcastReceiver {
    @Override public void onReceive(final Context c, final Intent i) {
        if ((c.getApplicationInfo().flags & android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) == 0) return;
        final PendingResult pr = goAsync();
        new Thread(() -> {
            JSONObject row = new JSONObject();
            try { String name = i.getStringExtra("tool"); JSONObject args = new JSONObject(i.getStringExtra("args") == null ? "{}" : i.getStringExtra("args"));
                row.put("tool", name).put("args", args);
                Tools.Tool t = new Tools(c).registry.get(name);
                if (t == null) row.put("error", "no such tool");
                else if (!t.consentFor(args).equals("none")) row.put("error", "refused: consent class " + t.consentFor(args) + " needs the agent's gate");
                else { String bad = Tools.validate(t, args); if (bad != null) row.put("error", bad);
                    else { long t0 = System.currentTimeMillis(); JSONObject r; boolean v;
                        try { r = t.execute(args); v = t.verify(args, r); } catch (Exception e) { r = new JSONObject().put("error", String.valueOf(e.getMessage())); v = false; }
                        row.put("result", r).put("verified", v).put("postcondition", t.postcondition).put("ms", System.currentTimeMillis() - t0); } }
            } catch (Exception e) { try { row.put("error", String.valueOf(e)); } catch (JSONException ignored) { } }
            try (FileWriter w = new FileWriter(new File(c.getFilesDir(), "tool_probe.jsonl"), true)) { w.write(row.toString() + "\n"); } catch (IOException ignored) { }
            pr.finish();
        }).start();
    }
}
