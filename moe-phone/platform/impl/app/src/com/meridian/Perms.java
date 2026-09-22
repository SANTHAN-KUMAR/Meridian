package com.meridian.app;

import android.app.Activity;
import android.content.*;
import android.content.pm.PackageManager;
import android.os.Bundle;
import java.util.*;
import java.util.concurrent.*;

/** Runtime permissions asked at first use (08_APP_AND_UX.md section 8: "the first task that needs each"), from the agent's
 *  worker thread, without touching the main UI: a translucent activity shows the system prompt and finishes. The caller blocks
 *  until the user answers (at most 90 s) and gets a PermissionRequired error if any permission is still missing. */
public final class Perms extends Activity {
    static volatile CountDownLatch pending; static volatile boolean shown;

    /** Returns when every permission in `perms` is granted; otherwise asks once and throws if the user declines. */
    static void need(Context ctx, String why, String... perms) throws Exception {
        List<String> miss = new ArrayList<>();
        for (String p : perms) if (ctx.checkSelfPermission(p) != PackageManager.PERMISSION_GRANTED) miss.add(p);
        if (miss.isEmpty()) return;
        CountDownLatch l = new CountDownLatch(1); pending = l; shown = false;
        ctx.startActivity(new Intent(ctx, Perms.class).putExtra("perms", miss.toArray(new String[0])).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK));
        // Android blocks activity starts from the background; if the prompt did not appear, say so now rather than wait
        if (!l.await(5, TimeUnit.SECONDS) && !shown) throw new IllegalStateException("PermissionRequired: " + why + " needs " + miss + "; the permission prompt could not be shown while Meridian is in the background. Open Meridian and ask again.");
        l.await(90, TimeUnit.SECONDS);
        List<String> still = new ArrayList<>(); for (String p : miss) if (ctx.checkSelfPermission(p) != PackageManager.PERMISSION_GRANTED) still.add(p.substring(p.lastIndexOf('.') + 1));
        if (!still.isEmpty()) throw new IllegalStateException("PermissionRequired: " + why + " needs " + still + ", which was not granted (Settings > Apps > Meridian > Permissions)");
    }
    /** Any one of `perms` is enough (e.g. precise or approximate location). */
    static void needAny(Context ctx, String why, String... perms) throws Exception {
        for (String p : perms) if (ctx.checkSelfPermission(p) == PackageManager.PERMISSION_GRANTED) return;
        try { need(ctx, why, perms); } catch (IllegalStateException e) { for (String p : perms) if (ctx.checkSelfPermission(p) == PackageManager.PERMISSION_GRANTED) return; throw e; }
    }

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b); shown = true;
        String[] p = getIntent().getStringArrayExtra("perms");
        if (p == null || p.length == 0) { done(); return; }
        requestPermissions(p, 11);
    }
    @Override public void onRequestPermissionsResult(int code, String[] p, int[] r) { done(); }
    void done() { CountDownLatch l = pending; if (l != null) l.countDown(); finish(); overridePendingTransition(0, 0); }
}
