package com.meridian.app;

import android.content.*;
import org.json.*;
import java.util.*;

/** Tools that read and operate other apps' screens through the accessibility service (A11y). Consent: the first action in an app
 *  during a task asks "operate <app> for this task?" (class `once`, scoped to that app and task, never carried to another task);
 *  a tap whose label sends, pays, deletes or submits asks every time with the label shown (Parse.sensitive). Every action's
 *  postcondition is read from the live screen afterwards, and the new screen is returned as the next observation. */
final class ToolsScreen {
    private ToolsScreen() { }
    static final Set<String> NAMES = new HashSet<>(Arrays.asList("screen_read", "screen_tap", "screen_type", "screen_scroll", "screen_nav"));

    static A11y service(Context ctx) {
        A11y a = A11y.inst; if (a != null) return a;
        try { ctx.startActivity(new Intent(android.provider.Settings.ACTION_ACCESSIBILITY_SETTINGS).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)); } catch (Exception ignored) { }
        throw new IllegalStateException("PermissionRequired: screen control is off. Opened Settings > Accessibility: turn on Meridian there (on Android 13+ for a sideloaded app, first allow 'restricted settings' in App info > menu), then ask again");
    }
    /** Refuse to act on Meridian itself: the agent's own screen is not a task target. */
    static A11y target(Context ctx) {
        A11y a = service(ctx); String p = a.activePackage();
        if (ctx.getPackageName().equals(p)) throw new IllegalStateException("Meridian itself is in front; open the app to work in first (open_app), then read the screen");
        return a;
    }
    static String appLabel(Context ctx, String pkg) { try { return String.valueOf(ctx.getPackageManager().getApplicationLabel(ctx.getPackageManager().getApplicationInfo(pkg, 0))); } catch (Exception e) { return pkg; } }

    static abstract class ScreenTool extends Tools.Tool {
        final Context ctx;
        ScreenTool(Context ctx, String n, String d, String eff, String consent, String post, JSONObject p) { super(n, d, eff, true, consent, post, p); this.ctx = ctx; }
        @Override int resultChars() { return 2200; }
        @Override boolean observes() { return true; }
        /** once per app per task; the scope names the app so the consent text can say which one */
        @Override String consentScope(JSONObject a) { A11y s = A11y.inst; return s == null ? null : "screen:" + s.activePackage(); }
        @Override String consentText(JSONObject a) { A11y s = A11y.inst; String app = s == null ? "the app in front" : appLabel(ctx, s.activePackage());
            return "Let Meridian read and operate " + app + " for this task (taps, typing, scrolling). Anything that sends, pays or deletes still asks you each time."; }
        JSONObject withScreen(JSONObject r, A11y a) throws JSONException { SystemClockSleep(600); return r.put("screen", a.snapshot()); }
        static void SystemClockSleep(long ms) { android.os.SystemClock.sleep(ms); }
    }

    static void register(Tools t) throws JSONException {
        final Context ctx = t.context();
        t.add(new ScreenTool(ctx, "screen_read", "Read what is on the screen of the app in front: elements with ids like n5, their text and whether they can be tapped or typed in.", "read", "once",
                "the returned elements were read from the live window of the app in front", Tools.obj()) {
            JSONObject execute(JSONObject a) throws Exception { A11y s = target(ctx); return new JSONObject().put("screen", s.snapshot()); }
            boolean verify(JSONObject a, JSONObject r) { A11y s = A11y.inst; return s != null && r.optString("screen").startsWith("app: ") && r.optString("screen").contains("(" + s.activePackage() + ")"); } });
        t.add(new ScreenTool(ctx, "screen_tap", "Tap an element on the screen by its id from screen_read (like n5) or by its visible text.", "write", "once",
                "the screen changed after the tap (or the tapped switch flipped)", Tools.obj("target", "string")) {
            @Override String consentFor(JSONObject a) { try { return Parse.sensitive(target(ctx).tapLabel(a.optString("target"))) ? "every_time" : consent; } catch (Exception e) { return consent; } }
            @Override String consentText(JSONObject a) { try { A11y s = target(ctx); String l = s.tapLabel(a.optString("target"));
                    if (Parse.sensitive(l)) return "Tap \"" + l + "\" in " + appLabel(ctx, s.activePackage()) + ". This may send, pay, delete or submit something."; } catch (Exception ignored) { }
                return super.consentText(a); }
            JSONObject execute(JSONObject a) throws Exception { A11y s = target(ctx); String tgt = a.getString("target"); String label = s.tapLabel(tgt);
                android.view.accessibility.AccessibilityNodeInfo n = s.find(tgt); Boolean was = n.isCheckable() ? n.isChecked() : null; String before = s.signature();
                boolean ok = s.tap(tgt); boolean changed = s.changedSince(before, 2500);
                JSONObject r = new JSONObject().put("tapped", label).put("accepted", ok).put("screen_changed", changed);
                if (was != null) { n.refresh(); r.put("switch_now", n.isChecked()).put("switch_before", was); }
                return withScreen(r, s); }
            boolean verify(JSONObject a, JSONObject r) { return r.optBoolean("accepted") && (r.has("switch_now") ? r.optBoolean("switch_now") != r.optBoolean("switch_before") || r.optBoolean("screen_changed") : r.optBoolean("screen_changed")); } });
        t.add(new ScreenTool(ctx, "screen_type", "Type text into a text field on the screen (by id from screen_read or its visible text/hint). Replaces the field's text.", "write", "once",
                "the field reads back exactly the typed text", Tools.obj("target", "string", "text", "string")) {
            JSONObject execute(JSONObject a) throws Exception { A11y s = target(ctx); boolean ok = s.type(a.getString("target"), a.getString("text")); SystemClockSleep(300);
                return withScreen(new JSONObject().put("typed", a.getString("text")).put("accepted", ok).put("field_now", s.fieldText(a.getString("target"))), s); }
            boolean verify(JSONObject a, JSONObject r) { return r.optBoolean("accepted") && a.optString("text").equals(r.optString("field_now")); } });
        t.add(new ScreenTool(ctx, "screen_scroll", "Scroll the screen of the app in front: direction down or up.", "ui", "once",
                "the screen content changed after the scroll", Tools.obj("direction", "string")) {
            JSONObject execute(JSONObject a) throws Exception { A11y s = target(ctx); String before = s.signature(); boolean ok = s.scroll(a.getString("direction"));
                return withScreen(new JSONObject().put("scrolled", a.getString("direction")).put("accepted", ok).put("screen_changed", s.changedSince(before, 2000)), s); }
            boolean verify(JSONObject a, JSONObject r) { return r.optBoolean("accepted") && r.optBoolean("screen_changed"); } });
        t.add(new ScreenTool(ctx, "screen_nav", "Press a system navigation key: back, home, recents, notifications or quick settings.", "ui", "none",
                "the screen changed after the key", Tools.obj("key", "string")) {
            @Override String consentScope(JSONObject a) { return null; }
            JSONObject execute(JSONObject a) throws Exception { A11y s = service(ctx); String before = s.signature(); boolean ok = s.global(a.getString("key"));
                JSONObject r = new JSONObject().put("pressed", a.getString("key")).put("accepted", ok).put("screen_changed", s.changedSince(before, 2000));
                SystemClockSleep(600); String p = s.activePackage(); r.put("app_in_front", appLabel(ctx, p));
                if (!ctx.getPackageName().equals(p)) try { r.put("screen", s.snapshot()); } catch (Exception e) { r.put("screen", "(unreadable: " + e.getMessage() + ")"); }
                return r; }
            boolean verify(JSONObject a, JSONObject r) { return r.optBoolean("accepted") && r.optBoolean("screen_changed"); } });
    }
}
