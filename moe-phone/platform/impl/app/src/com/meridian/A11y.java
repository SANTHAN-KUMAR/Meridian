package com.meridian.app;

import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.GestureDescription;
import android.graphics.*;
import android.os.*;
import android.view.*;
import android.view.accessibility.*;
import android.widget.*;
import java.util.*;
import java.util.concurrent.*;

/** The agent's eyes and hands inside other apps (07_AGENT_RUNTIME.md sections 5.1 and 7.2, "UI actions"). Enabled by the user in
 *  Settings > Accessibility > Meridian; nothing here runs until then. It reads the active window as a filtered, serialised node list
 *  with short per-snapshot ids, performs taps, typing, scrolls and global navigation on those ids, and shows the consent card on
 *  top of whatever app is in front (an accessibility overlay needs no extra permission), so approvals are visible mid-task.
 *  Password fields are never read: their text is replaced by "(password field)". */
public final class A11y extends AccessibilityService {
    static volatile A11y inst;
    volatile String fgPkg = ""; final java.util.concurrent.atomic.AtomicLong events = new java.util.concurrent.atomic.AtomicLong();
    private final Map<String, AccessibilityNodeInfo> ids = new HashMap<>(); private String lastPkg = "";
    private final Handler main = new Handler(Looper.getMainLooper());

    @Override protected void onServiceConnected() { inst = this; }
    @Override public boolean onUnbind(android.content.Intent i) { inst = null; return super.onUnbind(i); }
    @Override public void onDestroy() { inst = null; super.onDestroy(); }
    @Override public void onInterrupt() { }
    @Override public void onAccessibilityEvent(AccessibilityEvent e) {
        events.incrementAndGet();
        if (e.getEventType() == AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED && e.getPackageName() != null && !isOverlay) fgPkg = e.getPackageName().toString();
    }

    /** The app whose window is active (from the window list, falling back to the last window-state event). */
    String activePackage() {
        AccessibilityNodeInfo r = getRootInActiveWindow();
        if (r != null && r.getPackageName() != null) return r.getPackageName().toString();
        return fgPkg;
    }

    // ---------------- observation ----------------
    static final int MAX_NODES = 70, MAX_CHARS = 2000, MAX_LABEL = 70;
    /** One snapshot: header + one line per kept node, e.g. `n7 button "Send" [tap]`. Ids are valid until the next snapshot. */
    synchronized String snapshot() {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) throw new IllegalStateException("no readable window is active (the screen may be locked or showing a secure app)");
        ids.clear(); StringBuilder b = new StringBuilder(); int[] n = {0}; boolean[] cut = {false};
        lastPkg = root.getPackageName() == null ? "" : root.getPackageName().toString();
        walk(root, b, n, cut, 0);
        String label = lastPkg; try { label = String.valueOf(getPackageManager().getApplicationLabel(getPackageManager().getApplicationInfo(lastPkg, 0))); } catch (Exception ignored) { }
        return "app: " + label + " (" + lastPkg + ")\n" + (n[0] == 0 ? "(no readable elements: the app may draw its own content)\n" : b) + (cut[0] ? "... (more elements below; scroll to see them)\n" : "");
    }
    String signature() { try { AccessibilityNodeInfo r = getRootInActiveWindow(); if (r == null) return ""; StringBuilder b = new StringBuilder(); sig(r, b, 0); return b.toString(); } catch (Exception e) { return ""; } }
    private void sig(AccessibilityNodeInfo x, StringBuilder b, int d) {
        if (x == null || d > 40 || b.length() > 20000) return; b.append(x.getClassName()).append(x.getText()).append(x.isChecked()).append(x.getContentDescription()).append('|');
        for (int i = 0; i < x.getChildCount(); i++) sig(x.getChild(i), b, d + 1);
    }
    private void walk(AccessibilityNodeInfo x, StringBuilder b, int[] n, boolean[] cut, int depth) {
        if (x == null || depth > 40) return;
        if (!x.isVisibleToUser()) return;
        CharSequence t = x.getText(), d = x.getContentDescription(); CharSequence h = Build.VERSION.SDK_INT >= 26 ? x.getHintText() : null;
        String label = x.isPassword() ? "(password field)" : t != null && t.length() > 0 ? t.toString() : d != null && d.length() > 0 ? d.toString() : "";
        boolean tap = x.isClickable() || x.isLongClickable(), edit = x.isEditable(), check = x.isCheckable(), scroll = x.isScrollable();
        // unlabelled containers are skipped (a tap on any label walks up to its tappable parent): a WhatsApp chat row went from
        // ~7 lines to ~4, so the chat list fits the observation (15R, 2026-09-22); text fields and switches are kept even unlabelled
        if (!label.isEmpty() || edit || check) {
            if (n[0] >= MAX_NODES || b.length() > MAX_CHARS) { cut[0] = true; return; }
            String id = "n" + (++n[0]); ids.put(id, x);
            label = label.replaceAll("\\s+", " ").trim(); if (label.length() > MAX_LABEL) label = label.substring(0, MAX_LABEL) + "...";
            b.append(id).append(' ').append(role(x)).append(label.isEmpty() ? "" : " \"" + label + "\"");
            if (edit && h != null && h.length() > 0 && !h.toString().equals(label)) b.append(" hint=\"").append(h).append('"');
            if (tap) b.append(" [tap]"); if (edit) b.append(" [type]"); if (check) b.append(x.isChecked() ? " [on]" : " [off]"); if (scroll) b.append(" [scroll]");
            if (!x.isEnabled()) b.append(" [disabled]"); if (x.isSelected()) b.append(" [selected]");
            b.append('\n');
        }
        for (int i = 0; i < x.getChildCount(); i++) walk(x.getChild(i), b, n, cut, depth + 1);
    }
    static String role(AccessibilityNodeInfo x) {
        String c = x.getClassName() == null ? "" : x.getClassName().toString(); c = c.substring(c.lastIndexOf('.') + 1).toLowerCase(Locale.ROOT);
        if (x.isEditable() || c.contains("edittext")) return "field"; if (c.contains("switch") || c.contains("toggle")) return "switch"; if (c.contains("checkbox")) return "checkbox";
        if (c.contains("radio")) return "radio"; if (c.contains("button")) return "button"; if (c.contains("image")) return "image"; if (c.contains("recycler") || c.contains("list") || c.contains("scroll")) return "list";
        if (c.contains("tab")) return "tab"; return x.isClickable() ? "item" : "text";
    }

    // ---------------- targets ----------------
    /** `n12` from the last snapshot, or visible text/description (exact match preferred over contains). */
    synchronized AccessibilityNodeInfo find(String target) {
        String t = target.trim();
        if (t.matches("n\\d+")) { AccessibilityNodeInfo x = ids.get(t); if (x == null) throw new IllegalArgumentException("no element " + t + " in the last screen reading; read the screen again");
            if (!x.refresh()) throw new IllegalStateException("element " + t + " is no longer on screen; read the screen again"); return x; }
        AccessibilityNodeInfo root = getRootInActiveWindow(); if (root == null) throw new IllegalStateException("no readable window is active");
        List<AccessibilityNodeInfo> hits = root.findAccessibilityNodeInfosByText(t); AccessibilityNodeInfo best = null;
        for (AccessibilityNodeInfo h : hits) { if (!h.isVisibleToUser()) continue; String l = label(h);
            if (l.equalsIgnoreCase(t)) return h; if (best == null) best = h; }
        if (best == null) throw new IllegalArgumentException("nothing on screen reads '" + t + "'; read the screen and use an element id");
        return best;
    }
    static String label(AccessibilityNodeInfo x) { if (x.isPassword()) return "(password field)"; CharSequence t = x.getText(), d = x.getContentDescription();
        return t != null && t.length() > 0 ? t.toString() : d != null ? d.toString() : ""; }
    /** The label a tap on `target` would act on: the node's own text, else its nearest labelled descendant (for icon buttons). */
    String tapLabel(String target) {
        AccessibilityNodeInfo x = find(target); String l = label(x); if (!l.isEmpty()) return l;
        for (int i = 0; i < x.getChildCount(); i++) { AccessibilityNodeInfo c = x.getChild(i); if (c != null && !label(c).isEmpty()) return label(c); }
        return x.getViewIdResourceName() == null ? "" : x.getViewIdResourceName();
    }

    // ---------------- actions ----------------
    boolean tap(String target) {
        AccessibilityNodeInfo x = find(target), c = x;
        while (c != null && !c.isClickable()) c = c.getParent();
        if (c != null && c.performAction(AccessibilityNodeInfo.ACTION_CLICK)) return true;
        Rect r = new Rect(); x.getBoundsInScreen(r); return gesture(r.centerX(), r.centerY());
    }
    boolean gesture(int px, int py) {
        Path p = new Path(); p.moveTo(px, py); final CountDownLatch l = new CountDownLatch(1); final boolean[] ok = {false};
        dispatchGesture(new GestureDescription.Builder().addStroke(new GestureDescription.StrokeDescription(p, 0, 60)).build(),
            new GestureResultCallback() { @Override public void onCompleted(GestureDescription g) { ok[0] = true; l.countDown(); } @Override public void onCancelled(GestureDescription g) { l.countDown(); } }, main);
        try { l.await(3, TimeUnit.SECONDS); } catch (InterruptedException ignored) { } return ok[0];
    }
    boolean type(String target, String text) {
        AccessibilityNodeInfo x = find(target);
        if (!x.isEditable()) { AccessibilityNodeInfo f = x.findFocus(AccessibilityNodeInfo.FOCUS_INPUT); if (f != null && f.isEditable()) x = f; else throw new IllegalArgumentException("element '" + target + "' is not a text field"); }
        x.performAction(AccessibilityNodeInfo.ACTION_FOCUS);
        Bundle a = new Bundle(); a.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text);
        return x.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, a);
    }
    /** Text now in the field `target` refers to (re-read from the live node). */
    String fieldText(String target) { AccessibilityNodeInfo x = find(target); if (!x.isEditable()) { AccessibilityNodeInfo f = x.findFocus(AccessibilityNodeInfo.FOCUS_INPUT); if (f != null) x = f; }
        x.refresh(); return x.getText() == null ? "" : x.getText().toString(); }
    boolean scroll(String dir) {
        AccessibilityNodeInfo root = getRootInActiveWindow(); if (root == null) throw new IllegalStateException("no readable window is active");
        AccessibilityNodeInfo s = firstScrollable(root, 0); boolean fwd = !dir.toLowerCase(Locale.ROOT).matches(".*\\b(up|back|previous|left|top)\\b.*");
        if (s != null && s.performAction(fwd ? AccessibilityNodeInfo.ACTION_SCROLL_FORWARD : AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD)) return true;
        android.util.DisplayMetrics dm = getResources().getDisplayMetrics(); Path p = new Path(); int x = dm.widthPixels / 2, y0 = (int) (dm.heightPixels * (fwd ? 0.75 : 0.3)), y1 = (int) (dm.heightPixels * (fwd ? 0.3 : 0.75));
        p.moveTo(x, y0); p.lineTo(x, y1); final CountDownLatch l = new CountDownLatch(1); final boolean[] ok = {false};
        dispatchGesture(new GestureDescription.Builder().addStroke(new GestureDescription.StrokeDescription(p, 0, 350)).build(),
            new GestureResultCallback() { @Override public void onCompleted(GestureDescription g) { ok[0] = true; l.countDown(); } @Override public void onCancelled(GestureDescription g) { l.countDown(); } }, main);
        try { l.await(3, TimeUnit.SECONDS); } catch (InterruptedException ignored) { } return ok[0];
    }
    private AccessibilityNodeInfo firstScrollable(AccessibilityNodeInfo x, int d) {
        if (x == null || d > 40) return null; if (x.isScrollable() && x.isVisibleToUser()) return x;
        for (int i = 0; i < x.getChildCount(); i++) { AccessibilityNodeInfo r = firstScrollable(x.getChild(i), d + 1); if (r != null) return r; } return null;
    }
    boolean global(String what) {
        String w = what.toLowerCase(Locale.ROOT); int a = w.contains("home") ? GLOBAL_ACTION_HOME : w.contains("recent") ? GLOBAL_ACTION_RECENTS
            : w.contains("quick") ? GLOBAL_ACTION_QUICK_SETTINGS : w.contains("notif") ? GLOBAL_ACTION_NOTIFICATIONS : w.contains("back") ? GLOBAL_ACTION_BACK : -1;
        if (a < 0) throw new IllegalArgumentException("unknown navigation '" + what + "': use back, home, recents, notifications or quick settings");
        return performGlobalAction(a);
    }
    /** Waits up to `ms` for the screen signature to differ from `before`; true if it changed. */
    boolean changedSince(String before, long ms) { long end = SystemClock.uptimeMillis() + ms; while (SystemClock.uptimeMillis() < end) { if (!signature().equals(before)) return true; SystemClock.sleep(150); } return !signature().equals(before); }

    // ---------------- consent on top of any app ----------------
    private volatile boolean isOverlay;
    /** Blocks the calling (agent) thread until the user answers; denied after 120 s with no answer. */
    boolean askOnTop(final String title, final String body) {
        final CountDownLatch l = new CountDownLatch(1); final boolean[] ok = {false}; final View[] v = {null};
        main.post(() -> {
            float den = getResources().getDisplayMetrics().density;
            LinearLayout card = new LinearLayout(this); card.setOrientation(LinearLayout.VERTICAL); int pad = (int) (18 * den); card.setPadding(pad, pad, pad, pad);
            android.graphics.drawable.GradientDrawable bg = new android.graphics.drawable.GradientDrawable(); bg.setColor(0xFF1E1F24); bg.setCornerRadius(16 * den); card.setBackground(bg);
            TextView t = new TextView(this); t.setText(title); t.setTextColor(0xFFFFFFFF); t.setTextSize(18); t.setTypeface(Typeface.DEFAULT_BOLD); card.addView(t);
            TextView m = new TextView(this); m.setText(body); m.setTextColor(0xFFD0D3DA); m.setTextSize(15); m.setPadding(0, (int) (8 * den), 0, (int) (12 * den)); card.addView(m);
            LinearLayout row = new LinearLayout(this); row.setGravity(Gravity.END);
            Button no = new Button(this); no.setText("Deny"); no.setMinHeight((int) (48 * den)); Button yes = new Button(this); yes.setText("Allow once"); yes.setMinHeight((int) (48 * den));
            row.addView(no); row.addView(yes); card.addView(row);
            WindowManager.LayoutParams lp = new WindowManager.LayoutParams(WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY, WindowManager.LayoutParams.FLAG_DIM_BEHIND | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN, PixelFormat.TRANSLUCENT);
            lp.gravity = Gravity.BOTTOM; lp.dimAmount = 0.5f; lp.x = 0; lp.y = (int) (24 * den);
            FrameLayout wrap = new FrameLayout(this); int side = (int) (12 * den); wrap.setPadding(side, 0, side, 0); wrap.addView(card); v[0] = wrap;
            WindowManager wm = (WindowManager) getSystemService(WINDOW_SERVICE);
            View.OnClickListener close = btn -> { ok[0] = btn == yes; try { wm.removeView(wrap); } catch (Exception ignored) { } isOverlay = false; l.countDown(); };
            no.setOnClickListener(close); yes.setOnClickListener(close);
            try { isOverlay = true; wm.addView(wrap, lp); } catch (Exception e) { isOverlay = false; l.countDown(); }
        });
        try { if (!l.await(120, TimeUnit.SECONDS)) main.post(() -> { try { ((WindowManager) getSystemService(WINDOW_SERVICE)).removeView(v[0]); } catch (Exception ignored) { } isOverlay = false; }); } catch (InterruptedException ignored) { }
        return ok[0];
    }
}
