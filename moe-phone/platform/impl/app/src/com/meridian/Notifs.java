package com.meridian.app;

import android.app.*;
import android.content.*;
import android.os.*;
import android.service.notification.*;
import org.json.*;
import java.util.*;

/** Notification access for the agent (07_AGENT_RUNTIME.md section 7.2, "notifications"): list, reply through the app's own reply
 *  action, dismiss. Enabled by the user in Settings > Notification access; nothing here runs until then. Notification keys are long,
 *  so each listing hands out short ids (m1, m2 ...) that stay valid until the next listing. */
public final class Notifs extends NotificationListenerService {
    static volatile Notifs inst;
    static final Map<String, String> shortIds = new HashMap<>();

    @Override public void onListenerConnected() { inst = this; }
    @Override public void onListenerDisconnected() { inst = null; }

    static Notifs service(Context ctx) {
        Notifs n = inst; if (n != null) return n;
        try { ctx.startActivity(new Intent(Build.VERSION.SDK_INT >= 30 ? android.provider.Settings.ACTION_NOTIFICATION_LISTENER_DETAIL_SETTINGS : "android.settings.ACTION_NOTIFICATION_LISTENER_SETTINGS")
                .putExtra(android.provider.Settings.EXTRA_NOTIFICATION_LISTENER_COMPONENT_NAME, new ComponentName(ctx, Notifs.class).flattenToString()).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)); } catch (Exception ignored) { }
        throw new IllegalStateException("PermissionRequired: notification access is off. Opened the settings page: allow Meridian notification access, then ask again");
    }
    static String appLabel(Context ctx, String pkg) { try { return String.valueOf(ctx.getPackageManager().getApplicationLabel(ctx.getPackageManager().getApplicationInfo(pkg, 0))); } catch (Exception e) { return pkg; } }
    static CharSequence cs(Bundle e, String k) { CharSequence c = e.getCharSequence(k); return c == null ? "" : c; }

    /** Current notifications (newest first, Meridian's own and ongoing system ones excluded), optionally only from apps whose
     *  name or package contains `app`. */
    synchronized JSONArray list(String app) throws JSONException {
        String q = app == null ? "" : app.trim().toLowerCase(Locale.ROOT); if (q.matches("all|any|every.*|everything")) q = "";
        StatusBarNotification[] all = getActiveNotifications(); if (all == null) all = new StatusBarNotification[0];
        List<StatusBarNotification> l = new ArrayList<>(Arrays.asList(all)); Collections.sort(l, (x, y) -> Long.compare(y.getPostTime(), x.getPostTime()));
        shortIds.clear(); JSONArray out = new JSONArray(); int k = 0;
        for (StatusBarNotification s : l) {
            if (s.getPackageName().equals(getPackageName()) || (s.isOngoing() && !Notification.CATEGORY_CALL.equals(s.getNotification().category))) continue;
            Bundle e = s.getNotification().extras; String label = appLabel(this, s.getPackageName());
            if (!q.isEmpty() && !label.toLowerCase(Locale.ROOT).contains(q) && !s.getPackageName().toLowerCase(Locale.ROOT).contains(q)) continue;
            if ((s.getNotification().flags & Notification.FLAG_GROUP_SUMMARY) != 0) continue;
            String text = cs(e, Notification.EXTRA_BIG_TEXT).length() > 0 ? cs(e, Notification.EXTRA_BIG_TEXT).toString() : cs(e, Notification.EXTRA_TEXT).toString();
            CharSequence[] lines = e.getCharSequenceArray(Notification.EXTRA_TEXT_LINES); if (lines != null && lines.length > 0) text = android.text.TextUtils.join(" / ", lines);
            String id = "m" + (++k); shortIds.put(id, s.getKey());
            out.put(new JSONObject().put("id", id).put("app", label).put("title", cs(e, Notification.EXTRA_TITLE).toString()).put("text", Agent.clip(text.replaceAll("\\s+", " "), 240))
                .put("time", String.format(Locale.ROOT, "%1$tF %1$tR", s.getPostTime())).put("can_reply", replyAction(s.getNotification()) != null));
            if (out.length() >= 15) break;
        }
        return out;
    }
    static Notification.Action replyAction(Notification n) {
        if (n.actions == null) return null;
        for (Notification.Action a : n.actions) if (a.getRemoteInputs() != null) for (RemoteInput r : a.getRemoteInputs()) if (r.getAllowFreeFormInput()) return a;
        return null;
    }
    StatusBarNotification byId(String id) {
        String key = shortIds.get(id.trim()); if (key == null) throw new IllegalArgumentException("no notification " + id + " in the last listing; list notifications again");
        StatusBarNotification[] all = getActiveNotifications(); if (all != null) for (StatusBarNotification s : all) if (s.getKey().equals(key)) return s;
        throw new IllegalStateException("notification " + id + " is gone (it was dismissed or answered)");
    }
    /** Sends `text` through the notification's own reply action (the app delivers it). */
    JSONObject reply(String id, String text) throws Exception {
        StatusBarNotification s = byId(id); Notification.Action a = replyAction(s.getNotification());
        if (a == null) throw new IllegalStateException("this notification has no reply action; open the app instead");
        Intent fill = new Intent(); Bundle res = new Bundle(); for (RemoteInput r : a.getRemoteInputs()) res.putCharSequence(r.getResultKey(), text);
        RemoteInput.addResultsToIntent(a.getRemoteInputs(), fill, res);
        a.actionIntent.send(this, 0, fill);
        return new JSONObject().put("replied_to", cs(s.getNotification().extras, Notification.EXTRA_TITLE).toString()).put("app", appLabel(this, s.getPackageName())).put("key", s.getKey()).put("posted", s.getPostTime());
    }
    /** After a reply, messaging apps either remove the notification or re-post it with the reply appended; either is observable. */
    boolean replyLanded(String key, long posted, String text) {
        for (int i = 0; i < 20; i++) {
            StatusBarNotification hit = null; StatusBarNotification[] all = getActiveNotifications();
            if (all != null) for (StatusBarNotification s : all) if (s.getKey().equals(key)) hit = s;
            if (hit == null) return true;
            if (hit.getPostTime() != posted && hit.getNotification().extras.toString().contains(text)) return true;
            Parcelable[] msgs = hit.getNotification().extras.getParcelableArray(Notification.EXTRA_MESSAGES);
            if (msgs != null) for (Parcelable p : msgs) if (p instanceof Bundle && String.valueOf(((Bundle) p).getCharSequence("text")).contains(text)) return true;
            SystemClock.sleep(250);
        }
        return false;
    }
    boolean dismiss(String id) { StatusBarNotification s = byId(id); cancelNotification(s.getKey()); return true; }
    boolean active(String key) { StatusBarNotification[] all = getActiveNotifications(); if (all != null) for (StatusBarNotification s : all) if (s.getKey().equals(key)) return true; return false; }
}
