package com.meridian.app;

import android.app.*;
import android.content.*;
import android.database.Cursor;
import android.net.Uri;
import android.os.*;
import android.provider.*;
import org.json.*;
import java.util.*;
import java.util.concurrent.*;

/** Tools that reach other people or read what they sent: SMS send and read, phone calls, contact creation, notification listing,
 *  reply and dismiss. Everything that reaches a person has effects "communicate" and consent class "every_time": the consent text
 *  shows the exact recipient (name and resolved number) and the exact text, and Agent rate-limits communicate actions per task and
 *  per hour (07_AGENT_RUNTIME.md section 8). Each postcondition is read from the system, not from the tool's own return value. */
final class ToolsComms {
    private ToolsComms() { }

    /** Resolves a contact name or a number to {name, number}; a name matching several contacts is refused, not guessed. */
    static JSONObject recipient(Tools t, String to) throws Exception {
        if (Parse.looksLikeNumber(to)) return new JSONObject().put("name", "").put("number", to.replaceAll("[\\s()-]", ""));
        Perms.need(t.context(), "finding the contact", android.Manifest.permission.READ_CONTACTS);
        JSONArray m = t.contacts(to); if (m.length() == 0) throw new IllegalStateException("no contact matches '" + to + "'");
        String exact = null;
        for (int i = 0; i < m.length(); i++) { JSONObject c = m.getJSONObject(i); if (c.getString("name").equalsIgnoreCase(to.trim())) exact = c.toString(); }
        if (exact != null) return new JSONObject(exact);
        Set<String> names = new LinkedHashSet<>(); for (int i = 0; i < m.length(); i++) names.add(m.getJSONObject(i).getString("name"));
        if (names.size() > 1) throw new IllegalStateException("'" + to + "' matches several contacts " + names + "; say which one");
        return m.getJSONObject(0);
    }
    static String describe(Tools t, String to) { try { JSONObject r = recipient(t, to); return (r.optString("name").isEmpty() ? "" : r.optString("name") + " ") + "(" + r.optString("number") + ")"; } catch (Exception e) { return to + " (" + e.getMessage() + ")"; } }

    static JSONArray smsInbox(Context ctx, String from, int max) throws Exception {
        Perms.need(ctx, "reading your messages", android.Manifest.permission.READ_SMS);
        Set<String> keys = new HashSet<>(); String q = from == null ? "" : from.trim();
        if (!q.isEmpty() && !Parse.looksLikeNumber(q) && ctx.checkSelfPermission(android.Manifest.permission.READ_CONTACTS) == android.content.pm.PackageManager.PERMISSION_GRANTED) {
            try (Cursor c = ctx.getContentResolver().query(ContactsContract.CommonDataKinds.Phone.CONTENT_URI, new String[]{ContactsContract.CommonDataKinds.Phone.NUMBER},
                    ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME + " LIKE ?", new String[]{"%" + q + "%"}, null)) { while (c != null && c.moveToNext()) keys.add(Parse.digitsKey(c.getString(0))); } }
        if (Parse.looksLikeNumber(q)) keys.add(Parse.digitsKey(q));
        JSONArray out = new JSONArray();
        try (Cursor c = ctx.getContentResolver().query(Uri.parse("content://sms/inbox"), new String[]{"address", "body", "date", "read"}, null, null, "date DESC")) {
            while (c != null && c.moveToNext() && out.length() < max) { String addr = c.getString(0) == null ? "" : c.getString(0);
                boolean match = q.isEmpty() || keys.contains(Parse.digitsKey(addr)) || (!Parse.looksLikeNumber(q) && keys.isEmpty() && addr.toLowerCase(Locale.ROOT).contains(q.toLowerCase(Locale.ROOT)));
                if (!match) continue;
                out.put(new JSONObject().put("from", addr).put("text", Agent.clip(c.getString(1), 300)).put("time", String.format(Locale.ROOT, "%1$tF %1$tR", c.getLong(2))).put("unread", c.getInt(3) == 0)); } }
        return out;
    }
    static boolean inCall(Context ctx) { int m = ((android.media.AudioManager) ctx.getSystemService(Context.AUDIO_SERVICE)).getMode(); return m == android.media.AudioManager.MODE_IN_CALL || m == android.media.AudioManager.MODE_IN_COMMUNICATION; }

    static void register(final Tools t) throws JSONException {
        final Context ctx = t.context();
        t.add(new Tools.Tool("sms_send", "Send an SMS text message now to a phone number or contact name (the user approves the exact text first).", "communicate", false, "every_time",
                "the phone's radio reported every part of the SMS as sent (SENT result OK)", Tools.obj("to", "string", "text", "string")) {
            @Override String consentText(JSONObject a) { return "Send SMS to " + describe(t, a.optString("to")) + ":\n\"" + a.optString("text") + "\""; }
            JSONObject execute(JSONObject a) throws Exception {
                Perms.need(ctx, "sending an SMS", android.Manifest.permission.SEND_SMS);
                JSONObject r = recipient(t, a.getString("to")); String text = a.getString("text"); if (text.trim().isEmpty()) throw new IllegalArgumentException("empty message");
                android.telephony.SmsManager sm = Build.VERSION.SDK_INT >= 31 ? ctx.getSystemService(android.telephony.SmsManager.class) : android.telephony.SmsManager.getDefault();
                ArrayList<String> parts = sm.divideMessage(text); String action = ctx.getPackageName() + ".SMS_SENT." + System.nanoTime();
                final CountDownLatch l = new CountDownLatch(parts.size()); final List<Integer> codes = Collections.synchronizedList(new ArrayList<Integer>());
                BroadcastReceiver br = new BroadcastReceiver() { @Override public void onReceive(Context c, Intent i) { codes.add(getResultCode()); l.countDown(); } };
                if (Build.VERSION.SDK_INT >= 33) ctx.registerReceiver(br, new IntentFilter(action), Context.RECEIVER_NOT_EXPORTED); else ctx.registerReceiver(br, new IntentFilter(action));
                try { ArrayList<PendingIntent> sent = new ArrayList<>();
                    for (int i = 0; i < parts.size(); i++) sent.add(PendingIntent.getBroadcast(ctx, i, new Intent(action).setPackage(ctx.getPackageName()), PendingIntent.FLAG_IMMUTABLE));
                    sm.sendMultipartTextMessage(r.getString("number"), null, parts, sent, null); l.await(45, TimeUnit.SECONDS); }
                finally { ctx.unregisterReceiver(br); }
                return new JSONObject().put("to", r.optString("name").isEmpty() ? r.getString("number") : r.getString("name")).put("number", r.getString("number")).put("parts", parts.size())
                    .put("sent_ok", codes.size() == parts.size() && allOk(codes)).put("result_codes", new JSONArray(codes)); }
            boolean allOk(List<Integer> c) { for (int x : c) if (x != Activity.RESULT_OK) return false; return true; }
            boolean verify(JSONObject a, JSONObject r) { return r.optBoolean("sent_ok"); } });
        t.add(new Tools.Tool("phone_call", "Place a phone call now to a number or contact name (the user approves first).", "communicate", false, "every_time",
                "the phone entered a call within 8 seconds (audio mode in-call)", Tools.obj("to", "string")) {
            @Override String consentText(JSONObject a) { return "Call " + describe(t, a.optString("to")) + " now."; }
            JSONObject execute(JSONObject a) throws Exception {
                Perms.need(ctx, "placing a call", android.Manifest.permission.CALL_PHONE);
                JSONObject r = recipient(t, a.getString("to"));
                t.launch(new Intent(Intent.ACTION_CALL, Uri.parse("tel:" + Uri.encode(r.getString("number")))));
                boolean on = false; for (int i = 0; i < 32 && !on; i++) { SystemClock.sleep(250); on = inCall(ctx); }
                return new JSONObject().put("calling", r.optString("name").isEmpty() ? r.getString("number") : r.getString("name")).put("number", r.getString("number")).put("in_call", on); }
            boolean verify(JSONObject a, JSONObject r) { return r.optBoolean("in_call"); } });
        t.add(new Tools.Tool("contact_add", "Save a new contact with a name and phone number (email may be empty).", "write", true, "every_time",
                "a contact with this name and number is found in the contacts provider afterwards", Tools.obj("name", "string", "phone", "string", "email", "string")) {
            @Override String consentText(JSONObject a) { return "Save contact " + a.optString("name") + ", " + a.optString("phone") + (a.optString("email").isEmpty() ? "" : ", " + a.optString("email")) + " to this phone's contacts."; }
            JSONObject execute(JSONObject a) throws Exception {
                Perms.need(ctx, "saving a contact", android.Manifest.permission.WRITE_CONTACTS, android.Manifest.permission.READ_CONTACTS);
                if (!Parse.looksLikeNumber(a.getString("phone"))) throw new IllegalArgumentException("not a phone number: '" + a.getString("phone") + "'");
                ArrayList<ContentProviderOperation> ops = new ArrayList<>();
                ops.add(ContentProviderOperation.newInsert(ContactsContract.RawContacts.CONTENT_URI).withValue(ContactsContract.RawContacts.ACCOUNT_TYPE, null).withValue(ContactsContract.RawContacts.ACCOUNT_NAME, null).build());
                ops.add(ContentProviderOperation.newInsert(ContactsContract.Data.CONTENT_URI).withValueBackReference(ContactsContract.Data.RAW_CONTACT_ID, 0)
                    .withValue(ContactsContract.Data.MIMETYPE, ContactsContract.CommonDataKinds.StructuredName.CONTENT_ITEM_TYPE).withValue(ContactsContract.CommonDataKinds.StructuredName.DISPLAY_NAME, a.getString("name")).build());
                ops.add(ContentProviderOperation.newInsert(ContactsContract.Data.CONTENT_URI).withValueBackReference(ContactsContract.Data.RAW_CONTACT_ID, 0)
                    .withValue(ContactsContract.Data.MIMETYPE, ContactsContract.CommonDataKinds.Phone.CONTENT_ITEM_TYPE).withValue(ContactsContract.CommonDataKinds.Phone.NUMBER, a.getString("phone"))
                    .withValue(ContactsContract.CommonDataKinds.Phone.TYPE, ContactsContract.CommonDataKinds.Phone.TYPE_MOBILE).build());
                if (!a.getString("email").trim().isEmpty()) ops.add(ContentProviderOperation.newInsert(ContactsContract.Data.CONTENT_URI).withValueBackReference(ContactsContract.Data.RAW_CONTACT_ID, 0)
                    .withValue(ContactsContract.Data.MIMETYPE, ContactsContract.CommonDataKinds.Email.CONTENT_ITEM_TYPE).withValue(ContactsContract.CommonDataKinds.Email.ADDRESS, a.getString("email").trim()).build());
                ContentProviderResult[] res = ctx.getContentResolver().applyBatch(ContactsContract.AUTHORITY, ops);
                return new JSONObject().put("saved", a.getString("name")).put("phone", a.getString("phone")).put("raw_contact", String.valueOf(res[0].uri)); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { JSONArray m = t.contacts(a.getString("name")); String k = Parse.digitsKey(a.getString("phone"));
                for (int i = 0; i < m.length(); i++) if (Parse.digitsKey(m.getJSONObject(i).getString("number")).equals(k)) return true; return false; } });
        t.add(new Tools.Tool("sms_read", "Read recent received SMS messages, optionally only from one contact, number or sender name (empty for all).", "read", true, "none",
                "the returned messages re-query identically from the SMS inbox", Tools.obj("from", "string")) {
            JSONObject execute(JSONObject a) throws Exception { return new JSONObject().put("messages", smsInbox(ctx, a.getString("from"), 8)); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { return smsInbox(ctx, a.getString("from"), 8).toString().equals(r.getJSONArray("messages").toString()); } });
        t.add(new Tools.Tool("notifications_read", "Read the phone's current notifications (messages, missed calls, app alerts), optionally only from one app (empty for all). Each has an id like m1.", "read", true, "none",
                "every returned id maps to a notification key from the system's active list", Tools.obj("app", "string")) {
            @Override int resultChars() { return 1600; }
            JSONObject execute(JSONObject a) throws Exception { return new JSONObject().put("notifications", Notifs.service(ctx).list(a.getString("app"))); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { JSONArray x = r.getJSONArray("notifications");
                for (int i = 0; i < x.length(); i++) { String key = Notifs.shortIds.get(x.getJSONObject(i).getString("id")); if (key == null) return false; } return true; } });
        t.add(new Tools.Tool("notification_reply", "Reply to a message notification (id from notifications_read) with text, through that app's reply button (the user approves the exact text first).", "communicate", false, "every_time",
                "the app accepted the reply: the notification was cleared or re-posted containing the reply", Tools.obj("id", "string", "text", "string")) {
            @Override String consentText(JSONObject a) { String who = a.optString("id"); try { Notifs n = Notifs.service(ctx); android.service.notification.StatusBarNotification s = n.byId(a.optString("id"));
                    who = Notifs.cs(s.getNotification().extras, Notification.EXTRA_TITLE) + " in " + Notifs.appLabel(ctx, s.getPackageName()); } catch (Exception ignored) { }
                return "Reply to " + who + ":\n\"" + a.optString("text") + "\""; }
            JSONObject execute(JSONObject a) throws Exception { Notifs n = Notifs.service(ctx); JSONObject r = n.reply(a.getString("id"), a.getString("text"));
                return r.put("landed", n.replyLanded(r.getString("key"), r.getLong("posted"), a.getString("text"))); }
            boolean verify(JSONObject a, JSONObject r) { return r.optBoolean("landed"); } });
        t.add(new Tools.Tool("notification_dismiss", "Dismiss (clear) a notification by its id from notifications_read.", "write", false, "none",
                "the notification is no longer active", Tools.obj("id", "string")) {
            JSONObject execute(JSONObject a) throws Exception { Notifs n = Notifs.service(ctx); String key = Notifs.shortIds.get(a.getString("id").trim()); n.dismiss(a.getString("id")); SystemClock.sleep(500);
                return new JSONObject().put("dismissed", a.getString("id")).put("key", key); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { return !Notifs.service(ctx).active(r.getString("key")); } });
    }
}
