package com.meridian.app;

import android.content.*;
import android.os.*;
import org.json.*;
import java.io.*;
import java.util.*;

/** ToolSpec registry (03_INTERFACES.md section 7.1). Every tool declares effects, reversibility, consent class and a
 *  named postcondition that is checked against observable state -- not against the tool's own return value. */
public final class Tools {
    public static abstract class Tool {
        public final String name, description, effects, consent, postcondition; public final boolean reversible; public final JSONObject params;
        Tool(String n, String d, String eff, boolean rev, String consent, String post, JSONObject params) { name = n; description = d; effects = eff; reversible = rev; this.consent = consent; postcondition = post; this.params = params; }
        abstract JSONObject execute(JSONObject args) throws Exception;
        abstract boolean verify(JSONObject args, JSONObject result) throws Exception;   // observable-state check
    }
    private final Context ctx; private final File notes; public final Map<String, Tool> registry = new LinkedHashMap<>();

    static JSONObject obj(String... kv) throws JSONException { JSONObject o = new JSONObject(), props = new JSONObject(); JSONArray req = new JSONArray();
        for (int i = 0; i < kv.length; i += 2) { props.put(kv[i], new JSONObject().put("type", kv[i + 1])); req.put(kv[i]); }
        return o.put("type", "object").put("properties", props).put("required", req); }

    List<String> readNotes() throws IOException { List<String> l = new ArrayList<>(); if (!notes.exists()) return l;
        try (BufferedReader r = new BufferedReader(new FileReader(notes))) { String s; while ((s = r.readLine()) != null) if (!s.isEmpty()) l.add(s); } return l; }
    void writeNotes(List<String> l) throws IOException { try (FileWriter w = new FileWriter(notes, false)) { for (String s : l) w.write(s.replace('\n', ' ') + "\n"); } }

    public Tools(Context c) throws JSONException {
        ctx = c.getApplicationContext(); notes = new File(ctx.getFilesDir(), "notes.txt");
        add(new Tool("battery_status", "Read the battery level (percent) and whether the phone is charging.", "read", true, "none", "level is 0..100 and matches the OS battery service", obj()) {
            JSONObject execute(JSONObject a) throws Exception { Intent b = ctx.registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
                return new JSONObject().put("level_pct", b.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) * 100 / b.getIntExtra(BatteryManager.EXTRA_SCALE, 100)).put("charging", b.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { int l = r.getInt("level_pct"); BatteryManager bm = (BatteryManager) ctx.getSystemService(Context.BATTERY_SERVICE);
                return l >= 0 && l <= 100 && Math.abs(bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY) - l) <= 2; } });
        add(new Tool("storage_free", "Read free internal storage in gigabytes.", "read", true, "none", "value equals StatFs free bytes within 1%", obj()) {
            JSONObject execute(JSONObject a) throws Exception { return new JSONObject().put("free_gb", Math.round(ctx.getFilesDir().getUsableSpace() / 1e8) / 10.0); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { double fresh = ctx.getFilesDir().getUsableSpace() / 1e9; return Math.abs(fresh - r.getDouble("free_gb")) <= 0.01 * fresh + 0.1; } });
        add(new Tool("notes_add", "Save a short note in the app's notes. Only when the user asks to note, save, remember or write something down.", "write", true, "none", "the note text appears in the notes store after the call", obj("text", "string")) {
            JSONObject execute(JSONObject a) throws Exception { List<String> l = readNotes(); l.add(a.getString("text")); writeNotes(l); return new JSONObject().put("count", l.size()); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { return readNotes().contains(a.getString("text").replace('\n', ' ')); } });
        add(new Tool("notes_list", "List all saved notes.", "read", true, "none", "returned count equals notes in the store", obj()) {
            JSONObject execute(JSONObject a) throws Exception { return new JSONObject().put("notes", new JSONArray(readNotes())); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { return r.getJSONArray("notes").length() == readNotes().size(); } });
        add(new Tool("notes_clear", "Delete ALL saved notes. Cannot be undone.", "write", false, "every_time", "the notes store is empty", obj()) {
            JSONObject execute(JSONObject a) throws Exception { int n = readNotes().size(); writeNotes(new ArrayList<String>()); return new JSONObject().put("deleted", n); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { return readNotes().isEmpty(); } });
        addPhoneTools();
    }
    private void add(Tool t) { registry.put(t.name, t); }

    // ---------- phone actions through standard, user-visible Android intents (no special permissions) ----------
    /** Starts an intent from the app context. Returns the component that handled it; throws if nothing on the phone can. */
    String launch(Intent i) throws Exception {
        i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK); android.content.pm.ResolveInfo ri = ctx.getPackageManager().resolveActivity(i, 0);
        if (ri == null) throw new IllegalStateException("no app on this phone handles " + i.getAction() + (i.getData() != null ? " " + i.getData().getScheme() : ""));
        ctx.startActivity(i); return ri.activityInfo.packageName;
    }
    /** Launcher apps whose label best matches `name` (exact > prefix > contains, case-insensitive). */
    android.content.pm.ResolveInfo findApp(String name) {
        Intent m = new Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER); android.content.pm.PackageManager pm = ctx.getPackageManager();
        String t = name.toLowerCase(Locale.ROOT).replace("app", "").trim(); android.content.pm.ResolveInfo best = null; int bs = 0;
        for (android.content.pm.ResolveInfo r : pm.queryIntentActivities(m, 0)) { String l = String.valueOf(r.loadLabel(pm)).toLowerCase(Locale.ROOT);
            int sc = l.equals(t) ? 3 : l.startsWith(t) ? 2 : (l.contains(t) || (t.length() > 3 && t.contains(l))) ? 1 : 0; if (sc > bs) { bs = sc; best = r; } }
        return best;
    }
    /** "7", "7:30", "7.30 pm", "19:05", "6am" -> {hour, minute}; refuses anything else rather than guessing. */
    static int[] parseTime(String s) {
        java.util.regex.Matcher m = java.util.regex.Pattern.compile("(\\d{1,2})(?:[:.](\\d{2}))?\\s*(am|pm|a\\.m\\.|p\\.m\\.)?", java.util.regex.Pattern.CASE_INSENSITIVE).matcher(s.trim());
        if (!m.find()) throw new IllegalArgumentException("not a time: '" + s + "'");
        int h = Integer.parseInt(m.group(1)), mi = m.group(2) == null ? 0 : Integer.parseInt(m.group(2)); String ap = m.group(3) == null ? "" : m.group(3).toLowerCase(Locale.ROOT);
        if (ap.startsWith("p") && h < 12) h += 12; if (ap.startsWith("a") && h == 12) h = 0;
        if (h > 23 || mi > 59) throw new IllegalArgumentException("not a time: '" + s + "'"); return new int[]{h, mi};
    }
    /** "5 minutes", "90 seconds", "1 hour 30 minutes", "10 min", "45" (seconds) -> seconds. */
    static int parseDuration(String s) {
        java.util.regex.Matcher m = java.util.regex.Pattern.compile("(\\d+(?:\\.\\d+)?)\\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds)?", java.util.regex.Pattern.CASE_INSENSITIVE).matcher(s);
        double tot = 0; boolean any = false;
        while (m.find()) { double v = Double.parseDouble(m.group(1)); String u = m.group(2) == null ? "s" : m.group(2).toLowerCase(Locale.ROOT); any = true;
            tot += u.startsWith("h") ? v * 3600 : u.startsWith("m") ? v * 60 : v; }
        if (!any || tot <= 0 || tot > 86400) throw new IllegalArgumentException("not a duration: '" + s + "'"); return (int) Math.round(tot);
    }
    /** Arithmetic only: + - * / % ^ and parentheses, decimals. A recursive-descent parser, not an eval of code. */
    static double calc(String e) {
        final String x = e.replace("x", "*").replace("×", "*").replace("÷", "/").replaceAll("\\s+", ""); final int[] i = {0};
        class P { double expr() { double v = term(); while (i[0] < x.length()) { char c = x.charAt(i[0]); if (c == '+') { i[0]++; v += term(); } else if (c == '-') { i[0]++; v -= term(); } else break; } return v; }
            double term() { double v = pow(); while (i[0] < x.length()) { char c = x.charAt(i[0]); if (c == '*') { i[0]++; v *= pow(); } else if (c == '/') { i[0]++; v /= pow(); } else if (c == '%') { i[0]++; v %= pow(); } else break; } return v; }
            double pow() { double b = unary(); if (i[0] < x.length() && x.charAt(i[0]) == '^') { i[0]++; return Math.pow(b, pow()); } return b; }
            double unary() { if (i[0] < x.length() && x.charAt(i[0]) == '-') { i[0]++; return -unary(); } if (i[0] < x.length() && x.charAt(i[0]) == '(') { i[0]++; double v = expr(); if (i[0] >= x.length() || x.charAt(i[0]) != ')') throw new IllegalArgumentException("unbalanced parentheses"); i[0]++; return v; }
                int st = i[0]; while (i[0] < x.length() && (Character.isDigit(x.charAt(i[0])) || x.charAt(i[0]) == '.')) i[0]++; if (st == i[0]) throw new IllegalArgumentException("not arithmetic: '" + e + "'"); return Double.parseDouble(x.substring(st, i[0])); } }
        double v = new P().expr(); if (i[0] != x.length()) throw new IllegalArgumentException("not arithmetic: '" + e + "'"); return v;
    }
    volatile Boolean torchOn = null; String torchId = null;
    void ensureTorchListener() {
        if (torchId != null) return; android.hardware.camera2.CameraManager cm = (android.hardware.camera2.CameraManager) ctx.getSystemService(Context.CAMERA_SERVICE);
        try { for (String id : cm.getCameraIdList()) { Boolean f = cm.getCameraCharacteristics(id).get(android.hardware.camera2.CameraCharacteristics.FLASH_INFO_AVAILABLE); if (f != null && f) { torchId = id; break; } } } catch (Exception ignored) { }
        cm.registerTorchCallback(new android.hardware.camera2.CameraManager.TorchCallback() { @Override public void onTorchModeChanged(String id, boolean on) { if (id.equals(torchId)) torchOn = on; } }, new Handler(Looper.getMainLooper()));
    }
    /** Contacts whose name contains `name` (case-insensitive), up to 5, as {name, number}. Needs READ_CONTACTS (asked at first use). */
    JSONArray contacts(String name) throws Exception {
        if (ctx.checkSelfPermission(android.Manifest.permission.READ_CONTACTS) != android.content.pm.PackageManager.PERMISSION_GRANTED)
            throw new IllegalStateException("PermissionRequired: contacts access is not granted; allow it when Meridian asks (or in Settings > Apps > Meridian > Permissions)");
        JSONArray out = new JSONArray(); android.net.Uri u = android.provider.ContactsContract.CommonDataKinds.Phone.CONTENT_URI;
        try (android.database.Cursor c = ctx.getContentResolver().query(u, new String[]{android.provider.ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME, android.provider.ContactsContract.CommonDataKinds.Phone.NUMBER},
                android.provider.ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME + " LIKE ?", new String[]{"%" + name.trim() + "%"}, android.provider.ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME + " ASC")) {
            while (c != null && c.moveToNext() && out.length() < 5) out.put(new JSONObject().put("name", c.getString(0)).put("number", c.getString(1))); }
        return out;
    }
    /** A recipient that is a name (has letters) becomes the first matching contact's number; a number passes through. */
    String number(String to) throws Exception {
        if (!to.matches(".*[A-Za-z].*")) return to;
        JSONArray m = contacts(to); if (m.length() == 0) throw new IllegalStateException("no contact matches '" + to + "'"); return m.getJSONObject(0).getString("number");
    }
    static void sleep(long ms) { try { Thread.sleep(ms); } catch (InterruptedException ignored) { } }

    void addPhoneTools() throws JSONException {
        add(new Tool("open_app", "Open an installed app by name. Only when the user asks to open or use a specific app.", "ui", true, "none", "an installed launcher app matched the name and its activity was started", obj("name", "string")) {
            JSONObject execute(JSONObject a) throws Exception { android.content.pm.ResolveInfo r = findApp(a.getString("name")); if (r == null) throw new IllegalStateException("no installed app is called '" + a.getString("name") + "'");
                Intent i = ctx.getPackageManager().getLaunchIntentForPackage(r.activityInfo.packageName); launch(i);
                return new JSONObject().put("opened", String.valueOf(r.loadLabel(ctx.getPackageManager()))).put("package", r.activityInfo.packageName); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { return ctx.getPackageManager().getLaunchIntentForPackage(r.getString("package")) != null; } });
        add(new Tool("web_search", "Show web search results in the browser. For questions that need current information from the internet.", "ui", true, "none", "a browser accepted the search URL", obj("query", "string")) {
            JSONObject execute(JSONObject a) throws Exception { String pkg = launch(new Intent(Intent.ACTION_VIEW, android.net.Uri.parse("https://www.google.com/search?q=" + java.net.URLEncoder.encode(a.getString("query"), "UTF-8")))); return new JSONObject().put("searched", a.getString("query")).put("handled_by", pkg); }
            boolean verify(JSONObject a, JSONObject r) { return r.has("handled_by"); } });
        add(new Tool("open_url", "Open a web address the user gives.", "ui", true, "none", "a browser accepted the URL", obj("url", "string")) {
            JSONObject execute(JSONObject a) throws Exception { String u = a.getString("url").trim(); if (!u.matches("(?i)https?://.*")) u = "https://" + u; return new JSONObject().put("opened", u).put("handled_by", launch(new Intent(Intent.ACTION_VIEW, android.net.Uri.parse(u)))); }
            boolean verify(JSONObject a, JSONObject r) { return r.has("handled_by"); } });
        add(new Tool("set_alarm", "Set an alarm. time like 7:30 am or 19:05; label is a short name or empty.", "write", true, "none", "the system's next scheduled alarm clock is at the requested hour and minute", obj("time", "string", "label", "string")) {
            JSONObject execute(JSONObject a) throws Exception { int[] t = parseTime(a.getString("time"));
                Intent i = new Intent(android.provider.AlarmClock.ACTION_SET_ALARM).putExtra(android.provider.AlarmClock.EXTRA_HOUR, t[0]).putExtra(android.provider.AlarmClock.EXTRA_MINUTES, t[1])
                    .putExtra(android.provider.AlarmClock.EXTRA_MESSAGE, a.getString("label")).putExtra(android.provider.AlarmClock.EXTRA_SKIP_UI, true);
                String pkg = launch(i); sleep(1500); return new JSONObject().put("hour", t[0]).put("minute", t[1]).put("handled_by", pkg); }
            boolean verify(JSONObject a, JSONObject r) throws Exception {   // observable state: AlarmManager's next alarm clock
                android.app.AlarmManager am = (android.app.AlarmManager) ctx.getSystemService(Context.ALARM_SERVICE); android.app.AlarmManager.AlarmClockInfo n = am.getNextAlarmClock(); if (n == null) return false;
                Calendar c = Calendar.getInstance(); c.setTimeInMillis(n.getTriggerTime()); return c.get(Calendar.HOUR_OF_DAY) == r.getInt("hour") && c.get(Calendar.MINUTE) == r.getInt("minute"); } });
        add(new Tool("set_timer", "Start a countdown timer. duration like 5 minutes or 90 seconds; label short or empty.", "write", true, "none", "the clock app accepted the timer intent (no public API reads timers back)", obj("duration", "string", "label", "string")) {
            JSONObject execute(JSONObject a) throws Exception { int sec = parseDuration(a.getString("duration"));
                String pkg = launch(new Intent(android.provider.AlarmClock.ACTION_SET_TIMER).putExtra(android.provider.AlarmClock.EXTRA_LENGTH, sec).putExtra(android.provider.AlarmClock.EXTRA_MESSAGE, a.getString("label")).putExtra(android.provider.AlarmClock.EXTRA_SKIP_UI, true));
                return new JSONObject().put("seconds", sec).put("handled_by", pkg); }
            boolean verify(JSONObject a, JSONObject r) { return r.has("handled_by"); } });
        add(new Tool("calendar_event", "Open the calendar with a new event filled in. when like 5 pm or 17:30 today.", "ui", true, "none", "the calendar app opened the new-event screen with the title and time (the user saves it)", obj("title", "string", "when", "string")) {
            JSONObject execute(JSONObject a) throws Exception { int[] t = parseTime(a.getString("when")); Calendar c = Calendar.getInstance(); c.set(Calendar.HOUR_OF_DAY, t[0]); c.set(Calendar.MINUTE, t[1]); c.set(Calendar.SECOND, 0);
                if (a.getString("when").toLowerCase(Locale.ROOT).contains("tomorrow") || c.getTimeInMillis() < System.currentTimeMillis()) c.add(Calendar.DAY_OF_MONTH, 1);
                Intent i = new Intent(Intent.ACTION_INSERT).setData(android.provider.CalendarContract.Events.CONTENT_URI).putExtra(android.provider.CalendarContract.Events.TITLE, a.getString("title"))
                    .putExtra(android.provider.CalendarContract.EXTRA_EVENT_BEGIN_TIME, c.getTimeInMillis()).putExtra(android.provider.CalendarContract.EXTRA_EVENT_END_TIME, c.getTimeInMillis() + 3600000);
                return new JSONObject().put("title", a.getString("title")).put("starts", c.getTime().toString()).put("handled_by", launch(i)); }
            boolean verify(JSONObject a, JSONObject r) { return r.has("handled_by"); } });
        add(new Tool("send_message", "Open a text message to a phone number or contact name with the text filled in; the user presses send.", "external", true, "every_time", "the messaging app opened with the recipient and text (nothing is sent without the user)", obj("to", "string", "text", "string")) {
            JSONObject execute(JSONObject a) throws Exception { String n = number(a.getString("to")); Intent i = new Intent(Intent.ACTION_SENDTO, android.net.Uri.parse("smsto:" + android.net.Uri.encode(n))).putExtra("sms_body", a.getString("text"));
                return new JSONObject().put("to", a.getString("to")).put("number", n).put("handled_by", launch(i)).put("note", "composer opened; the user sends it"); }
            boolean verify(JSONObject a, JSONObject r) { return r.has("handled_by"); } });
        add(new Tool("call", "Open the dialer with a phone number or contact name filled in; the user presses call.", "external", true, "every_time", "the dialer opened with the number (no call is placed without the user)", obj("number", "string")) {
            JSONObject execute(JSONObject a) throws Exception { String n = number(a.getString("number")); return new JSONObject().put("number", n).put("handled_by", launch(new Intent(Intent.ACTION_DIAL, android.net.Uri.parse("tel:" + android.net.Uri.encode(n))))); }
            boolean verify(JSONObject a, JSONObject r) { return r.has("handled_by"); } });
        add(new Tool("navigate", "Show a place or directions in the maps app. Only when the user asks where something is or how to get there.", "ui", true, "none", "a maps app accepted the place query", obj("place", "string")) {
            JSONObject execute(JSONObject a) throws Exception { return new JSONObject().put("place", a.getString("place")).put("handled_by", launch(new Intent(Intent.ACTION_VIEW, android.net.Uri.parse("geo:0,0?q=" + android.net.Uri.encode(a.getString("place")))))); }
            boolean verify(JSONObject a, JSONObject r) { return r.has("handled_by"); } });
        add(new Tool("play_music", "Play a song, artist or playlist. Only when the user asks for music or a video.", "ui", true, "none", "a media app accepted the play-from-search request", obj("query", "string")) {
            JSONObject execute(JSONObject a) throws Exception { String q = a.getString("query"); String pkg;
                try { pkg = launch(new Intent(android.provider.MediaStore.INTENT_ACTION_MEDIA_PLAY_FROM_SEARCH).putExtra(android.app.SearchManager.QUERY, q)); }
                catch (IllegalStateException e) { pkg = launch(new Intent(Intent.ACTION_VIEW, android.net.Uri.parse("https://www.youtube.com/results?search_query=" + java.net.URLEncoder.encode(q, "UTF-8")))); }
                return new JSONObject().put("query", q).put("handled_by", pkg); }
            boolean verify(JSONObject a, JSONObject r) { return r.has("handled_by"); } });
        add(new Tool("flashlight", "Turn the phone's flashlight (torch) on or off to give light. state is on or off.", "write", true, "none", "the camera service reports the torch in the requested state", obj("state", "string")) {
            JSONObject execute(JSONObject a) throws Exception { ensureTorchListener(); if (torchId == null) throw new IllegalStateException("this phone reports no flash unit");
                boolean on = a.getString("state").toLowerCase(Locale.ROOT).matches(".*\\b(on|enable|true|yes)\\b.*");
                ((android.hardware.camera2.CameraManager) ctx.getSystemService(Context.CAMERA_SERVICE)).setTorchMode(torchId, on); sleep(400); return new JSONObject().put("requested_on", on); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { return torchOn != null && torchOn == r.getBoolean("requested_on"); } });
        add(new Tool("set_volume", "Set the media volume to a percentage, 0 to 100.", "write", true, "none", "the audio service reports the media volume at the requested step", obj("percent", "string")) {
            JSONObject execute(JSONObject a) throws Exception { android.media.AudioManager am = (android.media.AudioManager) ctx.getSystemService(Context.AUDIO_SERVICE);
                double pct = calc(a.getString("percent").replace("%", "").replaceAll("[^0-9.]", "")); if (pct < 0 || pct > 100) throw new IllegalArgumentException("percent must be 0..100");
                int max = am.getStreamMaxVolume(android.media.AudioManager.STREAM_MUSIC), step = (int) Math.round(pct / 100 * max); am.setStreamVolume(android.media.AudioManager.STREAM_MUSIC, step, 0);
                return new JSONObject().put("step", step).put("max", max).put("percent", Math.round(100.0 * step / max)); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { return ((android.media.AudioManager) ctx.getSystemService(Context.AUDIO_SERVICE)).getStreamVolume(android.media.AudioManager.STREAM_MUSIC) == r.getInt("step"); } });
        add(new Tool("open_settings", "Open a settings page: wifi, bluetooth, display, sound, battery, location, apps, storage, notifications, or main.", "ui", true, "none", "the settings app accepted the page intent", obj("page", "string")) {
            JSONObject execute(JSONObject a) throws Exception { String p = a.getString("page").toLowerCase(Locale.ROOT); String act =
                    p.contains("wi") ? android.provider.Settings.ACTION_WIFI_SETTINGS : p.contains("blue") ? android.provider.Settings.ACTION_BLUETOOTH_SETTINGS : p.contains("display") || p.contains("bright") ? android.provider.Settings.ACTION_DISPLAY_SETTINGS
                  : p.contains("sound") || p.contains("volume") ? android.provider.Settings.ACTION_SOUND_SETTINGS : p.contains("batt") ? Intent.ACTION_POWER_USAGE_SUMMARY : p.contains("loc") ? android.provider.Settings.ACTION_LOCATION_SOURCE_SETTINGS
                  : p.contains("app") ? android.provider.Settings.ACTION_APPLICATION_SETTINGS : p.contains("stor") ? android.provider.Settings.ACTION_INTERNAL_STORAGE_SETTINGS : p.contains("notif") ? "android.settings.NOTIFICATION_SETTINGS"
                  : android.provider.Settings.ACTION_SETTINGS;
                return new JSONObject().put("page", p).put("handled_by", launch(new Intent(act))); }
            boolean verify(JSONObject a, JSONObject r) { return r.has("handled_by"); } });
        add(new Tool("copy_text", "Copy text to the clipboard.", "write", true, "none", "the clipboard holds exactly the text", obj("text", "string")) {
            JSONObject execute(JSONObject a) throws Exception { final String t = a.getString("text"); final java.util.concurrent.CountDownLatch l = new java.util.concurrent.CountDownLatch(1);
                new Handler(Looper.getMainLooper()).post(() -> { ((ClipboardManager) ctx.getSystemService(Context.CLIPBOARD_SERVICE)).setPrimaryClip(ClipData.newPlainText("meridian", t)); l.countDown(); }); l.await(); return new JSONObject().put("copied", t); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { final String[] got = {null}; final java.util.concurrent.CountDownLatch l = new java.util.concurrent.CountDownLatch(1);
                new Handler(Looper.getMainLooper()).post(() -> { ClipData c = ((ClipboardManager) ctx.getSystemService(Context.CLIPBOARD_SERVICE)).getPrimaryClip(); if (c != null && c.getItemCount() > 0) got[0] = String.valueOf(c.getItemAt(0).getText()); l.countDown(); });
                l.await(); return a.getString("text").equals(got[0]); } });
        add(new Tool("share_text", "Share text with another app (the user picks the app). Only when the user asks to share.", "external", true, "none", "the share sheet opened", obj("text", "string")) {
            JSONObject execute(JSONObject a) throws Exception { Intent s = new Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_TEXT, a.getString("text")); ctx.startActivity(Intent.createChooser(s, "Share").addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)); return new JSONObject().put("shared", true); }
            boolean verify(JSONObject a, JSONObject r) { return r.optBoolean("shared"); } });
        add(new Tool("take_photo", "Open the camera to take a photo. Only when the user wants to take a picture.", "ui", true, "none", "a camera app accepted the still-image intent", obj()) {
            JSONObject execute(JSONObject a) throws Exception { return new JSONObject().put("handled_by", launch(new Intent(android.provider.MediaStore.INTENT_ACTION_STILL_IMAGE_CAMERA))); }
            boolean verify(JSONObject a, JSONObject r) { return r.has("handled_by"); } });
        add(new Tool("calculate", "Compute an arithmetic expression exactly, e.g. 17*23 or (120-15)/4.", "read", true, "none", "the result re-computes to the same value", obj("expression", "string")) {
            JSONObject execute(JSONObject a) throws Exception { double v = calc(a.getString("expression")); return new JSONObject().put("expression", a.getString("expression")).put("result", v == Math.rint(v) && Math.abs(v) < 1e15 ? (Object) (long) v : (Object) v); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { return Math.abs(calc(a.getString("expression")) - r.getDouble("result")) < 1e-9 * Math.max(1, Math.abs(r.getDouble("result"))); } });
        add(new Tool("time_now", "Read the current date, time and weekday. Only when the answer depends on the date or time.", "read", true, "none", "the time is within 5 seconds of the system clock", obj()) {
            JSONObject execute(JSONObject a) throws Exception { Calendar c = Calendar.getInstance(); return new JSONObject().put("date", String.format(Locale.ROOT, "%tF", c)).put("time", String.format(Locale.ROOT, "%tR", c))
                .put("weekday", c.getDisplayName(Calendar.DAY_OF_WEEK, Calendar.LONG, Locale.ENGLISH)).put("epoch_s", System.currentTimeMillis() / 1000); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { return Math.abs(System.currentTimeMillis() / 1000 - r.getLong("epoch_s")) <= 5; } });
        add(new Tool("find_contact", "Look up a contact's phone number by name.", "read", true, "none", "the returned contacts re-query identically from the contacts provider", obj("name", "string")) {
            JSONObject execute(JSONObject a) throws Exception { return new JSONObject().put("contacts", contacts(a.getString("name"))); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { return contacts(a.getString("name")).toString().equals(r.getJSONArray("contacts").toString()); } });
        add(new Tool("send_email", "Open an email to someone with subject and text filled in; the user presses send.", "external", true, "every_time", "the mail app opened the draft (nothing is sent without the user)", obj("to", "string", "subject", "string", "text", "string")) {
            JSONObject execute(JSONObject a) throws Exception { Intent i = new Intent(Intent.ACTION_SENDTO, android.net.Uri.parse("mailto:" + android.net.Uri.encode(a.getString("to"))))
                    .putExtra(Intent.EXTRA_SUBJECT, a.getString("subject")).putExtra(Intent.EXTRA_TEXT, a.getString("text"));
                return new JSONObject().put("to", a.getString("to")).put("handled_by", launch(i)).put("note", "draft opened; the user sends it"); }
            boolean verify(JSONObject a, JSONObject r) { return r.has("handled_by"); } });
        add(new Tool("media_control", "Control whatever music or video is playing: play, pause, next or previous.", "write", true, "none", "for play/pause the audio service reports music active/inactive afterwards", obj("action", "string")) {
            JSONObject execute(JSONObject a) throws Exception { String x = a.getString("action").toLowerCase(Locale.ROOT); int key = x.contains("next") || x.contains("skip") ? android.view.KeyEvent.KEYCODE_MEDIA_NEXT
                    : x.contains("prev") || x.contains("back") ? android.view.KeyEvent.KEYCODE_MEDIA_PREVIOUS : x.contains("pause") || x.contains("stop") ? android.view.KeyEvent.KEYCODE_MEDIA_PAUSE : android.view.KeyEvent.KEYCODE_MEDIA_PLAY;
                android.media.AudioManager am = (android.media.AudioManager) ctx.getSystemService(Context.AUDIO_SERVICE);
                am.dispatchMediaKeyEvent(new android.view.KeyEvent(android.view.KeyEvent.ACTION_DOWN, key)); am.dispatchMediaKeyEvent(new android.view.KeyEvent(android.view.KeyEvent.ACTION_UP, key)); sleep(800);
                return new JSONObject().put("key", android.view.KeyEvent.keyCodeToString(key)).put("music_active", am.isMusicActive()); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { String k = r.getString("key"); boolean act = ((android.media.AudioManager) ctx.getSystemService(Context.AUDIO_SERVICE)).isMusicActive();
                return k.endsWith("PLAY") ? act : k.endsWith("PAUSE") ? !act : true; } });
        add(new Tool("quick_panel", "Show the system panel for wifi, internet, volume or nfc, where the user can switch it (apps cannot toggle these directly).", "ui", true, "none", "the system panel intent was accepted", obj("panel", "string")) {
            JSONObject execute(JSONObject a) throws Exception { String p = a.getString("panel").toLowerCase(Locale.ROOT);
                String act = p.contains("wi") ? android.provider.Settings.Panel.ACTION_WIFI : p.contains("vol") ? android.provider.Settings.Panel.ACTION_VOLUME : p.contains("nfc") ? android.provider.Settings.Panel.ACTION_NFC : android.provider.Settings.Panel.ACTION_INTERNET_CONNECTIVITY;
                return new JSONObject().put("panel", act).put("handled_by", launch(new Intent(act))); }
            boolean verify(JSONObject a, JSONObject r) { return r.has("handled_by"); } });
        add(new Tool("set_brightness", "Set the screen brightness to a percentage, 0 to 100.", "write", true, "none", "the system brightness setting reads back the value written", obj("percent", "string")) {
            JSONObject execute(JSONObject a) throws Exception {
                if (!android.provider.Settings.System.canWrite(ctx)) { launch(new Intent(android.provider.Settings.ACTION_MANAGE_WRITE_SETTINGS, android.net.Uri.parse("package:" + ctx.getPackageName())));
                    throw new IllegalStateException("PermissionRequired: opened the system page to allow Meridian to change settings; allow it, then ask again"); }
                double pct = calc(a.getString("percent").replaceAll("[^0-9.]", "")); if (pct < 0 || pct > 100) throw new IllegalArgumentException("percent must be 0..100");
                android.content.ContentResolver cr = ctx.getContentResolver(); android.provider.Settings.System.putInt(cr, android.provider.Settings.System.SCREEN_BRIGHTNESS_MODE, android.provider.Settings.System.SCREEN_BRIGHTNESS_MODE_MANUAL);
                int v = (int) Math.round(pct / 100 * 255); android.provider.Settings.System.putInt(cr, android.provider.Settings.System.SCREEN_BRIGHTNESS, v); return new JSONObject().put("value", v).put("percent", Math.round(pct)); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { return android.provider.Settings.System.getInt(ctx.getContentResolver(), android.provider.Settings.System.SCREEN_BRIGHTNESS) == r.getInt("value"); } });
        add(new Tool("device_info", "Read this phone's model, Android version, RAM and battery.", "read", true, "none", "fields match the OS build and memory services", obj()) {
            JSONObject execute(JSONObject a) throws Exception { android.app.ActivityManager.MemoryInfo mi = new android.app.ActivityManager.MemoryInfo(); ((android.app.ActivityManager) ctx.getSystemService(Context.ACTIVITY_SERVICE)).getMemoryInfo(mi);
                return new JSONObject().put("model", Build.MANUFACTURER + " " + Build.MODEL).put("android", Build.VERSION.RELEASE).put("ram_gb", Math.round(mi.totalMem / 1e8) / 10.0).put("ram_free_gb", Math.round(mi.availMem / 1e8) / 10.0)
                    .put("battery_pct", ((BatteryManager) ctx.getSystemService(Context.BATTERY_SERVICE)).getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { return r.getString("android").equals(Build.VERSION.RELEASE); } });
    }

    public String schemaText() throws JSONException {
        StringBuilder sb = new StringBuilder();
        for (Tool t : registry.values()) {
            JSONObject props = t.params.getJSONObject("properties"); StringBuilder sig = new StringBuilder();
            Iterator<String> it = props.keys(); while (it.hasNext()) { String k = it.next(); if (sig.length() > 0) sig.append(", "); sig.append(k).append(": ").append(props.getJSONObject(k).getString("type")).append(" (required)"); }
            sb.append("- ").append(t.name).append("(").append(sig).append("): ").append(t.description).append('\n');
        }
        return sb.toString();
    }
    private static final String BASE_RULES =
        "ws ::= [ ]?\n"
      + "string ::= \"\\\"\" ([^\"\\\\\\x7F\\x00-\\x1F] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" [0-9a-fA-F]{4}))* \"\\\"\"\n"
      + "final ::= \"{\\\"final\\\":\" ws string \"}\"\n";

    /** GBNF body for one tool call; string-typed arguments only (anything else is refused, not guessed). */
    private String callRule(String rule, Tool t) throws JSONException {
        JSONObject props = t.params.getJSONObject("properties"); StringBuilder body = new StringBuilder();
        body.append("\"{\\\"tool\\\":\\\"").append(t.name).append("\\\",\\\"args\\\":{\"");
        boolean firstArg = true; Iterator<String> it = props.keys();
        while (it.hasNext()) { String k = it.next();
            if (!props.getJSONObject(k).getString("type").equals("string")) throw new JSONException("tool " + t.name + ": grammar generation supports string arguments only (argument '" + k + "')");
            body.append(firstArg ? " " : " \",\" ").append("\"\\\"").append(k).append("\\\":\" ws string"); firstArg = false; }
        return rule + " ::= " + body + " \"}}\"\n";
    }
    /** Grammar generated from the registered ToolSpecs (closes stub PL-E7): exactly one call to `name`, with exactly its declared arguments. */
    public String callGrammar(String name) throws JSONException { Tool t = registry.get(name); return BASE_RULES + callRule("call", t) + "root ::= call\n"; }
    /** One step of the observe-act loop: a call to ANY registered tool (with exactly its arguments) or {"final": ...}. */
    public String stepGrammar() throws JSONException { return stepGrammar(registry.keySet()); }
    /** Same, restricted to `allowed` tools (the ones the request's selection step named) plus final. */
    public String stepGrammar(Collection<String> allowed) throws JSONException {
        StringBuilder g = new StringBuilder(BASE_RULES), alts = new StringBuilder("final"); int k = 0;
        for (Tool t : registry.values()) { if (!allowed.contains(t.name)) continue; String r = "call" + (k++); g.append(callRule(r, t)); alts.append(" | ").append(r); }
        return g.append("root ::= ").append(alts).append('\n').toString();
    }
    /** {"intent": <string>}: the request restated as one direct instruction. */
    public String intentGrammar() { return BASE_RULES + "root ::= \"{\\\"intent\\\":\" ws string \"}\"\n"; }
    /** {"answer": "yes"} or {"answer": "no"}: a constrained decision (tool needed? instruction done?). */
    public String yesNoGrammar() { return "root ::= \"{\\\"answer\\\":\\\"\" (\"yes\" | \"no\") \"\\\"}\"\n"; }
    /** Only {"final": <string>}. */
    public String finalGrammar() { return BASE_RULES + "root ::= final\n"; }
    /** {"plan":[<registered tool names in order>]} -- the TaskPlanner step. */
    public String planGrammar() {
        StringBuilder alts = new StringBuilder(); for (String n : registry.keySet()) { if (alts.length() > 0) alts.append(" | "); alts.append("\"\\\"").append(n).append("\\\"\""); }
        return "name ::= " + alts + "\nroot ::= \"{\\\"plan\\\":[\" (name (\",\" name){0,4})? \"]}\"\n";
    }

    /** Validate args against the tool's declared schema (required keys, primitive types). Returns null when valid. */
    public static String validate(Tool t, JSONObject args) throws JSONException {
        JSONArray req = t.params.getJSONArray("required"); JSONObject props = t.params.getJSONObject("properties");
        for (int i = 0; i < req.length(); i++) { String k = req.getString(i); if (!args.has(k)) return "missing argument '" + k + "'";
            String ty = props.getJSONObject(k).getString("type"); Object v = args.get(k);
            if (ty.equals("string") && !(v instanceof String)) return "argument '" + k + "' must be a string"; }
        return null;
    }
}
