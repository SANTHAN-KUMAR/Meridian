package com.meridian.app;

import android.app.NotificationManager;
import android.content.*;
import android.database.Cursor;
import android.net.Uri;
import android.os.*;
import android.provider.*;
import org.json.*;
import java.io.*;
import java.util.*;

/** Everyday-task tools: weather, news, reminders, calendar read/write, nearby places, files, Do Not Disturb, ringer, exchange
 *  rates. Network sources need no API key: Open-Meteo (weather and geocoding), Google News RSS, OpenStreetMap (Nominatim
 *  geocoding, Overpass places), open.er-api.com (exchange rates). OpenStreetMap has no ratings; ratings are returned only when a
 *  Google Places key is configured (keys.json, see placesKey), and the result says which source every figure came from. */
final class ToolsDaily {
    private ToolsDaily() { }

    static String enc(String s) { try { return java.net.URLEncoder.encode(s, "UTF-8"); } catch (UnsupportedEncodingException e) { throw new RuntimeException(e); } }
    static String fmt(long ms) { return String.format(Locale.ROOT, "%1$ta %1$tF %1$tR", ms); }
    static boolean here(String s) { return s == null || s.trim().isEmpty() || s.trim().toLowerCase(Locale.ROOT).matches("(me|here|near me|nearby|current location|my location|around me|my area)"); }

    /** {lat, lon, name, source}: the device's location for "here", else the place geocoded (Open-Meteo, then Nominatim). */
    static JSONObject locate(Tools t, String place, boolean preferPrecise) throws Exception {
        if (here(place)) {
            if (preferPrecise) Perms.needAny(t.context(), "finding places near you", android.Manifest.permission.ACCESS_FINE_LOCATION, android.Manifest.permission.ACCESS_COARSE_LOCATION);
            else Perms.need(t.context(), "finding where you are", android.Manifest.permission.ACCESS_COARSE_LOCATION);
            JSONObject l = t.location(); String nm = l.optString("area", ""); if (nm.isEmpty() || nm.equals("null")) nm = l.optString("city", "your location");
            return new JSONObject().put("lat", l.getDouble("lat")).put("lon", l.getDouble("lon")).put("name", nm).put("source", "device location, accuracy " + l.optLong("accuracy_m") + " m, " + l.optLong("age_min") + " min old");
        }
        JSONObject g = new JSONObject(Tools.httpGet("https://geocoding-api.open-meteo.com/v1/search?count=1&language=en&format=json&name=" + enc(place.trim())));
        JSONArray r = g.optJSONArray("results");
        if (r != null && r.length() > 0) { JSONObject x = r.getJSONObject(0);
            return new JSONObject().put("lat", x.getDouble("latitude")).put("lon", x.getDouble("longitude")).put("name", x.optString("name") + (x.has("admin1") ? ", " + x.optString("admin1") : "") + (x.has("country") ? ", " + x.optString("country") : "")).put("source", "Open-Meteo geocoding"); }
        JSONArray n = new JSONArray(Tools.httpGet("https://nominatim.openstreetmap.org/search?format=json&limit=1&q=" + enc(place.trim()), Tools.OSM_UA));
        if (n.length() == 0) throw new IllegalArgumentException("could not find a place called '" + place + "'");
        JSONObject x = n.getJSONObject(0);
        return new JSONObject().put("lat", Double.parseDouble(x.getString("lat"))).put("lon", Double.parseDouble(x.getString("lon"))).put("name", Agent.clip(x.optString("display_name"), 80)).put("source", "OpenStreetMap Nominatim");
    }

    static String placesKey(Context c) {   // {"google_places_key": "..."} in files/keys.json or Android/data/com.meridian.app/files/keys.json
        for (File d : new File[]{c.getFilesDir(), c.getExternalFilesDir(null)}) { if (d == null) continue; File f = new File(d, "keys.json");
            if (f.exists()) try { String k = new JSONObject(Native.readFile(f.getAbsolutePath())).optString("google_places_key"); if (!k.isEmpty()) return k; } catch (Exception ignored) { } }
        return null;
    }

    static final String[] OVERPASS = {"https://overpass-api.de/api/interpreter", "https://maps.mail.ru/osm/tools/overpass/api/interpreter", "https://overpass.kumi.systems/api/interpreter"};
    /** OpenStreetMap places around (lat, lon) within radius metres, nearest first. */
    static JSONArray osmPlaces(String what, double lat, double lon, int radius) throws Exception {
        String[] tags = Parse.osmTags(what); StringBuilder q = new StringBuilder("[out:json][timeout:20];(");
        String around = "(around:" + radius + "," + lat + "," + lon + ");";
        if (tags != null) for (String tg : tags) { String[] kv = tg.split("="); q.append("nwr[\"").append(kv[0]).append("\"=\"").append(kv[1]).append("\"]").append(around); }
        else { String name = what.replaceAll("[^A-Za-z0-9 ]", "").trim(); if (name.isEmpty()) throw new IllegalArgumentException("say what kind of place to look for");
            q.append("nwr[\"name\"~\"").append(name).append("\",i]").append(around); }
        q.append(");out center 80;");
        JSONArray el = null; Exception last = null;   // public Overpass instances rate-limit (429) and drop connections; try the mirrors in turn
        for (String host : OVERPASS) { try { el = new JSONObject(Tools.httpGet(host + "?data=" + enc(q.toString()), Tools.OSM_UA)).getJSONArray("elements"); break; } catch (Exception e) { last = e; } }
        if (el == null) throw new IllegalStateException("the OpenStreetMap place service is unreachable right now (" + last.getMessage() + ")");
        List<JSONObject> l = new ArrayList<>();
        for (int i = 0; i < el.length(); i++) { JSONObject e = el.getJSONObject(i), tg = e.optJSONObject("tags"); if (tg == null || tg.optString("name").isEmpty()) continue;
            double la = e.has("lat") ? e.getDouble("lat") : e.getJSONObject("center").getDouble("lat"), lo = e.has("lon") ? e.getDouble("lon") : e.getJSONObject("center").getDouble("lon");
            JSONObject p = new JSONObject().put("name", tg.getString("name")).put("distance_m", Math.round(Parse.metres(lat, lon, la, lo))).put("lat", la).put("lon", lo);
            String kind = tg.optString("amenity", tg.optString("shop", tg.optString("tourism", tg.optString("leisure", "")))); if (!kind.isEmpty()) p.put("kind", kind);
            for (String k : new String[]{"cuisine", "opening_hours", "phone", "website"}) if (tg.has(k)) p.put(k, Agent.clip(tg.getString(k), 60));
            String addr = (tg.optString("addr:housenumber") + " " + tg.optString("addr:street")).trim(); if (!addr.isEmpty()) p.put("address", addr);
            l.add(p); }
        Collections.sort(l, (a, b) -> Long.compare(a.optLong("distance_m"), b.optLong("distance_m")));
        JSONArray out = new JSONArray(); Set<String> seen = new HashSet<>(); for (JSONObject p : l) if (seen.add(p.getString("name").toLowerCase(Locale.ROOT)) && out.length() < 8) out.put(p);
        return out;
    }

    // ---------------- files ----------------
    static File sdcard() { return Environment.getExternalStorageDirectory(); }
    static void needFiles(Context ctx) throws Exception {
        if (Build.VERSION.SDK_INT >= 30) { if (Environment.isExternalStorageManager()) return;
            try { ctx.startActivity(new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, Uri.parse("package:" + ctx.getPackageName())).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)); } catch (Exception ignored) { }
            throw new IllegalStateException("PermissionRequired: file access is off. Opened the settings page: allow Meridian 'access to all files', then ask again"); }
        Perms.need(ctx, "searching your files", android.Manifest.permission.READ_EXTERNAL_STORAGE);
    }
    /** Files under shared storage whose name contains every word of `query` (Android/ and hidden folders skipped), newest first. */
    static List<File> findFiles(String query, int max) {
        String[] words = query.toLowerCase(Locale.ROOT).trim().split("\\s+"); List<File> hits = new ArrayList<>(); int[] seen = {0};
        Deque<File> dirs = new ArrayDeque<>(); dirs.add(sdcard()); Map<File, Integer> depth = new HashMap<>(); depth.put(sdcard(), 0);
        while (!dirs.isEmpty() && seen[0] < 30000) { File d = dirs.poll(); File[] ls = d.listFiles(); if (ls == null) continue;
            for (File f : ls) { seen[0]++; String n = f.getName(); if (n.startsWith(".")) continue;
                if (f.isDirectory()) { if (depth.get(d) < 7 && !(d.equals(sdcard()) && n.equals("Android"))) { dirs.add(f); depth.put(f, depth.get(d) + 1); } continue; }
                String ln = n.toLowerCase(Locale.ROOT); boolean all = true; for (String w : words) all &= ln.contains(w); if (all) hits.add(f); } }
        Collections.sort(hits, (a, b) -> Long.compare(b.lastModified(), a.lastModified()));
        return hits.size() > max ? hits.subList(0, max) : hits;
    }
    static File resolveFile(String name) {
        File f = name.startsWith("/") ? new File(name) : new File(sdcard(), name); if (name.contains("/") && f.isFile()) return f;
        List<File> h = findFiles(name, 1); if (h.isEmpty()) throw new IllegalArgumentException("no file named like '" + name + "' in shared storage"); return h.get(0);
    }
    static String rel(File f) { String s = sdcard().getAbsolutePath(); return f.getAbsolutePath().startsWith(s) ? f.getAbsolutePath().substring(s.length() + 1) : f.getAbsolutePath(); }
    static final String TEXT_EXT = ".*\\.(txt|md|csv|json|log|xml|html?|ini|yaml|yml|tsv|srt|vcf|ics)$";
    static String readText(File f, int max) throws IOException { byte[] b = RemoteGguf.readUpTo(new FileInputStream(f), max * 4); String s = new String(b, "UTF-8"); return s.length() > max ? s.substring(0, max) + " ..." : s; }
    static String mime(String name) { String ext = android.webkit.MimeTypeMap.getFileExtensionFromUrl(Uri.encode(name.toLowerCase(Locale.ROOT)).replace("%2E", "."));
        String m = ext == null ? null : android.webkit.MimeTypeMap.getSingleton().getMimeTypeFromExtension(ext); return m == null ? "application/octet-stream" : m; }

    // ---------------- calendar ----------------
    static JSONArray events(Context ctx, long st, long en) throws Exception {
        Perms.need(ctx, "reading your calendar", android.Manifest.permission.READ_CALENDAR);
        Uri.Builder u = CalendarContract.Instances.CONTENT_URI.buildUpon(); ContentUris.appendId(u, st); ContentUris.appendId(u, en); JSONArray out = new JSONArray();
        try (Cursor c = ctx.getContentResolver().query(u.build(), new String[]{CalendarContract.Instances.TITLE, CalendarContract.Instances.BEGIN, CalendarContract.Instances.END, CalendarContract.Instances.ALL_DAY, CalendarContract.Instances.EVENT_LOCATION, CalendarContract.Instances.EVENT_ID},
                null, null, CalendarContract.Instances.BEGIN + " ASC")) {
            while (c != null && c.moveToNext() && out.length() < 25) { JSONObject e = new JSONObject().put("title", c.getString(0) == null ? "(no title)" : c.getString(0));
                if (c.getInt(3) == 1) e.put("all_day", String.format(Locale.ROOT, "%1$ta %1$tF", c.getLong(1))); else e.put("starts", fmt(c.getLong(1))).put("ends", String.format(Locale.ROOT, "%tR", c.getLong(2)));
                if (c.getString(4) != null && !c.getString(4).isEmpty()) e.put("location", c.getString(4)); out.put(e.put("event_id", c.getLong(5))); } }
        return out;
    }

    static void register(final Tools t) throws JSONException {
        final Context ctx = t.context();
        t.add(new Tools.Tool("weather", "Get the current weather and 3-day forecast for a place (empty for where the phone is).", "read", true, "none",
                "temperatures are physically possible and the forecast is for the requested coordinates", Tools.obj("place", "string")) {
            JSONObject execute(JSONObject a) throws Exception { JSONObject p = locate(t, a.getString("place"), false);
                JSONObject w = new JSONObject(Tools.httpGet("https://api.open-meteo.com/v1/forecast?latitude=" + p.getDouble("lat") + "&longitude=" + p.getDouble("lon")
                    + "&current=temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max&timezone=auto&forecast_days=3"));
                JSONObject cur = w.getJSONObject("current"), d = w.getJSONObject("daily"); JSONArray days = new JSONArray();
                for (int i = 0; i < d.getJSONArray("time").length(); i++) days.put(new JSONObject().put("date", d.getJSONArray("time").getString(i)).put("sky", Parse.wmo(d.getJSONArray("weather_code").getInt(i)))
                    .put("max_c", d.getJSONArray("temperature_2m_max").getDouble(i)).put("min_c", d.getJSONArray("temperature_2m_min").getDouble(i)).put("rain_chance_pct", d.getJSONArray("precipitation_probability_max").optInt(i, -1)));
                return new JSONObject().put("place", p.getString("name")).put("now", new JSONObject().put("sky", Parse.wmo(cur.getInt("weather_code"))).put("temp_c", cur.getDouble("temperature_2m"))
                        .put("feels_like_c", cur.getDouble("apparent_temperature")).put("humidity_pct", cur.getInt("relative_humidity_2m")).put("wind_kmh", cur.getDouble("wind_speed_10m")).put("rain_mm", cur.getDouble("precipitation")).put("as_of", cur.getString("time")))
                    .put("days", days).put("lat", w.getDouble("latitude")).put("lon", w.getDouble("longitude")).put("req_lat", p.getDouble("lat")).put("req_lon", p.getDouble("lon")).put("source", "Open-Meteo; place from " + p.getString("source")); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { JSONObject n = r.getJSONObject("now"); double tc = n.getDouble("temp_c");   // world records: -89.2 C (Vostok), 56.7 C (Death Valley)
                return tc > -90 && tc < 57 && n.getInt("humidity_pct") >= 0 && n.getInt("humidity_pct") <= 100 && Parse.metres(r.getDouble("lat"), r.getDouble("lon"), r.getDouble("req_lat"), r.getDouble("req_lon")) < 25000; } });
        t.add(new Tools.Tool("news", "Get the latest news headlines, on a topic or top stories (empty topic).", "read", true, "none",
                "the headlines were parsed from the news feed and each has a title and link", Tools.obj("topic", "string")) {
            @Override int resultChars() { return 1500; }
            JSONObject execute(JSONObject a) throws Exception { String cc = Locale.getDefault().getCountry(); if (cc.isEmpty()) cc = "US"; String loc = "hl=en-" + cc + "&gl=" + cc + "&ceid=" + cc + ":en";
                String topic = a.getString("topic").trim(), url = topic.isEmpty() || topic.matches("(?i)(top|latest|today'?s?|headlines?|news|top stories)") ? "https://news.google.com/rss?" + loc : "https://news.google.com/rss/search?q=" + enc(topic) + "&" + loc;
                JSONArray out = new JSONArray(); for (String[] it : Parse.rssItems(Tools.httpGet(url), 8)) out.put(new JSONObject().put("title", it[0]).put("source", it[1]).put("published", it[2]).put("link", it[3]));
                return new JSONObject().put("topic", topic.isEmpty() ? "top stories" : topic).put("headlines", out).put("source", "Google News RSS"); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { JSONArray x = r.getJSONArray("headlines"); if (x.length() == 0) return false;
                for (int i = 0; i < x.length(); i++) if (x.getJSONObject(i).optString("title").isEmpty() || !x.getJSONObject(i).optString("link").startsWith("http")) return false; return true; } });
        t.add(new Tools.Tool("reminder_add", "Set a reminder that notifies at a time: when like 'in 20 minutes', 'tomorrow 9am', 'friday 6pm'.", "write", true, "none",
                "the reminder is stored and its alarm is registered with the system", Tools.obj("text", "string", "when", "string")) {
            JSONObject execute(JSONObject a) throws Exception { if (Build.VERSION.SDK_INT >= 33) Perms.need(ctx, "showing reminders", "android.permission.POST_NOTIFICATIONS");
                Parse.When w = Parse.when(a.getString("when"), Calendar.getInstance()); int id = (int) (System.currentTimeMillis() / 1000 % 1000000000L);
                boolean exact = ReminderReceiver.schedule(ctx, id, w.at);
                JSONArray all = ReminderReceiver.load(ctx); all.put(new JSONObject().put("id", id).put("text", a.getString("text")).put("at", w.at).put("exact", exact)); ReminderReceiver.save(ctx, all);
                JSONObject r = new JSONObject().put("reminder", a.getString("text")).put("at", fmt(w.at)).put("id", id).put("exact", exact); if (!w.assumed.isEmpty()) r.put("assumed", w.assumed); return r; }
            @Override String note(JSONObject r) { String s = r.has("assumed") ? "Reminder time assumption: " + r.optString("assumed") + "." : null;
                if (!r.optBoolean("exact")) s = (s == null ? "" : s + " ") + "Exact alarms are not allowed for Meridian, so the reminder may arrive up to 10 minutes late (Settings > Apps > Meridian > Alarms & reminders)."; return s; }
            boolean verify(JSONObject a, JSONObject r) { int id = r.optInt("id"); JSONArray all = ReminderReceiver.load(ctx); boolean stored = false;
                for (int i = 0; i < all.length(); i++) if (all.optJSONObject(i).optInt("id") == id) stored = true; return stored && ReminderReceiver.scheduled(ctx, id); } });
        t.add(new Tools.Tool("reminder_list", "List the reminders that are still to come.", "read", true, "none", "the returned reminders re-read identically from the reminder store", Tools.obj()) {
            JSONArray future() { JSONArray all = ReminderReceiver.load(ctx), out = new JSONArray(); long now = System.currentTimeMillis();
                for (int i = 0; i < all.length(); i++) { JSONObject r = all.optJSONObject(i); if (r != null && r.optLong("at") > now) try { out.put(new JSONObject().put("text", r.optString("text")).put("at", fmt(r.optLong("at")))); } catch (JSONException ignored) { } }
                return out; }
            JSONObject execute(JSONObject a) throws Exception { return new JSONObject().put("reminders", future()); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { return future().toString().equals(r.getJSONArray("reminders").toString()); } });
        t.add(new Tools.Tool("reminder_cancel", "Cancel the reminders whose text contains the given words ('all' cancels every reminder).", "write", false, "every_time",
                "no cancelled reminder remains stored or scheduled", Tools.obj("text", "string")) {
            @Override String consentText(JSONObject a) { return "Cancel reminders matching \"" + a.optString("text") + "\"."; }
            JSONObject execute(JSONObject a) throws Exception { String q = a.getString("text").toLowerCase(Locale.ROOT).trim(); boolean all = q.equals("all") || q.isEmpty();
                JSONArray st = ReminderReceiver.load(ctx), keep = new JSONArray(), gone = new JSONArray();
                for (int i = 0; i < st.length(); i++) { JSONObject r = st.getJSONObject(i); if (all || r.optString("text").toLowerCase(Locale.ROOT).contains(q)) { ReminderReceiver.cancel(ctx, r.optInt("id")); gone.put(r); } else keep.put(r); }
                if (gone.length() == 0) throw new IllegalArgumentException("no reminder matches '" + a.getString("text") + "'");
                ReminderReceiver.save(ctx, keep); return new JSONObject().put("cancelled", gone); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { JSONArray g = r.getJSONArray("cancelled"), st = ReminderReceiver.load(ctx);
                for (int i = 0; i < g.length(); i++) { int id = g.getJSONObject(i).optInt("id"); if (ReminderReceiver.scheduled(ctx, id)) return false; for (int k = 0; k < st.length(); k++) if (st.getJSONObject(k).optInt("id") == id) return false; } return true; } });
        t.add(new Tools.Tool("calendar_read", "Read calendar events for today, tomorrow, this week, next week, a weekday or a date.", "read", true, "none",
                "the returned events re-query identically from the calendar provider", Tools.obj("when", "string")) {
            @Override int resultChars() { return 1500; }
            JSONObject execute(JSONObject a) throws Exception { long[] r = Parse.range(a.getString("when"), Calendar.getInstance());
                return new JSONObject().put("from", fmt(r[0])).put("to", fmt(r[1])).put("events", events(ctx, r[0], r[1])); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { long[] x = Parse.range(a.getString("when"), Calendar.getInstance()); return events(ctx, x[0], x[1]).toString().equals(r.getJSONArray("events").toString()); } });
        t.add(new Tools.Tool("calendar_add", "Add an event to the calendar directly: title, when (like 'friday 3pm'), duration (like '1 hour', empty for 1 hour), place (may be empty).", "write", true, "none",
                "the event is found in the calendar provider with this title and start time", Tools.obj("title", "string", "when", "string", "duration", "string", "place", "string")) {
            JSONObject execute(JSONObject a) throws Exception {
                Perms.need(ctx, "adding to your calendar", android.Manifest.permission.READ_CALENDAR, android.Manifest.permission.WRITE_CALENDAR);
                Parse.When w = Parse.when(a.getString("when"), Calendar.getInstance()); long dur = a.getString("duration").trim().isEmpty() ? 3600 : Tools.parseDuration(a.getString("duration"));
                long cal = -1; String calName = "";
                try (Cursor c = ctx.getContentResolver().query(CalendarContract.Calendars.CONTENT_URI, new String[]{CalendarContract.Calendars._ID, CalendarContract.Calendars.CALENDAR_DISPLAY_NAME},
                        CalendarContract.Calendars.VISIBLE + "=1 AND " + CalendarContract.Calendars.CALENDAR_ACCESS_LEVEL + ">=" + CalendarContract.Calendars.CAL_ACCESS_CONTRIBUTOR, null, CalendarContract.Calendars.IS_PRIMARY + " DESC")) {
                    if (c != null && c.moveToFirst()) { cal = c.getLong(0); calName = c.getString(1); } }
                if (cal < 0) throw new IllegalStateException("this phone has no writable calendar (add an account with a calendar, or use calendar_event to open the calendar app)");
                ContentValues v = new ContentValues(); v.put(CalendarContract.Events.CALENDAR_ID, cal); v.put(CalendarContract.Events.TITLE, a.getString("title")); v.put(CalendarContract.Events.DTSTART, w.at);
                v.put(CalendarContract.Events.DTEND, w.at + dur * 1000); v.put(CalendarContract.Events.EVENT_TIMEZONE, TimeZone.getDefault().getID()); if (!a.getString("place").trim().isEmpty()) v.put(CalendarContract.Events.EVENT_LOCATION, a.getString("place").trim());
                Uri u = ctx.getContentResolver().insert(CalendarContract.Events.CONTENT_URI, v); if (u == null) throw new IllegalStateException("the calendar provider refused the event");
                JSONObject r = new JSONObject().put("added", a.getString("title")).put("starts", fmt(w.at)).put("minutes", dur / 60).put("calendar", calName).put("event_id", ContentUris.parseId(u)).put("start_ms", w.at);
                if (!w.assumed.isEmpty()) r.put("assumed", w.assumed); return r; }
            @Override String note(JSONObject r) { return r.has("assumed") ? "Event time assumption: " + r.optString("assumed") + "." : null; }
            boolean verify(JSONObject a, JSONObject r) throws Exception {
                try (Cursor c = ctx.getContentResolver().query(ContentUris.withAppendedId(CalendarContract.Events.CONTENT_URI, r.getLong("event_id")), new String[]{CalendarContract.Events.TITLE, CalendarContract.Events.DTSTART}, null, null, null)) {
                    return c != null && c.moveToFirst() && a.getString("title").equals(c.getString(0)) && c.getLong(1) == r.getLong("start_ms"); } } });
        t.add(new Tools.Tool("places_nearby", "Find the nearest places of a kind (restaurant, cafe, pharmacy, ATM, hospital, petrol...) near where the phone is (near empty) or near a named place. Gives distance, cuisine, hours; ratings only if a ratings source is configured.", "read", true, "none",
                "every place's distance re-computes from its coordinates, lies within the search radius, and the list is nearest-first", Tools.obj("what", "string", "near", "string")) {
            @Override int resultChars() { return 1800; }
            JSONObject execute(JSONObject a) throws Exception { JSONObject c = locate(t, a.getString("near"), true); double la = c.getDouble("lat"), lo = c.getDouble("lon");
                JSONArray p = new JSONArray(); int radius = 1500;
                for (int rr : new int[]{1500, 4000, 10000}) { radius = rr; p = osmPlaces(a.getString("what"), la, lo, rr); if (p.length() >= 3) break; }
                JSONObject r = new JSONObject().put("near", c.getString("name")).put("center_lat", la).put("center_lon", lo).put("radius_m", radius).put("places", p).put("source", "OpenStreetMap (Overpass); location from " + c.getString("source"));
                String key = placesKey(ctx);
                if (key == null) r.put("ratings", "not available: OpenStreetMap has no ratings and no Google Places key is configured");
                else { JSONObject g = new JSONObject(Tools.httpGet("https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=" + la + "," + lo + "&radius=" + radius + "&keyword=" + enc(a.getString("what")) + "&key=" + key));
                    JSONArray res = g.optJSONArray("results"), rated = new JSONArray();
                    if (res != null) for (int i = 0; i < res.length() && rated.length() < 8; i++) { JSONObject x = res.getJSONObject(i); if (!x.has("rating")) continue; JSONObject loc = x.getJSONObject("geometry").getJSONObject("location");
                        rated.put(new JSONObject().put("name", x.optString("name")).put("rating", x.getDouble("rating")).put("reviews", x.optInt("user_ratings_total")).put("distance_m", Math.round(Parse.metres(la, lo, loc.getDouble("lat"), loc.getDouble("lng")))).put("address", x.optString("vicinity"))); }
                    r.put("rated", rated).put("ratings", "Google Places (" + g.optString("status") + ")"); }
                return r; }
            boolean verify(JSONObject a, JSONObject r) throws Exception { JSONArray p = r.getJSONArray("places"); double la = r.getDouble("center_lat"), lo = r.getDouble("center_lon"); long prev = -1;
                for (int i = 0; i < p.length(); i++) { JSONObject x = p.getJSONObject(i); long d = Math.round(Parse.metres(la, lo, x.getDouble("lat"), x.getDouble("lon")));
                    if (d != x.getLong("distance_m") || d > r.getInt("radius_m") * 1.05 || d < prev) return false; prev = d; }
                JSONArray g = r.optJSONArray("rated"); if (g != null) for (int i = 0; i < g.length(); i++) { double s = g.getJSONObject(i).getDouble("rating"); if (s < 1 || s > 5) return false; }
                return p.length() > 0 || (g != null && g.length() > 0); } });
        t.add(new Tools.Tool("files_find", "Find files on the phone's storage by words in their name (like 'invoice pdf' or 'resume').", "read", true, "none",
                "every returned file exists with the reported size", Tools.obj("query", "string")) {
            JSONObject execute(JSONObject a) throws Exception { needFiles(ctx); JSONArray out = new JSONArray();
                for (File f : findFiles(a.getString("query"), 10)) out.put(new JSONObject().put("name", f.getName()).put("path", rel(f)).put("size_kb", (f.length() + 1023) / 1024).put("modified", fmt(f.lastModified())));
                return new JSONObject().put("files", out); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { JSONArray x = r.getJSONArray("files");
                for (int i = 0; i < x.length(); i++) { File f = new File(sdcard(), x.getJSONObject(i).getString("path")); if (!f.isFile() || (f.length() + 1023) / 1024 != x.getJSONObject(i).getLong("size_kb")) return false; } return true; } });
        t.add(new Tools.Tool("file_read", "Read the text of a text file (txt, md, csv, json...) by name or path from files_find.", "read", true, "none",
                "the returned text re-reads identically from the file", Tools.obj("name", "string")) {
            @Override int resultChars() { return 1700; }
            JSONObject execute(JSONObject a) throws Exception { needFiles(ctx); File f = resolveFile(a.getString("name"));
                if (!f.getName().toLowerCase(Locale.ROOT).matches(TEXT_EXT)) throw new IllegalArgumentException(f.getName() + " is not a text file; only text formats can be read (open it with file_open instead)");
                return new JSONObject().put("path", rel(f)).put("text", readText(f, 1500)); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { return readText(new File(sdcard(), r.getString("path")), 1500).equals(r.getString("text")); } });
        t.add(new Tools.Tool("file_save", "Save text as a file in Download/Meridian (name like 'shopping list.txt').", "write", true, "none",
                "the saved file reads back exactly the text", Tools.obj("name", "string", "text", "string")) {
            JSONObject execute(JSONObject a) throws Exception { String n = a.getString("name").replaceAll("[\\\\/:*?\"<>|]", "_").trim(); if (n.isEmpty()) n = "note"; if (!n.contains(".")) n += ".txt";
                ContentValues v = new ContentValues(); v.put(MediaStore.MediaColumns.DISPLAY_NAME, n); v.put(MediaStore.MediaColumns.MIME_TYPE, mime(n)); v.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/Meridian");
                Uri u = ctx.getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, v); if (u == null) throw new IllegalStateException("storage refused the new file");
                try (OutputStream o = ctx.getContentResolver().openOutputStream(u)) { o.write(a.getString("text").getBytes("UTF-8")); }
                String shown = n; try (Cursor c = ctx.getContentResolver().query(u, new String[]{MediaStore.MediaColumns.DISPLAY_NAME}, null, null, null)) { if (c != null && c.moveToFirst()) shown = c.getString(0); }
                return new JSONObject().put("saved", "Download/Meridian/" + shown).put("uri", u.toString()).put("bytes", a.getString("text").getBytes("UTF-8").length); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { try (InputStream i = ctx.getContentResolver().openInputStream(Uri.parse(r.getString("uri")))) {
                return new String(RemoteGguf.readUpTo(i, r.getInt("bytes") + 16), "UTF-8").equals(a.getString("text")); } } });
        t.add(new Tools.Tool("file_open", "Open a file (by name or path from files_find) in the app that views it.", "ui", true, "none",
                "an app on the phone accepted the file", Tools.obj("name", "string")) {
            JSONObject execute(JSONObject a) throws Exception { needFiles(ctx); File f = resolveFile(a.getString("name")); Uri u = null;
                for (int k = 0; k < 2 && u == null; k++) { try (Cursor c = ctx.getContentResolver().query(MediaStore.Files.getContentUri("external"), new String[]{MediaStore.MediaColumns._ID}, MediaStore.MediaColumns.DATA + "=?", new String[]{f.getAbsolutePath()}, null)) {
                        if (c != null && c.moveToFirst()) u = ContentUris.withAppendedId(MediaStore.Files.getContentUri("external"), c.getLong(0)); }
                    if (u == null) { final java.util.concurrent.CountDownLatch l = new java.util.concurrent.CountDownLatch(1); android.media.MediaScannerConnection.scanFile(ctx, new String[]{f.getAbsolutePath()}, null, (p, x) -> l.countDown()); l.await(5, java.util.concurrent.TimeUnit.SECONDS); } }
                if (u == null) throw new IllegalStateException("the media index does not know " + rel(f) + "; it cannot be opened from here");
                String pkg = t.launch(new Intent(Intent.ACTION_VIEW).setDataAndType(u, mime(f.getName())).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION));
                return new JSONObject().put("opened", rel(f)).put("handled_by", pkg); }
            boolean verify(JSONObject a, JSONObject r) { return r.has("handled_by"); } });
        t.add(new Tools.Tool("do_not_disturb", "Turn Do Not Disturb on or off: state is on, off, alarms only, or total silence.", "write", true, "none",
                "the notification manager reports the requested interruption filter", Tools.obj("state", "string")) {
            JSONObject execute(JSONObject a) throws Exception { NotificationManager nm = (NotificationManager) ctx.getSystemService(Context.NOTIFICATION_SERVICE); policy(ctx, nm);
                String s = a.getString("state").toLowerCase(Locale.ROOT); int f = s.matches(".*\\b(off|disable|false|no)\\b.*") ? NotificationManager.INTERRUPTION_FILTER_ALL : s.contains("alarm") ? NotificationManager.INTERRUPTION_FILTER_ALARMS
                    : s.matches(".*\\b(total|silence|none|nothing)\\b.*") ? NotificationManager.INTERRUPTION_FILTER_NONE : NotificationManager.INTERRUPTION_FILTER_PRIORITY;
                nm.setInterruptionFilter(f); SystemClock.sleep(300); return new JSONObject().put("filter", f).put("dnd_on", f != NotificationManager.INTERRUPTION_FILTER_ALL); }
            boolean verify(JSONObject a, JSONObject r) { return ((NotificationManager) ctx.getSystemService(Context.NOTIFICATION_SERVICE)).getCurrentInterruptionFilter() == r.optInt("filter"); } });
        t.add(new Tools.Tool("ringer_mode", "Set the ringer to normal, vibrate or silent.", "write", true, "none",
                "the audio service reports the requested ringer mode", Tools.obj("mode", "string")) {
            JSONObject execute(JSONObject a) throws Exception { android.media.AudioManager am = (android.media.AudioManager) ctx.getSystemService(Context.AUDIO_SERVICE); String m = a.getString("mode").toLowerCase(Locale.ROOT);
                int mode = m.contains("vib") ? android.media.AudioManager.RINGER_MODE_VIBRATE : m.matches(".*\\b(silent|mute|quiet)\\b.*") ? android.media.AudioManager.RINGER_MODE_SILENT : android.media.AudioManager.RINGER_MODE_NORMAL;
                try { am.setRingerMode(mode); } catch (SecurityException e) { policy(ctx, (NotificationManager) ctx.getSystemService(Context.NOTIFICATION_SERVICE)); am.setRingerMode(mode); }
                SystemClock.sleep(300); return new JSONObject().put("mode", mode == 0 ? "silent" : mode == 1 ? "vibrate" : "normal").put("code", mode); }
            boolean verify(JSONObject a, JSONObject r) { return ((android.media.AudioManager) ctx.getSystemService(Context.AUDIO_SERVICE)).getRingerMode() == r.optInt("code"); } });
        t.add(new Tools.Tool("exchange_rate", "Convert an amount between currencies at today's rate (like 100, USD, INR).", "read", true, "none",
                "the converted amount re-computes from the returned rate", Tools.obj("amount", "string", "from", "string", "to", "string")) {
            JSONObject execute(JSONObject a) throws Exception { String f = Parse.currency(a.getString("from")), to = Parse.currency(a.getString("to")); double amt = Tools.calc(a.getString("amount").trim().isEmpty() ? "1" : a.getString("amount"));
                JSONObject j = new JSONObject(Tools.httpGet("https://open.er-api.com/v6/latest/" + f)); if (!"success".equals(j.optString("result"))) throw new IllegalStateException("rate service: " + j.optString("error-type", "failed"));
                if (!j.getJSONObject("rates").has(to)) throw new IllegalArgumentException("no rate for " + to); double rate = j.getJSONObject("rates").getDouble(to);
                return new JSONObject().put("amount", amt).put("from", f).put("to", to).put("rate", rate).put("result", Math.round(amt * rate * 100) / 100.0).put("rates_as_of", j.optString("time_last_update_utc")).put("source", "open.er-api.com"); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { double rate = r.getDouble("rate"); return rate > 0 && Math.abs(r.getDouble("amount") * rate - r.getDouble("result")) <= 0.006; } });
    }
    static void policy(Context ctx, NotificationManager nm) { if (nm.isNotificationPolicyAccessGranted()) return;
        try { ctx.startActivity(new Intent(Settings.ACTION_NOTIFICATION_POLICY_ACCESS_SETTINGS).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)); } catch (Exception ignored) { }
        throw new IllegalStateException("PermissionRequired: Do Not Disturb access is off. Opened the settings page: allow it for Meridian, then ask again"); }
}
