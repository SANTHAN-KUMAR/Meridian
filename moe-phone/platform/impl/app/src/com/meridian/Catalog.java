package com.meridian.app;

import android.content.Context;
import org.json.*;
import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.*;

/** Model discovery. A catalogue is a convenience over the architecture checks, never a gate (10_EXTENSION_POINTS.md section 3):
 *   - curated: assets/catalog.json, 20 ungated Hugging Face GGUFs whose ModelCards were derived from their real headers at
 *     build time (app/test/CatalogMain.java), so ranking and predictions work offline;
 *   - any URL: RemoteGguf reads the header of any https .gguf and the same Predictor evaluates it before download;
 *   - search: the public Hugging Face API lists GGUF repos and their files, each evaluable the same way. */
public final class Catalog {
    public static final class Entry { public JSONObject meta; public Planner.Card card; public String url, sha256, name; public long size; }

    public static List<Entry> curated(Context ctx) throws IOException, JSONException {
        String txt; try (InputStream in = ctx.getAssets().open("catalog.json")) { txt = new String(RemoteGguf.readUpTo(in, 8 << 20), "UTF-8"); }
        JSONArray a = new JSONObject(txt).getJSONArray("entries"); List<Entry> out = new ArrayList<>();
        for (int i = 0; i < a.length(); i++) { JSONObject e = a.getJSONObject(i); Entry x = new Entry(); x.meta = e; x.card = Planner.Card.fromJson(e.getJSONObject("card"));
            x.url = e.getString("url"); x.sha256 = e.optString("sha256", null); x.size = e.getLong("size_bytes"); x.name = e.optString("display_name", e.getString("filename")); out.add(x); }
        return out;
    }

    /** Evaluate every curated entry on this device and rank them (largest fluent model first). */
    public static JSONArray recommend(Context ctx, JSONObject profile, File modelDir) throws IOException, JSONException {
        List<JSONObject> evs = new ArrayList<>(); long free = modelDir.getUsableSpace(); String fs = Profile.filesystemOf(modelDir.getAbsolutePath());
        for (Entry e : curated(ctx)) {
            boolean have = new File(modelDir, e.meta.getString("filename")).exists();
            JSONObject ev = Predictor.evaluate(profile, e.card, free, have, fs);
            ev.put("name", e.name).put("url", e.url).put("sha256", e.sha256).put("size_bytes", e.size).put("quant", e.meta.optString("quant")).put("license", e.meta.optString("license"))
              .put("kind", e.meta.optString("kind")).put("params_note", e.meta.optString("params_note")).put("params_b", Predictor.paramsB(e.card)).put("downloaded", have).put("filename", e.meta.getString("filename"));
            evs.add(ev);
        }
        return Predictor.rank(evs);
    }

    /** Evaluate a remote GGUF URL without downloading it. */
    public static JSONObject evaluateUrl(JSONObject profile, String url, File modelDir) throws Exception {
        RemoteGguf.Result r = RemoteGguf.fetch(url); String name = url.substring(url.lastIndexOf('/') + 1); int q = name.indexOf('?'); if (q >= 0) name = name.substring(0, q);
        Planner.Card c = Planner.derive(r.gguf, name);
        JSONObject ev = Predictor.evaluate(profile, c, modelDir.getUsableSpace(), new File(modelDir, name).exists(), Profile.filesystemOf(modelDir.getAbsolutePath()));
        return ev.put("name", name).put("url", url).put("size_bytes", r.totalBytes).put("header_bytes_fetched", r.fetchedBytes).put("arch", c.arch).put("params_b", Predictor.paramsB(c)).put("card", c.toJson());
    }

    static String get(String url) throws IOException {
        HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection(); c.setConnectTimeout(15000); c.setReadTimeout(20000); c.setRequestProperty("User-Agent", "meridian/0.2");
        try { if (c.getResponseCode() != 200) throw new IOException("HTTP " + c.getResponseCode() + " for " + url); return new String(RemoteGguf.readUpTo(c.getInputStream(), 8 << 20), "UTF-8"); }
        finally { c.disconnect(); }
    }

    /** Hugging Face search: GGUF repos matching q, most downloaded first. */
    public static JSONArray searchRepos(String q) throws IOException, JSONException {
        JSONArray a = new JSONArray(get("https://huggingface.co/api/models?search=" + URLEncoder.encode(q, "UTF-8") + "&filter=gguf&sort=downloads&direction=-1&limit=15"));
        JSONArray out = new JSONArray(); for (int i = 0; i < a.length(); i++) { JSONObject m = a.getJSONObject(i); if (m.optBoolean("gated", false) || m.optBoolean("private", false)) continue;
            out.put(new JSONObject().put("repo", m.getString("id")).put("downloads", m.optLong("downloads"))); }
        return out;
    }

    /** Single-file GGUFs in a repo (split shards and vision projectors skipped), smallest first. */
    public static JSONArray repoFiles(String repo) throws IOException, JSONException {
        JSONArray a = new JSONArray(get("https://huggingface.co/api/models/" + repo + "/tree/main")); List<JSONObject> l = new ArrayList<>();
        for (int i = 0; i < a.length(); i++) { JSONObject f = a.getJSONObject(i); String p = f.optString("path");
            if (!p.endsWith(".gguf") || p.matches(".*-\\d{5}-of-\\d{5}\\.gguf") || p.toLowerCase(Locale.ROOT).contains("mmproj")) continue;
            JSONObject lfs = f.optJSONObject("lfs");
            l.add(new JSONObject().put("filename", p).put("size_bytes", f.optLong("size")).put("sha256", lfs == null ? JSONObject.NULL : lfs.optString("oid"))
                .put("url", "https://huggingface.co/" + repo + "/resolve/main/" + p)); }
        Collections.sort(l, (x, y) -> Long.compare(x.optLong("size_bytes"), y.optLong("size_bytes")));
        return new JSONArray(l);
    }
}
