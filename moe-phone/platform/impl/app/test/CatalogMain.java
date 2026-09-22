package com.meridian.app;

import org.json.*;
import java.io.*;
import java.nio.file.*;

/** Host tool: builds the app's catalog asset. For each curated entry it reads the REMOTE GGUF header (RemoteGguf, the same
 *  code the app runs) and stores the derived ModelCard, so the app can rank and predict offline and in airplane mode.
 *  Entries whose header size disagrees with the curated size, or whose card cannot be derived, are dropped and reported.
 *    java -cp <classes>:layoutlib.jar com.meridian.app.CatalogMain catalog_raw.json out/catalog.json
 *  With a third argument DIR, any entry whose file also exists in DIR is derived locally too, and the two cards must agree. */
public class CatalogMain {
    public static void main(String[] a) throws Exception {
        JSONArray raw = new JSONArray(new String(Files.readAllBytes(Paths.get(a[0])), "UTF-8"));
        JSONArray out = new JSONArray(); int dropped = 0, crossChecked = 0;
        for (int i = 0; i < raw.length(); i++) {
            JSONObject e = raw.getJSONObject(i);
            if (!e.has("url")) e.put("url", "https://huggingface.co/" + e.getString("repo") + "/resolve/main/" + e.getString("filename"));
            try {
                RemoteGguf.Result r = RemoteGguf.fetch(e.getString("url"));
                if (r.totalBytes != e.getLong("size_bytes")) throw new IOException("size " + r.totalBytes + " != curated " + e.getLong("size_bytes"));
                Planner.Card c = Planner.derive(r.gguf, e.getString("filename"));
                if (a.length > 2) { File loc = new File(a[2], e.getString("filename"));
                    if (loc.exists() && loc.length() == r.totalBytes) { String x = Planner.derive(loc).toJson().toString(), y = c.toJson().toString();
                        if (!x.equals(y)) throw new IOException("remote card != local card:\n local  " + x + "\n remote " + y); crossChecked++; } }
                JSONObject o = new JSONObject();
                for (String k : new String[]{"id", "display_name", "family", "kind", "repo", "filename", "url", "size_bytes", "sha256", "license", "quant", "params_note", "notes"}) o.put(k, e.opt(k));
                o.put("card", c.toJson()).put("header_bytes_fetched", r.fetchedBytes).put("chat_template", c.chatTemplateHint);
                out.put(o);
                System.err.printf("OK   %-30s %-8s arch=%-10s moe=%-5s active=%6.0f MiB  header %d KiB%n", e.getString("id"), e.getString("quant"), c.arch, c.moe, c.activeBytesPerToken / 1048576.0, r.fetchedBytes / 1024);
            } catch (Exception ex) { dropped++; System.err.println("DROP " + e.optString("id") + ": " + ex); }
        }
        JSONObject doc = new JSONObject().put("schema", 1).put("built_at", System.currentTimeMillis() / 1000).put("entries", out)
            .put("note", "cards derived from each file's real GGUF header by RemoteGguf + Planner.derive; sizes and sha256 from the Hugging Face API");
        Files.write(Paths.get(a[1]), doc.toString(1).getBytes("UTF-8"));
        System.err.println("entries=" + out.length() + " dropped=" + dropped + " cross_checked_local=" + crossChecked);
        if (dropped > 0) System.exit(2);
    }
}
