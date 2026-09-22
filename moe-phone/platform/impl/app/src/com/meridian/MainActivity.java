package com.meridian.app;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.*;
import android.graphics.Color;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.*;
import android.text.*;
import android.text.method.ScrollingMovementMethod;
import android.text.style.ForegroundColorSpan;
import android.view.*;
import android.widget.*;
import org.json.*;
import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.regex.*;

public class MainActivity extends Activity {
    static final int BG = 0xFF0F1419, FG = 0xFFE6E6E6, DIM = 0xFF8B949E, ACC = 0xFF3B82F6, GOOD = 0xFF6FCF97, WARN = 0xFFF2C94C, PANEL = 0xFF1A2129;
    final ExecutorService ex = Executors.newSingleThreadExecutor();
    final Handler ui = new Handler(Looper.getMainLooper());
    FrameLayout content; final Map<String, View> tabViews = new LinkedHashMap<>();
    JSONObject profile; File selectedModel; Chat chat; Planner.Card selectedCard; Tools tools; Recorder rec; int chatTurns = 0;
    LinearLayout recList, searchList;
    TextView deviceStatus, deviceReport, modelStatus, chatOut, chatInfo, agentLog, auditView; LinearLayout modelList; ProgressBar dlBar; Button loadBtn;
    CheckBox memGrant; boolean busy;

    // ---------- helpers ----------
    int dp(int v) { return (int) (v * getResources().getDisplayMetrics().density); }
    TextView tv(String s, int size, int color) { TextView t = new TextView(this); t.setText(s); t.setTextSize(size); t.setTextColor(color); t.setPadding(0, dp(4), 0, dp(4)); return t; }
    Button btn(String s, View.OnClickListener l) { Button b = new Button(this); b.setText(s); b.setAllCaps(false); b.setOnClickListener(l); return b; }
    EditText edit(String hint) { EditText e = new EditText(this); e.setHint(hint); e.setTextColor(FG); e.setHintTextColor(DIM); return e; }
    LinearLayout col() { LinearLayout l = new LinearLayout(this); l.setOrientation(LinearLayout.VERTICAL); l.setPadding(dp(14), dp(10), dp(14), dp(24)); return l; }
    ScrollView scroll(View v) { ScrollView s = new ScrollView(this); s.addView(v); return s; }
    void run(Runnable r) { ex.execute(r); }
    void onUi(Runnable r) { ui.post(r); }
    void setBusy(boolean b) { busy = b; onUi(() -> { if (b) getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON); else getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON); }); }
    static final Pattern TAG = Pattern.compile("\\[(measured|prior|unknown)[^\\]]*\\]|not measured|not calibrated|NOT the deployment regime|Infeasible|NotCalibrated|Refusal|verified");
    CharSequence colorize(String s) {
        SpannableString sp = new SpannableString(s); Matcher m = TAG.matcher(s);
        while (m.find()) { String g = m.group(); int c = g.startsWith("[measured") || g.equals("verified") ? GOOD : (g.startsWith("[prior") || g.contains("NOT the") || g.equals("Infeasible") ? WARN : DIM);
            if (g.equals("Refusal")) c = WARN; sp.setSpan(new ForegroundColorSpan(c), m.start(), m.end(), 0); }
        return sp;
    }
    TextView mono(String s) { TextView t = tv(s, 12, FG); t.setTypeface(Typeface.MONOSPACE); t.setTextIsSelectable(true); return t; }
    File internalModels() { File d = new File(getFilesDir(), "models"); d.mkdirs(); return d; }
    File externalModels() { File x = getExternalFilesDir(null); if (x == null) return null; File d = new File(x, "models"); d.mkdirs(); return d; }

    // ---------- lifecycle ----------
    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        if (getActionBar() != null) getActionBar().hide();
        rec = new Recorder(this);
        List<String> ask = new ArrayList<>();   // asked once up front so the agent's contact lookups and notifications work
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission("android.permission.POST_NOTIFICATIONS") != android.content.pm.PackageManager.PERMISSION_GRANTED) ask.add("android.permission.POST_NOTIFICATIONS");
        if (checkSelfPermission("android.permission.READ_CONTACTS") != android.content.pm.PackageManager.PERMISSION_GRANTED) ask.add("android.permission.READ_CONTACTS");
        if (!ask.isEmpty()) requestPermissions(ask.toArray(new String[0]), 1);
        try { tools = new Tools(this); } catch (JSONException e) { throw new RuntimeException(e); }
        File pf = new File(getFilesDir(), "profile.json");
        if (pf.exists()) try { profile = new JSONObject(Native.readFile(pf.getAbsolutePath())); } catch (Exception ignored) { }
        LinearLayout root = new LinearLayout(this); root.setOrientation(LinearLayout.VERTICAL); root.setBackgroundColor(BG);
        TextView title = tv("Meridian", 22, FG); title.setPadding(dp(14), dp(12), dp(14), 0); title.setTypeface(Typeface.DEFAULT_BOLD); root.addView(title);
        HorizontalScrollView hs = new HorizontalScrollView(this); LinearLayout bar = new LinearLayout(this); hs.addView(bar); root.addView(hs);
        content = new FrameLayout(this); root.addView(content, new LinearLayout.LayoutParams(-1, 0, 1f));
        tabViews.put("Device", deviceTab()); tabViews.put("Models", modelsTab()); tabViews.put("Chat", chatTab()); tabViews.put("Agent", agentTab()); tabViews.put("Lab", labTab()); tabViews.put("Audit", auditTab());
        for (final String name : tabViews.keySet()) bar.addView(btn(name, v -> show(name)));
        setContentView(root); show(profile == null ? "Device" : "Models");
        if (profile != null) try { deviceReport.setText(colorize(Report.render(profile))); } catch (JSONException ignored) { }
        refreshModels(); refreshRecommendations();
        autoRun(getIntent());
    }
    // Debug-build automation (ignored unless the APK is debuggable): am start -n com.meridian.app/.MainActivity --es auto load|agent|chat [--es task "..."]
    @Override protected void onNewIntent(Intent i) { super.onNewIntent(i); setIntent(i); autoRun(i); }
    void autoRun(Intent i) {
        if ((getApplicationInfo().flags & android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) == 0 || i == null || i.getStringExtra("auto") == null) return;
        final String what = i.getStringExtra("auto"), task = i.getStringExtra("task");
        saveText("auto.log", "autoRun " + what + " busy=" + busy + " chat=" + (chat != null) + " models=" + allModels().size() + "\n");
        ui.postDelayed(() -> {
            if (selectedModel == null) { List<File> ms = allModels(); if (!ms.isEmpty()) selectedModel = ms.get(0); }
            if (what.equals("load") && chat == null) toggleEngine();
            else if (what.equals("unload") && chat != null) toggleEngine();
            else if (what.equals("agent")) { show("Agent"); runAgent(task); }
            else if (what.equals("chat")) { show("Chat"); send(task); }
            else if (what.equals("placement")) { show("Device"); calibratePlacement(); }
            else if (what.equals("plan")) { show("Lab"); labPlan(); }
            else if (what.equals("calibrate")) { show("Lab"); labCalibrate(false); }
            else if (what.equals("calibrate_streamed")) { show("Lab"); labCalibrate(true); }
            else if (what.equals("loadplan")) { show("Lab"); labLoadPlan(); }
            else if (what.equals("validate")) { show("Lab"); labValidate(); }
            else if (what.equals("t3")) { show("Lab"); labParam.setText(task == null ? "10" : task); labT3(); }
            else if (what.equals("x1")) { show("Lab"); labParam.setText(task == null ? "" : task); labX1(); }
            else if (what.equals("eval")) { show("Lab"); evalVariants = task == null ? null : task.split(","); labEval(); }
            else if (what.equals("select")) { for (File f : allModels()) if (f.getName().contains(task)) selectedModel = f; saveText("auto.log", "selected " + selectedModel); }
            else if (what.equals("download")) { show("Models"); downloadUrl(task, i.getStringExtra("sha")); }
            else if (what.equals("profile")) { show("Device"); memGrant.setChecked(!"nomem".equals(task)); profileDevice(); }
            else if (what.equals("probe")) { show("Device"); measureCompute(); }
            else if (what.equals("recommend")) { show("Models"); refreshRecommendations(); }
            else if (what.equals("evalurl")) { show("Models"); evaluateUrl(task); }
            else if (what.equals("use")) { useModel(task); }
        }, 500);
    }
    @Override protected void onResume() { super.onResume(); Regime.appForeground = true; }
    @Override protected void onPause() { super.onPause(); Regime.appForeground = false; }
    @Override protected void onDestroy() { super.onDestroy(); if (chat != null) chat.close(); }
    void show(String name) { content.removeAllViews(); content.addView(tabViews.get(name)); if (name.equals("Audit")) refreshAudit(); if (name.equals("Models")) refreshModels(); }

    // ---------- Device ----------
    View deviceTab() {
        LinearLayout l = col();
        deviceStatus = tv("Profile this phone once. Everything is measured on-device, as this app, and every number carries whether it was measured or not.", 14, DIM);
        memGrant = new CheckBox(this); memGrant.setText("Include memory-grant probe (fills RAM; may close background apps)"); memGrant.setTextColor(FG); memGrant.setChecked(true);
        deviceReport = mono("");
        l.addView(deviceStatus); l.addView(memGrant);
        l.addView(btn("Profile this device", v -> profileDevice()));
        l.addView(btn("Measure engine compute only (~2 min)", v -> measureCompute()));
        l.addView(btn("Calibrate thread placement (needs a selected model)", v -> calibratePlacement()));
        l.addView(deviceReport);
        return scroll(l);
    }
    void profileDevice() {
        if (busy) { toast("Busy"); return; }
        if (chat != null) { toast("Unload the engine first: profiling needs an idle device"); return; }
        final boolean mem = memGrant.isChecked(); setBusy(true);
        run(() -> {
            try {
                File dir = internalModels();
                JSONObject p = Profile.run(this, dir, mem, s -> onUi(() -> deviceStatus.setText("Working: " + s + " ... keep this app in front, phone unplugged, don't touch it.")));
                attachCompute(p);
                profile = p;
                try (FileWriter w = new FileWriter(new File(getFilesDir(), "profile.json"))) { w.write(p.toString(2)); }
                final String rep = Report.render(p);
                onUi(() -> { deviceStatus.setText("Profile saved."); deviceReport.setText(colorize(rep)); refreshModels(); });
            } catch (Exception e) { onUi(() -> deviceStatus.setText("Profiling failed: " + e)); }
            finally { setBusy(false); }
        });
    }
    /** Runs the engine compute probe and stores it in the profile; a failure is recorded as unknown with its reason, never defaulted. */
    void attachCompute(JSONObject p) throws JSONException {
        try { p.getJSONObject("cpu").put("compute", ComputeProbe.run(this, p, s -> onUi(() -> deviceStatus.setText("Working: " + s + " ... keep this app in front, phone unplugged.")))); }
        catch (Exception e) { p.getJSONObject("cpu").put("compute", new JSONObject().put("value", JSONObject.NULL).put("provenance", "unknown").put("confidence", 0).put("source", "ComputeProbe failed: " + e.getMessage()));
            rec.write(new JSONObject().put("event", "compute_probe_failed").put("why", String.valueOf(e.getMessage())).put("t", System.currentTimeMillis() / 1000.0)); }
    }
    void measureCompute() {
        if (busy || profile == null) { toast(profile == null ? "Profile the device first" : "Busy"); return; }
        if (chat != null) { toast("Unload the engine first"); return; }
        setBusy(true);
        run(() -> { try { attachCompute(profile); saveProfile(); final String rep = Report.render(profile);
                onUi(() -> { deviceStatus.setText("Compute probe done."); deviceReport.setText(colorize(rep)); refreshModels(); refreshRecommendations(); saveText("last_probe.txt", rep); }); }
            catch (Exception e) { onUi(() -> deviceStatus.setText("Compute probe failed: " + e)); } finally { setBusy(false); } });
    }
    void saveProfile() throws IOException, JSONException { try (FileWriter w = new FileWriter(new File(getFilesDir(), "profile.json"))) { w.write(profile.toString(2)); } }
    void calibratePlacement() {
        if (busy || profile == null || selectedModel == null) { toast(profile == null ? "Profile the device first" : "Select a model first"); return; }
        if (chat != null) { toast("Unload the engine first"); return; }
        setBusy(true);
        run(() -> {
            try {
                JSONObject r = Placement.calibrate(this, selectedModel, profile.getJSONObject("cpu").getJSONArray("clusters"), s -> onUi(() -> deviceStatus.setText("Working: " + s)));
                JSONObject rec1 = r.getJSONObject("recommended");
                JSONObject m = new JSONObject().put("value", rec1).put("provenance", r.getBoolean("tie") ? "measured" : "measured").put("confidence", r.getBoolean("tie") ? 0.5 : 0.7).put("source", r.getString("basis")).put("arms", r.getJSONArray("arms")).put("tie", r.getBoolean("tie"))
                    .put("conditions", Regime.snapshot(this));
                profile.getJSONObject("cpu").put("recommended_compute_mask", m);
                try (FileWriter w = new FileWriter(new File(getFilesDir(), "profile.json"))) { w.write(profile.toString(2)); }
                final String rep = Report.render(profile) + "\nPlacement arms:\n" + r.getJSONArray("arms").toString(2);
                final String armsTxt = r.getJSONArray("arms").toString();
                final String msg = "Placement calibrated: " + rec1.getString("name") + (r.getBoolean("tie") ? " (tie -> fewest threads)" : "");
                onUi(() -> { deviceStatus.setText(msg); deviceReport.setText(colorize(rep)); saveText("last_placement.txt", msg + "\n" + armsTxt); });
            } catch (Exception e) { onUi(() -> deviceStatus.setText("Placement calibration failed: " + e)); }
            finally { setBusy(false); }
        });
    }

    // ---------- Models ----------
    View modelsTab() {
        LinearLayout l = col();
        modelStatus = tv("Models are not bundled. Download one (paste a direct https link to a .gguf file), import from storage, or copy into the app's models folder.", 14, DIM);
        final EditText url = edit("https://.../model.gguf"), sha = edit("sha256 (optional)");
        dlBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal); dlBar.setMax(1000);
        final Downloader[] dl = {null};
        Button go = btn("Download", v -> { dl[0] = downloadUrl(url.getText().toString().trim(), sha.getText().toString()); });
        Button cancel = btn("Cancel", v -> { if (dl[0] != null) dl[0].cancelled = true; });
        Button imp = btn("Import .gguf from storage", v -> { Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT); i.addCategory(Intent.CATEGORY_OPENABLE); i.setType("*/*"); startActivityForResult(i, 7); });
        modelList = new LinearLayout(this); modelList.setOrientation(LinearLayout.VERTICAL);
        recList = new LinearLayout(this); recList.setOrientation(LinearLayout.VERTICAL); searchList = new LinearLayout(this); searchList.setOrientation(LinearLayout.VERTICAL);
        final EditText q = edit("search Hugging Face GGUF models, e.g. qwen3");
        l.addView(tv("Recommended for this phone", 18, FG));
        l.addView(tv("Every model below was checked against this phone's measured memory and compute. Speeds are predictions with their range and basis; the app checks them against what it observes.", 13, DIM));
        l.addView(btn("Refresh recommendations", v -> refreshRecommendations())); l.addView(recList);
        l.addView(modelStatus); l.addView(dlBar); l.addView(cancel);
        l.addView(tv("Any model", 18, FG)); l.addView(url); l.addView(sha);
        l.addView(btn("Evaluate URL (reads only the file header)", v -> evaluateUrl(url.getText().toString().trim()))); l.addView(go);
        l.addView(q); l.addView(btn("Search Hugging Face", v -> searchHf(q.getText().toString().trim()))); l.addView(searchList);
        l.addView(imp); l.addView(tv("Your models", 18, FG)); l.addView(modelList);
        return scroll(l);
    }
    Downloader downloadUrl(final String url, final String sha) {
        if (busy) { toast("Busy"); return null; }
        final Downloader d = new Downloader(); setBusy(true);
        // under the foreground service: OxygenOS cut the app's network when the screen locked mid-download (15R, 2026-09-22)
        startForegroundService(new Intent(this, KeepAlive.class).putExtra("text", "Downloading " + url.substring(url.lastIndexOf('/') + 1)));
        run(() -> { try { File f = d.download(url, internalModels(), sha, (done, total) -> onUi(() -> { dlBar.setProgress(total > 0 ? (int) (done * 1000 / total) : 0); modelStatus.setText("Downloading " + (done >> 20) + " / " + (total >> 20) + " MiB"); }));
                onUi(() -> { modelStatus.setText("Downloaded and verified: " + f.getName()); refreshModels(); saveText("last_download.txt", "OK " + f.getName() + " " + f.length()); }); }
            catch (Exception e) { onUi(() -> { modelStatus.setText("Download failed: " + e.getMessage()); saveText("last_download.txt", "FAIL " + e.getMessage()); }); }
            finally { setBusy(false); onUi(() -> { if (chat == null) stopService(new Intent(this, KeepAlive.class)); refreshRecommendations(); }); } });
        return d;
    }
    @Override protected void onActivityResult(int req, int res, final Intent data) {
        super.onActivityResult(req, res, data);
        if (req != 7 || res != RESULT_OK || data == null) return;
        final Uri u = data.getData(); if (busy) { toast("Busy"); return; } setBusy(true);
        run(() -> { File out = null; try {
            String name = "imported.gguf"; try (android.database.Cursor c = getContentResolver().query(u, null, null, null, null)) { if (c != null && c.moveToFirst()) { int i = c.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME); if (i >= 0) name = c.getString(i); } }
            out = new File(internalModels(), name.replaceAll("[^A-Za-z0-9._-]", "_"));
            long total = 0; try (InputStream in = getContentResolver().openInputStream(u); OutputStream o = new FileOutputStream(out)) { byte[] b = new byte[1 << 20]; int n; while ((n = in.read(b)) > 0) { o.write(b, 0, n); total += n; final long t = total; onUi(() -> modelStatus.setText("Importing " + (t >> 20) + " MiB")); } }
            try { Gguf.read(out); } catch (Gguf.GgufError e) { out.delete(); throw new IOException("not a valid GGUF: " + e.getMessage()); }
            final String nm = out.getName(); onUi(() -> { modelStatus.setText("Imported " + nm); refreshModels(); });
        } catch (Exception e) { if (out != null) out.delete(); onUi(() -> modelStatus.setText("Import failed: " + e.getMessage())); } finally { setBusy(false); } });
    }
    static String gib(long b) { return String.format(Locale.ROOT, "%.2f GiB", b / 1073741824.0); }
    /** One line a person can read: what tier, how fast, how sure. Built from the evaluation JSON only. */
    static String evalText(JSONObject e) throws JSONException {
        StringBuilder b = new StringBuilder();
        String v = e.getString("verdict");
        if (v.equals("Runs")) {
            JSONObject d = e.getJSONObject("decode_tok_s"), t = e.getJSONObject("ttft_200_s");
            // a prior-basis figure is shown as a range only: it may not be the sole basis for a promise (05 section 2)
            if (d.getString("provenance").equals("prior")) b.append(String.format(Locale.ROOT, "%s tier: %.1f-%.1f tokens/s [prior], %s%n", e.getString("tier"), d.getDouble("lo"), d.getDouble("hi"), e.getString("speed_class")));
            else b.append(String.format(Locale.ROOT, "%s tier: about %.1f tokens/s (range %.1f-%.1f) [%s], %s%n", e.getString("tier"), d.getDouble("value"), d.getDouble("lo"), d.getDouble("hi"), d.getString("provenance"), e.getString("speed_class")));
            b.append(String.format(Locale.ROOT, "first reply to a 200-token prompt: %.1f s (%.1f-%.1f)", t.getDouble("value"), t.getDouble("lo"), t.getDouble("hi")));
            if (e.getString("tier").equals("streamed")) b.append(String.format(Locale.ROOT, "%nexperts stream from storage through a %s cache", gib(e.getLong("cache_bytes"))));
        } else b.append(v).append(": ").append(e.optString("detail"));
        return b.toString();
    }
    void refreshRecommendations() {
        if (recList == null) return;
        if (profile == null) { recList.removeAllViews(); recList.addView(tv("Profile this phone first (Device tab): recommendations come from its measurements.", 14, WARN)); return; }
        run(() -> { try { final JSONArray r = Catalog.recommend(this, profile, internalModels()); saveText("last_recommend.json", r.toString(2)); onUi(() -> renderRecs(r)); }
            catch (Exception e) { onUi(() -> { recList.removeAllViews(); recList.addView(tv("Recommendation failed: " + e, 14, WARN)); }); } });
    }
    void renderRecs(JSONArray r) {
        recList.removeAllViews();
        for (int i = 0; i < r.length(); i++) { try { final JSONObject e = r.getJSONObject(i);
            String head = (e.optBoolean("recommended") ? "RECOMMENDED  " : "") + e.getString("name") + "  (" + e.optString("params_note") + ", " + e.optString("quant") + ", " + gib(e.getLong("size_bytes")) + ", " + e.optString("kind") + ")";
            LinearLayout row = new LinearLayout(this); row.setOrientation(LinearLayout.VERTICAL); row.setBackgroundColor(PANEL); row.setPadding(dp(10), dp(8), dp(10), dp(8));
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2); lp.setMargins(0, dp(4), 0, dp(4));
            TextView h = tv(head, 14, e.optBoolean("recommended") ? GOOD : FG); h.setTypeface(Typeface.DEFAULT_BOLD); row.addView(h); row.addView(mono(evalText(e)));
            if (!e.optString("license").isEmpty() && !e.optString("license").startsWith("apache") && !e.optString("license").equals("mit")) row.addView(tv("license: " + e.optString("license"), 12, WARN));
            LinearLayout bs = new LinearLayout(this);
            if (e.optBoolean("downloaded")) bs.addView(btn("Use this model", v -> useModel(e.optString("filename"))));
            else if (e.getString("verdict").equals("Runs")) bs.addView(btn("Download " + gib(e.getLong("size_bytes")), v -> downloadUrl(e.optString("url"), e.optString("sha256"))));
            row.addView(bs); recList.addView(row, lp);
        } catch (JSONException ignored) { } }
    }
    void useModel(String filename) {
        for (File f : allModels()) if (f.getName().equals(filename)) selectedModel = f;
        if (selectedModel == null) { toast("not found: " + filename); return; }
        show("Chat"); if (chat == null) toggleEngine();
    }
    void evaluateUrl(final String url) {
        if (profile == null) { modelStatus.setText("Profile this phone first."); return; } if (url.isEmpty()) return;
        modelStatus.setText("Reading the header of " + url + " ...");
        run(() -> { try { JSONObject e = Catalog.evaluateUrl(profile, url, internalModels()); final String t = e.getString("name") + " (" + e.getString("arch") + ", " + gib(e.getLong("size_bytes")) + ", header " + (e.getLong("header_bytes_fetched") >> 10) + " KiB read)\n" + evalText(e);
                saveText("last_evaluate.json", e.toString(2)); onUi(() -> modelStatus.setText(colorize(t))); }
            catch (Exception e) { onUi(() -> modelStatus.setText("Could not evaluate: " + e.getMessage())); } });
    }
    void searchHf(final String q) {
        if (q.isEmpty()) return; searchList.removeAllViews(); searchList.addView(tv("Searching...", 13, DIM));
        run(() -> { try { final JSONArray repos = Catalog.searchRepos(q); onUi(() -> { searchList.removeAllViews(); if (repos.length() == 0) searchList.addView(tv("No ungated GGUF repos found.", 13, DIM));
                for (int i = 0; i < repos.length(); i++) { final String repo = repos.optJSONObject(i).optString("repo"); searchList.addView(btn(repo + "  (" + repos.optJSONObject(i).optLong("downloads") + " downloads)", v -> listRepo(repo))); } }); }
            catch (Exception e) { onUi(() -> { searchList.removeAllViews(); searchList.addView(tv("Search failed: " + e.getMessage(), 13, WARN)); }); } });
    }
    void listRepo(final String repo) {
        searchList.removeAllViews(); searchList.addView(tv("Files in " + repo + " ...", 13, DIM));
        run(() -> { try { final JSONArray fs = Catalog.repoFiles(repo); onUi(() -> { searchList.removeAllViews(); searchList.addView(tv(repo, 14, FG));
                for (int i = 0; i < fs.length(); i++) { final JSONObject f = fs.optJSONObject(i); LinearLayout row = new LinearLayout(this); row.setOrientation(LinearLayout.VERTICAL);
                    row.addView(tv(f.optString("filename") + "  " + gib(f.optLong("size_bytes")), 13, FG)); LinearLayout bs = new LinearLayout(this);
                    bs.addView(btn("Evaluate", v -> evaluateUrl(f.optString("url")))); bs.addView(btn("Download", v -> downloadUrl(f.optString("url"), f.optString("sha256"))));
                    row.addView(bs); searchList.addView(row); } }); }
            catch (Exception e) { onUi(() -> { searchList.removeAllViews(); searchList.addView(tv("Listing failed: " + e.getMessage(), 13, WARN)); }); } });
    }
    List<File> allModels() { List<File> l = new ArrayList<>(); for (File d : new File[]{internalModels(), externalModels()}) if (d != null && d.listFiles() != null) for (File f : d.listFiles()) if (f.getName().endsWith(".gguf")) l.add(f); return l; }
    void refreshModels() {
        if (modelList == null) return; modelList.removeAllViews();
        final List<File> ms = allModels(); if (ms.isEmpty()) modelList.addView(tv("(none yet)", 14, DIM));
        for (final File f : ms) {
            final TextView info = mono(f.getName() + "\n" + (f.length() >> 20) + " MiB   " + (f.getPath().startsWith(getFilesDir().getPath()) ? "internal" : "app external (FUSE path: profile it before trusting streaming)") + "\nreading header...");
            LinearLayout row = new LinearLayout(this); row.setOrientation(LinearLayout.VERTICAL); row.setBackgroundColor(PANEL); row.setPadding(dp(10), dp(8), dp(10), dp(8));
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2); lp.setMargins(0, dp(6), 0, dp(6));
            LinearLayout bs = new LinearLayout(this);
            bs.addView(btn("Select", v -> { selectedModel = f; selectedCard = null; toast("Selected " + f.getName()); }));
            bs.addView(btn("Chat", v -> useModel(f.getName())));
            bs.addView(btn("Delete", v -> new AlertDialog.Builder(this).setMessage("Delete " + f.getName() + "?").setPositiveButton("Delete", (d, w) -> { f.delete(); if (f.equals(selectedModel)) selectedModel = null; refreshModels(); }).setNegativeButton("Cancel", null).show()));
            row.addView(bs); row.addView(info); modelList.addView(row, lp);
            run(() -> { String s; try { Planner.Card c = Planner.derive(f); String pl;
                    if (profile == null) pl = "feasibility: profile the device first (Device tab)";
                    else { pl = evalText(Predictor.evaluate(profile, c, f.getParentFile().getUsableSpace(), true, Profile.filesystemOf(f.getAbsolutePath())));
                        List<Double> obs = rec.observedTokS(c.modelId); if (obs.size() >= 3) { List<Double> o = new ArrayList<>(obs); Collections.sort(o); pl += String.format(Locale.ROOT, "%nobserved on this phone: %.1f tokens/s median over %d turns [measured]", o.get(o.size() / 2), o.size()); } }
                    s = String.format("%s: %d layers, %s\nweights %.2f GiB, touched per token %.0f MiB, KV cache %d KiB per token (f16)\n%s", c.arch, c.nLayer, c.moe ? c.nExpert + " experts (" + c.nUsed + " used per token)" + (c.streamable ? ", streamable" : ", resident only") : "dense", c.totalBytes / 1073741824.0, c.activeBytesPerToken / 1048576.0, c.kvF16PerToken / 1024, pl);
                } catch (Planner.Refusal r) { s = "Refusal " + r.reason + ": " + r.getMessage(); } catch (Exception e) { s = "error: " + e; }
                final String out = f.getName() + "\n" + (f.length() >> 20) + " MiB\n" + s; onUi(() -> info.setText(colorize(out))); });
        }
    }

    static String summarize(JSONObject b) throws JSONException {
        StringBuilder sb = new StringBuilder();
        if (b.has("refusal")) return "Refusal " + b.getJSONObject("refusal").getString("reason") + ": " + b.getJSONObject("refusal").getString("detail");
        JSONObject t = b.getJSONObject("feasibility").getJSONObject("tiers");
        for (String k : new String[]{"resident", "streamed"}) { JSONObject v = t.getJSONObject(k);
            sb.append(k).append(": ").append(v.getString("verdict")).append(" (needs >= ").append(v.getLong("need_lower_bound") >> 20).append(" MiB, at most ").append(v.getLong("grant_upper_bound") >> 20).append(" MiB can be kept)\n"); }
        JSONObject e = b.getJSONObject("expectation");
        sb.append(e.getString("provenance").equals("measured") ? String.format("expected decode %.2f tok/s [measured]  (%s)", e.getDouble("decode_tok_s"), e.getString("basis")) : "speed: not calibrated yet [unknown] - " + e.getString("reason"));
        return sb.toString();
    }

    // ---------- Chat ----------
    View chatTab() {
        LinearLayout l = col();
        chatInfo = tv("Load a model to chat. Tokens/s shown are observed on this turn, not predictions.", 14, DIM);
        loadBtn = btn("Load engine with selected model", v -> toggleEngine());
        final EditText in = edit("Message"); chatOut = mono("");
        chatOut.setMovementMethod(new ScrollingMovementMethod());
        l.addView(chatInfo); l.addView(loadBtn);
        l.addView(btn("New chat", v -> { chatTurns = 0; chatOut.setText(""); }));
        l.addView(in);
        l.addView(btn("Send", v -> { String q = in.getText().toString().trim(); if (q.isEmpty()) return; in.setText(""); send(q); }));
        l.addView(btn("Stop", v -> { if (chat != null) chat.cancel(); }));
        l.addView(chatOut);
        return scroll(l);
    }
    String engineRefusal() {
        if (selectedModel == null) return "Select a model on the Models tab first.";
        if (profile == null) return "Profile the device first (Device tab).";
        try {
            if (!Topo.hasIsa(profile, "asimddp") || !(Topo.hasIsa(profile, "asimdhp") || Topo.hasIsa(profile, "fphp")))
                return "EngineUnsupported: the bundled engine needs armv8.2-a+dotprod+fp16 on every core.";
            selectedCard = Planner.derive(selectedModel); return null;
        } catch (Planner.Refusal r) { return "Refusal " + r.reason + ": " + r.getMessage(); } catch (Exception e) { return "error: " + e; }
    }
    Lease lease; Governor governor; Engine.Config curCfg; JSONObject curPlan;
    void toggleEngine() {
        if (busy) { toast("Busy"); return; }
        if (chat != null) { unloadEngine(); loadBtn.setText("Load engine with selected model"); chatInfo.setText("Engine unloaded."); return; }
        String why = engineRefusal(); if (why != null) { chatInfo.setText(colorize(why)); return; }
        try {
            JSONObject plan = AutoPlan.plan(this, profile, selectedCard, selectedModel); saveText("plan.json", plan.toString(2));
            if (plan.has("refusal")) { chatInfo.setText(colorize("Refusal " + plan.getJSONObject("refusal").getString("reason") + ": " + plan.getJSONObject("refusal").optString("detail"))); return; }
            loadEngine(PlanV2.configFromJson(plan.getJSONObject("config"), selectedModel), plan);
        } catch (Exception e) {   // no compute probe yet: the model still runs, on the topology placement, with no promise attached
            try { Engine.Config cfg = AutoPlan.config(profile, selectedCard, selectedModel, "resident", 2048, 0); rec.write(new JSONObject().put("event", "plan_unavailable").put("why", String.valueOf(e.getMessage())));
                chatInfo.setText(colorize("No prediction (" + e.getMessage() + "); loading without one.")); loadEngine(cfg, null); } catch (JSONException x) { chatInfo.setText("error: " + x); } }
    }
    void unloadEngine() {
        if (governor != null) { governor.stop(); governor = null; } if (lease != null) { lease.revoke(); lease = null; }
        if (chat != null) { chat.close(); chat = null; } stopService(new Intent(this, KeepAlive.class));
    }
    /** Load the engine with `cfg`; when a plan is given, register its lease and start the governor with its ladder and falsification rule. */
    void loadEngine(final Engine.Config cfg, final JSONObject plan) {
        setBusy(true); onUi(() -> chatInfo.setText("Loading " + cfg.describe() + " ..."));
        run(() -> { try {
            Chat c = new Chat(this, cfg); c.start(900000); chat = c; chatTurns = 0; curCfg = cfg; curPlan = plan;
            JSONObject lst = null; String leaseTxt = "no plan: no lease, no governor (the configuration is uncalibrated)";
            if (plan != null) { long floor = plan.getJSONObject("lease").getLong("floor"), target = plan.getJSONObject("lease").getLong("target");
                try { lease = Lease.request(profile, floor, target); lst = lease.verify(); leaseTxt = "lease granted " + (lease.granted >> 20) + " MiB, verified resident " + (lease.verifiedResident >> 20) + " MiB (swapped " + (lease.swapped >> 20) + " MiB)";
                    lease.startHeartbeat(ui, 5000, (why, st) -> { if (governor != null) governor.onMemoryPressure(why); }); }
                catch (Planner.Refusal r) { lease = null; leaseTxt = "no memory lease (" + r.reason + ": " + r.getMessage() + "); the governor still watches heat and speed"; rec.write(new JSONObject().put("event", "lease_refused").put("why", r.getMessage())); }
                JSONObject pd = plan.getJSONObject("chosen").getJSONObject("decode_tok_s"); leaseTxt += String.format(Locale.ROOT, "%npredicted %.1f tokens/s (range %.1f-%.1f) [%s]", pd.getDouble("value"), pd.getDouble("lo"), pd.getDouble("hi"), pd.getString("provenance"));
                governor = new Governor(this, new Governor.Host() {
                    public void applyRung(JSONObject r, String why) { applyRungAsync(r, why); }
                    public void invalidated(String why) { onUi(() -> chatInfo.setText(colorize("Plan falsified: " + why + ". Recalibrate before trusting it."))); } }, rec, plan); governor.start(); }
            final String fl = leaseTxt;
            onUi(() -> { startForegroundService(new Intent(this, KeepAlive.class).putExtra("model", cfg.model.getName())); loadBtn.setText("Unload engine"); chatInfo.setText(colorize("Engine ready: " + cfg.describe() + "\nload " + c.engine().readyInfo().optDouble("load_s") + " s; " + fl)); });
        } catch (Exception e) { chat = null; onUi(() -> chatInfo.setText("Engine failed: " + e.getMessage())); } finally { setBusy(false); } });
    }
    /** Execute a ladder rung chosen by the governor: reload with the rung's change, or unload. Queued behind whatever job is running. */
    void applyRungAsync(final JSONObject r, final String why) {
        run(() -> { try {
            String act = r.getString("action"); Engine.Config c = curCfg; JSONObject pl = curPlan; if (c == null) return;
            rec.write(new JSONObject().put("event", "ladder_rung").put("action", act).put("why", why).put("expected_cost", r.getJSONObject("expected_cost")).put("t", System.currentTimeMillis() / 1000.0));
            onUi(() -> chatInfo.setText(colorize("Governor: " + act + " because " + why)));
            if (act.equals("suspend") || act.equals("refuse")) { onUi(this::unloadEngine); return; }
            Engine.Config n = PlanV2.configFromJson(PlanV2.cfgJson(c), c.model); JSONObject p = r.getJSONObject("params");
            if (act.equals("change_placement")) { n.threads = p.getInt("threads"); n.cpuMask = p.getString("mask_hex"); } else if (act.equals("reduce_context")) n.ctx = p.getInt("ctx"); else if (act.equals("shrink_cache")) n.cacheCeilMb = n.cacheFloorMb;
            final Engine.Config nn = n; onUi(() -> { unloadEngine(); loadEngine(nn, pl); });
        } catch (Exception e) { onUi(() -> chatInfo.setText("Governor failed to apply rung: " + e)); } });
    }
    @Override public void onTrimMemory(int level) { super.onTrimMemory(level); if (level >= TRIM_MEMORY_RUNNING_LOW && governor != null) governor.onMemoryPressure("onTrimMemory level " + level); }
    void send(final String q) {
        if (chat == null) { toast("Load the engine first"); return; } if (busy) { toast("Busy"); return; } setBusy(true);
        final boolean fresh = chatTurns == 0; final Chat c = chat; final StringBuilder acc = new StringBuilder();
        onUi(() -> chatOut.append("\n> " + q + "\n"));
        run(() -> { try {
            c.ask(q, 384, fresh, t -> onUi(() -> chatOut.append(t))); chatTurns++;
            JSONObject d = c.lastResult; JSONObject cond = Regime.snapshot(this);
            rec.write(new JSONObject().put("turn_id", "chat." + System.currentTimeMillis()).put("model_id", selectedModel.getName()).put("prompt_tokens", d.optInt("n_prompt")).put("output_tokens", d.optInt("tokens"))
                .put("prefill_ms", d.optDouble("prefill_s") * 1000).put("predicted", curPlan == null ? JSONObject.NULL : curPlan.getJSONObject("chosen").getJSONObject("decode_tok_s")).put("observed", new JSONObject().put("tokens_per_s", d.optDouble("tok_s")).put("prefill_tokens_per_s", d.optDouble("prefill_tps")))
                .put("conditions", cond).put("in_regime", cond.getBoolean("in_regime")).put("outcome", "ok"));
            String pr = ""; if (curPlan != null) { JSONObject pd = curPlan.getJSONObject("chosen").getJSONObject("decode_tok_s"); double ob = d.optDouble("tok_s");
                pr = String.format(Locale.ROOT, "%npredicted %.1f (%.1f-%.1f) [%s]: %s", pd.getDouble("value"), pd.getDouble("lo"), pd.getDouble("hi"), pd.getString("provenance"), ob >= pd.getDouble("lo") && ob <= pd.getDouble("hi") ? "inside the range" : "OUTSIDE the range"); }
            final String info = String.format("decode %.2f tok/s, prefill %.1f tok/s (%d prompt tokens) - observed this turn [%s]", d.optDouble("tok_s"), d.optDouble("prefill_tps"), d.optInt("n_prompt"), cond.getBoolean("in_regime") ? "measured" : "prior: out of regime") + pr + "\nconfig: " + (curCfg == null ? "?" : curCfg.describe());
            onUi(() -> chatInfo.setText(colorize(info))); if (governor != null && cond.getBoolean("in_regime")) governor.observe(d.optDouble("tok_s"));
        } catch (Exception e) { onUi(() -> chatInfo.setText("Turn failed: " + e.getMessage())); } finally { setBusy(false); } });
    }

    // ---------- Agent ----------
    View agentTab() {
        LinearLayout l = col();
        l.addView(tv("Ask for anything the tools below can do, in your own words. The agent picks tools step by step, each result is checked against the phone's real state, and anything that reaches other people (messages, calls) or cannot be undone asks you first.", 14, DIM));
        try { l.addView(mono(tools.schemaText())); } catch (JSONException ignored) { }
        final EditText task = edit("e.g. Save a note that says buy milk, then tell me my battery level");
        agentLog = mono("");
        l.addView(task);
        l.addView(btn("Run task", v -> runAgent(task.getText().toString().trim())));
        l.addView(agentLog);
        return scroll(l);
    }
    void runAgent(final String task) {
        if (chat == null) { toast("Load the engine on the Chat tab first"); return; } if (task.isEmpty() || busy) return; setBusy(true); chatTurns = 0;
        onUi(() -> agentLog.setText("Task: " + task + "\n"));
        final Agent a = new Agent(this, chat, tools, rec, selectedModel.getName());
        run(() -> { try {
            String res = a.runLoop(task, 8, new Agent.UI() {
                public void log(String s) { onUi(() -> agentLog.append(s + "\n")); }
                public void token(String t) { }
                public boolean consent(String tool, String args) { final CountDownLatch cd = new CountDownLatch(1); final boolean[] ok = {false};
                    onUi(() -> new AlertDialog.Builder(MainActivity.this).setTitle("Allow " + tool + "?").setMessage(args + "\n\nThis cannot be undone.").setCancelable(false)
                        .setPositiveButton("Allow once", (d, w) -> { ok[0] = true; cd.countDown(); }).setNegativeButton("Deny", (d, w) -> cd.countDown()).show());
                    try { cd.await(); } catch (InterruptedException e) { return false; } return ok[0]; }
            });
            onUi(() -> { agentLog.append("\nResult: " + res + "\n"); saveText("last_agent.txt", agentLog.getText().toString()); });
        } catch (Exception e) { onUi(() -> { agentLog.append("\nFailed: " + e.getMessage() + "\n"); saveText("last_agent.txt", agentLog.getText().toString()); }); } finally { setBusy(false); } });
    }
    void saveText(String name, String t) { try (FileWriter w = new FileWriter(new File(getFilesDir(), name))) { w.write(t); } catch (IOException ignored) { } }


    // ---------- Lab: planner, calibration, validation, thermal, foreground grant, evaluation ----------
    String[] evalVariants;
    TextView labOut; EditText labCtx, labOutTok, labLat, labParam;
    View labTab() {
        LinearLayout l = col();
        l.addView(tv("Measurement lab. Everything here runs the real engine on this phone; results are saved and shown with their basis. Keep the app in front, phone unplugged.", 14, DIM));
        labCtx = edit("prompt tokens (default 200)"); labOutTok = edit("output tokens (default 64)"); labLat = edit("latency target to first token, ms (default 5000)"); labParam = edit("parameter (minutes / package name)");
        l.addView(labCtx); l.addView(labOutTok); l.addView(labLat); l.addView(labParam);
        l.addView(btn("Plan (selected model)", v -> labPlan()));
        l.addView(btn("Calibrate resident tier (~3 min)", v -> labCalibrate(false)));
        l.addView(btn("Calibrate streamed tier", v -> labCalibrate(true)));
        l.addView(btn("Load engine from plan", v -> labLoadPlan()));
        l.addView(btn("Validate planner (pre-registered)", v -> labValidate()));
        l.addView(btn("Sustained thermal test (T3)", v -> labT3()));
        l.addView(btn("Foreground memory grant (X1)", v -> labX1()));
        l.addView(btn("Run agent evaluation (suite-v1)", v -> labEval()));
        labOut = mono(""); l.addView(labOut); return scroll(l);
    }
    long num(EditText e, long d) { try { return Long.parseLong(e.getText().toString().trim()); } catch (Exception x) { return d; } }
    void labShow(final String name, final String text) { onUi(() -> { labOut.setText(colorize(text)); }); saveText("last_" + name + ".txt", text); }
    boolean labReady(boolean needIdle) {
        if (busy) { toast("Busy"); return false; } if (profile == null) { labShow("error", "Profile the device first."); return false; } if (selectedModel == null) { List<File> ms = allModels(); if (ms.isEmpty()) { labShow("error", "No model."); return false; } selectedModel = ms.get(0); }
        if (needIdle && chat != null) { labShow("error", "Unload the engine first: this experiment needs the device otherwise idle."); return false; } return true;
    }
    static String planSummary(JSONObject p) throws JSONException {
        StringBuilder b = new StringBuilder();
        if (p.has("refusal")) b.append("Refusal ").append(p.getJSONObject("refusal").getString("reason")).append(": ").append(p.getJSONObject("refusal").getString("detail")).append('\n');
        if (p.has("grant_basis")) b.append("memory basis: ").append(p.getString("grant_basis")).append('\n');
        JSONArray c = p.optJSONArray("candidates"); if (c != null) for (int i = 0; i < c.length(); i++) { JSONObject o = c.getJSONObject(i); b.append(o.getString("tier")).append(": ").append(o.getString("verdict"));
            if (o.has("detail")) b.append(" - ").append(o.getString("detail")); if (o.has("ttft_ms")) b.append(String.format("  TTFT %.0f-%.0f ms [%s], decode %.2f-%.2f tok/s", o.getJSONObject("ttft_ms").getDouble("lo_ms"), o.getJSONObject("ttft_ms").getDouble("hi_ms"), o.getString("calibration_state"), o.getJSONObject("decode_tok_s").getDouble("lo"), o.getJSONObject("decode_tok_s").getDouble("hi")));
            if (o.has("offer")) b.append("\n   offer: ").append(o.getString("offer")); b.append('\n'); }
        if (p.has("chosen")) b.append("CHOSEN ").append(p.getJSONObject("chosen").getString("tier")).append(" config ").append(p.getJSONObject("config")).append("\nlease ").append(p.getJSONObject("lease")).append("\nfalsification ").append(p.getJSONObject("falsification")).append("\nladder ").append(p.getJSONArray("ladder").length()).append(" rungs\n");
        return b.toString();
    }
    void labPlan() { if (!labReady(false)) return; run(() -> { try { Planner.Card c = Planner.derive(selectedModel); JSONObject p = PlanV2.plan(this, profile, c, selectedModel, num(labCtx, 200), num(labOutTok, 64), num(labLat, 5000)); saveText("plan.json", p.toString(2)); labShow("plan", planSummary(p)); } catch (Exception e) { labShow("plan", "Plan failed: " + e); } }); }
    void labCalibrate(final boolean streamed) {
        if (!labReady(true)) return; setBusy(true);
        run(() -> { try { Planner.Card c = Planner.derive(selectedModel); long grant = profile.getJSONObject("memory").getJSONObject("grantable_foreground").isNull("value") ? profile.getJSONObject("memory").getJSONObject("grantable_quiesced").getLong("value") : profile.getJSONObject("memory").getJSONObject("grantable_foreground").getLong("value");
            Engine.Config cfg = PlanV2.configFor(profile, c, selectedModel, streamed ? "streamed" : "resident", 1024, grant); long t0 = System.currentTimeMillis();
            Cells cells = Calibration.run(this, cfg, new int[]{32, 128, 512}, 2, s -> labShow("calibrate", "Calibrating " + cfg.describe() + "\n" + s));
            labShow("calibrate", "Calibrated in " + (System.currentTimeMillis() - t0) / 1000 + " s, in_regime=" + cells.inRegime + "\n" + cells.toJson().toString(2));
        } catch (Exception e) { labShow("calibrate", "Calibration failed: " + e); } finally { setBusy(false); } });
    }
    void labLoadPlan() {
        if (!labReady(true)) return;
        run(() -> { try { Planner.Card c = Planner.derive(selectedModel); JSONObject p = PlanV2.plan(this, profile, c, selectedModel, num(labCtx, 200), num(labOutTok, 64), num(labLat, 5000)); labShow("plan", planSummary(p));
            if (p.has("refusal")) return; Engine.Config cfg = PlanV2.configFromJson(p.getJSONObject("config"), selectedModel); onUi(() -> loadEngine(cfg, p)); } catch (Exception e) { labShow("plan", "failed: " + e); } });
    }
    void labValidate() {
        if (!labReady(true)) return; setBusy(true);
        run(() -> { try { Planner.Card c = Planner.derive(selectedModel); Engine.Config cfg = PlanV2.configFor(profile, c, selectedModel, "resident", 1024, 0);
            JSONObject r = Validate.run(this, cfg, c, profile, s -> labShow("validate", s)); labShow("validate", r.toString(2)); } catch (Exception e) { labShow("validate", "Validation failed: " + e); } finally { setBusy(false); } });
    }
    void labT3() {
        if (!labReady(true)) return; setBusy(true); final int secs = (int) num(labParam, 10) * 60;
        run(() -> { try { Planner.Card c = Planner.derive(selectedModel); Engine.Config cfg = PlanV2.baseConfig(profile, selectedModel, 1024);
            JSONObject r = Thermal.run(this, cfg, secs, s -> labShow("t3", s));
            JSONObject th = profile.getJSONObject("thermal"); boolean in = r.getJSONObject("conditions").getBoolean("in_regime"); JSONObject cond = r.getJSONObject("conditions");
            JSONArray dr = new JSONArray(); JSONArray d = r.getJSONArray("derate"); for (int i = 0; i < d.length(); i++) dr.put(d.getJSONObject(i));
            th.put("sustained_derate", new JSONObject().put("value", dr).put("provenance", in ? "measured" : "prior").put("confidence", in ? 0.6 : 0.3).put("source", r.getString("temperature_source") + "; " + r.getString("config")).put("conditions", cond));
            th.put("time_to_throttle_s", new JSONObject().put("value", r.get("time_to_throttle_s")).put("provenance", in ? "measured" : "prior").put("source", "T3 sustained run"));
            th.put("recovery_s", new JSONObject().put("value", r.get("recovery_s")).put("provenance", in ? "measured" : "prior").put("source", "T3 recovery polling"));
            try (FileWriter w = new FileWriter(new File(getFilesDir(), "profile.json"))) { w.write(profile.toString(2)); }
            r.remove("series"); labShow("t3", r.toString(2)); } catch (Exception e) { labShow("t3", "T3 failed: " + e); } finally { setBusy(false); } });
    }
    void labX1() {
        if (!labReady(true)) return; setBusy(true); final String pkg = labParam.getText().toString().trim().isEmpty() ? "com.google.android.youtube" : labParam.getText().toString().trim();
        run(() -> { try { if (!ForegroundGrant.usageAccess(this)) labShow("x1", "Usage Access is not granted: the foreground app cannot be verified, so the result will be labelled prior. Grant it in Settings > Usage access.");
            JSONObject m = ForegroundGrant.run(this, pkg, s -> labShow("x1", s));
            profile.getJSONObject("memory").put("grantable_foreground", m); try (FileWriter w = new FileWriter(new File(getFilesDir(), "profile.json"))) { w.write(profile.toString(2)); }
            onUi(() -> startActivity(new Intent(this, MainActivity.class).addFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT))); labShow("x1", m.toString(2)); } catch (Exception e) { labShow("x1", "X1 failed: " + e); } finally { setBusy(false); } });
    }
    void labEval() {
        if (chat == null) { labShow("eval", "Load the engine (Chat tab) first: the model-driven variants need it."); return; } if (busy) { toast("Busy"); return; } setBusy(true);
        run(() -> { try { String[] variants = evalVariants != null ? evalVariants : new String[]{"loop", "plan_act_answer", "keyword_router", "constant:say_hello", "constant:battery_status"};
            JSONObject r = Eval.run(this, chat, tools, rec, selectedModel.getName(), variants, s -> labShow("eval", "Evaluating: " + s)); labShow("eval", r.getJSONObject("summary").toString(2)); }
            catch (Exception e) { labShow("eval", "Eval failed: " + e); } finally { setBusy(false); } });
    }

    // ---------- Audit ----------
    View auditTab() { LinearLayout l = col(); l.addView(tv("Audit trail (structural fields only; no prompt or screen content is stored).", 14, DIM)); auditView = mono(""); l.addView(auditView);
        l.addView(btn("Copy profile JSON", v -> { if (profile != null) { ((android.content.ClipboardManager) getSystemService(CLIPBOARD_SERVICE)).setPrimaryClip(ClipData.newPlainText("profile", profile.toString())); toast("Copied"); } })); return scroll(l); }
    void refreshAudit() { auditView.setText(rec.tail(30)); }
    void toast(String s) { Toast.makeText(this, s, Toast.LENGTH_SHORT).show(); }
}
