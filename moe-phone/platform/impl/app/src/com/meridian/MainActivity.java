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
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission("android.permission.POST_NOTIFICATIONS") != android.content.pm.PackageManager.PERMISSION_GRANTED) requestPermissions(new String[]{"android.permission.POST_NOTIFICATIONS"}, 1);
        try { tools = new Tools(this); } catch (JSONException e) { throw new RuntimeException(e); }
        File pf = new File(getFilesDir(), "profile.json");
        if (pf.exists()) try { profile = new JSONObject(Native.readFile(pf.getAbsolutePath())); } catch (Exception ignored) { }
        LinearLayout root = new LinearLayout(this); root.setOrientation(LinearLayout.VERTICAL); root.setBackgroundColor(BG);
        TextView title = tv("Meridian", 22, FG); title.setPadding(dp(14), dp(12), dp(14), 0); title.setTypeface(Typeface.DEFAULT_BOLD); root.addView(title);
        HorizontalScrollView hs = new HorizontalScrollView(this); LinearLayout bar = new LinearLayout(this); hs.addView(bar); root.addView(hs);
        content = new FrameLayout(this); root.addView(content, new LinearLayout.LayoutParams(-1, 0, 1f));
        tabViews.put("Device", deviceTab()); tabViews.put("Models", modelsTab()); tabViews.put("Chat", chatTab()); tabViews.put("Agent", agentTab()); tabViews.put("Audit", auditTab());
        for (final String name : tabViews.keySet()) bar.addView(btn(name, v -> show(name)));
        setContentView(root); show(profile == null ? "Device" : "Models");
        if (profile != null) try { deviceReport.setText(colorize(Report.render(profile))); } catch (JSONException ignored) { }
        refreshModels();
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
            else if (what.equals("download")) { show("Models"); downloadUrl(task, i.getStringExtra("sha")); }
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
                profile = p; Native.readFile("/proc/version");
                try (FileWriter w = new FileWriter(new File(getFilesDir(), "profile.json"))) { w.write(p.toString(2)); }
                final String rep = Report.render(p);
                onUi(() -> { deviceStatus.setText("Profile saved."); deviceReport.setText(colorize(rep)); refreshModels(); });
            } catch (Exception e) { onUi(() -> deviceStatus.setText("Profiling failed: " + e)); }
            finally { setBusy(false); }
        });
    }
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
        l.addView(modelStatus); l.addView(url); l.addView(sha); l.addView(go); l.addView(cancel); l.addView(dlBar); l.addView(imp); l.addView(tv("Your models", 18, FG)); l.addView(modelList);
        return scroll(l);
    }
    Downloader downloadUrl(final String url, final String sha) {
        if (busy) { toast("Busy"); return null; }
        final Downloader d = new Downloader(); setBusy(true);
        run(() -> { try { File f = d.download(url, internalModels(), sha, (done, total) -> onUi(() -> { dlBar.setProgress(total > 0 ? (int) (done * 1000 / total) : 0); modelStatus.setText("Downloading " + (done >> 20) + " / " + (total >> 20) + " MiB"); }));
                onUi(() -> { modelStatus.setText("Downloaded and verified: " + f.getName()); refreshModels(); saveText("last_download.txt", "OK " + f.getName() + " " + f.length()); }); }
            catch (Exception e) { onUi(() -> { modelStatus.setText("Download failed: " + e.getMessage()); saveText("last_download.txt", "FAIL " + e.getMessage()); }); } finally { setBusy(false); } });
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
            bs.addView(btn("Delete", v -> new AlertDialog.Builder(this).setMessage("Delete " + f.getName() + "?").setPositiveButton("Delete", (d, w) -> { f.delete(); if (f.equals(selectedModel)) selectedModel = null; refreshModels(); }).setNegativeButton("Cancel", null).show()));
            row.addView(bs); row.addView(info); modelList.addView(row, lp);
            run(() -> { String s; try { Planner.Card c = Planner.derive(f); String pl;
                    if (profile == null) pl = "feasibility: profile the device first (Device tab)"; else pl = summarize(Binder.bind(profile, c, Math.min(4096, c.contextLimit), rec.observedTokS(c.modelId)));
                    s = String.format("%s: %d layers, %d experts (%d used per token)\nweights %.2f GiB, touched per token %.0f MiB, KV cache %d KiB per token (f16)\n%s", c.arch, c.nLayer, c.nExpert, c.nUsed, c.totalBytes / 1073741824.0, c.activeBytesPerToken / 1048576.0, c.kvF16PerToken / 1024, pl);
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
            JSONArray cl = profile.getJSONObject("cpu").getJSONArray("clusters"); boolean dot = false, fp16 = false;
            for (int i = 0; i < cl.length(); i++) { String f = cl.getJSONObject(i).getJSONArray("isa_features").toString(); if (f.contains("asimddp")) dot = true; if (f.contains("asimdhp") || f.contains("fphp")) fp16 = true; }
            if (!dot || !fp16) return "EngineUnsupported: the bundled engine is built for armv8.2-a+dotprod+fp16; this CPU reports dotprod=" + dot + " fp16=" + fp16 + ".";
            selectedCard = Planner.derive(selectedModel);
            JSONObject plan = Planner.plan(profile, selectedCard, 2048);
            if (plan.getJSONObject("tiers").getJSONObject("resident").getString("verdict").equals("Infeasible"))
                return "Infeasible for this app: the resident tier needs >= " + plan.getJSONObject("tiers").getJSONObject("resident").getLong("need_lower_bound") / 1048576 + " MiB but the largest grant ever kept is " + plan.getJSONObject("tiers").getJSONObject("resident").getLong("grant_upper_bound") / 1048576 + " MiB. The streamed tier is not exposed in this build (stub PL-E25).";
            return null;
        } catch (Planner.Refusal r) { return "Refusal " + r.reason + ": " + r.getMessage(); } catch (Exception e) { return "error: " + e; }
    }
    void toggleEngine() {
        if (busy) { toast("Busy"); return; }
        if (chat != null) { chat.close(); chat = null; stopService(new Intent(this, KeepAlive.class)); loadBtn.setText("Load engine with selected model"); chatInfo.setText("Engine unloaded."); return; }
        String why = engineRefusal(); if (why != null) { android.util.Log.i("meridian", "engine refusal: " + why); chatInfo.setText(colorize(why)); return; }
        setBusy(true); chatInfo.setText("Loading " + selectedModel.getName() + " ...");
        run(() -> { try {
            Engine.Config cfg = new Engine.Config(); cfg.model = selectedModel; cfg.ctx = 2048; cfg.ubatch = 512; String basis = "default 4 threads, unpinned [prior]";
            JSONObject cm = profile.getJSONObject("cpu").getJSONObject("recommended_compute_mask");
            if (!cm.isNull("value")) { JSONObject v = cm.getJSONObject("value"); cfg.threads = v.getInt("threads"); if (!v.isNull("mask_hex")) cfg.cpuMask = v.getString("mask_hex"); basis = "placement " + v.getString("name") + " [" + cm.getString("provenance") + "]"; }
            android.util.Log.i("meridian", "starting engine " + cfg.model); Chat c = new Chat(this, cfg); c.start(180000); chat = c; android.util.Log.i("meridian", "engine ready"); chatTurns = 0; final String b = basis;
            onUi(() -> { startForegroundService(new Intent(this, KeepAlive.class).putExtra("model", selectedModel.getName())); loadBtn.setText("Unload engine"); chatInfo.setText(colorize("Engine ready (" + b + ")  load " + c.engine().readyInfo().optDouble("load_s") + " s")); });
        } catch (Exception e) { android.util.Log.e("meridian", "engine failed", e); chat = null; onUi(() -> chatInfo.setText("Engine failed: " + e.getMessage())); } finally { setBusy(false); } });
    }
    void send(final String q) {
        if (chat == null) { toast("Load the engine first"); return; } if (busy) { toast("Busy"); return; } setBusy(true);
        final boolean fresh = chatTurns == 0; final Chat c = chat; final StringBuilder acc = new StringBuilder();
        onUi(() -> chatOut.append("\n> " + q + "\n"));
        run(() -> { try {
            c.ask(q, 384, fresh, t -> onUi(() -> chatOut.append(t))); chatTurns++;
            JSONObject d = c.lastResult; JSONObject cond = Regime.snapshot(this);
            rec.write(new JSONObject().put("turn_id", "chat." + System.currentTimeMillis()).put("model_id", selectedModel.getName()).put("prompt_tokens", d.optInt("n_prompt")).put("output_tokens", d.optInt("tokens"))
                .put("prefill_ms", d.optDouble("prefill_s") * 1000).put("predicted", JSONObject.NULL).put("observed", new JSONObject().put("tokens_per_s", d.optDouble("tok_s")).put("prefill_tokens_per_s", d.optDouble("prefill_tps")))
                .put("conditions", cond).put("in_regime", cond.getBoolean("in_regime")).put("outcome", "ok"));
            final String info = String.format("decode %.2f tok/s, prefill %.1f tok/s (%d prompt tokens) - observed this turn [%s]", d.optDouble("tok_s"), d.optDouble("prefill_tps"), d.optInt("n_prompt"), cond.getBoolean("in_regime") ? "measured" : "prior: out of regime");
            onUi(() -> chatInfo.setText(colorize(info)));
        } catch (Exception e) { onUi(() -> chatInfo.setText("Turn failed: " + e.getMessage())); } finally { setBusy(false); } });
    }

    // ---------- Agent ----------
    View agentTab() {
        LinearLayout l = col();
        l.addView(tv("The agent uses only the tools below. Each tool's result is checked against device state; destructive tools ask you first.", 14, DIM));
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
            String res = a.run(task, 6, new Agent.UI() {
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

    // ---------- Audit ----------
    View auditTab() { LinearLayout l = col(); l.addView(tv("Audit trail (structural fields only; no prompt or screen content is stored).", 14, DIM)); auditView = mono(""); l.addView(auditView);
        l.addView(btn("Copy profile JSON", v -> { if (profile != null) { ((android.content.ClipboardManager) getSystemService(CLIPBOARD_SERVICE)).setPrimaryClip(ClipData.newPlainText("profile", profile.toString())); toast("Copied"); } })); return scroll(l); }
    void refreshAudit() { auditView.setText(rec.tail(30)); }
    void toast(String s) { Toast.makeText(this, s, Toast.LENGTH_SHORT).show(); }
}
