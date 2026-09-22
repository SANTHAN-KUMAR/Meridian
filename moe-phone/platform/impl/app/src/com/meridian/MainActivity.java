package com.meridian.app;

import android.app.Activity;
import android.app.AlarmManager;
import android.app.AlertDialog;
import android.app.Dialog;
import android.app.NotificationManager;
import android.content.*;
import android.content.res.Configuration;
import android.graphics.Color;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.*;
import android.provider.Settings;
import android.speech.RecognizerIntent;
import android.text.*;
import android.text.method.ScrollingMovementMethod;
import android.text.style.ForegroundColorSpan;
import android.view.*;
import android.view.inputmethod.EditorInfo;
import android.widget.*;
import org.json.*;
import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.regex.*;

/** Meridian's single activity. The platform logic (profiling, recommendations, planning, the engine, the agent, the lab) is
 *  unchanged from the developer build; the UI on top follows the brand kit: a task hub, a live task view built from the agent's
 *  own log, plain-language problems, and every technical detail behind Developer tools. Debug automation (autoRun) is kept
 *  exactly: same intent extras, same saved files. */
public class MainActivity extends Activity {
    int BG, FG, DIM, ACC, GOOD, WARN, PANEL;   // developer-tools palette, derived from the theme
    final ExecutorService ex = Executors.newSingleThreadExecutor();
    final Handler ui = new Handler(Looper.getMainLooper());
    JSONObject profile; File selectedModel; Chat chat; Planner.Card selectedCard; Tools tools; Recorder rec; int chatTurns = 0;
    LinearLayout recList, searchList;
    TextView deviceStatus, deviceReport, modelStatus, chatOut, chatInfo, agentLog, auditView; LinearLayout modelList; ProgressBar dlBar; Button loadBtn;
    CheckBox memGrant; volatile boolean busy;

    // ---------- design system and navigation ----------
    UiTheme T; UiKit K; SharedPreferences prefs;
    FrameLayout root, screenHost; View drawer; String screen = "home"; final ArrayDeque<String> backStack = new ArrayDeque<>();
    final Set<String> seen = new HashSet<>();
    /** True the first time `key` is rendered: new items animate in once, refreshes never replay it. */
    boolean fresh(String key) { return seen.add(key); }
    TextView workingElapsed, runningElapsed;
    TextView headerSub; final Runnable ticker = new Runnable() { public void run() { tick(); } };
    LinearLayout body; ScrollView bodyScroll; Runnable bodyFill; boolean refreshQueued; final Set<String> pendingRefresh = new HashSet<>();
    // live state shown by the new screens
    UiTask task; String engineNote; boolean setupRunning, measuring; String setupStage; UiHumanize.Problem setupProblem, phoneProblem, modelProblem;
    JSONArray recs; String recsError; Downloader activeDl; String dlName; long dlDone, dlTotal; UiKit.Bar dlBarNew; TextView dlLabel;
    JSONObject lastEval; String evalMsg, modelMsg; JSONArray searchRepos, repoFiles; String repoName, searchMsg; String homeDraft = "", taskDraft = "", chatDraft = "";
    EditText voiceTarget; String devTab = "Device";
    final List<ChatMsg> chatMsgs = new ArrayList<>(); boolean thinkFirst;
    static final class ChatMsg { int renderedPhase = -1; String user; final StringBuilder raw = new StringBuilder(); boolean done, failed; String info; long t0 = System.currentTimeMillis(), tThinkEnd; TextView thinkView, answerView; }

    // ---------- helpers ----------
    int dp(int v) { return (int) (v * getResources().getDisplayMetrics().density); }
    TextView tv(String s, int size, int color) { TextView t = new TextView(this); t.setText(s); t.setTextSize(size); t.setTextColor(color); t.setTypeface(T.body); t.setPadding(0, dp(4), 0, dp(4)); return t; }
    Button btn(String s, View.OnClickListener l) { Button b = K.button(s, UiKit.Kind.SECONDARY, l); LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-2, -2); p.setMargins(0, dp(4), dp(6), dp(4)); b.setLayoutParams(p); return b; }
    EditText edit(String hint) { EditText e = K.field(hint, false); LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-1, -2); p.setMargins(0, dp(4), 0, dp(4)); e.setLayoutParams(p); return e; }
    LinearLayout col() { LinearLayout l = new LinearLayout(this); l.setOrientation(LinearLayout.VERTICAL); l.setPadding(dp(14), dp(10), dp(14), dp(24)); return l; }
    ScrollView scroll(View v) { ScrollView s = new ScrollView(this); s.addView(v); return s; }
    void run(Runnable r) { ex.execute(r); }
    void onUi(Runnable r) { ui.post(r); }
    void setBusy(boolean b) { busy = b; onUi(() -> { if (b) getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON); else getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON); refresh("home"); }); }
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
    void initColors() { BG = T.bg; FG = T.ink; DIM = T.ink3; ACC = T.primary; GOOD = T.good; WARN = T.warm; PANEL = T.surface; }

    // ---------- lifecycle ----------
    @Override protected void onCreate(Bundle b) {
        T = new UiTheme(this); setTheme(T.dark ? android.R.style.Theme_Material_NoActionBar : android.R.style.Theme_Material_Light_NoActionBar);
        super.onCreate(b);
        K = new UiKit(this, T); initColors(); prefs = getSharedPreferences("meridian_ui", MODE_PRIVATE);
        if (getActionBar() != null) getActionBar().hide();
        rec = new Recorder(this);
        try { tools = new Tools(this); } catch (JSONException e) { throw new RuntimeException(e); }
        File pf = new File(getFilesDir(), "profile.json");
        if (pf.exists()) try { profile = new JSONObject(Native.readFile(pf.getAbsolutePath())); } catch (Exception ignored) { }
        boolean onboarded = profile != null || prefs.getBoolean("onboarded", false);
        if (onboarded) requestStartupPermissions();   // a first-run user sees the welcome first; permissions are asked when they start
        buildDeveloperViews();
        applyWindowColors();
        root = new FrameLayout(this); root.setBackgroundColor(T.bg); screenHost = new FrameLayout(this); root.addView(screenHost, new FrameLayout.LayoutParams(-1, -1));
        setContentView(root);
        screen = onboarded ? "home" : "welcome"; render();
        if (profile != null) try { deviceReport.setText(colorize(Report.render(profile))); } catch (JSONException ignored) { }
        refreshModels(); refreshRecommendations();
        autoRun(getIntent());
    }
    void requestStartupPermissions() {
        List<String> ask = new ArrayList<>();   // asked once so the agent's contact lookups and notifications work
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission("android.permission.POST_NOTIFICATIONS") != android.content.pm.PackageManager.PERMISSION_GRANTED) ask.add("android.permission.POST_NOTIFICATIONS");
        for (String perm : new String[]{"android.permission.READ_CONTACTS", "android.permission.READ_CALL_LOG", "android.permission.ACCESS_COARSE_LOCATION"})
            if (checkSelfPermission(perm) != android.content.pm.PackageManager.PERMISSION_GRANTED) ask.add(perm);
        if (!ask.isEmpty()) requestPermissions(ask.toArray(new String[0]), 1);
    }
    @SuppressWarnings("deprecation")
    void applyWindowColors() {
        Window w = getWindow(); w.setStatusBarColor(T.bg); w.setNavigationBarColor(T.bg);
        int f = 0; if (!T.dark) f = View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR | View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR;
        w.getDecorView().setSystemUiVisibility(f);
    }
    @Override public void onConfigurationChanged(Configuration c) {
        super.onConfigurationChanged(c);
        T = new UiTheme(this); K = new UiKit(this, T); initColors(); applyWindowColors(); root.setBackgroundColor(T.bg);
        String keepDev = devTab; buildDeveloperViewsKeepingText(); devTab = keepDev;
        closeDrawer(); render();
        Consent pc = pendingConsent; if (pc != null && pc.dialog != null) { try { pc.dialog.dismiss(); } catch (Exception ignored) { } pc.dialog = null; reshowConsent(); }   // re-themed sheet
    }
    // Debug-build automation (ignored unless the APK is debuggable): am start -n com.meridian.app/.MainActivity --es auto load|agent|chat [--es task "..."]
    @Override protected void onNewIntent(Intent i) { super.onNewIntent(i); setIntent(i); autoRun(i); reshowConsent(); }
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
    @Override protected void onResume() { super.onResume(); Regime.appForeground = true; if (screen.equals("settings")) refresh("settings"); reshowConsent(); }
    @Override protected void onPause() { super.onPause(); Regime.appForeground = false; }
    @Override protected void onDestroy() { super.onDestroy(); if (chat != null) chat.close(); }
    /** Legacy tab names (debug automation) map onto the new screens. */
    void show(String name) {
        if (name.equals("Device")) { devTab = "Device"; go(prefs.getBoolean("dev", false) ? "developer" : "phone"); }
        else if (name.equals("Models")) go("models");
        else if (name.equals("Chat")) go("chat");
        else if (name.equals("Agent")) go("task");
        else if (name.equals("Lab") || name.equals("Audit")) { devTab = name; if (name.equals("Audit")) refreshAudit(); go("developer"); }
        else go(name);
    }
    @Override public void onBackPressed() {
        if (drawer != null) { closeDrawer(); return; }
        if (!backStack.isEmpty()) { screen = backStack.pop(); render(); return; }
        if (!screen.equals("home") && !screen.equals("welcome")) { screen = "home"; render(); return; }
        super.onBackPressed();
    }

    // =====================================================================================================================
    // Screen frame: header + scrolling body + optional footer, width-capped on large screens. Live updates rebuild only the
    // body's children (scroll position is kept), never the whole screen, so typing and scrolling are not interrupted.
    // =====================================================================================================================
    void go(String name) { go(name, true); }
    void go(String name, boolean push) {
        if (name.equals(screen)) { render(); return; }
        if (push && !screen.equals("welcome") && !screen.equals("setup") || push && name.equals("setup")) backStack.push(screen);
        if (name.equals("home")) backStack.clear();
        screen = name; render();
    }
    void render() {
        closeDrawer(); screenHost.removeAllViews(); body = null; bodyFill = null; bodyScroll = null;
        View v;
        switch (screen) {
            case "welcome": v = buildWelcome(); break;
            case "setup": v = buildSetup(); break;
            case "task": v = buildTask(); break;
            case "chat": v = buildChat(); break;
            case "phone": v = buildPhone(); break;
            case "models": v = buildModels(); break;
            case "settings": v = buildSettings(); break;
            case "developer": v = buildDeveloper(); break;
            default: screen = "home"; v = buildHome();
        }
        screenHost.addView(v, new FrameLayout.LayoutParams(-1, -1));
        if (!screen.equals(lastRendered) && UiKit.motion()) { v.setAlpha(0f); v.setTranslationY(dp(10)); v.animate().alpha(1f).translationY(0).setDuration(180).setInterpolator(new android.view.animation.DecelerateInterpolator()).start(); }
        lastRendered = screen; ui.removeCallbacks(ticker); ui.postDelayed(ticker, 1000);
    }
    String lastRendered = "";
    View frame(View header, Runnable fill, View footer) {
        LinearLayout f = K.col();
        if (header != null) f.addView(header, new LinearLayout.LayoutParams(-1, -2));
        body = K.col(); body.setPadding(dp(16), dp(4), dp(16), dp(24)); bodyFill = fill;
        bodyScroll = K.scroll(body); f.addView(bodyScroll, new LinearLayout.LayoutParams(-1, 0, 1f));
        if (footer != null) f.addView(footer, new LinearLayout.LayoutParams(-1, -2));
        if (fill != null) fill.run();
        return K.centered(f);
    }
    /** Rebuild the body of `names` if one of them is on screen; coalesced so bursts of agent log lines cost one layout pass. */
    void refresh(String... names) {
        for (String n : names) pendingRefresh.add(n);
        if (refreshQueued) return; refreshQueued = true;
        ui.postDelayed(() -> { refreshQueued = false; boolean hit = pendingRefresh.contains(screen); pendingRefresh.clear();
            if (hit && body != null && bodyFill != null) { final ScrollView sv = bodyScroll; final int y = sv == null ? 0 : sv.getScrollY();
                final boolean atBottom = sv != null && sv.getChildCount() > 0 && y + sv.getHeight() >= sv.getChildAt(0).getHeight() - dp(24);
                body.removeAllViews(); bodyFill.run();
                if (sv != null) sv.post(() -> { if (atBottom && screen.equals("task")) sv.fullScroll(View.FOCUS_DOWN); else sv.scrollTo(0, y); }); } }, 120);
    }
    LinearLayout header(String leftIcon, String leftDesc, View.OnClickListener left, CharSequence title, String sub, boolean brand, View... right) {
        LinearLayout h = K.row(); h.setPadding(dp(4), dp(6), dp(4), dp(6)); h.setMinimumHeight(dp(60));
        if (leftIcon != null) h.addView(K.iconButton(leftIcon, leftDesc, T.ink, left));
        LinearLayout tt = K.col(); LinearLayout.LayoutParams p = K.weight(); p.leftMargin = dp(leftIcon == null ? 12 : 4);
        TextView t = K.text(title, brand ? UiTheme.Text.TITLE_L : UiTheme.Text.TITLE, T.ink); if (brand) { t.setTypeface(T.display); t.setTextSize(21); }
        t.setMaxLines(1); t.setEllipsize(TextUtils.TruncateAt.END); t.setAccessibilityHeading(true); tt.addView(t);
        headerSub = null;
        if (sub != null) { TextView s = K.text(sub, UiTheme.Text.CAPTION, T.ink3); s.setMaxLines(1); s.setEllipsize(TextUtils.TruncateAt.END); tt.addView(s); headerSub = s; }
        h.addView(tt, p);
        for (View r : right) h.addView(r);
        return h;
    }
    LinearLayout backHeader(String title, String sub, View... right) { return header("back", "Back", v -> onBackPressed(), title, sub, false, right); }
    void addGap(View v, int gapDp) { K.add(body, v, gapDp); }
    void sectionTitle(String s, int gapDp) { addGap(K.section(s), gapDp); }

    /** The problem card (brand kit "Problem as an offer"): what happened, what to do, one button, raw text behind Details. */
    View problemCard(final UiHumanize.Problem pr) {
        LinearLayout c = pr.warm ? K.tintCard(T.warmTint) : K.card();
        LinearLayout top = K.row(); top.setGravity(Gravity.TOP);
        top.addView(K.badge(pr.warm ? "therm" : "info", pr.warm ? T.surface : T.surface2, pr.warm ? T.warm : T.ink2, 40));
        LinearLayout tx = K.col(); LinearLayout.LayoutParams p = K.weight(); p.leftMargin = dp(14);
        tx.addView(K.text(pr.title, UiTheme.Text.TITLE, T.ink)); tx.addView(K.text(pr.body, UiTheme.Text.BODY, T.ink2), K.lpGap(4)); top.addView(tx, p);
        c.addView(top);
        LinearLayout acts = K.row(); acts.setPadding(0, dp(12), 0, 0);
        if (pr.action != UiHumanize.Action.NONE && pr.actionLabel != null) acts.addView(K.button(pr.actionLabel, UiKit.Kind.PRIMARY, v -> doAction(pr.action)));
        if (pr.detail != null && !pr.detail.isEmpty()) { final TextView det = K.text(pr.detail, UiTheme.Text.MONO, T.ink2); det.setTextIsSelectable(true); det.setVisibility(View.GONE);
            det.setBackground(T.rounded(T.surface2, 12)); det.setPadding(dp(12), dp(10), dp(12), dp(10));
            final Button d = K.button("Details", UiKit.Kind.QUIET, null); d.setOnClickListener(v -> { boolean show = det.getVisibility() != View.VISIBLE; det.setVisibility(show ? View.VISIBLE : View.GONE); d.setText(show ? "Hide details" : "Details"); });
            acts.addView(d); c.addView(acts); c.addView(det, K.lpGap(8)); }
        else if (acts.getChildCount() > 0) c.addView(acts);
        return c;
    }
    void doAction(UiHumanize.Action a) {
        switch (a) {
            case SETUP: go("setup"); break;
            case OPEN_MODELS: case CHOOSE_SMALLER: case DOWNLOAD_AGAIN: go("models"); break;
            case MEASURE: measureCompute(); go("phone"); break;
            case FREE_SPACE: try { startActivity(new Intent(Settings.ACTION_INTERNAL_STORAGE_SETTINGS)); } catch (Exception e) { toast("Open Settings > Storage"); } break;
            case OPEN_SETTINGS: go("settings"); break;
            case RETRY: if (screen.equals("task") && task != null && task.finished()) runAgent(task.request); else if (screen.equals("setup")) profileDevice(); else refresh(screen); break;
            default: if (screen.equals("task")) go("home"); break;
        }
    }

    // =====================================================================================================================
    // Welcome and first-run check
    // =====================================================================================================================
    View buildWelcome() {
        LinearLayout c = K.col(); c.setPadding(dp(24), dp(56), dp(24), dp(24));
        ImageView mark = new ImageView(this); mark.setImageDrawable(T.mark(this)); mark.setContentDescription("Meridian");
        c.addView(mark, new LinearLayout.LayoutParams(dp(72), dp(72)));
        TextView h = K.heading("Hello, I'm Meridian.", UiTheme.Text.DISPLAY); K.add(c, h, 28);
        K.add(c, K.text("An assistant that gets things done on your phone: messages, reminders, notes, settings and more.", UiTheme.Text.BODY_L, T.ink2), 12);
        K.add(c, feature("lock", "Private by design", "The assistant runs on this phone. It goes online only when a task needs the web."), 32);
        K.add(c, feature("shield", "Asks before it acts", "Nothing is sent, called or changed without your OK."), 20);
        K.add(c, feature("gauge", "Made for your phone", "A one-time check measures what this phone can run."), 20);
        c.addView(K.fill());
        Button go = K.button("Get started", UiKit.Kind.PRIMARY, v -> { prefs.edit().putBoolean("onboarded", true).apply(); requestStartupPermissions(); go("setup"); });
        go.setMinHeight(dp(56)); K.add(c, go, 32);
        Button skip = K.button("Skip for now", UiKit.Kind.QUIET, v -> { prefs.edit().putBoolean("onboarded", true).apply(); requestStartupPermissions(); go("home"); });
        K.add(c, skip, 4);
        ScrollView s = K.scroll(c); return K.centered(s);
    }
    View feature(String icon, String title, String sub) {
        LinearLayout r = K.row(); r.setGravity(Gravity.TOP); r.addView(K.badge(icon, T.sunTint, T.sunInk, 40));
        LinearLayout tx = K.col(); LinearLayout.LayoutParams p = K.weight(); p.leftMargin = dp(14);
        tx.addView(K.text(title, UiTheme.Text.TITLE, T.ink)); tx.addView(K.text(sub, UiTheme.Text.BODY, T.ink2), K.lpGap(2)); r.addView(tx, p);
        return r;
    }
    View buildSetup() {
        return frame(backHeader("Getting to know your phone", null), () -> {
            if (setupRunning) {
                LinearLayout c = K.card(); LinearLayout r = K.row(); r.addView(K.pulse());
                TextView tt = K.text("Checking your phone…", UiTheme.Text.TITLE, T.ink); LinearLayout.LayoutParams p = K.weight(); p.leftMargin = dp(12); r.addView(tt, p); c.addView(r);
                workClock = K.text(clock((System.currentTimeMillis() - workStarted) / 1000), UiTheme.Text.CAPTION, T.ink3); r.addView(workClock);
                stageView = K.text(setupStage == null ? "Starting" : setupStage, UiTheme.Text.BODY, T.ink2); stageView.setPadding(dp(34), 0, 0, 0); K.add(c, stageView, 8);
                c.setAccessibilityLiveRegion(View.ACCESSIBILITY_LIVE_REGION_POLITE);
                addGap(K.heading("This takes a few minutes.", UiTheme.Text.HEADLINE), 8);
                addGap(K.text("Keep Meridian open and put the phone down. If you can, leave it unplugged: I measure the phone the way you'll use it.", UiTheme.Text.BODY_L, T.ink2), 8);
                addGap(c, 20);
                c.setContentDescription("Checking your phone. " + (setupStage == null ? "" : setupStage));
                return;
            }
            if (profile != null) {
                addGap(K.heading("All set.", UiTheme.Text.DISPLAY), 8);
                addGap(K.text("Here's what fits this phone. Download one model and Meridian works offline from then on.", UiTheme.Text.BODY_L, T.ink2), 8);
                addModelVerdicts(3, true);
                Button done = K.button("Continue", UiKit.Kind.PRIMARY, v -> go("home")); addGap(done, 24);
                return;
            }
            addGap(K.heading("A quick check, once.", UiTheme.Text.DISPLAY), 8);
            addGap(K.text("I'll measure this phone's memory, storage and processor so I only suggest models that run well here, and tell you honestly how fast they'll be.", UiTheme.Text.BODY_L, T.ink2), 12);
            if (setupProblem != null) addGap(problemCard(setupProblem), 16);
            LinearLayout box = K.card(); box.setPadding(0, dp(4), 0, dp(4));
            box.addView(K.switchRow("Include a memory test", "More accurate. It may close apps running in the background.", memGrant.isChecked(), v -> { memGrant.setChecked(!memGrant.isChecked()); refresh("setup"); }));
            addGap(box, 20);
            Button start = K.button("Start the check", UiKit.Kind.PRIMARY, v -> profileDevice()); start.setMinHeight(dp(56)); addGap(start, 20);
            addGap(K.text("Keep Meridian open while it runs. It takes a few minutes.", UiTheme.Text.CAPTION, T.ink3), 10);
        }, null);
    }

    // =====================================================================================================================
    // Home: the task hub
    // =====================================================================================================================
    View buildHome() {
        String sub = warming ? "Getting ready…" : chat != null && selectedModel != null ? "Ready · " + UiHumanize.modelName(selectedModel.getName()) : busy && task != null && task.status == UiTask.Status.STARTING ? "Getting ready…" : null;
        View right = K.iconButton("phone", "This phone", T.ink, v -> go("phone"));
        LinearLayout h = header("menu", "Open menu", v -> openDrawer(), "Meridian", sub, true, right);
        final UiKit.Composer comp = K.composer("Give Meridian a task", true, null, null);
        comp.input.setText(homeDraft); comp.input.addTextChangedListener(draftWatcher(0));
        comp.send.setOnClickListener(v -> submitTask(comp.input));
        comp.mic.setOnClickListener(v -> startVoice(comp.input));
        comp.input.setOnEditorActionListener((v, id, e) -> { if (id == EditorInfo.IME_ACTION_SEND) { submitTask(comp.input); return true; } return false; });
        LinearLayout foot = K.col(); foot.setPadding(dp(12), dp(6), dp(12), dp(12)); foot.addView(comp.view);
        TextView note = K.text("Runs on this phone. Online only when a task needs the web.", UiTheme.Text.CAPTION, T.ink3); note.setGravity(Gravity.CENTER);
        K.add(foot, note, 8);
        return frame(h, this::fillHome, foot);
    }
    void submitTask(EditText in) {
        String q = in.getText().toString().trim(); if (q.isEmpty()) return;
        if (busy) { toast("I'm still working on the current task."); return; }
        in.setText(""); homeDraft = ""; taskDraft = ""; hideKeyboard(in); runAgent(q);
    }
    TextWatcher draftWatcher(final int which) {
        return new TextWatcher() { public void beforeTextChanged(CharSequence s, int a, int b, int c) { } public void onTextChanged(CharSequence s, int a, int b, int c) { }
            public void afterTextChanged(Editable e) { if (which == 0) homeDraft = e.toString(); else if (which == 1) taskDraft = e.toString(); else chatDraft = e.toString(); } };
    }
    void fillHome() {
        boolean running = task != null && !task.finished();
        addGap(K.heading(running ? "I'm on it." : UiHumanize.greeting() + ".\nWhat should I take care of?", UiTheme.Text.HEADLINE), 12);
        // Needs you: the one thing standing between the user and a working assistant
        if (profile == null && !setupRunning) addGap(needsYou("gauge", "Let's check your phone first", "One quick check, so I only suggest models that run well here.", "Check my phone", v -> go("setup")), 20);
        else if (setupRunning) addGap(needsYou("gauge", "Checking your phone…", setupStage == null ? "This takes a few minutes. Keep Meridian open." : setupStage, "See progress", v -> go("setup")), 20);
        else if (allModels().isEmpty() && activeDl == null) { JSONObject best = bestRecommendation();
            String bodyTxt = best == null ? "Download one that fits this phone. After that it works offline." : "Recommended for this phone: " + best.optString("name") + " (" + UiHumanize.gb(best.optLong("size_bytes")) + "). After that it works offline.";
            addGap(needsYou("download", "Choose an assistant model", bodyTxt, "See models that fit", v -> go("models")), 20); }
        else if (activeDl != null) addGap(needsYou("download", "Downloading " + UiHumanize.modelName(dlName), dlTotal > 0 ? UiHumanize.gb(dlDone) + " of " + UiHumanize.gb(dlTotal) : "Starting", "See progress", v -> go("models")), 20);
        if (running) { sectionTitle("Running", 24); addGap(runningCard(), 8); }
        List<JSONObject> hist = history(8);
        if (task != null && task.finished() && (hist.isEmpty() || !hist.get(0).optString("request").equals(task.request))) { sectionTitle("Just now", 24); addGap(taskRow(task.request, task.answer != null ? task.answer : task.now, "now", v -> go("task")), 8); }
        if (hist.isEmpty() && !running) {
            sectionTitle("Try asking", 24);
            String[][] s = {{"bolt", "What's my battery level?"}, {"auto", "Set a timer for 10 minutes"}, {"file", "Save a note: buy milk"}, {"phone", "Who did I miss calls from?"}};
            for (final String[] x : s) { LinearLayout r = K.pressCard(v -> { homeDraft = x[1]; render(); });
                LinearLayout row = K.row(); row.addView(K.iconView(x[0], T.ink2, 20)); TextView t = K.text(x[1], UiTheme.Text.BODY_L, T.ink); LinearLayout.LayoutParams p = K.weight(); p.leftMargin = dp(12); row.addView(t, p);
                r.addView(row); r.setContentDescription("Suggestion: " + x[1]); addGap(r, 8); }
        } else if (!hist.isEmpty()) {
            sectionTitle("Recent", 24);
            for (final JSONObject j : hist) addGap(taskRow(j.optString("request"), j.optString("answer"), j.optString("time"), v -> openHistory(j)), 8);
        }
    }
    View needsYou(String icon, String title, String sub, String action, View.OnClickListener l) {
        LinearLayout c = K.tintCard(T.sunTint);
        LinearLayout r = K.row(); r.setGravity(Gravity.TOP); r.addView(K.iconView(icon, T.sunInk, 22));
        LinearLayout tx = K.col(); LinearLayout.LayoutParams p = K.weight(); p.leftMargin = dp(12);
        tx.addView(K.text(title, UiTheme.Text.TITLE, T.ink)); tx.addView(K.text(sub, UiTheme.Text.BODY, T.ink2), K.lpGap(2)); r.addView(tx, p); c.addView(r);
        LinearLayout a = K.row(); a.setPadding(dp(34), dp(10), 0, 0); a.addView(K.button(action, UiKit.Kind.PRIMARY, l)); c.addView(a);
        return c;
    }
    View runningCard() {
        LinearLayout c = K.pressCard(v -> go("task"));
        LinearLayout r = K.row(); r.addView(K.pulse()); TextView t = K.text(task.request, UiTheme.Text.TITLE, T.ink); t.setMaxLines(2); t.setEllipsize(TextUtils.TruncateAt.END);
        LinearLayout.LayoutParams p = K.weight(); p.leftMargin = dp(12); r.addView(t, p); r.addView(K.iconView("chevron", T.ink3, 20)); c.addView(r);
        LinearLayout nr = K.row(); nr.setPadding(dp(34), 0, 0, 0); TextView now = K.text(task.now, UiTheme.Text.BODY, T.ink2); nr.addView(now, K.weight());
        runningElapsed = K.text(clock(task.elapsedS()), UiTheme.Text.CAPTION, T.ink3); nr.addView(runningElapsed); K.add(c, nr, 6);
        if (!task.steps.isEmpty()) { LinearLayout br = K.row(); br.setPadding(dp(34), dp(10), 0, 0); UiKit.Bar b = K.bar(T.primary); b.glide("home" + task.started, task.doneCount() / (float) task.steps.size());
            br.addView(b, new LinearLayout.LayoutParams(0, dp(4), 1f)); TextView n = K.text("Step " + Math.min(task.steps.size(), task.doneCount() + 1) + " of " + task.steps.size(), UiTheme.Text.CAPTION, T.ink3);
            LinearLayout.LayoutParams np = new LinearLayout.LayoutParams(-2, -2); np.leftMargin = dp(10); br.addView(n, np); c.addView(br); }
        c.setContentDescription("Running task: " + task.request + ". " + task.now);
        return c;
    }
    View taskRow(String request, String answer, String when, View.OnClickListener l) {
        LinearLayout c = K.pressCard(l);
        TextView t = K.text(request, UiTheme.Text.TITLE, T.ink); t.setMaxLines(2); t.setEllipsize(TextUtils.TruncateAt.END); c.addView(t);
        if (answer != null && !answer.isEmpty()) { TextView a = K.text(answer.replace('\n', ' '), UiTheme.Text.BODY, T.ink2); a.setMaxLines(2); a.setEllipsize(TextUtils.TruncateAt.END); K.add(c, a, 4); }
        if (when != null && !when.isEmpty()) K.add(c, K.text(when, UiTheme.Text.CAPTION, T.ink3), 6);
        return c;
    }
    /** Task memory written by the agent (files/memory.jsonl), newest first. */
    List<JSONObject> history(int max) {
        List<JSONObject> out = new ArrayList<>(); File f = new File(getFilesDir(), "memory.jsonl"); if (!f.exists()) return out;
        List<String> lines = new ArrayList<>();
        try (BufferedReader r = new BufferedReader(new FileReader(f))) { String l; while ((l = r.readLine()) != null) if (!l.trim().isEmpty()) lines.add(l); } catch (IOException ignored) { }
        for (int i = lines.size() - 1; i >= 0 && out.size() < max; i--) try { out.add(new JSONObject(lines.get(i))); } catch (JSONException ignored) { }
        return out;
    }
    void openHistory(JSONObject j) {
        UiTask t = new UiTask(j.optString("request"));
        JSONArray pl = j.optJSONArray("plan"); if (pl != null) for (int i = 0; i < pl.length(); i++) { UiTask.Step s = new UiTask.Step(); s.title = UiTask.sentence(pl.optString(i)); s.state = UiKit.StepState.DONE; t.steps.add(s); }
        StringBuilder acts = new StringBuilder(); JSONArray a = j.optJSONArray("actions");
        if (a != null) for (int i = 0; i < a.length(); i++) { JSONObject x = a.optJSONObject(i); if (x == null) continue; acts.append(UiTask.done(x.optString("tool"))).append(x.optBoolean("verified") ? "" : " (couldn't confirm it worked)").append('\n'); }
        t.answer = j.optString("answer"); t.actionsTaken = acts.length() == 0 ? null : acts.toString().trim(); t.status = UiTask.Status.DONE; t.now = "Done · " + j.optString("time");
        viewing = t; go("task");
    }
    UiTask viewing;   // a past task opened from history (read-only); null = the live task

    // =====================================================================================================================
    // Live task
    // =====================================================================================================================
    View buildTask() {
        final UiTask tk = viewing != null ? viewing : task;
        if (tk == null) return frame(backHeader("Task", null), () -> addGap(K.text("No task yet. Give Meridian one from the home screen.", UiTheme.Text.BODY_L, T.ink2), 16), null);
        View foot; taskFooterRunning = !tk.finished() && tk == task;
        if (taskFooterRunning) {
            LinearLayout f = K.col(); f.setPadding(dp(12), dp(8), dp(12), dp(12)); f.setBackgroundColor(T.bg);
            Button stop = K.button("Stop", "stop", UiKit.Kind.SECONDARY, v -> stopTask()); stop.setMinHeight(dp(52)); f.addView(stop, new LinearLayout.LayoutParams(-1, -2));
            foot = f;
        } else {
            final UiKit.Composer comp = K.composer("Give Meridian another task", true, null, null);
            comp.input.setText(taskDraft); comp.input.addTextChangedListener(draftWatcher(1));
            comp.send.setOnClickListener(v -> { viewing = null; submitTask(comp.input); }); comp.mic.setOnClickListener(v -> startVoice(comp.input));
            comp.input.setOnEditorActionListener((v, id, e) -> { if (id == EditorInfo.IME_ACTION_SEND) { viewing = null; submitTask(comp.input); return true; } return false; });
            LinearLayout f = K.col(); f.setPadding(dp(12), dp(6), dp(12), dp(12)); f.addView(comp.view); foot = f;
        }
        LinearLayout h = header("back", "Back", v -> { viewing = null; onBackPressed(); }, "Task", statusLine(tk), false);
        return frame(h, () -> fillTask(tk), foot);
    }
    String statusLine(UiTask tk) {
        if (tk.stopRequested && !tk.finished()) return "Stopping…";
        switch (tk.status) { case STARTING: return "Getting ready · " + tk.elapsedS() + " s"; case RUNNING: return "Working · " + tk.elapsedS() + " s"; case NEEDS_YOU: return "Waiting for you";
            case DONE: return tk == task ? "Done in " + tk.elapsedS() + " s" : tk.now; case STOPPED: return "Stopped"; default: return "Didn't finish"; }
    }
    void tick() {
        ui.removeCallbacks(ticker);
        if ((measuring || setupRunning) && (screen.equals("phone") || screen.equals("setup"))) {
            if (workClock != null) workClock.setText(clock((System.currentTimeMillis() - workStarted) / 1000)); ui.postDelayed(ticker, 1000); return; }
        UiTask tk = viewing != null ? viewing : task;
        if (tk == null || tk.finished()) { if (headerSub != null && tk != null && screen.equals("task")) headerSub.setText(statusLine(tk)); return; }
        if (screen.equals("task")) { if (headerSub != null) headerSub.setText(statusLine(tk)); if (workingElapsed != null) workingElapsed.setText(clock(tk.elapsedS())); }
        else if (screen.equals("home") && runningElapsed != null) runningElapsed.setText(clock(tk.elapsedS()));
        else return;
        ui.postDelayed(ticker, 1000);
    }
    static String clock(long s) { return s < 60 ? s + " s" : String.format(Locale.ROOT, "%d:%02d", s / 60, s % 60); }
    void fillTask(UiTask tk) {
        if (headerSub != null) headerSub.setText(statusLine(tk));
        ui.removeCallbacks(ticker); if (!tk.finished()) ui.postDelayed(ticker, 1000);
        TextView req = K.text(tk.request, UiTheme.Text.BODY_L, T.ink); req.setBackground(T.rounded(T.surface2, 20)); req.setPadding(dp(16), dp(12), dp(16), dp(12)); req.setTextIsSelectable(true);
        LinearLayout.LayoutParams rp = new LinearLayout.LayoutParams(-2, -2); rp.gravity = Gravity.END; rp.topMargin = dp(8); rp.leftMargin = dp(48); body.addView(req, rp);
        if (tk.status == UiTask.Status.STARTING) {
            LinearLayout c = K.card(); LinearLayout r = K.row(); r.addView(K.pulse()); LinearLayout.LayoutParams p = K.weight(); p.leftMargin = dp(12);
            r.addView(K.text("Getting ready", UiTheme.Text.TITLE, T.ink), p); c.addView(r);
            K.add(c, K.text(engineNote != null ? engineNote : "Waking up the assistant. The first task after opening Meridian takes the longest.", UiTheme.Text.BODY, T.ink2), 6);
            c.setContentDescription("Getting ready. " + (engineNote == null ? "" : engineNote)); addGap(c, 16);
        } else if (!tk.finished()) {
            LinearLayout c = K.tintCard(T.surface2); LinearLayout r = K.row(); r.addView(K.pulse()); LinearLayout.LayoutParams p = K.weight(); p.leftMargin = dp(12);
            r.addView(K.text(tk.status == UiTask.Status.NEEDS_YOU ? "Waiting for your OK" : "Working now", UiTheme.Text.LABEL, T.ink), p);
            workingElapsed = K.text(clock(tk.elapsedS()), UiTheme.Text.CAPTION, T.ink3); r.addView(workingElapsed);
            c.addView(r);
            K.add(c, K.text(tk.now, UiTheme.Text.BODY_L, T.ink), 8);
            liveView = K.text("", UiTheme.Text.BODY, T.ink2); liveView.setMaxLines(4); K.add(c, liveView, 6);
            String l0 = tk.live(); if (l0 == null) liveView.setVisibility(View.GONE); else liveView.setText(withCaret(l0));
            if (tk.status == UiTask.Status.NEEDS_YOU && pendingConsent != null) { LinearLayout ra = K.row(); ra.setPadding(0, dp(10), 0, 0); ra.addView(K.button("Review", UiKit.Kind.PRIMARY, v -> reshowConsent())); c.addView(ra); }
            c.setAccessibilityLiveRegion(View.ACCESSIBILITY_LIVE_REGION_POLITE); c.setContentDescription("Working now: " + tk.now); addGap(c, 16);
        }
        if (!tk.steps.isEmpty()) {
            LinearLayout c = K.card(); int n = tk.steps.size();
            c.addView(K.text("Plan · " + tk.doneCount() + " of " + n + " done", UiTheme.Text.OVERLINE, T.ink2));
            UiKit.Bar pb = K.bar(tk.status == UiTask.Status.DONE ? T.good : T.primary); K.add(c, pb, 10); pb.glide("plan" + tk.started, tk.doneCount() / (float) n);
            int i = 0, cascade = 0;
            for (UiTask.Step s : tk.steps) {
                String sub = s.sub; if (!s.did.isEmpty()) sub = (sub == null ? "" : sub + "\n") + TextUtils.join("\n", s.did);
                LinearLayout row = K.step(s.state, s.title == null ? "" : s.title, sub); K.add(c, row, 14);
                String k = tk.started + ":" + i;
                if (fresh(k + ":row")) K.appear(row, 60L * cascade++);
                if (s.state == UiKit.StepState.DONE && fresh(k + ":done")) K.pop(row.getChildAt(0));
                if (!s.did.isEmpty() && fresh(k + ":did" + s.did.size())) K.appear(row.getChildAt(1), 0);
                i++;
            }
            addGap(c, 12); if (fresh(tk.started + ":plan")) K.appear(c, 0);
        }
        if (!tk.feed.isEmpty()) {
            LinearLayout c = K.card(); int n = tk.feed.size(); boolean all = feedAll || !tk.finished(); int from = all ? 0 : Math.max(0, n - 5);
            LinearLayout hr = K.row(); hr.addView(K.text("Activity", UiTheme.Text.OVERLINE, T.ink2), K.weight());
            if (n > 5 && tk.finished()) { Button tg = K.button(feedAll ? "Show less" : "Show all " + n, UiKit.Kind.QUIET, v -> { feedAll = !feedAll; refresh("task"); }); tg.setMinHeight(dp(40)); hr.addView(tg); }
            c.addView(hr);
            for (int i = from; i < n; i++) { UiTask.Event e = tk.feed.get(i);
                LinearLayout r = K.row(); r.setGravity(Gravity.TOP);
                int dot = e.kind.equals("ok") ? T.good : e.kind.equals("warn") ? T.warm : e.kind.equals("act") ? T.sun : e.kind.equals("step") ? T.ink : T.ink3;
                View d = new View(this); d.setBackground(T.rounded(dot, 999)); LinearLayout.LayoutParams dp8 = new LinearLayout.LayoutParams(dp(8), dp(8)); dp8.topMargin = dp(8); r.addView(d, dp8);
                TextView tx = K.text(e.text, e.kind.equals("step") ? UiTheme.Text.LABEL : UiTheme.Text.BODY, e.kind.equals("step") ? T.ink : T.ink2); LinearLayout.LayoutParams tp = K.weight(); tp.leftMargin = dp(12); r.addView(tx, tp);
                TextView tm = K.text(clock(e.t / 1000), UiTheme.Text.CAPTION, T.ink3); LinearLayout.LayoutParams mp = new LinearLayout.LayoutParams(-2, -2); mp.leftMargin = dp(8); mp.topMargin = dp(2); r.addView(tm, mp);
                r.setContentDescription(e.text + ", at " + clock(e.t / 1000)); K.add(c, r, 10);
                if (fresh(tk.started + ":ev" + i)) K.appear(r, 0);
            }
            addGap(c, 12);
        }
        if (tk.status == UiTask.Status.DONE && tk.answer != null) {
            LinearLayout c = K.card();
            TextView a = K.text(tk.answer, UiTheme.Text.BODY_L, T.ink); a.setTextIsSelectable(true); c.addView(a);
            if (tk.actionsTaken != null) { K.add(c, K.text("What I did", UiTheme.Text.OVERLINE, T.ink2), 16);
                TextView at = K.text(tk.actionsTaken, UiTheme.Text.BODY, T.ink); at.setTextIsSelectable(true); at.setBackground(T.rounded(T.surface2, 12)); at.setPadding(dp(12), dp(10), dp(12), dp(10)); K.add(c, at, 6); }
            LinearLayout acts = K.row(); acts.setPadding(0, dp(8), 0, 0);
            final String all = tk.answer + (tk.actionsTaken == null ? "" : "\n\n" + tk.actionsTaken);
            acts.addView(K.button("Copy", "copy", UiKit.Kind.QUIET, v -> { ((android.content.ClipboardManager) getSystemService(CLIPBOARD_SERVICE)).setPrimaryClip(ClipData.newPlainText("Meridian", all)); toast("Copied"); }));
            if (tk == task) acts.addView(K.button("Run again", "history", UiKit.Kind.QUIET, v -> runAgent(tk.request)));
            c.addView(acts); addGap(c, 12); if (fresh(tk.started + ":result")) K.appear(c, 80);
        }
        if (tk.status == UiTask.Status.PROBLEM && tk.problem != null) { View pc = problemCard(tk.problem); addGap(pc, 12); if (fresh(tk.started + ":problem")) K.appear(pc, 80); }
        if (tk.status == UiTask.Status.STOPPED) { View sp = K.pill("stop", "Stopped. Nothing else will happen for this task.", T.surface2, T.ink2); addGap(sp, 12); if (fresh(tk.started + ":stopped")) K.appear(sp, 0); }
    }
    /** Stop means stop: deny any pending approval and cancel the agent, which checks before every model call and every tool
     *  (Agent.cancel). Cancelling only the current generation was not enough: on the Nord (2026-09-22) a tool ran after Stop. */
    volatile Agent runningAgent; volatile boolean warming;
    void stopTask() {
        if (task == null || task.finished()) return;
        task.stopRequested = true; task.now = "Stopping…";
        answerConsent(pendingConsent, false);
        Agent a = runningAgent; if (a != null) a.cancel();   // checked before every model call and tool run (harness, 2026-09-22)
        refresh("task", "home");
    }
    TextView liveView; boolean liveQueued; boolean feedAll;
    /** The live line under "Working now": what the model is writing this instant (the plan forming, the action it is choosing,
     *  the answer typing out). Updated in place ~12 times a second at most; never rebuilds the screen. */
    void scheduleLive() {
        if (liveQueued) return; liveQueued = true;
        ui.postDelayed(() -> { liveQueued = false; UiTask tk = task; if (tk == null || tk.finished() || liveView == null || !liveView.isAttachedToWindow()) return;
            String l = tk.live(); if (l == null) { liveView.setVisibility(View.GONE); return; }
            liveView.setVisibility(View.VISIBLE); liveView.setText(withCaret(l)); }, 80);
    }
    boolean taskFooterRunning;
    /** The body refreshes in place; the footer (Stop vs. the next-task composer) swaps only when the task starts or ends. */
    void taskChanged() {
        if (screen.equals("task") && viewing == null && task != null && task.finished() == taskFooterRunning) { ui.post(this::render); refresh("home"); }
        else refresh("task", "home");
    }

    /** A consent request the agent thread is waiting on. It lives in the activity, not in the dialog: a cancel is never an answer
     *  (15R rehearsal, 2026-09-22: a sheet shown while Meridian was in the background was cancelled on return and counted as a
     *  denial). Only "Allow once", "Don't allow" or Stop answer it; the sheet is re-shown on resume, new intent and config change. */
    static final class Consent { final UiTask tk; final String tool, text; final CountDownLatch cd = new CountDownLatch(1); volatile boolean ok, answered; Dialog dialog;
        Consent(UiTask tk, String tool, String text) { this.tk = tk; this.tool = tool; this.text = text; } }
    volatile Consent pendingConsent;
    boolean askConsent(final UiTask tk, final String tool, final String text) {
        if (tk != null && tk.stopRequested) return false;
        final Consent c = new Consent(tk, tool, text); pendingConsent = c;
        onUi(() -> { if (tk != null && !tk.finished()) { tk.status = UiTask.Status.NEEDS_YOU; tk.now = "Waiting for your OK"; taskChanged(); } showConsent(c); });
        try { c.cd.await(); } catch (InterruptedException e) { return false; } finally { if (pendingConsent == c) pendingConsent = null; }
        return c.ok;
    }
    void answerConsent(Consent c, boolean ok) {
        if (c == null || c.answered) return; c.answered = true; c.ok = ok;
        if (c.dialog != null) { try { c.dialog.dismiss(); } catch (Exception ignored) { } c.dialog = null; }
        if (pendingConsent == c) pendingConsent = null;
        if (c.tk != null && !c.tk.finished()) { c.tk.status = UiTask.Status.RUNNING; c.tk.now = ok ? "Doing it now" : "Skipping that"; taskChanged(); }
        c.cd.countDown();
    }
    /** Show (or re-show) the sheet for a pending consent; idempotent. */
    void showConsent(final Consent c) {
        if (c == null || c.answered || isFinishing() || isDestroyed()) return;
        if (c.dialog != null && c.dialog.isShowing()) return;
        LinearLayout box = K.col();
        box.addView(K.heading("Allow this?", UiTheme.Text.HEADLINE));
        K.add(box, K.text(UiTask.verbing(c.tool), UiTheme.Text.CAPTION, T.ink2), 4);
        TextView what = K.text(c.text, UiTheme.Text.BODY_L, T.ink); what.setBackground(T.rounded(T.surface2, 16, 1, T.line)); what.setPadding(dp(16), dp(14), dp(16), dp(14)); what.setTextIsSelectable(true);
        K.add(box, what, 16);
        K.add(box, K.text("Meridian will do exactly this and nothing else.", UiTheme.Text.BODY, T.ink2), 12);
        Button allow = K.button("Allow once", UiKit.Kind.PRIMARY, v -> answerConsent(c, true)); allow.setMinHeight(dp(56));
        Button deny = K.button("Don't allow", UiKit.Kind.SECONDARY, v -> answerConsent(c, false)); deny.setMinHeight(dp(52));
        K.add(box, allow, 20); K.add(box, deny, 8);
        Dialog d = K.sheet(box, false); d.setCancelable(false); d.setCanceledOnTouchOutside(false);
        c.dialog = d;
        try { d.show(); } catch (Exception e) { c.dialog = null; }   // no window yet (e.g. mid-restart): onResume shows it
    }
    void reshowConsent() { Consent c = pendingConsent; if (c != null) { if (c.dialog != null && !c.dialog.isShowing()) c.dialog = null; showConsent(c); } }

    // =====================================================================================================================
    // Chat (no phone actions), with the model's thinking when it produces some
    // =====================================================================================================================
    View buildChat() {
        final UiKit.Composer comp = K.composer("Ask anything", true, null, null);
        comp.input.setText(chatDraft); comp.input.addTextChangedListener(draftWatcher(2));
        View.OnClickListener sendL = v -> { String q = comp.input.getText().toString().trim(); if (q.isEmpty()) return; if (busy) { toast("I'm still busy with the last one."); return; } comp.input.setText(""); chatDraft = ""; hideKeyboard(comp.input); send(q); };
        comp.send.setOnClickListener(sendL); comp.mic.setOnClickListener(v -> startVoice(comp.input));
        LinearLayout foot = K.col(); foot.setPadding(dp(12), dp(6), dp(12), dp(12));
        LinearLayout opts = K.row(); opts.setPadding(dp(4), 0, 0, dp(6));
        final Button think = K.button(thinkFirst ? "Thinking first: on" : "Thinking first: off", "deep", thinkFirst ? UiKit.Kind.SUN : UiKit.Kind.SECONDARY, null);
        think.setOnClickListener(v -> { thinkFirst = !thinkFirst; render(); }); think.setContentDescription("Think before answering, " + (thinkFirst ? "on" : "off") + ". Works with models that support it.");
        opts.addView(think); foot.addView(opts); foot.addView(comp.view);
        View newChat = K.iconButton("plus", "New chat", T.ink, v -> { chatMsgs.clear(); chatTurns = 0; chatOut.setText(""); render(); });
        return frame(backHeader("Chat", "Answers only. No phone actions.", newChat), this::fillChat, foot);
    }
    void fillChat() {
        if (chatMsgs.isEmpty()) { addGap(K.heading("Ask me anything.", UiTheme.Text.HEADLINE), 16);
            addGap(K.text("For things on your phone, like messages, reminders and settings, give me a task from the home screen instead.", UiTheme.Text.BODY_L, T.ink2), 8); return; }
        for (ChatMsg m : chatMsgs) {
            TextView q = K.text(m.user, UiTheme.Text.BODY_L, T.ink); q.setBackground(T.rounded(T.surface2, 20)); q.setPadding(dp(16), dp(12), dp(16), dp(12)); q.setTextIsSelectable(true);
            LinearLayout.LayoutParams rp = new LinearLayout.LayoutParams(-2, -2); rp.gravity = Gravity.END; rp.topMargin = dp(16); rp.leftMargin = dp(48); body.addView(q, rp);
            if (fresh("chat:" + m.t0 + ":" + m.user.hashCode())) K.appear(q, 0);
            String[] parts = splitThinking(m.raw.toString()); int ph = phase(m); m.renderedPhase = ph; m.thinkView = null;
            if (ph == 0 || !parts[0].isEmpty()) {   // waiting for the first token, or the model's thinking (live while it thinks, folded after)
                final LinearLayout box = K.tintCard(T.surface2); box.setPadding(dp(14), dp(10), dp(14), dp(10));
                LinearLayout hr = K.row(); if (ph <= 1) hr.addView(K.pulse());
                long secs = Math.max(1, ((m.tThinkEnd > 0 ? m.tThinkEnd : System.currentTimeMillis()) - m.t0) / 1000);
                String label = ph == 0 ? "Working on it" : ph == 1 ? "Thinking" : "Thought for " + secs + " s";
                TextView lt = K.text(label, UiTheme.Text.LABEL, T.ink2); LinearLayout.LayoutParams lp = K.weight(); lp.leftMargin = dp(ph <= 1 ? 10 : 0); hr.addView(lt, lp);
                if (ph >= 2) hr.addView(K.iconView("down", T.ink3, 18));
                box.addView(hr);
                final TextView tv = K.text(parts[0], UiTheme.Text.BODY, T.ink2); tv.setTextIsSelectable(true); K.add(box, tv, 6);
                tv.setVisibility(ph == 1 ? View.VISIBLE : View.GONE);
                if (ph >= 2) { box.setClickable(true); box.setFocusable(true); box.setOnClickListener(v -> tv.setVisibility(tv.getVisibility() == View.VISIBLE ? View.GONE : View.VISIBLE));
                    box.setContentDescription(label + ". Double tap to show or hide the thinking."); }
                if (ph == 1) m.thinkView = tv; addGap(box, 10);
            }
            TextView a = K.text(ph == 2 ? caret(parts[1]) : parts[1], UiTheme.Text.BODY_L, T.ink); a.setTextIsSelectable(ph == 3); m.answerView = a; if (!parts[1].isEmpty() || ph >= 2) addGap(a, 10);
            if (m.failed && m.info != null) addGap(problemCard(UiHumanize.of(m.info)), 8);
            else if (m.done && m.info != null) addGap(K.text(m.info, UiTheme.Text.CAPTION, T.ink3), 6);
        }
    }
    /** The streaming answer ends in a marigold caret, so text visibly arriving reads as live. */
    CharSequence withCaret(String s) { if (s.length() > 160) s = "…" + s.substring(s.length() - 160); return caret(s); }
    CharSequence caret(String s) { SpannableString sp = new SpannableString(s + " \u258D"); sp.setSpan(new ForegroundColorSpan(T.sun), sp.length() - 1, sp.length(), 0); return sp; }
    /** 0 waiting for output, 1 thinking, 2 answering, 3 done. The body is rebuilt only when the phase changes. */
    static int phase(ChatMsg m) { if (m.done) return 3; String[] p = splitThinking(m.raw.toString()); return !p[1].isEmpty() ? 2 : !p[0].isEmpty() ? 1 : 0; }
    /** {thinking, answer} from a raw stream that may contain <think>..</think>. */
    static String[] splitThinking(String raw) {
        int a = raw.indexOf("<think>"); if (a < 0) return new String[]{"", raw.trim()};
        int b = raw.indexOf("</think>", a);
        if (b < 0) return new String[]{raw.substring(a + 7).trim(), ""};
        return new String[]{raw.substring(a + 7, b).trim(), (raw.substring(0, a) + raw.substring(b + 8)).trim()};
    }
    boolean chatFlushQueued;
    void chatToken(final ChatMsg m, String t) {
        m.raw.append(t); if (m.tThinkEnd == 0 && m.raw.indexOf("</think>") >= 0) m.tThinkEnd = System.currentTimeMillis();
        if (chatFlushQueued) return; chatFlushQueued = true;
        ui.postDelayed(() -> { chatFlushQueued = false;   // conflated: one UI update per ~60 ms, never per token (08 section 4)
            if (!screen.equals("chat")) return;
            if (phase(m) != m.renderedPhase || m.answerView == null || !m.answerView.isAttachedToWindow()) { refresh("chat"); return; }
            String[] p = splitThinking(m.raw.toString());
            if (m.renderedPhase == 1 && m.thinkView != null) m.thinkView.setText(p[0]); else m.answerView.setText(caret(p[1]));
            if (bodyScroll != null) bodyScroll.post(() -> bodyScroll.fullScroll(View.FOCUS_DOWN)); }, 60);
    }

    // =====================================================================================================================
    // This phone
    // =====================================================================================================================
    View buildPhone() {
        return frame(backHeader("This phone", deviceName()), () -> {
            if (profile == null) { addGap(needsYou("gauge", "I haven't checked this phone yet", "A one-time check measures what it can run.", "Check my phone", v -> go("setup")), 12); return; }
            addGap(K.heading(phoneVerdict(), UiTheme.Text.HEADLINE), 8);
            if (measuring || setupRunning) { LinearLayout c = K.card(); LinearLayout r = K.row(); r.addView(K.pulse()); LinearLayout.LayoutParams p = K.weight(); p.leftMargin = dp(12);
                r.addView(K.text(measuring ? "Measuring speed…" : "Checking your phone…", UiTheme.Text.TITLE, T.ink), p);
                workClock = K.text(clock((System.currentTimeMillis() - workStarted) / 1000), UiTheme.Text.CAPTION, T.ink3); r.addView(workClock); c.addView(r);
                stageView = K.text(setupStage == null ? "Starting" : setupStage, UiTheme.Text.BODY, T.ink2); stageView.setPadding(dp(34), 0, 0, 0); K.add(c, stageView, 6);
                K.add(c, K.text(measuring ? "Takes about 2 minutes. Keep Meridian open." : "Takes a few minutes. Keep Meridian open.", UiTheme.Text.CAPTION, T.ink3), 8);
                c.setAccessibilityLiveRegion(View.ACCESSIBILITY_LIVE_REGION_POLITE); addGap(c, 16); }
            if (phoneProblem != null) addGap(problemCard(phoneProblem), 16);
            else if (!measuring && !setupRunning && notMeasured()) addGap(needsYou("gauge", "Measure this phone's speed", "A 2-minute test with small sample models. After it, I can tell you how fast each model will run here.", "Measure now", v -> measureCompute()), 16);
            addModelVerdicts(6, false);
            sectionTitle("About this phone", 24);
            LinearLayout facts = K.row(); facts.setGravity(Gravity.TOP);
            facts.addView(fact("chip", "Memory I can use", memoryFact()), tileLp(0)); facts.addView(fact("file", "Free storage", storageFact()), tileLp(8)); facts.addView(fact("gauge", "Processor", coresFact()), tileLp(8));
            addGap(facts, 8);
            sectionTitle("Measurements", 24);
            LinearLayout box = K.card(); box.setPadding(dp(4), dp(4), dp(4), dp(4));
            box.addView(K.navRow("gauge", "Measure speed", "about 2 min", v -> measureCompute()));
            box.addView(K.navRow("history", "Check my phone again", null, v -> { go("setup"); profileDevice(); }));
            box.addView(K.navRow("info", "Full technical report", null, v -> showReport()));
            addGap(box, 8);
        }, null);
    }
    LinearLayout.LayoutParams tileLp(int leftDp) { LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(0, -1, 1f); p.leftMargin = dp(leftDp); return p; }
    View fact(String icon, String label, String value) {
        LinearLayout c = K.col(); c.setBackground(T.rounded(T.surface, 16, 1, T.line)); c.setPadding(dp(12), dp(12), dp(12), dp(12));
        c.addView(K.iconView(icon, T.ink2, 20)); K.add(c, K.text(label, UiTheme.Text.CAPTION, T.ink2), 8); K.add(c, K.text(value, UiTheme.Text.LABEL, T.ink), 2);
        c.setContentDescription(label + ": " + value); return c;
    }
    String deviceName() { try { JSONObject id = profile.getJSONObject("identity"); return id.getString("manufacturer") + " " + id.getString("model"); } catch (Exception e) { return null; } }
    String memoryFact() { try { JSONObject m = profile.getJSONObject("memory").getJSONObject("grantable_quiesced"); return m.isNull("value") ? "Not measured" : UiHumanize.gb(m.getLong("value")); } catch (Exception e) { return "Not measured"; } }
    String storageFact() { try { return UiHumanize.gb(profile.getJSONObject("storage").getLong("free_bytes")); } catch (Exception e) { return "Unknown"; } }
    String coresFact() { try { JSONArray cl = profile.getJSONObject("cpu").getJSONArray("clusters"); int n = 0; for (int i = 0; i < cl.length(); i++) n += cl.getJSONObject(i).getJSONArray("core_ids").length(); return n + " cores"; } catch (Exception e) { return "Unknown"; } }
    String phoneVerdict() {
        if (recs == null) return recsError != null ? "I couldn't work out what fits yet." : "Working out what fits…";
        double best = 0; int runs = 0;
        for (int i = 0; i < recs.length(); i++) { JSONObject e = recs.optJSONObject(i); if (e == null || !"Runs".equals(e.optString("verdict"))) continue; runs++; best = Math.max(best, conservativeTokS(e)); }
        if (runs == 0 && notMeasured()) return "I need to measure this phone before I can say what it runs.";
        if (runs == 0) return "None of the listed models fit this phone yet.";
        if (best >= 20) return "Great for everyday tasks.";
        if (best >= 8) return "Good for everyday tasks. Bigger models take their time.";
        return "This phone runs small models. Expect slower replies.";
    }
    /** The speed a promise may rest on: the measured value, else the low end of the predicted range (never the midpoint). */
    static double conservativeTokS(JSONObject e) {
        JSONObject d = e.optJSONObject("decode_tok_s"); if (d == null) return 0;
        return UiKit.basisOf(d.optString("provenance")) == UiKit.Basis.MEASURED && !Double.isNaN(d.optDouble("value", Double.NaN)) ? d.optDouble("value") : d.optDouble("lo", 0);
    }
    boolean notMeasured() {
        if (recs == null) return false;
        for (int i = 0; i < recs.length(); i++) { JSONObject e = recs.optJSONObject(i); if (e != null && (e.optString("verdict") + e.optString("detail")).toLowerCase(Locale.ROOT).contains("notcalibrated")) return true; }
        try { JSONObject cp = profile.getJSONObject("cpu").optJSONObject("compute"); return cp == null || cp.isNull("value"); } catch (Exception e) { return false; }
    }
    JSONObject bestRecommendation() { if (recs == null) return null; for (int i = 0; i < recs.length(); i++) { JSONObject e = recs.optJSONObject(i); if (e != null && e.optBoolean("recommended")) return e; } return null; }
    /** One card per model from the recommendation list: plain speed words first, the measured-or-not chip, the number second. */
    void addModelVerdicts(int max, boolean fitsOnly) {
        if (recs == null) { if (recsError != null) addGap(problemCard(UiHumanize.of(recsError)), 16); else { LinearLayout r = K.row(); r.addView(K.pulse()); TextView t = K.text("Working out which models fit…", UiTheme.Text.BODY, T.ink2); LinearLayout.LayoutParams p = K.weight(); p.leftMargin = dp(10); r.addView(t, p); addGap(r, 16); } return; }
        int shown = 0;
        for (int pass = 0; pass < 3; pass++) for (int i = 0; i < recs.length() && shown < max; i++) {
            final JSONObject e = recs.optJSONObject(i); if (e == null) continue; boolean runs = "Runs".equals(e.optString("verdict")), have = e.optBoolean("downloaded");
            if (pass == 0 && !have || pass == 1 && (have || !runs) || pass == 2 && (have || runs)) continue; if (fitsOnly && !runs && !have) continue;
            addGap(modelCard(e), shown == 0 ? 16 : 10); shown++;
        }
    }
    View modelCard(final JSONObject e) {
        boolean runs = "Runs".equals(e.optString("verdict"));
        LinearLayout c = runs ? K.card() : K.col(); if (!runs) { c.setBackground(T.dashed(T.surface, 18, T.lineStrong)); c.setPadding(dp(16), dp(14), dp(16), dp(14)); }
        LinearLayout top = K.row(); top.setGravity(Gravity.TOP);
        LinearLayout tx = K.col(); TextView nm = K.text(e.optString("name"), UiTheme.Text.TITLE, T.ink); tx.addView(nm);
        String meta = (e.optString("params_note").isEmpty() ? "" : e.optString("params_note") + " · ") + UiHumanize.gb(e.optLong("size_bytes")) + (e.optBoolean("downloaded") ? " · On this phone" : "");
        tx.addView(K.text(meta, UiTheme.Text.CAPTION, T.ink2), K.lpGap(2)); top.addView(tx, K.weight());
        JSONObject d = e.optJSONObject("decode_tok_s");
        if (runs && d != null) top.addView(K.chip(UiKit.basisOf(d.optString("provenance"))));
        if (e.optBoolean("recommended") && !e.optBoolean("downloaded")) { LinearLayout.LayoutParams bp = new LinearLayout.LayoutParams(-2, -2); bp.topMargin = dp(8); tx.addView(K.pill(null, "Best fit for this phone", T.sunTint, T.sunInk), bp); }
        c.addView(top);
        if (runs && d != null) {
            UiKit.Basis b = UiKit.basisOf(d.optString("provenance")); double v = d.optDouble("value", Double.NaN), lo = d.optDouble("lo", Double.NaN), hi = d.optDouble("hi", Double.NaN);
            double mid = conservativeTokS(e);
            LinearLayout sp = K.row(); sp.setPadding(0, dp(10), 0, dp(6));
            sp.addView(K.text(UiHumanize.speedWords(mid), UiTheme.Text.LABEL, T.ink), K.weight());
            // a single number only for a measured cell; anything predicted is a range (08 section 6)
            String num = b == UiKit.Basis.MEASURED && !Double.isNaN(v) ? String.format(Locale.ROOT, "%.1f tokens/s", v) : (Math.round(lo) == Math.round(hi) ? String.format(Locale.ROOT, "about %.0f tokens/s", lo) : String.format(Locale.ROOT, "about %.0f–%.0f tokens/s", lo, hi));
            sp.addView(K.text(num, UiTheme.Text.CAPTION, T.ink2)); c.addView(sp);
            LinearLayout meter = K.row(); for (int k = 0; k < 3; k++) { View s = new View(this); s.setBackground(T.rounded(k < UiHumanize.speedLevel(mid) ? T.ink : T.line, 999)); LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(0, dp(6), 1f); if (k > 0) p.leftMargin = dp(4); meter.addView(s, p); }
            c.addView(meter);
            if ("streamed".equals(e.optString("tier"))) K.add(c, K.text("Bigger than this phone's memory, so it reads from fast storage as it goes.", UiTheme.Text.CAPTION, T.ink2), 8);
        } else if (!runs) {
            UiHumanize.Problem pr = UiHumanize.of(e.optString("verdict") + ": " + e.optString("detail"));
            K.add(c, K.text(pr.title.equals("Something went wrong on my side.") ? "Doesn't fit this phone." : pr.title, UiTheme.Text.BODY, T.ink2), 8);
        }
        if (!e.optString("license").isEmpty() && !e.optString("license").startsWith("apache") && !e.optString("license").equals("mit"))
            K.add(c, K.text("Licence: " + e.optString("license") + ". Check its terms before using it for work.", UiTheme.Text.CAPTION, T.warm), 8);
        LinearLayout acts = K.row(); acts.setPadding(0, dp(10), 0, 0);
        if (e.optBoolean("downloaded")) { boolean inUse = selectedModel != null && selectedModel.getName().equals(e.optString("filename"));
            acts.addView(K.button(inUse ? "In use" : "Use this one", inUse ? UiKit.Kind.QUIET : UiKit.Kind.SECONDARY, inUse ? null : v -> useModel(e.optString("filename")))); }
        else if (runs && activeDl == null) acts.addView(K.button("Download · " + UiHumanize.gb(e.optLong("size_bytes")), "download", e.optBoolean("recommended") ? UiKit.Kind.PRIMARY : UiKit.Kind.SECONDARY, v -> { downloadUrl(e.optString("url"), e.optString("sha256")); go("models"); }));
        if (acts.getChildCount() > 0) c.addView(acts);
        return c;
    }
    void showReport() {
        LinearLayout c = K.col(); c.addView(K.heading("Technical report", UiTheme.Text.HEADLINE));
        String rep; try { rep = profile == null ? "No profile yet." : Report.render(profile); } catch (JSONException e) { rep = "Report unavailable: " + e.getMessage(); }
        TextView t = K.text(colorize(rep), UiTheme.Text.MONO, T.ink); t.setTextIsSelectable(true); K.add(c, t, 12);
        final String r2 = rep; LinearLayout a = K.row(); a.setPadding(0, dp(12), 0, 0);
        a.addView(K.button("Copy", "copy", UiKit.Kind.SECONDARY, v -> { ((android.content.ClipboardManager) getSystemService(CLIPBOARD_SERVICE)).setPrimaryClip(ClipData.newPlainText("report", r2)); toast("Copied"); }));
        c.addView(a); K.sheet(c, true).show();
    }

    // =====================================================================================================================
    // Models
    // =====================================================================================================================
    View buildModels() {
        return frame(backHeader("Models", "Download once, then works offline"), () -> {
            if (activeDl != null) {
                LinearLayout c = K.card(); LinearLayout r = K.row(); r.addView(K.iconView("download", T.ink2, 22));
                LinearLayout.LayoutParams p = K.weight(); p.leftMargin = dp(12); TextView n = K.text("Downloading " + UiHumanize.modelName(dlName), UiTheme.Text.TITLE, T.ink); r.addView(n, p); c.addView(r);
                dlBarNew = K.bar(T.primary); dlBarNew.set(dlTotal > 0 ? dlDone / (float) dlTotal : 0); K.add(c, dlBarNew, 12);
                dlLabel = K.text(dlProgressText(), UiTheme.Text.CAPTION, T.ink2); K.add(c, dlLabel, 6);
                LinearLayout a = K.row(); a.setPadding(0, dp(8), 0, 0); a.addView(K.button("Cancel", UiKit.Kind.SECONDARY, v -> { if (activeDl != null) activeDl.cancelled = true; })); c.addView(a);
                c.setContentDescription("Downloading " + dlName); addGap(c, 12);
            }
            if (modelProblem != null) addGap(problemCard(modelProblem), 12);
            else if (modelMsg != null) addGap(modelMsg.endsWith("…") ? K.pill("info", modelMsg, T.surface2, T.ink2) : K.pill("check", modelMsg, T.goodTint, T.good), 12);
            List<File> ms = allModels();
            sectionTitle("On this phone", 16);
            if (ms.isEmpty()) addGap(K.text("No models yet. Pick one below that fits this phone.", UiTheme.Text.BODY, T.ink2), 8);
            for (final File f : ms) addGap(installedCard(f), 8);
            if (profile != null) { sectionTitle("Fits this phone", 24); addModelVerdicts(8, true); }
            else addGap(needsYou("gauge", "Check this phone first", "Then I can tell you which models fit.", "Check my phone", v -> go("setup")), 16);
            sectionTitle("Find more", 24);
            final EditText q = K.field("Search, e.g. qwen3", true); q.setImeOptions(EditorInfo.IME_ACTION_SEARCH);
            q.setOnEditorActionListener((v, id, e) -> { if (id == EditorInfo.IME_ACTION_SEARCH) { hideKeyboard(q); searchHf(q.getText().toString().trim()); return true; } return false; });
            addGap(q, 8); addGap(K.button("Search", "search", UiKit.Kind.SECONDARY, v -> { hideKeyboard(q); searchHf(q.getText().toString().trim()); }), 8);
            if (searchMsg != null) addGap(K.text(searchMsg, UiTheme.Text.CAPTION, T.ink2), 8);
            if (repoFiles != null) { addGap(K.text(repoName, UiTheme.Text.TITLE, T.ink), 12);
                for (int i = 0; i < repoFiles.length(); i++) { final JSONObject f = repoFiles.optJSONObject(i); if (f == null) continue;
                    LinearLayout c = K.card(); c.addView(K.text(f.optString("filename"), UiTheme.Text.BODY, T.ink)); K.add(c, K.text(UiHumanize.gb(f.optLong("size_bytes")), UiTheme.Text.CAPTION, T.ink2), 2);
                    LinearLayout a = K.row(); a.setPadding(0, dp(8), 0, 0); a.addView(K.button("Will it fit?", UiKit.Kind.SECONDARY, v -> evaluateUrl(f.optString("url"))));
                    if (activeDl == null) a.addView(K.button("Download", UiKit.Kind.QUIET, v -> downloadUrl(f.optString("url"), f.optString("sha256")))); c.addView(a); addGap(c, 8); } }
            else if (searchRepos != null) for (int i = 0; i < searchRepos.length(); i++) { final String repo = searchRepos.optJSONObject(i).optString("repo");
                addGap(K.navRow("layers", repo, searchRepos.optJSONObject(i).optLong("downloads") + " downloads", v -> listRepo(repo)), 2); }
            if (lastEval != null || evalMsg != null) addGap(evalCard(), 12);
            sectionTitle("Add from a link", 24);
            final EditText url = K.field("https://…/model.gguf", true); addGap(url, 8);
            LinearLayout a = K.row(); a.addView(K.button("Will it fit?", UiKit.Kind.SECONDARY, v -> { hideKeyboard(url); evaluateUrl(url.getText().toString().trim()); }));
            if (activeDl == null) a.addView(K.button("Download", UiKit.Kind.QUIET, v -> { hideKeyboard(url); downloadUrl(url.getText().toString().trim(), ""); }));
            addGap(a, 8);
            addGap(K.button("Import a model file", "file", UiKit.Kind.SECONDARY, v -> { Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT); i.addCategory(Intent.CATEGORY_OPENABLE); i.setType("*/*"); startActivityForResult(i, 7); }), 24);
        }, null);
    }
    String dlProgressText() { return dlTotal > 0 ? UiHumanize.gb(dlDone) + " of " + UiHumanize.gb(dlTotal) + " · " + (int) (100 * dlDone / Math.max(1, dlTotal)) + "%" : "Starting…"; }
    View installedCard(final File f) {
        LinearLayout c = K.card(); boolean inUse = f.equals(selectedModel);
        LinearLayout top = K.row(); LinearLayout tx = K.col(); tx.addView(K.text(UiHumanize.modelName(f.getName()), UiTheme.Text.TITLE, T.ink));
        tx.addView(K.text(UiHumanize.gb(f.length()) + (inUse ? (chat != null ? " · In use" : " · Selected") : ""), UiTheme.Text.CAPTION, T.ink2), K.lpGap(2)); top.addView(tx, K.weight());
        if (inUse && chat != null) top.addView(K.pill(null, "Ready", T.goodTint, T.good));
        c.addView(top);
        if (!f.getPath().startsWith(getFilesDir().getPath())) K.add(c, K.text("Stored outside the app. If it runs slowly, re-import it.", UiTheme.Text.CAPTION, T.warm), 6);
        LinearLayout a = K.row(); a.setPadding(0, dp(8), 0, 0);
        if (!inUse || chat == null) a.addView(K.button("Use this one", UiKit.Kind.SECONDARY, v -> useModel(f.getName())));
        a.addView(K.button("Remove", UiKit.Kind.QUIET, v -> confirmRemove(f)));
        c.addView(a); return c;
    }
    void confirmRemove(final File f) {
        LinearLayout c = K.col(); c.addView(K.heading("Remove " + UiHumanize.modelName(f.getName()) + "?", UiTheme.Text.HEADLINE));
        K.add(c, K.text("This frees " + UiHumanize.gb(f.length()) + ". You can download it again later.", UiTheme.Text.BODY_L, T.ink2), 8);
        final Dialog[] d = {null};
        Button rm = K.button("Remove", UiKit.Kind.DESTRUCTIVE, v -> { d[0].dismiss(); if (f.equals(selectedModel)) { if (chat != null) unloadEngine(); selectedModel = null; } f.delete(); modelMsg = "Removed " + UiHumanize.modelName(f.getName()); refreshModels(); refreshRecommendations(); refresh("models", "home"); });
        Button keep = K.button("Keep it", UiKit.Kind.SECONDARY, v -> d[0].dismiss());
        K.add(c, rm, 20); K.add(c, keep, 8); d[0] = K.sheet(c, true); d[0].show();
    }
    View evalCard() {
        if (lastEval == null) { LinearLayout c = K.card(); LinearLayout r = K.row(); r.addView(K.pulse()); LinearLayout.LayoutParams p = K.weight(); p.leftMargin = dp(12); r.addView(K.text(evalMsg, UiTheme.Text.BODY, T.ink2), p); c.addView(r); return c; }
        return modelCard(lastEval);
    }

    // =====================================================================================================================
    // Settings
    // =====================================================================================================================
    View buildSettings() {
        return frame(backHeader("Settings", null), () -> {
            sectionTitle("Phone access", 8);
            addGap(K.text("Meridian asks for each of these only when a task needs it. You can turn them on here too.", UiTheme.Text.BODY, T.ink2), 6);
            LinearLayout box = K.card(); box.setPadding(0, dp(4), 0, dp(4));
            box.addView(K.switchRow("Use other apps for you", "Accessibility: lets Meridian tap and type in apps when you ask.", A11y.inst != null, v -> openSettings(Settings.ACTION_ACCESSIBILITY_SETTINGS, false)));
            box.addView(K.switchRow("Read your notifications", "For summaries and finding messages you haven't answered.", Notifs.inst != null, v -> openSettings("android.settings.ACTION_NOTIFICATION_LISTENER_SETTINGS", false)));
            box.addView(K.switchRow("Files", "To find and open documents when you ask.", Build.VERSION.SDK_INT >= 30 && Environment.isExternalStorageManager(), v -> openSettings("android.settings.MANAGE_APP_ALL_FILES_ACCESS_PERMISSION", true)));
            NotificationManager nm = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
            box.addView(K.switchRow("Do Not Disturb", "To silence the phone when you ask.", nm != null && nm.isNotificationPolicyAccessGranted(), v -> openSettings(Settings.ACTION_NOTIFICATION_POLICY_ACCESS_SETTINGS, false)));
            if (Build.VERSION.SDK_INT >= 31) { AlarmManager am = (AlarmManager) getSystemService(ALARM_SERVICE);
                box.addView(K.switchRow("Alarms & reminders", "So reminders go off on time.", am != null && am.canScheduleExactAlarms(), v -> openSettings("android.settings.REQUEST_SCHEDULE_EXACT_ALARM", true))); }
            addGap(box, 8);
            sectionTitle("Privacy", 24);
            LinearLayout pv = K.card();
            pv.addView(K.text("The assistant model runs on this phone. Your tasks, notes and history are stored only here. Meridian goes online only for web searches and links, and only when a task needs them.", UiTheme.Text.BODY, T.ink));
            LinearLayout pa = K.row(); pa.setPadding(0, dp(8), 0, 0); pa.addView(K.button("Delete task history", UiKit.Kind.DESTRUCTIVE, v -> confirmClearHistory())); pv.addView(pa);
            addGap(pv, 8);
            sectionTitle("Places (optional)", 24);
            LinearLayout pl = K.card(); pl.addView(K.text("A Google Places key lets Meridian show ratings when you look for restaurants or shops. It is stored only on this phone.", UiTheme.Text.BODY, T.ink2));
            final EditText key = K.field("Google Places API key", true); key.setText(readPlacesKey()); key.setContentDescription("Google Places API key"); K.add(pl, key, 10);
            LinearLayout ka = K.row(); ka.setPadding(0, dp(8), 0, 0); ka.addView(K.button("Save key", UiKit.Kind.SECONDARY, v -> { hideKeyboard(key); savePlacesKey(key.getText().toString().trim()); })); pl.addView(ka);
            addGap(pl, 8);
            sectionTitle("Advanced", 24);
            LinearLayout adv = K.card(); adv.setPadding(0, dp(4), 0, dp(4));
            final boolean dev = prefs.getBoolean("dev", false);
            adv.addView(K.switchRow("Developer tools", "Measurements, logs and the engine controls.", dev, v -> { prefs.edit().putBoolean("dev", !dev).apply(); refresh("settings"); }));
            if (dev) adv.addView(K.navRow("code", "Open developer tools", null, v -> go("developer")));
            addGap(adv, 8);
            sectionTitle("About", 24);
            LinearLayout ab = K.card(); ab.setPadding(dp(4), dp(4), dp(4), dp(4));
            String ver = "?"; try { ver = getPackageManager().getPackageInfo(getPackageName(), 0).versionName; } catch (Exception ignored) { }
            ab.addView(K.navRow("info", "Meridian", ver == null ? "" : ver, v -> { }));
            ab.addView(K.navRow("file", "Font licences", null, v -> showLicences()));
            addGap(ab, 8);
        }, null);
    }
    void openSettings(String action, boolean withPackage) {
        try { Intent i = new Intent(action); if (withPackage) i.setData(Uri.parse("package:" + getPackageName())); startActivity(i); }
        catch (Exception e) { try { startActivity(new Intent(Settings.ACTION_SETTINGS)); } catch (Exception ignored) { } }
    }
    String readPlacesKey() { for (File d : new File[]{getFilesDir(), getExternalFilesDir(null)}) { if (d == null) continue; File f = new File(d, "keys.json"); if (!f.exists()) continue;
        try { return new JSONObject(Native.readFile(f.getAbsolutePath())).optString("google_places_key", ""); } catch (Exception ignored) { } } return ""; }
    void savePlacesKey(String k) {
        try (FileWriter w = new FileWriter(new File(getFilesDir(), "keys.json"))) { w.write(new JSONObject().put("google_places_key", k).toString()); toast(k.isEmpty() ? "Key removed" : "Key saved on this phone"); }
        catch (Exception e) { toast("Couldn't save the key"); }
    }
    void confirmClearHistory() {
        LinearLayout c = K.col(); c.addView(K.heading("Delete task history?", UiTheme.Text.HEADLINE));
        K.add(c, K.text("Meridian forgets your past tasks and their results. This can't be undone.", UiTheme.Text.BODY_L, T.ink2), 8);
        final Dialog[] d = {null};
        K.add(c, K.button("Delete history", UiKit.Kind.DESTRUCTIVE, v -> { d[0].dismiss(); new File(getFilesDir(), "memory.jsonl").delete(); toast("History deleted"); refresh("settings"); }), 20);
        K.add(c, K.button("Keep it", UiKit.Kind.SECONDARY, v -> d[0].dismiss()), 8);
        d[0] = K.sheet(c, true); d[0].show();
    }
    void showLicences() {
        LinearLayout c = K.col(); c.addView(K.heading("Font licences", UiTheme.Text.HEADLINE));
        for (String n : new String[]{"license_figtree", "license_bricolage"}) { int id = getResources().getIdentifier(n, "raw", getPackageName()); if (id == 0) continue;
            StringBuilder b = new StringBuilder(); try (BufferedReader r = new BufferedReader(new InputStreamReader(getResources().openRawResource(id)))) { String l; while ((l = r.readLine()) != null) b.append(l).append('\n'); } catch (IOException ignored) { }
            TextView t = K.text(b.toString().trim(), UiTheme.Text.MONO, T.ink2); t.setTextIsSelectable(true); K.add(c, t, 16); }
        K.sheet(c, true).show();
    }

    // =====================================================================================================================
    // Drawer
    // =====================================================================================================================
    void openDrawer() {
        if (drawer != null) return;
        FrameLayout layer = new FrameLayout(this);
        View scrim = new View(this); scrim.setBackgroundColor(T.scrim); scrim.setContentDescription("Close menu"); scrim.setOnClickListener(v -> closeDrawer());
        layer.addView(scrim, new FrameLayout.LayoutParams(-1, -1));
        LinearLayout p = K.col(); p.setBackground(T.rounded(T.bg, 0)); p.setPadding(dp(12), dp(16), dp(12), dp(12)); p.setClickable(true);
        android.graphics.drawable.GradientDrawable g = new android.graphics.drawable.GradientDrawable(); g.setColor(T.bg); float r = dp(28); g.setCornerRadii(new float[]{0, 0, r, r, r, r, 0, 0}); p.setBackground(g);
        LinearLayout brand = K.row(); brand.setPadding(dp(12), 0, 0, dp(12)); ImageView m = new ImageView(this); m.setImageDrawable(T.mark(this)); m.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);
        brand.addView(m, new LinearLayout.LayoutParams(dp(32), dp(32))); TextView bt = K.text("Meridian", UiTheme.Text.TITLE_L, T.ink); bt.setTypeface(T.display); LinearLayout.LayoutParams bp = K.weight(); bp.leftMargin = dp(12); brand.addView(bt, bp);
        p.addView(brand);
        p.addView(K.navRow("plus", "New task", null, v -> { viewing = null; go("home"); }));
        p.addView(K.navRow("msg", "Chat", null, v -> go("chat")));
        p.addView(K.navRow("phone", "This phone", null, v -> go("phone")));
        p.addView(K.navRow("layers", "Models", null, v -> go("models")));
        p.addView(K.navRow("gear", "Settings", null, v -> go("settings")));
        if (prefs.getBoolean("dev", false)) p.addView(K.navRow("code", "Developer tools", null, v -> go("developer")));
        List<JSONObject> hist = history(12);
        if (!hist.isEmpty()) { TextView rt = K.section("Recent tasks"); rt.setPadding(dp(12), dp(20), 0, dp(6)); p.addView(rt);
            LinearLayout list = K.col(); for (final JSONObject j : hist) list.addView(K.navRow(null, j.optString("request"), null, v -> openHistory(j)));
            ScrollView sv = K.scroll(list); p.addView(sv, new LinearLayout.LayoutParams(-1, 0, 1f)); }
        int w = Math.min(dp(320), (int) (getResources().getDisplayMetrics().widthPixels * 0.86f));
        layer.addView(p, new FrameLayout.LayoutParams(w, -1, Gravity.START));
        root.addView(layer, new FrameLayout.LayoutParams(-1, -1)); drawer = layer;
        if (ValueAnimatorsOn()) { p.setTranslationX(-w); p.animate().translationX(0).setDuration(180).start(); scrim.setAlpha(0f); scrim.animate().alpha(1f).setDuration(180).start(); }
        p.requestFocus();
    }
    boolean ValueAnimatorsOn() { return android.animation.ValueAnimator.areAnimatorsEnabled(); }
    void closeDrawer() { if (drawer == null) return; root.removeView(drawer); drawer = null; }

    // =====================================================================================================================
    // Developer tools: the original lab, unchanged in function, restyled
    // =====================================================================================================================
    final Map<String, View> tabViews = new LinkedHashMap<>();
    void buildDeveloperViews() {
        memGrant = new CheckBox(this); memGrant.setChecked(true);
        tabViews.clear();
        tabViews.put("Device", deviceTab()); tabViews.put("Models", modelsTab()); tabViews.put("Chat", chatTab()); tabViews.put("Agent", agentTab()); tabViews.put("Lab", labTab()); tabViews.put("Audit", auditTab());
    }
    /** After a theme change: rebuild the developer views but keep what they were showing. */
    void buildDeveloperViewsKeepingText() {
        CharSequence ds = deviceStatus.getText(), dr = deviceReport.getText(), ms = modelStatus.getText(), co = chatOut.getText(), ci = chatInfo.getText(), al = agentLog.getText(), lo = labOut.getText(); boolean mg = memGrant.isChecked();
        buildDeveloperViews(); deviceStatus.setText(ds); deviceReport.setText(dr); modelStatus.setText(ms); chatOut.setText(co); chatInfo.setText(ci); agentLog.setText(al); labOut.setText(lo); memGrant.setChecked(mg);
        refreshModels(); if (recs != null) renderRecs(recs);
    }
    View buildDeveloper() {
        LinearLayout f = K.col();
        f.addView(backHeader("Developer tools", "Raw measurements and controls"));
        HorizontalScrollView hs = new HorizontalScrollView(this); hs.setHorizontalScrollBarEnabled(false); LinearLayout bar = K.row(); bar.setPadding(dp(12), 0, dp(12), dp(8)); hs.addView(bar);
        for (final String name : tabViews.keySet()) { boolean on = name.equals(devTab);
            Button b = K.button(name, on ? UiKit.Kind.PRIMARY : UiKit.Kind.SECONDARY, v -> { devTab = name; if (name.equals("Audit")) refreshAudit(); if (name.equals("Models")) refreshModels(); render(); });
            LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-2, -2); p.rightMargin = dp(6); bar.addView(b, p); }
        f.addView(hs);
        View tab = tabViews.get(devTab); if (tab.getParent() != null) ((ViewGroup) tab.getParent()).removeView(tab);
        f.addView(tab, new LinearLayout.LayoutParams(-1, 0, 1f));
        return K.centered(f);
    }

    // ---------- Device ----------
    View deviceTab() {
        LinearLayout l = col();
        deviceStatus = tv("Profile this phone once. Everything is measured on-device, as this app, and every number carries whether it was measured or not.", 14, DIM);
        memGrant.setText("Include memory-grant probe (fills RAM; may close background apps)"); memGrant.setTextColor(FG);
        deviceReport = mono("");
        l.addView(deviceStatus); l.addView(memGrant);
        l.addView(btn("Profile this device", v -> profileDevice()));
        l.addView(btn("Measure engine compute only (~2 min)", v -> measureCompute()));
        l.addView(btn("Calibrate thread placement (needs a selected model)", v -> calibratePlacement()));
        l.addView(deviceReport);
        return scroll(l);
    }
    void deviceStage(final String s) { onUi(() -> { deviceStatus.setText(s); String h = stageWords(s); boolean changed = !h.equals(setupStage); setupStage = h;
        if (stageView != null && stageView.isAttachedToWindow()) stageView.setText(h); else if (changed) refresh("setup", "phone", "home"); }); }
    long workStarted; TextView stageView, workClock;
    /** Profile.run / ComputeProbe progress strings in plain words, e.g. "compute probe Q4_0_L3N@dot (rep 1/2)" ->
     *  "Timing the processor on test models · pass 1 of 2". Unknown stages fall back to a generic line, never to raw text. */
    static String stageWords(String raw) {
        String l = raw == null ? "" : raw.toLowerCase(Locale.ROOT), what;
        if (l.contains("compute")) what = "Timing the processor on test models";
        else if (l.contains("grant") || l.contains("memprobe") || l.contains("fill")) what = "Checking how much memory Meridian can use";
        else if (l.contains("dram") || l.contains("bandwidth")) what = "Measuring memory speed";
        else if (l.contains("ufs") || l.contains("storage") || l.contains("read") || l.contains("write") || l.contains("i/o")) what = "Measuring storage speed";
        else if (l.contains("placement") || l.contains("core") || l.contains("cluster") || l.contains("thread")) what = "Finding the fastest cores";
        else if (l.contains("thermal") || l.contains("temp")) what = "Checking how warm the phone gets";
        else if (l.contains("device") || l.contains("identity") || l.contains("cpu")) what = "Reading this phone's details";
        else what = "Measuring";
        Matcher m = Pattern.compile("rep (\\d+)/(\\d+)").matcher(l);
        return m.find() ? what + " · pass " + m.group(1) + " of " + m.group(2) : what;
    }
    void profileDevice() {
        if (busy) { toast("Meridian is busy. Try again when the current task finishes."); return; }
        if (chat != null) { unloadEngine(); if (loadBtn != null) loadBtn.setText("Load engine with selected model"); }   // profiling needs an idle device
        final boolean mem = memGrant.isChecked(); setBusy(true); setupRunning = true; setupProblem = null; phoneProblem = null; setupStage = "Starting"; workStarted = System.currentTimeMillis(); ui.postDelayed(ticker, 1000); refresh("setup", "phone", "home");
        run(() -> {
            try {
                File dir = internalModels();
                JSONObject p = Profile.run(this, dir, mem, s -> deviceStage("Working: " + s + " ... keep this app in front, phone unplugged, don't touch it."));
                attachCompute(p);
                profile = p;
                try (FileWriter w = new FileWriter(new File(getFilesDir(), "profile.json"))) { w.write(p.toString(2)); }
                final String rep = Report.render(p);
                onUi(() -> { deviceStatus.setText("Profile saved."); deviceReport.setText(colorize(rep)); refreshModels(); refreshRecommendations(); });
            } catch (Exception e) { onUi(() -> { deviceStatus.setText("Profiling failed: " + e); setupProblem = UiHumanize.of("Profiling failed: " + e); phoneProblem = setupProblem; }); }
            finally { setBusy(false); onUi(() -> { setupRunning = false; setupStage = null; refresh("setup", "phone", "home"); }); }
        });
    }
    /** Runs the engine compute probe and stores it in the profile; a failure is recorded as unknown with its reason, never defaulted. */
    void attachCompute(JSONObject p) throws JSONException {
        try { p.getJSONObject("cpu").put("compute", ComputeProbe.run(this, p, s -> deviceStage("Working: " + s + " ... keep this app in front, phone unplugged."))); }
        catch (Exception e) { p.getJSONObject("cpu").put("compute", new JSONObject().put("value", JSONObject.NULL).put("provenance", "unknown").put("confidence", 0).put("source", "ComputeProbe failed: " + e.getMessage()));
            rec.write(new JSONObject().put("event", "compute_probe_failed").put("why", String.valueOf(e.getMessage())).put("t", System.currentTimeMillis() / 1000.0)); }
    }
    void measureCompute() {
        if (busy || profile == null) { toast(profile == null ? "Check your phone first." : "Meridian is busy. Try again when the current task finishes."); return; }
        if (chat != null) { unloadEngine(); if (loadBtn != null) loadBtn.setText("Load engine with selected model"); }
        setBusy(true); measuring = true; phoneProblem = null; setupStage = "Starting"; workStarted = System.currentTimeMillis(); ui.postDelayed(ticker, 1000); refresh("phone");
        run(() -> { try { attachCompute(profile); saveProfile(); final String rep = Report.render(profile);
                onUi(() -> { deviceStatus.setText("Compute probe done."); deviceReport.setText(colorize(rep)); refreshModels(); refreshRecommendations(); saveText("last_probe.txt", rep); }); }
            catch (Exception e) { onUi(() -> { deviceStatus.setText("Compute probe failed: " + e); phoneProblem = UiHumanize.of("Compute probe failed: " + e); }); }
            finally { setBusy(false); onUi(() -> { measuring = false; setupStage = null; refresh("phone"); }); } });
    }
    void saveProfile() throws IOException, JSONException { try (FileWriter w = new FileWriter(new File(getFilesDir(), "profile.json"))) { w.write(profile.toString(2)); } }
    void calibratePlacement() {
        if (busy || profile == null || selectedModel == null) { toast(profile == null ? "Profile the device first" : busy ? "Busy" : "Select a model first"); return; }
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
        Button go = btn("Download", v -> downloadUrl(url.getText().toString().trim(), sha.getText().toString()));
        Button cancel = btn("Cancel", v -> { if (activeDl != null) activeDl.cancelled = true; });
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
        if (busy) { toast("Meridian is busy. Try again when the current task finishes."); return null; }
        if (url == null || !url.startsWith("http")) { modelProblem = UiHumanize.p("That doesn't look like a download link.", "Paste a direct link that starts with https and ends with .gguf.", UiHumanize.Action.NONE, null, url); refresh("models"); return null; }
        final Downloader d = new Downloader(); setBusy(true); activeDl = d; dlName = url.substring(url.lastIndexOf('/') + 1); dlDone = 0; dlTotal = 0; modelProblem = null; modelMsg = null; refresh("models", "home");
        // under the foreground service: OxygenOS cut the app's network when the screen locked mid-download (15R, 2026-09-22)
        startForegroundService(new Intent(this, KeepAlive.class).putExtra("text", "Downloading " + url.substring(url.lastIndexOf('/') + 1)));
        run(() -> { try { File f = d.download(url, internalModels(), sha, (done, total) -> onUi(() -> { dlBar.setProgress(total > 0 ? (int) (done * 1000 / total) : 0); modelStatus.setText("Downloading " + (done >> 20) + " / " + (total >> 20) + " MiB");
                        dlDone = done; dlTotal = total; if (dlBarNew != null) dlBarNew.set(total > 0 ? done / (float) total : 0); if (dlLabel != null) dlLabel.setText(dlProgressText()); }));
                onUi(() -> { modelStatus.setText("Downloaded and verified: " + f.getName()); modelMsg = "Downloaded " + UiHumanize.modelName(f.getName()) + ". It's ready to use."; if (selectedModel == null) selectedModel = f; refreshModels(); saveText("last_download.txt", "OK " + f.getName() + " " + f.length()); }); }
            catch (Exception e) { onUi(() -> { modelStatus.setText("Download failed: " + e.getMessage()); if (!d.cancelled) modelProblem = UiHumanize.of("Download failed: " + e.getMessage()); else modelMsg = null; saveText("last_download.txt", "FAIL " + e.getMessage()); }); }
            finally { setBusy(false); onUi(() -> { activeDl = null; dlBarNew = null; dlLabel = null; if (chat == null) stopService(new Intent(this, KeepAlive.class)); refreshRecommendations(); refresh("models", "home"); }); } });
        return d;
    }
    @Override protected void onActivityResult(int req, int res, final Intent data) {
        super.onActivityResult(req, res, data);
        if (req == 9) { if (res == RESULT_OK && data != null && voiceTarget != null) { ArrayList<String> r = data.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS);
                if (r != null && !r.isEmpty()) { voiceTarget.setText(r.get(0)); voiceTarget.setSelection(voiceTarget.getText().length()); } } return; }
        if (req != 7 || res != RESULT_OK || data == null) return;
        final Uri u = data.getData(); if (busy) { toast("Meridian is busy. Try again when the current task finishes."); return; } setBusy(true); modelProblem = null; modelMsg = "Importing…"; refresh("models");
        run(() -> { File out = null; try {
            String name = "imported.gguf"; try (android.database.Cursor c = getContentResolver().query(u, null, null, null, null)) { if (c != null && c.moveToFirst()) { int i = c.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME); if (i >= 0) name = c.getString(i); } }
            out = new File(internalModels(), name.replaceAll("[^A-Za-z0-9._-]", "_"));
            long total = 0; try (InputStream in = getContentResolver().openInputStream(u); OutputStream o = new FileOutputStream(out)) { byte[] b = new byte[1 << 20]; int n; while ((n = in.read(b)) > 0) { o.write(b, 0, n); total += n; final long t = total; onUi(() -> modelStatus.setText("Importing " + (t >> 20) + " MiB")); } }
            try { Gguf.read(out); } catch (Gguf.GgufError e) { out.delete(); throw new IOException("not a valid GGUF: " + e.getMessage()); }
            final String nm = out.getName(); onUi(() -> { modelStatus.setText("Imported " + nm); modelMsg = "Imported " + UiHumanize.modelName(nm); refreshModels(); refreshRecommendations(); });
        } catch (Exception e) { if (out != null) out.delete(); onUi(() -> { modelStatus.setText("Import failed: " + e.getMessage()); modelMsg = null; modelProblem = UiHumanize.of("Import failed: " + e.getMessage()); }); }
        finally { setBusy(false); onUi(() -> refresh("models")); } });
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
        if (profile == null) { recList.removeAllViews(); recList.addView(tv("Profile this phone first (Device tab): recommendations come from its measurements.", 14, WARN)); recs = null; return; }
        run(() -> { try { final JSONArray r = Catalog.recommend(this, profile, internalModels()); saveText("last_recommend.json", r.toString(2)); onUi(() -> { recs = r; recsError = null; renderRecs(r); refresh("phone", "models", "home", "setup"); }); }
            catch (Exception e) { onUi(() -> { recList.removeAllViews(); recList.addView(tv("Recommendation failed: " + e, 14, WARN)); recsError = "Recommendation failed: " + e; refresh("phone", "models", "setup"); }); } });
    }
    void renderRecs(JSONArray r) {
        recList.removeAllViews();
        for (int i = 0; i < r.length(); i++) { try { final JSONObject e = r.getJSONObject(i);
            String head = (e.optBoolean("recommended") ? "RECOMMENDED  " : "") + e.getString("name") + "  (" + e.optString("params_note") + ", " + e.optString("quant") + ", " + gib(e.getLong("size_bytes")) + ", " + e.optString("kind") + ")";
            LinearLayout row = new LinearLayout(this); row.setOrientation(LinearLayout.VERTICAL); row.setBackground(T.rounded(PANEL, 12, 1, T.line)); row.setPadding(dp(10), dp(8), dp(10), dp(8));
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2); lp.setMargins(0, dp(4), 0, dp(4));
            TextView h = tv(head, 14, e.optBoolean("recommended") ? GOOD : FG); h.setTypeface(T.bodySemi); row.addView(h); row.addView(mono(evalText(e)));
            if (!e.optString("license").isEmpty() && !e.optString("license").startsWith("apache") && !e.optString("license").equals("mit")) row.addView(tv("license: " + e.optString("license"), 12, WARN));
            LinearLayout bs = new LinearLayout(this);
            if (e.optBoolean("downloaded")) bs.addView(btn("Use this model", v -> useModel(e.optString("filename"))));
            else if (e.getString("verdict").equals("Runs")) bs.addView(btn("Download " + gib(e.getLong("size_bytes")), v -> downloadUrl(e.optString("url"), e.optString("sha256"))));
            row.addView(bs); recList.addView(row, lp);
        } catch (JSONException ignored) { } }
    }
    /** Select a model and get the engine ready for it in the background; the user never has to press "Load". */
    void useModel(String filename) {
        File pick = null; for (File f : allModels()) if (f.getName().equals(filename)) pick = f;
        if (pick == null) { toast("That model isn't on this phone any more."); return; }
        if (pick.equals(selectedModel) && chat != null) { toast(UiHumanize.modelName(pick.getName()) + " is ready."); return; }
        if (chat != null) unloadEngine();
        selectedModel = pick; selectedCard = null; modelMsg = "Getting " + UiHumanize.modelName(pick.getName()) + " ready…"; refresh("models", "phone", "home");
        toggleEngine();
    }
    void evaluateUrl(final String url) {
        if (profile == null) { modelStatus.setText("Profile this phone first."); evalMsg = null; lastEval = null; modelProblem = UiHumanize.of("profile this phone"); refresh("models"); return; } if (url == null || url.isEmpty()) return;
        modelStatus.setText("Reading the header of " + url + " ..."); lastEval = null; evalMsg = "Checking whether it fits this phone…"; modelProblem = null; refresh("models");
        run(() -> { try { JSONObject e = Catalog.evaluateUrl(profile, url, internalModels()); final String t = e.getString("name") + " (" + e.getString("arch") + ", " + gib(e.getLong("size_bytes")) + ", header " + (e.getLong("header_bytes_fetched") >> 10) + " KiB read)\n" + evalText(e);
                if (!e.has("url")) e.put("url", url);
                saveText("last_evaluate.json", e.toString(2)); onUi(() -> { modelStatus.setText(colorize(t)); lastEval = e; evalMsg = null; refresh("models"); }); }
            catch (Exception e) { onUi(() -> { modelStatus.setText("Could not evaluate: " + e.getMessage()); lastEval = null; evalMsg = null; modelProblem = UiHumanize.of("Could not evaluate: " + e.getMessage()); refresh("models"); }); } });
    }
    void searchHf(final String q) {
        if (q.isEmpty()) return; searchList.removeAllViews(); searchList.addView(tv("Searching...", 13, DIM)); searchMsg = "Searching…"; searchRepos = null; repoFiles = null; refresh("models");
        run(() -> { try { final JSONArray repos = Catalog.searchRepos(q); onUi(() -> { searchList.removeAllViews(); if (repos.length() == 0) searchList.addView(tv("No ungated GGUF repos found.", 13, DIM));
                for (int i = 0; i < repos.length(); i++) { final String repo = repos.optJSONObject(i).optString("repo"); searchList.addView(btn(repo + "  (" + repos.optJSONObject(i).optLong("downloads") + " downloads)", v -> listRepo(repo))); }
                searchRepos = repos; searchMsg = repos.length() == 0 ? "Nothing found. Try a shorter name." : null; refresh("models"); }); }
            catch (Exception e) { onUi(() -> { searchList.removeAllViews(); searchList.addView(tv("Search failed: " + e.getMessage(), 13, WARN)); searchMsg = null; modelProblem = UiHumanize.of("Search failed: " + e.getMessage()); refresh("models"); }); } });
    }
    void listRepo(final String repo) {
        searchList.removeAllViews(); searchList.addView(tv("Files in " + repo + " ...", 13, DIM)); searchMsg = "Looking inside " + repo + "…"; refresh("models");
        run(() -> { try { final JSONArray fs = Catalog.repoFiles(repo); onUi(() -> { searchList.removeAllViews(); searchList.addView(tv(repo, 14, FG));
                for (int i = 0; i < fs.length(); i++) { final JSONObject f = fs.optJSONObject(i); LinearLayout row = new LinearLayout(this); row.setOrientation(LinearLayout.VERTICAL);
                    row.addView(tv(f.optString("filename") + "  " + gib(f.optLong("size_bytes")), 13, FG)); LinearLayout bs = new LinearLayout(this);
                    bs.addView(btn("Evaluate", v -> evaluateUrl(f.optString("url")))); bs.addView(btn("Download", v -> downloadUrl(f.optString("url"), f.optString("sha256"))));
                    row.addView(bs); searchList.addView(row); }
                repoName = repo; repoFiles = fs; searchMsg = null; refresh("models"); }); }
            catch (Exception e) { onUi(() -> { searchList.removeAllViews(); searchList.addView(tv("Listing failed: " + e.getMessage(), 13, WARN)); searchMsg = null; modelProblem = UiHumanize.of("Listing failed: " + e.getMessage()); refresh("models"); }); } });
    }
    List<File> allModels() { List<File> l = new ArrayList<>(); for (File d : new File[]{internalModels(), externalModels()}) if (d != null && d.listFiles() != null) for (File f : d.listFiles()) if (f.getName().endsWith(".gguf")) l.add(f); return l; }
    void refreshModels() {
        if (modelList == null) return; modelList.removeAllViews();
        final List<File> ms = allModels(); if (ms.isEmpty()) modelList.addView(tv("(none yet)", 14, DIM));
        for (final File f : ms) {
            final TextView info = mono(f.getName() + "\n" + (f.length() >> 20) + " MiB   " + (f.getPath().startsWith(getFilesDir().getPath()) ? "internal" : "app external (FUSE path: profile it before trusting streaming)") + "\nreading header...");
            LinearLayout row = new LinearLayout(this); row.setOrientation(LinearLayout.VERTICAL); row.setBackground(T.rounded(PANEL, 12, 1, T.line)); row.setPadding(dp(10), dp(8), dp(10), dp(8));
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2); lp.setMargins(0, dp(6), 0, dp(6));
            LinearLayout bs = new LinearLayout(this);
            bs.addView(btn("Select", v -> { selectedModel = f; selectedCard = null; toast("Selected " + f.getName()); }));
            bs.addView(btn("Chat", v -> { useModel(f.getName()); go("chat"); }));
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

    // ---------- Chat (engine) ----------
    View chatTab() {
        LinearLayout l = col();
        chatInfo = tv("Load a model to chat. Tokens/s shown are observed on this turn, not predictions.", 14, DIM);
        loadBtn = btn("Load engine with selected model", v -> toggleEngine());
        final EditText in = edit("Message"); chatOut = mono("");
        chatOut.setMovementMethod(new ScrollingMovementMethod());
        l.addView(chatInfo); l.addView(loadBtn);
        l.addView(btn("New chat", v -> { chatTurns = 0; chatOut.setText(""); chatMsgs.clear(); }));
        l.addView(in);
        l.addView(btn("Send", v -> { String q = in.getText().toString().trim(); if (q.isEmpty()) return; in.setText(""); send(q); }));
        l.addView(btn("Stop", v -> { if (chat != null) chat.cancel(); }));
        l.addView(chatOut);
        return scroll(l);
    }
    /** Engine status goes to the developer view verbatim and, while a task is starting, to the task screen in plain words. */
    void engineStatus(CharSequence s) { chatInfo.setText(s); }
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
        if (busy) { toast("Meridian is busy. Try again when the current task finishes."); return; }
        if (chat != null) { unloadEngine(); loadBtn.setText("Load engine with selected model"); engineStatus("Engine unloaded."); refresh("home", "models"); return; }
        String why = engineRefusal(); if (why != null) { engineStatus(colorize(why)); modelProblem = UiHumanize.of(why); modelMsg = null; refresh("models", "home"); return; }
        try {
            JSONObject plan = AutoPlan.plan(this, profile, selectedCard, selectedModel); saveText("plan.json", plan.toString(2));
            if (plan.has("refusal")) { String r = "Refusal " + plan.getJSONObject("refusal").getString("reason") + ": " + plan.getJSONObject("refusal").optString("detail"); engineStatus(colorize(r)); modelProblem = UiHumanize.of(r); modelMsg = null; refresh("models", "home"); return; }
            loadEngine(PlanV2.configFromJson(plan.getJSONObject("config"), selectedModel), plan);
        } catch (Exception e) {   // no compute probe yet: the model still runs, on the topology placement, with no promise attached
            try { Engine.Config cfg = AutoPlan.config(profile, selectedCard, selectedModel, "resident", 2048, 0); rec.write(new JSONObject().put("event", "plan_unavailable").put("why", String.valueOf(e.getMessage())));
                engineStatus(colorize("No prediction (" + e.getMessage() + "); loading without one.")); loadEngine(cfg, null); } catch (JSONException x) { engineStatus("error: " + x); } }
    }
    void unloadEngine() {
        if (governor != null) { governor.stop(); governor = null; } if (lease != null) { lease.revoke(); lease = null; }
        if (chat != null) { chat.close(); chat = null; } stopService(new Intent(this, KeepAlive.class));
    }
    /** Load the engine with `cfg`; when a plan is given, register its lease and start the governor with its ladder and falsification rule. */
    void loadEngine(final Engine.Config cfg, final JSONObject plan) {
        setBusy(true); onUi(() -> engineStatus("Loading " + cfg.describe() + " ..."));
        run(() -> { try { loadEngineNow(cfg, plan); onUi(() -> { modelMsg = UiHumanize.modelName(cfg.model.getName()) + " is ready."; modelProblem = null; refresh("models", "home"); }); warmAgent(); }
            catch (Exception e) { chat = null; onUi(() -> { engineStatus("Engine failed: " + e.getMessage()); modelMsg = null; modelProblem = UiHumanize.of("Engine failed: " + e.getMessage()); refresh("models", "home"); }); } finally { setBusy(false); } });
    }
    /** Prefill the agent's tool prefix right after a load, so the first task does not pay for it (Agent.warm, ~2 min on the 15R).
     *  Queued on the same single worker, so a task sent meanwhile simply starts after it; it shows as "Getting ready". */
    void warmAgent() {
        run(() -> { final Chat c = chat; if (c == null || selectedModel == null) return;
            warming = true; onUi(() -> { engineNote = "Preparing the assistant. This happens once after it loads."; refresh("home", "task"); });
            try { Agent.warm(this, c, tools, rec, selectedModel.getName()); }
            catch (Exception e) { try { rec.write(new JSONObject().put("event", "warm_failed").put("why", String.valueOf(e.getMessage()))); } catch (JSONException ignored) { } }
            finally { warming = false; onUi(() -> { engineNote = null; refresh("home", "task"); }); } });
    }
    /** Autonomy: the agent and chat never make the user press "Load": if no engine is running, plan and load the selected model
     *  (or the most capable downloaded model that fits, by the same ranking the recommendations use). Runs on the executor thread. */
    boolean ensureEngineNow(StringBuilder why) {
        if (chat != null) return true;
        try {
            if (profile == null) { why.append("Profile this phone first (Device tab)."); return false; }
            File pick = selectedModel; JSONObject bestPlan = null;
            if (pick == null) { double bestP = -1;
                for (File f : allModels()) { try { Planner.Card c = Planner.derive(f); JSONObject pl = AutoPlan.plan(this, profile, c, f);
                    if (!pl.has("refusal") && Predictor.paramsB(c) > bestP) { bestP = Predictor.paramsB(c); pick = f; bestPlan = pl; } } catch (Exception ignored) { } } }
            if (pick == null) { why.append("No model that fits this phone is downloaded yet (Models tab)."); return false; }
            selectedModel = pick; selectedCard = Planner.derive(pick);
            JSONObject plan = bestPlan != null ? bestPlan : AutoPlan.plan(this, profile, selectedCard, pick);
            if (plan.has("refusal")) { why.append(plan.getJSONObject("refusal").optString("detail")); return false; }
            final String nm = pick.getName();
            onUi(() -> { engineStatus("Loading " + nm + " ..."); engineNote = "Loading " + UiHumanize.modelName(nm) + ". The first task after opening Meridian takes the longest."; taskChanged(); });
            saveText("plan.json", plan.toString(2));
            loadEngineNow(PlanV2.configFromJson(plan.getJSONObject("config"), pick), plan); return chat != null;
        } catch (Exception e) { why.append("could not load a model: ").append(e.getMessage()); return false; }
    }
    void loadEngineNow(final Engine.Config cfg, final JSONObject plan) throws Exception {
        {
            Chat c = new Chat(this, cfg); c.start(900000); chat = c; chatTurns = 0; curCfg = cfg; curPlan = plan;
            JSONObject lst = null; String leaseTxt = "no plan: no lease, no governor (the configuration is uncalibrated)";
            if (plan != null) { long floor = plan.getJSONObject("lease").getLong("floor"), target = plan.getJSONObject("lease").getLong("target");
                try { lease = Lease.request(profile, floor, target); lst = lease.verify(); leaseTxt = "lease granted " + (lease.granted >> 20) + " MiB, verified resident " + (lease.verifiedResident >> 20) + " MiB (swapped " + (lease.swapped >> 20) + " MiB)";
                    lease.startHeartbeat(ui, 5000, (why, st) -> { if (governor != null) governor.onMemoryPressure(why); }); }
                catch (Planner.Refusal r) { lease = null; leaseTxt = "no memory lease (" + r.reason + ": " + r.getMessage() + "); the governor still watches heat and speed"; rec.write(new JSONObject().put("event", "lease_refused").put("why", r.getMessage())); }
                JSONObject pd = plan.getJSONObject("chosen").getJSONObject("decode_tok_s"); leaseTxt += String.format(Locale.ROOT, "%npredicted %.1f tokens/s (range %.1f-%.1f) [%s]", pd.getDouble("value"), pd.getDouble("lo"), pd.getDouble("hi"), pd.getString("provenance"));
                governor = new Governor(this, new Governor.Host() {
                    public void applyRung(JSONObject r, String why) { applyRungAsync(r, why); }
                    public void invalidated(String why) { onUi(() -> engineStatus(colorize("Plan falsified: " + why + ". Recalibrate before trusting it."))); } }, rec, plan); governor.start(); }
            final String fl = leaseTxt;
            onUi(() -> { startForegroundService(new Intent(this, KeepAlive.class).putExtra("model", cfg.model.getName())); loadBtn.setText("Unload engine"); engineStatus(colorize("Engine ready: " + cfg.describe() + "\nload " + c.engine().readyInfo().optDouble("load_s") + " s; " + fl)); refresh("home"); });
        }
    }
    /** Execute a ladder rung chosen by the governor: reload with the rung's change, or unload. Queued behind whatever job is running. */
    void applyRungAsync(final JSONObject r, final String why) {
        run(() -> { try {
            String act = r.getString("action"); Engine.Config c = curCfg; JSONObject pl = curPlan; if (c == null) return;
            rec.write(new JSONObject().put("event", "ladder_rung").put("action", act).put("why", why).put("expected_cost", r.getJSONObject("expected_cost")).put("t", System.currentTimeMillis() / 1000.0));
            onUi(() -> engineStatus(colorize("Governor: " + act + " because " + why)));
            if (act.equals("suspend") || act.equals("refuse")) { onUi(this::unloadEngine); return; }
            Engine.Config n = PlanV2.configFromJson(PlanV2.cfgJson(c), c.model); JSONObject p = r.getJSONObject("params");
            if (act.equals("change_placement")) { n.threads = p.getInt("threads"); n.cpuMask = p.getString("mask_hex"); } else if (act.equals("reduce_context")) n.ctx = p.getInt("ctx"); else if (act.equals("shrink_cache")) n.cacheCeilMb = n.cacheFloorMb;
            final Engine.Config nn = n; onUi(() -> { unloadEngine(); loadEngine(nn, pl); });
        } catch (Exception e) { onUi(() -> engineStatus("Governor failed to apply rung: " + e)); } });
    }
    @Override public void onTrimMemory(int level) { super.onTrimMemory(level); if (level >= TRIM_MEMORY_RUNNING_LOW && governor != null) governor.onMemoryPressure("onTrimMemory level " + level); }
    void send(final String q) {
        if (q == null || q.isEmpty()) return;
        if (busy) { toast("Meridian is busy. Try again when the current task finishes."); return; } setBusy(true);
        final ChatMsg m = new ChatMsg(); m.user = q; chatMsgs.add(m); final Boolean think = thinkFirst ? Boolean.TRUE : null;
        onUi(() -> { chatOut.append("\n> " + q + "\n"); refresh("chat"); });
        run(() -> { try {
            StringBuilder why = new StringBuilder(); if (!ensureEngineNow(why)) { final String w = why.toString(); onUi(() -> { engineStatus(colorize(w)); m.done = true; m.failed = true; m.info = w; refresh("chat"); }); return; }
            final boolean fresh = chatTurns == 0; final Chat c = chat;
            m.t0 = System.currentTimeMillis();
            c.ask(q, thinkFirst ? 1024 : 384, fresh, null, false, think, t -> onUi(() -> { chatOut.append(t); chatToken(m, t); })); chatTurns++;
            JSONObject d = c.lastResult; JSONObject cond = Regime.snapshot(this);
            rec.write(new JSONObject().put("turn_id", "chat." + System.currentTimeMillis()).put("model_id", selectedModel.getName()).put("prompt_tokens", d.optInt("n_prompt")).put("output_tokens", d.optInt("tokens"))
                .put("prefill_ms", d.optDouble("prefill_s") * 1000).put("predicted", curPlan == null ? JSONObject.NULL : curPlan.getJSONObject("chosen").getJSONObject("decode_tok_s")).put("observed", new JSONObject().put("tokens_per_s", d.optDouble("tok_s")).put("prefill_tokens_per_s", d.optDouble("prefill_tps")))
                .put("conditions", cond).put("in_regime", cond.getBoolean("in_regime")).put("outcome", "ok"));
            String pr = ""; if (curPlan != null) { JSONObject pd = curPlan.getJSONObject("chosen").getJSONObject("decode_tok_s"); double ob = d.optDouble("tok_s");
                pr = String.format(Locale.ROOT, "%npredicted %.1f (%.1f-%.1f) [%s]: %s", pd.getDouble("value"), pd.getDouble("lo"), pd.getDouble("hi"), pd.getString("provenance"), ob >= pd.getDouble("lo") && ob <= pd.getDouble("hi") ? "inside the range" : "OUTSIDE the range"); }
            final String info = String.format("decode %.2f tok/s, prefill %.1f tok/s (%d prompt tokens) - observed this turn [%s]", d.optDouble("tok_s"), d.optDouble("prefill_tps"), d.optInt("n_prompt"), cond.getBoolean("in_regime") ? "measured" : "prior: out of regime") + pr + "\nconfig: " + (curCfg == null ? "?" : curCfg.describe());
            final String quiet = String.format(Locale.ROOT, "%s · %.1f tokens/s", UiHumanize.modelName(selectedModel.getName()), d.optDouble("tok_s"));
            onUi(() -> { engineStatus(colorize(info)); m.done = true; m.info = quiet; refresh("chat"); }); if (governor != null && cond.getBoolean("in_regime")) governor.observe(d.optDouble("tok_s"));
        } catch (Exception e) { onUi(() -> { engineStatus("Turn failed: " + e.getMessage()); m.done = true; m.failed = true; m.info = "Turn failed: " + e.getMessage(); refresh("chat"); }); } finally { setBusy(false); } });
    }

    // ---------- Agent ----------
    View agentTab() {
        LinearLayout l = col();
        l.addView(tv("Ask for anything the tools below can do, in your own words. The agent picks tools step by step, each result is checked against the phone's real state, and anything that reaches other people (messages, calls) or cannot be undone asks you first.", 14, DIM));
        try { l.addView(mono(tools.schemaText())); } catch (JSONException ignored) { }
        final EditText taskIn = edit("e.g. Save a note that says buy milk, then tell me my battery level");
        agentLog = mono("");
        l.addView(taskIn);
        l.addView(btn("Run task", v -> runAgent(taskIn.getText().toString().trim())));
        l.addView(agentLog);
        return scroll(l);
    }
    void runAgent(final String taskText) {
        if (taskText == null || taskText.trim().isEmpty()) return;
        if (busy) { toast("I'm still working on the current task."); return; }
        setBusy(true); chatTurns = 0;
        final UiTask tk = new UiTask(taskText); task = tk; viewing = null; engineNote = null;
        agentLog.setText("Task: " + taskText + "\n");
        go("task");
        run(() -> { try {
            StringBuilder why = new StringBuilder(); if (!ensureEngineNow(why)) { final String w = why.toString(); onUi(() -> { agentLog.append("Cannot start: " + w + "\n"); tk.onFailure(w); taskChanged(); }); saveText("last_agent.txt", "Task: " + taskText + "\nFailed: " + w); return; }
            if (tk.stopRequested) throw new CancellationException("Stopped by the user before the task started.");
            onUi(() -> { if (tk.status == UiTask.Status.STARTING) { tk.status = UiTask.Status.RUNNING; tk.now = "Planning"; } taskChanged(); });
            final Agent a = new Agent(this, chat, tools, rec, selectedModel.getName()); runningAgent = a;
            String res = a.runLoop(taskText, 12, new Agent.UI() {
                // runLoop clears its cancel flag when it starts, so a Stop that lands in that gap is re-issued here.
                public void log(String s) { onUi(() -> { agentLog.append(s + "\n"); if (tk.onLog(s)) taskChanged(); }); if (tk.stopRequested) a.cancel(); }
                public void token(String t) { tk.onToken(t); scheduleLive(); }
                public boolean consent(String tool, String text) { return !tk.stopRequested && askConsent(tk, tool, text); }
            });
            onUi(() -> { agentLog.append("\nResult: " + res + "\n"); saveText("last_agent.txt", agentLog.getText().toString()); tk.onResult(res); taskChanged(); });
        } catch (Exception e) { onUi(() -> { agentLog.append("\nFailed: " + e.getMessage() + "\n"); saveText("last_agent.txt", agentLog.getText().toString()); tk.onFailure(String.valueOf(e.getMessage())); taskChanged(); }); }
        finally { runningAgent = null; setBusy(false); } });
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

    // ---------- input helpers ----------
    void startVoice(EditText target) {
        Intent i = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            .putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true).putExtra(RecognizerIntent.EXTRA_PROMPT, "Say your task");
        if (i.resolveActivity(getPackageManager()) == null) { toast("Voice typing isn't available on this phone."); return; }
        voiceTarget = target; try { startActivityForResult(i, 9); } catch (Exception e) { toast("Voice typing isn't available on this phone."); }
    }
    void hideKeyboard(View v) { android.view.inputmethod.InputMethodManager im = (android.view.inputmethod.InputMethodManager) getSystemService(INPUT_METHOD_SERVICE); if (im != null) im.hideSoftInputFromWindow(v.getWindowToken(), 0); }
}
