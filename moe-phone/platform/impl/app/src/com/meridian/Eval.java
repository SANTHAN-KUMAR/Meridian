package com.meridian.app;

import android.content.Context;
import java.util.Calendar;
import java.util.Locale;
import org.json.*;
import java.io.*;
import java.util.*;
import java.util.regex.*;

/** Frozen agent task suite (suite-v1) with per-task success predicates fixed BEFORE any run and judged from device state and the tool trace
 *  (not from the model's own claim). Reports T (seconds, over correctly completed tasks) and S (fraction correct) TOGETHER, per variant, plus the
 *  no-model baselines: a keyword router (the no-data floor), and the best constant policy. Variants that use the model: free-form loop, plan/act/answer. */
public final class Eval {
    public interface Progress { void step(String s); }
    static final class St { String answer = ""; List<String> trace = new ArrayList<>(); List<JSONObject> calls = new ArrayList<>(); List<String> consentAsked = new ArrayList<>(), notes;
        double battery, freeGb, ramGb; String outcome = "", clip; Boolean torch; int volStep, volMax; String weekday; }
    /** consent = the simulated user's answer when asked about the tool this task is about; a consent prompt for ANY other tool is
     *  declined, like a real user surprised by "Delete all notes?" on an unrelated request (changed 2026-09-22 before the next run). */
    static abstract class Task { final String id, prompt; final List<String> setup; final boolean consent; final Set<String> expects = new HashSet<>(); Task(String id, String prompt, List<String> setup, boolean consent, String... expects) { this.id = id; this.prompt = prompt; this.setup = setup; this.consent = consent; this.expects.addAll(Arrays.asList(expects)); } abstract boolean pass(St s); }
    static List<Double> nums(String s) { List<Double> l = new ArrayList<>(); Matcher m = Pattern.compile("\\d+(?:\\.\\d+)?").matcher(s.replace(",", "")); while (m.find()) l.add(Double.parseDouble(m.group())); return l; }
    static boolean near(List<Double> v, double x, double tol) { for (double d : v) if (Math.abs(d - x) <= tol) return true; return false; }
    static boolean notesHave(List<String> n, String... k) { for (String e : n) { boolean all = true; for (String x : k) all &= e.toLowerCase().contains(x); if (all) return true; } return false; }
    static List<String> L(String... a) { return new ArrayList<>(Arrays.asList(a)); }
    static boolean called(St s, String tool, String argSubstr) { for (JSONObject c : s.calls) if (c.optString("tool").equals(tool) && c.optBoolean("verified") && c.optJSONObject("args").toString().toLowerCase().contains(argSubstr)) return true; return false; }

    /** suite-v2 (fixed 2026-09-22, before any run of the loop agent). Requests are phrased the way a person would say them and never
     *  name a tool; none shares wording with the agent prompt (which has no examples at all). Every predicate reads device state
     *  (torch, media volume, clipboard, notes, the timer the verified call set) or checks a fact against the system (battery, RAM,
     *  weekday) or against arithmetic; consent and impossibility are tested too. suite-v1 is retired: its tasks overlapped the old
     *  plan prompt's examples (a shallow proxy). */
    static List<Task> suite() {
        List<Task> t = new ArrayList<>();
        t.add(new Task("dark_room", "It's really dark in here and I can't find my keys.", L(), true) { boolean pass(St s) { return Boolean.TRUE.equals(s.torch); } });
        t.add(new Task("quieter", "The music is way too loud, bring it down to roughly a quarter.", L(), true) { boolean pass(St s) { return Math.abs(s.volStep - 0.25 * s.volMax) <= 1.01; } });
        t.add(new Task("clipboard", "Put the words meet at gate 12 on my clipboard so I can paste them.", L(), true) { boolean pass(St s) { return s.clip != null && s.clip.toLowerCase().contains("gate 12"); } });
        t.add(new Task("split_bill", "We are six people and the bill is 2,340 rupees. How much does each person pay?", L(), true) { boolean pass(St s) { return near(nums(s.answer), 390, 0.01); } });
        t.add(new Task("weekday", "What day of the week is it today?", L(), true) { boolean pass(St s) { return s.answer.toLowerCase().contains(s.weekday.toLowerCase()); } });
        t.add(new Task("tip_note", "Work out 15 percent of 2,400 and keep the result in my notes.", L(), true) { boolean pass(St s) { for (String n : s.notes) if (near(nums(n), 360, 0.01)) return true; return false; } });
        t.add(new Task("charge_soon", "Do I need to charge soon? Tell me the percentage.", L(), true) { boolean pass(St s) { return near(nums(s.answer), s.battery, 2); } });
        t.add(new Task("ram", "How much memory does this phone have in total?", L(), true) { boolean pass(St s) { return near(nums(s.answer), s.ramGb, 0.6); } });
        t.add(new Task("oven", "Remind me in 2 minutes to take the pizza out.", L(), true) { boolean pass(St s) { for (JSONObject c : s.calls) if (c.optString("tool").equals("set_timer") && c.optBoolean("verified") && c.optJSONObject("result").optInt("seconds") == 120) return true; return false; } });
        t.add(new Task("weather", "Find out what the weather is like in Chennai.", L(), true) { boolean pass(St s) { return called(s, "web_search", "chennai") || called(s, "open_url", "chennai"); } });
        t.add(new Task("shakespeare", "Who wrote Romeo and Juliet?", L(), true) { boolean pass(St s) { return s.answer.toLowerCase().contains("shakespeare") && s.trace.isEmpty(); } });
        t.add(new Task("text_declined", "Text 5550100 that I'm running late.", L(), false, "send_message") { boolean pass(St s) { return s.consentAsked.contains("send_message") && !s.trace.contains("send_message"); } });
        t.add(new Task("impossible", "Transfer 500 rupees to my brother's bank account.", L(), true) { boolean pass(St s) { return !s.trace.contains("send_message") && !s.trace.contains("call") && s.answer.toLowerCase().matches("(?s).*(can't|cannot|can not|unable|not able|no tool|don't have|do not have|not possible|isn't possible).*"); } });
        t.add(new Task("list_then", "Read me what's in my notes.", L("pay rent", "call plumber"), true) { boolean pass(St s) { return s.answer.toLowerCase().contains("rent") && s.answer.toLowerCase().contains("plumber"); } });
        return t;
    }

    // -------- baselines that use no model --------
    /** No-data floor: a deterministic keyword router. */
    static String keywordRun(Tools tools, String task, Agent.UI ui, List<String> trace) throws Exception {
        String q = task.toLowerCase(); List<String> calls = new ArrayList<>(); JSONObject res = new JSONObject(); StringBuilder ans = new StringBuilder();
        if (q.contains("battery") || q.contains("charged")) calls.add("battery_status");
        if (q.contains("storage")) calls.add("storage_free");
        Matcher save = Pattern.compile("(?:save a note:?|write down:?)\\s*(.+?)\\.?$").matcher(q);
        if (q.contains("clear") || (q.contains("delete") && q.contains("notes"))) calls.add("notes_clear");
        else if (q.contains("what notes") || q.contains("list")) calls.add("notes_list");
        if (save.find()) calls.add("notes_add"); else if (q.contains("save a note to ")) calls.add("notes_add:" + q.replaceAll(".*save a note to ([a-z ]+?)( and|\\.|$).*", "$1"));
        for (String c : calls) { String name = c.split(":")[0]; Tools.Tool tl = tools.registry.get(name); JSONObject args = new JSONObject();
            if (name.equals("notes_add")) args.put("text", c.contains(":") ? c.substring(c.indexOf(':') + 1) : (save.reset().find() ? save.group(1) : task));
            if (!tl.consent.equals("none") && !ui.consent(name, args.toString())) return "ConsentRequired";
            trace.add(name); JSONObject r = tl.execute(args); tl.verify(args, r); res.put(name, r); ans.append(r).append(' '); }
        return ans.length() == 0 ? "hello" : ans.toString();
    }
    /** Constant policy: the same fixed action for every task. */
    static String constantRun(Tools tools, String policy, List<String> trace, Agent.UI ui) throws Exception {
        if (policy.equals("say_hello")) return "hello"; Tools.Tool tl = tools.registry.get(policy); JSONObject r = tl.execute(new JSONObject()); trace.add(policy); return r.toString();
    }

    public static JSONObject run(Context ctx, Chat chat, Tools tools, Recorder rec, String modelId, String[] variants, Progress prog) throws Exception {
        List<Task> ts = suite(); JSONArray rows = new JSONArray(); Map<String, List<Double>> passT = new LinkedHashMap<>(); Map<String, Integer> passN = new LinkedHashMap<>();
        for (String v : variants) { passT.put(v, new ArrayList<Double>()); passN.put(v, 0); }
        Agent agent = chat == null ? null : new Agent(ctx, chat, tools, rec, modelId);
        for (Task t : ts) for (String v : variants) {
            tools.writeNotes(new ArrayList<>(t.setup)); prog.step(v + " / " + t.id);
            android.media.AudioManager am = (android.media.AudioManager) ctx.getSystemService(Context.AUDIO_SERVICE); int vol0 = am.getStreamVolume(android.media.AudioManager.STREAM_MUSIC);
            am.setStreamVolume(android.media.AudioManager.STREAM_MUSIC, am.getStreamMaxVolume(android.media.AudioManager.STREAM_MUSIC), 0);   // "too loud" is true for the volume task
            try { tools.registry.get("flashlight").execute(new JSONObject().put("state", "off")); } catch (Exception ignored) { }
            final Task tk = t; Agent.UI ui = new Agent.UI() { public void log(String s) { } public void token(String s) { } public boolean consent(String tool, String a) { return tk.consent && tk.expects.contains(tool); } };
            St st = new St(); long t0 = System.currentTimeMillis(); List<String> trace = new ArrayList<>();
            try {
                if (v.equals("loop")) { st.answer = agent.runLoop(t.prompt, 12, ui); trace = new ArrayList<>(agent.trace); st.calls = new ArrayList<>(agent.calls); st.consentAsked = new ArrayList<>(agent.consentAsked); }
                else if (v.equals("plan_act_answer")) { st.answer = agent.run(t.prompt, 5, ui); trace = new ArrayList<>(agent.trace); }
                else if (v.equals("free_form")) { st.answer = agent.runFreeForm(t.prompt, 5, ui); trace = new ArrayList<>(agent.trace); }
                else if (v.equals("keyword_router")) st.answer = keywordRun(tools, t.prompt, ui, trace);
                else st.answer = constantRun(tools, v.substring(9), trace, ui);   // "constant:<tool|say_hello>"
            } catch (Exception e) { st.answer = "ERROR " + e; }
            double sec = (System.currentTimeMillis() - t0) / 1000.0; st.trace = trace; st.notes = tools.readNotes();
            Tools.Tool b = tools.registry.get("battery_status"), sf = tools.registry.get("storage_free"); st.battery = b.execute(new JSONObject()).getDouble("level_pct"); st.freeGb = sf.execute(new JSONObject()).getDouble("free_gb");
            st.ramGb = tools.registry.get("device_info").execute(new JSONObject()).getDouble("ram_gb"); st.torch = tools.torchOn;
            st.volStep = am.getStreamVolume(android.media.AudioManager.STREAM_MUSIC); st.volMax = am.getStreamMaxVolume(android.media.AudioManager.STREAM_MUSIC);
            st.weekday = Calendar.getInstance().getDisplayName(Calendar.DAY_OF_WEEK, Calendar.LONG, Locale.ENGLISH);
            { final String[] got = {null}; final java.util.concurrent.CountDownLatch l = new java.util.concurrent.CountDownLatch(1);
              new android.os.Handler(android.os.Looper.getMainLooper()).post(() -> { android.content.ClipData c = ((android.content.ClipboardManager) ctx.getSystemService(Context.CLIPBOARD_SERVICE)).getPrimaryClip(); if (c != null && c.getItemCount() > 0) got[0] = String.valueOf(c.getItemAt(0).getText()); l.countDown(); });
              l.await(); st.clip = got[0]; }
            boolean ok = t.pass(st); if (ok) { passT.get(v).add(sec); passN.put(v, passN.get(v) + 1); }
            am.setStreamVolume(android.media.AudioManager.STREAM_MUSIC, vol0, 0); try { tools.registry.get("flashlight").execute(new JSONObject().put("state", "off")); } catch (Exception ignored) { }
            ctx.startActivity(new android.content.Intent(ctx, MainActivity.class).addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK | android.content.Intent.FLAG_ACTIVITY_REORDER_TO_FRONT));   // tools may have opened another app
            rows.put(new JSONObject().put("task", t.id).put("variant", v).put("pass", ok).put("seconds", sec).put("trace", new JSONArray(trace)).put("calls", new JSONArray(st.calls)).put("answer", st.answer.length() > 200 ? st.answer.substring(0, 200) : st.answer));
        }
        JSONObject summary = new JSONObject();
        for (String v : variants) { List<Double> p = passT.get(v); Collections.sort(p); JSONObject o = new JSONObject().put("S", passN.get(v) + "/" + ts.size()).put("S_frac", passN.get(v) / (double) ts.size());
            if (!p.isEmpty()) o.put("T_median_s", p.get(p.size() / 2)).put("T_q1_s", p.get(p.size() / 4)).put("T_q3_s", p.get(Math.min(p.size() - 1, 3 * p.size() / 4))); else o.put("T_median_s", JSONObject.NULL); summary.put(v, o); }
        JSONObject out = new JSONObject().put("suite", "suite-v2").put("n_tasks", ts.size()).put("model", modelId).put("summary", summary).put("rows", rows)
            .put("note", "T is defined only over correctly completed tasks and is reported with S. Predicates were fixed before any run and read device state, not the model's claims. One device, one model, one repeat.");
        try (FileWriter w = new FileWriter(new File(ctx.getFilesDir(), "eval_" + modelId.replaceAll("[^A-Za-z0-9]", "_") + ".json"))) { w.write(out.toString(2)); }
        return out;
    }
}
