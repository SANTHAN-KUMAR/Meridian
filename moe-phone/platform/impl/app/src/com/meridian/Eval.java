package com.meridian.app;

import android.content.Context;
import org.json.*;
import java.io.*;
import java.util.*;
import java.util.regex.*;

/** Frozen agent task suite (suite-v1) with per-task success predicates fixed BEFORE any run and judged from device state and the tool trace
 *  (not from the model's own claim). Reports T (seconds, over correctly completed tasks) and S (fraction correct) TOGETHER, per variant, plus the
 *  no-model baselines: a keyword router (the no-data floor), and the best constant policy. Variants that use the model: free-form loop, plan/act/answer. */
public final class Eval {
    public interface Progress { void step(String s); }
    static final class St { String answer = ""; List<String> trace = new ArrayList<>(); List<String> notes; double battery, freeGb; String outcome = ""; }
    static abstract class Task { final String id, prompt; final List<String> setup; final boolean consent; Task(String id, String prompt, List<String> setup, boolean consent) { this.id = id; this.prompt = prompt; this.setup = setup; this.consent = consent; } abstract boolean pass(St s); }
    static List<Double> nums(String s) { List<Double> l = new ArrayList<>(); Matcher m = Pattern.compile("\\d+(?:\\.\\d+)?").matcher(s); while (m.find()) l.add(Double.parseDouble(m.group())); return l; }
    static boolean near(List<Double> v, double x, double tol) { for (double d : v) if (Math.abs(d - x) <= tol) return true; return false; }
    static boolean notesHave(List<String> n, String... k) { for (String e : n) { boolean all = true; for (String x : k) all &= e.toLowerCase().contains(x); if (all) return true; } return false; }
    static List<String> L(String... a) { return new ArrayList<>(Arrays.asList(a)); }

    static List<Task> suite() {
        List<Task> t = new ArrayList<>();
        t.add(new Task("battery", "What is my battery level?", L(), true) { boolean pass(St s) { return s.trace.contains("battery_status") && near(nums(s.answer), s.battery, 2); } });
        t.add(new Task("storage", "How much free storage do I have, in gigabytes?", L(), true) { boolean pass(St s) { return s.trace.contains("storage_free") && near(nums(s.answer), s.freeGb, 0.05 * s.freeGb + 0.2); } });
        t.add(new Task("note_milk", "Save a note: buy milk.", L(), true) { boolean pass(St s) { return notesHave(s.notes, "milk"); } });
        t.add(new Task("note_dentist", "Write down: dentist at 3pm.", L(), true) { boolean pass(St s) { return notesHave(s.notes, "dentist", "3"); } });
        t.add(new Task("list_notes", "What notes do I have?", L("alpha", "beta"), true) { boolean pass(St s) { return s.trace.contains("notes_list") && s.answer.toLowerCase().contains("alpha") && s.answer.toLowerCase().contains("beta"); } });
        t.add(new Task("eggs_battery", "Save a note to buy eggs and tell me my battery level.", L(), true) { boolean pass(St s) { return notesHave(s.notes, "eggs") && s.trace.contains("battery_status") && near(nums(s.answer), s.battery, 2); } });
        t.add(new Task("storage_note", "Check how much storage is free and save that number as a note.", L(), true) { boolean pass(St s) { for (String n : s.notes) if (near(nums(n), s.freeGb, 0.05 * s.freeGb + 0.2)) return true; return false; } });
        t.add(new Task("hello", "Say hello.", L(), true) { boolean pass(St s) { return s.trace.isEmpty() && s.answer.trim().length() >= 2; } });
        t.add(new Task("clear_deny", "Delete all my notes.", L("a", "b"), false) { boolean pass(St s) { return s.notes.size() == 2; } });
        t.add(new Task("clear_allow", "Please clear all my notes.", L("a", "b"), true) { boolean pass(St s) { return s.notes.isEmpty(); } });
        t.add(new Task("france", "What is the capital of France?", L(), true) { boolean pass(St s) { return s.trace.isEmpty() && s.answer.toLowerCase().contains("paris"); } });
        t.add(new Task("phrase", "how charged is my phone", L(), true) { boolean pass(St s) { return s.trace.contains("battery_status") && near(nums(s.answer), s.battery, 2); } });
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
            final Task tk = t; Agent.UI ui = new Agent.UI() { public void log(String s) { } public void token(String s) { } public boolean consent(String tool, String a) { return tk.consent; } };
            St st = new St(); long t0 = System.currentTimeMillis(); List<String> trace = new ArrayList<>();
            try {
                if (v.equals("plan_act_answer")) { st.answer = agent.run(t.prompt, 5, ui); trace = new ArrayList<>(agent.trace); }
                else if (v.equals("free_form")) { st.answer = agent.runFreeForm(t.prompt, 5, ui); trace = new ArrayList<>(agent.trace); }
                else if (v.equals("keyword_router")) st.answer = keywordRun(tools, t.prompt, ui, trace);
                else st.answer = constantRun(tools, v.substring(9), trace, ui);   // "constant:<tool|say_hello>"
            } catch (Exception e) { st.answer = "ERROR " + e; }
            double sec = (System.currentTimeMillis() - t0) / 1000.0; st.trace = trace; st.notes = tools.readNotes();
            Tools.Tool b = tools.registry.get("battery_status"), sf = tools.registry.get("storage_free"); st.battery = b.execute(new JSONObject()).getDouble("level_pct"); st.freeGb = sf.execute(new JSONObject()).getDouble("free_gb");
            boolean ok = t.pass(st); if (ok) { passT.get(v).add(sec); passN.put(v, passN.get(v) + 1); }
            rows.put(new JSONObject().put("task", t.id).put("variant", v).put("pass", ok).put("seconds", sec).put("trace", new JSONArray(trace)).put("answer", st.answer.length() > 200 ? st.answer.substring(0, 200) : st.answer));
        }
        JSONObject summary = new JSONObject();
        for (String v : variants) { List<Double> p = passT.get(v); Collections.sort(p); JSONObject o = new JSONObject().put("S", passN.get(v) + "/" + ts.size()).put("S_frac", passN.get(v) / (double) ts.size());
            if (!p.isEmpty()) o.put("T_median_s", p.get(p.size() / 2)).put("T_q1_s", p.get(p.size() / 4)).put("T_q3_s", p.get(Math.min(p.size() - 1, 3 * p.size() / 4))); else o.put("T_median_s", JSONObject.NULL); summary.put(v, o); }
        JSONObject out = new JSONObject().put("suite", "suite-v1").put("n_tasks", ts.size()).put("model", modelId).put("summary", summary).put("rows", rows)
            .put("note", "T is defined only over correctly completed tasks and is reported with S. Predicates were fixed before any run and read device state, not the model's claims. One device, one model, one repeat.");
        try (FileWriter w = new FileWriter(new File(ctx.getFilesDir(), "eval_" + modelId.replaceAll("[^A-Za-z0-9]", "_") + ".json"))) { w.write(out.toString(2)); }
        return out;
    }
}
