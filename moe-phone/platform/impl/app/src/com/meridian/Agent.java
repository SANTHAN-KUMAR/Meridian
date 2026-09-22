package com.meridian.app;

import android.content.Context;
import org.json.*;
import java.util.*;

/** intent -> act -> verify -> recover (07_AGENT_RUNTIME.md). The model never sees compute detail and never names a model or
 *  quantisation. Tool calls are validated against the ToolSpec, classified by SafetyGate (consent for non-"none" classes),
 *  executed, and verified against observable device state; a failed verification is fed back as evidence, not ignored. */
public final class Agent {
    public interface UI { void log(String line); boolean consent(String tool, String args); void token(String t); }
    private final Context ctx; private final Chat chat; private final Tools tools; private final Recorder rec; private final String modelId;
    private long taskCounter = System.currentTimeMillis();
    public final List<String> trace = new ArrayList<>();   // tools executed by the last run, in order (for task predicates)
    public final List<JSONObject> calls = new ArrayList<>();   // executed calls with their arguments and verification (runLoop)
    public final List<String> consentAsked = new ArrayList<>();

    public Agent(Context c, Chat chat, Tools tools, Recorder rec, String modelId) { ctx = c; this.chat = chat; this.tools = tools; this.rec = rec; this.modelId = modelId; }

    // ---------- SafetyGate (07_AGENT_RUNTIME.md section 8) ----------
    /** Communicate actions (send, call, reply) allowed per task and per rolling hour, so a looping agent cannot send twenty messages. */
    static final int COMM_PER_TASK = 5, COMM_PER_HOUR = 20;
    private final Set<String> granted = new HashSet<>();   // "once" grants of the current task only; cleared at every task start
    private int commThisTask = 0; public int overflows = 0;   // context overflows this task (counted, reported in the task memory)
    /** null when the call may run; otherwise the reason it may not (declined, or over the rate limit). Every decision is recorded. */
    String gate(Tools.Tool tool, JSONObject args, String taskId, UI ui) throws Exception {
        String cls = tool.consentFor(args); JSONObject act = new JSONObject().put("tool", tool.name).put("consent", cls);
        if (tool.effects.equals("communicate")) {
            int hour = recentComms(); String why = commThisTask >= COMM_PER_TASK ? "limit of " + COMM_PER_TASK + " messages/calls per task reached" : hour >= COMM_PER_HOUR ? "limit of " + COMM_PER_HOUR + " messages/calls per hour reached" : null;
            if (why != null) { rec.write(new JSONObject().put("turn_id", taskId + ".gate:" + tool.name).put("task_id", taskId).put("outcome", "RateLimited").put("action", act.put("accepted", false))); return "blocked: " + why; }
        }
        if (cls.equals("none")) return null;
        String scope = cls.equals("once") ? tool.consentScope(args) : null;
        if (scope != null && granted.contains(scope)) return null;   // granted earlier in THIS task for this scope (one app)
        consentAsked.add(tool.name);
        boolean ok = askConsent(ui, tool.name, tool.consentText(args));
        rec.write(new JSONObject().put("turn_id", taskId + ".gate:" + tool.name).put("task_id", taskId).put("outcome", ok ? "ConsentGiven" : "ConsentRequired").put("action", act.put("accepted", ok).put("scope", scope == null ? JSONObject.NULL : scope)));
        if (!ok) return "the user declined " + tool.name;
        if (scope != null) granted.add(scope);
        if (tool.effects.equals("communicate")) { commThisTask++; logComm(tool.name); }
        return null;
    }
    /** Consent must be visible: with Meridian in front the app's own dialog is used; with another app in front the accessibility
     *  overlay shows it on top of that app; with neither, a notification brings the user back to the waiting dialog. */
    boolean askConsent(UI ui, String tool, String text) {
        if (!inFront()) {
            A11y a = A11y.inst; if (a != null) return a.askOnTop("Allow Meridian to do this?", text);
            try { android.app.NotificationManager nm = (android.app.NotificationManager) ctx.getSystemService(Context.NOTIFICATION_SERVICE);
                nm.createNotificationChannel(new android.app.NotificationChannel("meridian_consent", "Approvals", android.app.NotificationManager.IMPORTANCE_HIGH));
                nm.notify(77, new android.app.Notification.Builder(ctx, "meridian_consent").setSmallIcon(android.R.drawable.ic_dialog_alert).setContentTitle("Meridian needs your approval").setContentText(text)
                    .setContentIntent(android.app.PendingIntent.getActivity(ctx, 0, new android.content.Intent(ctx, MainActivity.class).addFlags(android.content.Intent.FLAG_ACTIVITY_REORDER_TO_FRONT), android.app.PendingIntent.FLAG_IMMUTABLE)).setAutoCancel(true).build());
            } catch (Exception e) { ui.log("could not post the approval notification: " + e.getMessage()); }
        }
        return ui.consent(tool, text);
    }
    boolean inFront() { android.app.ActivityManager.RunningAppProcessInfo i = new android.app.ActivityManager.RunningAppProcessInfo(); android.app.ActivityManager.getMyMemoryState(i);
        return i.importance == android.app.ActivityManager.RunningAppProcessInfo.IMPORTANCE_FOREGROUND; }
    int recentComms() { java.io.File f = new java.io.File(ctx.getFilesDir(), "comm_log.txt"); if (!f.exists()) return 0; long cut = System.currentTimeMillis() - 3600_000L; int n = 0;
        try (java.io.BufferedReader r = new java.io.BufferedReader(new java.io.FileReader(f))) { String l; while ((l = r.readLine()) != null) try { if (Long.parseLong(l.split(" ")[0]) > cut) n++; } catch (NumberFormatException ignored) { } } catch (java.io.IOException e) { return COMM_PER_HOUR; }
        return n; }   // an unreadable log counts as the limit reached: fail closed
    void logComm(String tool) { try (java.io.FileWriter w = new java.io.FileWriter(new java.io.File(ctx.getFilesDir(), "comm_log.txt"), true)) { w.write(System.currentTimeMillis() + " " + tool + "\n"); } catch (java.io.IOException ignored) { } }
    void newTask() { granted.clear(); commThisTask = 0; overflows = 0; }

    static String firstJsonObject(String s) {
        int st = s.indexOf('{'); if (st < 0) return null; int depth = 0; boolean inStr = false, esc = false;
        for (int i = st; i < s.length(); i++) { char ch = s.charAt(i);
            if (inStr) { if (esc) esc = false; else if (ch == '\\') esc = true; else if (ch == '"') inStr = false; continue; }
            if (ch == '"') inStr = true; else if (ch == '{') depth++; else if (ch == '}' && --depth == 0) return s.substring(st, i + 1); }
        return null;
    }

    private JSONObject ask(String prompt, int n, boolean clearKv, String grammar, String taskId, String step, UI ui) throws Exception { return ask(prompt, n, clearKv, false, grammar, taskId, step, ui); }
    /** resetHistory: start a new conversation but keep the KV cache (the tool list prefix is reused); agent turns never "think". */
    private JSONObject ask(String prompt, int n, boolean clearKv, boolean resetHistory, String grammar, String taskId, String step, UI ui) throws Exception {
        long t0 = System.currentTimeMillis();
        String reply = chat.ask(prompt, n, clearKv, grammar, resetHistory, Boolean.FALSE, ui::token); String one = firstJsonObject(reply); if (one != null) reply = one;
        ui.log(step + "> " + reply.trim().replace('\n', ' '));
        JSONObject done = chat.lastResult, cond = Regime.snapshot(ctx);
        rec.write(new JSONObject().put("turn_id", taskId + "." + step).put("task_id", taskId).put("model_id", modelId).put("step", step).put("t_start", t0 / 1000.0).put("t_end", System.currentTimeMillis() / 1000.0)
            .put("prompt_tokens", done.optInt("n_prompt")).put("output_tokens", done.optInt("tokens")).put("prefill_ms", done.optDouble("prefill_s") * 1000)
            .put("predicted", JSONObject.NULL).put("observed", new JSONObject().put("tokens_per_s", done.optDouble("tok_s")).put("prefill_tokens_per_s", done.optDouble("prefill_tps")))
            .put("conditions", cond).put("in_regime", cond.getBoolean("in_regime")).put("outcome", "ok"));
        try { return new JSONObject(reply); } catch (JSONException e) { return null; }
    }

    /** Plan -> act (one grammar-constrained call per planned tool, each verified) -> answer. */
    public String run(String task, int maxSteps, UI ui) throws Exception {
        trace.clear(); newTask(); String taskId = "t" + (taskCounter++);
        if (!chat.engine().supportsGrammar()) return "EngineUnsupported: this engine build has no grammar-constrained decoding, which the agent requires.";
        ui.log("decoding: grammar-constrained (plan, calls and answer are valid by construction)");
        String prefix = "You control an Android phone through tools. Work in steps.\nTools:\n" + tools.schemaText()
            + "\nExample task: Save a note to call mom and tell me the battery level. Plan: {\"plan\":[\"notes_add\",\"battery_status\"]}\n"
            + "Example task: How much storage is free? Plan: {\"plan\":[\"storage_free\"]}\n"
            + "Example task: Say hello. Plan: {\"plan\":[]}\n";
        JSONObject plan = ask(prefix + "\nTask: " + task + "\nList the tools needed for the task, in order, each part of the task needing one. Plan:", 60, true, tools.planGrammar(), taskId, "plan", ui);
        JSONArray steps = plan == null ? null : plan.optJSONArray("plan");
        if (steps == null) return "NoProgress: the planning step did not return a plan.";
        List<String> names = new ArrayList<>(); for (int i = 0; i < steps.length() && names.size() < maxSteps; i++) { String n = steps.optString(i); if (tools.registry.containsKey(n)) names.add(n); }
        ui.log("plan: " + names);
        JSONArray results = new JSONArray();
        for (String name : names) {
            Tools.Tool tool = tools.registry.get(name); JSONObject last = null; boolean ok = false; String evidence = "";
            for (int attempt = 0; attempt < 2 && !ok; attempt++) {
                JSONObject call = ask("Task: " + task + "\nNow call the tool " + name + " for the part of the task it serves. " + evidence + "Call:", 80, false, tools.callGrammar(name), taskId, "call:" + name, ui);
                JSONObject args = call == null ? null : call.optJSONObject("args");
                if (args == null) { evidence = "Your previous call was unusable. "; continue; }
                String problem = Tools.validate(tool, args);
                if (problem != null) { evidence = "Error: " + problem + ". "; continue; }
                JSONObject act = new JSONObject().put("tool", name).put("consent", tool.consent);
                String denied = gate(tool, args, taskId, ui);   // consent is never inferred from an earlier task's grant
                if (denied != null) return "ConsentRequired: " + denied + "; nothing was changed by it.";
                JSONObject result; boolean verified; trace.add(name);
                try { result = tool.execute(args); verified = tool.verify(args, result); } catch (Exception e) { result = new JSONObject().put("error", String.valueOf(e.getMessage())); verified = false; }
                ui.log("tool " + name + args + " -> " + result + (verified ? "  [verified: " + tool.postcondition + "]" : "  [NOT verified]"));
                rec.write(new JSONObject().put("turn_id", taskId + ".act:" + name).put("task_id", taskId).put("outcome", verified ? "ok" : "NoProgress").put("action", act.put("accepted", true).put("verified", verified)));
                last = result; ok = verified; if (!verified) evidence = "The last attempt failed its postcondition (" + tool.postcondition + "). ";
            }
            if (!ok) return "NoProgress: '" + name + "' could not be completed and verified; stopped without claiming success.";
            results.put(new JSONObject().put("tool", name).put("result", last));
        }
        JSONObject fin = ask("Task: " + task + "\nAll steps are done. Verified tool results: " + results + "\nWrite the final answer for the user in one sentence, using the facts from the results. Answer:", 90, false, tools.finalGrammar(), taskId, "final", ui);
        String ans = fin == null ? "" : fin.optString("final").trim();
        if (ans.length() < 4) return "NoProgress: the model returned an empty answer. Verified results: " + results;
        return ans;
    }

    /** Observe-act loop (07_AGENT_RUNTIME.md section 2): each step the model emits ONE grammar-constrained JSON object, either a call
     *  to any registered tool (exactly its declared arguments) or a final answer; the tool runs behind the consent gate, its result is
     *  verified against observable device state, and the verified result is shown to the model before the next step.
     *  No few-shot examples: the grammar fixes the format, so no example can leak an answer pattern into an evaluation task
     *  (the plan/act prompt's examples were near-copies of three suite-v1 tasks: the shallow-proxy risk, removed 2026-09-22). */
    // ---------- cancellation: checked before every model call and every tool execution ----------
    private volatile boolean cancelled;   // one Agent per task: never reset, so a Stop that arrives before runLoop starts is kept
    /** Stops the running task at its next model call or tool execution; the current generation is aborted too. */
    public void cancel() { cancelled = true; try { chat.cancel(); } catch (Exception ignored) { } }
    void checkCancel() { if (cancelled) throw new java.util.concurrent.CancellationException("Stopped by the user"); }

    String prefix() throws JSONException {
        return "You are Meridian, an assistant that operates this Android phone for its owner using tools.\nTools (name(arguments): what it does):\n" + tools.schemaText()
            + "Rules:\n- Reply with exactly one JSON object as asked.\n"
            + "- Use tools for anything that needs the phone, the internet or exact facts. One action per reply; you will see its result.\n"
            + "- People often describe a situation or a need instead of naming an action; pick the tool that addresses it.\n"
            + "- Take only the actions the request asks for; do not open apps or save notes unless asked.\n"
            + "- Use only facts from tool results. If something failed or no tool can do it, say so. Never claim an action you did not take.\n"
            + "- When saving or sending a list (places, results, items), put the whole list in one note or message, best first.\n"
            + "- To work inside another app: open it, read the screen, then tap or type using the element ids shown (n5). Each action shows the new screen.\n";
    }
    /** Prefills the stable prefix (instructions + every tool schema) right after the engine loads, so the first task does not pay
     *  for it: the engine keeps it in the KV cache across tasks (patch 0021). Measured cost on the 15R with Qwen3-4B: ~2 minutes. */
    public static void warm(Context c, Chat chat, Tools tools, Recorder rec, String modelId) throws Exception {
        Agent a = new Agent(c, chat, tools, rec, modelId);
        chat.ask(a.prefix() + "Request: hello\nBreak the request into the ordered steps needed to fulfil it.", 1, false, tools.stepsGrammar(), true, Boolean.FALSE, t -> { });
    }
    /** A plan is kept only if its steps are instructions, not bare tool names, and a short single-clause request stays one step
     *  (2026-09-22: a weak planner turned "What's my battery level?" into 8 unrelated tool names, which then ran). */
    List<String> sanePlan(String task, List<String> steps) {
        List<String> ok = new ArrayList<>();
        for (String x : steps) { String t = x.trim(); if (tools.registry.containsKey(t) || t.matches("[a-z_]+") || t.split("\\s+").length < 2) continue; if (!ok.contains(t)) ok.add(t); }
        boolean multi = task.trim().split("\\s+").length > 10 || task.toLowerCase(Locale.ROOT).matches("(?s).*(\\band\\b|\\bthen\\b|\\balso\\b|,|;|\\. ).*");
        if (ok.isEmpty() || (!multi && ok.size() > 1)) { ok.clear(); ok.add(task.trim()); }
        return ok;
    }
    static final java.util.regex.Pattern MULTI = java.util.regex.Pattern.compile("(?i)\\b(each|every|all|them|both|those|these)\\b");

    /** Planner-executor (07_AGENT_RUNTIME.md section 2). Per step: ONE grammar-constrained call that is either a tool call or the step's
     *  written result; a verified single-action step ends there (no "is it done?" call); a follow-up call happens only for multi-item
     *  steps ("each caller"), screen flows, or a failed verification. Each result is verified against device state. Hard caps: tool
     *  calls per step, per task (24), and cancel() at any point. */
    /** Agent tasks in progress; the Governor defers engine-unloading rungs while this is > 0 (D-11). */
    static final java.util.concurrent.atomic.AtomicInteger running = new java.util.concurrent.atomic.AtomicInteger();
    public String runLoop(String task, int maxSteps, UI ui) throws Exception {
        running.incrementAndGet();
        try { return runLoopInner(task, maxSteps, ui); } finally { running.decrementAndGet(); }
    }
    private String runLoopInner(String task, int maxSteps, UI ui) throws Exception {
        trace.clear(); calls.clear(); consentAsked.clear(); newTask(); String taskId = "r" + (taskCounter++);
        if (!chat.engine().supportsGrammar()) return "EngineUnsupported: this engine build has no grammar-constrained decoding, which the agent requires.";
        String prefix = prefix(), grammar = tools.stepGrammar(), finalOnly = tools.finalGrammar();
        checkCancel();
        JSONObject pl = ask(prefix + "Request: " + task + "\nBreak the request into the ordered steps needed to fulfil it, each one short direct instruction. A simple request is one step; a question that needs no phone is one step.",
            160, false, true, tools.stepsGrammar(), taskId, "plan", ui);
        List<String> raw = new ArrayList<>(); JSONArray ja = pl == null ? null : pl.optJSONArray("steps");
        if (ja != null) for (int i = 0; i < ja.length(); i++) raw.add(ja.optString(i));
        List<String> steps = sanePlan(task, raw);
        if (!steps.equals(raw)) ui.log("plan corrected: " + raw + " -> " + steps);
        ui.log("plan: " + steps);
        StringBuilder scratch = new StringBuilder(); int budget = Math.min(24, Math.max(maxSteps, 4 * steps.size())), used = 0;
        for (int si = 0; si < steps.size() && used < budget; si++) {
            ui.log("step " + (si + 1) + "/" + steps.size() + ": " + steps.get(si));
            String st = steps.get(si); boolean multi = MULTI.matcher(st).find();
            String ctxText = prefix + "Request: " + task + "\nPlan: " + steps + "\nResults so far:\n" + (scratch.length() == 0 ? "(none)\n" : scratch) + "Current step " + (si + 1) + ": " + st + "\n";
            String prompt = "Carry out the current step: reply with one tool call, or with final (the step's result in words) if it needs no tool.", lastSig = null, obs = null;
            boolean stepDone = false, fresh = true; int cap = 6;
            for (int k = 0; k < cap && used < budget && !stepDone; k++) {
                checkCancel(); JSONObject out;
                String label = "s" + (si + 1) + "." + (k + 1);
                try { out = fresh ? ask(ctxText + prompt, 200, false, true, grammar, taskId, label, ui)
                        : obs != null ? ask(ctxText + "Current screen:\n" + obs + "\n" + prompt, 160, false, true, grammar, taskId, label, ui)
                        : ask(prompt, 160, false, grammar, taskId, label, ui); }
                catch (IllegalStateException e) { if (!String.valueOf(e.getMessage()).contains("exceeds the session n_ctx")) throw e;
                    overflows++; ui.log("context full (" + overflows + " this task): step " + (si + 1) + " stopped");
                    scratch.append("step ").append(si + 1).append(" (").append(st).append("): stopped, the context was full\n"); break; }
                fresh = false;
                if (out == null) { prompt = "That reply was not usable. Reply with one tool call, or final."; continue; }
                if (out.has("final")) { scratch.append("step ").append(si + 1).append(" (").append(st).append("): ").append(clip(out.optString("final"), 400)).append('\n'); stepDone = true; break; }
                String name = out.optString("tool"); JSONObject args = out.optJSONObject("args"); if (args == null) args = new JSONObject();
                Tools.Tool tool = tools.registry.get(name); String problem = tool == null ? "unknown tool '" + name + "'" : Tools.validate(tool, args);
                if (problem != null) { prompt = "Error: " + problem + ". Reply with a valid tool call, or final."; continue; }
                String sig = name + args; if (sig.equals(lastSig)) { stepDone = true; break; }   // a repeat: nothing new to do for this step
                lastSig = sig;
                JSONObject act = new JSONObject().put("tool", name).put("consent", tool.consentFor(args));
                String denied = gate(tool, args, taskId, ui);   // asked every time (or once per app for this task) and never inferred
                if (denied != null) { scratch.append("step ").append(si + 1).append(" (").append(st).append("): ").append(denied).append("; not done\n"); stepDone = true; break; }
                checkCancel(); used++;
                JSONObject result; boolean verified; trace.add(name);
                try { result = tool.execute(args); verified = tool.verify(args, result); } catch (Exception e) { result = new JSONObject().put("error", String.valueOf(e.getMessage())); verified = false; }
                ui.log("tool " + name + args + " -> " + clip(result.toString(), 300) + (verified ? "  [verified: " + tool.postcondition + "]" : "  [NOT verified]"));
                calls.add(new JSONObject().put("tool", name).put("args", args).put("result", result).put("verified", verified));
                rec.write(new JSONObject().put("turn_id", taskId + ".act:" + name).put("task_id", taskId).put("outcome", verified ? "ok" : "NoProgress").put("action", act.put("accepted", true).put("verified", verified)));
                String res = clip(result.toString(), tool.resultChars()) + (verified ? " (verified)" : " (NOT verified: " + tool.postcondition + " did not hold)");
                if (tool.observes()) { cap = 8; obs = result.optString("screen", null); JSONObject brief = new JSONObject(result.toString()); brief.remove("screen");   // the scratch keeps the action, not the screen
                    scratch.append("step ").append(si + 1).append(" ").append(name).append(args).append(" -> ").append(clip(brief.toString(), 300)).append(verified ? " (verified)" : " (NOT verified)").append('\n'); }
                else { obs = null; scratch.append("step ").append(si + 1).append(" ").append(name).append(args).append(" -> ").append(res).append('\n'); }
                // a verified single action completes a single-item step without another model call
                if (verified && !multi && !tool.observes()) { stepDone = true; break; }
                prompt = "Result of " + name + ": " + (tool.observes() ? "(the screen is shown above) " + clip(new JSONObject(result.toString()).put("screen", "").toString(), 300) : res)
                    + "\nIf the current step (" + st + ") still needs an action, reply with the next tool call; if it is done, reply with final (the step's result in words).";
            }
        }
        if (used >= budget) ui.log("action budget reached (" + budget + " tool calls); stopping");
        checkCancel();
        // the final answer must not be lost to a context overflow after every action succeeded (15R rehearsal, 2026-09-22): retry with
        // each result line clipped, and as a last resort answer from the verified action log itself (counted in context_overflows)
        String fq = "Write the final answer for the user covering every step in at most 5 short sentences, using only these results. Refer to people and places by name; never re-type phone numbers, codes or long lists (the exact list of actions is shown to the user separately). A drafted message or email is drafted, not sent; an opened settings page is opened, not switched.";
        JSONObject fin = null; String ans = "";
        for (int attempt = 0; attempt < 3 && fin == null; attempt++) {
            String res = attempt == 0 ? scratch.toString() : compact(scratch.toString(), attempt == 1 ? 220 : 90);
            try { fin = ask(prefix + "Request: " + task + "\nPlan: " + steps + "\nResults:\n" + res + fq, attempt == 0 ? 200 : 160, false, true, finalOnly, taskId, "final", ui); break; }
            catch (IllegalStateException e) { if (!String.valueOf(e.getMessage()).contains("exceeds the session n_ctx")) throw e; overflows++; ui.log("context full (" + overflows + " this task): shortening the results for the final answer"); }
        }
        ans = fin == null ? salvageFinal(chat.lastText) : fin.optString("final").trim();
        if (ans.length() < 2 && !calls.isEmpty()) ans = "Done. The steps and their verified results are listed below.";
        String out = (ans.length() < 2 ? "NoProgress: empty answer" : ans) + groundTruth();
        remember(task, steps, ans);
        return out;
    }
    /** A final answer cut off by the token cap is still text: keep what was written (the JSON did not close, so it did not parse). */
    static String salvageFinal(String raw) {
        if (raw == null) return ""; int i = raw.indexOf("\"final\""); if (i < 0) return ""; int q = raw.indexOf('"', raw.indexOf(':', i) + 1); if (q < 0) return "";
        String t = raw.substring(q + 1).replaceAll("\"\\s*}\\s*$", "").replace("\\n", "\n").replace("\\\"", "\"").trim();
        return t.isEmpty() ? "" : t + " ...";
    }
    /** Each result line clipped to n characters (the scratchpad for a final answer that would not fit the context). */
    static String compact(String scratch, int n) { StringBuilder b = new StringBuilder(); for (String l : scratch.split("\n")) b.append(clip(l, n)).append('\n'); return b.toString(); }
    static String clip(String s, int n) { return s == null ? "" : s.length() > n ? s.substring(0, n) + " ..." : s; }
    /** Task memory for later "summary" requests (the recall tool): request, plan, answer and verified actions, one JSON line per task. */
    void remember(String task, List<String> steps, String answer) {
        try (java.io.FileWriter w = new java.io.FileWriter(new java.io.File(ctx.getFilesDir(), "memory.jsonl"), true)) {
            JSONArray acts = new JSONArray(); for (JSONObject c : calls) acts.put(new JSONObject().put("tool", c.optString("tool")).put("args", c.optJSONObject("args")).put("verified", c.optBoolean("verified")).put("result", clip(String.valueOf(c.opt("result")), 600)));
            w.write(new JSONObject().put("time", String.format(Locale.ROOT, "%1$tF %1$tR", System.currentTimeMillis())).put("request", task).put("plan", new JSONArray(steps)).put("answer", clip(answer, 800)).put("actions", acts).put("context_overflows", overflows).toString() + "\n");
        } catch (Exception ignored) { }
    }
    /** What actually happened, generated from the verified call log rather than the model's words (the model may overstate). */
    String groundTruth() {
        if (calls.isEmpty()) return "\n[actions taken: none]";
        StringBuilder todo = new StringBuilder();
        for (JSONObject c : calls) { Tools.Tool t = tools.registry.get(c.optString("tool")); String n = t == null || !c.optBoolean("verified") ? null : t.note(c.optJSONObject("result"));
            if (n != null && todo.indexOf(n) < 0) todo.append("\n").append(n); }
        StringBuilder b = new StringBuilder("\n[actions taken: "); for (int i = 0; i < calls.size(); i++) { JSONObject c = calls.get(i); if (i > 0) b.append("; ");
            b.append(c.optString("tool")).append(c.optJSONObject("args") == null || c.optJSONObject("args").length() == 0 ? "" : " " + c.optJSONObject("args")).append(c.optBoolean("verified") ? " (verified)" : " (NOT verified)"); }
        return b.append(']').append(todo).toString();
    }

    /** Ablation: the earlier free-form loop (no grammar, no planning step): one JSON reply per step, validated and verified, but the model chooses
     *  tools, arguments and when to stop on its own. Kept to measure what the plan/act/answer structure buys (baseline in Eval). */
    public String runFreeForm(String task, int maxSteps, UI ui) throws Exception {
        trace.clear(); newTask(); String taskId = "ff" + (taskCounter++);
        String prompt = "You control an Android phone through tools. Reply with exactly one JSON object and nothing else.\n"
            + "A tool call looks like {\"tool\":\"<name>\",\"args\":{<argument name>:<value>}}. Call ONE tool per reply and wait for its result. When the task is done, reply {\"final\":\"<your answer to the user>\"}.\nTools:\n" + tools.schemaText()
            + "\nExample 1\nUser: Remember to call mom.\nAssistant: {\"tool\":\"notes_add\",\"args\":{\"text\":\"call mom\"}}\nTool result: {\"count\":1}\nAssistant: {\"final\":\"Saved the note: call mom.\"}\n"
            + "Example 2\nUser: How much battery do I have?\nAssistant: {\"tool\":\"battery_status\",\"args\":{}}\nTool result: {\"level_pct\":57,\"charging\":false}\nAssistant: {\"final\":\"Your battery is at 57% and not charging.\"}\n"
            + "\nNow the real task.\nUser: " + task + "\nAssistant:";
        boolean first = true; int bad = 0; String lastCall = null;
        for (int step = 1; step <= maxSteps; step++) {
            JSONObject call = ask(prompt, 96, first, null, taskId, "ff" + step, ui); first = false;
            if (call == null) { if (++bad > 2) return "NoProgress: invalid JSON"; prompt = "Error: reply with exactly one JSON object."; continue; }
            if (call.has("final")) { String f = call.optString("final").trim(); return f.length() < 4 ? "NoProgress: empty answer" : f; }
            String name = call.optString("tool", ""); JSONObject args = call.optJSONObject("args"); if (args == null) args = new JSONObject(); Tools.Tool tool = tools.registry.get(name);
            String problem = tool == null ? "unknown tool '" + name + "'" : Tools.validate(tool, args);
            if (problem != null) { if (++bad > 2) return "NoProgress: " + problem; prompt = "Error: " + problem + ". Try again."; continue; }
            String sig = name + args; if (sig.equals(lastCall)) return "NoProgress: repeated call " + name; lastCall = sig;
            String denied = gate(tool, args, taskId, ui); if (denied != null) return "ConsentRequired: " + denied;
            JSONObject result; boolean verified; trace.add(name);
            try { result = tool.execute(args); verified = tool.verify(args, result); } catch (Exception e) { result = new JSONObject().put("error", String.valueOf(e.getMessage())); verified = false; }
            ui.log("tool " + name + args + " -> " + result + (verified ? " [verified]" : " [NOT verified]"));
            prompt = "Tool result: " + result + (verified ? "" : "\nWARNING: postcondition not verified.");
        }
        return "NoProgress: step budget exhausted";
    }
}
