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

    public Agent(Context c, Chat chat, Tools tools, Recorder rec, String modelId) { ctx = c; this.chat = chat; this.tools = tools; this.rec = rec; this.modelId = modelId; }

    static String firstJsonObject(String s) {
        int st = s.indexOf('{'); if (st < 0) return null; int depth = 0; boolean inStr = false, esc = false;
        for (int i = st; i < s.length(); i++) { char ch = s.charAt(i);
            if (inStr) { if (esc) esc = false; else if (ch == '\\') esc = true; else if (ch == '"') inStr = false; continue; }
            if (ch == '"') inStr = true; else if (ch == '{') depth++; else if (ch == '}' && --depth == 0) return s.substring(st, i + 1); }
        return null;
    }

    private JSONObject ask(String prompt, int n, boolean clearKv, String grammar, String taskId, String step, UI ui) throws Exception {
        long t0 = System.currentTimeMillis();
        String reply = chat.ask(prompt, n, clearKv, grammar, ui::token); String one = firstJsonObject(reply); if (one != null) reply = one;
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
        trace.clear(); String taskId = "t" + (taskCounter++);
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
                if (!tool.consent.equals("none") && !ui.consent(name, args.toString())) {   // consent is never inferred from an earlier grant
                    rec.write(new JSONObject().put("turn_id", taskId + ".gate:" + name).put("task_id", taskId).put("outcome", "ConsentRequired").put("action", act.put("accepted", false)));
                    return "ConsentRequired: you declined '" + name + "'; nothing was changed by it.";
                }
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

    /** Ablation: the earlier free-form loop (no grammar, no planning step): one JSON reply per step, validated and verified, but the model chooses
     *  tools, arguments and when to stop on its own. Kept to measure what the plan/act/answer structure buys (baseline in Eval). */
    public String runFreeForm(String task, int maxSteps, UI ui) throws Exception {
        trace.clear(); String taskId = "ff" + (taskCounter++);
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
            if (!tool.consent.equals("none") && !ui.consent(name, args.toString())) return "ConsentRequired: declined " + name;
            JSONObject result; boolean verified; trace.add(name);
            try { result = tool.execute(args); verified = tool.verify(args, result); } catch (Exception e) { result = new JSONObject().put("error", String.valueOf(e.getMessage())); verified = false; }
            ui.log("tool " + name + args + " -> " + result + (verified ? " [verified]" : " [NOT verified]"));
            prompt = "Tool result: " + result + (verified ? "" : "\nWARNING: postcondition not verified.");
        }
        return "NoProgress: step budget exhausted";
    }
}
