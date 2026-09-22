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

    /** Observe-act loop (07_AGENT_RUNTIME.md section 2): each step the model emits ONE grammar-constrained JSON object, either a call
     *  to any registered tool (exactly its declared arguments) or a final answer; the tool runs behind the consent gate, its result is
     *  verified against observable device state, and the verified result is shown to the model before the next step.
     *  No few-shot examples: the grammar fixes the format, so no example can leak an answer pattern into an evaluation task
     *  (the plan/act prompt's examples were near-copies of three suite-v1 tasks: the shallow-proxy risk, removed 2026-09-22). */
    public String runLoop(String task, int maxSteps, UI ui) throws Exception {
        trace.clear(); calls.clear(); consentAsked.clear(); String taskId = "r" + (taskCounter++);
        if (!chat.engine().supportsGrammar()) return "EngineUnsupported: this engine build has no grammar-constrained decoding, which the agent requires.";
        String prompt = "You are Meridian, an assistant that operates this Android phone for its owner using tools.\nTools (name(arguments): what it does):\n" + tools.schemaText()
            + "Rules:\n- Reply with exactly one JSON object: a tool call {\"tool\": name, \"args\": {...}} or {\"final\": \"answer to the user\"}.\n"
            + "- Use tools for anything that needs the phone or exact facts (time, battery, storage, arithmetic). One action per reply; you will see its result.\n"
            + "- People often describe a situation or a need instead of naming an action; pick the tool that addresses it.\n"
            + "- Take only the actions the request asks for. Most requests need one or two actions; do not open apps or save notes unless asked.\n"
            + "- As soon as the request is satisfied, reply final and say what you did, using only the results. If something failed, say so.\n"
            + "- If no tool can do what is asked, reply final and say what is missing. Never claim an action you did not take.\n"
            + "Request: " + task;
        // Tool selection first (no examples; grammar = registered names only). The action steps may then use only the selected tools
        // plus final: a small model offered every tool at every step opened unrelated apps (maps, camera) on a flashlight request
        // (15R, Qwen2.5-3B, 2026-09-22), and the camera it opened then blocked the torch.
        // Step 0, query rewriting: indirect requests ("it's dark and I can't find my keys") failed selection on a 3B model, so the
        // model first restates the request as one direct instruction; selection then works on the instruction. General, no examples.
        JSONObject in = ask(prompt + "\nFirst, restate what the user wants done as one short direct instruction (if it is only a question, restate the question):", 48, true, tools.intentGrammar(), taskId, "intent", ui);
        String intent = in == null ? task : in.optString("intent", task).trim(); if (intent.isEmpty()) intent = task;
        // (A separate tool-SELECTION step was tried and removed the same day: with an optional list its grammar allowed, both
        //  Qwen2.5-3B and Qwen3-1.7B returned an empty list for every suite-v2 task and then invented answers; 2/14 on the 15R.)
        String grammar = tools.stepGrammar(), finalOnly = tools.finalGrammar();
        // Constrained decisions instead of open-ended ones: the small model kept calling tools on knowledge questions and after the
        // request was already satisfied (suite-v2, Qwen3-1.7B, 15R: 7/14, including notes_clear on "Who wrote Romeo and Juliet?").
        JSONObject need = ask("Instruction: " + intent + "\nDoes carrying this out need the phone or an exact fact from it (the time, battery, storage, a calculation, an app, a setting)? Answer yes or no.", 8, false, tools.yesNoGrammar(), taskId, "need", ui);
        boolean needsTool = need == null || !"no".equals(need.optString("answer"));
        prompt = needsTool ? "Instruction: " + intent + "\nCarry it out: reply with one tool call." : "Answer the request directly: " + task;
        String g = needsTool ? grammar : finalOnly;
        String lastSig = null; int repeats = 0, bad = 0; boolean first = false;
        for (int step = 1; step <= maxSteps; step++) {
            JSONObject out = ask(prompt, 128, first, g, taskId, "step" + step, ui); first = false; g = grammar;
            if (out == null) { if (++bad > 1) return "NoProgress: the model produced no usable step."; prompt = "That reply was not usable. Reply with one tool call or final."; continue; }
            if (out.has("final")) { String f = out.optString("final").trim(); return (f.length() < 2 ? "NoProgress: empty answer" : f) + groundTruth(); }
            String name = out.optString("tool"); JSONObject args = out.optJSONObject("args"); if (args == null) args = new JSONObject();
            Tools.Tool tool = tools.registry.get(name); String problem = tool == null ? "unknown tool '" + name + "'" : Tools.validate(tool, args);
            if (problem != null) { prompt = "Error: " + problem + ". Reply with a valid tool call or final."; continue; }
            String sig = name + args;
            if (sig.equals(lastSig)) {   // a repeat means the model has nothing new to do: answer from the results so far
                prompt = "You already did that. Using only the results you have, write the final answer."; g = finalOnly; continue; }
            lastSig = sig; repeats = 0;
            JSONObject act = new JSONObject().put("tool", name).put("consent", tool.consent);
            if (!tool.consent.equals("none")) consentAsked.add(name);
            if (!tool.consent.equals("none") && !ui.consent(name, args.toString())) {   // consent is asked every time and never inferred
                rec.write(new JSONObject().put("turn_id", taskId + ".gate:" + name).put("task_id", taskId).put("outcome", "ConsentRequired").put("action", act.put("accepted", false)));
                prompt = "The user declined " + name + ". Do not retry it. Reply final and say it was not done."; continue;
            }
            JSONObject result; boolean verified; trace.add(name);
            try { result = tool.execute(args); verified = tool.verify(args, result); } catch (Exception e) { result = new JSONObject().put("error", String.valueOf(e.getMessage())); verified = false; }
            ui.log("tool " + name + args + " -> " + result + (verified ? "  [verified: " + tool.postcondition + "]" : "  [NOT verified]"));
            calls.add(new JSONObject().put("tool", name).put("args", args).put("result", result).put("verified", verified));
            rec.write(new JSONObject().put("turn_id", taskId + ".act:" + name).put("task_id", taskId).put("outcome", verified ? "ok" : "NoProgress").put("action", act.put("accepted", true).put("verified", verified)));
            String obs = "Result of " + name + ": " + result + (verified ? " (verified: " + tool.postcondition + ")" : " (NOT verified: " + tool.postcondition + " did not hold)");
            JSONObject done = ask(obs + "\nIs the instruction (" + intent + ") now completely carried out? Answer yes or no.", 8, false, tools.yesNoGrammar(), taskId, "done" + step, ui);
            if (done != null && "yes".equals(done.optString("answer"))) { prompt = "Write the final answer for the user, using only these results."; g = finalOnly; }
            else prompt = "Take the next action the instruction still needs: one tool call.";
        }
        return "NoProgress: step budget exhausted (" + maxSteps + " steps)." + groundTruth();
    }
    /** What actually happened, generated from the verified call log rather than the model's words (the model may overstate). */
    String groundTruth() {
        if (calls.isEmpty()) return "\n[actions taken: none]";
        StringBuilder b = new StringBuilder("\n[actions taken: "); for (int i = 0; i < calls.size(); i++) { JSONObject c = calls.get(i); if (i > 0) b.append("; ");
            b.append(c.optString("tool")).append(c.optJSONObject("args") == null || c.optJSONObject("args").length() == 0 ? "" : " " + c.optJSONObject("args")).append(c.optBoolean("verified") ? " (verified)" : " (NOT verified)"); }
        return b.append(']').toString();
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
