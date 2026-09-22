package com.meridian.app;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** One agent task as the user sees it: a plan of steps, what is happening now, and the outcome. It is built only from the
 *  agent's own log lines (Agent.UI.log) and its result string, so the screen can never show progress the agent did not make.
 *  Log shapes (agreed with the harness session, 2026-09-22, loop v2): "plan: [..]", "plan corrected: [raw] -> [kept]",
 *  "step i/n: text", "s<N>.<k>> reply" (a tool call or {"final":..}), "final> ..", "tool name{args} -> result [verified: ..]|
 *  [NOT verified]", "context full ...". The older need<N>/done<N>.<k>/text<N> labels are still understood. */
final class UiTask {
    enum Status { STARTING, RUNNING, NEEDS_YOU, DONE, PROBLEM, STOPPED }
    static final class Step { String title; UiKit.StepState state = UiKit.StepState.NEXT; String sub; final List<String> did = new ArrayList<>(); }

    final String request; final long started = System.currentTimeMillis();
    Status status = Status.STARTING; String now = "Getting ready"; final List<Step> steps = new ArrayList<>();
    String answer, actionsTaken, raw; UiHumanize.Problem problem; boolean stopRequested;
    int current = -1;

    UiTask(String request) { this.request = request; }

    static final Pattern STEP = Pattern.compile("^step (\\d+)/(\\d+):\\s*(.*)$", Pattern.DOTALL);
    static final Pattern LABEL = Pattern.compile("^(need|s|text|done)(\\d+)(?:\\.(\\d+))?> (.*)$", Pattern.DOTALL);
    static final Pattern TOOL = Pattern.compile("^tool ([a-z_]+)(\\{.*?\\})? -> (.*?)\\s*(\\[verified[^\\]]*\\]|\\[NOT verified\\])\\s*$", Pattern.DOTALL);
    static final Pattern CTX = Pattern.compile("^context full \\(\\d+ this task\\)(?:: step (\\d+) stopped)?");

    /** Feed one agent log line. Returns true when the visible state changed. */
    boolean onLog(String line) {
        if (line == null) return false; String s = line.trim(); Matcher m;
        if (s.startsWith("plan: [") && s.endsWith("]")) {
            if (steps.isEmpty()) { for (String x : s.substring(7, s.length() - 1).split(", ")) { if (x.trim().isEmpty()) continue; Step st = new Step(); st.title = sentence(x); steps.add(st); } }
            status = Status.RUNNING; now = "Planning done"; return true;
        }
        if (s.startsWith("plan corrected: ") && s.contains("] -> [") && s.endsWith("]")) {   // the harness dropped junk steps: show only what it kept
            String kept = s.substring(s.lastIndexOf("] -> [") + 6, s.length() - 1); steps.clear();
            for (String x : kept.split(", ")) { if (x.trim().isEmpty()) continue; Step st = new Step(); st.title = sentence(x); steps.add(st); }
            status = Status.RUNNING; now = "Planning done"; return true;
        }
        if ((m = STEP.matcher(s)).matches()) {
            int i = Integer.parseInt(m.group(1)) - 1, n = Integer.parseInt(m.group(2));
            if (steps.size() != n) {   // the bracketed plan was split on commas inside a step: rebuild from the authoritative count
                List<Step> old = new ArrayList<>(steps); steps.clear();
                for (int k = 0; k < n; k++) { Step st = new Step(); st.title = k < i && k < old.size() ? old.get(k).title : "Step " + (k + 1); if (k < i) st.state = UiKit.StepState.DONE; steps.add(st); }
            }
            for (int k = 0; k < i && k < steps.size(); k++) if (steps.get(k).state == UiKit.StepState.NOW) steps.get(k).state = UiKit.StepState.DONE;
            Step st = steps.get(i); st.title = sentence(m.group(3)); st.state = UiKit.StepState.NOW; current = i;
            status = Status.RUNNING; now = st.title; return true;
        }
        if ((m = LABEL.matcher(s)).matches()) {
            String kind = m.group(1), body = m.group(4); int i = Integer.parseInt(m.group(2)) - 1; Step st = at(i);
            if (st == null) return false;
            if (kind.equals("need") && body.contains("\"no\"")) { st.sub = "Worked this out myself"; now = "Thinking it through"; return true; }
            if (kind.equals("need")) { now = "Choosing what to do"; return true; }
            if (kind.equals("s")) { if (body.contains("\"final\"")) { st.state = UiKit.StepState.DONE; now = "Writing it up"; return true; }
                String tool = field(body, "tool"); if (tool != null) now = verbing(tool); return tool != null; }
            if (kind.equals("text") || kind.equals("done") && body.contains("\"yes\"")) { st.state = UiKit.StepState.DONE; if (kind.equals("text") && st.sub == null) st.sub = "Worked this out myself"; return true; }
            return false;
        }
        if (s.startsWith("final> ")) { now = "Writing up the result"; for (Step st : steps) if (st.state == UiKit.StepState.NOW) st.state = UiKit.StepState.DONE; return true; }
        if ((m = TOOL.matcher(s)).matches()) {
            Step st = at(current); boolean ok = !m.group(4).startsWith("[NOT");
            String what = done(m.group(1)) + (ok ? "" : " (couldn't confirm it worked)");
            if (st != null) { st.did.add(what); if (!ok) st.sub = "I couldn't confirm this worked"; }
            return true;
        }
        if ((m = CTX.matcher(s)).find()) {
            if (m.group(1) != null) { Step st = at(Integer.parseInt(m.group(1)) - 1); if (st != null) { st.state = UiKit.StepState.FAILED; st.sub = "That was too much to hold at once"; } }
            return true;
        }
        return false;
    }
    /** The agent's return value: an answer plus the harness-written "[actions taken: ...]" block, or a typed failure. */
    void onResult(String result) {
        raw = result; String r = result == null ? "" : result;
        int k = r.indexOf("\n[actions taken:");
        answer = (k >= 0 ? r.substring(0, k) : r).trim(); actionsTaken = k >= 0 ? r.substring(k + 1).trim() : null;
        if (stopRequested) { status = Status.STOPPED; now = "Stopped"; }
        else if (answer.startsWith("NoProgress") || answer.startsWith("EngineUnsupported") || answer.startsWith("ConsentRequired")) {
            problem = UiHumanize.of(answer); status = Status.PROBLEM; now = problem.title; answer = null;
        } else { status = Status.DONE; now = "Done"; }
        for (Step st : steps) if (st.state == UiKit.StepState.NOW) st.state = status == Status.DONE ? UiKit.StepState.DONE : UiKit.StepState.SKIPPED;
        if (status != Status.DONE) for (Step st : steps) if (st.state == UiKit.StepState.NEXT) st.state = UiKit.StepState.SKIPPED;
    }
    void onFailure(String message) {
        raw = message; problem = UiHumanize.of(message); status = stopRequested ? Status.STOPPED : Status.PROBLEM; now = stopRequested ? "Stopped" : problem.title;
        for (Step st : steps) if (st.state == UiKit.StepState.NOW || st.state == UiKit.StepState.NEXT) st.state = UiKit.StepState.SKIPPED;
    }
    int doneCount() { int n = 0; for (Step s : steps) if (s.state == UiKit.StepState.DONE) n++; return n; }
    boolean finished() { return status == Status.DONE || status == Status.PROBLEM || status == Status.STOPPED; }
    long elapsedS() { return (System.currentTimeMillis() - started) / 1000; }

    private Step at(int i) { return i >= 0 && i < steps.size() ? steps.get(i) : null; }
    private static String field(String json, String name) { Matcher m = Pattern.compile("\"" + name + "\"\\s*:\\s*\"([^\"]*)\"").matcher(json); return m.find() ? m.group(1) : null; }
    static String sentence(String s) { s = s.trim(); if (s.startsWith("\"") && s.endsWith("\"") && s.length() > 1) s = s.substring(1, s.length() - 1); return s.isEmpty() ? s : Character.toUpperCase(s.charAt(0)) + s.substring(1); }

    // ---------- tool names as plain words ----------
    static final Map<String, String[]> WORDS = new HashMap<>();   // tool -> {while doing, after doing}
    static void w(String tool, String doing, String done) { WORDS.put(tool, new String[]{doing, done}); }
    static {
        w("battery_status", "Checking the battery", "Checked the battery"); w("storage_free", "Checking free storage", "Checked free storage");
        w("notes_add", "Saving a note", "Saved a note"); w("notes_list", "Reading your notes", "Read your notes"); w("notes_clear", "Clearing notes", "Cleared notes");
        w("calculate", "Working out the numbers", "Did the maths"); w("calendar_event", "Adding to your calendar", "Added a calendar event");
        w("call", "Starting a call", "Started a call"); w("copy_text", "Copying text", "Copied text"); w("device_info", "Reading phone details", "Read phone details");
        w("find_contact", "Looking up a contact", "Found a contact"); w("flashlight", "Switching the flashlight", "Switched the flashlight");
        w("media_control", "Controlling playback", "Controlled playback"); w("missed_calls", "Checking missed calls", "Checked missed calls");
        w("my_location", "Finding where you are", "Found your location"); w("navigate", "Opening directions", "Opened directions");
        w("open_app", "Opening an app", "Opened an app"); w("open_settings", "Opening settings", "Opened settings"); w("open_url", "Opening a link", "Opened a link");
        w("play_music", "Starting music", "Started music"); w("quick_panel", "Opening quick settings", "Opened quick settings"); w("read_page", "Reading the page", "Read the page");
        w("recall", "Remembering earlier tasks", "Looked back at earlier tasks"); w("send_email", "Preparing an email", "Prepared an email");
        w("send_message", "Preparing a message", "Prepared a message"); w("set_alarm", "Setting an alarm", "Set an alarm"); w("set_brightness", "Adjusting brightness", "Adjusted brightness");
        w("set_timer", "Setting a timer", "Set a timer"); w("set_volume", "Adjusting volume", "Adjusted volume"); w("share_text", "Sharing text", "Shared text");
        w("take_photo", "Opening the camera", "Opened the camera"); w("time_now", "Checking the time", "Checked the time");
        w("web_results", "Reading search results", "Read search results"); w("web_search", "Searching the web", "Searched the web");
    }
    static String verbing(String tool) { String[] x = WORDS.get(tool); return x != null ? x[0] : "Using " + tool.replace('_', ' '); }
    static String done(String tool) { String[] x = WORDS.get(tool); return x != null ? x[1] : "Used " + tool.replace('_', ' '); }
}
