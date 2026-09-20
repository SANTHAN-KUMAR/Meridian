package com.meridian.app;

import android.content.*;
import android.os.*;
import org.json.*;
import java.io.*;
import java.util.*;

/** ToolSpec registry (03_INTERFACES.md section 7.1). Every tool declares effects, reversibility, consent class and a
 *  named postcondition that is checked against observable state -- not against the tool's own return value. */
public final class Tools {
    public static abstract class Tool {
        public final String name, description, effects, consent, postcondition; public final boolean reversible; public final JSONObject params;
        Tool(String n, String d, String eff, boolean rev, String consent, String post, JSONObject params) { name = n; description = d; effects = eff; reversible = rev; this.consent = consent; postcondition = post; this.params = params; }
        abstract JSONObject execute(JSONObject args) throws Exception;
        abstract boolean verify(JSONObject args, JSONObject result) throws Exception;   // observable-state check
    }
    private final Context ctx; private final File notes; public final Map<String, Tool> registry = new LinkedHashMap<>();

    static JSONObject obj(String... kv) throws JSONException { JSONObject o = new JSONObject(), props = new JSONObject(); JSONArray req = new JSONArray();
        for (int i = 0; i < kv.length; i += 2) { props.put(kv[i], new JSONObject().put("type", kv[i + 1])); req.put(kv[i]); }
        return o.put("type", "object").put("properties", props).put("required", req); }

    List<String> readNotes() throws IOException { List<String> l = new ArrayList<>(); if (!notes.exists()) return l;
        try (BufferedReader r = new BufferedReader(new FileReader(notes))) { String s; while ((s = r.readLine()) != null) if (!s.isEmpty()) l.add(s); } return l; }
    void writeNotes(List<String> l) throws IOException { try (FileWriter w = new FileWriter(notes, false)) { for (String s : l) w.write(s.replace('\n', ' ') + "\n"); } }

    public Tools(Context c) throws JSONException {
        ctx = c.getApplicationContext(); notes = new File(ctx.getFilesDir(), "notes.txt");
        add(new Tool("battery_status", "Read the battery level (percent) and whether the phone is charging.", "read", true, "none", "level is 0..100 and matches the OS battery service", obj()) {
            JSONObject execute(JSONObject a) throws Exception { Intent b = ctx.registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
                return new JSONObject().put("level_pct", b.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) * 100 / b.getIntExtra(BatteryManager.EXTRA_SCALE, 100)).put("charging", b.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { int l = r.getInt("level_pct"); BatteryManager bm = (BatteryManager) ctx.getSystemService(Context.BATTERY_SERVICE);
                return l >= 0 && l <= 100 && Math.abs(bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY) - l) <= 2; } });
        add(new Tool("storage_free", "Read free internal storage in gigabytes.", "read", true, "none", "value equals StatFs free bytes within 1%", obj()) {
            JSONObject execute(JSONObject a) throws Exception { return new JSONObject().put("free_gb", Math.round(ctx.getFilesDir().getUsableSpace() / 1e8) / 10.0); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { double fresh = ctx.getFilesDir().getUsableSpace() / 1e9; return Math.abs(fresh - r.getDouble("free_gb")) <= 0.01 * fresh + 0.1; } });
        add(new Tool("notes_add", "Save a short note in the app's private notes.", "write", true, "none", "the note text appears in the notes store after the call", obj("text", "string")) {
            JSONObject execute(JSONObject a) throws Exception { List<String> l = readNotes(); l.add(a.getString("text")); writeNotes(l); return new JSONObject().put("count", l.size()); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { return readNotes().contains(a.getString("text").replace('\n', ' ')); } });
        add(new Tool("notes_list", "List all saved notes.", "read", true, "none", "returned count equals notes in the store", obj()) {
            JSONObject execute(JSONObject a) throws Exception { return new JSONObject().put("notes", new JSONArray(readNotes())); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { return r.getJSONArray("notes").length() == readNotes().size(); } });
        add(new Tool("notes_clear", "Delete ALL saved notes. Cannot be undone.", "write", false, "every_time", "the notes store is empty", obj()) {
            JSONObject execute(JSONObject a) throws Exception { int n = readNotes().size(); writeNotes(new ArrayList<String>()); return new JSONObject().put("deleted", n); }
            boolean verify(JSONObject a, JSONObject r) throws Exception { return readNotes().isEmpty(); } });
    }
    private void add(Tool t) { registry.put(t.name, t); }

    public String schemaText() throws JSONException {
        StringBuilder sb = new StringBuilder();
        for (Tool t : registry.values()) {
            JSONObject props = t.params.getJSONObject("properties"); StringBuilder sig = new StringBuilder();
            Iterator<String> it = props.keys(); while (it.hasNext()) { String k = it.next(); if (sig.length() > 0) sig.append(", "); sig.append(k).append(": ").append(props.getJSONObject(k).getString("type")).append(" (required)"); }
            sb.append("- ").append(t.name).append("(").append(sig).append("): ").append(t.description).append('\n');
        }
        return sb.toString();
    }
    private static final String BASE_RULES =
        "ws ::= [ ]?\n"
      + "string ::= \"\\\"\" ([^\"\\\\\\x7F\\x00-\\x1F] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" [0-9a-fA-F]{4}))* \"\\\"\"\n"
      + "final ::= \"{\\\"final\\\":\" ws string \"}\"\n";

    /** GBNF body for one tool call; string-typed arguments only (anything else is refused, not guessed). */
    private String callRule(String rule, Tool t) throws JSONException {
        JSONObject props = t.params.getJSONObject("properties"); StringBuilder body = new StringBuilder();
        body.append("\"{\\\"tool\\\":\\\"").append(t.name).append("\\\",\\\"args\\\":{\"");
        boolean firstArg = true; Iterator<String> it = props.keys();
        while (it.hasNext()) { String k = it.next();
            if (!props.getJSONObject(k).getString("type").equals("string")) throw new JSONException("tool " + t.name + ": grammar generation supports string arguments only (argument '" + k + "')");
            body.append(firstArg ? " " : " \",\" ").append("\"\\\"").append(k).append("\\\":\" ws string"); firstArg = false; }
        return rule + " ::= " + body + " \"}}\"\n";
    }
    /** Grammar generated from the registered ToolSpecs (closes stub PL-E7): exactly one call to `name`, with exactly its declared arguments. */
    public String callGrammar(String name) throws JSONException { Tool t = registry.get(name); return BASE_RULES + callRule("call", t) + "root ::= call\n"; }
    /** Only {"final": <string>}. */
    public String finalGrammar() { return BASE_RULES + "root ::= final\n"; }
    /** {"plan":[<registered tool names in order>]} -- the TaskPlanner step. */
    public String planGrammar() {
        StringBuilder alts = new StringBuilder(); for (String n : registry.keySet()) { if (alts.length() > 0) alts.append(" | "); alts.append("\"\\\"").append(n).append("\\\"\""); }
        return "name ::= " + alts + "\nroot ::= \"{\\\"plan\\\":[\" (name (\",\" name){0,4})? \"]}\"\n";
    }

    /** Validate args against the tool's declared schema (required keys, primitive types). Returns null when valid. */
    public static String validate(Tool t, JSONObject args) throws JSONException {
        JSONArray req = t.params.getJSONArray("required"); JSONObject props = t.params.getJSONObject("properties");
        for (int i = 0; i < req.length(); i++) { String k = req.getString(i); if (!args.has(k)) return "missing argument '" + k + "'";
            String ty = props.getJSONObject(k).getString("type"); Object v = args.get(k);
            if (ty.equals("string") && !(v instanceof String)) return "argument '" + k + "' must be a string"; }
        return null;
    }
}
