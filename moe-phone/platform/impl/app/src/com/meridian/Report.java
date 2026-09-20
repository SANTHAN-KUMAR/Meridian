package com.meridian.app;

import org.json.*;

/** Plain-language capability report generated FROM the profile JSON (02_ARCHITECTURE.md section 9). No number is typed here. */
public final class Report {
    static String fmt(JSONObject m, String unit, double scale, int digits) throws JSONException {
        if (m == null || m.isNull("value")) return "not measured (" + (m == null ? "?" : m.optString("source")) + ")";
        double v = m.getDouble("value") / scale; String s = String.format("%." + digits + "f%s [%s", v, unit, m.getString("provenance"));
        JSONArray iv = m.optJSONArray("interval");
        if (iv != null && iv.length() == 2) s += String.format(", observed %." + digits + "f-%." + digits + "f", iv.getDouble(0) / scale, iv.getDouble(1) / scale);
        return s + "]";
    }
    public static String render(JSONObject p) throws JSONException {
        StringBuilder b = new StringBuilder(); JSONObject id = p.getJSONObject("identity"), mem = p.getJSONObject("memory"), st = p.getJSONObject("storage"), cond = p.getJSONObject("validity").getJSONObject("conditions");
        b.append(id.getString("manufacturer")).append(' ').append(id.getString("model")).append("  -  ").append(id.getString("soc_vendor")).append(' ').append(id.getString("soc_model")).append("\nAndroid ").append(id.getString("os_version")).append(" (API ").append(id.getString("api_level")).append("), rooted: ").append(id.getBoolean("rooted")).append("\n\n");
        b.append("CPU\n"); JSONArray cl = p.getJSONObject("cpu").getJSONArray("clusters");
        for (int i = 0; i < cl.length(); i++) { JSONObject c = cl.getJSONObject(i); b.append("  ").append(c.getString("name")).append(": cores ").append(c.getJSONArray("core_ids")).append(", ").append(String.format("%.2f GHz", c.getLong("max_khz") / 1e6)).append(", ISA ").append(c.getJSONArray("isa_features").toString().replace("\"", "")).append('\n'); }
        JSONObject cm = p.getJSONObject("cpu").getJSONObject("recommended_compute_mask");
        b.append("  best thread placement: ").append(cm.isNull("value") ? "not calibrated" : cm.get("value") + " [" + cm.getString("provenance") + "]").append("\n\nMemory\n");
        b.append("  total ").append(String.format("%.1f GiB", mem.getLong("total") / 1073741824.0)).append(", available now ").append(fmt(mem.getJSONObject("available_at_probe"), " GiB", 1073741824.0, 1)).append('\n');
        b.append("  one process can keep (quiesced): ").append(fmt(mem.getJSONObject("grantable_quiesced"), " GiB", 1073741824.0, 2)).append('\n');
        b.append("  ...with a foreground app: ").append(fmt(mem.getJSONObject("grantable_foreground"), "", 1, 0)).append("  (the number an agent lives on; never measured yet)\n");
        b.append("  DRAM read: ").append(fmt(mem.getJSONObject("dram_read_gbps"), " GB/s", 1, 2)).append("\n\nStorage (").append(st.getString("filesystem")).append(", ").append(st.getString("path")).append(")\n");
        b.append("  free ").append(String.format("%.1f GiB", st.getLong("free_bytes") / 1073741824.0)).append(", direct I/O: ").append(st.getBoolean("direct_io_supported")).append(", efficient request size ").append(fmt(st.getJSONObject("efficient_request_size"), " KiB", 1024, 0)).append('\n');
        b.append("  sequential write: ").append(fmt(st.getJSONObject("sequential_write_mbps"), " MB/s", 1, 0)).append('\n');
        JSONArray rr = st.getJSONArray("random_read");
        for (int i = 0; i < rr.length(); i++) { JSONObject r = rr.getJSONObject(i); if (r.getInt("threads") == 8 || i == rr.length() - 1) b.append("    ").append(r.getLong("size_bytes") / 1024).append(" KiB x").append(r.getInt("threads")).append(": ").append(fmt(r.getJSONObject("mbps"), " MB/s", 1, 0)).append('\n'); }
        b.append("\nConditions: power ").append(cond.getString("power")).append(", ").append(cond.getString("wakefulness")).append(", foreground ").append(cond.getString("foreground")).append(", battery ").append(cond.optInt("battery_pct")).append("%\n");
        if (!cond.getBoolean("in_regime")) b.append("  NOT the deployment regime: every figure above is labelled [prior], not [measured]. Re-run unplugged with the screen on and this app in front.\n");
        int rej = p.getJSONObject("validity").getInt("rejected_runs"); if (rej > 0) b.append("  ").append(rej).append(" contaminated probe row(s) rejected, not averaged.\n");
        b.append("\nNot measured in this build: per-cluster compute throughput, thermal derate, foreground memory grant, NPU. No tokens/s is promised for any model; see the Models tab for what can be said.\n");
        return b.toString();
    }
}
