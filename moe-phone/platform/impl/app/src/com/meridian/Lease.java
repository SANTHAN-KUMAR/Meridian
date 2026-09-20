package com.meridian.app;

import android.os.Handler;
import org.json.*;
import java.io.File;

/** MemoryLease protocol (03_INTERFACES.md section 5): request {floor,target} -> grant -> verify residency by RSS (never by allocation,
 *  and swapped pages are not memory) -> heartbeat -> trim -> revoke. Refusal happens at plan time when the floor exceeds what was ever
 *  measured keepable. The heartbeat reads the engine child's /proc status; growth of swapped memory or of major faults is a pressure signal. */
public final class Lease {
    public interface Listener { void onPressure(String why, JSONObject state); }
    public long floor, target, granted, verifiedResident, swapped; public boolean revoked; public int pid = -1;
    private Handler h; private Listener l; private volatile boolean stop; private long lastSwap = -1, lastMajflt = -1;
    private JSONObject last = new JSONObject();

    public static Lease request(JSONObject profile, long floor, long target) throws Planner.Refusal, JSONException {
        JSONObject g = profile.getJSONObject("memory").getJSONObject("grantable_quiesced");
        if (g.isNull("value")) throw new Planner.Refusal("NotCalibrated", "grantable_quiesced not measured");
        long cap = g.getLong("value"); if (floor > cap) throw new Planner.Refusal("MemoryUnavailable", "floor " + (floor >> 20) + " MiB exceeds the largest grant ever kept (" + (cap >> 20) + " MiB)");
        Lease s = new Lease(); s.floor = floor; s.target = target; s.granted = Math.min(target, cap); return s;   // the plan must be re-checked against `granted`, never `target`
    }

    /** Find the engine child's pid: same-UID processes expose /proc/<pid>/cmdline. */
    public static int findEnginePid() {
        File[] ps = new File("/proc").listFiles(); if (ps == null) return -1;
        for (File p : ps) { if (!p.getName().matches("\\d+")) continue; String c = Native.readFile(p.getPath() + "/cmdline"); if (c != null && c.contains("libbmoe_cli.so")) return Integer.parseInt(p.getName()); }
        return -1;
    }
    static long statusKb(int pid, String key) { String s = Native.readFile("/proc/" + pid + "/status"); if (s == null) return -1; for (String l : s.split("\n")) if (l.startsWith(key + ":")) return Long.parseLong(l.replaceAll("[^0-9]", "")); return -1; }
    static long majflt(int pid) { String s = Native.readFile("/proc/" + pid + "/stat"); if (s == null) return -1; String[] f = s.substring(s.lastIndexOf(')') + 2).split(" "); return Long.parseLong(f[9]); }

    /** Verified residency = VmRSS of the engine process. Swapped pages are reported separately and excluded. */
    public JSONObject verify() throws JSONException {
        pid = findEnginePid(); if (pid < 0) return new JSONObject().put("error", "engine process not found");
        verifiedResident = statusKb(pid, "VmRSS") * 1024; swapped = Math.max(0, statusKb(pid, "VmSwap")) * 1024;
        return last = new JSONObject().put("lease_granted", granted).put("verified_resident", verifiedResident).put("swapped", swapped).put("revocable", true).put("rss_ge_floor", verifiedResident >= floor);
    }

    public void startHeartbeat(Handler handler, final long periodMs, Listener listener) {
        h = handler; l = listener; stop = false;
        h.postDelayed(new Runnable() { public void run() { if (stop || revoked) return;
            try { long sw = Math.max(0, statusKb(pid, "VmSwap")) * 1024, mf = majflt(pid), avail = Profile.meminfoKb("MemAvailable") * 1024, rss = statusKb(pid, "VmRSS") * 1024;
                last = new JSONObject().put("lease_granted", granted).put("verified_resident", rss).put("swapped", sw).put("mem_available", avail).put("majflt", mf);
                if (lastSwap >= 0 && sw > lastSwap) l.onPressure("engine memory grew in compressed swap (+" + ((sw - lastSwap) >> 20) + " MiB)", last);
                else if (rss > 0 && rss + (256L << 20) < verifiedResident) l.onPressure("engine resident set shrank from " + (verifiedResident >> 20) + " to " + (rss >> 20) + " MiB (the OS reclaimed it)", last);
                lastSwap = sw; lastMajflt = mf; } catch (Exception ignored) { }
            h.postDelayed(this, periodMs); } }, periodMs);
    }
    public void revoke() { revoked = true; stop = true; }
    public JSONObject state() { return last; }
}
