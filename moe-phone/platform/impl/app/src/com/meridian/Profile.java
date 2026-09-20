package com.meridian.app;

import android.content.Context;
import android.os.Build;
import android.os.StatFs;
import org.json.*;
import java.io.File;
import java.util.*;

/** Layer 0 device profiler: T0 static inventory + T1 fast probes, run as the app's own UID.
 *  Produces the DeviceProfile of platform/03_INTERFACES.md section 1. Anything not measured is provenance "unknown"
 *  with a named reason; nothing is defaulted. Intervals are the real min/max over repeated probe runs. */
public final class Profile {
    public static final String PROBE_SUITE = "meridian-app-0.1.0";
    public interface Progress { void step(String what); }

    static JSONObject meas(Object value, boolean inRegime, JSONObject cond, double lo, double hi, String source) throws JSONException {
        JSONObject m = new JSONObject().put("value", value).put("provenance", inRegime ? "measured" : "prior")
            .put("confidence", inRegime ? 0.7 : 0.4).put("source", source).put("conditions", cond);
        m.put("interval", new JSONArray().put(lo).put(hi));   // observed spread over repeats
        return m;
    }
    static JSONObject unknown(String source) throws JSONException {
        return new JSONObject().put("value", JSONObject.NULL).put("provenance", "unknown").put("confidence", 0.0).put("source", source);
    }
    static String prop(String key) {
        try { Process p = new ProcessBuilder("/system/bin/getprop", key).start();
            try (java.util.Scanner s = new java.util.Scanner(p.getInputStream()).useDelimiter("\\A")) { return s.hasNext() ? s.next().trim() : ""; } }
        catch (Exception e) { return ""; }
    }

    // ---------------- T0 ----------------
    public static JSONObject identity() throws JSONException {
        boolean rooted = false;
        try { Process p = new ProcessBuilder("su", "-c", "id -u").start(); p.getOutputStream().close();
            try (java.util.Scanner s = new java.util.Scanner(p.getInputStream())) { rooted = s.hasNext() && s.next().trim().equals("0"); } }
        catch (Exception e) { rooted = false; }   // verified by executing, not by finding a binary
        String soc = Build.VERSION.SDK_INT >= 31 ? Build.SOC_MODEL : "UNKNOWN";
        return new JSONObject().put("manufacturer", Build.MANUFACTURER).put("model", Build.MODEL)
            .put("soc_vendor", Build.VERSION.SDK_INT >= 31 ? Build.SOC_MANUFACTURER : "UNKNOWN").put("soc_model", soc)
            .put("board", Build.BOARD).put("os_version", Build.VERSION.RELEASE).put("api_level", String.valueOf(Build.VERSION.SDK_INT))
            .put("build_fingerprint", Build.FINGERPRINT).put("rooted", rooted).put("verified_boot", prop("ro.boot.verifiedbootstate"));
    }

    /** Cores grouped by (CPU part, max kHz); features = intersection over the cluster's cores. */
    public static JSONArray cpuClusters() throws JSONException {
        String ci = Native.readFile("/proc/cpuinfo");
        if (ci == null) return new JSONArray();
        Map<Integer, String> part = new TreeMap<>(); Map<Integer, Set<String>> feats = new HashMap<>(); int core = -1;
        for (String l : ci.split("\n")) {
            if (l.startsWith("processor")) core = Integer.parseInt(l.split(":")[1].trim());
            else if (l.startsWith("CPU part") && core >= 0) part.put(core, l.split(":")[1].trim());
            else if (l.startsWith("Features") && core >= 0) feats.put(core, new TreeSet<>(Arrays.asList(l.split(":")[1].trim().split("\\s+"))));
        }
        Map<String, List<Integer>> groups = new LinkedHashMap<>(); Map<Integer, Long> khz = new HashMap<>();
        for (int c : part.keySet()) {
            String f = Native.readFile("/sys/devices/system/cpu/cpu" + c + "/cpufreq/cpuinfo_max_freq");
            long k = f == null ? 0 : Long.parseLong(f.trim()); khz.put(c, k);
            groups.computeIfAbsent(part.get(c) + "@" + k, x -> new ArrayList<>()).add(c);
        }
        List<List<Integer>> ordered = new ArrayList<>(groups.values());
        Collections.sort(ordered, (a, b) -> Long.compare(khz.get(a.get(0)), khz.get(b.get(0))));
        String[] names = {"efficiency", "performance", "prime"};
        JSONArray out = new JSONArray();
        for (int i = 0; i < ordered.size(); i++) {
            List<Integer> ids = ordered.get(i); Set<String> common = null;
            for (int c : ids) { Set<String> f = feats.get(c); if (f == null) continue; if (common == null) common = new TreeSet<>(f); else common.retainAll(f); }
            out.put(new JSONObject().put("name", i < names.length ? names[i] : "cluster" + i).put("core_ids", new JSONArray(ids))
                .put("max_khz", khz.get(ids.get(0))).put("isa_features", new JSONArray(common == null ? new ArrayList<String>() : new ArrayList<>(common)))
                .put("cpu_part", part.get(ids.get(0))).put("matmul_gbps", unknown("compute_probe:not_implemented (PL-E11)")));
        }
        return out;
    }

    static long meminfoKb(String key) {
        String m = Native.readFile("/proc/meminfo"); if (m == null) return -1;
        for (String l : m.split("\n")) if (l.startsWith(key + ":")) return Long.parseLong(l.replaceAll("[^0-9]", ""));
        return -1;
    }

    /** Filesystem type of the mount backing `path` (longest matching mount point), or "UNKNOWN". */
    static String filesystemOf(String path) {
        String m = Native.readFile("/proc/mounts"); if (m == null) return "DENIED";
        String best = "UNKNOWN"; int bestLen = -1;
        for (String l : m.split("\n")) { String[] p = l.split(" "); if (p.length > 2 && path.startsWith(p[1]) && p[1].length() > bestLen) { best = p[2]; bestLen = p[1].length(); } }
        return best;
    }

    // ---------------- T1 ----------------
    static List<Map<String, String>> csv(String text, String headerPrefix) {
        List<Map<String, String>> rows = new ArrayList<>(); String[] hdr = null;
        for (String l : text.split("\n")) {
            if (l.startsWith("#") || l.trim().isEmpty()) continue;
            if (l.startsWith(headerPrefix)) { hdr = l.split(","); continue; }
            if (hdr == null) continue;
            String[] v = l.split(","); if (v.length != hdr.length) continue;
            Map<String, String> r = new HashMap<>(); for (int i = 0; i < hdr.length; i++) r.put(hdr[i], v[i]); rows.add(r);
        }
        return rows;
    }
    static double[] stats(List<Double> v) { Collections.sort(v); return new double[]{v.get(v.size() / 2), v.get(0), v.get(v.size() - 1)}; }

    public static JSONObject run(Context ctx, File modelDir, boolean includeMemGrant, Progress prog) throws Exception {
        modelDir.mkdirs();
        Regime regime = new Regime(ctx); regime.begin();
        JSONObject idn = identity(); JSONArray clusters = cpuClusters();
        long memTotal = meminfoKb("MemTotal") * 1024, memAvail = meminfoKb("MemAvailable") * 1024;
        StatFs sf = new StatFs(modelDir.getAbsolutePath());
        JSONObject validity = new JSONObject(); int rejected = 0; JSONArray reasons = new JSONArray();
        String base = new File(ctx.getFilesDir(), "probe").getAbsolutePath();

        prog.step("device nodes (as this app)");
        String dev = Native.present(ctx, "devprobe") ? Native.run(ctx, "devprobe", new ArrayList<String>(), 15000, null, true) : "";

        prog.step("DRAM bandwidth (3 repeats)");
        List<Double> dram = new ArrayList<>(); String dramWhy = null;
        try {
            StringBuilder err = new StringBuilder();
            String o = Native.run(ctx, "dramprobe", Arrays.asList("--mb", "512", "--threads", "1,2,4,6,8", "--seconds", "1.5", "--repeats", "3", "--out", base + "_dram.csv"), 120000, err, true);
            for (Map<String, String> r : csv(o, "repeat")) {
                if (Double.parseDouble(r.get("min_cpu_over_wall")) < 0.90) { rejected++; reasons.put("descheduled: dramprobe threads=" + r.get("threads") + " ratio=" + r.get("min_cpu_over_wall")); }
                else dram.add(Double.parseDouble(r.get("GBps")));
            }
            if (dram.isEmpty()) dramWhy = "no clean rows: " + err.toString().trim();
        } catch (Exception e) { dramWhy = e.getMessage(); }

        prog.step("storage random-read curve on the model path (3 repeats)");
        String ufsWhy = null; List<Map<String, String>> ufs = new ArrayList<>(); String fsType = filesystemOf(modelDir.getAbsolutePath());
        long freeBytes = sf.getAvailableBytes(); File probeFile = new File(modelDir, ".meridian_probe.bin");
        if (freeBytes < (1L << 30) + (512L << 20)) ufsWhy = "insufficient free space (" + freeBytes + " B) for a 1 GiB probe file";
        else try {
            String o = Native.run(ctx, "ufsbench", Arrays.asList("--file", probeFile.getAbsolutePath(), "--size-mb", "1024", "--create", "--seconds", "1.5", "--repeats", "3",
                "--sizes-kb", "4,64,256,1024,4096", "--threads", "1,4,8", "--modes", "buffered", "--patterns", "rand", "--out", base + "_ufs.csv"), 400000, null, false);
            ufs = csv(o, "repeat");
        } catch (Exception e) { ufsWhy = e.getMessage(); }
        boolean directOk = false;
        if (ufsWhy == null) try {
            String o = Native.run(ctx, "ufsbench", Arrays.asList("--file", probeFile.getAbsolutePath(), "--size-mb", "1024", "--seconds", "1", "--repeats", "1",
                "--sizes-kb", "4,1024", "--threads", "4", "--modes", "direct", "--patterns", "rand", "--out", base + "_ufsd.csv"), 60000, null, false);
            for (Map<String, String> r : csv(o, "repeat")) if (r.get("mode").equals("direct") && r.get("errors").equals("0")) directOk = true;
        } catch (Exception ignored) { directOk = false; }
        probeFile.delete();

        prog.step("sequential write (fsync included)");
        List<Double> wr = new ArrayList<>(); String wrWhy = null;
        if (freeBytes < (1L << 30) + (512L << 20)) wrWhy = "insufficient free space";
        else try {
            String o = Native.run(ctx, "wrbench", Arrays.asList("--file", new File(modelDir, ".meridian_wr.bin").getAbsolutePath(), "--mb", "1024", "--repeats", "3"), 120000, null, false);
            for (Map<String, String> r : csv(o, "repeat")) if (r.get("errors").equals("0")) wr.add(Double.parseDouble(r.get("MBps")));
            if (wr.isEmpty()) wrWhy = "all repeats errored";
        } catch (Exception e) { wrWhy = e.getMessage(); }

        List<Double> grants = new ArrayList<>(); String grantWhy = null;
        if (includeMemGrant) {
            for (int i = 0; i < 2; i++) {
                prog.step("memory grant probe " + (i + 1) + "/2 (may close background apps)");
                try {
                    String o = Native.run(ctx, "memprobe", Arrays.asList("--oom-adj", "0"), 120000, null, false);
                    java.util.regex.Matcher m = java.util.regex.Pattern.compile("max_VmRSS_MB=(\\d+)").matcher(o);
                    if (m.find()) grants.add(Double.parseDouble(m.group(1)) * 1048576.0); else grantWhy = "no max_VmRSS in output";
                } catch (Exception e) { grantWhy = e.getMessage(); }
            }
        } else grantWhy = "skipped by user (may close background apps)";

        JSONObject cond = regime.end(); boolean in = cond.getBoolean("in_regime");
        prog.step("assembling profile");

        JSONArray rr = new JSONArray(); JSONObject knee = null;
        if (!ufs.isEmpty()) {
            TreeSet<Integer> sizes = new TreeSet<>(), threads = new TreeSet<>();
            for (Map<String, String> r : ufs) { sizes.add(Integer.parseInt(r.get("size_kb"))); threads.add(Integer.parseInt(r.get("threads"))); }
            for (int s : sizes) for (int t : threads) {
                List<Double> v = new ArrayList<>(); String p50 = "0", p99 = "0";
                for (Map<String, String> r : ufs) if (Integer.parseInt(r.get("size_kb")) == s && Integer.parseInt(r.get("threads")) == t) { v.add(Double.parseDouble(r.get("MBps"))); p50 = r.get("lat_p50_us"); p99 = r.get("lat_p99_us"); }
                if (v.isEmpty()) continue; double[] st = stats(v);
                rr.put(new JSONObject().put("size_bytes", s * 1024L).put("threads", t).put("mbps", meas(st[0], in, cond, st[1], st[2], "ufsbench n_repeats=" + v.size()))
                    .put("p50_us", Double.parseDouble(p50)).put("p99_us", Double.parseDouble(p99)));
            }
            // knee: per repeat, smallest size within 90% of that repeat's bulk rate at the max thread count
            int maxT = threads.last(); List<Double> knees = new ArrayList<>();
            TreeSet<String> reps = new TreeSet<>(); for (Map<String, String> r : ufs) reps.add(r.get("repeat"));
            for (String rep : reps) {
                double bulk = 0; for (Map<String, String> r : ufs) if (r.get("repeat").equals(rep) && Integer.parseInt(r.get("threads")) == maxT) bulk = Math.max(bulk, Double.parseDouble(r.get("MBps")));
                for (int s : sizes) { boolean hit = false; for (Map<String, String> r : ufs) if (r.get("repeat").equals(rep) && Integer.parseInt(r.get("threads")) == maxT && Integer.parseInt(r.get("size_kb")) == s && Double.parseDouble(r.get("MBps")) >= 0.9 * bulk) hit = true; if (hit) { knees.add(s * 1024.0); break; } }
            }
            if (!knees.isEmpty()) { double[] k = stats(knees); knee = meas((long) k[0], in, cond, k[1], k[2], "ufsbench knee over " + knees.size() + " repeats"); }
        }

        JSONObject mem = new JSONObject().put("total", memTotal)
            .put("available_at_probe", new JSONObject().put("value", memAvail).put("provenance", "measured").put("confidence", 0.9).put("source", "/proc/meminfo"))
            .put("grantable_foreground", unknown("PL-S2:never_measured"))
            .put("zram_or_swap", new JSONObject().put("present", meminfoKb("SwapTotal") > 0).put("fault_cost", unknown("swap_fault_probe:not_implemented")))
            .put("grantable_quiesced", grants.isEmpty() ? unknown("memprobe: " + grantWhy) : meas((long) stats(grants)[0], in, cond, stats(grants)[1], stats(grants)[2], "memprobe n_runs=" + grants.size()))
            .put("dram_read_gbps", dram.isEmpty() ? unknown("dramprobe: " + dramWhy) : meas(stats(dram)[0], in, cond, stats(dram)[1], stats(dram)[2], "dramprobe n_clean=" + dram.size()));
        JSONObject storage = new JSONObject().put("path", modelDir.getAbsolutePath()).put("filesystem", fsType).put("free_bytes", freeBytes)
            .put("random_read", rr).put("efficient_request_size", knee == null ? unknown("ufsbench: " + ufsWhy) : knee)
            .put("sequential_write_mbps", wr.isEmpty() ? unknown("wrbench: " + wrWhy) : meas(stats(wr)[0], in, cond, stats(wr)[1], stats(wr)[2], "wrbench n_repeats=" + wr.size() + " (fsync included)"))
            .put("direct_io_supported", directOk);
        boolean gpuOpen = dev.contains("/dev/kgsl-3d0") && dev.matches("(?s).*kgsl-3d0\\s+O_RDWR\\s+OPEN_OK.*");
        JSONArray acc = new JSONArray()
            .put(new JSONObject().put("kind", "cpu").put("backend_id", "cpu-generic").put("available", true).put("reachable_unprivileged", true).put("fidelity_class", "bit-exact"))
            .put(new JSONObject().put("kind", "gpu").put("backend_id", "opencl-adreno").put("available", gpuOpen).put("reachable_unprivileged", gpuOpen).put("fidelity_class", "unverified").put("note", "device node openable by this app; no engine backend probed"));
        JSONObject prof = new JSONObject().put("schema_version", 1).put("probe_suite", PROBE_SUITE).put("state", "partial").put("created_at", System.currentTimeMillis() / 1000.0)
            .put("profile_id", Integer.toHexString((idn.getString("build_fingerprint") + PROBE_SUITE).hashCode()))
            .put("identity", idn).put("cpu", new JSONObject().put("clusters", clusters).put("recommended_compute_mask", unknown("placement probe not run")).put("recommended_io_mask", unknown("placement probe not run")))
            .put("memory", mem).put("storage", storage).put("accelerators", acc)
            .put("thermal", new JSONObject().put("sustained_derate", unknown("T3 not implemented (PL-E14)")))
            .put("power", new JSONObject().put("rails_available", false).put("method", "none"))
            .put("validity", new JSONObject().put("conditions", cond).put("rejected_runs", rejected).put("rejection_reasons", reasons));
        return prof;
    }
}
