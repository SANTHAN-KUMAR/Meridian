package com.meridian.app;

import android.content.Context;
import org.json.JSONException;
import org.json.JSONObject;
import java.io.*;
import java.util.*;

/** One warm bmoe-cli --session process (BigMoeOnEdge engine, docs/telemetry.md "Session mode").
 *  Requests are JSON lines on stdin; BMOE_<TAG> {json} control and per-token lines come back on stdout. */
public final class Engine {
    public interface Listener {
        void onReady(JSONObject ready);
        void onToken(String text, JSONObject progress);
        void onDone(JSONObject done);
        void onError(String message);
    }
    public static final class Config {
        public File model; public int threads = 4, ctx = 2048, ubatch = 512; public String cpuMask = null; public boolean chatml = true;
        // streamed tier (experts read from flash through a per-layer cache); all values are derived by PlanV2 from measurements
        public boolean moeStream; public int cacheFloorMb, cacheCeilMb, ioThreads = 4; public String ioMask = null;
        public String describe() { return (moeStream ? "streamed" : "resident") + " threads=" + threads + " mask=" + cpuMask + " ctx=" + ctx + (moeStream ? " cache=[" + cacheFloorMb + "," + cacheCeilMb + "]MiB io=" + ioThreads + "@" + ioMask : ""); }
    }

    private final Context ctx; private final Config cfg; private Process proc; private Listener listener;
    private Writer stdin; private volatile boolean ready, closed;
    private final Deque<String> stderrTail = new ArrayDeque<>();
    private JSONObject readyInfo;

    public Engine(Context c, Config cfg) { this.ctx = c.getApplicationContext(); this.cfg = cfg; }
    public boolean isReady() { return ready && !closed; }
    public JSONObject readyInfo() { return readyInfo; }
    public synchronized String stderrTail() { return String.join("\n", stderrTail); }

    public void start(Listener l) throws IOException {
        this.listener = l;
        List<String> a = new ArrayList<>(Arrays.asList("-m", cfg.model.getAbsolutePath(), "--session",
                "-t", String.valueOf(cfg.threads), "-c", String.valueOf(cfg.ctx), "--ubatch", String.valueOf(cfg.ubatch)));
        if (cfg.chatml) a.add("--chatml");
        if (cfg.cpuMask != null) { a.add("--cpu-mask"); a.add(cfg.cpuMask); }
        if (cfg.moeStream) {   // flags adopted by the research project (results/2026-09-19); values come from the plan
            a.addAll(Arrays.asList("--moe-stream", "--cache-mb", "auto", "--cache-floor-mb", String.valueOf(cfg.cacheFloorMb), "--cache-ceil-mb", String.valueOf(cfg.cacheCeilMb),
                "--overlap", "--dense-weights", "anon", "--expert-slru", "--predict-prefetch", "--spec-adopt-selective", "--io-threads", String.valueOf(cfg.ioThreads)));
            if (cfg.ioMask != null) { a.add("--io-cpu-mask"); a.add(cfg.ioMask); }
        }
        proc = Native.builder(ctx, "bmoe_cli", a).start();
        stdin = new BufferedWriter(new OutputStreamWriter(proc.getOutputStream(), "UTF-8"));
        Thread out = new Thread(this::readStdout, "engine-stdout"), err = new Thread(this::readStderr, "engine-stderr");
        out.setDaemon(true); err.setDaemon(true); out.start(); err.start();
    }

    private void readStderr() {
        try (BufferedReader r = new BufferedReader(new InputStreamReader(proc.getErrorStream(), "UTF-8"))) {
            String l; while ((l = r.readLine()) != null) synchronized (this) { stderrTail.addLast(l); if (stderrTail.size() > 40) stderrTail.removeFirst(); }
        } catch (IOException ignored) { }
    }

    private void readStdout() {
        try (BufferedReader r = new BufferedReader(new InputStreamReader(proc.getInputStream(), "UTF-8"))) {
            String line;
            while ((line = r.readLine()) != null) {
                int sp = line.indexOf(' ');
                if (!line.startsWith("BMOE_") || sp < 0) continue;
                String tag = line.substring(0, sp);
                JSONObject j;
                try { j = new JSONObject(line.substring(sp + 1)); } catch (JSONException e) { continue; }
                switch (tag) {
                    case "BMOE_READY": ready = true; readyInfo = j; listener.onReady(j); break;
                    case "BMOE_PROGRESS": listener.onToken(j.optString("delta_text", ""), j); break;
                    case "BMOE_DONE": listener.onDone(j); break;
                    default: break;
                }
            }
        } catch (IOException ignored) { }
        if (!closed) { closed = true; String why = stderrTail(); listener.onError("engine exited" + (why.isEmpty() ? "" : ": " + lastLines(why, 6))); }
    }

    private static String lastLines(String s, int n) { String[] p = s.split("\n"); StringBuilder b = new StringBuilder(); for (int i = Math.max(0, p.length - n); i < p.length; i++) b.append(p[i]).append('\n'); return b.toString().trim(); }

    private synchronized void send(JSONObject j) throws IOException { stdin.write(j.toString()); stdin.write('\n'); stdin.flush(); }

    public boolean supportsGrammar() { return readyInfo != null && readyInfo.optBoolean("grammar", false); }

    public void generate(int id, String prompt, int nPredict, boolean clearKv, String grammar) throws IOException {
        try { JSONObject j = new JSONObject().put("cmd", "generate").put("id", id).put("prompt", prompt).put("n_predict", nPredict).put("clear_kv", clearKv);
            if (grammar != null) { if (!supportsGrammar()) throw new IOException("this engine build does not support grammar-constrained decoding"); j.put("grammar", grammar); }
            send(j); }
        catch (JSONException e) { throw new IOException(e); }
    }
    public void cancel() { try { send(new JSONObject().put("cmd", "cancel")); } catch (Exception ignored) { } }
    public void close() {
        if (closed) return; closed = true;
        try { send(new JSONObject().put("cmd", "close")); } catch (Exception ignored) { }
        if (proc != null) { final Process p = proc; new Thread(() -> { try { Thread.sleep(2000); } catch (InterruptedException ignored) { } p.destroyForcibly(); }).start(); }
    }
}
