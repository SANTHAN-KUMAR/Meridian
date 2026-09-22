package com.meridian.app;

import android.content.Context;
import org.json.JSONObject;
import java.io.File;
import java.util.concurrent.*;

/** Synchronous facade over one Engine session: ask() blocks until the turn's BMOE_DONE. */
public final class Chat implements Engine.Listener {
    public interface TokenSink { void token(String t); }
    private final Engine engine; private volatile TokenSink sink; private volatile String error;
    private final BlockingQueue<JSONObject> done = new LinkedBlockingQueue<>();
    private final CountDownLatch ready = new CountDownLatch(1); private final StringBuilder text = new StringBuilder(); private int nextId = 1;

    public Chat(Context c, Engine.Config cfg) { engine = new Engine(c, cfg); }
    public Engine engine() { return engine; }

    public void start(long timeoutMs) throws Exception {
        engine.start(this);
        if (!ready.await(timeoutMs, TimeUnit.MILLISECONDS) || error != null) throw new IllegalStateException(error != null ? error : "engine did not become ready in " + timeoutMs + " ms: " + engine.stderrTail());
    }
    public JSONObject lastResult; public String lastText = "";

    /** Returns the model's reply text; lastResult holds the BMOE_DONE telemetry. */
    public synchronized String ask(String prompt, int nPredict, boolean clearKv, TokenSink s) throws Exception { return ask(prompt, nPredict, clearKv, null, s); }
    public synchronized String ask(String prompt, int nPredict, boolean clearKv, String grammar, TokenSink s) throws Exception { return ask(prompt, nPredict, clearKv, grammar, false, null, s); }
    public synchronized String ask(String prompt, int nPredict, boolean clearKv, String grammar, boolean resetHistory, Boolean think, TokenSink s) throws Exception {
        if (error != null) throw new IllegalStateException(error);
        sink = s; text.setLength(0); done.clear();
        engine.generate(nextId++, prompt, nPredict, clearKv, grammar, resetHistory, think);
        JSONObject d = done.poll(600, TimeUnit.SECONDS);
        if (d == null) { engine.cancel(); throw new IllegalStateException(error != null ? error : "generation timed out"); }
        if (d.optBoolean("error")) { String m = d.optString("msg", error == null ? "engine error" : error); if (d.optBoolean("fatal")) error = m; throw new IllegalStateException(m); }
        lastResult = d; lastText = text.toString(); return lastText;
    }
    public void close() { engine.close(); }
    public void cancel() { engine.cancel(); }

    @Override public void onReady(JSONObject r) { ready.countDown(); }
    @Override public void onToken(String t, JSONObject p) { text.append(t); TokenSink s = sink; if (s != null) s.token(t); }
    @Override public void onDone(JSONObject d) { done.add(d); }
    @Override public void onError(String m) { error = m; ready.countDown(); try { done.add(new JSONObject().put("error", true)); } catch (Exception ignored) { } }
}
