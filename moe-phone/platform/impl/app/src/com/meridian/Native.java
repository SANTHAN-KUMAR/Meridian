package com.meridian.app;

import android.content.Context;
import java.io.*;
import java.util.*;

/** Executables ship as lib*.so in the APK's native lib dir (the only place an app may exec from on Android 10+). */
public final class Native {
    public static File libDir(Context c) { return new File(c.getApplicationInfo().nativeLibraryDir); }

    public static boolean present(Context c, String name) { return new File(libDir(c), "lib" + name + ".so").canExecute(); }

    public static ProcessBuilder builder(Context c, String name, List<String> args) {
        List<String> cmd = new ArrayList<>();
        cmd.add(new File(libDir(c), "lib" + name + ".so").getAbsolutePath());
        cmd.addAll(args);
        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.environment().put("LD_LIBRARY_PATH", libDir(c).getAbsolutePath());
        pb.directory(c.getFilesDir());
        return pb;
    }

    /** Run to completion; returns stdout. stderr goes to `err` (may be null). Throws on non-zero exit unless allowNonZero. */
    public static String run(Context c, String name, List<String> args, long timeoutMs, StringBuilder err, boolean allowNonZero) throws IOException, InterruptedException {
        Process p = builder(c, name, args).start();
        p.getOutputStream().close();
        final StringBuilder out = new StringBuilder(), er = new StringBuilder();
        Thread t1 = drain(p.getInputStream(), out), t2 = drain(p.getErrorStream(), er);
        long end = System.currentTimeMillis() + timeoutMs;
        while (true) {
            try { p.exitValue(); break; } catch (IllegalThreadStateException e) { /* still running */ }
            if (System.currentTimeMillis() > end) { p.destroyForcibly(); throw new IOException(name + " timed out after " + timeoutMs + " ms"); }
            Thread.sleep(100);
        }
        t1.join(2000); t2.join(2000);
        if (err != null) err.append(er);
        if (p.exitValue() != 0 && !allowNonZero) throw new IOException(name + " exited " + p.exitValue() + ": " + er);
        return out.toString();
    }

    private static Thread drain(final InputStream in, final StringBuilder sink) {
        Thread t = new Thread(() -> {
            try (BufferedReader r = new BufferedReader(new InputStreamReader(in))) {
                String l; while ((l = r.readLine()) != null) synchronized (sink) { sink.append(l).append('\n'); }
            } catch (IOException ignored) { }
        });
        t.setDaemon(true); t.start(); return t;
    }

    public static String readFile(String path) {
        try (BufferedReader r = new BufferedReader(new FileReader(path))) {
            StringBuilder sb = new StringBuilder(); String l;
            while ((l = r.readLine()) != null) sb.append(l).append('\n');
            return sb.toString();
        } catch (IOException e) { return null; }   // null == DENIED_OR_ABSENT, never "empty"
    }
}
