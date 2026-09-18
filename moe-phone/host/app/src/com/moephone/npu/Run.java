package com.moephone.npu;

import android.app.Activity;
import android.os.Bundle;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

// moe-phone: runs a packaged native executable inside this app's own process (SELinux domain
// untrusted_app), which is the domain FastRPC's HAL path allows to reach the Hexagon NPU. The adb
// shell domain and run-as (runas_app) are both refused with AEE_ECONNREFUSED (0x72).
// The command line arrives as intent extra "args", separated by "~~"; "exe" picks the binary.
// stdout+stderr are written to filesDir/out.txt, which adb can read with run-as.
public class Run extends Activity {
    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        new Thread(new Runnable() {
            public void run() {
                File out = new File(getFilesDir(), "out.txt");
                try {
                    String lib = getApplicationInfo().nativeLibraryDir;
                    if (getIntent().getStringExtra("bench") != null) {
                        // llama-bench inside this process: the only place the HTP session opens
                        String[] ba = getIntent().getStringExtra("bench").split("~~");
                        File bo = new File(getFilesDir(), "bench.txt");
                        String res = Probe.runBench(lib, bo.getAbsolutePath(), ba,
                                                    getIntent().getStringExtra("env"));
                        FileOutputStream fb = new FileOutputStream(out);
                        fb.write((res + "\nEXIT=bench\n").getBytes());
                        fb.close();
                        finish();
                        return;
                    }
                    if (getIntent().getStringExtra("probe") != null) {
                        // in-process Hexagon backend init (child-vs-app-process test)
                        String res = Probe.run(lib, getIntent().getStringExtra("probe"));
                        FileOutputStream fp = new FileOutputStream(out);
                        fp.write((res + "\nEXIT=probe\n").getBytes());
                        fp.close();
                        finish();
                        return;
                    }
                    String argStr = getIntent().getStringExtra("args");
                    if (argStr == null) { argStr = "--version"; }
                    String exe = getIntent().getStringExtra("exe");
                    if (exe == null) { exe = "libexe_llama_completion.so"; }
                    java.util.List<String> cmd = new java.util.ArrayList<String>();
                    cmd.add(lib + "/" + exe);
                    String[] parts = argStr.split("~~");
                    for (int i = 0; i < parts.length; i++) {
                        if (parts[i].length() > 0) { cmd.add(parts[i]); }
                    }
                    ProcessBuilder pb = new ProcessBuilder(cmd);
                    pb.environment().put("LD_LIBRARY_PATH", lib + ":/vendor/lib64");
                    pb.environment().put("ADSP_LIBRARY_PATH", lib);
                    pb.environment().put("GGML_HEXAGON_ARCH", "v81");
                    pb.redirectErrorStream(true);
                    pb.directory(getFilesDir());
                    Process p = pb.start();
                    InputStream is = p.getInputStream();
                    FileOutputStream fo = new FileOutputStream(out);
                    byte[] buf = new byte[8192];
                    int n;
                    while ((n = is.read(buf)) > 0) { fo.write(buf, 0, n); fo.flush(); }
                    is.close();
                    fo.close();
                    int rc = p.waitFor();
                    FileOutputStream fa = new FileOutputStream(out, true);
                    fa.write(("\nEXIT=" + rc + "\n").getBytes());
                    fa.close();
                } catch (Throwable t) {
                    try {
                        FileOutputStream fa = new FileOutputStream(out, true);
                        fa.write(("\nEXCEPTION " + t + "\n").getBytes());
                        fa.close();
                    } catch (Exception ignored) { }
                }
                finish();
            }
        }).start();
    }
}
