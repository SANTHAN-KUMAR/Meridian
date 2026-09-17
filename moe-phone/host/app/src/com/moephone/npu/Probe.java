package com.moephone.npu;

// moe-phone: the app-side shim that gives our engine NPU access. The Hexagon (HTP) session opens for an
// app's OWN process (with libOpenCL.so / libcdsprpc.so declared via <uses-native-library>) but is refused
// for any binary the app spawns, so the engine must be called in-process instead of exec'd.
public class Probe {
    public static native String probe(String device);
    public static native int setEnv(String key, String value);
    public static native int bench(String outPath, String[] args);

    private static final String[] LIBS = {
        "c++_shared", "ggml-base", "ggml-cpu", "ggml-opencl", "ggml-hexagon", "ggml",
        "llama", "llama-common", "matmulbench", "npuprobe"
    };

    private static String load(String libDir) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < LIBS.length; i++) {
            try {
                System.loadLibrary(LIBS[i]);
            } catch (Throwable t) {
                sb.append("load ").append(LIBS[i]).append(": ").append(t.getMessage()).append(" | ");
            }
        }
        try {
            setEnv("ADSP_LIBRARY_PATH", libDir);
            setEnv("GGML_HEXAGON_ARCH", "v81");
        } catch (Throwable t) {
            sb.append("setEnv: ").append(t).append(" | ");
        }
        return sb.toString();
    }

    // Opens the backend and allocates a buffer: the child-vs-app-process test.
    public static String run(String libDir, String device) {
        String pre = load(libDir);
        try {
            return pre + probe(device);
        } catch (Throwable t) {
            return pre + "probe threw: " + t;
        }
    }

    // Runs llama.cpp's llama-bench inside this process; its table goes to outPath.
    public static String runBench(String libDir, String outPath, String[] args) {
        String pre = load(libDir);
        try {
            int rc = bench(outPath, args);
            return pre + "bench rc=" + rc;
        } catch (Throwable t) {
            return pre + "bench threw: " + t;
        }
    }
}
