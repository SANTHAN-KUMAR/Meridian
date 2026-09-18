// moe-phone: does the Hexagon (HTP) backend open its DSP session when initialised INSIDE an app's own
// process, rather than in a binary the app spawns? Geekbench AI's app process opens a session through the
// vendor HAL on this phone (results/2026-09-17/npu_app_domain/README.md), while our app-spawned child is
// refused with 0x72. This JNI entry point runs the same ggml backend init in the app process itself and
// returns what happened, so the child-vs-app-process hypothesis can be settled.
#include <jni.h>

#include <string>

#include <cstdlib>
#include <cstdio>
#include <vector>

#include <dlfcn.h>

#include "ggml.h"
#include "ggml-backend.h"

extern "C" JNIEXPORT jstring JNICALL
Java_com_moephone_npu_Probe_probe(JNIEnv * env, jclass, jstring jname) {
    const char * cname = env->GetStringUTFChars(jname, nullptr);
    std::string name = cname ? cname : "HTP0";
    env->ReleaseStringUTFChars(jname, cname);

    std::string out;
    const size_t n_dev = ggml_backend_dev_count();
    out += "devices=" + std::to_string(n_dev);
    for (size_t i = 0; i < n_dev; i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        out += std::string(" [") + ggml_backend_dev_name(d) + "]";
    }

    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
    if (!dev) {
        out += " ; no device named " + name;
        return env->NewStringUTF(out.c_str());
    }

    // The session is opened here, on this thread, in the app's own process.
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) {
        out += " ; " + name + " init FAILED (session not opened)";
        return env->NewStringUTF(out.c_str());
    }
    out += " ; " + name + " init OK: " + ggml_backend_name(be);

    // A buffer allocation is the next thing a real run needs from the session.
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    if (buft) {
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, 8u << 20);
        out += buf ? " ; 8 MiB buffer OK" : " ; 8 MiB buffer FAILED";
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
    }
    ggml_backend_free(be);
    return env->NewStringUTF(out.c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_moephone_npu_Probe_setEnv(JNIEnv * env, jclass, jstring jk, jstring jv) {
    const char * k = env->GetStringUTFChars(jk, nullptr);
    const char * v = env->GetStringUTFChars(jv, nullptr);
    // An empty value UNSETS the variable. The app process survives between runs, so a knob set for one
    // arm of a sweep would otherwise still be set for the next one and the arms would accumulate
    // instead of being independent.
    const int rc = (v && v[0]) ? setenv(k, v, 1) : unsetenv(k);
    env->ReleaseStringUTFChars(jk, k);
    env->ReleaseStringUTFChars(jv, v);
    return rc;
}

// Runs llama.cpp's llama-bench IN THIS PROCESS (the Hexagon session opens for the app process but is
// refused for any binary the app spawns). argv arrives as a Java String[]; stdout/stderr are redirected
// to outPath so the caller can read the table.
extern "C" JNIEXPORT jint JNICALL
Java_com_moephone_npu_Probe_bench(JNIEnv * env, jclass, jstring joutPath, jobjectArray jargs) {
    const char * outPath = env->GetStringUTFChars(joutPath, nullptr);
    FILE * fo = freopen(outPath, "w", stdout);
    FILE * fe = freopen(outPath, "a", stderr);
    setvbuf(stdout, nullptr, _IOLBF, 0);
    setvbuf(stderr, nullptr, _IOLBF, 0);
    env->ReleaseStringUTFChars(joutPath, outPath);

    // A first argument of "completion" selects llama-completion (where the expert-streaming flags live)
    // instead of llama-bench. Both are plain int(int, char **) entry points in their impl library.
    const char * libName = "libllama-bench-impl.so";
    const char * symbol  = "_Z11llama_benchiPPc";
    const char * argv0   = "llama-bench";
    jsize skip = 0;
    if (env->GetArrayLength(jargs) > 0) {
        auto js0 = (jstring) env->GetObjectArrayElement(jargs, 0);
        const char * c0 = env->GetStringUTFChars(js0, nullptr);
        if (c0 && std::string(c0) == "completion") {
            libName = "libllama-completion-impl.so";
            symbol  = "_Z16llama_completioniPPc";
            argv0   = "llama-completion";
            skip    = 1;
        } else if (c0 && std::string(c0) == "mmb") {
            // per-op matmul benchmark (host/app/ggml_matmul_bench.cpp): the device question answered
            // directly, without the end-to-end rate's spread hiding the effect
            libName = "libmatmulbench.so";
            symbol  = "matmul_bench_main";
            argv0   = "matmul_bench";
            skip    = 1;
        } else if (c0 && std::string(c0) == "bmoe") {
            // our own engine (BigMoeOnEdge), built as libbmoe_entry.so with an extern "C" entry point
            libName = "libbmoe_entry.so";
            symbol  = "bmoe_main";
            argv0   = "bmoe-cli";
            skip    = 1;
        }
        env->ReleaseStringUTFChars(js0, c0);
    }
    void * h = dlopen(libName, RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        fprintf(stderr, "dlopen %s failed: %s\n", libName, dlerror());
        fflush(stderr);
        return -1;
    }
    using bench_fn = int (*)(int, char **);
    auto fn = (bench_fn) dlsym(h, symbol);
    if (!fn) {
        fprintf(stderr, "dlsym %s failed: %s\n", symbol, dlerror());
        fflush(stderr);
        return -2;
    }
    const jsize n = env->GetArrayLength(jargs);
    std::vector<std::string> store;
    store.push_back(argv0);
    for (jsize i = skip; i < n; i++) {
        auto js = (jstring) env->GetObjectArrayElement(jargs, i);
        const char * c = env->GetStringUTFChars(js, nullptr);
        store.push_back(c ? c : "");
        env->ReleaseStringUTFChars(js, c);
    }
    std::vector<char *> argv;
    for (auto & a : store) { argv.push_back(const_cast<char *>(a.c_str())); }
    const int rc = fn((int) argv.size(), argv.data());
    fflush(stdout);
    fflush(stderr);
    if (fo) { fclose(fo); }
    if (fe) { fclose(fe); }
    return rc;
}
