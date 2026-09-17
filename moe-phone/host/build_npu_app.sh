#!/bin/bash
# build_npu_app.sh — build and install the app package that lets OUR engine reach the Hexagon NPU.
#
# WHY AN APP AT ALL. On this phone (OnePlus 15R, Snapdragon 8 Elite Gen 5, stock, unrooted) a FastRPC
# session to the cDSP is refused with 0x72 = AEE_ECONNREFUSED from the `shell` domain (adb), from
# `runas_app` (run-as) and from Termux, but opens normally inside an app's own process once the app
# declares the vendor libraries it uses. Evidence: results/2026-09-17/npu_app_domain (Geekbench AI's
# logcat opening a session) and results/2026-09-17/npu_inprocess (our own, in- vs out-of-process).
# A binary the app spawns is refused too, so the engine cannot be exec'd: it must be called in-process.
# That is what patch 0009 adds (`bmoe_main`, the CLI as a shared-library symbol) and what the JNI shim
# in npu-jni/npu_probe.cpp dlsym's.
#
# WHAT IT PRODUCES. One APK (default com.moephone.bmoe3) containing
#   libbmoe_entry.so      our engine, entry point bmoe_main(argc, argv)
#   libggml-*.so          its ggml backends, including ggml-hexagon
#   libggml-htp-v81.so    the DSP-side library, found through ADSP_LIBRARY_PATH = nativeLibraryDir
#   libc++_shared.so      one libc++ for the whole process; a statically linked libc++ inside the ggml
#                         libraries aborts with "Pointer tag ... was truncated" when the app's
#                         allocator frees memory those libraries allocated
#   libnpuprobe.so        the JNI shim (probe / setEnv / bench)
# and the two Java classes that load them in the app process and pass a command line through.
#
# The APK is debuggable only so that `run-as` can read the files it writes; nothing in the measurement
# depends on that. It is signed with a throwaway key.
#
# Usage:
#   moe-phone/host/build_npu_app.sh [PKG] [ENGINE_DIR] [APK_DIR]
# Environment:
#   ANDROID_SDK, ANDROID_NDK, BUILD_TOOLS, KEYSTORE, ANDROID_SERIAL
set -eu

PKG=${1:-com.moephone.bmoe3}
ENGINE=${2:-$HOME/moework/BigMoeOnEdge}
APKDIR=${3:-$HOME/moework/npu-apk3}
SDK=${ANDROID_SDK:-$HOME/Android/Sdk}
NDK=${ANDROID_NDK:-$SDK/ndk/30.0.16248370}
BT=${BUILD_TOOLS:-$SDK/build-tools/36.1.0}
KS=${KEYSTORE:-$HOME/moework/debug.keystore}
JNI=${JNI_DIR:-$(cd "$(dirname "$0")/../../../moe-work/npu-jni" 2>/dev/null && pwd || echo "$HOME/moework/npu-jni")}
BUILD=$ENGINE/build-android-hex
AJ=$(ls -d "$SDK"/platforms/android-*/android.jar | tail -1)

echo "== 1. engine for Android with the Hexagon and OpenCL backends"
# -DANDROID_STL=c++_shared: see libc++_shared.so above. armv8.6-a+i8mm is what the 8 Elite Gen 5 has and
# what every other measured row on this phone was built with, so this build is comparable to them.
if [ ! -f "$BUILD/CMakeCache.txt" ]; then
  cmake -S "$ENGINE" -B "$BUILD" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_shared \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DGGML_HEXAGON=ON -DGGML_OPENCL=ON -DGGML_OPENMP=OFF \
    -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm+fp16
fi
cmake --build "$BUILD" -j"$(nproc)"

echo "== 2. JNI shim"
"$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++" \
  -O2 -fPIC -shared -std=c++17 -I"$ENGINE/third_party/llama.cpp/ggml/include" \
  "$JNI/npu_probe.cpp" -o "$APKDIR/lib/arm64-v8a/libnpuprobe.so" \
  -L"$BUILD/bin" -lggml -lggml-base -llog

echo "== 3. stage native libraries"
mkdir -p "$APKDIR/lib/arm64-v8a" "$APKDIR/classes"
cp "$BUILD/cli/libbmoe_entry.so" "$APKDIR/lib/arm64-v8a/"
for l in ggml-base ggml-cpu ggml-opencl ggml-hexagon ggml llama llama-common; do
  cp "$BUILD/bin/lib$l.so" "$APKDIR/lib/arm64-v8a/"
done
cp "$BUILD/third_party/llama.cpp/ggml/src/ggml-hexagon/libggml-htp-v81.so" "$APKDIR/lib/arm64-v8a/"
cp "$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" \
   "$APKDIR/lib/arm64-v8a/"

echo "== 4. dex + link + sign"
cd "$APKDIR"
javac -nowarn -source 8 -target 8 -bootclasspath "$AJ" -classpath "$AJ" -d classes src/com/moephone/npu/*.java
"$BT/d8" --lib "$AJ" --output . classes/com/moephone/npu/*.class   # ALL classes: Run's inner Runnable too
"$BT/aapt2" link -I "$AJ" --manifest AndroidManifest.xml -o base.apk --min-sdk-version 28 --target-sdk-version 34
rm -f unsigned.apk aligned.apk app.apk
cp base.apk unsigned.apk
zip -q -X unsigned.apk classes.dex
# stored (-0), not deflated: extractNativeLibs=true copies them out at install and they load faster
find lib -name '*.so' | while read -r f; do zip -q -X -0 unsigned.apk "$f"; done
"$BT/zipalign" -p -f 4 unsigned.apk aligned.apk
"$BT/apksigner" sign --ks "$KS" --ks-pass pass:android --ks-key-alias moephone --key-pass pass:android \
  --out app.apk aligned.apk

echo "== 5. install"
adb install -r app.apk
adb shell "am start -n $PKG/com.moephone.npu.Run --es probe HTP0" >/dev/null
sleep 12
echo "-- in-process HTP probe:"
adb shell "run-as $PKG sh -c 'cat files/out.txt'"
echo
echo "Expected: 'devices=3 [GPUOpenCL] [HTP0] [CPU] ; HTP0 init OK: HTP0 ; 8 MiB buffer OK'."
echo "A refusal here (0x72) means the manifest's <uses-native-library> entries or the app process itself"
echo "is not what opened the session — do not report any NPU number until this line says init OK."
