#!/bin/bash
# build.sh — build gx (fused expert FFN) and its references.
#   host/gpu_ffn/build.sh inc       regenerate gx_kernels.inc from gx_kernels.cl (committed; the engine includes it)
#   host/gpu_ffn/build.sh host      out/host/{gx_test, gx_pool_test, ggml_ref_x86}   (laptop OpenCL + laptop ggml)
#   host/gpu_ffn/build.sh arm-ref   out/arm/ggml_ref_arm                  (aarch64 static, run with qemu-aarch64-static)
#   host/gpu_ffn/build.sh android   out/android/{libgx.a (for the engine), gx_test, gx_pool_test, gx_bench (M6)}
# Env: NDK, CL_HDR (Khronos headers), GGML_SRC (a llama.cpp ggml tree, headers), GGML_HOST (x86 libggml*.so),
#      GGML_A64 (a static aarch64 ggml build: see NOTE.md "ARM reference" for its exact configure line).
# Memory: single translation units, one at a time; well under 1 GB.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
WORK=${WORK:-$(cd "$REPO/.." && pwd)/moe-work}
NDK=${NDK:-$HOME/Android/Sdk/ndk/30.0.16248370}
CL_HDR=${CL_HDR:-$WORK/opencl/OpenCL-Headers}
GGML_SRC=${GGML_SRC:-$WORK/BigMoeOnEdge/third_party/llama.cpp/ggml}
GGML_HOST=${GGML_HOST:-/tmp/claude-1000/gxdeps/x86/bin}
GGML_A64=${GGML_A64:-/tmp/claude-1000/gxdeps/a64/ggml/src}
CXX_A64="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++ --target=aarch64-linux-android29"
what=${1:-host}

gen_inc() {   # the kernel source as one C++ raw string literal
  { printf 'R"GXCL('; cat "$HERE/gx_kernels.cl"; printf ')GXCL"\n'; } > "$HERE/gx_kernels.inc"
}

case "$what" in
  inc) gen_inc ;;
  host)
    gen_inc
    O=$HERE/out/host; mkdir -p "$O"
    c++ -std=c++17 -O2 -ffp-contract=off -I"$HERE" -I"$CL_HDR" "$HERE/gx.cpp" "$HERE/gx_test.cpp" \
      -l:libOpenCL.so.1 -ldl -o "$O/gx_test"
    c++ -std=c++17 -O2 -ffp-contract=off -I"$HERE" -I"$CL_HDR" "$HERE/gx.cpp" "$HERE/gx_pool_test.cpp" \
      -l:libOpenCL.so.1 -ldl -lpthread -o "$O/gx_pool_test"
    c++ -std=c++17 -O2 -ffp-contract=off -I"$HERE" -I"$CL_HDR" "$HERE/gx.cpp" "$HERE/gx_bench.cpp" \
      -l:libOpenCL.so.1 -ldl -lpthread -o "$O/gx_bench"
    c++ -std=c++17 -O2 -I"$GGML_SRC/include" "$HERE/ggml_ref.cpp" -L"$GGML_HOST" -lggml -lggml-base -lggml-cpu \
      -Wl,-rpath,"$GGML_HOST" -o "$O/ggml_ref_x86"
    echo "built $O/gx_test $O/gx_pool_test $O/gx_bench $O/ggml_ref_x86" ;;
  arm-ref)
    O=$HERE/out/arm; mkdir -p "$O"
    $CXX_A64 -std=c++17 -O2 -static -I"$GGML_SRC/include" "$HERE/ggml_ref.cpp" \
      "$GGML_A64/libggml.a" "$GGML_A64/libggml-cpu.a" "$GGML_A64/libggml-base.a" -lm -ldl -o "$O/ggml_ref_arm"
    echo "built $O/ggml_ref_arm" ;;
  android)
    gen_inc
    O=${ANDROID_OUT:-$HERE/out/android}; mkdir -p "$O"
    $CXX_A64 -std=c++17 -O2 -ffp-contract=off -fPIC -I"$HERE" -I"$CL_HDR" -c "$HERE/gx.cpp" -o "$O/gx.o"
    "$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar" rcs "$O/libgx.a" "$O/gx.o"
    # M6 tools: OpenCL reached through cl_shim.cpp (dlopen of the device's libOpenCL.so); ggml_ref_arm (arm-ref)
    # is static and runs on the phone unchanged
    for t in gx_test gx_pool_test gx_bench; do
      $CXX_A64 -std=c++17 -O2 -ffp-contract=off -I"$HERE" -I"$CL_HDR" "$HERE/$t.cpp" "$O/gx.o" "$HERE/cl_shim.cpp" \
        -static-libstdc++ -ldl -o "$O/$t"
    done
    echo "built $O/libgx.a $O/gx_test $O/gx_pool_test $O/gx_bench" ;;
  *) echo "unknown target $what"; exit 2 ;;
esac
