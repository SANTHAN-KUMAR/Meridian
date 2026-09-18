#!/bin/bash
# build.sh — build zcbench for the phone (arm64, NDK) and/or the laptop (x86, self-test on any OpenCL GPU).
#
#   host/gpu_zerocopy/build.sh android   -> out/android/{zcbench, libggml*.so}   (push the whole dir)
#   host/gpu_zerocopy/build.sh host      -> out/host/zcbench
#
# ggml comes from an existing BigMoeOnEdge build, so the CPU arm runs the engine's own kernels:
#   BMOE_SRC      the BigMoeOnEdge tree (headers)           default ../moe-work/BigMoeOnEdge
#   BMOE_ANDROID  its Android build with libggml*.so        default $BMOE_SRC/build-android-i8mm/bin
#   BMOE_HOST     a host build with libggml*.so             default /tmp/claude-1000/bmoe-prefetch/build-host/bin
#   NDK           Android NDK                                default ~/Android/Sdk/ndk/30.0.16248370
# OpenCL headers: Khronos OpenCL-Headers (CL_HDR). The library is dlopen'ed at run time; nothing links it.
# Memory: a single translation unit, -j1; well under 1 GB.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
WORK=${WORK:-$(cd "$REPO/.." && pwd)/moe-work}
BMOE_SRC=${BMOE_SRC:-$WORK/BigMoeOnEdge}
BMOE_ANDROID=${BMOE_ANDROID:-$BMOE_SRC/build-android-i8mm/bin}
BMOE_HOST=${BMOE_HOST:-/tmp/claude-1000/bmoe-prefetch/build-host/bin}
NDK=${NDK:-$HOME/Android/Sdk/ndk/30.0.16248370}
CL_HDR=${CL_HDR:-$WORK/opencl/OpenCL-Headers}
GGML_INC=$BMOE_SRC/third_party/llama.cpp/ggml/include
SRC=$HERE/zcbench.cpp
what=${1:-both}

if [ "$what" = android ] || [ "$what" = both ]; then
  O=$HERE/out/android; mkdir -p "$O"
  cp "$BMOE_ANDROID"/libggml.so "$BMOE_ANDROID"/libggml-base.so "$BMOE_ANDROID"/libggml-cpu.so "$O/"
  "$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++" --target=aarch64-linux-android29 -std=c++17 -O2 \
    -I"$CL_HDR" -I"$GGML_INC" "$SRC" -L"$O" -lggml -lggml-base -lggml-cpu -ldl -static-libstdc++ \
    -Wl,-rpath,'$ORIGIN' -o "$O/zcbench"
  echo "built $O/zcbench"
fi
if [ "$what" = host ] || [ "$what" = both ]; then
  O=$HERE/out/host; mkdir -p "$O"
  c++ -std=c++17 -O2 -I"$CL_HDR" -I"$GGML_INC" "$SRC" -L"$BMOE_HOST" -lggml -lggml-base -lggml-cpu -ldl -lpthread \
    -Wl,-rpath,"$BMOE_HOST" -o "$O/zcbench"
  echo "built $O/zcbench"
fi
