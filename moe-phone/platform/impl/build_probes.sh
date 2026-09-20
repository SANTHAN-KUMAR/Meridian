#!/bin/sh
# Cross-compiles the existing, already-validated T0/T1 probes
# (../../device/{devprobe,dramprobe,memprobe,ufsbench}.c) for arm64-android
# using the Android NDK, so the L1 profiler can push and run them on any
# attached device without needing a compiler on the phone.
#
# These probes were NOT written for this platform; they are the C probe
# suite 00_PROBLEM.md section 5 and 04_DEVICE_PROFILING.md say "already
# exists ... and becomes a library". This script is that library's build
# step, unmodified from the source.
#
# Usage: NDK=/path/to/ndk sh build_probes.sh [output_dir]

set -e
: "${NDK:?set NDK to the Android NDK root, e.g. \$HOME/Android/Sdk/ndk/<version>}"
: "${API:=28}"
OUT="${1:-$(dirname "$0")/bin}"
DEVICE_SRC="$(dirname "$0")/../../device"
CC="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${API}-clang"

if [ ! -x "$CC" ]; then
  echo "compiler not found: $CC" >&2
  exit 1
fi

mkdir -p "$OUT"
"$CC" -O2 "$DEVICE_SRC/devprobe.c" -o "$OUT/devprobe"
"$CC" -O3 -pthread "$DEVICE_SRC/dramprobe.c" -o "$OUT/dramprobe"
"$CC" -O2 "$DEVICE_SRC/memprobe.c" -o "$OUT/memprobe"
"$CC" -O2 -pthread "$DEVICE_SRC/ufsbench.c" -o "$OUT/ufsbench" -lm
"$CC" -O2 "$(dirname "$0")/native/wrbench.c" -o "$OUT/wrbench"
echo "built: $OUT/devprobe $OUT/dramprobe $OUT/memprobe $OUT/ufsbench"
