#!/bin/bash
# run_accept.sh — M3 acceptance for gx on the laptop, end to end, into a new stamped results directory.
#   1. build gx_test + ggml_ref_x86 (host) and ggml_ref_arm (aarch64 static; its numeric functions are checked
#      instruction-identical to the engine's Android libggml-cpu.so, step 2)
#   2. disassembly identity: quantize_row_q8_{0,1}, vec_dot_q4_{0,1}_q8_{0,1}, ggml_vec_swiglu_f32
#   3. cases: random stress cases + real Qwen3-30B-A3B expert slices read by offset (make_cases.py)
#   4. references: ggml_ref_arm under qemu-aarch64-static (bit-exact target) and ggml_ref_x86
#   5. gx_test (bit comparison + errors), negative_controls.sh (each mutation must be detected) and
#      gx_pool_test (pool allocator, dispatch from pool slots, concurrent map/unmap, dispatch-while-mapped)
# Output: results/<date>/gx_m3_<time>/ (never overwritten): gx_test.json, gx_test.out, controls.out,
#         pool_test.out, disasm_identity.txt, manifest.json, STAMP. Case files stay in a scratch directory (hundreds of MB).
#   host/gpu_ffn/run_accept.sh <scratch_dir> [gguf]
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
MP=$(cd "$HERE/../.." && pwd)
S=${1:?scratch dir for cases}
GGUF=${2:-$(cd "$MP/../.." && pwd)/moe-work/models/Qwen3-30B-A3B-Q4_0.gguf}
ENGINE_SO=${ENGINE_SO:-$(cd "$MP/../.." && pwd)/moe-work/BigMoeOnEdge/build-android-i8mm/bin/libggml-cpu.so}
NDK=${NDK:-$HOME/Android/Sdk/ndk/30.0.16248370}
OBJ=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-objdump
R="$MP/results/$(date +%F)/gx_m3_$(date +%H%M%S)"
[ -e "$R" ] && { echo "exists: $R"; exit 1; }
mkdir -p "$R"

"$HERE/build.sh" host >/dev/null   # host only: out/android is whatever the next phone run was built with
"$HERE/build.sh" arm-ref >/dev/null

{
  for f in quantize_row_q8_0 quantize_row_q8_1 ggml_vec_dot_q4_0_q8_0 ggml_vec_dot_q4_1_q8_1 ggml_vec_swiglu_f32; do
    dis() { "$OBJ" -d --no-show-raw-insn "$1" | awk "/<$f>:/{p=1} p{print} p&&/^\$/{exit}" | awk '{$1=""; print}' |
            sed -E 's/0x[0-9a-f]+//g; s/<[^>]*>//g' | md5sum | cut -c1-16; }
    a=$(dis "$HERE/out/arm/ggml_ref_arm"); b=$(dis "$ENGINE_SO")
    echo "$f ref=$a engine=$b $([ "$a" = "$b" ] && echo IDENTICAL || echo DIFFERENT)"
  done
} | tee "$R/disasm_identity.txt"

rm -rf "$S"
python3 "$HERE/make_cases.py" "$S" --gguf "$GGUF"
cp "$S/manifest.json" "$R/"
for c in "$S"/*.bin; do
  qemu-aarch64-static -cpu max "$HERE/out/arm/ggml_ref_arm" "$c" "$c.arm.out" 2
  "$HERE/out/host/ggml_ref_x86" "$c" "$c.x86.out" 2
done
qemu-aarch64-static -cpu max "$HERE/out/arm/ggml_ref_arm" --quant "$S/quant_blocks.f32" \
  "$S/quant_blocks.f32.q8_0.arm" "$S/quant_blocks.f32.q8_1.arm"
qemu-aarch64-static -cpu max "$HERE/out/arm/ggml_ref_arm" --swiglu "$S/swiglu_pairs.f32" "$S/swiglu_pairs.f32.arm"

set +e
"$HERE/out/host/gx_test" "$S" "$R/gx_test.json" | tee "$R/gx_test.out"
t=${PIPESTATUS[0]}
"$HERE/negative_controls.sh" "$S" | tee "$R/controls.out"
c=${PIPESTATUS[0]}
"$HERE/out/host/gx_pool_test" "$S" | tee "$R/pool_test.out"
pt=${PIPESTATUS[0]}
{
  echo "date=$(date -Is) git=$(git -C "$MP" rev-parse --short HEAD) dirty=$(git -C "$MP" status --porcelain -- host/gpu_ffn | wc -l)"
  echo "gguf=$GGUF engine_so=$ENGINE_SO qemu=$(qemu-aarch64-static --version | head -1)"
  echo "kernel_md5=$(md5sum < "$HERE/gx_kernels.cl" | cut -c1-16) gx_test_exit=$t controls_exit=$c pool_test_exit=$pt"
} > "$R/STAMP"
cat "$R/STAMP"
echo "results in $R"
