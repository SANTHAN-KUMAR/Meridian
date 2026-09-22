#!/bin/sh
# Builds build/meridian.apk: ONE self-contained arm64 APK (engine + probes + planner + agent), SDK command-line tools only.
#   NDK=$HOME/Android/Sdk/ndk/<ver> ENGINE=<BigMoeOnEdge checkout with build-android-ref> [DEBUGGABLE=true] sh build.sh
# Models are NOT bundled; the app downloads/imports them.
set -e
: "${SDK:=$HOME/Android/Sdk}"; : "${DEBUGGABLE:=false}"
: "${ENGINE:?set ENGINE to the BigMoeOnEdge checkout containing build-android-ref}"
: "${NDK:?set NDK to the Android NDK root}"
D=$(cd "$(dirname "$0")" && pwd); B="$D/build"
BT=$(ls -d "$SDK"/build-tools/* | sort -V | tail -1); PL=$(ls -d "$SDK"/platforms/android-* | sort -V | tail -1)
rm -rf "$B"; mkdir -p "$B/classes" "$B/apk/lib/arm64-v8a"
# native payload: engine builds and the C probe suite
#   ENGINE_BUILD = portable armv8.2-a+dotprod+fp16 build (required); I8_BUILD = armv8.6-a+i8mm build (optional, same source).
#   The i8mm build's libraries are renamed in place (elf_rename.py) so both builds coexist; ComputeProbe picks one per device.
E="${ENGINE_BUILD:-$ENGINE/build-android-ref}"
L="$B/apk/lib/arm64-v8a"; STRIP=$(ls "$NDK"/toolchains/llvm/prebuilt/*/bin/llvm-strip | head -1)
cp "$E/cli/bmoe-cli" "$L/libbmoe_cli.so"
for l in libggml-base.so libggml-cpu.so libggml.so libllama-common.so libllama.so; do cp "$E/bin/$l" "$L/$l"; done
if [ -n "${I8_BUILD:-}" ]; then
  python3 "$D/elf_rename.py" "$I8_BUILD/cli/bmoe-cli" "$L/libbmoe_cli_i8.so" >/dev/null
  for l in libggml-base libggml-cpu libggml libllama libllama-common; do n=$(echo $l | sed -e 's/libggml/libg8ml/' -e 's/libllama/libl8ama/'); python3 "$D/elf_rename.py" "$I8_BUILD/bin/$l.so" "$L/$n.so" >/dev/null; done
fi
for f in "$L"/*.so; do "$STRIP" --strip-unneeded "$f"; done
# assets: the model catalogue (cards from real headers, app/test/CatalogMain.java) and the compute-probe calibration models
mkdir -p "$B/assets/calib"; cp "$D/assets/catalog.json" "$B/assets/catalog.json"
python3 "$D/../tools/make_calib_models.py" "$B/assets/calib" >/dev/null
sh "$D/../build_probes.sh" "$B/probes" >/dev/null
for p in devprobe dramprobe memprobe ufsbench wrbench; do cp "$B/probes/$p" "$L/lib$p.so"; done
sed "s/@DEBUGGABLE@/$DEBUGGABLE/g" "$D/AndroidManifest.xml" > "$B/AndroidManifest.xml"
"$BT/aapt2" link -o "$B/base.apk" -I "$PL/android.jar" --manifest "$B/AndroidManifest.xml" --min-sdk-version 29 --target-sdk-version 33
javac --release 8 -Xlint:-options -cp "$PL/android.jar" -d "$B/classes" "$D"/src/com/meridian/*.java
jar cf "$B/classes.jar" -C "$B/classes" .
"$BT/d8" --min-api 29 --lib "$PL/android.jar" --output "$B" "$B/classes.jar"
cp "$B/base.apk" "$B/unsigned.apk"
(cd "$B" && zip -q unsigned.apk classes.dex && zip -q -9 -r unsigned.apk assets && cd apk && zip -q -r ../unsigned.apk lib)
"$BT/zipalign" -f -p 4 "$B/unsigned.apk" "$B/aligned.apk"
KS="$D/debug.keystore"
[ -f "$KS" ] || keytool -genkeypair -keystore "$KS" -storepass android -keypass android -alias debug -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=Meridian Debug" >/dev/null 2>&1
"$BT/apksigner" sign --ks "$KS" --ks-pass pass:android --key-pass pass:android --out "$B/meridian.apk" "$B/aligned.apk"
ls -la "$B/meridian.apk"
