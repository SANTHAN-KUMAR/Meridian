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
# native payload: engine (portable armv8.2-a+dotprod+fp16 build) and the C probe suite
E="${ENGINE_BUILD:-$ENGINE/build-android-ref}"
cp "$E/cli/bmoe-cli" "$B/apk/lib/arm64-v8a/libbmoe_cli.so"
for l in libggml-base.so libggml-cpu.so libggml.so libllama-common.so libllama.so; do cp "$E/bin/$l" "$B/apk/lib/arm64-v8a/$l"; done
sh "$D/../build_probes.sh" "$B/probes" >/dev/null
for p in devprobe dramprobe memprobe ufsbench wrbench; do cp "$B/probes/$p" "$B/apk/lib/arm64-v8a/lib$p.so"; done
sed "s/@DEBUGGABLE@/$DEBUGGABLE/g" "$D/AndroidManifest.xml" > "$B/AndroidManifest.xml"
"$BT/aapt2" link -o "$B/base.apk" -I "$PL/android.jar" --manifest "$B/AndroidManifest.xml" --min-sdk-version 29 --target-sdk-version 33
javac --release 8 -Xlint:-options -cp "$PL/android.jar" -d "$B/classes" "$D"/src/com/meridian/*.java
jar cf "$B/classes.jar" -C "$B/classes" .
"$BT/d8" --min-api 29 --lib "$PL/android.jar" --output "$B" "$B/classes.jar"
cp "$B/base.apk" "$B/unsigned.apk"
(cd "$B" && zip -q unsigned.apk classes.dex && cd apk && zip -q -r ../unsigned.apk lib)
"$BT/zipalign" -f -p 4 "$B/unsigned.apk" "$B/aligned.apk"
KS="$D/debug.keystore"
[ -f "$KS" ] || keytool -genkeypair -keystore "$KS" -storepass android -keypass android -alias debug -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=Meridian Debug" >/dev/null 2>&1
"$BT/apksigner" sign --ks "$KS" --ks-pass pass:android --key-pass pass:android --out "$B/meridian.apk" "$B/aligned.apk"
ls -la "$B/meridian.apk"
