#!/bin/sh
# Host-JVM gates for the generalization layer (GatesMain.java). org.json comes from Android Studio's layoutlib.jar.
set -e
D=$(cd "$(dirname "$0")/.." && pwd); LL=${LAYOUTLIB:-/opt/android-studio/plugins/design-tools/lib/layoutlib.jar}
AJ=$(ls -d ${SDK:-$HOME/Android/Sdk}/platforms/android-* | sort -V | tail -1)/android.jar   # compile-time API; runtime org.json is layoutlib's real one
[ -f "$LL" ] || { echo "SKIP: layoutlib.jar (org.json) not found"; exit 0; }
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
javac -nowarn -cp "$AJ" -d "$T" "$D"/src/com/meridian/Gguf.java "$D"/src/com/meridian/Planner.java "$D"/src/com/meridian/Predictor.java \
  "$D"/src/com/meridian/Topo.java "$D"/src/com/meridian/ComputeProbe.java "$D"/src/com/meridian/PlanV2.java "$D"/src/com/meridian/Engine.java \
  "$D"/src/com/meridian/Cells.java "$D"/src/com/meridian/Calibration.java "$D"/src/com/meridian/Chat.java "$D"/src/com/meridian/Native.java \
  "$D"/src/com/meridian/Profile.java "$D"/src/com/meridian/Regime.java "$D"/test/GatesMain.java 2>&1 | grep -v '^Note' || true
java -cp "$T:$LL" com.meridian.app.GatesMain
# agent tools' pure layer (Parse.java): time/date parsing, distances, WMO/ISO tables, the sensitive-tap rule, RSS
T2=$(mktemp -d); javac -nowarn -d "$T2" "$D"/src/com/meridian/Parse.java "$D"/test/ToolGatesMain.java 2>&1 | grep -v '^Note' || true
java -cp "$T2" com.meridian.app.ToolGatesMain; rm -rf "$T2"
