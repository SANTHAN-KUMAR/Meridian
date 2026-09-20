# Meridian Android app

One self-contained arm64 APK: the inference engine, the device-probe suite, the model planner, the agent runtime and the UI.
Models are **not** bundled; the app downloads them (paste a direct https link to a `.gguf`), or imports one from storage.

## Install and use

```sh
adb install -r app/dist/meridian.apk        # or copy the APK to a phone and open it (allow "install unknown apps")
```

1. **Device** tab -> *Profile this device* (about 2 minutes; keep the app in front, phone unplugged). Every figure is labelled
   `[measured]` only if the run was unplugged, awake and this app was in front for the *whole* run (sampled every second);
   otherwise `[prior]`. Nothing is guessed: unmeasured fields say so.
2. **Models** tab -> download or import a `.gguf`. Each model card shows the exact byte budget derived from the file and a
   feasibility verdict. Verdicts are `Infeasible` (only by a sound lower-bound argument) or `NotCalibrated` naming what is
   missing; the app never promises a speed it has not measured.
3. **Device** tab -> *Calibrate thread placement* (about 3 minutes, needs a selected model): A/Bs CPU masks with the real engine.
4. **Chat** tab -> *Load engine*, then talk. Observed tokens/s are shown per turn and recorded.
5. **Agent** tab -> give it a task. Plan -> constrained tool calls -> each result verified against device state -> answer.
   Irreversible tools (`notes_clear`) ask for consent every time.

Requirements: Android 10+ (API 29), arm64, and a CPU reporting dotprod and fp16 (the bundled engine is built for
armv8.2-a+dotprod+fp16; the app refuses with `EngineUnsupported` otherwise).

## Build

```sh
NDK=$HOME/Android/Sdk/ndk/<ver> ENGINE=<moe-work/BigMoeOnEdge checkout> ENGINE_BUILD=<dir with cli/bmoe-cli and bin/*.so> sh app/build.sh
# DEBUGGABLE=true adds a debug-only adb automation hook and allows cleartext http (for LAN download tests). Release: DEBUGGABLE=false.
```
The engine payload must include the grammar patch `tools/patches/0020-*.patch` (the app refuses to run the agent on an engine
whose `BMOE_READY` lacks `"grammar":true`). Tests: `python3 app/test/test_parity.py` (Java planner vs Python planner on real GGUFs).

See [`../STATUS.md`](../STATUS.md) for the conformance level and every registered stub.
