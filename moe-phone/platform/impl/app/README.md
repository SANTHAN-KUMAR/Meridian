# Meridian Android app

One self-contained arm64 APK: the inference engine, the device-probe suite, the model planner, the agent runtime and the UI.
Models are **not** bundled; the app downloads them (paste a direct https link to a `.gguf`), or imports one from storage.

## Install and use

```sh
adb install -r app/dist/meridian.apk        # or copy the APK to a phone and open it (allow "install unknown apps")
```

1. **Device** tab -> *Profile this device* (a few minutes; keep the app in front, phone unplugged). It measures memory, storage and
   DRAM with the C probes, then runs the **compute probe**: synthetic calibration models through the real engine, per weight
   type, and picks the faster engine build for this CPU. Every figure is labelled `[measured]` only if the whole run was
   unplugged, awake and in front; otherwise `[prior]`.
2. **Models** tab -> *Recommended for this phone*: every catalogue model with the tier it would run in (resident or streamed),
   its predicted speed range and basis, and time to first reply, ranked by the lower end of the range. *Download* fetches it
   (resumable, SHA-256 verified before use). *Evaluate URL* does the same for any https `.gguf` by reading only its header;
   *Search Hugging Face* lists GGUF repos and files to evaluate or download. Nothing is promised without a measurement behind it.
3. **Use this model** (or Chat on a downloaded model): the app plans the configuration (tier, context, threads, CPU mask,
   expert cache, engine build) and loads it. Each turn shows the observed speed next to the prediction; the Governor flags the
   plan if observed speeds leave the predicted range.
4. **Agent** tab -> ask for something in your own words. The agent rewrites it as a direct instruction, uses the phone's tools
   step by step (apps, alarms, timers, calendar, messages and calls via the composer/dialer, email, maps, music and media keys,
   flashlight, volume, brightness, settings panels, clipboard, contacts, calculator, time, device info, notes), checks each
   result against the phone's real state, and ends every answer with the list of actions actually taken. Anything that reaches
   other people asks first, every time. It cannot operate inside other apps' screens (see STATUS.md).

Requirements: Android 10+ (API 29), arm64, a CPU with dotprod and fp16. An i8mm engine build is also bundled and used when it
measures faster on the device.

## Build

```sh
NDK=$HOME/Android/Sdk/ndk/<ver> ENGINE=<engine source> ENGINE_BUILD=<armv8.2 dotprod build dir> I8_BUILD=<armv8.6 i8mm build dir, optional> sh app/build.sh
# DEBUGGABLE=true adds a debug-only adb automation hook and allows cleartext http (for LAN download tests). Release: DEBUGGABLE=false.
```
The engine is BigMoeOnEdge 74ba18f + `tools/patches/0019-SNAPSHOT` + `0020` (per-request grammar) + `0021` (history reset that keeps
the KV cache). The payload must include the grammar patch (the app refuses to run the agent on an engine
whose `BMOE_READY` lacks `"grammar":true`). Tests: `sh app/test/run_gates.sh` (host gates for the planner, predictor, probe fit and topology rule). The catalogue asset is
rebuilt with `app/test/CatalogMain.java` from `tools/catalog_raw.json`.

See [`../STATUS.md`](../STATUS.md) for the conformance level and every registered stub.
