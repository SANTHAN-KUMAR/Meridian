# `moe-phone/host/` — the laptop side of a phone measurement

Everything under [`../device/`](../device/) runs *on* the phone in the `shell` domain. The scripts here
run on the laptop and drive runs that **cannot** happen in that domain: anything that needs the Hexagon
NPU, which only opens a DSP session inside an app's own process on this device.

| file | what it does |
|---|---|
| [`build_npu_app.sh`](build_npu_app.sh) | builds the engine for Android with the Hexagon + OpenCL backends, compiles the JNI shim, packages both into an APK, installs it and prints the in-process HTP probe. The comment block at the top is the argument for why an app is required at all. |
| [`app/`](app/) | the app's entire source: manifest (with the `<uses-native-library>` entries that make the session open), the two Java classes, and the JNI shim that `dlsym`s `bmoe_main` / `llama_bench` / `llama_completion` in-process. |
| [`overnight_npu.sh`](overnight_npu.sh) | upstream llama.cpp (the PR #25294 streaming branch) in-app, across HTP / OpenCL / CPU. Produces `results/<date>/app_engine/`. |
| [`bmoe_attn_ab.sh`](bmoe_attn_ab.sh) | our engine in-app, A/B'd on `--attn-device` (CPU vs HTP0 vs GPUOpenCL) with the flag string of the best measured CPU cell. Produces `results/<date>/bmoe_attn_ab/`. |
| [`chain_attn.sh`](chain_attn.sh) | sequences the above so two campaigns never share the flash, and copies the model into app storage (an app cannot read `/data/local/tmp`). |

Analysis of what they produce: [`../gates/app_engine_analyze.py`](../gates/app_engine_analyze.py) (rates,
failure counts, per-arm medians) and [`../gates/device_bandwidth.py`](../gates/device_bandwidth.py) (the
weight-byte throughput each device achieves, and the ceiling that implies for a target model).

## Two things to know before reading any number these produce

1. **In-app is not the same environment as `adb shell`.** An app is subject to a different SELinux
   domain, a different memory-pressure regime (the low-memory killer treats it as a foreground app, not
   as a shell process) and Doze. Rows from `bmoe_attn_ab.sh` are therefore comparable to each other, and
   to the `device/` rows only with that caveat stated. `bmoe_attn_ab.sh` uses the same flag string as the
   best `device/bmoe_cache.sh` cell precisely so that the app-vs-shell difference is the only thing left
   over when the attention device is held at CPU.
2. **The model is copied, not shared.** Each app package needs its own copy in its own `files/`
   directory, because `untrusted_app` cannot read `/data/local/tmp`. Two app packages each holding a
   copy of the model costs twice the model's size in phone storage on top of the original;
   `chain_attn.sh` checks the copy's size against the source and refuses to run the A/B if they differ,
   rather than measuring a truncated model.
