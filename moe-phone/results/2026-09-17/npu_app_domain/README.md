# Hexagon NPU: unreachable for an unprivileged third-party process on this ROM (2026-09-17)

llama.cpp's Hexagon backend (upstream 9f31776, built with Qualcomm's Hexagon SDK 6.6.0.0 image, HTP v81
skel built) registers the device (`HTP0`, arch v81 when `GGML_HEXAGON_ARCH=v81`; its own capability query
fails with the same code) but **every session open fails with 0x72 = AEE_ECONNREFUSED**, in three domains:

| caller | SELinux domain | result |
|---|---|---|
| `adb shell` | `u:r:shell:s0` | `avc: denied { open } /dev/fastrpc-nsp1000`, then HAL fallback refused |
| `adb shell run-as com.moephone.npu` | `u:r:runas_app:s0` | same 0x72 |
| a normal app process (`com.moephone.npu2`, targetSdk 34, launches the binary itself) | `u:r:untrusted_app:s0` | `open_device_node: no access to default device of domain 3, open thru HAL` -> `errno 13, Permission denied` -> `apps_dev_init failed ... Error 0x72` |

`remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)` succeeds in all three; the refusal is at device-node
open and at the vendor HAL fallback. **Correction, same evening (23:51):** a THIRD-PARTY app does reach HTP v81 on this phone. With logcat
captured while Geekbench AI (com.primatelabs.banff, uid 10xxx, installed from Play, requesting only INTERNET
and ACCESS_NETWORK_STATE) ran its AI benchmark, its own process opened the session through the vendor HAL and
loaded the skel from its APK:
  I/dsp-client(28094): DspClient.cpp (244): close_hal_session: closed device fd 178 on domain 3
  fastrpc_apps_user.c:3613: remote_handle64_close: ... name libQnnHtpV81Skel.so
  dspqueue_cpu.c:981: dspqueue_close: closed Queue 0 ... for domain 3
So the HAL path is available to an ordinary app, and no special manifest permission is involved. What differs
for us is not established: candidates are (a) our engine runs as a CHILD process spawned by the app rather than
in the app process itself, (b) the HAL admits QNN's client library (libQnnHtp + dspqueue) but not a direct
FastRPC session open from ggml-hexagon, (c) a vendor-side condition we have not identified. Status: OPEN, not
closed. Next steps that stay inside this project's rules: report the device, ROM and these logs upstream
(ggml-org/llama.cpp Hexagon backend, qualcomm/fastrpc), and test the QNN client path rather than a raw
FastRPC session. The earlier wording in this file said the branch was closed by platform policy; that was
wrong and is replaced by this paragraph.
