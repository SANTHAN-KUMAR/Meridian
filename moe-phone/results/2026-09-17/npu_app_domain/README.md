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
open and at the vendor HAL fallback. Apps that DO reach HTP v81 on this phone (G0-NPU, 2026-09-14: Google
AICore, OnePlus Gallery, Geekbench AI) are system-privileged or OEM-permitted. Conclusion for this project's
regime (unrooted, third-party): the NPU branch is closed by platform policy, not by the backend, the skel
version, the domain name, `ADSP_LIBRARY_PATH`, or `security.perf_harden`. Reopening it would need an
OEM-permitted app identity or a rooted device - both outside the regime this project reports in.
