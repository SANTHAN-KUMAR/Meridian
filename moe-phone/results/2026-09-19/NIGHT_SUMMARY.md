# Night of 2026-09-18/19: engine session summary (written 01:05, updated as runs land)

Level: L3 (what can the GPU and memory levers deliver at all?) and L5 (engine plumbing). No new tok/s claim was made tonight.

## Measured on the phone
| what | result | artifact |
|---|---|---|
| gtier engine smoke (gx v1, 380 warm slots) | text identical 2/2; 0 device failures; 0 risk recomputes; 2845 experts on device (1.61 per dispatch); CPU-side wait 0.24 ms per dispatch | results/2026-09-18/bmoe_gtier/bmoe_gtier_smoke_20260919_0004 |
| first smoke | invalid: CANNOT LINK (LD_LIBRARY_PATH included /vendor/lib64); script fixed | .../bmoe_gtier_smoke_20260918_2344/FAILED.txt |
| gx kernel, M6 run 3 (gx session) | best is v0 + spin-wait: k=1 0.58 / 0.27 ms, k=3 1.06 / 0.72 ms (host / device); v2 regressed ~10x (0.85 GB/s, a page-locality hypothesis) | results/2026-09-19/gx_m6_001107 |

## Built and checked on the laptop (OLMoE; no Qwen3 on the laptop)
- The engine takes gx variants 2 and 3 (repack on slot writes, unpack for the CPU paths). v2 and v3 match v1 byte for byte,
  with 0 risk flags (results/2026-09-19/gtier_v2_laptop). The first v3 rows were void: my variant check wrote v3 slots
  unrepacked. The gx session found this; it is fixed and recorded in that README.
- `--arena-kgsl`: the slot arena in OpenCL ALLOC_HOST_PTR memory, used by the CPU only. The aim is to keep the anonymous arena's
  -18 ms of cache management without its +14 ms of zram-fault compute. Text is identical to off and to the stack
  (results/2026-09-19/arena_kgsl_laptop). A phone run is gated on host/pinned_arena/pinprobe.cpp (pre-registered).

## The ceiling (L3, arithmetic on committed numbers)
- The GPU tier covers 2.19 of 8 experts per layer (gtier_size). Expert arithmetic is about half of compute. So even a
  zero-time GPU saves at most ~14 ms/token: +10-13% on the 5.5-6.3 tok/s stack.
- A whole-cache CPU+GPU split (zcbench: 1.535x aggregate) would be worth ~16 ms/token. It is not built: the CPU's cache
  would have to be GPU-visible without map/unmap (measured at 1.4 / 2.7 ms per call).
- The pinned arena, if it works, is worth up to ~18 ms/token (the mgmt term).
- None of these reaches 10 tok/s alone. Together, at best, they reach ~7-8 at capped clocks. This agrees with
  research/2026-09-18_ceiling_handoff.md.

## What stopped the night
- **Phone adb was lost at ~00:38**, right after the gx session's run 3 exited cleanly (exit=0, normal thermal state at
  00:37:30). Afterwards 192.168.0.65 answers ping, but port 5555 refuses, no other port is open, and no mDNS service exists.
  **Whether the phone rebooted is unknown.** host/chain_night2.sh logs /proc/uptime the moment adb returns.
- Nothing ran on the phone after 00:38. Restoring adb needs the user (USB, then `adb tcpip 5555`, or re-enable Wireless
  debugging).

## Queued (host/chain_night2.sh, waiting for adb)
1. pinprobe (~1 min).
2. The gx session's run 4 (memory probe plus the tiled v3 kernel).
3. The --arena-kgsl smoke, then ABBA x6, only if pinprobe says BUILD.
4. GPU tier v0+spin: smoke (identical text, risk <= 1%), then ABBA x6.
Each row is refused below 25% battery. The shared phone lock serialises the two sessions.
