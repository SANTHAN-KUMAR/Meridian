# Why this thread sweep is invalid

Every row ran as a new `am start` intent inside ONE long-lived `com.moephone.npu2` process. The app was
never force-stopped between rows. By 15:06 that process held:
- 3.0 GB RSS and 21 GB of virtual memory;
- 685% CPU.

MemAvailable was 1.9 GB. It went back to 7.3 GB the moment the app was force-stopped.

The rates degrade with position in the run, not with thread count. The t6 arm went 49.18 (rep1) ->
44.42 (rep2) -> 0.90 (rep3), and t8 went 31.87 -> 4.80. This is the same defect class as the voided NPU
knob sweep (state carried across arms in one process).

The driver was stopped at 15:08 after 9 rows. No number from this directory is quoted.
`host/thread_sweep.sh` now force-stops the app before every row and logs the free memory each row
started with. It also takes a lock.
