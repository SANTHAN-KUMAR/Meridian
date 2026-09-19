# Head-to-head vs BigMoeOnEdge (same session, same phone, same Q4_0 file): ours is 1.22x, decisive

Run `bmoe_h2h/bmoe_h2h_ab_20260919_2155/` (pre-registered in `device/bmoe_h2h.sh`; unplugged on Wi-Fi adb, held Awake; ABBA x6, 24/24 rows
kept by `stack_summary.py --any-budget-prereg --require-awake` -> `h2h_ab_summary.json`).
- BigMoeOnEdge: the reference build (bmoe-ref, armv8.2-a+dotprod+fp16) with its documented command (cache ceiling 4000 MiB): **5.46 tok/s**.
- Ours (engine bmoe-i8mm-0027, the plain stack: pinned compute cpu4-7 and I/O cpu0-3, SLRU, predictive prefetch with selective adoption,
  cache ceiling 5000; our arm was "plain" because the repacked kernels failed their pre-registered fidelity gate): **6.67 tok/s**.
- **Ratio 1.22 (+1.21 tok/s), faster in 6/6 repeats, SE/|diff| = 0.10: decisive.**
- **Retraction:** the earlier "~1.7x the published result" compared our 6.76 (awake, unplugged, 09-19) with the reference's 3.98
  (on the charger, 09-17). The controlled number is 1.22x. The ~1.7x was deleted from the summaries per CLAUDE.md §7.6.
- Text: each engine is deterministic (12/12 identical rows within each), but the two engines' texts differ from word 71 onward. They are
  different builds (armv8.2 dotprod vs armv8.6 i8mm) running the Q4_0 dot products through different instruction paths:
  floating-point reordering, not a model change. A "same output as the reference" claim would need a KL check or the same build flags.
  An earlier measurement showed the build flags alone give the reference no speed (`bmoe_i8mm_no_gain`), so the 1.22x is attributable
  to the engine changes.
- The speed advantage is smaller in this session partly because the reference itself ran much faster awake and unplugged (5.46) than in
  its 09-17 reproduction on the charger (3.98). Conditions matter as much as engines, which is the benchmark-validity point.
