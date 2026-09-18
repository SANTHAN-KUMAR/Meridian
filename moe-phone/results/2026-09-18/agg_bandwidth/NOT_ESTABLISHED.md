# The "pairs add" reading of this campaign is not established

The CPU arm is fixed in this run: it runs on the phone. Every row has a tg number.

The pre-registration (`../PREREG_zram_and_aggregate.md`, Q2) required the per-row start/end stamps to show
that the two members' windows overlapped before a positive result could be reported. They cannot show it:
- The stamps are host-side and bracket both members together (same start, same end second).
- Each member generates for only about 1.5 s (`-n 64 -r 1`).
- The two members' model loads differ by device: the app members first upload 3.7 GB into device buffers.

So a pair's CPU generation may have run mostly before the device member's generation began. The pair
sums (about 84 tok/s against a best solo of about 50) are not evidence that the devices add throughput.
They were quoted as such in conversation at 16:27 and are withdrawn. No claim was made from them.

Also unexplained: `solo_cpu_rep3` came in at 7.27 tok/s against 45-50 in the other CPU solo rows.

**Replacement:** `host/agg_overlap.sh`.
- The CPU member runs `-r 40` in jsonl for about 2 minutes, and the device member starts 30 s into it.
- Each repetition's wall interval is reconstructed from llama-bench's own jsonl timings.
- Only CPU repetitions fully inside the device's generation window count as concurrent.
