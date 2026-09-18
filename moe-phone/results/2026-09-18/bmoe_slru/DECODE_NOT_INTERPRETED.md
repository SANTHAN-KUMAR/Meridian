# SLRU campaign 2026-09-18 14:08: decode rates are not interpreted

**Retracted:** the "+4 to +7% decode" range for SLRU vs LRU (prose in commit acef356's message only; it
was never in the claims map or any table).

**Why:** an orphaned `host/agg_bandwidth.sh` driver ran from 13:40 to 14:35 (see
`../agg_bandwidth_INVALID_cpu_arm_on_host/driver.log`) and launched OLMoE `llama-bench` runs in the
`com.moephone.npu2` app throughout this campaign (14:08-14:31):
- Its bench rows completed at 14:24:52, 14:25:23, 14:26:14, 14:26:56, 14:27:16, 14:28:04, 14:28:51 and
  14:29:38, overlapping `slru_rep3a`, `slru_rep3b` and `lru_rep3b`.
- The earlier ones timed out after 10 minutes each. Whether they computed anything before a row's
  quiesce force-stopped the app is unknown.

Every row's rate is therefore suspect, not just the three with the documented overlap.

**Still valid:** the cache counters, which are the campaign's pre-stated primary outcome:
- `slru_phone_read_mib`: 106.53 vs 119.80 MiB/token.
- `slru_phone_hit`: 86.6 vs 85.3%.

Both are functions of routing, budget and policy, not of the clock. `lru_rep1b` decoded at 0.205
s/token (the slowest LRU row) and still read exactly 119.8 MiB/token, the same as every other
5000-MiB LRU row.

**Re-run:** `host/chain_after.sh`, pre-registered in its header. It adds a per-row `foreign=[...]` log
of any other benchmark process and a per-row memory gate.
