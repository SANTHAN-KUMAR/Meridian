# Stopped after repeat 1: patch 0011 list-corruption bug, not a lever result

Repeat 1 of the stacked comparison (binary 0013):

| row | decode | flash read | cache mgmt | stall |
|---|---|---|---|---|
| base | 6.05 / 5.69 tok/s | 119.8 MiB/token | 19 ms | 47 / 53 ms |
| stack | 3.17 / 3.18 tok/s | 256.3 / 239.7 MiB/token | 47 / 45 ms | 136 ms |

Flash MiB/token is a clock-independent counter. Separately, SLRU read 106.5 and selective adoption 130.7,
so a combined 2.0-2.1x is a mechanism, not noise. The run was stopped rather than spend ~50 min confirming
it.

**Cause.** `retain()` is called by the predictor for predicted experts that are already resident. It does
`lru_unlink(id); lru_push_front(id);`.
- Under `--expert-slru`, the unlink correctly takes a PROTECTED entry off the protected list.
- `lru_push_front` then links it into the PROBATION list, while `cprot_[id]` stays 1 and its bytes stay in
  `cprot_bytes_`.
- From then on, both lists' head and tail pointers are wrong for every later unlink and eviction.

The same pattern was in the staging loop's duplicate-id path. SLRU had only been measured without the
predictor, where `retain()` is never called, so the SLRU claims (106.53 vs 119.80 MiB/token) are
unaffected.

**Output was not affected.** The text of base_rep1a/b and stack_rep1a/b is identical (md5 4fce938e2f94).
stack_rep2a was cut short by the stop.

**Fix:** `tools/patches/0015-bigmoeonedge-slru-retain-keeps-protected-list.patch`. Both sites re-link a
protected entry into the protected list. The comparison was re-run on that binary with the
pre-registration unchanged.
