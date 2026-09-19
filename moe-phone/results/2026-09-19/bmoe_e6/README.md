# E6 — Tier-A quality frontier (research spec E6; binding bar = spec §0 item 4 "negligible"). VERDICT: no option is negligible

KL phase: `bmoe_e6_kl_20260919_1653/`, scored by `gates/e6_score.py` into `../e6_summary.json` / `../e6_summary.txt`. Reference: Q8_0 (sha256 a68fe734…).
Corpus: `../e6_corpus` (3,258 scored tokens, --ppl-step, -c 4096, K=8192, largest reference tail 0.052).

| arm | PPL | mean KL | p99 KL | flips | negligible (all four KL/PPL bars) | old 2x rule |
|---|---|---|---|---|---|---|
| FLOOR Q4_0 top-8 | 3.977 | 0.118 | 1.76 | 323 (9.9%) | margin | margin |
| T7 | 4.043 | 0.146 | 2.31 | 347 (10.7%) | no (mean, p99, PPL) | pass |
| T6 | 4.196 | 0.210 | 2.87 | 423 (13.0%) | no (all four) | pass |
| RA1 | 4.080 | 0.168 | 2.60 | 352 (10.8%) | no (mean, p99, PPL) | pass |
| DC05 | 4.015 | 0.126 | 2.06 | 315 (9.7%) | no (p99 only) | pass |
| DC10 | 4.184 | 0.209 | 3.10 | 398 (12.2%) | no (all four) | pass |
| SUB | 4.048 | 0.163 | 2.60 | 369 (11.3%) | no (all four) | pass |

- Heat: the REF row ended at thermal status 3 / shell 49 C. From then on the rows ran screen-off with a cooldown gate (the script
  REVISION 17:40). FLOOR's first attempt was stopped unscored and rerun. Quality numbers do not depend on clock or wake state.
- The knowledge probe (`bmoe_e6_mc_20260919_1938/`) was launched by the chain from the stale 2x arm list (bash had buffered the chain's
  lines before the negligible-bar edit reached the file) and was stopped by the agent, unscored. No arm is negligible on the KL/PPL
  measures, and the knowledge criterion can only add failures, so the probe cannot change the verdict.
- The phone-side determinism control (`bmoe_e6_ident_*`, Q4_0 scored against itself) runs next. It must read KL 0, 0 flips.
