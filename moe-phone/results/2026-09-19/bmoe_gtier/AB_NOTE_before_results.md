# Note written 2026-09-19 02:40, before any row of the tier A/B (bmoe_gtier ab, v1 + spin, bmoe-i8mm-0022) was read

The smoke's base row (bmoe_gtier_smoke_20260919_0229, the first engine row after the stopped kgsl campaign) had every term
inflated: compute 150, mgmt 50, stall 80 ms/token. The tier row in the same smoke had 87 / 24 / 45 ms. Both rows ran at
the same caps (2.02 / 2.48 GHz at exit). Suspected cause: swap state left by the kgsl thrash, paid by the first row(s).
In ABBA the first row of repeat 1 is base, so any such residue biases repeat 1 against base.
Therefore, alongside the PRE-REGISTERED verdict (unchanged, all 6 repeats), a SENSITIVITY analysis excluding repeat 1 will
be reported, labelled as sensitivity. The verdict is the pre-registered one. Also reported: per-row
compute/mgmt/stall, and whether repeat 1's base rows are outliers against later base rows.
