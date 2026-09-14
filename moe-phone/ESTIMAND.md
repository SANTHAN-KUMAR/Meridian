# ESTIMAND.md — what moe-phone measures, exactly

Prerequisite for any engine work (`CLAUDE.md` §2). This is a **systems** project, so the "estimands"
are measured quantities of an engine–model–device triple. The same discipline applies: every
quantity is defined, its averaging measure is stated, and the gap between what we measure and what
we claim is written down (§6).

A run is a triple **(engine e, model m in format f, device d with memory budget M)**.

## 1. Decode rate `r`

`r(e, m, f, d, M)` = generated tokens ÷ wall-clock seconds over the decode phase, excluding prefill,
for a fixed prompt set (§5), batch size 1, greedy decoding.

- **Steady state**: tokens 65–512 of each generation (the first 64 absorb cache warm-up and are
  reported separately as the cold-start rate).
- **Sustained**: the same, measured after 30 minutes of continuous generation; reported alongside
  steady state, never instead of it.
- Reported as median and IQR over ≥ 5 runs per prompt, plus the full distribution. A single run is
  never a result.

## 2. Flash traffic

- `B_flash` — bytes read from storage per decoded token, measured from the process's
  `/proc/<pid>/io` `read_bytes` (block-layer reads; page-cache hits excluded) — cross-checked against
  the engine's own read log (`CLAUDE.md` §6.5: a derived number is measured two ways).
- `N_req` and the **request-size distribution** per token — because on UFS, time depends on request
  size, not only on bytes (`POSITION.md` A3).
- `h` — expert-cache hit rate, per layer and overall.

## 3. Fidelity

- **Tier E (exact)**: for every token of the evaluation corpus under teacher forcing, the engine's
  logits equal the reference engine's (llama.cpp CPU, same weights, same format) bit-for-bit, or
  within floating-point reordering tolerance declared in advance. Claims of "no quality loss" are
  Tier-E claims.
- **Tier A (approximate)**: mean and 99th-percentile **KL(p_BF16 ‖ p_engine)** per token under
  teacher forcing on a held-out corpus; top-1 **flip rate**; and task accuracy on a fixed suite.
  The KL margin is fixed **before** any sparsity result is seen, from the KL that the published
  4-bit format itself introduces relative to BF16 (the "format floor"): an approximate method is
  "within margin" if it adds no more KL than the quantization everybody already accepts. This
  derives the margin from an external reference point rather than from our results (`CLAUDE.md`
  §6.4).
- The reference for stock models is the model's own BF16 checkpoint. For TurboSparse-Mixtral the
  reference is TurboSparse-Mixtral, not Mixtral — its retraining is PowerInfer-2's choice, and its
  quality relative to Mixtral is their claim, reported as theirs.

## 4. The scale frontier `S(τ)`

`S(τ)` = the largest total-parameter count among candidate models for which, on the 15R at budget
M, steady-state `r ≥ τ` **and** fidelity is Tier E or within the Tier-A margin.

`τ` is derived, not chosen: silent reading rate for English non-fiction is ~238 words/min
(Brysbaert, *Journal of Memory and Language* 109:104047, 2019), converted to tokens with each
model's **own measured** tokens-per-word ratio on the evaluation corpus. So `τ` is model-specific
and nobody picks it by looking at results. `S` is reported as a curve over τ too, so a reader can
apply their own threshold.

## 5. Averaging measure

- Prompt set: fixed, committed, mixed lengths (short chat, long-form generation, code), drawn before
  any engine exists. Context lengths 512 and 4096 reported separately (KV-cache traffic grows with
  context and competes for DRAM).
- Device state: airplane mode, fixed brightness, battery 50–80%, ambient temperature recorded;
  thermal state recorded at start and end of every run.
- Memory budget M: enforced by the engine's own accounting **and** checked against the process's
  proportional set size (PSS); PowerInfer-2's budgets (7–19 GB "available memory") are matched by
  the same definition, and any mismatch in accounting is stated.

## 6. The gap between what we measure and what we claim

- **Steady-state vs cold start.** Expert caches are cold at the start of a conversation. A
  steady-state-only number flatters every caching method; both are reported.
- **Budget accounting.** "7 GB" in PowerInfer-2 is their memory budget on a 24 GB phone. On a 12 GB
  phone the OS takes a larger share and the low-memory killer acts sooner; the same nominal budget
  is not the same operating point. The head-to-head is stated at matched *budgets*, with the device
  difference named as a limitation rather than hidden.
- **Prompt distribution.** Routing locality — and therefore hit rate — depends on the text. Rates are
  reported per prompt category; a single pooled number is not the claim.
- **Energy.** Whole-device energy includes the display and radios; per-rail energy (if Perfetto rails
  exist on the 15R, gate G0) isolates SoC and storage. Which one is reported is always stated.

## 7. Estimability (L3) of the comparisons

A speedup is reported only if the run-to-run spread lets it be distinguished: the IQR of the
two conditions must not overlap, or a paired difference over prompts must have a 95% interval
excluding zero. Before any campaign, G1's repeat measurements set the device's run-to-run noise;
the number of runs per condition is chosen from that noise, not from the results.
