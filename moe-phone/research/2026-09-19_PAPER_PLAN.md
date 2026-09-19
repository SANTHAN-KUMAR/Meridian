# Paper plan (the next phase after the 2026-09-19 closure): from a one-phone study to a top-venue paper

Agreed with the user on 2026-09-19. Targets: MobiSys / MLSys / EuroSys, with a workshop version (e.g. EdgeFM) as the fallback.
Inputs: `2026-09-19_CLOSURE.md` (what is proven), `2026-09-19_paper_framing.md` (framings), `2026-09-19_RESEARCH_SPEC.md` §9 (method rules).

## 1. The claim
> **Sustained, not burst: the real performance ceiling of flash-streamed MoE LLMs on phones, a model that predicts it for any
> device and model, and an engine that reaches it.**

Contributions:
1. A decomposition of a decode token into compute, stall and cache management, measured under sustained thermal load with the device
   held awake. The instruments exist: the replay compute floor `--fixed-routing`, the KL fidelity tool, and oracle routing.
2. An analytical performance model: tok/s = f(device probes, model shape). Pre-registered predictions before each device/model
   measurement, then validated. This is also the product's hardware-detection core (see memory: moe-phone-long-term-goals).
3. The engine, with a clean ablation, where every row is measured wall time:
   - core placement: +35% (`pin2_*`);
   - cache size and policy;
   - prefetch with selective adoption;
   - repacked kernels (−20.7 ms compute, E1). Only if they pass the negligible KL bar (see §5.1); they failed the 0.2% PPL gate.
4. What does not work, and why, each with its measured mechanism:
   - the GPU as a helper (E4) and as the main engine (E7);
   - the NPU;
   - draft speculation;
   - the routing-shortcut quality frontier (E6);
   - memory relocation levers.
5. Benchmark validity: unplugged phones autosuspend mid-run, and the clock cap tracks skin temperature. We show the error this
   introduces in naive numbers and release the controlled harness.

## 2. Device matrix (user-confirmed availability)
| device | SoC | RAM | role |
|---|---|---|---|
| OnePlus 15R | Snapdragon 8 Gen 5 class (SM8845) | 12 GB | primary (done) |
| Samsung Galaxy S25 | Snapdragon 8 Elite (SM8750) | 12 GB | same vendor, different thermal design and OEM governor |
| OnePlus Nord CE (Snapdragon 888 class, to be confirmed) | older Snapdragon | 12 GB | an older generation: the model's range |
| a MediaTek Dimensity phone | Dimensity (to be chosen) | ≥ 12 GB | a different vendor (CPU, GPU and storage stack) |
| iPhone 16 Pro | A18 Pro | 8 GB | **stretch**: needs a Metal backend and an iOS app; a separate porting project |
| laptop / PC | x86 + NVIDIA | 16 GB | the "PC" row for the product story; no phone thermal constraints |

## 3. Model matrix
| model | total / active | status |
|---|---|---|
| Qwen3-30B-A3B | 30B / 3B | done on the 15R |
| gpt-oss-20b (MXFP4) | 21B / 3.6B | downloaded (`moe-work/models/gpt-oss-20b-MXFP4.gguf`, also on the phone) |
| OLMoE-1B-7B | 7B / 1B | on hand; the small-model anchor for the performance model |
| DeepSeek-V2-Lite | 16B / 2.4B | to download |
| one larger model (e.g. Qwen3-Next-80B-A3B, or another 100B-class MoE) | 80B+ / ~3B | the ">5 tok/s for bigger models" goal |

## 4. Protocol per (device, model) cell (fixed; the tree terminates per cell)
1. Probes (minutes): G0/G1 storage, DRAM, compute, thermal-cap curve. Then the performance model's **pre-registered prediction**.
2. Sustained decode, 10 min, awake, unplugged, 2 baselines + our stack, ABBA: tok/s, **J/token** (`current_now × voltage_now`),
   caps and skin temperature.
3. The replay compute floor (E1 method) and oracle-prefetch stall (E2 method), only on the 2 primary models.
4. Quality: KL vs **BF16** (a rented GPU for the reference logits), plus 2–3 task benchmarks, for every non-bit-identical configuration.
The prediction error of step 1 vs step 2 across all cells is the paper's model-validation result.

## 5. Baselines and open items
- Baselines: plain llama.cpp (mmap), BigMoeOnEdge (the same-session head-to-head, running on the 15R on 2026-09-19), upstream llama.cpp
  expert-streaming, and MLC-LLM / ExecuTorch where they can run the model.
- 5.1 Repacked kernels: failed the pre-registered 0.2% PPL gate (−0.33% PPL, i.e. better, on 616 tokens). The next check is the
  E6 KL bar against Q8_0/BF16. If negligible, they are reported as "negligible-loss" with their compute gain; otherwise they are excluded
  from the lossless headline.
- Novelty: redo `2026-09-19_novelty_search.md` as a verified-absent search (ACM/IEEE venues, arXiv, llama.cpp and MLC issues), dated.
- Artifact: open-source the engine patches, harness and scripts for artifact evaluation.

## 6. Order and effort (about 3–5 weeks)
1. gpt-oss-20b on the 15R (protocol §4); build the performance model from the 15R data.
2. The S25 and the Nord CE (all models), then the Dimensity; each device starts with its pre-registered predictions.
3. BF16 references and task benchmarks on a rented GPU; energy on every cell.
4. Write-up. The iPhone and the larger model only if time allows.
