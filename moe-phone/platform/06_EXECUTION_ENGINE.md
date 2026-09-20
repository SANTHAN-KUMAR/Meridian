# 6 — The execution engine: tiers, placement, residency, MoE streaming, memory, storage, fidelity

Layer A of [`02_ARCHITECTURE.md`](02_ARCHITECTURE.md). This is the generalisation of the `moe-phone`
engine — a fork of BigMoeOnEdge over llama.cpp — into the compute layer of the platform. Most of
its parts exist and were measured; this document says which, what each becomes, what is added, and
what was tried and must not be re-tried without new evidence.

---

## 1. Two tiers, one engine

The engine serves two kinds of configuration through one code path:

| tier | definition | what dominates cost | measured on the primary device |
|---|---|---|---|
| **resident** | every weight the forward pass touches is in RAM for the session | arithmetic and placement; no storage traffic at steady state | a 1B-active MoE at 44.5 [E:olmoe_cpu_decode] (CPU), 50.0 [E:olmoe_gpu_decode] (GPU), 40.7 [E:olmoe_npu_decode] (NPU) tok/s decode |
| **streamed** | routed experts exceed RAM; the always-needed part is resident and experts are read from storage per token, through a cache | compute at the sustained clock, then un-hidden storage stalls, then cache management | a 30B-A3B MoE at 6.68 tok/s [E:qwen3_h2h_ours], with a lossless ceiling of 7.8 [E:qwen3_e1_close_tok_s] on this device; a second 20B-class MoE at 4.3 [E:gptoss_decode] |

The tier is a **property of (model, device, grant)**, not of the model alone: the same checkpoint
is resident on a device that grants enough memory and streamed on one that does not. The
`ModelRegistry` computes feasibility per device from `resident_bytes`, `expert_bytes` and
`grantable_foreground`; the engine is told which tier it is running and never guesses.

**The resident tier is the platform's default for agent work** ([`00_PROBLEM.md`](00_PROBLEM.md) §2).
The streamed tier is the escalation path. Both must be first-class in the engine; the prior project
built only the second.

---

## 2. Module map: what exists, what it becomes

The existing engine is a ports-and-adapters layer over llama.cpp's public API. The table maps its
source to the platform components of [`02_ARCHITECTURE.md`](02_ARCHITECTURE.md) §3.3 so an
implementer knows where to start reading and what is new.

| existing engine module | platform component | change |
|---|---|---|
| `core/include/bmoe/recipe.h`, `core/src/moe/arch_registry.cpp` | `ModelRegistry` architecture seam | keep as is. A new MoE family is one table row naming its expert-tensor suffixes; everything else is discovered from the file |
| `core/src/engine/session.cpp`, `session.h` | `SessionManager` ↔ `EngineSession` | keep: model load once, expert cache warm across turns, KV prefix reuse via `clear_kv=false`. Add: lease shrink and checkpoint hooks |
| `cli/main.cpp` `--session` loop (JSON requests over stdin) | the engine-process protocol | keep the shape; formalise the request/response schema (§11) |
| `core/src/moe/expert_stream_source.cpp` | `ExpertStream` | keep: per-layer cache, LRU/SLRU, event-atomic fetch, overlap lanes, selective adoption of speculative reads. Parameterise lanes, masks and request size from the profile |
| `core/src/io/platform_io.cpp`, `file_reader.cpp` | `ExpertStream` I/O | keep direct I/O; add path-rate verification at open |
| `core/src/moe/dense_weights.cpp` | `ResidencyManager` (resident part) | keep the anonymous-copy versus mmap policy as a **measured** choice per model, not a default — it reversed sign between two models on the same phone |
| `core/src/moe/gpu_tier.cpp`, `host/gpu_ffn/` | `PlacementController` (helper role) | keep the code; **disabled by policy** on parts where measured dispatch overhead exceeds the threshold (§3.3) |
| `--repack-experts`, `--experts-prerepacked`, `--repack-dense`, `host/repack_bench/repack_gguf.cpp` | `LayoutService` | becomes an on-device transform with a `TransformRecord` and a fidelity gate |
| `--ppl`, `--ppl-ref`, `--ppl-dump`, `--ppl-kl-out` | `FidelityService` | becomes a library call; thresholds pre-registered per transform kind |
| `--fixed-routing` (replay) | calibration: compute-floor cell | keep; it is how true compute is measured without storage traffic |
| `--compute-trace`, `--io-trace`, `--route-trace` | calibration and diagnostics | keep; never in steady-state telemetry |
| `--ngram`, `--mtp`, `--draft` | `speculation` in `EngineConfig` | keep `ngram` as a candidate for copy-heavy agent outputs; `draft` is off by default (§7) |
| `--drop-cold-experts`, `--expert-substitute`, `--route-ahead`, `--n-expert-used` | lossy options | **excluded from automatic selection**; every one was measured non-negligible (§8.2) |
| `core/src/metrics/*` | `Telemetry` | keep the wall-additive contract; rename the residual (§10) |

New components with no existing counterpart: `PlacementController` as a *selector* (the code paths
exist, the decision logic does not), `ResidencyManager` as a *lease holder*, `KVManager` policy,
the per-device **autotune** procedure (§9), and the engine-side half of the `RuntimeGovernor` hooks.

---

## 3. Placement: which compute unit runs what

### 3.1 Regions

The forward pass is split into regions that can be placed independently: **attention**, **dense
feed-forward and shared experts**, **routed experts**, and — for prefill only — the **batched**
versions of each. Placement is chosen **separately for prefill and decode**, because the measured
winner differs: on the primary device the NPU prefilled a resident model at 284.1 tok/s
[E:olmoe_npu_prefill] against the GPU's 135.3 [E:olmoe_gpu_prefill] and the CPU's 20.4
[E:olmoe_cpu_prefill], while at decode the GPU led at 50.0 [E:olmoe_gpu_decode] and the NPU was
last at 40.7 [E:olmoe_npu_decode]. That is a latency-versus-throughput signature: a compute-strong
unit whose per-call overhead dominates at batch one.

### 3.2 The selection rule

For each (region, phase), candidates are the backends whose `fidelity_class` for this model is not
`unverified`. Each candidate is scored by the profile's measured rate for that region's shapes,
minus its dispatch overhead multiplied by the number of dispatches the region implies per token. The
highest score wins **only if** its predicted advantage exceeds the resolution of the measurement;
otherwise the CPU wins the tie, because it is the backend with the fewest failure modes.

Placement is then **confirmed** by a short on-device A/B in the autotune step (§9), because a
per-region microbenchmark is a proxy and proxies need a transfer test before they are trusted.

### 3.3 What was measured about accelerators for the streamed tier, so nobody re-derives it

| candidate role | measured outcome on the primary device | rule that encodes it |
|---|---|---|
| GPU as a **helper** for a few experts per layer | the fixed per-dispatch round trip alone exceeded the CPU's whole cost for that work; net neutral at best in the full engine | a helper role is admissible only if `dispatch_overhead_ms × dispatches_per_token` is below a fixed fraction of the predicted token time |
| GPU as the **main** engine, whole graph resident | 8 layers took 51.5 ms/token [E:e7_gpu_l8_ms] sustained against the CPU's 17.5 [E:e7_cpu_l8_ms]; the full model projects well above the CPU by monotonicity | GPU-main is a candidate only where the profile's measured decode rate on model-shaped kernels beats the CPU's |
| NPU for **expert matmuls** at batch one | slower than the CPU on the expert shapes; latency-bound | same rule as the helper role |
| NPU for **prefill** | fastest unit by a wide margin on a resident model | prefill placement is chosen independently, and this is why |

None of these is a permanent verdict about hardware. They are verdicts about *this part* under
*these dispatch costs*, and the rules above are written so that a vendor path with a lower dispatch
cost ([`09_VENDOR_INTEGRATION.md`](09_VENDOR_INTEGRATION.md) §5) is admitted automatically once it
measures better — without anyone editing a policy.

### 3.4 Threads and cores

Compute threads are pinned to the cluster the profile recommends; I/O lanes are pinned to a
different cluster. Thread count is a **first-order** parameter, not a detail: unpinned, four threads
on the primary device fell off a cliff to 4.0 tok/s [E:cliff_unpinned] where pinned threads on the
same cores reached 30.1 [E:cliff_pinned]. The mechanism was scheduling across clusters with different
clock domains, and it recurs on any big.LITTLE part. Adding threads beyond the fast cluster did not
help and contending with the system on every core hurt.

---

## 4. Expert streaming: the measured design

This is the part of the prior work that transfers unchanged. Each element was measured against its
alternatives and the alternatives lost.

### 4.1 Cache structure

- **Scope: per layer.** Each layer has its own quota out of the total byte budget. Layer *l*'s
  experts are never candidates for layer *l+1*, so one shared pool spends its capacity holding the
  whole layer cycle before anything recurs: at a 10% cache on real routing traces the shared pool
  scored 0.0 [E:sim_global_pool] where per-layer LRU scored 0.28 [E:sim_lru_10pct], against an
  offline optimum of 0.45 [E:sim_belady_10pct].
- **Fetch: event-atomic.** A layer's whole top-k set is decided for residency before anything is
  evicted, so a miss early in the fetch cannot evict an expert the same fetch still needs. This is
  what the hardware does anyway; simulating otherwise was a defect that produced a published zero.
- **Eviction: recency.** Plain LRU beat LFU, warm-start and static pinning at every cache size on a
  held-out split — including a static oracle that knew the whole trace's frequencies. Segmented LRU
  read 11% fewer bytes on the device, exactly as simulated, without a resolved decode gain; it is
  kept as a byte-saving option and not claimed as a speed one.
- **Warming is a time-to-first-token feature**, not a throughput one: LRU re-converges within a few
  hundred tokens regardless.

### 4.2 Reads

- Every read is at least `efficient_request_size` from the profile. Expert slices in every candidate
  model are already megabytes, so this is naturally satisfied — but a future model with tiny experts
  would need bundling, and the check is explicit.
- Direct I/O, on lanes pinned away from compute, count from the profile's storage plateau.
- **Traffic is not determined by hit rate alone.** On the primary device an 85.3% [E:qwen3_cache_hit]
  hit rate still moved 119.8 MiB [E:qwen3_read_mib] of experts per token, because what the cache
  misses is re-read in full. The planner predicts bytes, not hit rates
  ([`05_PERFORMANCE_MODEL.md`](05_PERFORMANCE_MODEL.md) §4.3).
- **Overlap**: lanes read the next needed experts while compute proceeds. The stall that overlap
  cannot hide has a structural floor, because a layer's routing is known only immediately before its
  experts are needed; on the primary device that un-hidden stall was 34 ms/token
  [E:qwen3_h2h_stall_ms] beside 24 ms [E:qwen3_h2h_mgmt_ms] of cache management, at an 87.5%
  [E:qwen3_h2h_hit] hit rate.

### 4.3 Prefetch and speculation of reads

- **Selective adoption** of speculative reads is on: a guess that turns out right is adopted rather
  than thrown away and re-read. Measured lossless (identical text) and decisive as part of the
  stacked configuration.
- **Cheap cross-token predictors are closed.** A persistence predictor is informationally empty (the
  engine already knows the current token's set); fitted Markov predictors were slightly worse than
  LRU because their errors are plausible experts that get wrongly protected. Only an *exact* lookahead
  of about four tokens reaches the offline optimum, and only a multi-token verification pass can
  supply one.
- **Within-token cross-layer prefetch** pays only when compute per token is large relative to the
  read; at the primary device's operating point it raised traffic through eviction churn. It stays
  off unless the profile's compute-to-read ratio crosses the measured threshold.

### 4.4 The ledger, for the planner

At steady state on the primary device, per token: compute at the sustained clock (the floor was
71.0 ms [E:qwen3_e1_repack_ms] with the fast kernels, 93.7 [E:qwen3_e1_generic_ms] without), then
stall, then management. **Compute is the wall once the cache works**, and the device's hard ceiling
from CPU weight throughput alone was 16.9 tok/s [E:qwen3_ceiling_cpu] — so the streamed tier on
this class of part is bounded by arithmetic, and the levers that remain are kernel efficiency and
clock, not I/O.

---

## 5. Residency and the memory lease

`ResidencyManager` owns every byte the engine holds and is the engine-side holder of the
`MemoryLease` ([`03_INTERFACES.md`](03_INTERFACES.md) §5).

### 5.1 What is resident

| class | policy |
|---|---|
| non-expert weights (attention, dense FFN, shared experts, embeddings, router) | resident for the session; **anonymous copy or mmap is a measured choice per model** — the copy was 13.6 ms/token cheaper on one model and 22.5 ms/token *more expensive* on another, on the same phone |
| expert cache | sized between `floor` and `target` from the lease; shrinkable |
| KV | sized by the context cap; quantisable; per-session |
| engine working set (compute buffers, batch buffers) | sized by `n_ubatch`; this competes with the cache and is part of the request |
| speculative / prefetch structures | first to go under pressure |

### 5.2 The shrink order (fixed)

1. drop speculation and prefetch;
2. shrink the expert cache toward `floor` (re-reads cost rate, not correctness);
3. quantise or trim KV of inactive sessions;
4. suspend inactive sessions with a KV checkpoint;
5. unload the streamed tier; rebind to resident;
6. refuse.

### 5.3 Swap is not memory

The engine must assume that anything the OS can reclaim, it will. Two measured facts drive the design:

- Pinning the expert cache into unswappable driver memory pushed reclaim onto the engine's **hot**
  anonymous data and raised compute cost by tens of milliseconds per token. Pin nothing cold.
- A never-released slot arena pushed hundreds of megabytes of the engine into compressed swap;
  every touch became a synchronous fault while the other compute threads waited at the next barrier.

So: grow the cache incrementally, verify residency (RSS, not allocation) after each step, and treat
a rise in major faults per token as a lease-pressure signal, not as noise.

### 5.4 Foreground coexistence

The agent's operating condition. No measurement exists (`PL-S2`). The engine's obligations are
mechanical: honour a shrink within a bounded time, checkpoint before suspend, report
`verified_resident` honestly, and never hold a wake lock for background work. What the achievable
tier *is* under a foreground app is experiment X1, and the engine must be instrumented so that
experiment can be run on day one.

---

## 6. Sessions and KV

### 6.1 The session

One process, one loaded model, one warm expert cache, serving many turns. Already implemented:
`Session::open` once, `generate` per turn, `clear_kv=false` to continue a conversation. The
platform adds:

- **lease hooks**: `shrink_to(bytes)`, `checkpoint()`, `restore()`;
- **admission metadata**: priority, deadline;
- **cancellation** mid-generation, which exists, and must remain cheap.

### 6.2 Prefix reuse, and the ordering it forces on the agent

A turn re-ingests only the tokens not already in the KV. On an agent workload the observation
(screen, state) changes every turn, so the only reusable part is a **stable prefix**: system
instructions, tool schemas, persistent task context. The agent must therefore build prompts as
`[stable prefix][task history][volatile observation]`, in that order, and never interleave. This is
specified as a hard requirement in [`07_AGENT_RUNTIME.md`](07_AGENT_RUNTIME.md) §5 because the
saving is the entire difference between a usable and an unusable turn time.

### 6.3 The context tax, and which mechanism matters

Context costs the streamed tier two ways: **eviction** (KV takes RAM the expert cache needed, so more
experts are read from storage) and **traffic** (attention re-reads the whole KV every token). The
prior phase's registered prediction found eviction dominant and traffic a minority at the context
lengths an agent uses, and *withdrew* an earlier claim that KV quantisation was decisive. So KV
quantisation is a modest lever on the streamed tier — and a **measured compute fact bounds the whole
thing**: at ~3,000 tokens the compute floor alone rose to 144.3 ms/token [E:qwen3_e1_long_ms]. The
`KVManager` therefore enforces the plan's context cap rather than hoping quantisation pays for
overrun.

---

## 7. Speculation

- **n-gram / prompt-lookup drafting** stays available: agent outputs echo element ids, field names
  and literal strings straight from the prompt, which is the regime it was built for. Its ceiling is
  bounded by the measured verify cost — verifying two positions cost 1.7x a single decode on the
  streamed tier — so it is a candidate, enabled by autotune only if it wins an A/B on the target
  workload.
- **Draft-model speculation** is off by default on the streamed tier: best case measured at ~1.0x,
  because a verify pass reads the union of the window's experts. On the resident tier it is a
  legitimate candidate and is treated like any other autotune option.

---

## 8. Fidelity: proving an optimisation did not change the model

### 8.1 The service

`FidelityService.compare(candidate, reference, corpus) -> {kl_mean, kl_p99, top1_flip_rate,
ppl_delta_frac, n_tokens}` under teacher forcing, so every configuration scores the **same** token
sequence. Comparing generated text does not work: greedy output can stay byte-identical under a
large perturbation and flip under a small one, depending only on whether an argmax boundary
happened to be crossed.

A **determinism control** runs first: the candidate against itself must give zero KL and zero flips.
Anything else means the measurement is broken and nothing downstream may be read.

### 8.2 Thresholds

Pre-registered per transform kind, relative to the **format floor** — the loss the user already
accepted by choosing a 4-bit file. On the primary device that floor was 0.118 nats [E:e6_floor_kl]
mean KL and 1.76 [E:e6_floor_p99] at the 99th percentile against an 8-bit reference. The prior
phase's "negligible" bar (mean and p99 KL within 10% of the floor, flips within one point, perplexity
within 1%, plus a task probe) is the template. Against it:

- **kernel repack / pre-layout**: expected bit-identical or within floating-point reordering; gate at
  the reordering tolerance, not the negligible bar;
- **requantisation, KV quantisation**: the negligible bar;
- **lossy routing (top-k reduction, drop-cold, substitution, route-ahead)**: measured across the
  whole frontier and **no option passed** — dropping one expert of eight cost 0.146 nats
  [E:e6_topk7_kl] against the floor's 0.118 [E:e6_floor_kl], and the closest arm still failed on
  its p99 at 2.06 [E:e6_best_arm_p99]. These stay excluded from automatic selection.

### 8.3 Where the reference comes from

For a transform, the reference is the untransformed file on the same device. For a lossy option, the
reference is a higher-precision file — which may not fit on the device, in which case the comparison
uses dumped reference logits produced elsewhere and shipped with the model card. The gate is only
as honest as its reference, so the reference's provenance is part of the `TransformRecord`.

---

## 9. Autotune: per-device configuration by measurement

The procedure that replaces every hard-coded constant of the prior phase with a measured choice. It
is run once per (device, model) at T2/T3, and re-run on invalidation.

### 9.1 Rules, inherited from the research method

1. **Floor first.** Before any A/B, write down the term it attacks and the lever's ceiling in
   milliseconds per token, both from the profile. A lever whose ceiling is below the device's
   run-to-run resolution is not A/B'd; it is bundled.
2. **Wall time is the only verdict.** Term deltas are diagnostics.
3. **Remove, don't move.** A lever must delete bytes, instructions or a serial dependency;
   relocation levers are rejected at design time, because five of them were measured net-neutral.
4. **Resolution check.** Expected effect must be at least twice the design's standard error, or the
   design changes (more rows, or the replay mode with lower noise), or the lever is bundled.
5. **ABBA, thermally gated, awake, in the deployment regime.** Rows that fail the validity guard are
   discarded, never averaged.
6. **Pre-registered decision rule** per lever, written before the first row.

### 9.2 The sequence

1. thread placement (masks from topology) — the largest lever measured;
2. I/O lane count and request size — read off the storage curve, confirmed by one A/B;
3. cache size — grow until the grant probe says stop, verifying residency;
4. weight layout — apply the transform, run the fidelity gate, A/B only if it passes;
5. dense-weight residency policy (copy versus mmap) — measured per model;
6. speculation mode — on the target workload, not on an essay prompt;
7. placement confirmation — the §3.2 choice against the CPU baseline.

Each step writes its verdict and its rows into the profile. The whole sequence is bounded in time
and heat by the calibration scheduler.

---

## 10. Telemetry semantics

Inherited contract, with one renaming and one rule.

- `wall_ms` per token is the ground truth. `io_ms`, `stall_ms`, `mgmt_ms` are measured. Under
  overlap, `stall_ms` is the **union** of intervals during which any compute thread was blocked on a
  streamed expert — the critical-path quantity — not a per-thread mean.
- The remaining term is `compute_ms_residual`, named as such. It absorbs page faults, scheduler
  stalls and clock caps. It is shown to developers and **never fitted**
  ([`05_PERFORMANCE_MODEL.md`](05_PERFORMANCE_MODEL.md) §3.4).
- `read_bytes` is taken from the process's block-layer accounting **and** the engine's own log, and
  the two are compared; a derived number is measured two ways.
- Major faults per token are always reported; a rise is a lease-pressure signal.

---

## 11. The engine-process protocol

The engine runs as its own process ([`02_ARCHITECTURE.md`](02_ARCHITECTURE.md) §8) and speaks a
line-delimited request/response protocol, which is the existing `--session` design formalised.

Requests: `open(EngineConfig, model)`, `generate(prompt, n_predict, clear_kv, think, grammar?,
deadline?)`, `cancel`, `shrink_to(bytes)`, `checkpoint`, `restore`, `fidelity(reference, corpus)`,
`probe(kind, params)`, `close`.

Responses: a `ready` line with load time, architecture, context size and the think-control mode the
model's own template supports; per-token progress lines; a `done` line with the turn's summary; typed
errors from the taxonomy of [`03_INTERFACES.md`](03_INTERFACES.md) §8. A bad request leaves the
session usable; a decode failure does not.

**Grammar-constrained decoding** is exposed on `generate`, because the agent's outputs are structured
tool calls and constraining the sampler is both cheaper and more reliable than parsing free text.

---

## 12. Model acquisition and storage

- **Path first.** Before a byte is downloaded, the storage probe runs on the destination directory
  and the planner confirms that a feasible configuration exists on that path. A model that would
  need streaming is not downloaded to a path that cannot stream it.
- **Resumable, hashed, quarantined.** A file is not a model until its hash matches the card.
- **Split files are fine**: the engine reads sharded files in place with no merge step.
- **Transforms are stored beside the original**, never in place of it, each with its record; the
  original is what every gate compares against.
- **Retention**: under storage pressure, transforms are dropped before originals, and originals are
  dropped by least-recent-binding with the user's confirmation.

---

## 13. Invariants and the tests that pin them

Each is a property whose expected value is fixed independently of whether the engine is fast
([`../../CLAUDE.md`](../../CLAUDE.md) §9.1).

| invariant | test |
|---|---|
| a resident-tier run performs zero storage reads at steady state | block-layer read counter is flat over tokens 65+ |
| event-atomic fetch never scores below sequential fetch on any trace | property test on the simulator, already exists |
| per-layer quotas preserve the total byte budget exactly | unit test on `split_capacity`, already exists |
| any transform's output passes the determinism control against itself | fidelity self-test before every gate |
| the sum of measured terms plus the residual equals wall time per token | telemetry identity test, with the clamp case documented |
| `verified_resident` never exceeds the process's measured RSS | lease unit test |
| shrinking the cache never changes generated text at temperature zero | lossless-by-construction test, identical-text comparison |
| every automatically adopted configuration has a `TransformRecord` or an autotune verdict | registry audit |
| the engine refuses a checkpoint whose architecture is not in the registry | negative test |
| a `Refusal` from the lease is surfaced as a typed outcome, never as a crash | fault-injection test |
