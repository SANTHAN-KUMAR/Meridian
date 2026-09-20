# 3 — Interfaces and contracts

The schemas and APIs that hold the system together. An implementer building any single layer should
be able to work from this document plus [`02_ARCHITECTURE.md`](02_ARCHITECTURE.md) alone.

Types are written in a language-neutral form. `?` marks optional. Durations are milliseconds,
sizes are bytes unless a field name says otherwise, rates are per second. Every schema is versioned
and every consumer must reject a version it does not understand rather than guess.

---

## 0. Two rules that apply to every schema here

**Rule 1 — provenance is part of the value.** Any field that carries a performance-relevant number
is a `Measured<T>`, not a `T`:

```
Measured<T> = {
  value:        T,
  provenance:   "measured" | "calibrated" | "prior" | "declared" | "unknown",
  confidence:   float,           // 0..1, the model's own; NOT a p-value
  interval?:    [T, T],          // required when provenance != "measured"
  observed_at?: timestamp,
  conditions?:  ValidityConditions,   // §2.3 — what was true when it was measured
  source:       string           // probe id, artifact path, or the prior's identifier
}
```

- `measured` — observed on **this device**, under stated conditions, by a named probe.
- `calibrated` — an analytic model fitted to measurements on this device, evaluated inside its
  fitted range.
- `prior` — a device-class prior or an analytic model evaluated **outside** its fitted range. Any
  consumer that treats a `prior` as a promise is defective.
- `declared` — from a file's own metadata (a model config), true by definition, not a measurement.
- `unknown` — the probe could not run. **Never substitute a default and call it a value.**

**Rule 2 — refusal is a value, not an exception.** Every producer of a prediction or a plan may
return `Refusal`, and every consumer must handle it:

```
Refusal = {
  reason:   RefusalCode,       // §8
  detail:   string,            // human-readable, specific
  missing?: [string],          // what would have to be measured to answer
  nearest?: object             // the closest feasible alternative, if one exists
}
```

---

## 1. `DeviceProfile`

Produced by layer 0. The single description of what this phone is and does. Detail on how each
field is obtained: [`04_DEVICE_PROFILING.md`](04_DEVICE_PROFILING.md).

```
DeviceProfile {
  schema_version:  int,
  profile_id:      string,          // stable per device+OS build+probe-suite version
  probe_suite:     string,          // version of the code that produced this
  created_at:      timestamp,
  state:           "fresh" | "stale" | "suspect" | "partial",

  identity {
    manufacturer, model, soc_vendor, soc_model, board: string,
    os_version, api_level:  string,
    build_fingerprint:      string,
    rooted:                 bool,       // verified by executing, not by finding a binary
    verified_boot:          string
  },

  cpu {
    clusters: [ {
        name:          string,          // e.g. "prime", "performance", "efficiency"
        core_ids:      [int],
        max_khz:       int,             // hardware maximum, NOT what it sustains
        isa_features:  [string],        // "i8mm", "dotprod", "fp16", "sve2", ...
        matmul_gbps:   Measured<float>  // on MODEL-SHAPED kernels, per cluster
    } ],
    recommended_compute_mask: Measured<bitmask>,   // from topology, confirmed by an on-device A/B
    recommended_io_mask:      Measured<bitmask>
  },

  memory {
    total:                 int,
    available_at_probe:    Measured<int>,
    grantable_quiesced:    Measured<int>,   // GrantProbe, nothing else running
    grantable_foreground:  Measured<int>,   // GrantProbe with a target app foreground — RQ2
    zram_or_swap:          { present: bool, size?: int, fault_cost: Measured<float> },
    dram_read_gbps:        Measured<float>,
    lmk_behaviour?:        { observed_kills: int, at_rss: int }
  },

  storage {
    path:              string,          // where models live
    filesystem:        string,          // fuseblk, f2fs, ext4, ... — this decides feasibility
    free_bytes:        int,
    random_read: [ { size_bytes: int, threads: int, mbps: Measured<float>, p50_us, p99_us: float } ],
    efficient_request_size: Measured<int>,   // the knee: smallest size reaching bulk rate
    sequential_write_mbps:  Measured<float>,
    direct_io_supported:    bool
  },

  accelerators: [ {
      kind:        "cpu" | "gpu" | "npu" | "vendor",
      backend_id:  string,             // "opencl-adreno", "qnn-htp-v81", "cpu-i8mm", ...
      available:   bool,
      reachable_unprivileged: bool,    // reachable by an ordinary app: the only case that matters
      decode_gbps:  Measured<float>,   // weight-byte throughput at batch 1
      prefill_gbps: Measured<float>,   // batched — measured separately, because they differ
      dispatch_overhead_ms: Measured<float>,   // the fixed per-call cost; it decides helper roles
      fidelity_class: "bit-exact" | "tolerant" | "unverified"
  } ],

  thermal {
    sustained_derate:  Measured<[ {skin_c: float, clock_frac: float, throughput_frac: float} ]>,
    time_to_throttle_s: Measured<float>,
    recovery_s:         Measured<float>,
    zones:              [ {name: string, usable: bool} ]
  },

  power {
    rails_available:  bool,
    method:           "per-rail" | "battery-counter" | "none",
    idle_w:           Measured<float>?
  },

  validity {
    conditions:        ValidityConditions,
    rejected_runs:     int,
    rejection_reasons: [string]
  }
}
```

**Field rules.** `grantable_foreground` is the field the planner must use for anything the agent
does; `grantable_quiesced` exists only to quantify the gap. `efficient_request_size` is the knee of
the storage curve, and every streaming read must be at least that size — the measured penalty for
ignoring it on the primary device was 8.9x [E:dev_flash_ratio]. A profile whose `filesystem` is a
userspace bridge must carry a measured rate for **that** path, never one inherited from the block
device.

### 1.1 `ValidityConditions`

Attached to every measurement, so a consumer can see the regime it came from.

```
ValidityConditions {
  wakefulness:   "awake" | "dozing" | "unknown",
  foreground:    "self" | "other-app" | "none" | "unknown",
  power:         "unplugged" | "ac" | "usb",
  thermal_status: int,               // OS thermal level at start and end
  skin_c_start, skin_c_end: float,
  battery_pct:   int,
  descheduled:   bool,               // the process did not run when it thought it did
  concurrent_load: [string]          // other known heavy processes
}
```

A measurement whose conditions differ from the deployment regime (`awake`, `unplugged`,
thermally settled, app foreground) is usable as a diagnostic and **not** as a planning input.

---

## 2. `ModelCard`

Produced by `ModelRegistry`. Declared facts come from the model file; derived facts are computed
from it, never typed.

```
ModelCard {
  schema_version: int,
  model_id:       string,
  display_name:   string,
  source:         { url: string, sha256: string, licence: string, revision?: string },

  declared {                         // read from the checkpoint's own metadata
    architecture:      string,       // the registry key, e.g. "qwen3moe"
    is_moe:            bool,
    n_layer:           int,
    n_expert?:         int,
    n_expert_used?:    int,
    n_shared_expert?:  int,
    hidden_size, intermediate_size: int,
    context_limit:     int,
    quantisation:      string,       // "Q4_0", "MXFP4", "Q4_K_M", ...
    tokenizer:         string,
    files:             [ {path: string, bytes: int} ]
  },

  derived {                          // computed from the file; the arithmetic is the definition
    total_bytes:            int,
    resident_bytes:         int,     // everything that is not a routed expert
    expert_bytes:           int,
    active_bytes_per_token: int,     // what one token's forward pass must touch
    expert_slice_bytes:     int,     // one expert's contiguous read — compare to efficient_request_size
    kv_bytes_per_token:     { f16: int, q8: int, q4: int }
  },

  capabilities {                     // declared by the catalogue, verified by evaluation
    tool_calling:   "native" | "prompted" | "none",
    structured_output: bool,
    vision:         bool,
    languages:      [string],
    context_effective?: Measured<int>,   // where quality degrades, if ever measured
    task_scores?:   [ {suite: string, score: float, measured_on: string} ]
  },

  transforms: [ TransformRecord ]    // §6
}
```

**Adding a model family** must remain a registry row plus a byte-identity check, not a code change
in the streaming path. That property is inherited from the engine and is what makes "run any model"
a real feature. A model whose architecture is not in the registry is **refused**, not guessed at.

---

## 3. `PerfPrediction`

Produced by `PerformanceModel`. The contract of decision **D2**.

```
PerfPrediction {
  model_id, device_profile_id: string,
  config:        EngineConfig,       // §4.1
  context_len:   int,                // predictions are context-conditional, always

  prefill_tok_s: Measured<float>,
  decode_tok_s:  Measured<float>,
  ttft_ms:       Measured<float>,
  joules_per_token: Measured<float>?,

  memory_required: { resident: int, cache_min: int, cache_target: int, kv: int },
  storage_read_bytes_per_token: Measured<int>?,

  calibration_state: "measured" | "interpolated" | "extrapolated" | "uncalibrated",
  basis:        [string],            // which profile fields and which calibration cells were used
  assumptions:  [string],            // in words, e.g. "expert-cache hit rate from a 200-token warm sample"
  falsification: {                   // what the governor will check, decided HERE not at runtime
     metric: "decode_tok_s" | "prefill_tok_s" | "ttft_ms",
     tolerance_frac: float,
     min_observations: int
  }
}
```

**Hard rules.**
- If `calibration_state == "extrapolated"`, the interval must widen accordingly and the UI must not
  present the point value alone.
- `uncalibrated` is only ever returned together with a `Refusal` to plan on it, or with an explicit
  request to run calibration first.
- Predictions for **prefill and decode are separate quantities with separate bases**. On the primary
  device the ranking of compute units inverts between them: 284.1 [E:olmoe_npu_prefill] against
  20.4 [E:olmoe_cpu_prefill] tok/s at prefill, but 40.7 [E:olmoe_npu_decode] against
  44.5 [E:olmoe_cpu_decode] at decode. Any interface that collapses them to one "speed" is wrong.

---

## 4. `ExecutionPlan`

Produced by `ConfigurationPlanner`. What the engine is actually told to do.

```
ExecutionPlan {
  plan_id:       string,
  model_id:      string,
  config:        EngineConfig,
  prediction:    PerfPrediction,
  lease:         MemoryLeaseRequest,
  ladder:        [LadderRung],        // ordered degradations, §4.2
  valid_while:   {                    // re-plan immediately if any of these changes
     thermal_status_max: int,
     foreground_app?:    string,
     power_state?:       string,
     profile_id:         string
  },
  expires_at?:   timestamp
}
```

### 4.1 `EngineConfig`

The complete, reproducible description of how a model is run. Everything the prior project passed as
a command-line flag belongs here, chosen by the planner rather than by a human.

```
EngineConfig {
  tier:            "resident" | "streamed",
  quantisation:    string,
  placement: {
     prefill: { attention: BackendId, dense: BackendId, experts: BackendId },
     decode:  { attention: BackendId, dense: BackendId, experts: BackendId }
  },
  threads:         { count: int, cpu_mask: bitmask },
  io:              { lanes: int, cpu_mask: bitmask, request_bytes: int, direct: bool },
  expert_cache:    { floor_bytes: int, target_bytes: int, policy: "lru" | "slru",
                     scope: "per_layer", fetch: "event_atomic" },
  kv:              { quantisation: string, max_context: int, prefix_reuse: bool },
  speculation:     { mode: "off" | "ngram" | "draft", params?: object },
  weight_layout:   string,            // which TransformRecord's output to use, or "original"
  overlap:         bool
}
```

`expert_cache.scope` and `fetch` are fixed rather than tunable: per-layer scope and event-atomic
fetch were measured against the alternatives, and the shared-pool alternative scored 0.0
[E:sim_global_pool] against 0.28 [E:sim_lru_10pct] at the same total bytes. They are in the schema
for the reader's benefit and as a guard against a future change that silently reverts them.

### 4.2 `LadderRung`

```
LadderRung {
  trigger:   "memory" | "thermal" | "latency" | "battery",
  action:    "drop_speculation" | "shrink_cache" | "quantise_kv" | "reduce_context"
           | "change_placement" | "change_tier" | "suspend" | "refuse",
  params:    object,
  expected_cost: { rate_frac: float, quality: "none" | "bounded" | "unbounded" },
  reversible: bool,
  hysteresis_ms: int
}
```

Rungs are ordered by **quality preserved per unit of pressure relieved**. A rung whose `quality` is
`unbounded` may never be entered automatically — it needs either a fidelity gate that bounds it or a
user decision.

---

## 5. `MemoryLease` — decision D5 as a protocol

```
MemoryLeaseRequest  { floor: int, target: int, purpose: string, priority: "interactive" | "background" }
MemoryLease         { lease_id: string, granted: int, verified_resident: int,
                      heartbeat_ms: int, revocable: bool }
```

Protocol:

1. **Request** `{floor, target}`. A request whose `floor` exceeds `grantable_foreground` is refused
   at plan time, not attempted.
2. **Grant** returns `granted`, which may be below `target`. **The planner must re-check the plan
   against `granted`, never against `target`** — on the primary device, requested and granted
   diverged routinely and the largest budget ever granted across 238 runs was 5806 MiB
   [E:dev_budget_max_mib].
3. **Verify** `verified_resident` by measuring actual resident set, not by trusting the allocator.
   Pages compressed into swap are not memory for this purpose.
4. **Heartbeat.** The holder confirms it still needs the lease; a missed heartbeat releases it.
5. **Trim.** On pressure the lease shrinks along the ladder; the holder must honour a shrink within
   a bounded time or be revoked.
6. **Revoke.** The holder checkpoints and releases. Revocation is normal, not exceptional.

---

## 6. `TransformRecord` — automatic optimisation with a proof

Every transform the platform performs on a model file is recorded, and no transform is used before
it passes its gate (decision **D8**).

```
TransformRecord {
  transform_id:  string,
  kind:          "kernel_repack" | "expert_prelayout" | "requantise" | "kv_policy" | "vendor_compile",
  produced_at:   timestamp,
  device_profile_id: string,          // a transform is device-specific and must not be shared blindly
  output_path:   string,
  gate: {
     status:      "passed" | "failed" | "pending",
     reference:   string,             // what it was compared against
     metrics:     { kl_mean: float, kl_p99: float, top1_flip_rate: float, ppl_delta_frac: float },
     thresholds:  { ... },            // fixed BEFORE the transform was run
     n_tokens:    int
  },
  adopted:       bool
}
```

The thresholds are pre-registered per transform kind. A transform with `status: "failed"` has
`adopted: false` and the original weights are used — the precedent being a repack path that won its
microbenchmark and then missed its perplexity gate end to end
([`00_PROBLEM.md`](00_PROBLEM.md) §8.8).

---

## 7. The agent-facing contract

The interface across which decision **D7** is enforced. The agent says what it needs; it never names
a model, a quantisation or a backend.

```
CapabilityRequest {
  task_class:     "classify" | "extract" | "summarise" | "plan" | "tool_call" | "converse" | "code",
  quality_floor:  "draft" | "standard" | "high",
  latency_target_ms: int,            // to first action, not to last token
  context_tokens_estimate: int,
  output_tokens_estimate:  int,
  needs:          [ "structured_output" | "tool_calling" | "vision" | "long_context" ],
  privacy:        "on_device_only" | "may_use_network",
  budget?:        { max_joules?: float, max_wall_ms?: int }
}

ModelBinding {
  binding_id:  string,
  model_id:    string,
  plan:        ExecutionPlan,
  expectation: { ttft_ms: Measured<float>, tokens_per_s: Measured<float> },
  escalation?: {                      // what a step up would cost, so the caller can decide
     available:     bool,
     model_id?:     string,
     added_latency_ms: Measured<float>,
     added_joules?:    Measured<float>
  }
}
```

`bind(CapabilityRequest) -> ModelBinding | Refusal`. A refusal carries `nearest`: the closest
request that *would* succeed (for example, the same task with a 2x latency target).

### 7.1 Tool contract

```
ToolSpec {
  name:        string,
  schema:      JSONSchema,           // parameters; this is what goes in the model's prompt
  effects:     "read" | "write" | "communicate" | "transact",
  reversible:  bool,
  consent:     "none" | "once" | "every_time",
  cost_hint:   { wall_ms: int, needs_network: bool, needs_foreground: bool },
  postcondition: string,             // a checkable predicate, named; the Verifier calls it
  checkpointable: bool               // required true for anything runnable in the background
}
```

`effects` and `reversible` drive `SafetyGate`; `postcondition` drives verification; `cost_hint`
enters the agent's own planning so it does not choose a 30-second tool for a 3-second task.

---

## 8. Error and refusal taxonomy

One closed set, used at every layer. Typed refusals are how the system stays honest under pressure.

| code | meaning | typical response |
|---|---|---|
| `NotCalibrated` | no measurement covers this configuration | offer calibration; give a prior with a wide interval, clearly labelled |
| `OutOfRange` | the request is outside what any measurement supports | refuse; name what would have to be measured |
| `Infeasible` | no configuration satisfies the constraints on this device | return `nearest` |
| `MemoryUnavailable` | the lease cannot be granted or was revoked | ladder, then refuse |
| `ThermalLimited` | the device cannot sustain the required rate now | defer with an estimated ready time |
| `StorageTooSlow` | the model's path cannot serve the plan's reads | refuse streaming; suggest moving the model |
| `FidelityGateFailed` | a transform or backend changed the model's outputs beyond threshold | do not adopt; fall back; record |
| `ArchitectureUnsupported` | the checkpoint's family is not in the registry | refuse; name the registry row that would be needed |
| `IntegrityFailed` | hash or structure check failed | quarantine |
| `Preempted` | a higher-priority request took the device | requeue background work; report for interactive |
| `ConsentRequired` | the action needs user authorisation | ask, with the specific action named |
| `NoProgress` | the agent loop is not advancing | stop, report state, hand back to the user |

---

## 9. Telemetry events

The governor's input, the audit trail, and the evaluation dataset — one schema for all three.

```
TurnEvent {
  turn_id, task_id, binding_id: string,
  t_start, t_end: timestamp,
  prompt_tokens, new_prompt_tokens, output_tokens: int,   // new_* is what prefix reuse did NOT save
  ttft_ms, prefill_ms, decode_ms, tool_ms: float,
  predicted: { ttft_ms: float, tokens_per_s: float },
  observed:  { ttft_ms: float, tokens_per_s: float },
  conditions: ValidityConditions,
  lease:      { granted: int, verified_resident: int, shrinks: int, revoked: bool },
  ladder_rung: int,
  action:     { tool: string, accepted: bool, verified: bool, consent: string },
  outcome:    "ok" | RefusalCode
}
```

```
TokenEvent {                          // sampled, not every token, unless tracing
  turn_id: string, index: int,
  wall_ms: float,
  io_ms: float, stall_ms: float, mgmt_ms: float,
  compute_ms_residual: float,         // NAMED as a residual — see below
  read_bytes: int, cache_hit_frac: float, major_faults: int
}
```

> **`compute_ms_residual` is named that way on purpose.** In the existing engine this term is
> `wall − stall − mgmt` and it silently absorbs page faults, scheduler stalls and frequency caps
> ([`00_PROBLEM.md`](00_PROBLEM.md) §8.1). It is a diagnostic for humans. **It must never be a
> regression input to the performance model**, which fits on wall-clock quantities only. A separate
> tracing mode measures true compute when that is the question, at a cost that makes it a
> diagnostic rather than telemetry.

Telemetry leaves the device only with explicit opt-in, and never carries prompt or screen content —
only the structural fields above.

---

## 10. Versioning and compatibility

- Every schema carries `schema_version`; consumers reject unknown majors.
- `DeviceProfile` is invalidated by: an OS build change, a probe-suite version change, a model
  storage path change, or repeated prediction drift.
- A `TransformRecord` is bound to a `device_profile_id` and must not be reused across devices
  without re-gating.
- Prediction artifacts are immutable and timestamped; a re-prediction creates a new record rather
  than editing one, so that a falsification can always be traced to the prediction it falsified.
