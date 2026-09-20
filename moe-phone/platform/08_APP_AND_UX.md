# 8 — The mobile app: stack, architecture, screens and performance budget

Layer E of [`02_ARCHITECTURE.md`](02_ARCHITECTURE.md), plus the Android-side plumbing that hosts
layers A–D. This document is prescriptive: it names the stack, the process model, the screens, the
interaction rules and the frame budget, because "the app is slow" is the fastest way to lose a
phone-first product no matter how good the engine is.

Two constraints shape everything here:

1. **The app shares a phone with the inference engine.** Every megabyte the UI holds is a megabyte
   the expert cache does not get, and every frame the UI spends is CPU the decode loop wanted. The
   UI is a **guest** in its own app.
2. **The product's honesty is a UI problem.** The platform refuses, gives intervals and says "I do
   not know yet" ([`05_PERFORMANCE_MODEL.md`](05_PERFORMANCE_MODEL.md) §5). If the UI renders those
   as a single confident number, the entire discipline underneath is wasted.

---

## 1. Stack

| concern | choice | why this and not the alternative |
|---|---|---|
| language | **Kotlin** | first-class Android, coroutines and structured concurrency match the streaming workload |
| UI toolkit | **Jetpack Compose** with Material 3 | declarative UI over a token stream is far simpler than imperative view updates; the existing reference app for this engine is already Compose |
| architecture | **unidirectional data flow**: immutable UI state, events up, state down, one `ViewModel` per screen | a token stream mutating shared state from a background thread is the classic source of jank and races |
| async | **coroutines + `Flow`**, `StateFlow` for state, `SharedFlow` for events, `conflate()` on high-rate streams | back-pressure on a token stream must be explicit, not accidental |
| engine binding | **a separate OS process**, driven over a line-delimited protocol ([`06_EXECUTION_ENGINE.md`](06_EXECUTION_ENGINE.md) §11) | a native OOM or crash must not take the UI with it, and the OS accounts its memory separately |
| host for the session | **foreground service** with a media-style persistent notification | the only way to keep an inference session alive and not be killed; it is also where the wake lock and the memory lease live |
| background work | **`WorkManager`** with constraints (charging, idle, unmetered) | model downloads and T3 calibration must survive process death and respect the user's battery |
| persistence | **Room** for tasks, recordings, model registry, profiles; **DataStore** for preferences | structured queries over the audit log are needed by the evaluation harness |
| DI | **Hilt** | standard, and keeps the layer boundaries of §2 enforceable |
| serialization | **kotlinx.serialization** | schema versioning of the contracts in [`03_INTERFACES.md`](03_INTERFACES.md) |
| native | **C++ over the NDK**, engine built per ABI with runtime feature detection | the engine is C++; the ISA variants are selected at runtime, not at install |
| navigation | **Navigation Compose**, few destinations, no deep nesting | |
| testing | JUnit + Turbine for flows, Compose UI tests, Macrobenchmark for startup and jank, Baseline Profiles | jank and startup are measured, not eyeballed |

**Deliberately excluded:** cross-platform UI frameworks (a second runtime competing for the memory
the model needs), heavyweight image loading in list rows, reflection-based JSON, and any dependency
that pulls a large native library the engine does not already need.

---

## 2. App architecture

```
app/
  ui/            Compose screens + ViewModels, one package per screen
  domain/        use cases; pure Kotlin, no Android types — the layer tests run on the JVM
  data/          repositories: ModelRegistry, ProfileStore, TaskStore, Recorder
  service/       InferenceService (foreground), EngineClient (process + protocol), WorkManager workers
  platform/      HAL bindings: thermal, power, memory-pressure, perf hints, accessibility
  native/        JNI surface and the engine build
```

Rules:

- `ui` never touches `service` or `native` directly; it goes through `domain`.
- `domain` is pure Kotlin so that planner and agent logic is unit-testable without a device.
- The engine process is reached only through `EngineClient`, which owns the protocol, the timeouts
  and the typed errors of [`03_INTERFACES.md`](03_INTERFACES.md) §8.
- **All contract types are shared** with the harness and the evaluation tooling, so the app and the
  research pipeline cannot drift apart.

### 2.1 Process and lifecycle

| process | holds | dies when |
|---|---|---|
| UI process | Compose, ViewModels, Room | user leaves and the system reclaims; must restore from persisted state |
| `InferenceService` (same process as UI or its own, configurable) | the lease, the wake lock, the `EngineClient`, the agent loop | user stops the session; system pressure with no active task |
| engine process | model weights, expert cache, KV | session close, lease revocation, or crash (restarted with a backoff, state restored from checkpoint) |

A crash of the engine process surfaces as a typed `EngineUnavailable` state in the UI with a retry —
never as a blank screen.

---

## 3. Screens

### 3.1 Task console — the home screen

The primary surface. A single input (text or voice) and a running transcript of the current task.

- **Input first.** The keyboard target and the microphone are reachable with one thumb.
- **The transcript shows steps, not tokens.** Each step is a row: what the agent is doing, the tool
  it is using, and its verification result. Raw model output is available behind a disclosure, not
  in the main flow.
- **The action confirmation is a sheet**, never a dialog buried in a list: the exact content,
  recipient and effect, with a primary confirm and an obvious cancel.
- **Undo** is offered inline for reversible writes, for as long as the tool supports it.
- **Live cost**: a compact, non-animated indicator of tokens per second and the current tier. It is
  informative, not decorative, and it is the thing a hackathon judge will look at.

### 3.2 Capability report — "what can this phone do"

Shown after first-launch profiling and reachable any time. This screen is the product's thesis
made visible.

- Plain-language summary: what this phone can run well, what it can run slowly, what it cannot run.
- Per candidate model: predicted prefill and decode rates **as a range**, the memory it needs,
  whether it must stream, and a **confidence chip** — `measured`, `calibrated`, `estimated` — with a
  one-line explanation on tap.
- An explicit "measure this properly" action that runs the relevant calibration and shows the
  interval narrowing afterwards. **A number that moves from `estimated` to `measured` in front of
  the user is the single most persuasive thing this product does.**
- The device facts that drove the verdict: cores, memory the OS will actually grant, storage read
  rate, accelerators available.

### 3.3 Model shelf

- Installed models with size, tier, last used, and the transforms applied with their gate results.
- A catalogue plus "add by URL". Before any download: the predicted performance, the space needed,
  and a refusal with a reason if the device cannot run it.
- Download progress with pause and resume; downloads obey `WorkManager` constraints.
- A model whose transform failed its fidelity gate says so plainly and runs untransformed.

### 3.4 Routines

Saved multi-step tasks with triggers (time, place class, device state). Each shows its last run and
its success rate. Editing a routine is editing a list of steps, not writing a prompt.

### 3.5 Activity and audit

Every task with its steps, actions, consents and outcomes; filterable; exportable. This is the audit
trail from [`07_AGENT_RUNTIME.md`](07_AGENT_RUNTIME.md) §11 rendered for a human, and it is what
makes an agent with device permissions trustworthy.

### 3.6 Settings

Consent policies per app and per contact, privacy mode (`on_device_only` as the default), battery
and thermal policy, calibration schedule, telemetry opt-in (off by default), developer mode.

### 3.7 Developer mode

Behind a toggle: the term ledger per token, the live plan and its ladder rung, lease state, thermal
state, prediction versus observation, and a one-tap export of the session's telemetry. This is a
demo asset as much as a debugging one.

---

## 4. Rendering a token stream without jank

The specific problem: tokens arrive at up to tens per second, each one mutating a string that a
Compose text node renders. Done naively this recomposes a large subtree per token and drops frames
while the engine is trying to use the CPU.

Rules:

1. **Conflate the stream.** The engine emits per token; the UI collects at a fixed cadence
   (about 60 ms) and renders the accumulated text. Users cannot read faster than that, and it cuts
   recompositions by an order of magnitude.
2. **Isolate the streaming node.** The growing text lives in its own composable reading its own
   `State`, so recomposition never escapes it. Everything around it is stable.
3. **Immutable snapshots.** UI state is a data class; no mutable shared buffers across threads.
4. **No markdown re-parse per token.** Parse incrementally, or render plain text while streaming and
   format once on completion.
5. **Stable keys** in every list; `LazyColumn` with explicit `key`, no index keys.
6. **No animations during generation** beyond a single lightweight indicator. Animation competes
   directly with decode for the same cores.
7. **`derivedStateOf`** for anything computed from the stream (token counts, rates) so it does not
   recompose per character.
8. **Autoscroll** follows only when the user is already at the bottom.

## 5. The UI's performance budget

Treated as a contract, measured with Macrobenchmark, and regressions fail the build.

| quantity | budget | why |
|---|---|---|
| cold start to interactive | **< 1.5 s** | Baseline Profiles; the engine loads lazily, never on the startup path |
| first meaningful capability report | **< 10 s** from first launch | T0+T1 profiling only ([`04_DEVICE_PROFILING.md`](04_DEVICE_PROFILING.md) §2) |
| frame time while streaming | **p99 < 16 ms**, zero frozen frames | the UI must not steal the decode loop's cores |
| UI process resident memory | **< 150 MB** steady | every megabyte here is taken from the expert cache |
| recompositions per generated token | **≤ 1** in the streaming subtree, **0** elsewhere | measured with the Compose recomposition counters |
| main-thread work per token batch | **< 4 ms** | serialisation and parsing happen off the main thread |
| time from action confirmed to executed | **< 150 ms** | consent must not feel like a tax |

**The UI thread never:** parses a model file, serialises an accessibility tree, computes a
prediction, or touches the engine protocol.

---

## 6. Communicating uncertainty without being annoying

The hardest UX problem in this product. Rules that keep it honest and usable:

- **A range, not a number, wherever the basis is not `measured`**: "about 12–18 tokens/s". A point
  value appears only for a measured cell.
- **One chip, three states**: `measured` (this phone, this model), `calibrated` (predicted from this
  phone's measurements), `estimated` (predicted from similar phones). Tapping explains it in one
  sentence.
- **Refusals are offers.** `NotCalibrated` renders as "I can measure this in about 40 seconds" with a
  button, not as an error. `Infeasible` renders as the nearest thing that would work.
- **Never show a prediction the system would not act on.** If the planner called two configurations
  tied, the UI says they are equivalent on this phone, and why that is interesting.
- **Degradation is visible, not silent.** When the thermal ladder steps down, a quiet line says the
  phone is warm and what changed. A user who understands why it slowed down forgives it; one who
  does not, uninstalls.

---

## 7. Accessibility, input and localisation

- The app **uses** the accessibility service to act, so it must be exemplary in its own
  accessibility: full TalkBack labelling, focus order, 48 dp touch targets, no colour-only state,
  dynamic type to the largest size without clipping.
- **Voice input** is first-class: phone tasks are often stated while walking. Transcription runs
  on-device where available.
- Dark theme and dynamic colour from the start; both tested, since demos happen in dark rooms.
- Strings externalised and the layout tested at long-string widths; an Indian-market app should
  expect multilingual input in the task console even before the UI itself is translated.

---

## 8. Permissions and onboarding

Permissions are requested **at the moment of first use**, with the task in view, never in a
front-loaded wall at install:

| permission | asked when | fallback if denied |
|---|---|---|
| notification listener | the first triage task | triage is disabled; other classes work |
| accessibility service | the first UI-action task | intent-and-deep-link tools only |
| contacts / calendar / SMS | the first task that needs each | that tool is unavailable; the agent says so and offers an alternative |
| camera | the first capture task | gallery import |
| ignore battery optimisation | before the first long background routine | routines run only while the app is open |

Onboarding is three screens: what it does, that it runs on the phone with nothing leaving it, and
the profiling run that produces the capability report. The capability report **is** the onboarding
payoff.

---

## 9. Demo and judging considerations

Stated because the immediate venue is a phone-first hackathon judged on phone-first execution, AI
integration and defended trade-offs — and because these are also the right engineering choices:

- **Everything works in airplane mode.** That is the demo: turn the radios off and keep going.
- The **capability report on the judge's own phone** is the strongest single moment available. It is
  the platform's whole thesis in one screen: an unknown device, measured in seconds, with honest
  numbers, including the ones it refuses to give.
- **Developer mode** carries the defence of the trade-offs: the term ledger, the plan, the ladder,
  the prediction against the observation.
- Keep a **pre-warmed session** for the demo path; first-load time for a large model is real and
  should be shown once deliberately rather than hit by accident.
- Do not demo the streamed tier for a task the resident tier does better. The escalation story is
  stronger when the escalation is *justified* by a task the small model visibly fails.

---

## 10. UI invariants and tests

| invariant | test |
|---|---|
| no engine call on the main thread | StrictMode in debug; lint rule |
| the streaming subtree recomposes at most once per collected batch | Compose recomposition counter test |
| every action-confirmation sheet shows the exact arguments that will be executed | UI test against a mock tool |
| a prediction with a non-`measured` basis never renders as a bare number | UI test over the three bases |
| the app restores an in-flight task after process death | process-death test |
| cold start stays within budget | Macrobenchmark in CI, fails on regression |
| the app is fully operable with TalkBack | accessibility scanner in CI plus a manual checklist |
| no network call occurs in `on_device_only` mode | network-denied instrumentation test |
