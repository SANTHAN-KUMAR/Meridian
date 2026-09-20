# 7 — The agent runtime: a phone-native harness for everyday tasks

Layer D of [`02_ARCHITECTURE.md`](02_ARCHITECTURE.md). The component that turns the compute platform
into something a person uses. It is shaped like a coding-agent harness — a loop of observe, decide,
act, verify, with tools and a context manager — but its workload is the opposite of a coding
agent's, and that difference drives every decision below.

---

## 1. Scope: what the agent is for, and what it is not

**For:** tasks a person does on their phone many times a day and would hand to a capable assistant:

| task class | examples | dominant tool |
|---|---|---|
| **triage** | summarise the morning's notifications; find the message that needs a reply; flag the calendar conflict | notifications, calendar, messages (read) |
| **compose** | draft a reply in the user's tone; write the message that reschedules a meeting | messages, contacts, calendar |
| **extract** | pull the amount and date out of a receipt photo; read the tracking number off a screen; turn a poster into a calendar entry | camera and OCR, screen, calendar |
| **navigate and set** | turn on the setting that is buried four menus deep; get to the right screen in an app | accessibility actions, intents, deep links |
| **capture and organise** | file a note; save the address from a chat into contacts; add the list from a photo to the shopping app | notes, contacts, clipboard, files |
| **multi-step routines** | "when I get to the office, silence, set the status, open the schedule" | scheduler, device state, several of the above |

**Not for:** software development, long document authoring, anything that needs a repository or an
unbounded context. That reading was made explicitly in the prior side-track and the correction
stands ([`../agentic/ESTIMAND.md`](../agentic/ESTIMAND.md) §0).

The workload difference, and why it matters:

| | coding agent | phone-task agent |
|---|---|---|
| context | grows without bound | roughly bounded by one observation |
| prefix reuse across turns | high | **low — the observation changes every turn** |
| output length | long | **short — one structured action** |
| dominant cost | decode | **prefill** |
| what else is running | nothing that matters | **the target app, in the foreground** |

Every row of the right column is why the resident tier is the default, why prompt order is a hard
requirement, and why the coexistence experiment comes first.

---

## 2. The loop, as a state machine

```
        ┌──────────┐
        │   Idle   │◀───────────────────────────────────────────┐
        └────┬─────┘                                            │
             │ user input / trigger                             │
        ┌────▼─────────┐  ambiguous   ┌──────────────┐          │
        │ Understanding├─────────────▶│  NeedsUser   │──────────┤ (answer / cancel)
        └────┬─────────┘              └──────────────┘          │
             │ TaskSpec                                         │
        ┌────▼─────┐  no capability   ┌──────────────┐          │
        │ Planning ├─────────────────▶│   Refused    │──────────┤
        └────┬─────┘                  └──────────────┘          │
             │ Plan (steps)                                     │
        ┌────▼──────────────────────────────────────┐           │
        │ Executing(step i)                          │           │
        │   observe → decide → gate → act → verify   │           │
        └────┬───────────┬──────────────┬────────────┘           │
             │ verified  │ failed       │ consent needed        │
             │           ▼              ▼                        │
             │      retry/escalate   NeedsUser                   │
             │ all steps done                                    │
        ┌────▼─────┐                                             │
        │   Done   ├─────────────────────────────────────────────┘
        └──────────┘        also: Failed(NoProgress | Refusal | Cancelled)
```

Transitions are recorded as `TurnEvent`s ([`03_INTERFACES.md`](03_INTERFACES.md) §9). No transition
is implicit; a state that cannot name the event that moved it is a bug.

---

## 3. Understanding: `IntentParser`

Input: the user's utterance (text or transcribed speech) plus ambient context the user has
consented to (foreground app, time, location class). Output: a `TaskSpec`:

```
TaskSpec {
  task_class:      one of §1's classes,
  goal:            a one-sentence restatement the user can confirm,
  success_criteria: [checkable predicate],       // what "done" means, machine-checkable where possible
  constraints:     { deadline?, privacy: "on_device_only" | "may_use_network", consent_mode },
  inputs:          [ {kind, ref} ],               // a photo, a screen, a contact, a message thread
  ambiguity:       [ {question, options?} ]       // empty means proceed
}
```

**Clarification policy.** Ask **once, up front**, only when the ambiguity would change which tools
are used or whether an irreversible action occurs. Never ask mid-execution unless a consent gate
requires it. A restatement of the goal is shown before execution for any task that writes, sends
or transacts; read-only tasks proceed.

The parser runs on the resident tier with `task_class: classify`, structured output constrained by
grammar. It is the cheapest step and it must feel instantaneous.

---

## 4. Planning: skills first, model second

**Deterministic skills are the no-data baseline, and they win when they apply.** A skill is a
hand-written procedure for a recurring task: "add a calendar event from these fields", "toggle this
setting", "reply to the latest message in this thread". It uses no model, or uses one only for a
sub-step (drafting text). The `TaskPlanner` matches a `TaskSpec` against the skill library first and
falls through to model-generated plans only when no skill fits.

This is not a shortcut. It is [`../../CLAUDE.md`](../../CLAUDE.md) §8's floor made structural: the
agent must beat scripted automation on the tasks scripts can do, and the cheapest way to guarantee
that is to *be* the script on those tasks.

For the fall-through case, the planner asks the resident tier for a plan of typed steps:

```
PlanStep { id, description, tool?: ToolSpec.name, args_schema, expected_observation, verifier, budget_ms }
```

Plans are bounded (a maximum step count declared per task class), and each step has a verifier
before it has an action.

---

## 5. Observation and the context manager

### 5.1 What an observation is

A **serialised, filtered accessibility tree** of the current screen, plus a small state block
(foreground app, time, notification count, connectivity). Screenshots are used only when the tree
is insufficient (a canvas-drawn app, an image the task is about) and a vision-capable model is bound.

Serialisation rules that keep prefill affordable:

- keep interactable and text-bearing nodes; drop decorative and layout-only nodes;
- assign short stable ids per turn; the action refers to ids, never to coordinates, unless a
  screenshot path is in use;
- collapse repeated list items after a fixed count with a marker;
- hard cap the observation's token count per plan; truncate from the least-interactable nodes;
- **diff against the previous observation** when the screen changed little, sending the delta plus
  an anchor.

### 5.2 Prompt order is a hard requirement

```
[ stable prefix ]  system instructions · tool schemas · skill hints      (unchanged across turns)
[ task context  ]  goal · plan · step history · verified facts           (append-only)
[ observation   ]  the current screen or state                            (replaced every turn)
```

The engine reuses the KV for whatever prefix is unchanged
([`06_EXECUTION_ENGINE.md`](06_EXECUTION_ENGINE.md) §6.2). Interleaving volatile content into the
prefix destroys that reuse and turns every turn into a full prefill. The context manager enforces
the order structurally: the three blocks are separate buffers and the prompt is their concatenation.

### 5.3 Budgeting

The context manager holds a per-plan token budget from the `ModelBinding`'s context cap and spends
it: prefix (fixed), history (compacted when over budget — verified facts are kept, step narration is
summarised), observation (capped). It never exceeds the cap and asks for a rebinding with
`needs: ["long_context"]` when a task genuinely needs more.

---

## 6. Deciding: model routing inside the agent

The agent never picks a model. Per step it issues a `CapabilityRequest`
([`03_INTERFACES.md`](03_INTERFACES.md) §7) and acts on the `ModelBinding`.

| step kind | request | why |
|---|---|---|
| intent classification | `classify`, `draft` quality, tight latency | cheapest possible; grammar-constrained |
| next action from an observation | `tool_call`, `standard`, latency to first action ≤ the interactivity target | the common case; resident tier, NPU or GPU prefill |
| drafting user-facing text | `converse`, `standard` or `high` by the user's preference | quality matters more than speed here |
| a step that failed verification twice | same request with `quality_floor: "high"` | **escalation** |
| extraction from an image | `extract` with `needs: ["vision"]` | binds a vision-capable model or refuses |

**Escalation triggers** are explicit and logged: two failed verifications of the same step; a parse
failure of the model's structured output; a low-confidence self-report the grammar makes the model
emit; or a task class the policy marks as high-stakes. The binding's `escalation` field carries the
predicted added latency and energy **before** the step is retried, so the runtime — or the user, by
policy — can decline it. This is RQ3's mechanism in the product: the streamed tier is reached
through a measured decision, and every escalation is a datum on whether it was worth it.

---

## 7. Tools

### 7.1 The registry

Every tool is a `ToolSpec` ([`03_INTERFACES.md`](03_INTERFACES.md) §7.1) with effects, reversibility,
consent class, cost hint, a named postcondition and a checkpointability flag. The registry is the
single source for the schemas placed in the model's stable prefix, so a tool that is not registered
does not exist to the model.

### 7.2 Categories, and what each needs from the OS

| category | examples | mechanism | consent |
|---|---|---|---|
| **system intents** | open an app, share content, dial, navigate to a settings page | standard intents and deep links | none for reads, once for launches that leave the app |
| **UI actions** | tap, type, scroll, back, on a named node | accessibility service | once per app the user enables it for; every time for `transact` effects |
| **notifications** | list, read, dismiss, act on an action button | notification listener | once |
| **calendar / contacts / messages** | read, create, send | content providers and the messaging apps' own intents | reads once; **sends every time** unless the user has set a standing policy for a named contact |
| **files and media** | read a document, save a note, pick a photo | storage access framework | per pick |
| **camera and OCR** | capture, read text | camera plus on-device text recognition | per capture |
| **clipboard** | read on demand, write | clipboard manager | write freely; read only in a user-initiated step |
| **scheduler** | run a routine later or on a condition | job scheduling with constraints | once per routine |
| **network** (optional) | fetch a page, call a service | HTTP | **off by default**; `privacy: on_device_only` blocks it entirely |

### 7.3 Tool authoring rules

- A tool's postcondition is checkable from observable state, or the tool is marked `verifier: none`
  and the plan treats its result as unverified.
- Anything that can run in the background is checkpointable, so preemption
  ([`02_ARCHITECTURE.md`](02_ARCHITECTURE.md) §6.3) never loses work.
- Cost hints are measured on the device and updated by the `Recorder`, so the planner's step
  budgets are real.

---

## 8. Acting: the safety gate

`SafetyGate` sits between the model's chosen action and its execution. It is a classifier over
`ToolSpec.effects` and `reversible`, plus content rules, and it is **not** the model: the model can
propose, and only the gate can execute.

| class | examples | behaviour |
|---|---|---|
| **read, reversible** | open, scroll, read a message | execute |
| **write, reversible** | create a note, add a calendar entry | execute; show an undo affordance |
| **communicate** | send a message, post, email | **confirm with the exact content and recipient shown**, unless a standing policy covers this recipient |
| **transact / irreversible** | pay, delete, submit a form that cannot be recalled, change security settings | confirm every time; never covered by a standing policy |
| **outward-facing with data** | any network egress | blocked under `on_device_only`; otherwise confirm, with the data shown |

Content rules: personal data detected in an action's arguments is highlighted in the confirmation;
a message drafted in the user's voice is always shown before sending. Consent given in one task
never extends to another task ([`02_ARCHITECTURE.md`](02_ARCHITECTURE.md) §7).

Rate limits bound the number of communicate and transact actions per task and per hour, so a
looping agent cannot send twenty messages.

---

## 9. Verifying, retrying, and stopping

- After every action, the step's verifier runs against a fresh observation. Verified → next step.
- Failed → **retry with evidence**: the failed observation and the verifier's reason are appended to
  the task context, so the model sees why. Two failures → escalate (§6). Escalation failed → stop
  with `NoProgress`, report state, hand back to the user.
- A **progress detector** watches for cycles: the same action on the same observation twice is a
  cycle, and the runtime stops rather than spending a third turn.
- Every stop leaves the device in a **known state**: any partially executed reversible step is
  undone if the tool supports it; anything else is reported precisely.

---

## 10. Memory and persistence

- **Task state** persists across app restarts so a preempted or interrupted task can resume.
- **User preferences** (tone, standing consent policies, favoured contacts) are structured records,
  editable by the user, never inferred silently.
- **Verified facts** learned during a task (a contact's number, an address) are offered for saving,
  not saved automatically.
- Nothing leaves the device unless the user opts in, and then only structural telemetry
  ([`03_INTERFACES.md`](03_INTERFACES.md) §9) — never prompt or screen content.

---

## 11. Evaluation: the harness is part of the product

`Recorder` writes every turn; the evaluation harness replays task suites against the same runtime
in a controlled regime, and it is the **only** source from which `T` and `S`
([`01_RESEARCH.md`](01_RESEARCH.md) §2) are computed.

- **Task suite**: fixed, versioned, with programmatic success predicates. Public mobile-agent
  benchmarks ([`01_RESEARCH.md`](01_RESEARCH.md) §5.1) supply candidates; the everyday classes of §1
  supply the ones that matter for this product. Frozen before any run (`PL-S1`).
- **Baselines, all three, every time**: the skill library alone (scripted, no model); the most
  common action always (constant); the resident tier alone with escalation disabled (the
  one-parameter analogue — it is what decides whether the streamed tier earns its place, RQ3).
- **Reported**: median and IQR of `T` over completed tasks, `S`, energy per completed task, memory
  retention, escalation rate and escalation *value* (tasks completed only after escalation), the
  distribution of turns per task, and the failure taxonomy — never a mean alone.
- **Regime**: the validity guard is active; a task run during which the device dozed or the engine
  lost its lease is reported as such and excluded from `T`, counted in the failure taxonomy.

---

## 12. Output format contract

The model's action output is a **grammar-constrained** structured object, decoded with the engine's
grammar support so that parsing cannot fail on well-formed generations:

```
Action { step_id, tool: name, args: object, confidence: "high" | "medium" | "low", note?: string }
```

`confidence` is required, produced by the model, and is one of the escalation signals. It is a
self-report and is treated as one: it enters the log and the escalation rule, never a metric.

---

## 13. Invariants and tests

| invariant | test |
|---|---|
| the prompt's stable prefix is byte-identical across turns of one task | prefix hash check per turn |
| no `communicate` or `transact` action executes without a consent record | audit query over the recorder |
| every executed action has a preceding gate decision and a following verification attempt | event-order test |
| a task stops within its declared step budget | bounded-loop test with a non-progressing mock tool |
| a preempted background task resumes from its last checkpoint with no repeated side effects | fault-injection test |
| `on_device_only` tasks generate zero network calls | network-denied environment test |
| `T` is never computed over a set that includes failed tasks | harness unit test |
| the observation serialiser is deterministic for the same tree | golden-file test |
