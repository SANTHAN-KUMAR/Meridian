# Implementation status

What exists under `platform/impl/`, what conformance level it reaches, and
what it does not do. This extends the stub registry of
[`../10_EXTENSION_POINTS.md`](../10_EXTENSION_POINTS.md) section 4 — an
unregistered stub is a false claim (CLAUDE.md section 7.5), so everything
this code skips is listed here, not silently omitted.

## 2026-09-22: engine generalization, pre-download prediction, agent (read this first)

What changed, what was measured, and what is still not true. Every figure below is read from a committed artifact
under `../../results/2026-09-22/` (`oneplus15r/`, `nord/`); `tools/validation_report.py` regenerates the prediction table.

**Any model, not a registry.** `Planner.derive` accepts every GGUF llama.cpp loads: dense models run resident, MoE models
(expert tensors found by llama.cpp's `_exps` naming, with integrity checks) run resident or, for the ten architectures the
engine can stream (`Planner.STREAMABLE`), streamed. `RemoteGguf` reads a remote file's header with HTTP Range requests, so a
model is evaluated before it is downloaded. `assets/catalog.json` carries ungated Hugging Face GGUFs whose cards were
derived from their real headers (`app/test/CatalogMain.java`, which also cross-checks any entry present locally); any other
https `.gguf` URL and Hugging Face search results go through the same path.

**Compute is measured on the phone (closes PL-E11).** `ComputeProbe` runs synthetic calibration GGUFs (constant weights, so they
compress to a few MB in the APK; `tools/make_calib_models.py`) through the real engine and fits
`t_token = t0 + n_layer * t_layer + sum_T bytes_T / W_T` for Q4_0, Q8_0, Q4_K, Q6_K and MXFP4. Rows taken with the screen off
are excluded and counted. When the i8mm engine build is bundled and the CPU has i8mm, both builds run and the one with the
shorter representative turn (200-token prompt + 64 tokens) is chosen: on the 15R that was the portable dotprod build, because
the i8mm build prefilled slower (`oneplus15r/profile.json`, `cpu.compute.value.variant_basis`).

**Predictions before download, checked after.** `Predictor` prices compute, the KV re-read and (streamed) flash stall and cache
management from measured inputs only, labels each figure `calibrated` or `prior`, ranks on the lower bound, and refuses without
a compute probe. Prior-basis figures are shown as ranges only. Checked on the 15R against real chat turns
(`oneplus15r/prediction_validation.json`, generated from the app's audit trail):

two models so far (Qwen3-0.6B and Qwen2.5-3B-Instruct, both Q4_0): each row of the artifact has the prediction, every
observed turn, the median, the point error and how many turns fell inside the predicted range. Regenerate with
`python3 tools/validation_report.py ../../results/2026-09-22/oneplus15r/audit.jsonl out.json`.
The streamed tier was run in the app on the 15R with OLMoE-1B-7B (larger than the phone's measured grant, so the plan chose
streaming with the cache sized from the grant); its turns are in the same artifact. During that run the governor saw the OS
reclaim part of the engine's memory and stepped down its ladder (smaller expert cache, reload), as designed; the plan's
prediction was not recomputed for the smaller cache, which is a known gap.
Known gap: the streamed tier's prediction is `prior` (hit-rate curve and cache-management cost transferred from the research's
Qwen3-30B-A3B measurements); a back-test against the research's Nord measurement of Qwen3-30B-A3B
(`../../results/2026-09-19/nord/`) had the measured rate inside the predicted range but the point estimate well above it,
which is why prior figures are shown as ranges and ranked by their lower end.

**Auto-configuration (closes PL-E25, PL-E23).** `AutoPlan` picks the tier (resident if it fits the measured memory grant, else
streamed with the cache sized from the grant), the largest context of 4096/3072/2048 that fits, threads and CPU mask
(`Topo`: fastest non-little cores, at most 4, until the placement A/B has run), I/O lanes on the other cores, and the engine
build. Two engine builds share one APK through same-length ELF renames (`app/elf_rename.py`). Plans feed the existing
Lease and Governor; the Governor falsifies a plan whose observed speed leaves the predicted range.

**Agent.** `Agent.runLoop` is a planner-executor. The model first breaks the request into ordered steps (grammar-constrained
list; one step for a simple request). For each step a constrained yes/no decides whether a tool is needed; tool steps are
grammar-constrained calls over the registered tools, each executed behind the consent gate and verified against device state,
followed by a constrained "is this step done?"; writing/thinking steps produce text that later steps can use. Results carry
forward in a bounded scratchpad (long results are clipped), the final answer is written from the results only, and the harness
appends the list of verified actions plus a note for anything left to the user (a drafted message is not sent; an opened
settings panel is not switched). Every finished task is appended to a task memory (`files/memory.jsonl`) that the `recall`
tool reads for later "what did you do / summarise" requests. The engine keeps the tool-list prefix in its KV cache across
tasks (engine patch 0021), so only the new request is prefilled.

Tools registered (`Tools.java`, one class per tool with arguments, consent class and a verification check):
- phone state and settings: battery, storage, device info, flashlight (torch state read back), media volume (read back),
  brightness (read back; needs the system "modify settings" permission, which the tool opens), media keys, settings pages and
  system quick panels (Wi-Fi, internet, volume, NFC), clipboard (read back);
- apps and intents: open an installed app by name, web search in the browser, open a URL, maps/navigation, play music by search,
  camera, share text;
- time: alarm (verified against the system's next alarm clock), timer, calendar new-event screen, current date and time;
- information: web results read as text (DuckDuckGo HTML), read a page's text, arithmetic with percentages;
- people: contact lookup by name; message, call and email only as drafts in the user's own apps, consent every time;
- memory: notes (add, list, clear with consent), missed calls (call-log permission), coarse location with area name, recall of
  earlier tasks.

Evaluation (`Eval.java` suite-v2, requests that never name a tool, predicates on device state, a simulated user who approves
consent only for the tool a task is about), all on the 15R, results in `oneplus15r/eval_suite-v2_*.json` (`summary.S` in each):
Qwen3-1.7B with the earlier loop variants, Qwen3-4B with the observe-act loop, and Qwen3-4B with the planner-executor. The
planner-executor scored below the observe-act loop on this suite; the misses are recorded in the file as they happened (the
suite's predicates were fixed before the runs and were not changed after seeing results). The no-model baselines are in the
first Qwen3-1.7B file.

**Not done, stated plainly.**
- **Agent scope, 2026-09-22 (second pass): the three missing modules now exist.** Retracted: the previous statement that they were
  not built. What was added, and what was checked:
  - *Screen control* (`A11y.java`, `ToolsScreen.java`: `screen_read`, `screen_tap`, `screen_type`, `screen_scroll`, `screen_nav`).
    The accessibility service serialises the active window into interactable/text nodes with per-snapshot ids (60 nodes and
    1,800 chars max; password fields never read). Every action's postcondition is read from the live window afterwards, and the new
    screen is returned as the next observation. Consent: once per app per task (shared with `open_app`); a tap whose label sends,
    pays, deletes or submits (`Parse.sensitive`) asks every time with the label shown. The consent card is drawn as an accessibility
    overlay when another app is in front. **Checked:** the service binds on the 15R. **Not yet checked:** an end-to-end in-app task.
  - *Communication* (`ToolsComms.java`: `sms_send`, `phone_call`, `contact_add`, `sms_read`, `notifications_read`,
    `notification_reply`, `notification_dismiss`; `Notifs.java` listener). Sends and calls are real, each gated `every_time`,
    with the resolved recipient and exact text shown; at most 5 per task and 20 per rolling hour (`Agent.gate`, `files/comm_log.txt`).
    Verification: SMS by the radio's SENT result for every part, a call by the audio mode entering a call, a reply by the notification
    being cleared or re-posted with the text. **Checked:** listing notifications on the 15R. **Not checked:** a real send or call
    (no test recipient was used).
  - *Daily tasks* (`ToolsDaily.java`): `weather` (Open-Meteo), `news` (Google News RSS), `reminder_add/list/cancel` (AlarmManager +
    `ReminderReceiver`, re-armed after reboot), `calendar_read/add` (provider, direct insert), `places_nearby` (OpenStreetMap
    Overpass with mirror fallback; ratings only with a Google Places key in `keys.json`, otherwise the result says ratings are
    unavailable), `files_find/file_read/file_save/file_open`, `do_not_disturb`, `ringer_mode`, `exchange_rate` (open.er-api.com).
    **Checked on the 15R through the debug `ToolProbe`, each against its own postcondition:** weather, news, exchange_rate,
    reminder_add/list, ringer_mode, do_not_disturb, file_save, places_nearby (verified after the Overpass user-agent fix; the
    public instance is slow and rate-limits). Runtime permissions are asked at first use (`Perms.java`).
  - Pure parsing (`Parse.java`: dates and times, distances, WMO codes, currencies, the sensitive-label rule, RSS) has 34 host
    gates in `test/ToolGatesMain.java`, whose expected values come from the calendar and published constants (`test/run_gates.sh`).
- **Defects found and fixed in this pass.**
  - *D-7 (hang).* `Engine` dropped `BMOE_ERROR` lines, so a recoverable engine error (context overflow) left `Chat.ask` waiting
    600 s. Fixed: the error ends the turn; overflows are caught per step, counted (`context_overflows` in the task memory) and
    reported.
  - *D-8 (safety, found by the UI session on the Nord with OLMoE).* "What's my battery level?" was planned as 8 bare tool names, and
    7 unrelated no-consent actions ran (open_app, take_photo, set_brightness...), including one after the user pressed Stop.
    Fixed: `Agent.sanePlan` drops bare-tool-name steps and keeps a short single-clause request as one step; `Agent.cancel()` is
    checked before every model call and tool execution; set_brightness, take_photo, open_settings, quick_panel and open_app now ask
    once per task; at most 24 tool calls per task.
  - *D-9 (latency).* Qwen3-4B prefills at 9-16 tok/s on the 15R in every recorded run (`files/audit.jsonl`), and the loop spent up
    to three model calls per step ("need a tool?", the call, "done?"), each re-reading the scratchpad: one yes/no call took 143 s.
    Changed: a step now opens with one call (a tool call or the step's written result), and a verified single action ends the step.
    `Agent.warm()` prefills the ~1.8k-token tool prefix at load time. The suite-v2 scores were measured on the previous loop and
    were not re-run.
- **Extension point.** A new capability is one `Tools.Tool` subclass registered in `Tools` (name, description, argument schema,
  effects, reversibility, consent class `none`, `once` or `every_time`, a named postcondition, `execute`, `verify`, and optionally
  `note`, `consentText`, `consentFor`, `consentScope`, `resultChars`, `observes`). The grammar, the planner-executor, the consent
  gate, verification, the action summary and the task memory pick it up with no other change. Runtime permissions are asked by
  the tool itself through `Perms.need`.
- **Limits that remain.** WhatsApp and other messengers can only be read through their notifications or their on-screen content
  (there is no API for chat history). OpenStreetMap has no ratings. An e-mail can be sent only by the screen tools pressing Send
  in the mail app (asked every time). A task run in the background competes with the foreground app for CPU.
- Thermal derate (T3) is not in the predictions; they describe the first minutes of use.
- The 15R's memory grant was measured with many user apps resident (`oneplus15r/profile.json`, `memory.grantable_quiesced`);
  the research measured much larger grants on the same phone when quiesced (`EVIDENCE.md`, `dev_budget_max_mib`), so the
  recommendation list on this phone is conservative.
- The Nord's compute probe in this session ran while charging and dozing and is labelled `prior` (`nord/`).

## Conformance reached (per `../10_EXTENSION_POINTS.md` section 1)

**Delivered: `app/dist/meridian.apk`** (build: `app/build.sh`; details: [`app/README.md`](app/README.md)). One arm64 APK containing the
engine, the probe suite, the planner and the agent runtime. Models are downloaded or imported, never bundled.

| level | state | what exists |
|---|---|---|
| **L0 runner** | done | research engine (`bmoe-cli`, session protocol) + resident tier + Android app shell + foreground service; typed refusals (`EngineUnsupported`, `ArchitectureUnsupported`, `Infeasible`, `IntegrityFailed`) |
| **L1 profiled runner** | done | on-device T0/T1 profile with a per-second regime sampler; every number carries measured/prior/unknown; capability report |
| **L2 planned runner** | **partial** | ModelCard from real GGUF headers; feasibility verdicts (sound `Infeasible` or `NotCalibrated`); measured thread-placement A/B with the real engine. **No PerformanceModel, no memory lease protocol, no degradation ladder** (so no rate is ever predicted) |
| **L3 governed platform** | **no** | no governor, no streamed tier exposed, no T2/T3 profiling, no fidelity-gated transforms |
| **L4 agent platform** | **partial** | ToolSpec registry, grammar-constrained plan/act/answer loop, consent gate, postcondition verification, audit trail. **No frozen task suite and no three baselines (PL-S1)**, so no success rate `S` is claimed |

The engine in the APK is the research fork (BigMoeOnEdge + `tools/patches`, version string 0.23.0) with one more patch,
`tools/patches/0020-*` (per-request grammar). The app currently runs it in resident mode only; its streaming flags exist in
the binary but are not exposed (`PL-E25`).

## Remaining stubs introduced or changed by the app

| id | what | state |
|---|---|---|
| ~~`PL-E7`~~ | grammar from ToolSpecs | **closed**: `Tools.grammar*()` + engine patch 0020; string-typed arguments only (others refused at generation time) |
| ~~`PL-E23`~~ | i8mm engine variant | **closed 2026-09-22**: both builds ship (the i8mm one with same-length renamed libraries, `app/elf_rename.py`); `ComputeProbe` picks one per device by measurement |
| `PL-E24` | engine crash isolation | the engine is a child process of the app; a foreground service keeps priority, but there is no separate-process supervisor/restart |
| ~~`PL-E25`~~ | streamed tier in the app | **closed 2026-09-22**: `AutoPlan` chooses streamed when resident does not fit and sizes the cache from the measured grant (quiesced grant until PL-S2 is measured; stated in the plan's `grant_basis`) |
| `PL-E26` | Java tests | only `app/test/test_parity.py` (Java planner == Python planner on 3 real GGUFs); no UI or agent unit tests. Agent behaviour was verified by hand on a Nord |
| `PL-S1` | frozen task suite + baselines | still open; the agent's success rate is unmeasured. Observed: with grammar-constrained plan/act/answer, one 2-tool task succeeded and verified on a 3B model; a free-form loop failed on the same model |

## What was actually measured, on a real device, over adb

Device: OnePlus Nord (AC2001), Qualcomm SM7250, Android 12. Not the OnePlus
15R named in `00_PROBLEM.md` — the 15R was unavailable this session; the
Nord already has prior calibration history in this repository
(`results/2026-09-19/nord/`), and the architecture is device-generic, so the
choice of device does not change any code path (`03_INTERFACES.md` — the
whole point of `DeviceProfile` is that layer 0 is where device specifics
live).

- **T0 static inventory** (`meridian/probes/static_inventory.py`, driving
  the existing `device/g0_probe.sh` + cross-compiled `device/devprobe.c`):
  identity, CPU topology (3 clusters resolved from `/proc/cpuinfo` part IDs
  and `cpuinfo_max_freq`), storage mount/filesystem resolution, accelerator
  device-node presence.
- **T1 fast probes**, cross-compiled with the NDK
  (`platform/impl/build_probes.sh`) from the existing, already-validated C
  sources in `device/`:
  - `dramprobe.c` → `memory.dram_read_gbps` (rows failing the probe's own
    `min_cpu_over_wall` check are rejected and counted, not averaged).
  - `memprobe.c` (anonymous mode, 2 runs) → `memory.grantable_quiesced`.
  - `ufsbench.c` (3 repeats) → `storage.random_read` and
    `efficient_request_size`; one direct-I/O confirmation pair.
  Values: read them from `DeviceProfile.json`, not from this file.
- **ValidityGuard** (`meridian/validity.py`): reads `dumpsys battery` and
  `dumpsys power` to attach real `ValidityConditions` to every measurement,
  and gates `measured` on power=unplugged, foreground=none and
  wakefulness=awake. Any other state downgrades to `prior` with an interval
  from real repeats. The current run (`meridian_l1_nord_awake`) was unplugged, awake, launcher
  foreground, and is `measured`; the earlier `_wireless` run (screen off) is
  `prior` and kept only as evidence for D-5.

Raw artifacts and the assembled profile:
[`../../results/2026-09-20/meridian_l1_nord_awake/`](../../results/2026-09-20/meridian_l1_nord_awake/).

## Defects found in this code after the first commit, and how they were fixed

These are recorded as defects, not as "limitations", because each one was a
mistake in code this pass wrote — not an inherent constraint of the device.

**D-1. Fabricated uncertainty intervals (commit `2fc6e48`).** When a
measurement was out of the deployment regime, `profiler.py` downgraded it
from `measured` to `prior` and attached an interval — because
`contracts.Measured` refuses a non-`measured` value without one. The
intervals were invented to satisfy that check: `[mbps * 0.7, mbps * 1.3]`
for every storage read point (a made-up ±30%, no measurement behind it),
`[knee, knee]` for `efficient_request_size` (a zero-width "interval"), and
`[0, observed]` for `grantable_quiesced` (a lower bound of zero says nothing).
This is the CLAUDE.md §0/§6.4 failure exactly: a plausible substitute that
made a check pass, with nothing asking whether it was the quantity. It also
meant the capability report displayed intervals that looked like uncertainty
quantification and were not. **Fix:** `ufsbench` now runs 3 repeats and
`memprobe` 2 independent runs, and every interval is the real observed
min/max across those repeats (`probes/storage.py`, `probes/memory.py`
`parse_multi`). Regression test:
`test_storage_interval_is_real_spread_not_fabricated_multiplier`. The
committed `results/2026-09-20/meridian_l1_nord/` directory was produced by
the defective version and its `DeviceProfile.json` was deleted
(retraction is deletion, CLAUDE.md 7.6); raw artifacts remain as test fixtures.
**D-5.** The regime gate ignored wakefulness; a screen-off (`dozing`) run was
labelled `measured`. Gate now requires `awake`.

**D-2. Foreground state was hardcoded, not measured.** `build_conditions`
took `foreground="none"` as a default, so a run taken while the user was
watching YouTube would have been reported as quiesced. The user's own
session was in exactly that state. **Fix:** foreground is now read from
`dumpsys activity activities` before and after the T1 window and the worse
of the two is used (`validity.classify_foreground`); empty input yields
`unknown`, never `none`. The first wireless run correctly caught YouTube
(`concurrent_load: ['com.google.android.youtube']`) and downgraded every
field.

**D-3. `ROOT:` substring match.** `"ROOT:" in text` is also true of
`"NOT_ROOT:"`; every unrooted device would have been reported as rooted.
Caught by a test written before the first commit; fixed in the same commit.

**D-4. kHz/GHz mislabel in the capability report** (÷1000 labelled GHz).
Fixed before the first commit.

**Not a defect, recorded for the next reader:** over wireless adb,
`g0_probe.sh` takes ~37 s against <1 s over USB (many forked subshells,
higher per-exec latency on the TCP transport). It looked like a hang; it is
not one. Timeout raised to 90 s.

## Planner slice (L2, feasibility only)

`meridian/gguf.py` reads real GGUF headers (sizes computed from dims and the
ggml block table, cross-checked against file layout); `meridian/planner.py`
derives a `ModelCard` and issues per-tier verdicts. It never returns
"feasible" and never predicts a rate: it refuses `Infeasible` only via a
sound lower-bound argument (assumption A1: a foreground app cannot raise what
one process may keep) and otherwise returns `NotCalibrated` naming every
missing measurement. Registry rows (olmoe, gpt-oss, granitemoe) were observed
in real local checkpoints; qwen3moe is refused because no local file verifies
its pattern. Defect found while testing: offset-gap sizing over-stated tensor
bytes by alignment padding (D-6, fixed).

## Registered stubs (extends `10_EXTENSION_POINTS.md` section 4)

| id | what | current state | removed when |
|---|---|---|---|
| `PL-E21` | engine working-set bytes | excluded from tier need (makes need a lower bound; verdicts stay sound) | measured from a real engine session |
| `PL-E22` | further registry rows | qwen3moe added (observed); others refused `ArchitectureUnsupported` | a real checkpoint's expert pattern is observed |
| ~~`PL-E10`~~ | L0 engine + session protocol | **closed**: engine bundled in the APK and driven over its session protocol (`app/src/com/meridian/Engine.java`) | - |
| ~~`PL-E11`~~ | T2 compute probe | **closed 2026-09-22** by `ComputeProbe` (the real engine on synthetic calibration models, per weight type) rather than a separate ggml benchmark; `cpu.clusters[].matmul_gbps` stays unknown because the probe measures the engine at its chosen placement, not per cluster | - |
| ~~`PL-E12`~~ | thread-placement A/B | **closed for the compute mask** (I/O mask still unknown): app `Placement.java` A/Bs masks with the real engine (ABBA, 2 reps). On the Nord, olmoe: cores 6-7 = 17.3 tok/s vs default unpinned-4 = 8.3 (results in the app's `profile.json`, `cpu.recommended_compute_mask.arms`) | - |
| `PL-E13` | per-accelerator decode/prefill rate, dispatch overhead | `Measured.unknown` for CPU/GPU/NPU; requires an engine (`PL-E10`) | T2 profiling ships |
| `PL-E14` | thermal derate curve, time-to-throttle, recovery (T3) | not attempted; would need a sustained decode load this pass has no engine to generate | T3 profiling ships, per `04_DEVICE_PROFILING.md` §3.5 |
| `PL-E15` | thermal-status OS field in `ValidityConditions` | left `None`; only wakefulness/power/battery are read from `dumpsys` | a thermal-status API call is added to `validity.py` |
| `PL-E16` | ISA features per cluster | parsed from `/proc/cpuinfo` (intersection over cluster cores); captured in `meridian_l1_nord_awake` (cpuinfo read after the run; static file) | done | |
| `PL-E17` | sequential write throughput | `Measured.unknown`; `ufsbench` was only run in random-read mode | a `--patterns seq --modes buffered` sweep is added to the T1 run |
| `PL-E18` | swap/zram fault cost | `Measured.unknown`; `memprobe --file` mode (file-backed residency) was not run | the file-backed regime is added and its fault cost measured |
| `PL-S2` | foreground-coexistence memory grant (`grantable_foreground`) | still never measured anywhere in this project, per `01_RESEARCH.md`/`10_EXTENSION_POINTS.md` — this pass does not change that, since it needs a real target app foregrounded and no app exists | experiment X1 runs, per the docs' own priority order in §7 |
| `PL-E19` | per-rail power | not readable by an unprivileged process on this device either (`battery/current_now` etc. all `DENIED_OR_ABSENT` in `g0_static.txt`) — consistent with `04_DEVICE_PROFILING.md` §7's documented limitation on the 15R | a device grants rails, or a validated battery-counter method is built (`PL-S5`) |
| ~~`PL-E20`~~ | thermal-zone data quality | **fixed**: `validity.valid_thermal_zones()` ports `ufsbench.c`'s own `max_thermal_c()` filter (magnitude + name exclusion) and applies it to every zone in `g0_static.txt`, not one hand-picked name. `ufsbench`'s in-process picks still surface `mmw-pa4-usr` (81.5 °C, unchanging across every row this session) because that zone genuinely passes the magnitude filter — it is real millidegree-range data from a plausible-sounding name, just not a useful proxy for skin temperature. The filter removes clearly-invalid sensors; it does not and cannot validate that a passing zone tracks anything physically meaningful, which only a T3 sustained-load run against a reference could do | T3 profiling ships and cross-checks zone readings against measured throttle behaviour |

## What is deliberately NOT a stub

- The four cross-compiled probes (`dramprobe`, `memprobe`, `ufsbench`,
  `devprobe`) are unmodified from `device/*.c` — their own correctness
  rules (descheduling detection, zram-residency refusal, permutation-based
  random reads, O_DIRECT fallback detection) are inherited intact, not
  reimplemented and possibly weakened.
- `contracts.py`'s `Measured` dataclass enforces its own invariant in
  `__post_init__`: a non-`measured` value must carry an interval. This is a
  runtime check, not a docstring promise — the profiler would crash, not
  silently emit a bad `DeviceProfile`, if a probe tried to report a `prior`
  point value with no interval.

## How to reproduce

```sh
NDK=$HOME/Android/Sdk/ndk/<version> sh platform/impl/build_probes.sh /tmp/meridian_bin
cd platform/impl
python3 -m meridian.cli profile --bin-dir /tmp/meridian_bin \
    --out-dir ../../results/$(date +%F)/meridian_l1_<device>
```

Requires an adb-authorized device attached. Rebuilding the profile from
already-captured artifacts (no device needed): add `--from-existing` and
drop `--bin-dir`.
