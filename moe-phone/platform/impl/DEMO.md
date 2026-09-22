# Meridian demo — 3–4 minute video runbook

Recorded on the OnePlus 15R with the phone's built-in screen recorder (Quick Settings), edited at up to 2x.
Every number shown on screen is listed in section 4 with the artifact it comes from. Do not put any other number on screen.

## 1. Before recording (5 minutes)

1. Phone unplugged, battery above 50%, Wi-Fi on, location on. Close all recent apps (swipe them away): the 30B model needs the RAM.
2. Do Not Disturb on for the recording (so no personal calls or messages appear), then turn it off only if the WhatsApp scene needs a fresh notification.
3. Settings > Accessibility > Meridian agent: on. Settings > Notification access > Meridian: on.
4. Open Meridian, pick **Qwen3-4B**, tap "Use this one" and wait for "Getting ready" to finish (it pre-reads the tool list: ~2 min, do this before recording).
5. Start the screen recorder.

## 2. The three scenes

### Scene 1 — A model bigger than the phone's memory (~60 s after editing)
- Show the Models screen: **Qwen3-30B-A3B, 17.4 GB file**, and the phone's **11.4 GB RAM** (This phone screen).
- Tap it, "Use this one": the app plans the configuration itself (streamed tier: the model is streamed from storage, only the active experts are in RAM).
- Chat: *"In three sentences, explain why running AI on the phone itself matters for privacy."* Keep the tokens/s line visible.
- Overlay (see section 4): "Qwen3-30B-A3B on a phone with 11.4 GB RAM" · "1.22x faster than BigMoeOnEdge, same phone, same model file, 6/6 runs" · the tokens/s shown live in the app.

### Scene 2 — One sentence, a whole errand list (~70 s after editing, 2x)
Switch back to Qwen3-4B (resident, fast). Task:
> Set an alarm for 6:30 am. Then check my recent missed calls and draft an SMS to each of them saying I was busy and will call back. Then find the nearest restaurants where I can have dinner tonight and save them in my notes.
- Show: the plan appearing as steps; each step ticking **verified**; the approval card for each SMS draft ("Allow once"); the final answer with the **actions taken** ledger.
- Cut to Notes/Clock to show the alarm and the saved restaurant list are real.
- Follow-up task: *"I'm back. Give me a summary of everything you did for me earlier."* (the agent's task memory).

### Scene 3 — Hands inside WhatsApp (~45 s)
Task: *"Open WhatsApp and tell me who sent me the most recent messages."* The agent asks once to operate WhatsApp, opens it, reads the screen, and answers.
Optional second beat (only with a friendly recipient who expects it): *"Reply to <name> on WhatsApp: I'll be there in 10 minutes."* The tap on Send raises the approval card **on top of WhatsApp** — the safety moment.

Closing card: on-device model · verified actions · approval before anything reaches another person · nothing leaves the phone except the web lookups the task asks for.

## 3. Narration (one line per beat)
1. "This phone has 11 GB of memory. This model is 17 GB. It's running on the phone, not in the cloud."
2. "Our engine streams only the experts each token needs — 1.22x faster than the best published engine on the same phone."
3. "Now the agent: one sentence, five steps."
4. "Every step is checked against the phone's real state. Anything that reaches another person waits for my OK."
5. "It remembers what it did."
6. "And it can work inside other apps — but a Send button always asks me first."

## 4. Numbers allowed on screen, and their sources
| on screen | value | source |
|---|---|---|
| head-to-head vs BigMoeOnEdge | 6.67 vs 5.46 tok/s, 1.22x, faster in 6/6 ABBA repeats | `results/2026-09-19/H2H_VERDICT.md` (research repo) |
| model size / phone RAM | 17,379,988,032 bytes / MemTotal 11,366,296 kB | `ls -l` of the GGUF, `/proc/meminfo` on the 15R |
| live tokens/s in the app | whatever the app shows during the take | the app's own audit row for that turn (`files/audit.jsonl`) |
| in-app burst earlier | 9.2–10.3 tok/s decode, 384 tokens, awake, unplugged, in front | 15R `files/audit.jsonl`, 2026-09-22 ~17:30 turns; label as a short burst, not sustained |

**Never on screen:** the retracted "~1.7x" (it compared runs under different conditions); any sustained-speed claim above 7.8 tok/s
(the closure's measured bound, `research/2026-09-19_CLOSURE.md`).
