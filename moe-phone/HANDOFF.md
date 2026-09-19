# HANDOFF — read this first if you are a new session continuing moe-phone

Written 2026-09-19 ~19:30 IST by the engine session, before a usage-limit gap. The user will continue from another account.
Everything needed is in this repository. The chat history is not. **Update this file whenever the state changes.**

---

## 1. What the project is (30 seconds)
- **Goal.** Qwen3-30B-A3B, unmodified (Q4_0 GGUF, unsloth, 17,379,988,032 B), ≥ 10 tok/s decode on a OnePlus 15R phone, by
  streaming experts from flash (engine: a fork of BigMoeOnEdge + patched llama.cpp/ggml).
- **Best measured:** 6.76 tok/s, lossless, awake and unplugged (`results/2026-09-19/bmoe_gtier/bmoe_gtier_ab_20260919_0447`).
  BigMoeOnEdge's published engine reproduces at 3.98 on the same phone (claim `bmoe_reproduced_decode`; not a controlled head-to-head).
- **The single source of truth for this phase:** `research/2026-09-19_RESEARCH_SPEC.md` (v2, after a hostile review). **§0 there
  is the user's CLOSING RULE:**
  1. Tier A (lossy routing) is adopted only if some option is "negligible" on all five quality measures (§0 item 4) **and**
     measures ≥ 10.0 tok/s on the phone (awake, unplugged, pre-registered ABBA).
  2. Otherwise run **E1 and E4 only**, decide whether this device can ever reach 10 tok/s, and **close the research** (write it up).
  3. Nothing else is built or tested.
- **Long-term goals of the user** (not blockers): a top-venue paper; a cross-device app/PC product built on the engine.

## 2. Where things stand right now (updated 20:20 IST)
**Latest:** E6 closed (no option negligible). The phone determinism control PASSED (Q4_0 against itself: KL 0.000000, 0/3258 flips,
`results/2026-09-19/bmoe_e6/bmoe_e6_ident_*`). adb was switched to Wi-Fi (192.168.0.65:5555), the phone was UNPLUGGED at 20:17,
and `E1_GO` was written. **E4 = KILL** (20:19; `results/2026-09-19/E4_VERDICT.md`: v5 k=2 host 1.47 ms vs kill 0.40), so S_gpu = 0 and the closure needs
C_R(capped) <= 43.4 ms. E1 is next (conversion started 20:19). What remains is steps 2-3 of §3 below.


### E6 (Tier A quality): essentially decided NEGATIVE
Reference Q8_0 (sha256 a68fe734…), margin = our Q4_0. Data in `results/2026-09-19/bmoe_e6/` (pulled when the chain finishes).
Binding bar "negligible" (user, 18:05, set before any arm was scored): mean KL ≤ 0.130, p99 ≤ 1.94, flips ≤ 10.9%, PPL ≤ 4.02,
MMLU same answer as Q4_0 on ≥ 95/100 and ≤ 2 fewer correct.

| arm | PPL | mean KL | p99 KL | flips | negligible? |
|---|---|---|---|---|---|
| REF Q8_0 | 3.70 | — | — | — | reference |
| FLOOR Q4_0 top-8 | 3.98 | 0.118 | 1.76 | 9.9% | margin |
| T7 (7 experts) | 4.04 | 0.146 | 2.31 | 10.7% | no |
| T6 | 4.20 | 0.210 | 2.87 | 13.0% | no |
| RA1 (route-ahead 1) | 4.08 | 0.168 | 2.60 | 10.8% | no |
| DC05 (drop-cold 0.5) | 4.015 | 0.126 | **2.06** | 9.7% | no (p99 only) |
| DC10 (drop-cold 1.0) | 4.18 | 0.209 | 3.10 | 12.2% | no |
| SUB (substitute 0.15) | 4.048 | 0.163 | 2.60 | 11.3% | no |
Reading: the only options that could reach 10 tok/s (top-k cuts, route-ahead) are clearly non-negligible. Drop-cold 0.5 nearly
passes, but it touches ~2% of experts, worth ~5 ms/token (est.), so it cannot reach 10 even if it passed.
**E6 VERDICT (19:45): no option is negligible — Tier A is CLOSED** under the user's bar. Per §0 the remaining work is E4 + E1, then close.
(The E6 scoring by `gates/e6_score.py` and the phone determinism control confirm these numbers formally; see the chain log.)

### Running unattended (detached processes on the laptop; they survive this session)
1. `host/chain_e6b.sh` finishes E6: SUB, then scoring (`gates/e6_score.py` → `results/2026-09-19/e6_kl_summary.json`). It runs the
   knowledge probe only if some arm is negligible on KL/PPL (else it copies the KL summary to `e6_summary.json`), then touches
   `results/2026-09-19/CHAIN_E6_DONE`.
2. `host/chain_e6_ident.sh` (after CHAIN_E6_DONE) runs the **phone determinism control**: Q4_0 scored against itself. It MUST give
   `ppl-kl: mean 0.000000 … flips 0/…` (in `results/2026-09-19/chain_e6.log`, "[ident] result:"). Anything else means every E6
   number must be re-examined before any conclusion. Then it touches `CHAIN_E6_IDENT_DONE`.
3. `host/chain_e1.sh` (after CHAIN_E6_IDENT_DONE) **PAUSES and waits for `results/2026-09-19/E1_GO`**. The user unplugs for E4/E1
   (the deployment condition). To release it:
   ```
   export ANDROID_SERIAL=3C15CK0028J00000       # while still on USB
   adb tcpip 5555                                # only when NO phone job is running (it kills phone jobs)
   adb connect 192.168.0.65:5555                 # phone Wi-Fi IP at handoff; re-check with: adb shell ip -4 addr show wlan0
   echo 192.168.0.65:5555 > "results/2026-09-19/E1_GO"   # then tell the user to unplug
   ```
   Then it runs **E4** (`host/gpu_ffn/run_phone_e4.sh /tmp/claude-1000/gxcases`, the gx session's variant 5, ~15 min, log
   `results/2026-09-19/e4_run.log`), deletes the Q8_0 from the phone, converts the pre-repacked Q4_0 on the phone (marker must say
   `repacked_tensors 138`), and runs **E1** (`device/bmoe_e1.sh`, ~1.5 h), then pulls it and touches `CHAIN_E1_DONE`.
   Log: `results/2026-09-19/chain_e1.log`.
   *If `/tmp/claude-1000/gxcases` has been wiped, regenerate it with `host/gpu_ffn/make_cases.py` (ask the gx session / read its NOTE).*

### Check liveness
`pgrep -fa "host/chain_"` on the laptop. On the phone: `adb shell 'cat /data/local/tmp/moe-stream/.phone_busy; ps -A -o ARGS | grep -E "bmoe|gx_"'`.
If a chain died, re-launch it with `setsid nohup bash host/<chain>.sh > /dev/null 2>&1 < /dev/null &`. Each one waits for the
previous marker file, so the order is preserved.

### E7 (added 20:30 at the user's request: "run it tonight")
E4 closed the GPU as a HELPER. E7 tests the GPU as the MAIN engine (the whole graph on llama.cpp's OpenCL/Adreno backend, which barely
throttles). It is pre-registered in `device/bmoe_e7.sh`: Qwen3 truncated to 4 and 8 layers, t(L) = a + bL projected to 48 layers,
and a 10-minute sustained GPU run. Decision: POSITIVE if C_gpu48 <= 28.5 ms, KILL if > 50.5 ms, MIDDLE = "not supported".
`host/chain_e7.sh` runs it after CHAIN_E1_DONE (log `results/2026-09-19/chain_e7.log`), then touches CHAIN_E7_DONE.
Scoring by hand from the csv files: median avg_ts (tok/s) per row -> ms/token; fit a, b; apply the rule.

## 3. How to close (after E4, E1 and E7 land)
1. **E4 verdict** (pre-registered in spec §6 E4, v2). Take `BENCH spin=1 variant=5` host-visible `median_ms` at k=2 and k=1 from
   `e4_run.log` or the gx results dir it names. Positive: k=2 ≤ 0.25 ms and k=1 ≤ 0.15 ms. Kill: k=2 ≥ 0.40. Middle: 3 more rounds
   (`E4_BENCH_ARGS="--variants 5 --ks 1,2 --iters 300"`), else kill. The saving it allows is S_gpu ≤ 10 ms/token (0 if killed).
2. **E1 score:** `python3 gates/e1_score.py results/2026-09-19/bmoe_e1/bmoe_e1_<stamp> --s-gpu <S_gpu> --out results/2026-09-19/e1_summary.json`.
   Rows are valid only with ~0 MiB/token and ~0 stall (the replay). **Closing formula** (pre-registered in `device/bmoe_e1.sh`):
   10 tok/s lossless is reachable iff `C_R(capped) + 56.6 − S_gpu ≤ 100` ms, i.e. `C_R ≤ 43.4 + S_gpu`, where 56.6 ms is the
   measured awake/unplugged stall + cache management.
3. **Write the closure:** update `research/2026-09-19_RESEARCH_SPEC.md` §12/§0 status and `research/2026-09-19_STATE_OF_RESEARCH.md`,
   add a `research/2026-09-19_CLOSURE.md` with the verdict and every number with its artifact, then commit, push, and sync the
   dedicated repo (§5).
   The paper framing options are in `research/2026-09-19_paper_framing.md`; the novelty search is `research/2026-09-19_novelty_search.md`.

## 4. Key artifacts
| what | where |
|---|---|
| rules of the repo (read first) | `../CLAUDE.md` |
| definitions (Tier E / Tier A) | `ESTIMAND.md` |
| claims (checked) | `CLAIMS.md`, `gates/claims_check.py` |
| state of research + gut-check analysis | `research/2026-09-19_STATE_OF_RESEARCH.md` |
| spec + hostile review | `research/2026-09-19_RESEARCH_SPEC.md`, `research/2026-09-19_HOSTILE_REVIEW.md` |
| tonight's night summary | `results/2026-09-19/NIGHT_SUMMARY.md` |
| engine source (cumulative, verified) | `tools/patches/0019-SNAPSHOT-bigmoeonedge-engine-tree-vs-74ba18f.patch` (+ NOTE); live tree `../../moe-work/BigMoeOnEdge` |
| phone binaries (persistent) | `/run/media/santhankumar/New Volume/moe-work/stage/bmoe-i8mm-0026`, `-0027` (MD5 files inside) |
| models (laptop) | `/run/media/santhankumar/New Volume/moe-work/models/` (Q4_0, Q8_0 + sha256, OLMoE) |
| GPU kernel work (gx session) | `host/gpu_ffn/NOTE.md`, results `results/2026-09-19/gx_*` |
| E6 corpus | `results/2026-09-19/e6_corpus/` |

## 5. Working rules the user set (also in the Claude memory dir of this Linux user, if the new account shares it)
- Commit and push after every milestone; stage only the files you touched (the checkout shows ~200 CRLF-only modified files: never
  `git add -A`). Commit as SANTHAN-KUMAR with **no Claude co-author trailer**.
- After pushing, sync the dedicated repo SANTHAN-KUMAR/moe-phone `main` (GitHub counts contributions only there):
  clone the branch `moe-phone/15r-bringup-measured` into a temp dir,
  `git filter-repo --force --path moe-phone --path CLAUDE.md --path LICENSE.txt --path .gitignore --path .gitattributes`, check
  `git merge-base --is-ancestor <remote main> HEAD`, check authors are only SANTHAN-KUMAR with 0 co-author trailers, then
  `git push https://github.com/SANTHAN-KUMAR/moe-phone.git HEAD:main` (never forced).
- Subagents: Sonnet only. No Qwen3 runs on the laptop (RAM). Laptop builds are memory-capped:
  `systemd-run --user --scope -q -p MemoryMax=$((MemAvailable_MiB-3072))M -p MemorySwapMax=0 cmake --build <dir> -j2 --target bmoe-cli`.
- Phone: USB serial 3C15CK0028J00000, Wi-Fi 192.168.0.65:5555 when tcpip is on. Shared lock file
  `/data/local/tmp/moe-stream/.phone_busy` (the gx session uses it too). Battery guard 25%. Unplugged runs must hold the phone
  **Awake** (waker in the device scripts; the unlock PIN is in `~/.config/moe-phone/phone_pin`, never in the repo), or rows are
  bimodal from SoC autosuspend. Quality-only runs (KL/PPL) may run screen-off. Watch heat: thermal status 3 + shell ~49 °C happened
  on USB power with the screen on; the E6 script has a cooldown gate.
- Method (spec §9): floor first; wall time is the only result; remove work, don't move it; oracle before predictor; a transfer
  test before any proxy; the resolution check; one architecture A/B per phone-day; pre-register before the first row; retract by
  deletion.
- The user wants direct, honest reporting (no hype), decisive tests, and no testing rabbit holes.

## 6. Other session
A second Claude session (the gx/libgx GPU-kernel session, name `identifying-variation-1f` at handoff) owns `host/gpu_ffn/`. It
built E4 variant 5 (commit 8dbcfb9). Coordinate with it through the lock file and messages; don't edit its files.
