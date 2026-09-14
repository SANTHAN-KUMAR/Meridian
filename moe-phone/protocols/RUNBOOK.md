# moe-phone runbook — the tests you run

Two tracks that do not depend on each other. Do them in parallel.

| track | gates | where | your time |
|---|---|---|---|
| **Phone** | G0 (device facts, memory), G1 (storage) | OnePlus 15R, in Termux | ~1.5 hours, mostly waiting |
| **GPU** | G2 (routing traces, cache), G3 (sparsity fidelity) | Kaggle first, then a rented H100 | ~30 min setup, then hours unattended |

Nothing here needs root, and nothing modifies a model.

---

## Phone track (G0 + G1) — OnePlus 15R

### 0. Prepare (once)

1. **Check which Android user the phone is in.** Termux refuses to start outside the primary
   user with *"Termux can only be run as the primary user"*. OnePlus's System Clone / Second
   Space is a secondary user, and a phone left in it will fail here for reasons that look
   like an install problem.
   ```bash
   adb shell pm list users        # the one marked {0:...} is primary
   adb shell am get-current-user  # must print 0
   adb shell am switch-user 0     # if it does not
   ```
2. Install **Termux from F-Droid** (<https://f-droid.org/packages/com.termux/>). The Play Store
   build is outdated and its package mirrors are broken. Open it once so it unpacks its
   bootstrap.
3. In Termux:
   ```sh
   pkg install -y clang openssh tmux
   ```
4. Get `moe-phone/device/{ufsbench.c,memprobe.c,g0_probe.sh}` onto the phone and build them.

   **Line endings matter.** These files run under Android's `/bin/sh`, which fails on CRLF at
   the first blank line (`g0_probe.sh: 12: : not found`) and at every `\` continuation. The
   repository's `/.gitattributes` forces LF on checkout; if you copy them any other way, check
   with `file g0_probe.sh` — it must *not* say "CRLF line terminators".

   *Either* copy via `/sdcard` (needs `termux-setup-storage`, answer "Allow"):
   ```sh
   mkdir -p ~/moe && cp /sdcard/Download/{ufsbench.c,memprobe.c,g0_probe.sh} ~/moe/
   ```
   *or* — better if you want to drive the runs from the computer — start Termux's sshd and use
   `scp`. In Termux, once:
   ```sh
   mkdir -p ~/.ssh && chmod 700 ~/.ssh      # then put your public key in ~/.ssh/authorized_keys
   chmod 600 ~/.ssh/authorized_keys
   whoami                                   # the ssh username, e.g. u0_a432
   sshd                                     # listens on 8022
   ```
   From the computer, over USB (`adb forward tcp:8022 tcp:8022`, then host `127.0.0.1`) or over
   Wi-Fi (the phone's LAN address — this is the one that survives unplugging):
   ```bash
   scp -P 8022 moe-phone/device/{ufsbench.c,memprobe.c,g0_probe.sh} <user>@<phone>:moe/
   ```
   Then build, on the phone:
   ```sh
   cd ~/moe
   clang -O2 -pthread ufsbench.c -o ufsbench -lm
   clang -O2 memprobe.c -o memprobe
   ```

### 1. Conditions for every run
- Battery 50–80%, **unplugged**, airplane mode on, screen on at a fixed low brightness, Termux
  in the foreground, phone at room temperature and not warm from use.
- Free storage of at least 10 GB (`df -h ~`).
- Run long jobs under `tmux` and hold `termux-wake-lock`, so a dropped connection does not end
  the run.
- **Keep the screen on for the whole run, and Termux in the foreground.** A wake lock keeps the
  process alive but does *not* keep it scheduled: with the screen off, a 2-second ufsbench
  configuration took 4.88 s and reported 108 MB/s where the same configuration with the screen on
  gave 814 MB/s — with p50 latency unchanged, which is the signature of a descheduled process
  rather than slow storage. Termux in the foreground also puts it in the `top-app` cpuset;
  backgrounding it changes the operating point mid-run. Enforce it:
  ```bash
  adb shell svc power stayon true     # stay awake while charging; revert with `false`
  adb shell am start -n com.termux/com.termux.app.TermuxActivity
  ```
  `g1_analyze.py` detects and excludes such rows (`n_invalid_process_descheduled`), but a run that
  loses a third of its rows to this is a run to repeat, not to analyse.

**Energy may not be measurable at all, and that is a G0 finding, not a setup error.** The
"unplugged" condition exists so that battery current means something. On the OnePlus 15R
(ColorOS 16, unrooted) *every* `/sys/class/power_supply/battery/*` node is denied — to Termux
**and** to `adb shell` — so `ufsbench --power` emits `nan` and says so on stderr. Check G0's
`power measurement` section before planning any J/token work: if the rails are denied, the
unplugged condition only buys thermal realism, and per-token energy needs the framework
(`adb shell dumpsys battery`, coarse) or external instrumentation.

### 2. G0 — device facts (2 minutes)
```sh
sh g0_probe.sh > g0_15r.txt 2>&1
```
Lines reading `DENIED_OR_ABSENT`, `FAILED rc=N` and `EMPTY rc=0` are results, not noise: they
record what an ordinary app cannot see, which is `POSITION.md` finding F6.

### 3. G0 — how much memory one app can hold (5 minutes)
Close apps you care about first: this can make Android close **background** apps. It stops
itself before anything is killed.
```sh
./memprobe > memprobe_15r.csv          # repeat at least 3x; the spread is large
tail -3 memprobe_15r.csv
```
The figure to use is the last line's `max_VmRSS_MB` — bytes actually **resident in DRAM**, not
bytes allocated. On a phone with a large zram swap the two differ by several times, and only
the resident bytes can be read at DRAM speed. The ceiling is a memory-*pressure* effect rather
than a fixed cap, so it depends on what else is running: record the device's state, and repeat.

### 4. G1 — storage (about 60–75 minutes for the full 3-repeat matrix)
```sh
./ufsbench --file ~/moe/ufs.bin --size-mb 8192 --seconds 2 --repeats 3 --power --out ufs_15r.csv
rm ~/moe/ufs.bin              # free the 8 GB afterwards
```
If it prints `O_DIRECT ... unsupported`, that is expected on encrypted storage and is recorded —
buffered mode (with page cache dropped before every configuration) is used instead. Use
`--repeats 1` for a ~20-minute smoke run, but the run-to-run spread that `ESTIMAND.md` §7 needs
comes only from repeats.

### 5. Bring the results back
```sh
cp g0_15r.txt memprobe_15r.csv ufs_15r.csv /sdcard/Download/     # or scp them off
```
Copy them to the computer into `moe-phone/results/<today>/`, then:
```bash
python moe-phone/gates/g0_summarize.py moe-phone/results/<today>/g0_15r.txt \
       --memprobe moe-phone/results/<today>/memprobe_15r.csv
python moe-phone/gates/g1_analyze.py moe-phone/results/<today>/ufs_15r.csv \
       --memprobe moe-phone/results/<today>/memprobe_15r.csv
```
`g1_analyze.py` prints the **G-ROOF v2** command with the measured storage and memory values.
Run it. Read its thermal line first: if it says the temperature column is suspect, that column
is void for the run and only the bandwidths may be used.

### If the 15R fails
Only if G0 shows the phone unusable for this work — not merely slow — repeat on the Galaxy S25.
Do not unlock the S25's bootloader: it permanently trips Samsung Knox.

---

## GPU track (G2 + G3)

### Step 1 — Kaggle, OLMoE-1B-7B (free, fits two T4s)

Use the ready-made notebook, **`moe-phone/kaggle/g2_g3_olmoe.ipynb`**. It embeds
`gates/traces_sparsity.py` and `gates/cache_sim.py` and runs the whole track.

1. Kaggle → **Create → New Notebook → File → Import Notebook**, upload that `.ipynb`.
2. Right panel: **Accelerator: GPU T4 x2**, **Internet: On**.
3. **Run All.** Every step runs through `subprocess` and raises on a non-zero exit, so the run
   stops at the first failure instead of producing an empty archive. The self-test cell must
   print `0 failure(s)`; nothing after it means anything otherwise.
4. When the last cell finishes, download `/kaggle/working/moe_phone_out.zip` from the **Output**
   panel and unpack it into `moe-phone/results/<today>/`.

The notebook is **generated** — do not edit its code cells. Edit `moe-phone/gates/*.py`, then:
```bash
python moe-phone/kaggle/build_notebook.py          # regenerate
python moe-phone/kaggle/build_notebook.py --check  # CI-style check; a test asserts this too
```
Otherwise the Kaggle numbers stop being numbers from the code in this repository.

T4s have no bfloat16, so the reference runs in float16; the artifact records this. The self-test
was last run locally on CPU against the pinned stack (`transformers==4.56.2`, torch 2.14):
`0 failure(s)` — so a failure on Kaggle is a Kaggle-environment problem, not a code problem.

### Step 2 — rented H100 (80 GB), the models that matter
Same commands, on one H100 for a few hours:
```bash
pip install -q "transformers==4.56.2" accelerate datasets
python traces_sparsity.py --selftest
python traces_sparsity.py --model Qwen/Qwen3-30B-A3B --windows 64 --out-dir out
python traces_sparsity.py --model deepseek-ai/DeepSeek-V2-Lite --trust-remote-code --windows 64 --out-dir out
python traces_sparsity.py --model mistralai/Mixtral-8x7B-v0.1 --windows 32 --out-dir out
```
- Mixtral in 16-bit is ~94 GB and does **not** fit one H100: either rent 2×H100, or run Mixtral on
  Kaggle traces-only later. It is the PowerInfer-2 comparison model, so it is worth the second GPU.
- gpt-oss, Qwen3-Next and TurboSparse use fused or custom expert layouts that the script refuses
  rather than approximates. They are the next engineering item, not part of this first run.

### What the GPU results decide
- **G3 fidelity**: at which density does gate-first sparsity stay within the Q4_0 floor? If none below
  1.0 does, "no retraining" loses its main lever → finding F1 in `POSITION.md`.
- **G2 locality**: does LRU beat the no-locality floor at phone-sized caches, and by how much
  once the hit rate is decomposed against the shuffled and uniform controls? Read the answer per
  *scope* and *replay* — a shared-pool, per-access simulation reports a different and wrong number
  (see the defect table in `README.md`). If LRU does not beat the floor → finding F4, and
  G-ROOF's floor-case numbers are the realistic ones.
- **Lookahead horizon**: how many tokens of future routing must an eviction policy see to reach
  Belady? This is the constructive number, because a batched multi-token verification pass
  supplies exactly its own window for free. Read `--lookahead`, and read `--pred-accuracy` for
  how far a noisy predictor gets under each of the two eviction rules.
