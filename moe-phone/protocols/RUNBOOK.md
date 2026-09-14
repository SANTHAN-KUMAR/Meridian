# moe-phone runbook — the tests you run

Two tracks that do not depend on each other. Do them in parallel.

| track | gates | where | your time |
|---|---|---|---|
| **Phone** | G0 (device facts, memory), G1 (storage) | OnePlus 15R, in Termux | ~1 hour, mostly waiting |
| **GPU** | G2 (routing traces, cache), G3 (sparsity fidelity) | Kaggle first, then a rented H100 | ~30 min setup, then hours unattended |

Nothing here needs root, and nothing modifies a model.

---

## Phone track (G0 + G1) — OnePlus 15R

### 0. Prepare (once)
1. Install **Termux from F-Droid** (<https://f-droid.org/packages/com.termux/>). The Play Store build
   is outdated and its package mirrors are broken.
2. In Termux:
   ```sh
   pkg update && pkg install -y clang git
   termux-setup-storage          # allow access to /sdcard, answer "Allow"
   ```
3. Put the three files from `moe-phone/device/` on the phone: `ufsbench.c`, `memprobe.c`,
   `g0_probe.sh`. Easiest: copy them over USB into the phone's `Download/` folder, then
   ```sh
   mkdir -p ~/moe && cp /sdcard/Download/{ufsbench.c,memprobe.c,g0_probe.sh} ~/moe/ && cd ~/moe
   clang -O2 -pthread ufsbench.c -o ufsbench -lm
   clang -O2 memprobe.c -o memprobe
   ```

### 1. Conditions for every run
- Battery 50–80%, **unplugged** (charging corrupts the power readings), airplane mode on,
  screen on at a fixed low brightness, Termux in the foreground, phone at room temperature and
  not warm from use.
- Free storage of at least 10 GB (`df -h ~`).

### 2. G0 — device facts (2 minutes)
```sh
sh g0_probe.sh > g0_15r.txt 2>&1
```

### 3. G0 — how much memory one app can hold (5 minutes)
Close apps you care about first: this can make Android close **background** apps. It stops
itself before anything is killed.
```sh
./memprobe > memprobe_15r.csv
tail -3 memprobe_15r.csv
```

### 4. G1 — storage (about 30–40 minutes)
```sh
./ufsbench --file ~/moe/ufs.bin --size-mb 8192 --seconds 2 --repeats 3 --power --out ufs_15r.csv
rm ~/moe/ufs.bin              # free the 8 GB afterwards
```
If it prints `O_DIRECT ... unsupported`, that is expected on encrypted storage and is recorded —
buffered mode (with page cache dropped before every configuration) is used instead.

### 5. Bring the results back
```sh
cp g0_15r.txt memprobe_15r.csv ufs_15r.csv /sdcard/Download/
```
Copy them to the computer into `moe-phone/results/<today>/`, then:
```bash
python moe-phone/gates/g1_analyze.py moe-phone/results/<today>/ufs_15r.csv --memprobe moe-phone/results/<today>/memprobe_15r.csv
```
It prints the **G-ROOF v2** command with the measured storage and memory values. Run it.

### If the 15R fails
Only if G0 shows the phone unusable for this work — not merely slow — repeat on the Galaxy S25.
Do not unlock the S25's bootloader: it permanently trips Samsung Knox.

---

## GPU track (G2 + G3)

### Step 1 — Kaggle, OLMoE-1B-7B (free, fits two T4s)
1. New notebook → Settings: **Accelerator: GPU T4 x2**, **Internet: on**.
2. Upload `moe-phone/gates/traces_sparsity.py` (Add Input → Upload), or paste it into a cell with
   `%%writefile traces_sparsity.py`.
3. Cells:
   ```python
   !pip install -q "transformers==4.56.2" accelerate datasets
   !python traces_sparsity.py --selftest
   !python traces_sparsity.py --model allenai/OLMoE-1B-7B-0924 --windows 64 --calib-windows 16 --seq-len 512 --out-dir /kaggle/working/out
   ```
   The self-test must print `0 failure(s)` before the real run means anything.
4. Download `/kaggle/working/out/` (`g3_OLMoE-1B-7B-0924.json`, `traces_OLMoE-1B-7B-0924.npz`) into
   `moe-phone/results/<today>/`, then locally:
   ```bash
   python moe-phone/gates/cache_sim.py moe-phone/results/<today>/traces_OLMoE-1B-7B-0924.npz
   ```

T4s have no bfloat16, so the reference runs in float16; the artifact records this.

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
- **G2 locality**: does Belady — and does LRU — beat the no-locality floor at phone-sized caches?
  If not → finding F4, and G-ROOF's floor-case numbers are the realistic ones.
- **Lookahead recall**: can the next layer's experts be prefetched without training?
