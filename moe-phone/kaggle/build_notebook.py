"""
Generate moe-phone/kaggle/g2_g3_olmoe.ipynb from the repository's gate scripts.

Why this exists: the notebook carries a COPY of gates/traces_sparsity.py and
gates/cache_sim.py inside `%%writefile` cells, because Kaggle runs a single
uploaded file rather than a repository. A copy that is edited by hand drifts
from the original, and then the Kaggle result is not a result of the code in
this repository (CLAUDE.md section 7.1). The notebook's own header has claimed
"generated ... do not edit the code here" since it was written; until now no
generator existed. This is it, and tests/test_gates.py asserts the checked-in
notebook equals this script's output.

  python moe-phone/kaggle/build_notebook.py            # write the notebook
  python moe-phone/kaggle/build_notebook.py --check    # exit 1 if out of date
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
NOTEBOOK = os.path.join(HERE, "g2_g3_olmoe.ipynb")
EMBED = ("traces_sparsity.py", "cache_sim.py")

INTRO = """\
# moe-phone G2 + G3 on Kaggle — OLMoE-1B-7B

## ⚠ Before you press anything

Open the **right-hand panel → Session options** and set:

- **Accelerator: `GPU T4 x2`** — the default is `None`, and with `None` this notebook stops at
  the third cell. Wait for the session to restart after changing it.
- **Internet: `On`** — the model and the corpus are downloaded.

Then **Run All** (or *Save Version → Save & Run All*). The accelerator must be set **before** the
run starts; changing it afterwards does not re-run anything.

This notebook is **generated** by `moe-phone/kaggle/build_notebook.py` from
`moe-phone/gates/traces_sparsity.py` and `moe-phone/gates/cache_sim.py`.
Do not edit the code cells here — edit the repo files and regenerate, or the
Kaggle result stops being a result of the code in the repository.

Every step below runs through `subprocess` and **raises on a non-zero exit**, so
*Run All* stops at the first failure instead of producing an empty archive. In
particular the self-test must print `0 failure(s)`; if it does not, nothing
after it means anything.

What comes back:

| gate | question | artifact |
|---|---|---|
| **G2** | which experts the model's own router picks, per layer and token; whether the next layer's experts can be predicted without training | `traces_OLMoE-1B-7B-0924.npz`, then LRU / Belady / no-locality-floor hit rates |
| **G3** | how much gate-first sparsity the model tolerates, against the Q4_0 quantization floor as the pre-registered Tier-A margin | `g3_OLMoE-1B-7B-0924.json` |

T4s have no bfloat16, so the reference runs in float16 and the artifact records
that. When the last cell finishes, download `/kaggle/working/moe_phone_out.zip`
from the **Output** panel and unpack it into `moe-phone/results/<today>/`.
"""

ENV_CELL = '''\
# Pinned because a gate's numbers are only reproducible against a fixed
# dependency set (CLAUDE.md section 6). Versions are printed, not assumed.
!pip install -q "transformers==4.56.2" accelerate datasets
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("CUDA devices:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}, {p.total_memory / 1e9:.1f} GB, bf16={torch.cuda.is_bf16_supported()}")

# Checked here, before anything expensive. OLMoE-1B-7B in float16 is ~14 GB of
# weights: on CPU this would not merely be slow, it would be a different
# measurement, and it would run for hours before saying so.
if torch.cuda.device_count() < 1:
    raise SystemExit(
        "\\n*** NO GPU ATTACHED — nothing below this cell can run. ***\\n"
        "Kaggle reports Accelerator: None.\\n"
        "Fix: open the right-hand panel -> Session options -> Accelerator -> 'GPU T4 x2',\\n"
        "wait for the session to restart, THEN Run All (or Save Version -> Save & Run All).\\n"
        "The accelerator must be set BEFORE the run starts; changing it afterwards does\\n"
        "not re-run the notebook.")
'''

RUN_HELPER = """\
import subprocess, sys, time

def run(*args, label):
    \"\"\"Run a gate step and STOP the notebook if it fails.

    A bare `!command` in Jupyter reports a failure and carries on to the next
    cell, which is how a broken run ends as a zip file full of nothing. Raising
    here makes 'Run All' halt at the first real problem.\"\"\"
    t0 = time.time()
    print(f"$ {' '.join(args)}", flush=True)
    rc = subprocess.run([sys.executable, *args]).returncode
    print(f"[{label}] exit={rc} in {time.time() - t0:.0f}s", flush=True)
    if rc != 0:
        raise SystemExit(f"{label} FAILED (exit {rc}) - stop here; later numbers mean nothing.")
"""

SELFTEST_CELL = """\
# Must print "0 failure(s)". Runs on CPU with tiny random models, so it costs
# nothing and it is the only thing standing between a bug and a plausible number.
run("traces_sparsity.py", "--selftest", label="G2/G3 self-test")
"""

MODEL_CELL = """\
# G2 + G3 on the stock checkpoint. ~20-40 min on T4 x2.
run("traces_sparsity.py", "--model", "allenai/OLMoE-1B-7B-0924",
    "--windows", "64", "--calib-windows", "16", "--seq-len", "512",
    "--out-dir", "/kaggle/working/out", label="G3 fidelity + G2 traces")
"""

CACHE_CELL = """\
# G2 offline half: replay the routing traces through an expert cache.
#
# --controls decomposes the hit rate into floor + popularity skew + recency, so
#   the result is not "there is locality" but how much of it is which, against
#   a locality-free null.
# --both-scopes keeps the retracted shared-pool number visible beside the
#   per-layer one that replaces it.
# --lookahead sweeps how many tokens of FUTURE routing a cache needs to reach
#   the offline optimum. This is the constructive number, because a batched
#   verification pass over W tokens supplies exactly that much for free: layer l
#   routes all W positions in one matmul, before it touches layer l's experts.
# Cost note: the lookahead rule scans the cache on every eviction, so runtime
# grows with the cache size. Fractions are capped at 0.30 and horizons at 8 to
# keep this cell to a few minutes; nothing above those changed the conclusion
# when swept locally.
run("cache_sim.py", "/kaggle/working/out/traces_OLMoE-1B-7B-0924.npz",
    "--controls", "--both-scopes", "--both-replays",
    "--lookahead", "0,1,2,4,8", "--fractions", "0.05,0.10,0.125,0.20,0.30",
    "--out-dir", "/kaggle/working/out", label="G2 cache simulation")

# How far a NOISY lookahead (a predictor, rather than a draft) gets, and under
# which of the two eviction rules. Fewer fractions: this is the most expensive
# sweep in the notebook, and the break-even accuracy is what it exists to print.
run("cache_sim.py", "/kaggle/working/out/traces_OLMoE-1B-7B-0924.npz",
    "--lookahead", "4", "--pred-accuracy", "1.0,0.9,0.8,0.7,0.5,0.3,0.0",
    "--fractions", "0.10,0.20", "--max-tokens", "16384",
    "--out-name", "cache_pred_OLMoE-1B-7B-0924",
    "--out-dir", "/kaggle/working/out", label="G2c predictor-accuracy sweep")

# The pre-registered sweep across f_crit = k/E, the cache fraction at which a
# per-layer cache first holds one token's whole working set.
run("cache_sim.py", "/kaggle/working/out/traces_OLMoE-1B-7B-0924.npz",
    "--fractions-around-crit", "--both-replays",
    "--out-name", "cache_fcrit_OLMoE-1B-7B-0924",
    "--out-dir", "/kaggle/working/out", label="G2 critical-fraction sweep")
"""

ZIP_CELL = """\
# Pack for download. Uses zipfile rather than the `zip` binary, which is not
# installed on every Kaggle image.
import os, zipfile
out, archive = "/kaggle/working/out", "/kaggle/working/moe_phone_out.zip"
files = sorted(os.listdir(out))
assert files, f"{out} is empty - an earlier cell failed."
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(os.path.join(out, f), f)
        print(f"  {f}  {os.path.getsize(os.path.join(out, f)) / 1e6:.2f} MB")
print(f"\\nwrote {archive} ({os.path.getsize(archive) / 1e6:.2f} MB)")
print("Download it from the Output panel, then unpack into moe-phone/results/<today>/")
"""


def _cell(kind, source, cell_id):
    # nbformat >= 4.5 requires a unique `id` per cell. Without it Kaggle logs
    # "MissingIDFieldWarning: ... this will become a hard error in future
    # nbformat versions". Ids are derived from position so the generator stays
    # deterministic: regenerating an unchanged notebook must produce identical
    # bytes, or --check and the sync test are meaningless.
    lines = source.splitlines(keepends=True)
    cell = {"cell_type": kind, "id": cell_id, "metadata": {}, "source": lines}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build():
    cells = [_cell("markdown", INTRO, "intro")]
    for name in EMBED:
        path = os.path.join(PROJECT, "gates", name)
        with open(path, encoding="utf-8") as f:
            body = f.read()
        cells.append(_cell("code", f"%%writefile {name}\n{body}",
                           "write-" + name.replace(".", "-")))
    for tag, src in (("env", ENV_CELL), ("runner", RUN_HELPER), ("selftest", SELFTEST_CELL),
                     ("g3-model", MODEL_CELL), ("g2-cache", CACHE_CELL), ("pack", ZIP_CELL)):
        cells.append(_cell("code", src, tag))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def serialise(nb):
    # ensure_ascii=False so em dashes survive as UTF-8. The checked-in notebook
    # had literal U+FFFD replacement characters where dashes belonged, because
    # it was written through a non-UTF-8 encoder.
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true",
                   help="exit 1 if the checked-in notebook differs from the generated one")
    a = p.parse_args()
    text = serialise(build())
    if a.check:
        try:
            with open(NOTEBOOK, encoding="utf-8") as f:
                current = f.read()
        except FileNotFoundError:
            print(f"{NOTEBOOK} does not exist; run without --check")
            return 1
        if current != text:
            print(f"OUT OF DATE: {NOTEBOOK} differs from build_notebook.py output.\n"
                  f"Regenerate with: python {os.path.relpath(__file__)}")
            return 1
        print(f"up to date: {NOTEBOOK}")
        return 0
    with open(NOTEBOOK, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {NOTEBOOK} ({len(text)} chars, {len(build()['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
