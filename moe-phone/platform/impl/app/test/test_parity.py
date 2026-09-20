#!/usr/bin/env python3
"""Java planner (host JVM, org.json from layoutlib.jar) vs Python planner on real GGUFs. Skips models that are absent."""
import os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); APP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(APP, ".."))
from meridian import planner
LAYOUTLIB = "/opt/android-studio/plugins/design-tools/lib/layoutlib.jar"
MODELS = ["/run/media/santhankumar/New Volume/moe-work/models/olmoe-1b-7b-0924-q4_0.gguf",
          "/run/media/santhankumar/New Volume/models/gpt-oss-20b/openai_gpt-oss-20b-MXFP4.gguf",
          "/run/media/santhankumar/New Volume/moe-work/models/granite-3.1-3b-a800m-instruct-Q4_0.gguf"]
def main():
    if not os.path.exists(LAYOUTLIB): print("SKIP: layoutlib.jar (org.json) not found"); return 0
    with tempfile.TemporaryDirectory() as d:
        srcs = [os.path.join(APP, "src/com/meridian", f) for f in ("Gguf.java", "Planner.java")] + [os.path.join(HERE, "ParityMain.java")]
        subprocess.run(["javac", "-cp", LAYOUTLIB, "-d", d] + srcs, check=True, capture_output=True)
        fails = 0
        for m in MODELS:
            if not os.path.exists(m): print("SKIP", os.path.basename(m)); continue
            out = subprocess.run(["java", "-cp", d + ":" + LAYOUTLIB, "com.meridian.app.ParityMain", m], capture_output=True, text=True, check=True).stdout
            j = dict(l.split("=") for l in out.strip().splitlines())
            c = planner.derive_card(m)
            want = {"total_bytes": c.total_bytes, "resident_bytes": c.resident_bytes, "expert_bytes": c.expert_bytes, "active_bytes_per_token": c.active_bytes_per_token,
                    "kv_f16": c.kv_bytes_per_token["f16"], "slice_sum_max": sum(hi for _, hi in c.expert_slice_bytes.values())}
            bad = {k: (j.get(k), v) for k, v in want.items() if str(v) != j.get(k)}
            print(("FAIL " if bad else "PASS ") + os.path.basename(m), bad if bad else ""); fails += bool(bad)
        return 1 if fails else 0
sys.exit(main())
