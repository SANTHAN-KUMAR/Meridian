"""
M4 plumbing acceptance for the GPU expert tier (patch 0019; design §12), LAPTOP, CPU stand-in backend: the engine's
generated text must be byte-identical with the tier OFF and ON, and the device must actually have computed experts.
Rows: off, t500 (500 MiB tier, promote after 2 hits), t800h1 (800 MiB, promote after 1 hit); OLMoE, 96 tokens, the
stack configuration. Also reported: the tier's own counters (promotions, evictions, experts per dispatch), which
show the policy problems recorded in design §13 (churn; GPU share above its rate). No speed claim: the stand-in
runs on the same CPU as the compute threads.
Run:
  python moe-phone/gates/gtier_m4.py results/2026-09-18/gtier_m4_laptop --out results/2026-09-18/gtier_m4.json
"""
import argparse, hashlib, json, os, re
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--out", required=True); a = ap.parse_args()
rows = {}
for tag in ("off", "t500", "t800h1"):
    t = open(os.path.join(a.root, tag + ".out")).read()
    text = t[:t.index("\ngeneration:")] if "\ngeneration:" in t else t
    err = open(os.path.join(a.root, tag + ".err")).read()
    m = re.search(r"gpu-tier: backend \S+ slots (\d+), promotions (\d+), evictions (\d+), flushes (\d+), gpu-hits (\d+), "
                  r"dispatches (\d+), experts on device (\d+) \(([0-9.]+)/dispatch\), wait ([0-9.]+) ms, failures (\d+)", err)
    r = dict(text_md5=hashlib.md5(text.encode()).hexdigest(), fatal="FATAL" in err)
    if m:
        k = ("slots", "promotions", "evictions", "flushes", "gpu_hits", "dispatches", "experts_on_device",
             "experts_per_dispatch", "wait_ms", "failures")
        r.update({kk: float(v) if "." in v else int(v) for kk, v in zip(k, m.groups())})
    rows[tag] = r
out = dict(source=os.path.abspath(a.root), rows=rows,
           text_identical=len({r["text_md5"] for r in rows.values()}) == 1,
           any_fatal=any(r["fatal"] for r in rows.values()),
           experts_on_device_total=sum(r.get("experts_on_device", 0) for r in rows.values()))
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "source"}, indent=1))
