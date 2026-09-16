"""
G-REPRO — re-run every committed gate from committed inputs and compare every
numeric leaf against the committed artifact.

CLAUDE.md §7.1 makes every reported number a function of a named script and an
artifact; this checks the other half, that the artifact is still a function of
the script. Outputs go to --out-dir (never results/: `_paths.write_path` is
patched for each child process, CLAUDE.md §6.6), and the check fails if a dated
results directory appears during the run.

Keys present in only one side are reported, not failed: a script that has since
gained output fields (e.g. byte_budget's compute columns) still reproduces if
every shared number matches.

Run:
  python moe-phone/gates/repro_check.py --out-dir /tmp/moe_repro [--jobs 11]
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _paths import RESULTS_ROOT  # noqa: E402

REF = os.path.join(RESULTS_ROOT, "2026-09-14")
T = os.path.join(REF, "traces_OLMoE-1B-7B-0924.npz")
MEMPROBE = ["memprobe_15r.csv", "memprobe_15r_r2.csv", "memprobe_15r_r3.csv",
            "memprobe_15r_file_r1.csv", "memprobe_15r_file_r2.csv", "memprobe_15r_file_r3.csv"]
BB = ["--ram-gb", "4.85", "--flash-gbps", "2.806", "--dram-gbps", "45", "--assumed", "dram",
      "--threshold-tps", "5"]

JOBS = [
    ("cache_sim.py", [T, "--controls", "--both-scopes", "--both-replays", "--lookahead", "0,1,2,4,8",
                      "--fractions", "0.05,0.10,0.125,0.20,0.30", "--out-name", "cache_OLMoE-1B-7B-0924"],
     "cache_OLMoE-1B-7B-0924.json"),
    ("cache_sim.py", [T, "--fractions-around-crit", "--both-replays",
                      "--out-name", "cache_fcrit_OLMoE-1B-7B-0924"], "cache_fcrit_OLMoE-1B-7B-0924.json"),
    ("cache_sim.py", [T, "--lookahead", "4", "--pred-accuracy", "1.0,0.9,0.8,0.7,0.5,0.3,0.0",
                      "--fractions", "0.10,0.20", "--max-tokens", "16384",
                      "--out-name", "cache_pred_OLMoE-1B-7B-0924"], "cache_pred_OLMoE-1B-7B-0924.json"),
    ("scope_compare.py", [T, "--max-tokens", "8000", "--fractions", "0.10,0.20"],
     "scope_compare_OLMoE-1B-7B-0924.json"),
    ("expert_policy.py", [T, "--fractions", "0.05,0.10,0.20,0.30"], "expert_policy_OLMoE-1B-7B-0924.json"),
    ("engine_sim.py", [T, "--bulk-gbps", "2.806", "--expert-mb", "3.54"], "engine_sim_OLMoE-1B-7B-0924.json"),
    ("predictor.py", [T, "--max-tokens", "16384"], "predictor_OLMoE-1B-7B-0924.json"),
    ("byte_budget.py", BB + ["--flash-gbps-scattered", "0.315", "--tag", "measured"],
     "byte_budget_measured.json"),
    ("byte_budget.py", BB + ["--flash-gbps-scattered", "2.806", "--tag", "measured_bundled_ideal"],
     "byte_budget_measured_bundled_ideal.json"),
    ("g1_analyze.py", [os.path.join(REF, "ufs_15r.csv"), "--memprobe"] +
     [os.path.join(REF, m) for m in MEMPROBE], "g1_storage.json"),
]
# engine_target.json of 2026-09-14 predates the 2026-09-16 fix (resident / clamped rows and
# the DRAM term), so its reproduction is against the 2026-09-16 artifact instead.
D16 = os.path.join(RESULTS_ROOT, "2026-09-16")
Q4 = os.path.join(D16, "traces_OLMoE-1B-7B-0924-q4_0-llamacpp.npz")
GR = os.path.join(D16, "traces_granite-3.1-1b-a400m-q4_0-llamacpp.npz")
MB = ["olmoe-1b-7b-0924-q4_0.gguf=3928036960",
      "granite-3.1-1b-a400m-instruct-Q4_0.gguf=771466560"]
# The measured forward-pass cost the 2026-09-16 artifacts were produced with. It is not a
# tuned constant: it is active_bytes / effective_nonflash_GBps, both measured, and the
# claims map asserts the artifacts carry this exact setting.
FWD = "10,20,30.07,40,80"

JOBS_NEW = [
    ("engine_target.py", ["--bulk-gbps", "2.806", "--dram-gbps", "59.74", "--nonflash-gbps",
                          "23.196", "--ram-gb", "4.85", "--target-tps", "5"],
     "engine_target.json", "2026-09-16"),
    ("external_validation.py", [], "external_validation_colibri.json", "2026-09-16"),
    ("dram_analyze.py", [f"app={D16}/dram_15r_app.csv", f"shell={D16}/dram_15r_shell.csv"],
     "dram_15r.json", "2026-09-16"),
    ("decode_analyze.py", [f"{D16}/decode_15r", "--model-bytes"] + MB,
     "decode_15r.json", "2026-09-16"),
    ("decode_analyze.py", [f"{D16}/decode_15r_cpu", "--out-name", "decode_15r_cpu.json",
                           "--model-bytes"] + MB, "decode_15r_cpu.json", "2026-09-16"),
    ("decode_analyze.py", [f"{D16}/decode_15r_threads", "--out-name", "decode_15r_threads.json",
                           "--model-bytes"] + MB, "decode_15r_threads.json", "2026-09-16"),
    ("s11_nonflash.py", [], "s11_nonflash_15r.json", "2026-09-16"),
    ("cache_sim.py", [Q4, "--fractions-around-crit", "--out-name",
                      "cache_fcrit_OLMoE-1B-7B-0924-q4_0-llamacpp"],
     "cache_fcrit_OLMoE-1B-7B-0924-q4_0-llamacpp.json", "2026-09-16"),
    ("cache_sim.py", [GR, "--fractions-around-crit", "--out-name",
                      "cache_fcrit_granite-3.1-1b-a400m"],
     "cache_fcrit_granite-3.1-1b-a400m.json", "2026-09-16"),
    ("llamacpp_traces.py", ["compare", os.path.join(REF, "traces_OLMoE-1B-7B-0924.npz"), Q4,
                            "--out", os.path.join("{OUT}", "traces_confound_olmoe_q4_vs_fp16.json")],
     "traces_confound_olmoe_q4_vs_fp16.json", "2026-09-16"),
    ("s9_score.py", ["--prereg", f"{D16}/s9_prereg_granite-3.1-1b-a400m.json",
                     "--target-curve", f"{D16}/cache_fcrit_granite-3.1-1b-a400m.json",
                     "--confound-curve", f"{D16}/cache_fcrit_OLMoE-1B-7B-0924-q4_0-llamacpp.json"],
     "s9_result_granite-3.1-1b-a400m.json", "2026-09-16"),
    ("engine_sim.py", [os.path.join(REF, "traces_OLMoE-1B-7B-0924.npz"), "--bulk-gbps", "2.806",
                       "--expert-mb", "3.54", "--fwd-ms", FWD, "--verify-beta", "0,0.25,1"],
     "engine_sim_OLMoE-1B-7B-0924.json", "2026-09-16"),
    ("prefetch_sim.py", [os.path.join(REF, "traces_OLMoE-1B-7B-0924.npz"), "--g3",
                         os.path.join(REF, "g3_OLMoE-1B-7B-0924.json"), "--bulk-gbps", "2.806",
                         "--expert-mb", "3.54", "--fwd-ms", "0,30.07,80,160"],
     "prefetch_sim_OLMoE-1B-7B-0924.json", "2026-09-16"),
]

SHIM = ("import os,sys,runpy; sys.path.insert(0,{g!r}); import _paths; "
        "_paths.write_path=lambda f: os.path.join({o!r}, f); "
        "sys.argv=[{s!r}]+{a!r}; runpy.run_path({s!r}, run_name='__main__')")
SKIP = {"source", "path", "runtime_s", "npz", "out_dir", "out_name", "byte_budget_json",
        "cache_json", "source_csv"}


def leaves(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if k not in SKIP:
                yield from leaves(v, f"{pre}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{pre}[{i}]")
    else:
        yield pre, x


def compare(ref_path, new_path):
    la = dict(leaves(json.load(open(ref_path, encoding="utf-8"))))
    lb = dict(leaves(json.load(open(new_path, encoding="utf-8"))))
    n = nd = 0
    worst = 0.0
    ex = []
    for k in set(la) & set(lb):
        va, vb = la[k], lb[k]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) \
                and not isinstance(va, bool) and not isinstance(vb, bool):
            n += 1
            d = abs(va - vb) / max(1e-12, abs(va))
            worst = max(worst, d)
            if d > 1e-9:
                nd += 1
                ex.append((k, va, vb))
        elif va != vb and not (isinstance(va, str) and ("/" in va or "\\" in va)):
            nd += 1
            ex.append((k, va, vb))
    return {"numeric_leaves": n, "n_differ": nd, "max_rel_diff": worst, "examples": ex[:5],
            "n_only_in_committed": len(set(la) - set(lb)), "n_only_in_rerun": len(set(lb) - set(la))}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--jobs", type=int, default=len(JOBS) + len(JOBS_NEW))
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    before = set(os.listdir(RESULTS_ROOT))
    todo = [(s, ar, art, "2026-09-14") for s, ar, art in JOBS] + JOBS_NEW

    def run(job):
        script, args, art, ref_date = job
        od = os.path.join(a.out_dir, ref_date)
        os.makedirs(od, exist_ok=True)
        args = [x.replace("{OUT}", od) if isinstance(x, str) else x for x in args]
        code = SHIM.format(g=HERE, o=od, s=os.path.join(HERE, script), a=args)
        with open(os.path.join(od, art + ".log"), "w") as log:
            rc = subprocess.call([sys.executable, "-c", code], stdout=log, stderr=subprocess.STDOUT,
                                 cwd=HERE)
        return job, rc, od

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(run, todo))
    report, bad = {}, 0
    for (script, args, art, ref_date), rc, od in res:
        e = {"rc": rc, "reference": f"{ref_date}/{art}"}
        if rc == 0:
            e.update(compare(os.path.join(RESULTS_ROOT, ref_date, art), os.path.join(od, art)))
        ok = rc == 0 and e.get("n_differ") == 0
        bad += not ok
        report[f"{ref_date}/{art}"] = e
        print(f"[{'ok' if ok else 'FAIL':4}] {ref_date}/{art}: "
              f"{e.get('numeric_leaves', '-')} numbers, {e.get('n_differ', '-')} differ")
    stray = set(os.listdir(RESULTS_ROOT)) - before
    if stray:
        bad += 1
        print("FAIL: new results directories appeared during the run:", sorted(stray))
    report["_meta"] = {"when": datetime.datetime.now().isoformat(), "seconds": time.time() - t0,
                       "stray_results_dirs": sorted(stray)}
    with open(os.path.join(a.out_dir, "REPRO_REPORT.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"{len(todo) - bad + (1 if stray else 0)}/{len(todo)} artifacts reproduce")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
