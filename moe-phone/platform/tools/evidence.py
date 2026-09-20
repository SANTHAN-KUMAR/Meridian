"""
EVIDENCE — every number the platform documents assert, read from an artifact.

`CLAUDE.md` §7.1: no number is written into a document by hand. The platform
design documents under `moe-phone/platform/` are design documents, but they
lean on measured results from the moe-phone research phase, and a design
argument built on a mis-remembered number is exactly the §0 failure mode.

So each number in those documents is written as

    6.67 tok/s [E:qwen3_h2h_ours]

and this script does two things:

  --emit    regenerate platform/EVIDENCE.md, one row per id, from the artifacts
  (default) verify every [E:id] marker in platform/**.md against the artifacts,
            non-zero exit on drift or on an unknown id

Sources, in order of preference:
  1. moe-phone/CLAIMS.md          — already generated and checked by gates/claims_check.py
  2. results/**/*.json            — read directly, with the reduction named here
  3. a pre-registration document  — parsed by a regex that is part of this file

A number that cannot be sourced from one of those does not go in a document.
This script never reads the platform documents to decide a value; it only ever
checks them against the artifacts.
"""
import argparse
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLATFORM = os.path.dirname(HERE)
PROJECT = os.path.dirname(PLATFORM)
RESULTS = os.path.join(PROJECT, "results")


def _jload(rel):
    with open(os.path.join(RESULTS, rel), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _claims():
    """Parse the generated claims map into {id: (value, artifact, text)}."""
    path = os.path.join(PROJECT, "CLAIMS.md")
    out = {}
    row = re.compile(r"^\|\s*`([a-z0-9_]+)`\s*\|(.*?)\|\s*`([^`]+)`\s*\|\s*([-\d.]+)\s*\|\s*(\w+)\s*\|\s*$")
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            m = row.match(line)
            if m:
                out[m.group(1)] = (float(m.group(4)), m.group(3), m.group(2).strip(), m.group(5))
    if not out:
        raise SystemExit("CLAIMS.md parsed to zero rows — the format changed; fix this parser")
    return out


CLAIMS = None


def claim(cid):
    """A value from the claims map, which gates/claims_check.py verifies against its own artifact."""
    if cid not in CLAIMS:
        raise KeyError("claim id not in CLAIMS.md: " + cid)
    value, artifact, _text, status = CLAIMS[cid]
    if status != "ok":
        raise SystemExit("claim %s is not ok in CLAIMS.md (status=%s)" % (cid, status))
    return value, "CLAIMS.md:" + cid + " -> " + artifact


def median_rows(rows, key, where=None):
    vals = [r[key] for r in rows if (where is None or where(r)) and r.get(key) is not None]
    if not vals:
        raise SystemExit("no rows matched for key " + key)
    return statistics.median(vals)


# ── the table ────────────────────────────────────────────────────────────────
# Each entry: id -> (callable returning (value, source), unit, one-line meaning)
# The callable does the reduction; the reduction IS the definition of the number.

def _e1(field, arm):
    d = _jload("2026-09-19/e1_summary.json")
    return d[field][arm], "2026-09-19/e1_summary.json:" + field + "." + arm


def _h2h(cell, key):
    d = _jload("2026-09-19/h2h_ab_summary.json")
    rows = [r for r in d["rows"] if r.get("kept") and r.get("cell") == cell]
    if not rows:
        raise SystemExit("h2h: no kept rows for cell " + cell)
    return median_rows(rows, key), "2026-09-19/h2h_ab_summary.json:median(kept,%s).%s" % (cell, key)


def _h2h_ratio():
    a, _ = _h2h("stack", "decode_tok_s")
    b, _ = _h2h("base", "decode_tok_s")
    return a / b, "2026-09-19/h2h_ab_summary.json:median(stack)/median(base) decode_tok_s"


def _gptoss(key):
    d = _jload("2026-09-17/bmoe_gptoss20b.json")
    return median_rows(d["runs"], key), "2026-09-17/bmoe_gptoss20b.json:median(runs).%s" % key


def _appengine(prefix, key):
    d = _jload("2026-09-18/app_engine.json")
    rows = [r for r in d["rows"] if r["file"].startswith(prefix) and not r.get("failed")]
    return median_rows(rows, key), "2026-09-18/app_engine.json:median(%s*).%s" % (prefix, key)


def _g0(path):
    d = _jload("2026-09-14/g0_device.json")
    cur = d
    for p in path.split("."):
        cur = cur[p]
    return cur, "2026-09-14/g0_device.json:" + path


def _g1(path):
    d = _jload("2026-09-14/g1_storage.json")
    cur = d
    for p in path.split("."):
        cur = cur[p]
    return cur, "2026-09-14/g1_storage.json:" + path


def _e6(arm, key):
    d = _jload("2026-09-19/e6_kl_summary.json")
    return d["arms"][arm][key], "2026-09-19/e6_kl_summary.json:arms.%s.%s" % (arm, key)


def _e7(key):
    d = _jload("2026-09-19/e7_summary.json")
    if key in d:
        return d[key], "2026-09-19/e7_summary.json:" + key
    return d["ms_per_token"][key], "2026-09-19/e7_summary.json:ms_per_token." + key


_NORD_RUN = "2026-09-19/nord/bmoe_sustain_20260919_2223"
_NORD_LINE = re.compile(
    r"generation: \d+ tokens, [\d.]+ s/token \(([\d.]+) tok/s\).*?"
    r"compute ([\d.]+) \+ cache mgmt ([\d.]+) .*?"
    r"stall ([\d.]+) s/token", re.S)


def _nord(which):
    """Steady state = median over the second half of the sustained runs (bmoe_sustain.sh's own rule)."""
    path = os.path.join(RESULTS, _NORD_RUN, "log.txt")
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    rows = _NORD_LINE.findall(text)
    if len(rows) < 4:
        raise SystemExit("nord log parsed to %d rows" % len(rows))
    half = rows[len(rows) // 2:]
    idx = {"tok_s": 0, "compute_s": 1, "mgmt_s": 2, "stall_s": 3}[which]
    vals = [float(r[idx]) for r in half]
    v = statistics.median(vals)
    if which != "tok_s":
        v *= 1000.0
    return v, "%s/log.txt:median(second half).%s" % (_NORD_RUN, which)


_NORD_PRED = re.compile(r"\*\*~([\d.]+) tok/s \(range ([\d.]+)-([\d.]+)\)")


def _nord_pred(which):
    path = os.path.join(RESULTS, "2026-09-19/nord/PREDICTION.md")
    with open(path, "r", encoding="utf-8") as fh:
        m = _NORD_PRED.search(fh.read())
    if not m:
        raise SystemExit("nord PREDICTION.md: the pre-registered range line did not parse")
    return float(m.group({"point": 1, "lo": 2, "hi": 3}[which])), \
        "2026-09-19/nord/PREDICTION.md:pre-registered." + which


TABLE = {
    # ── the device the research was done on (OnePlus 15R, SM8845, unrooted app context)
    "dev_ram_total_gib": (lambda: _g0("memory.MemTotal_GiB"), "GiB",
                          "total RAM the kernel reports on the primary device"),
    "dev_ram_avail_kb": (lambda: _g0("memory.MemAvailable_kB"), "kB",
                         "MemAvailable at probe time — what an app could hope to get"),
    "dev_budget_max_mib": (lambda: claim("budget_reach_max"), "MiB",
                           "largest expert-cache budget the phone actually GRANTED on any of 238 logged runs"),
    "dev_flash_bulk_mbps": (lambda: _g1("bulk_rand_best.MBps_median"), "MB/s",
                            "random 1 MiB O_DIRECT reads, 8 threads — the rate an expert read gets"),
    "dev_flash_4k_mbps": (lambda: _g1("small_4k_rand_best.MBps_median"), "MB/s",
                          "random 4 KiB reads, 8 threads — the rate a naive row-granular read gets"),
    "dev_flash_ratio": (lambda: _g1("bulk_to_small_ratio"), "x",
                        "bulk/small read-rate ratio: why request GRANULARITY, not bytes, bounds streaming"),
    "dev_dram_gbps": (lambda: claim("dram_app_gbps"), "GB/s",
                      "DRAM read bandwidth available to an unprivileged app"),

    # ── a small model held fully resident, measured IN AN APP PROCESS on the same phone
    "olmoe_cpu_decode": (lambda: claim("inapp_olmoe_cpu_tok_s"), "tok/s", "OLMoE-1B-7B Q4_0 resident, CPU decode"),
    "olmoe_gpu_decode": (lambda: claim("inapp_olmoe_gpu_tok_s"), "tok/s", "the same model, Adreno OpenCL decode"),
    "olmoe_npu_decode": (lambda: claim("inapp_olmoe_htp_tok_s"), "tok/s", "the same model, Hexagon NPU decode"),
    "olmoe_cpu_prefill": (lambda: _appengine("olmoe_cpu", "prefill_tok_s"), "tok/s", "the same model, CPU prefill"),
    "olmoe_gpu_prefill": (lambda: claim("inapp_olmoe_gpu_prefill"), "tok/s", "the same model, Adreno prefill"),
    "olmoe_npu_prefill": (lambda: claim("inapp_olmoe_htp_prefill"), "tok/s", "the same model, Hexagon NPU prefill"),

    # ── the streamed 30B MoE on the same phone
    "qwen3_h2h_ours": (lambda: _h2h("stack", "decode_tok_s"), "tok/s",
                       "our engine, Qwen3-30B-A3B Q4_0 streamed, awake and unplugged, same-session A/B"),
    "qwen3_h2h_base": (lambda: _h2h("base", "decode_tok_s"), "tok/s",
                       "the published baseline engine in the same session"),
    "qwen3_h2h_ratio": (_h2h_ratio, "x", "the controlled speed ratio over the published baseline"),
    "qwen3_h2h_stall_ms": (lambda: _h2h("stack", "stall_ms"), "ms/token",
                           "wall time lost to flash reads that overlap could not hide"),
    "qwen3_h2h_mgmt_ms": (lambda: _h2h("stack", "mgmt_ms"), "ms/token", "expert-cache management time"),
    "qwen3_h2h_hit": (lambda: _h2h("stack", "hit_pct"), "%", "expert-cache hit rate at that budget"),
    "qwen3_e1_generic_ms": (lambda: _e1("C_capped_ms", "G"), "ms/token",
                            "pure compute floor, generic kernels, replayed routing (no I/O), sustained clock"),
    "qwen3_e1_repack_ms": (lambda: _e1("C_capped_ms", "R"), "ms/token",
                           "the same floor with repacked i8mm kernels — the best lossless compute measured"),
    "qwen3_e1_long_ms": (lambda: _e1("long_ms", "R"), "ms/token",
                         "the same, at ~3,000 tokens of context: compute ALONE, before any flash cost"),
    "qwen3_e1_close_ms": (lambda: (_jload("2026-09-19/e1_summary.json")["closing_ms_per_token"],
                                   "2026-09-19/e1_summary.json:closing_ms_per_token"), "ms/token",
                          "best achievable token time on this device: compute floor + measured stall + mgmt"),
    "qwen3_e1_close_tok_s": (lambda: (_jload("2026-09-19/e1_summary.json")["closing_tok_s"],
                                      "2026-09-19/e1_summary.json:closing_tok_s"), "tok/s",
                             "the same, as a rate — the ceiling of the CPU-centric streamed architecture here"),
    "qwen3_cache_hit": (lambda: claim("cache5000_hit"), "%", "hit rate at the 5 GB cache the phone would grant"),
    "qwen3_read_mib": (lambda: claim("cache5000_read"), "MiB/token", "flash bytes read per decoded token at that hit rate"),
    "qwen3_ceiling_cpu": (lambda: claim("device_ceiling_qwen3_cpu"), "tok/s",
                          "hard ceiling from the CPU's measured weight-byte throughput"),

    # ── a second streamed model, same phone
    "gptoss_decode": (lambda: _gptoss("decode_tok_s"), "tok/s", "gpt-oss-20b MXFP4 streamed, decode"),
    "gptoss_hit": (lambda: _gptoss("cache_hit_pct"), "%", "its hit rate at the ~3.4 GB budget the phone granted"),
    "gptoss_read_mib": (lambda: _gptoss("read_MiB_per_token"), "MiB/token", "its flash traffic per token"),
    "gptoss_rereads": (lambda: _gptoss("rereads_per_token"), "/token",
                       "experts re-read per token — bytes the cache had already paid for once"),
    "gptoss_prefill_short": (lambda: _gptoss("prefill_tok_s"), "tok/s",
                             "prefill at a 26-token prompt: a cold-start artifact, NOT an ingestion rate"),
    "gptoss_prompt_tokens": (lambda: _gptoss("n_prompt"), "tokens", "the prompt length every prefill figure was taken at"),

    # ── a second device (OnePlus Nord, SD765G) — the cross-device transfer test
    "nord_decode": (lambda: _nord("tok_s"), "tok/s", "the same 30B model and engine, sustained steady state"),
    "nord_compute_ms": (lambda: _nord("compute_s"), "ms/token", "its compute term"),
    "nord_stall_ms": (lambda: _nord("stall_s"), "ms/token", "its un-hidden flash stall"),
    "nord_pred_point": (lambda: _nord_pred("point"), "tok/s", "the PRE-REGISTERED prediction from spec-sheet scaling"),
    "nord_pred_lo": (lambda: _nord_pred("lo"), "tok/s", "the bottom of its pre-registered range"),

    # ── the accelerators, measured rather than assumed
    "e7_gpu_l8_ms": (lambda: _e7("sustained_tail_median_ms"), "ms/token",
                     "8 Qwen3 layers resident on the Adreno, sustained"),
    "e7_cpu_l8_ms": (lambda: _e7("cpu_L8"), "ms/token", "the same 8 layers on the CPU"),

    # ── the quality frontier for training-free routing shortcuts
    "e6_floor_kl": (lambda: _e6("FLOOR", "kl_mean"), "nats",
                    "mean KL that 4-bit quantisation itself costs against an 8-bit reference"),
    "e6_floor_p99": (lambda: _e6("FLOOR", "kl_p99"), "nats", "its 99th percentile"),
    "e6_best_arm_p99": (lambda: _e6("DC05", "kl_p99"), "nats",
                        "the best lossy routing arm's p99 — the one that came closest to negligible and still failed"),
    "e6_topk7_kl": (lambda: _e6("T7", "kl_mean"), "nats", "dropping one expert of eight: mean KL"),

    # ── offline cache theory, on real routing traces
    "sim_lru_10pct": (lambda: claim("lru_10pct"), "", "per-layer LRU hit rate at a 10% expert cache"),
    "sim_belady_10pct": (lambda: claim("belady_10pct"), "", "the offline optimum at the same cache"),
    "sim_global_pool": (lambda: claim("scope_global_lru_is_zero"), "",
                        "one shared pool across layers at the same total bytes: no hits at all"),

    # ── engine-level facts that generalise
    "overhead_ratio": (lambda: claim("overhead_ratio"), "x",
                       "the streaming engine running the SAME arithmetic slower than plain llama.cpp on a cached model"),
    "cliff_unpinned": (lambda: claim("cliff_unpinned_t4"), "tok/s", "4 unpinned threads on a resident model"),
    "cliff_pinned": (lambda: claim("cliff_fix_pinned_t4"), "tok/s",
                     "the same 4 threads pinned: thread PLACEMENT, not arithmetic, was the binding constraint"),
}


def build():
    out = {}
    for eid, (fn, unit, meaning) in sorted(TABLE.items()):
        value, source = fn()
        out[eid] = {"value": float(value), "unit": unit, "meaning": meaning, "source": source}
    return out


def emit(table):
    lines = [
        "# platform — evidence map",
        "",
        "**Generated by `platform/tools/evidence.py --emit`. Do not edit by hand.**",
        "",
        "Every number asserted in the `platform/` documents, the artifact it is read from, and the",
        "reduction that defines it. Documents cite a row as `` [E:<id>] `` immediately after the number;",
        "`python moe-phone/platform/tools/evidence.py` fails if any document drifts from this table.",
        "",
        "These are measurements from the moe-phone research phase (2026-09-14 to 2026-09-19), on the",
        "devices named in each row's source. They are **inputs to a design argument**, not claims about",
        "any other device. `CLAUDE.md` §8: a number measured on one phone is not a property of phones.",
        "",
        "| id | value | unit | what it is | source |",
        "|---|---|---|---|---|",
    ]
    for eid, row in table.items():
        v = row["value"]
        s = ("%.4f" % v).rstrip("0").rstrip(".") if abs(v) < 10000 else "%.0f" % v
        lines.append("| `%s` | %s | %s | %s | `%s` |" % (eid, s, row["unit"] or "—", row["meaning"], row["source"]))
    lines.append("")
    return "\n".join(lines)


MARKER = re.compile(r"\[E:([a-z0-9_]+)\]")
NUM = re.compile(r"(-?\d+(?:\.\d+)?)")
CODE = re.compile(r"```.*?```|``.*?``|`[^`\n]*`", re.S)


def blank_code(text):
    """Blank out fenced blocks and inline code spans, preserving offsets.

    A document that SHOWS the marker syntax (`` value [E:id] ``) must not be checked as if it
    asserted a number. Markers that are meant to be checked are therefore written in plain prose,
    never inside backticks — which is also where a reader wants them.
    """
    return CODE.sub(lambda m: " " * len(m.group(0)), text)


def check(table):
    """Every [E:id] in platform/**.md must be preceded by a number that agrees with the table."""
    bad, seen = [], set()
    for root, _dirs, files in os.walk(PLATFORM):
        for name in sorted(files):
            if not name.endswith(".md") or name == "EVIDENCE.md":
                continue
            path = os.path.join(root, name)
            with open(path, "r", encoding="utf-8") as fh:
                text = blank_code(fh.read())
            for m in MARKER.finditer(text):
                eid = m.group(1)
                rel = os.path.relpath(path, PROJECT)
                if eid not in table:
                    bad.append("%s: unknown evidence id [E:%s]" % (rel, eid))
                    continue
                seen.add(eid)
                nums = NUM.findall(text[max(0, m.start() - 40):m.start()])
                if not nums:
                    bad.append("%s: [E:%s] has no number in front of it" % (rel, eid))
                    continue
                doc = float(nums[-1])
                ref = table[eid]["value"]
                dec = len(nums[-1].split(".")[1]) if "." in nums[-1] else 0
                tol = max(0.5 * 10 ** (-dec), 0.005 * abs(ref))
                if abs(doc - ref) > tol:
                    bad.append("%s: [E:%s] document says %s, artifact says %.6g (tol %.6g)"
                               % (rel, eid, nums[-1], ref, tol))
    return bad, seen


def main():
    global CLAIMS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit", action="store_true", help="regenerate platform/EVIDENCE.md")
    args = ap.parse_args()
    CLAIMS = _claims()
    table = build()
    if args.emit:
        with open(os.path.join(PLATFORM, "EVIDENCE.md"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(emit(table))
        print("wrote platform/EVIDENCE.md (%d rows)" % len(table))
        return 0
    bad, seen = check(table)
    for b in bad:
        print("DRIFT: " + b)
    unused = sorted(set(table) - seen)
    print("checked %d markers over %d evidence rows; %d rows unused: %s"
          % (len(seen), len(table), len(unused), ", ".join(unused) or "none"))
    if bad:
        print("\n%d problem(s). Fix the DOCUMENT, or re-derive the artifact — never the other way round." % len(bad))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
