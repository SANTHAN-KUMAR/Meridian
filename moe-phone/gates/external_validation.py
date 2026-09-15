"""
G-VALID-3 — does the flash-only decode formula predict an INDEPENDENT engine?

colibri (github.com/JustVugg/colibri, Apache-2.0) is an open C engine that
streams MoE experts from disk with a per-layer LRU cache, and publishes community
measurements with disk bandwidth, hit rate and tok/s, and on some rows the
fraction of decode time spent waiting on disk. Those rows are external data
nobody in this project tuned anything against (POSITION.md §0, validation point 1).

The rows below are QUOTED from colibri docs/benchmarks.md at commit a8f2ca6
(2026-09-15); each carries its line number so it can be re-checked. Nothing is
paraphrased: `quote` is the measured cell verbatim, and the parsed fields are
checked against it by a test.

What is computed per row (all in this file, none transcribed from elsewhere):

  bytes_per_token_cold  GLM-5.2: 75 MoE layers x top-8 x expert bytes. colibri
                        states "~11.4 GB of expert reads" per cold token
                        (docs/benchmarks.md:86) and elsewhere 12.7 (README.md:427);
                        the two are unreconciled in the source, so BOTH are run.
  flash_only_tok_s      bandwidth / (bytes_cold * (1 - h))   (ARCHITECTURE.md §1)
  ratio                 flash_only / measured. 1 = the formula is right.
  eta_in_engine         where a disk share of decode time is reported:
                        (bytes_cold * (1-h) / (disk_share / measured_tok_s)) / bandwidth
                        = the fraction of the benchmark bandwidth the engine
                        actually achieved while decoding.
  non_disk_s_per_tok    (1 - disk_share) / measured_tok_s — the term the
                        flash-only formula sets to zero.

Rows with MTP (speculation) active are EXCLUDED from the formula check, because
MTP changes the expert union per forward pass, so bytes/token is not
bytes_cold * (1 - h) (row 141 reports 9.9 GB/token read with MTP on; row 142
reports 600 vs 1,062 experts per token with MTP off vs on). They are listed with
reason "MTP on" rather than dropped silently (CLAUDE.md §6.3).

Run:
  python moe-phone/gates/external_validation.py
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SOURCE = {"repo": "https://github.com/JustVugg/colibri", "commit": "a8f2ca6",
          "file": "docs/benchmarks.md", "retrieved": "2026-09-16"}
GLM_COLD_GB = {"benchmarks.md:86": 11.4, "README.md:427": 12.7}

# line, machine, bandwidth GB/s (as quoted, O_DIRECT preferred), hit, tok/s,
# disk share of decode time or None, mtp_on, verbatim measured cell
ROWS = [
    (116, "Ryzen 9 9950X, Crucial P3 QLC Gen3", 1.51, 0.53, 0.10, 0.66, True,
     "0.10 tok/s · hit 53% · profile 66% disk"),
    (117, "Ryzen 9 9950X, Samsung 9100 PRO Gen5", 8.81, 0.57, 0.28, 0.32, True,
     "**0.28 tok/s** · hit 57% · profile flips: 32% disk / **57% matmul**"),
    (118, "Ryzen AI Max+ 395, Optane 905p", 3.27, 0.57, 0.16, 0.49, True,
     "0.16 tok/s · hit 57% · profile 49% disk / 47% matmul"),
    (120, "Ryzen 7 9800X3D, WSL2, Samsung 9100 PRO", 10.51, 0.54, 0.41, 36.5 / (36.5 + 24.0), False,
     "**0.41 tok/s** · disk-bound (36.5 s disk vs 24.0 s matmul)"),
    (127, "i5-13600K, Samsung 980 PRO", 5.90, 0.547, 0.98, None, False,
     "**0.98 tok/s** (peak 1.07) · hit 54.7% (pin 37.9 + lru 16.7)"),
    (142, "i9-12900K, Samsung 990 Pro, MTP=0", 6.44, 0.549, 0.34, None, False,
     "**0.34 tok/s** · hit **54.9%** · 600 experts/token vs 1,062 with MTP on"),
    (111, "M1 Ultra, Metal fmt=2", 6.89, 0.787, 1.50, 0.57, False,
     "**1.50 tok/s** · hit 78.7% · ... disk wait 57% of decode"),
    (121, "EPYC 7443, TrueNAS VM", 1.0, 0.98, 1.00, None, False,
     "**1.00 tok/s** · **hit 98%** · disk eliminated → **RAM-bandwidth + matmul bound**"),
    (147, "EPYC 9V45 Azure D128", 17.9, 0.97, 2.83, None, True,
     "**2.83 tok/s** rotating median · hit 97%"),
    (148, "EPYC 9V45 Azure E64", 9.0, 0.97, 2.34, None, True,
     "**2.34 tok/s** rotating median · hit 97% · CPU-bound"),
    (141, "i9-12900K, MTP on", 6.44, 0.41, 0.32, 0.48, True,
     "0.31 cold / 0.33 warm · hit 40-42% · ... profile **48% disk / 34% matmul / 12% attn** · ~9.9 GB read/token"),
]


def analyse(cold_gb):
    out = []
    for line, mach, bw, h, tps, disk, mtp, quote in ROWS:
        r = {"line": line, "machine": mach, "bandwidth_GBps": bw, "hit": h,
             "measured_tok_s": tps, "disk_share": disk, "mtp_on": mtp, "quote": quote}
        miss_gb = cold_gb * (1 - h)
        r["flash_only_tok_s"] = bw / miss_gb
        r["ratio_pred_over_measured"] = r["flash_only_tok_s"] / tps
        if disk is not None:
            t_disk = disk / tps
            r["eta_in_engine"] = (miss_gb / t_disk) / bw
            r["non_disk_s_per_tok"] = (1 - disk) / tps
        r["used_in_check"] = not mtp
        r["excluded_reason"] = "MTP on: bytes/token is not cold*(1-h)" if mtp else None
        out.append(r)
    used = [r for r in out if r["used_in_check"]]
    disk_bound = [r for r in used if r["hit"] < 0.9]
    summary = {
        "n_rows": len(out), "n_used": len(used),
        "n_excluded_mtp": sum(1 for r in out if r["mtp_on"]),
        "disk_bound_ratio_median": statistics.median(r["ratio_pred_over_measured"] for r in disk_bound),
        "disk_bound_ratio_range": [min(r["ratio_pred_over_measured"] for r in disk_bound),
                                   max(r["ratio_pred_over_measured"] for r in disk_bound)],
        "high_hit_ratio": [r["ratio_pred_over_measured"] for r in used if r["hit"] >= 0.9],
        "eta_in_engine_all_rows_with_disk_share": [round(r["eta_in_engine"], 3) for r in out
                                                   if r.get("eta_in_engine") is not None],
        "non_disk_share_all_rows_with_disk_share": [round(1 - r["disk_share"], 3) for r in out
                                                    if r["disk_share"] is not None],
    }
    return out, summary


def main():
    res = {"source": SOURCE, "by_cold_bytes": {}}
    for src, gb in GLM_COLD_GB.items():
        rows, summ = analyse(gb)
        res["by_cold_bytes"][src] = {"cold_GB_per_token": gb, "rows": rows, "summary": summ}
        print(f"\ncold bytes/token = {gb} GB ({src})")
        print(f"{'line':>5} {'machine':<40} {'pred':>6} {'meas':>6} {'ratio':>6} {'eta':>6}  used")
        for r in rows:
            eta = r.get("eta_in_engine")
            print(f"{r['line']:>5} {r['machine'][:40]:<40} {r['flash_only_tok_s']:6.2f} "
                  f"{r['measured_tok_s']:6.2f} {r['ratio_pred_over_measured']:6.2f} "
                  f"{(eta if eta is not None else float('nan')):6.2f}  "
                  f"{'yes' if r['used_in_check'] else r['excluded_reason']}")
        s = summ
        print(f"disk-bound rows used: ratio median {s['disk_bound_ratio_median']:.2f}, "
              f"range {s['disk_bound_ratio_range'][0]:.2f}-{s['disk_bound_ratio_range'][1]:.2f}; "
              f"high-hit rows {['%.1f' % x for x in s['high_hit_ratio']]}")
    from _paths import write_path
    path = write_path("external_validation_colibri.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
