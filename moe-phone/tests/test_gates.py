"""
moe-phone gates whose expected values are fixed by mathematics, not by results
(CLAUDE.md §9.1). Tests never write to results/ — they call pure functions.
"""
import math
import os
import random
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gates"))

import cache_sim  # noqa: E402
import g1_analyze  # noqa: E402


# ---- cache simulation ----------------------------------------------------

def test_uniform_iid_lru_hit_rate_equals_cache_fraction():
    # Under the independent reference model with uniform popularity, every
    # key is symmetric, so the steady-state hit rate of LRU is exactly C/N.
    rng = np.random.default_rng(0)
    N, C, R = 100, 20, 200_000
    seq = rng.integers(0, N, size=R)
    hit = cache_sim.lru_hits(seq, C) / R
    se = math.sqrt(0.2 * 0.8 / R)
    assert abs(hit - C / N) < 5 * se + C / R  # + compulsory-miss warm-up


def test_cyclic_scan_lru_zero_belady_closed_form():
    # N+1 keys accessed cyclically through a cache of N: LRU always evicts the
    # key needed next (0 hits); Belady misses once per N accesses, so its hit
    # rate tends to (N-1)/N.
    N, cycles = 10, 10_000
    seq = np.tile(np.arange(N + 1), cycles)
    assert cache_sim.lru_hits(seq, N) == 0
    hit = cache_sim.belady_hits(seq, N) / len(seq)
    assert abs(hit - (N - 1) / N) < 2e-3


def test_belady_never_below_lru():
    rng = np.random.default_rng(1)
    for seed in range(5):
        seq = rng.zipf(1.3, size=20_000) % 300
        for C in (5, 30, 120):
            assert cache_sim.belady_hits(seq, C) >= cache_sim.lru_hits(seq, C)


def test_request_stream_is_token_major_and_layer_offset():
    traces = {3: np.array([[0, 1], [2, 3]]), 7: np.array([[1, 0], [3, 2]])}
    seq, T, L, k = cache_sim.request_stream(traces, num_experts=4)
    assert (T, L, k) == (2, 2, 2)
    assert seq.tolist() == [0, 1, 5, 4, 2, 3, 7, 6]


# ---- G1 analysis -----------------------------------------------------------

HDR = ("repeat,mode,pattern,size_kb,threads,bytes,seconds,MBps,iops,lat_p50_us,lat_p99_us,"
       "errors,first_errno,power_w,temp_start_c,temp_end_c,temp_zone")


def _rows(lines):
    import csv
    rows = []
    for r in csv.DictReader([HDR] + lines):
        for k in g1_analyze.NUMERIC:
            r[k] = float(r[k])
        rows.append(r)
    return rows


def test_reads_above_interface_ceiling_are_invalid_not_averaged():
    rows = _rows([
        "1,direct,rand,4,1,1,1,450,1,1,1,0,0,nan,nan,nan,none",
        "1,direct,rand,512,1,1,1,3500,1,1,1,0,0,nan,nan,nan,none",
        "2,direct,rand,512,1,1,1,9000,1,1,1,0,0,nan,nan,nan,none",  # impossible: cache-served
    ])
    res = g1_analyze.analyse(rows, True)
    assert res["n_invalid_above_interface_ceiling"] == 1
    assert res["bulk_rand_best"]["MBps_median"] == 3500
    assert res["small_4k_rand_best"]["MBps_median"] == 450
    assert res["bulk_to_small_ratio"] == pytest.approx(3500 / 450)


def test_energy_subtracts_idle_power():
    rows = _rows([
        "0,idle,none,0,0,0,5,0,0,0,0,0,0,1.0,30,30,ddr",
        "1,direct,rand,512,1,1,1,2000,1,1,1,0,0,3.0,30,31,ddr",
    ])
    res = g1_analyze.analyse(rows, True)
    # (3 W - 1 W) / 2 GB/s = 1 J/GB
    assert res["bulk_rand_best"]["J_per_GB_above_idle"] == pytest.approx(1.0)


def _temp_rows(temps, zone="socd"):
    return _rows([f"1,direct,rand,512,1,1,1,{1000 + i},1,1,1,0,0,nan,{t},{t},{zone}"
                  for i, t in enumerate(temps)])


def test_constant_temperature_under_sustained_load_is_flagged_not_reported():
    # A die whose temperature is bit-identical across every configuration of a
    # multi-minute I/O benchmark is not being measured. This is the OnePlus 15R
    # `socd` pseudo-zone, which reads a bare 75 and is a perfectly plausible
    # 75.0 C. The value is in band; only its constancy gives it away.
    res = g1_analyze.analyse(_temp_rows([75.0] * 12), True)
    assert res["thermal"]["temp_column_suspect_constant"] is True
    assert res["thermal"]["n_distinct_temps"] == 1


def test_varying_temperature_is_not_flagged():
    res = g1_analyze.analyse(_temp_rows([40.0 + 0.5 * i for i in range(12)]), True)
    assert res["thermal"]["temp_column_suspect_constant"] is False
    assert res["thermal"]["max_temp_c"] == pytest.approx(45.5)
    assert res["thermal"]["max_temp_zone"] == "socd"


def test_temperatures_outside_physical_band_are_excluded_from_the_maximum():
    # -273 is the disabled-sensor sentinel; 300 C is not a phone.
    res = g1_analyze.analyse(_temp_rows([-273.0, 41.0, 300.0, 44.0]), True)
    assert res["thermal"]["n_rows_temp_implausible"] == 2
    assert res["thermal"]["max_temp_c"] == pytest.approx(44.0)


def test_g0_thermal_rejects_levels_and_sentinels_by_unit_not_by_name():
    """thermal sysfs `temp` is millidegrees Celsius. A zone reporting a bare 75
    is a level (the 15R's `socd`), even though 75.0 C is a perfectly believable
    die temperature and the name gives nothing away. Rejecting it must not
    depend on knowing that name, or the next device ships a new one and the
    maximum temperature of every run is silently wrong again."""
    import g0_summarize
    probe = "\n".join([
        "== thermal zones ==",
        "thermal_zone0 cpullc-0-0 43200",      # real
        "thermal_zone1 cpu-0-0-1 46500",       # real, hottest
        "thermal_zone85 socd 75",              # level, not millidegrees
        "thermal_zone51 cpu-hw-trip-0 95000",  # threshold, plausible value
        "thermal_zone55 pm8550-bcl-lvl0 0",    # level
        "thermal_zone65 pm7550ba-ibat-lvl0 -843",   # level, negative
        "thermal_zone86 sdr0 -273000",         # disabled-sensor sentinel
        "",
    ])
    t = g0_summarize.parse(probe)["thermal"]
    assert t["max_sensor_c"] == 46.5
    assert t["max_sensor_zone"] == "cpu-0-0-1"
    assert t["n_zones_sensor"] == 2
    assert t["excluded_by_reason"]["not_millidegrees"] == ["socd"]
    assert t["excluded_by_reason"]["outside_physical_band"] == ["sdr0"]
    assert set(t["excluded_by_reason"]["threshold_or_level_name"]) == {
        "cpu-hw-trip-0", "pm8550-bcl-lvl0", "pm7550ba-ibat-lvl0"}


def test_g0_records_denied_reads_rather_than_dropping_them():
    """An unreadable power rail must appear in the artifact as unreadable. If it
    were simply absent, 'we did not measure energy' and 'energy was zero' would
    be indistinguishable later."""
    import g0_summarize
    probe = "\n".join([
        "== power measurement ==",
        "battery/current_now=DENIED_OR_ABSENT",
        "battery/voltage_now=DENIED_OR_ABSENT",
        "FAILED rc=127: dumpsys powerstats",
        "",
    ])
    res = g0_summarize.parse(probe)
    assert res["power"]["any_rail_readable"] is False
    assert res["power"]["dumpsys_powerstats_available"] is False
    assert res["probe_outcomes"]["n_denied_or_absent"] == 2
    assert res["probe_outcomes"]["n_failed"] == 1


def test_g0_root_detection_requires_uid_zero_not_an_su_binary():
    """Termux ships its own `su`. A probe that reports root because some `su`
    exists on PATH would call a locked, verified-boot-green phone rooted."""
    import g0_summarize
    not_root = g0_summarize.parse(
        "== root ==\nNOT_ROOT: su at /usr/bin/su exists but 'su -c id -u' gave '<nothing>'\n"
        "ro.boot.verifiedbootstate=green\nro.boot.flash.locked=1\n")
    assert not_root["root"]["rooted"] is False
    rooted = g0_summarize.parse("== root ==\nROOT: su at /system/xbin/su returned uid 0\n")
    assert rooted["root"]["rooted"] is True


MEMPROBE_CSV = """\
oom_score_adj=50 MemTotal_MB=11099 SwapTotal_MB=12287
held_MB,chunk_s,fill_MBps,VmRSS_MB,VmSwap_MB,MemAvailable_MB,SwapFree_MB,stop_reason
256,0.227,1126,259,0,3208,8052,
512,0.215,1189,515,0,2941,8059,
768,1.335,192,700,300,1749,6195,rss_plateau
max_VmRSS_MB=720 chunks=3
"""


def test_memprobe_operating_point_is_resident_not_allocated(tmp_path):
    """A parser that reports `held_MB` reports pages the kernel already
    compressed into zram. The budget an engine can use is VmRSS. Here 768 MB
    was allocated but only 720 MB was ever resident, so 720 is the answer."""
    import g0_summarize
    p = tmp_path / "memprobe.csv"
    p.write_text(MEMPROBE_CSV, encoding="utf-8")
    for parsed in (g0_summarize.parse_memprobe(str(p)), g1_analyze.load_memprobe(str(p))):
        assert "error" not in parsed
        assert parsed["max_VmRSS_MB"] == 720
        # Rounded to 3 dp on purpose: this value is printed into a --ram-gb argument.
        assert parsed["resident_GB_decimal"] == round(720 * 1048576 / 1e9, 3)
        assert parsed["last_held_MB"] == 768


def test_memprobe_from_an_older_collector_is_rejected_not_misread(tmp_path):
    """The pre-2026-09-14 collector emitted 5 columns and no max_VmRSS_MB line.
    Silently reading `held_MB` from it would report allocated memory under a
    resident-memory name; both parsers must refuse instead."""
    import g0_summarize
    p = tmp_path / "old.csv"
    p.write_text("oom_score_adj=50 MemTotal_MB=11099\n"
                 "held_MB,chunk_s,MemAvailable_MB,SwapFree_MB,stop_reason\n"
                 "256,0.227,3103,8363,\n"
                 "512,1.316,2314,8034,fill_slowdown_swap\n", encoding="utf-8")
    for parsed in (g0_summarize.parse_memprobe(str(p)), g1_analyze.load_memprobe(str(p))):
        assert "error" in parsed
        assert "max_VmRSS_MB" not in parsed


def test_missing_direct_io_trailer_does_not_become_a_buffered_claim():
    """ufsbench writes its `# direct_io_supported=` trailer only on a clean
    exit. A truncated CSV must not silently report buffered bandwidth as the
    direct-I/O result."""
    rows = _rows([
        "1,direct,rand,512,1,1,1,3000,1,1,1,0,0,nan,40,41,ddr",
        "1,buffered,rand,512,1,1,1,1000,1,1,1,0,0,nan,40,41,ddr",
    ])
    res = g1_analyze.analyse(rows, None)          # trailer absent
    assert res["mode_used"] == "direct"
    assert "truncated" in res["mode_determined_by"]
    assert res["bulk_rand_best"]["MBps_median"] == 3000
    res_ok = g1_analyze.analyse(rows, True)       # trailer present
    assert res_ok["mode_determined_by"] == "collector trailer"


def test_descheduled_rows_are_excluded_not_averaged_in():
    """ufsbench stops each configuration at a deadline, so a row that took far
    longer than the others was not running for part of its interval and its
    bytes/elapsed measures CPU share, not storage. These are the real
    2026-09-14 rows: identical configuration, screen on vs screen off, p50
    latency essentially unchanged while throughput fell 7.5x."""
    rows = _rows([
        "1,direct,rand,16,8,1627406336,2.0004,813.55,49655.4,149.7,360.0,0,0,nan,50.9,53.6,qmx-1-0",
        "2,direct,rand,16,8,1600000000,2.0010,800.00,49000.0,150.0,360.0,0,0,nan,50.9,53.6,qmx-1-0",
        "3,direct,rand,16,8,529743872,4.8798,108.56,6625.9,157.6,452.1,0,0,nan,51.7,49.8,pm8550_tz",
    ])
    res = g1_analyze.analyse(rows, True)
    assert res["n_invalid_process_descheduled"] == 1
    assert res["descheduled_rows"][0]["seconds"] == pytest.approx(4.8798)
    # The surviving cell must not have been dragged down by the bad row.
    cell = [t for t in res["table"] if t["size_kb"] == 16][0]
    assert cell["n"] == 2
    assert cell["MBps_median"] == pytest.approx((813.55 + 800.00) / 2)


def test_rows_within_deadline_tolerance_are_kept():
    rows = _rows([
        "1,direct,rand,16,8,1,2.00,800,1,1,1,0,0,nan,40,41,ddr",
        "2,direct,rand,16,8,1,2.10,790,1,1,1,0,0,nan,40,41,ddr",
    ])
    res = g1_analyze.analyse(rows, True)
    assert res["n_invalid_process_descheduled"] == 0


def test_absent_power_rail_reports_no_energy_rather_than_zero():
    rows = _rows(["1,direct,rand,512,1,1,1,2000,1,1,1,0,0,nan,40,41,ddr"])
    res = g1_analyze.analyse(rows, True)
    assert res["idle_power_w"] is None
    assert res["bulk_rand_best"]["J_per_GB_above_idle"] is None


# ---- Kaggle notebook is a copy, so prove it is the SAME copy ---------------

def test_kaggle_notebook_is_regenerable_and_matches_the_gate_scripts():
    """The notebook embeds gates/traces_sparsity.py and gates/cache_sim.py in
    %%writefile cells, because Kaggle runs one uploaded file, not a repo. If the
    embedded copy drifts from the repo file, the Kaggle numbers are numbers from
    some other version of the code. The expected value here is an identity, not
    a result: it holds regardless of whether any gate passes."""
    kaggle = os.path.join(os.path.dirname(HERE), "kaggle")
    sys.path.insert(0, kaggle)
    import build_notebook

    with open(build_notebook.NOTEBOOK, encoding="utf-8") as f:
        current = f.read()
    assert current == build_notebook.serialise(build_notebook.build()), (
        "moe-phone/kaggle/g2_g3_olmoe.ipynb is out of date; regenerate with "
        "python moe-phone/kaggle/build_notebook.py")

    # And the embedded bodies really are the repo files, byte for byte.
    import json as _json
    nb = _json.loads(current)
    embedded = {}
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if src.startswith("%%writefile "):
            head, _, body = src.partition("\n")
            embedded[head.split()[1]] = body
    assert set(embedded) == set(build_notebook.EMBED)
    for name, body in embedded.items():
        with open(os.path.join(os.path.dirname(HERE), "gates", name), encoding="utf-8") as f:
            assert body == f.read(), f"{name} in the notebook differs from gates/{name}"


def test_kaggle_notebook_validates_against_the_nbformat_schema():
    """Kaggle warned `MissingIDFieldWarning: ... this will become a hard error
    in future nbformat versions` on the first generated notebook. Validate with
    warnings raised, so a schema regression fails here rather than on Kaggle
    after the user has queued a session."""
    nbformat = pytest.importorskip("nbformat")
    import warnings
    path = os.path.join(os.path.dirname(HERE), "kaggle", "g2_g3_olmoe.ipynb")
    nb = nbformat.read(path, as_version=4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nbformat.validate(nb)
    ids = [c.get("id") for c in nb.cells]
    assert all(ids) and len(set(ids)) == len(ids)


def test_kaggle_notebook_refuses_to_run_without_a_gpu():
    """The default Kaggle accelerator is None, and the first run of this
    notebook hit exactly that. The guard must come BEFORE the model download,
    or a CPU session spends an hour producing a different measurement."""
    import json as _json
    path = os.path.join(os.path.dirname(HERE), "kaggle", "g2_g3_olmoe.ipynb")
    with open(path, encoding="utf-8") as f:
        cells = _json.load(f)["cells"]
    sources = ["".join(c["source"]) for c in cells]
    guard = next(i for i, s in enumerate(sources) if "cuda.device_count() < 1" in s)
    model = next(i for i, s in enumerate(sources) if "allenai/OLMoE-1B-7B-0924" in s
                 and not s.startswith("%%writefile"))
    assert guard < model, "the GPU check must precede the model run"
    assert "Accelerator" in sources[guard]


def test_kaggle_notebook_carries_no_replacement_characters():
    """U+FFFD in a deliverable means it was written through a lossy encoder;
    the checked-in notebook had them where em dashes belonged."""
    with open(os.path.join(os.path.dirname(HERE), "kaggle", "g2_g3_olmoe.ipynb"),
              encoding="utf-8") as f:
        assert "�" not in f.read()


# ---- G2/G3 instrumentation (needs torch + transformers) --------------------

def test_traces_sparsity_selftest():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import traces_sparsity
    assert traces_sparsity.selftest() == 0
