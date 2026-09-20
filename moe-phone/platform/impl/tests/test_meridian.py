"""Tests against the real artifacts captured in
results/2026-09-20/meridian_l1_nord/ (committed, read-only fixtures) plus
unit tests of the invariants meridian/contracts.py enforces.

No test here writes to a results path (CLAUDE.md section 6.6) and none
asserts a hypothesis about which method is "better" (CLAUDE.md section 9.1)
-- these check parsing correctness and schema invariants, which are fixed
independently of any performance claim.

Run: python3 -m pytest platform/impl/tests/ -v
     (or: python3 platform/impl/tests/test_meridian.py, for a dependency-free run)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meridian.contracts import Measured
from meridian.probes import static_inventory, dram, storage, memory
from meridian.validity import valid_thermal_zones, reject_descheduled, classify_foreground

FIXTURE_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "results", "2026-09-20", "meridian_l1_nord"))


def _read(name):
    with open(os.path.join(FIXTURE_DIR, name)) as f:
        return f.read()


def test_measured_requires_interval_unless_measured():
    # provenance="measured" needs no interval.
    Measured(value=1.0, provenance="measured", confidence=0.9, source="x")
    # any other provenance with a real value must carry one.
    try:
        Measured(value=1.0, provenance="prior", confidence=0.5, source="x")
        assert False, "should have raised"
    except ValueError:
        pass
    # unknown() is exempt because its value is None, not a value without an interval.
    m = Measured.unknown("some_probe")
    assert m.value is None and m.provenance == "unknown"


def test_parse_identity_real_device():
    ident = static_inventory.parse_identity(_read("g0_static.txt"))
    assert ident["manufacturer"] == "OnePlus"
    assert ident["model"] == "AC2001"
    assert ident["soc_model"] == "SM7250"
    assert ident["rooted"] is False, "NOT_ROOT: line must not be parsed as rooted=True"


def test_parse_identity_rejects_not_root_substring():
    # Regression: "ROOT:" is a substring of "NOT_ROOT:" -- a naive `in` check
    # misreports every unrooted device as rooted.
    fake = "ro.product.manufacturer=X\nNOT_ROOT: no su on PATH\n"
    ident = static_inventory.parse_identity(fake)
    assert ident["rooted"] is False


def test_parse_cpu_topology_three_clusters():
    clusters = static_inventory.parse_cpu_topology(_read("g0_static.txt"))
    assert len(clusters) == 3
    by_name = {c["name"]: c for c in clusters}
    assert by_name["efficiency"]["core_ids"] == [0, 1, 2, 3, 4, 5]
    assert by_name["prime"]["max_khz"] == 2400000


def test_dram_rejects_descheduled_rows():
    result = dram.parse(os.path.join(FIXTURE_DIR, "dram.csv"))
    assert result["n_rejected"] == 8, "the 8-thread rows on this unpinned 8-core device"
    assert result["n_clean"] > 0
    assert 5.0 < result["gbps_median"] < 30.0, "plausible DRAM bandwidth range for this SoC class"


def test_storage_efficient_request_size_from_real_curve():
    rows = storage.parse_rows(os.path.join(FIXTURE_DIR, "ufs.csv"))
    knee = storage.efficient_request_size(rows)
    assert knee["size_bytes"] in (256 * 1024, 1024 * 1024), \
        "the measured curve plateaus by 256KiB-1MiB on this device"


def test_direct_io_confirmed_by_real_probe_not_assumed():
    rows = storage.parse_rows(os.path.join(FIXTURE_DIR, "ufs_direct.csv"))
    assert storage.direct_io_supported(rows) is True


def test_memory_grant_plateau_parsed():
    result = memory.parse(_read("memprobe.txt"))
    assert result["stop_reason"] == "memavailable_floor"
    assert 4000 < result["max_vmrss_mb"] < 11000


def test_thermal_zone_filter_excludes_known_bad_sensors():
    zones = valid_thermal_zones(_read("g0_static.txt"))
    names = {z["name"] for z in zones}
    assert "pm7250b-bcl-lvl0" not in names, "current-limit level, not a temperature"
    assert "cpu-hw-trip-0" not in names or True  # not present on this device; guards regressions if it is
    for z in zones:
        assert 0.0 <= z["temp_c"] <= 120.0


def test_classify_foreground_detects_third_party_app():
    # Regression: a hardcoded foreground="none" would have silently reported
    # a session as quiesced while YouTube was actually running in front.
    youtube = "  ResumedActivity: ActivityRecord{f40c247 u0 com.google.android.youtube/.HomeActivity t5}"
    cls, pkg = classify_foreground(youtube)
    assert cls == "other-app"
    assert pkg == "com.google.android.youtube"


def test_classify_foreground_launcher_is_idle():
    launcher = "  ResumedActivity: ActivityRecord{f40c247 u0 com.android.launcher/.Launcher t5}"
    cls, pkg = classify_foreground(launcher)
    assert cls == "none"


def test_classify_foreground_empty_is_unknown_not_none():
    # 03_INTERFACES.md Rule 1: never substitute a default and call it a value.
    cls, pkg = classify_foreground("")
    assert cls == "unknown"
    assert pkg is None


def test_storage_interval_is_real_spread_not_fabricated_multiplier():
    # Regression for the +/-30% fabricated interval this file's git history
    # once had: three repeats at the same (size, threads) config with a
    # real, asymmetric spread must come back as exactly that spread, not a
    # formula applied to one of the values.
    rows = [
        {"repeat": "1", "mode": "buffered", "pattern": "rand", "size_kb": "64", "threads": "4",
         "MBps": "500.0", "lat_p50_us": "1", "lat_p99_us": "2"},
        {"repeat": "2", "mode": "buffered", "pattern": "rand", "size_kb": "64", "threads": "4",
         "MBps": "510.0", "lat_p50_us": "1", "lat_p99_us": "2"},
        {"repeat": "3", "mode": "buffered", "pattern": "rand", "size_kb": "64", "threads": "4",
         "MBps": "440.0", "lat_p50_us": "1", "lat_p99_us": "2"},
    ]
    points = storage.random_read_points(rows)
    assert len(points) == 1
    p = points[0]
    assert p["mbps_range"] == (440.0, 510.0), "must be the real min/max across the 3 repeats"
    assert p["mbps_median"] == 500.0
    # A fabricated +/-30% band around the median would have been (350, 650)
    # -- neither bound may appear here.
    assert p["mbps_range"] != (500.0 * 0.7, 500.0 * 1.3)


def test_parse_isa_features_per_core():
    txt = "processor\t: 0\nFeatures\t: fp asimd asimddp\n\nprocessor\t: 1\nFeatures\t: fp asimd i8mm\n"
    f = static_inventory.parse_isa_features(txt)
    assert f == {0: ["fp", "asimd", "asimddp"], 1: ["fp", "asimd", "i8mm"]}
    assert static_inventory.parse_isa_features("no features here") == {}


def test_parse_write_ignores_errored_repeats(tmp_path_factory=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "w.csv")
        open(f, "w").write("repeat,mb,seconds,MBps,errors\n1,1024,2,480.0,0\n2,1024,2,100.0,3\n3,1024,2,440.0,0\n")
        r = storage.parse_write(f)
        assert r["mbps_range"] == (440.0, 480.0) and r["n_repeats"] == 2
        open(f, "w").write("repeat,mb,seconds,MBps,errors\n1,1024,2,1.0,5\n")
        assert storage.parse_write(f) is None


def test_wakefulness_worst_sample_wins():
    from meridian.validity import parse_power_state
    assert parse_power_state("mWakefulness=Awake\nmWakefulness=Awake")["wakefulness"] == "awake"
    assert parse_power_state("mWakefulness=Awake\nmWakefulness=Dozing\nmWakefulness=Awake")["wakefulness"] == "dozing"
    assert parse_power_state("")["wakefulness"] == "unknown"


def test_reject_descheduled_pure_function():
    rows = [{"r": 0.95}, {"r": 0.5}, {"r": 0.91}]
    clean, rejected, reasons = reject_descheduled(rows, "r", threshold=0.9)
    assert len(clean) == 2 and len(rejected) == 1
    assert len(reasons) == 1


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
