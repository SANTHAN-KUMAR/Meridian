"""L1 profiler orchestrator: T0 (static inventory) + T1 (fast probes).

Runs the cross-compiled probes on a live device over adb, applies the
validity guard, and assembles a DeviceProfile. Every field that this
codebase cannot yet measure is left as Measured.unknown(...) with a named
reason in STATUS.md -- never filled in with a plausible-looking number
(CLAUDE.md section 0).

This does NOT reach T2/T3 (per-backend decode/prefill rates, the thermal
derate curve, foreground-coexistence grants): those require an actual
inference engine and an installed app, which are L0/L2+ work this pass does
not build. See platform/impl/STATUS.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

from .adb import AdbDevice
from .contracts import (
    DeviceProfile, Identity, Cpu, CpuCluster, Memory, ZramOrSwap, Storage,
    RandomReadPoint, Accelerator, Thermal, Power, Validity, Measured,
)
from .validity import build_conditions, valid_thermal_zones
from .probes import static_inventory, dram, storage, memory

PROBE_SUITE_VERSION = "meridian-l1-0.1.0"
REMOTE_DIR = "/data/local/tmp"


def run_probes(device: AdbDevice, bin_dir: str, out_dir: str) -> dict:
    """Push binaries, run T0+T1 on the live device, pull results. Returns the
    dict of local artifact paths, so the caller can build the profile from
    files on disk rather than from values held only in memory -- the same
    discipline the rest of this repo uses for every reported number."""
    os.makedirs(out_dir, exist_ok=True)
    for b in ("devprobe", "dramprobe", "memprobe", "ufsbench"):
        device.push(os.path.join(bin_dir, b), f"{REMOTE_DIR}/{b}")
    device.shell(f"chmod 755 {REMOTE_DIR}/devprobe {REMOTE_DIR}/dramprobe "
                 f"{REMOTE_DIR}/memprobe {REMOTE_DIR}/ufsbench")

    g0_script_local = os.path.join(os.path.dirname(__file__), "..", "..", "..", "device", "g0_probe.sh")
    g0_script_local = os.path.normpath(g0_script_local)
    device.push(g0_script_local, f"{REMOTE_DIR}/g0_probe.sh")
    # g0_probe.sh forks dozens of subshells (one per `try`/`kv` call); this
    # measured ~37s over wireless adb against <1s over USB on the same
    # script and device -- not a hang, just higher per-exec latency on the
    # TCP transport. Timeout generous enough to cover that.
    g0_text = device.shell(f"sh {REMOTE_DIR}/g0_probe.sh", timeout=90)
    with open(os.path.join(out_dir, "g0_static.txt"), "w") as f:
        f.write(g0_text)

    cpuinfo_text = device.shell("cat /proc/cpuinfo", timeout=30)
    with open(os.path.join(out_dir, "cpuinfo.txt"), "w") as f:
        f.write(cpuinfo_text)

    devprobe_text = device.shell(f"{REMOTE_DIR}/devprobe", timeout=15)
    with open(os.path.join(out_dir, "devprobe.txt"), "w") as f:
        f.write(devprobe_text)

    battery_text = device.shell("dumpsys battery", timeout=15)
    with open(os.path.join(out_dir, "battery_dumpsys.txt"), "w") as f:
        f.write(battery_text)
    power_text = device.shell("dumpsys power | grep -i mWakefulness", timeout=15)
    with open(os.path.join(out_dir, "power_state.txt"), "w") as f:
        f.write(power_text)

    fg_before = device.shell("dumpsys activity activities | grep -i resumed", timeout=15)
    with open(os.path.join(out_dir, "foreground_before.txt"), "w") as f:
        f.write(fg_before)

    device.shell(
        f"{REMOTE_DIR}/dramprobe --mb 512 --threads 1,2,4,6,8 --seconds 1.5 "
        f"--repeats 3 --out {REMOTE_DIR}/dram.csv", timeout=90)
    device.pull(f"{REMOTE_DIR}/dram.csv", os.path.join(out_dir, "dram.csv"))

    # repeats=3, not 1: a single-shot read at each (size, threads) config
    # gives no real basis for an uncertainty interval when this run turns
    # out to be out-of-regime. Real repeats let storage.py report the
    # actual observed min/max instead of the fabricated +/-30% band an
    # earlier version of this profiler invented to satisfy the schema.
    device.shell(
        f"{REMOTE_DIR}/ufsbench --file {REMOTE_DIR}/ufs_test.bin --size-mb 2048 --create "
        f"--seconds 1.5 --repeats 3 --sizes-kb 4,64,256,1024,4096 --threads 1,4,8 "
        f"--modes buffered --patterns rand --out {REMOTE_DIR}/ufs.csv", timeout=300)
    device.pull(f"{REMOTE_DIR}/ufs.csv", os.path.join(out_dir, "ufs.csv"))

    device.shell(
        f"{REMOTE_DIR}/ufsbench --file {REMOTE_DIR}/ufs_test.bin --size-mb 2048 "
        f"--seconds 1 --repeats 1 --sizes-kb 4,1024 --threads 4 --modes direct "
        f"--patterns rand --out {REMOTE_DIR}/ufs_direct.csv", timeout=60)
    device.pull(f"{REMOTE_DIR}/ufs_direct.csv", os.path.join(out_dir, "ufs_direct.csv"))

    # Two independent memprobe runs, not one: same reasoning as above --
    # grantable_quiesced needs a real min/max across repeated allocate-and-
    # verify passes, not an invented [0, observed] band.
    mem_texts = []
    for i in range(2):
        t = device.shell(f"{REMOTE_DIR}/memprobe --oom-adj 0", timeout=90)
        mem_texts.append(t)
        with open(os.path.join(out_dir, f"memprobe_{i}.txt"), "w") as f:
            f.write(t)
    with open(os.path.join(out_dir, "memprobe.txt"), "w") as f:
        f.write(mem_texts[0])  # kept for backward-compatible single-run readers

    fg_after = device.shell("dumpsys activity activities | grep -i resumed", timeout=15)
    with open(os.path.join(out_dir, "foreground_after.txt"), "w") as f:
        f.write(fg_after)

    return {"out_dir": out_dir}


def build_profile(out_dir: str) -> DeviceProfile:
    """Read the artifacts run_probes() produced and assemble a DeviceProfile.
    Split from run_probes() so a profile can be rebuilt from a saved
    artifact directory without touching the device again."""
    def read(name):
        with open(os.path.join(out_dir, name)) as f:
            return f.read()

    g0_text = read("g0_static.txt")
    devprobe_text = read("devprobe.txt")
    battery_text = read("battery_dumpsys.txt")
    power_text = read("power_state.txt")
    fg_before = read("foreground_before.txt") if os.path.exists(os.path.join(out_dir, "foreground_before.txt")) else ""
    fg_after = read("foreground_after.txt") if os.path.exists(os.path.join(out_dir, "foreground_after.txt")) else ""
    mem_texts = []
    for i in range(2):
        p = os.path.join(out_dir, f"memprobe_{i}.txt")
        if os.path.exists(p):
            mem_texts.append(read(f"memprobe_{i}.txt"))
    if not mem_texts:
        mem_texts = [read("memprobe.txt")]  # older single-run artifact directory

    now = time.time()
    ident = static_inventory.parse_identity(g0_text)
    identity = Identity(**ident)

    clusters_raw = static_inventory.parse_cpu_topology(g0_text)
    cpuinfo_path = os.path.join(out_dir, "cpuinfo.txt")
    isa = static_inventory.parse_isa_features(read("cpuinfo.txt")) if os.path.exists(cpuinfo_path) else {}
    for c in clusters_raw:
        per_core = [set(isa[i]) for i in c["core_ids"] if i in isa]
        # features common to every core in the cluster: a feature only some
        # cores have cannot be assumed by a kernel pinned to the cluster.
        c["isa_features"] = sorted(set.intersection(*per_core)) if per_core else []
    clusters = [
        CpuCluster(
            name=c["name"], core_ids=c["core_ids"], max_khz=c["max_khz"],
            isa_features=c["isa_features"],
            matmul_gbps=Measured.unknown("compute_probe:not_implemented_L1"),
        )
        for c in clusters_raw
    ]
    cpu = Cpu(
        clusters=clusters,
        recommended_compute_mask=Measured.unknown("thread_placement_ab:not_implemented_L1"),
        recommended_io_mask=Measured.unknown("thread_placement_ab:not_implemented_L1"),
    )

    # Use the worse of the two samples: if a third-party app was foreground
    # at either end of the T1 window, the whole window is not quiesced
    # (04_DEVICE_PROFILING.md section 4 -- a contaminant anywhere in the
    # run invalidates the run, it is not diluted by the clean parts).
    from .validity import classify_foreground
    fg_before_cls, fg_before_pkg = classify_foreground(fg_before) if fg_before else ("unknown", None)
    fg_after_cls, fg_after_pkg = classify_foreground(fg_after) if fg_after else ("unknown", None)
    if "other-app" in (fg_before_cls, fg_after_cls):
        worst_text = fg_before if fg_before_cls == "other-app" else fg_after
    else:
        worst_text = fg_before or fg_after
    conditions = build_conditions(battery_text, power_text, resumed_activity_text=worst_text)
    conditions_dict = {
        "wakefulness": conditions.wakefulness, "foreground": conditions.foreground,
        "power": conditions.power, "thermal_status": conditions.thermal_status,
        "battery_pct": conditions.battery_pct, "concurrent_load": conditions.concurrent_load,
    }
    # D4 (02_ARCHITECTURE.md): the deployment regime is awake, UNPLUGGED,
    # thermally settled, and for T1's "quiesced" measurements, no foreground
    # app. Both power and foreground must hold or every affected value is
    # downgraded from measured to prior with a widened interval -- this is
    # not diluted by only one of the two conditions being clean.
    in_regime = (conditions.power == "unplugged" and conditions.foreground == "none"
                 and conditions.wakefulness == "awake")

    dram_csv = os.path.join(out_dir, "dram.csv")
    dram_result = dram.parse(dram_csv)
    dram_measured = Measured(
        value=dram_result.get("gbps_median"),
        provenance="measured" if in_regime else "prior",
        confidence=0.7 if in_regime else 0.4,
        interval=None if in_regime else
            [dram_result.get("gbps_min"), dram_result.get("gbps_max")],
        observed_at=now, conditions=conditions_dict,
        source=f"dramprobe:{dram_csv}",
    ) if dram_result.get("gbps_median") is not None else Measured.unknown("dramprobe:no_clean_rows")

    mem_result = memory.parse_multi(mem_texts)
    lo_mb, hi_mb = mem_result.get("range_mb", (None, None))
    grant_quiesced = Measured(
        value=(mem_result["max_vmrss_mb"] * 1024 * 1024) if mem_result.get("max_vmrss_mb") else None,
        provenance="measured" if in_regime else "prior",
        confidence=0.7 if in_regime else 0.4,
        # real min/max across mem_result["n_runs"] independent memprobe
        # runs, not the [0, value] placeholder an earlier version used.
        interval=None if in_regime else [lo_mb * 1024 * 1024, hi_mb * 1024 * 1024],
        observed_at=now, conditions=conditions_dict,
        source=f"memprobe:{out_dir}/memprobe_*.txt n_runs={mem_result.get('n_runs')} "
               f"stop_reason={mem_result.get('stop_reason')}",
    ) if mem_result.get("max_vmrss_mb") else Measured.unknown("memprobe:parse_failed")

    total_kb = None
    import re as _re
    m = _re.search(r"MemTotal:\s+(\d+) kB", g0_text)
    if m:
        total_kb = int(m.group(1))
    avail_kb = None
    m = _re.search(r"MemAvailable:\s+(\d+) kB", g0_text)
    if m:
        avail_kb = int(m.group(1))

    memory_block = Memory(
        total=total_kb * 1024 if total_kb else 0,
        available_at_probe=Measured(
            value=avail_kb * 1024 if avail_kb else None, provenance="measured", confidence=0.9,
            observed_at=now, conditions=conditions_dict, source="g0_probe.sh:/proc/meminfo",
        ) if avail_kb else Measured.unknown("g0_probe.sh"),
        grantable_quiesced=grant_quiesced,
        grantable_foreground=Measured.unknown("PL-S2:never_measured_no_app_exists"),
        zram_or_swap=ZramOrSwap(present=True, size=None,
                                 fault_cost=Measured.unknown("swap_fault_probe:not_implemented_L1")),
        dram_read_gbps=dram_measured,
    )

    mount = static_inventory.parse_storage_mount(g0_text)
    ufs_rows = storage.parse_rows(os.path.join(out_dir, "ufs.csv"))
    points_raw = storage.random_read_points(ufs_rows)
    random_read = [
        RandomReadPoint(
            size_bytes=p["size_bytes"], threads=p["threads"],
            mbps=Measured(
                value=p["mbps_median"],
                provenance="measured" if in_regime else "prior",
                confidence=0.7 if in_regime else 0.4,
                # real min/max across p["n_repeats"] repeats of this exact
                # (size, threads) config, not an invented +/-30% band.
                interval=None if in_regime else list(p["mbps_range"]),
                observed_at=now, conditions=conditions_dict,
                source=f"ufsbench:{os.path.join(out_dir, 'ufs.csv')} n_repeats={p['n_repeats']}",
            ),
            p50_us=p["p50_us"], p99_us=p["p99_us"],
        )
        for p in points_raw
    ]
    knee = storage.efficient_request_size(ufs_rows)
    knee_lo, knee_hi = knee.get("size_bytes_range", (None, None))
    efficient_size = Measured(
        value=knee.get("size_bytes"),
        provenance="measured" if in_regime else "prior",
        confidence=0.7 if in_regime else 0.4,
        # real range of the knee found independently per repeat, not a
        # degenerate [x, x] "interval".
        interval=None if in_regime else [knee_lo, knee_hi],
        observed_at=now, conditions=conditions_dict,
        source=f"ufsbench:{os.path.join(out_dir, 'ufs.csv')} n_repeats={knee.get('n_repeats')}",
    ) if knee.get("size_bytes") else Measured.unknown("ufsbench:no_data")

    direct_ok = storage.direct_io_supported(
        storage.parse_rows(os.path.join(out_dir, "ufs_direct.csv"))
    ) if os.path.exists(os.path.join(out_dir, "ufs_direct.csv")) else False

    storage_block = Storage(
        path=mount["path"], filesystem=mount["filesystem"],
        free_bytes=mount.get("free_bytes") or 0,
        random_read=random_read,
        efficient_request_size=efficient_size,
        sequential_write_mbps=Measured.unknown("sequential_write_probe:not_run_L1"),
        direct_io_supported=direct_ok,
    )

    nodes = static_inventory.parse_accelerator_nodes(devprobe_text)
    gpu_open = nodes.get("/dev/kgsl-3d0", {}).get("O_RDWR", "") == "OPEN_OK"
    accelerators = [
        Accelerator(
            kind="cpu", backend_id="cpu-generic", available=True, reachable_unprivileged=True,
            decode_gbps=Measured.unknown("engine_backend_probe:no_engine_built_L1"),
            prefill_gbps=Measured.unknown("engine_backend_probe:no_engine_built_L1"),
            dispatch_overhead_ms=Measured.unknown("engine_backend_probe:no_engine_built_L1"),
            fidelity_class="bit-exact",
        ),
        Accelerator(
            kind="gpu", backend_id="opencl-adreno",
            available=gpu_open, reachable_unprivileged=gpu_open,
            decode_gbps=Measured.unknown("engine_backend_probe:no_engine_built_L1"),
            prefill_gbps=Measured.unknown("engine_backend_probe:no_engine_built_L1"),
            dispatch_overhead_ms=Measured.unknown("engine_backend_probe:no_engine_built_L1"),
            fidelity_class="unverified",
        ),
        Accelerator(
            kind="npu", backend_id="qnn-htp",
            available="present:" in devprobe_text or "libcdsprpc" in g0_text,
            # devprobe.c's own docstring: a denied open here does NOT mean
            # unreachable by an ordinary app (Qualcomm FastRPC domain
            # routing). This is an existence signal, not a reachability
            # measurement -- see the source comment quoted in STATUS.md.
            reachable_unprivileged=False,
            decode_gbps=Measured.unknown("engine_backend_probe:no_engine_built_L1"),
            prefill_gbps=Measured.unknown("engine_backend_probe:no_engine_built_L1"),
            dispatch_overhead_ms=Measured.unknown("engine_backend_probe:no_engine_built_L1"),
            fidelity_class="unverified",
        ),
    ]

    zones = valid_thermal_zones(g0_text)
    thermal = Thermal(
        sustained_derate=Measured.unknown("thermal_probe:T3_not_implemented"),
        time_to_throttle_s=Measured.unknown("thermal_probe:T3_not_implemented"),
        recovery_s=Measured.unknown("thermal_probe:T3_not_implemented"),
        # "usable" here means "passed the sysfs sanity filter", not
        # "validated against a reference thermometer" -- T3 (04_DEVICE_
        # PROFILING.md section 3.5) is what would establish that.
        zones=[{"name": z["name"], "usable": True} for z in zones],
    )

    power_block = Power(rails_available=False, method="none", idle_w=None)

    reasons = list(dram_result.get("reasons", []))
    validity = Validity(
        conditions=conditions,
        rejected_runs=dram_result.get("n_rejected", 0),
        rejection_reasons=reasons,
    )

    profile_id_src = f"{ident['build_fingerprint']}|{PROBE_SUITE_VERSION}"
    profile_id = hashlib.sha256(profile_id_src.encode()).hexdigest()[:16]

    state = "partial"  # T2/T3 fields are unknown by design at L1

    return DeviceProfile(
        profile_id=profile_id, probe_suite=PROBE_SUITE_VERSION, state=state,
        identity=identity, cpu=cpu, memory=memory_block, storage=storage_block,
        accelerators=accelerators, thermal=thermal, power=power_block,
        validity=validity, created_at=now,
    )


def save_profile(profile: DeviceProfile, out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(profile.to_dict(), f, indent=2, default=str)
