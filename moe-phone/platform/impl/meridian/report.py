"""Capability report: the plain-language summary from 02_ARCHITECTURE.md
section 9, item 1 -- "what this phone can run, how fast, and how confident
the estimate is". This is prose generated FROM the DeviceProfile object,
never a hand-written number (CLAUDE.md section 7.1): every figure below is
read out of the `Measured` values the profiler already attached provenance
and source to.
"""
from __future__ import annotations

from .contracts import DeviceProfile, Measured


def _fmt(m: Measured, unit: str = "", digits: int = 1) -> str:
    if m.provenance == "unknown" or m.value is None:
        return f"not measured ({m.source})"
    val = f"{m.value:.{digits}f}" if isinstance(m.value, float) else str(m.value)
    tag = m.provenance
    if m.interval:
        lo, hi = m.interval
        lo_s = f"{lo:.{digits}f}" if isinstance(lo, float) else str(lo)
        hi_s = f"{hi:.{digits}f}" if isinstance(hi, float) else str(hi)
        return f"{val}{unit} [{tag}, interval {lo_s}-{hi_s}{unit}]"
    return f"{val}{unit} [{tag}]"


def render(profile: DeviceProfile) -> str:
    lines = []
    lines.append(f"# Capability report — {profile.identity.manufacturer} {profile.identity.model}")
    lines.append("")
    lines.append(f"Conformance: **L1 (profiled runner)**. State: `{profile.state}`. "
                  f"Probe suite `{profile.probe_suite}`. Profile id `{profile.profile_id}`.")
    lines.append("")
    lines.append(f"SoC: {profile.identity.soc_vendor} {profile.identity.soc_model} "
                 f"({profile.identity.board}), Android {profile.identity.os_version} "
                 f"(API {profile.identity.api_level}). Rooted: {profile.identity.rooted}.")
    lines.append("")
    lines.append("## What was actually measured on this device")
    lines.append("")
    lines.append(f"- **DRAM read bandwidth** (sequential, per unprivileged process): "
                  f"{_fmt(profile.memory.dram_read_gbps, ' GB/s', 2)}")
    lines.append(f"- **Memory this device will grant one quiesced process and let it keep**: "
                  f"{_fmt(profile.memory.grantable_quiesced, ' bytes', 0)}")
    lines.append(f"- **Memory grantable with a real app in the foreground**: "
                  f"{_fmt(profile.memory.grantable_foreground)} "
                  f"— this is the number an agent actually lives on and this build has not measured it "
                  f"anywhere (registered stub PL-S2). Do not plan agent memory against the quiesced figure above.")
    lines.append(f"- **Storage** ({profile.storage.filesystem} at `{profile.storage.path}`): "
                  f"efficient request size {_fmt(profile.storage.efficient_request_size, ' bytes', 0)}, "
                  f"direct I/O supported: {profile.storage.direct_io_supported}")
    for p in profile.storage.random_read:
        lines.append(f"    - {p.size_bytes // 1024} KiB @ {p.threads} threads: "
                      f"{_fmt(p.mbps, ' MB/s', 1)} (p50 {p.p50_us:.0f} us, p99 {p.p99_us:.0f} us)")
    lines.append("")
    lines.append("## CPU topology (identity only — no throughput measured at L1)")
    for c in profile.cpu.clusters:
        # max_khz is in kHz (sysfs cpuinfo_max_freq units); /1e6 for GHz.
        lines.append(f"- **{c.name}**: cores {c.core_ids}, max {c.max_khz/1e6:.3f} GHz, "
                      f"matmul throughput: {_fmt(c.matmul_gbps)}")
    lines.append("")
    lines.append("## Accelerators")
    for a in profile.accelerators:
        lines.append(f"- **{a.kind}** (`{a.backend_id}`): available={a.available}, "
                      f"reachable_unprivileged={a.reachable_unprivileged}, "
                      f"fidelity_class={a.fidelity_class}, "
                      f"decode={_fmt(a.decode_gbps)}, prefill={_fmt(a.prefill_gbps)}")
    lines.append("")
    lines.append("## Conditions these measurements were taken under")
    c = profile.validity.conditions
    lines.append(f"- power: **{c.power}**, wakefulness: {c.wakefulness}, foreground: {c.foreground}, "
                  f"battery: {c.battery_pct}%")
    if c.power != "unplugged":
        lines.append(f"  **This is not the deployment regime (D4 requires unplugged).** Every "
                      f"`measured` value above is downgraded to `prior` with a widened interval for "
                      f"exactly this reason — it is a real number taken under charging, not a "
                      f"sustained-unplugged operating point, and must not be presented as one.")
    if profile.validity.rejected_runs:
        lines.append(f"- {profile.validity.rejected_runs} run(s) rejected as contaminated: "
                      f"{profile.validity.rejection_reasons}")
    lines.append("")
    lines.append("## What this report is not")
    lines.append("- No tokens-per-second prediction for any model. That requires the PerformanceModel "
                  "(05_PERFORMANCE_MODEL.md), which is not built in this pass.")
    lines.append("- No thermal derate curve (T3), no per-backend decode/prefill rate, no "
                  "foreground-coexistence memory grant. See STATUS.md for the full stub list.")
    return "\n".join(lines)
