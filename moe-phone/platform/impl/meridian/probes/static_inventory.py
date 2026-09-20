"""T0 static inventory parser: turns device/g0_probe.sh's output and
devprobe's output into Identity, CPU topology and storage facts.

Follows 04_DEVICE_PROFILING.md section 3.1's three rules:
  - DENIED_OR_ABSENT is carried through verbatim, never turned into a
    default or a zero.
  - the storage device is resolved from the mount that backs the model
    directory (g0_probe.sh already does this; we just read its answer).
  - root is judged by g0_probe.sh's executed `su -c id -u`, not by finding a
    binary on the path.
"""
from __future__ import annotations

import re


def kv(text: str, key: str) -> str | None:
    m = re.search(rf"^{re.escape(key)}=(.*)$", text, re.M)
    if not m:
        return None
    v = m.group(1).strip()
    return None if v in ("", "DENIED_OR_ABSENT") else v


def parse_identity(g0_text: str) -> dict:
    fp = kv(g0_text, "ro.build.fingerprint") or "UNKNOWN"
    # g0_probe.sh prints one of "ROOT: su at ... returned uid 0" or
    # "NOT_ROOT: ...". Must match at line start -- "ROOT:" is a substring of
    # "NOT_ROOT:" too.
    rooted = bool(re.search(r"^ROOT: su at .* returned uid 0", g0_text, re.M))
    return {
        "manufacturer": kv(g0_text, "ro.product.manufacturer") or "UNKNOWN",
        "model": kv(g0_text, "ro.product.model") or "UNKNOWN",
        "soc_vendor": kv(g0_text, "ro.soc.manufacturer") or "UNKNOWN",
        "soc_model": kv(g0_text, "ro.soc.model") or "UNKNOWN",
        "board": kv(g0_text, "ro.board.platform") or "UNKNOWN",
        "os_version": kv(g0_text, "ro.build.version.release") or "UNKNOWN",
        "api_level": kv(g0_text, "ro.build.version.sdk") or "UNKNOWN",
        "build_fingerprint": fp,
        "rooted": rooted,
        "verified_boot": kv(g0_text, "ro.boot.verifiedbootstate") or "UNKNOWN",
    }


def parse_cpu_topology(g0_text: str) -> list:
    """Group cores by (CPU part, max_khz) into clusters. This is topology,
    not a performance claim -- matmul_gbps per cluster is filled in
    separately (probes/compute.py at L2; unmeasured at L1, see STATUS.md).
    """
    parts = {}  # core_id -> cpu part
    for m in re.finditer(r"processor\s*:\s*(\d+)\s*CPU part\s*:\s*(0x[0-9a-fA-F]+)", g0_text):
        parts[int(m.group(1))] = m.group(2)
    khz = {}
    for m in re.finditer(r"cpu(\d+) max_khz=(\d+)", g0_text):
        khz[int(m.group(1))] = int(m.group(2))

    groups: dict = {}
    for core_id in sorted(parts):
        key = (parts.get(core_id), khz.get(core_id))
        groups.setdefault(key, []).append(core_id)

    # Name clusters by ascending max_khz: efficiency < performance < prime.
    ordered = sorted(groups.items(), key=lambda kv: (kv[0][1] or 0))
    names = ["efficiency", "performance", "prime"]
    clusters = []
    for i, ((part, max_khz), core_ids) in enumerate(ordered):
        name = names[i] if i < len(names) else f"cluster{i}"
        clusters.append({
            "name": name,
            "core_ids": core_ids,
            "max_khz": max_khz or 0,
            "isa_features": [],  # not probed at L1; see STATUS.md PL-E9
            "cpu_part": part,
        })
    return clusters


def _size_to_bytes(s: str) -> int | None:
    m = re.match(r"([\d.]+)([KMGT]?)$", s.strip())
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2)
    mult = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[unit]
    return int(val * mult)


def parse_storage_mount(g0_text: str) -> dict:
    mount = kv(g0_text, "data_mount") or "UNKNOWN"
    fs_match = re.search(rf"\S+\s+{re.escape(mount)}\s+(\S+)\s+", g0_text)
    filesystem = fs_match.group(1) if fs_match else "UNKNOWN"
    # `df -h` line: "<dev> <size> <used> <avail> <use%> <mounted-on>"
    df_match = re.search(
        rf"^\S+\s+\S+\s+\S+\s+(\S+)\s+\d+%\s+{re.escape(mount)}$", g0_text, re.M
    )
    free_bytes = _size_to_bytes(df_match.group(1)) if df_match else None
    return {
        "path": mount,
        "filesystem": filesystem,
        "device": kv(g0_text, "data_device") or "UNKNOWN",
        "free_bytes": free_bytes,
    }


def parse_accelerator_nodes(devprobe_text: str) -> dict:
    """Which device nodes an adb-shell process could open. Per devprobe.c's
    own header: EACCES here does NOT mean an ordinary app is refused too
    (Qualcomm FastRPC's domain probing differs by caller). This function
    reports exactly what it measured -- reachability by *this* process --
    and the caller must not generalise it to "the app" without a same-process
    workload proving it (00_PROBLEM.md section 4, devprobe.c docstring).
    """
    nodes = {}
    for line in devprobe_text.splitlines():
        m = re.match(r"(\S+)\s+(O_RD\w+)\s+(.*)$", line.strip())
        if m:
            path, mode, result = m.groups()
            nodes.setdefault(path, {})[mode.strip()] = result.strip()
    return nodes
