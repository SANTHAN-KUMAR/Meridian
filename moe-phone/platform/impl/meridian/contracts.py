"""DeviceProfile contract, per platform/03_INTERFACES.md section 1.

This module is a direct transcription of that schema into Python dataclasses.
It carries no logic beyond serialisation: every value that reaches a
DeviceProfile field is produced by a probe in meridian/probes/, never
invented here. Fields this codebase cannot yet measure are populated with
Measured(provenance="unknown", ...) rather than omitted or defaulted -- per
03_INTERFACES.md Rule 1, "never substitute a default and call it a value".

schema_version 1 matches the schema as specified on 2026-09-20.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import time

SCHEMA_VERSION = 1

Provenance = str  # "measured" | "calibrated" | "prior" | "declared" | "unknown"


@dataclass
class Measured:
    value: Any
    provenance: Provenance
    confidence: float
    interval: Optional[list] = None
    observed_at: Optional[float] = None
    conditions: Optional[dict] = None
    source: str = ""

    def __post_init__(self):
        if self.provenance != "measured" and self.value is not None and self.interval is None:
            raise ValueError(
                f"Measured value with provenance={self.provenance!r} must carry an interval "
                f"(03_INTERFACES.md Rule 1); source={self.source!r}"
            )

    @staticmethod
    def unknown(source: str) -> "Measured":
        return Measured(value=None, provenance="unknown", confidence=0.0, source=source)


@dataclass
class ValidityConditions:
    wakefulness: str = "unknown"       # "awake" | "dozing" | "unknown"
    foreground: str = "unknown"        # "self" | "other-app" | "none" | "unknown"
    power: str = "unknown"             # "unplugged" | "ac" | "usb"
    thermal_status: Optional[int] = None
    skin_c_start: Optional[float] = None
    skin_c_end: Optional[float] = None
    battery_pct: Optional[int] = None
    descheduled: bool = False
    concurrent_load: list = field(default_factory=list)


@dataclass
class CpuCluster:
    name: str
    core_ids: list
    max_khz: int
    isa_features: list
    matmul_gbps: Measured  # engine-kernel matmul throughput; usually "unknown" at L1


@dataclass
class Cpu:
    clusters: list  # [CpuCluster]
    recommended_compute_mask: Measured
    recommended_io_mask: Measured


@dataclass
class ZramOrSwap:
    present: bool
    size: Optional[int]
    fault_cost: Measured


@dataclass
class Memory:
    total: int
    available_at_probe: Measured
    grantable_quiesced: Measured
    grantable_foreground: Measured
    zram_or_swap: ZramOrSwap
    dram_read_gbps: Measured
    lmk_behaviour: Optional[dict] = None


@dataclass
class RandomReadPoint:
    size_bytes: int
    threads: int
    mbps: Measured
    p50_us: float
    p99_us: float


@dataclass
class Storage:
    path: str
    filesystem: str
    free_bytes: int
    random_read: list  # [RandomReadPoint]
    efficient_request_size: Measured
    sequential_write_mbps: Measured
    direct_io_supported: bool


@dataclass
class Accelerator:
    kind: str            # "cpu" | "gpu" | "npu" | "vendor"
    backend_id: str
    available: bool
    reachable_unprivileged: bool
    decode_gbps: Measured
    prefill_gbps: Measured
    dispatch_overhead_ms: Measured
    fidelity_class: str  # "bit-exact" | "tolerant" | "unverified"


@dataclass
class Thermal:
    sustained_derate: Measured
    time_to_throttle_s: Measured
    recovery_s: Measured
    zones: list  # [{"name": str, "usable": bool}]


@dataclass
class Power:
    rails_available: bool
    method: str  # "per-rail" | "battery-counter" | "none"
    idle_w: Optional[Measured] = None


@dataclass
class Identity:
    manufacturer: str
    model: str
    soc_vendor: str
    soc_model: str
    board: str
    os_version: str
    api_level: str
    build_fingerprint: str
    rooted: bool
    verified_boot: str


@dataclass
class Validity:
    conditions: ValidityConditions
    rejected_runs: int
    rejection_reasons: list


@dataclass
class DeviceProfile:
    profile_id: str
    probe_suite: str
    state: str  # "fresh" | "stale" | "suspect" | "partial"
    identity: Identity
    cpu: Cpu
    memory: Memory
    storage: Storage
    accelerators: list  # [Accelerator]
    thermal: Thermal
    power: Power
    validity: Validity
    schema_version: int = SCHEMA_VERSION
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)
