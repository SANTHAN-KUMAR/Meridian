"""ModelCard derivation and the feasibility half of the ConfigurationPlanner
(05_PERFORMANCE_MODEL.md section 6.2, steps 1 and the refusal rules of 5).

What this module does NOT do, on purpose: predict tokens per second. That
needs kernel throughput per cluster, the thermal derate curve and (streamed)
a measured cache hit rate; the profile has none of them (STATUS.md PL-E11,
PL-E14). Faced with that, plan() returns typed Refusals naming exactly what
is missing -- it never fills the gap with a formula (D10).

Soundness argument for the only two verdicts it will issue on its own:
  * Infeasible for a tier: the tier's byte need EXCLUDES the engine working
    set (unmeasured, PL-E21), so the need is a lower bound; a lower bound above
    an upper bound on the grant is a sound refusal. The grant upper bound is
    grantable_quiesced, valid under assumption A1: a foreground app cannot
    increase what one process may keep.
  * Otherwise NotCalibrated -- never "feasible".
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional

from . import gguf

# Architecture registry (03_INTERFACES.md section 2): one row per family, the
# expert-tensor name pattern. Rows are added only after the pattern is
# observed in a real checkpoint of that family (tools: gguf.read). Observed
# 2026-09-20 in local files: olmoe, gpt-oss, granitemoe, qwen3moe (the last from the device copy of Qwen3-30B-A3B-Q4_0).
EXPERT_SUFFIXES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                   "ffn_gate_exps.bias", "ffn_up_exps.bias", "ffn_down_exps.bias")
ARCH_REGISTRY = {"olmoe": EXPERT_SUFFIXES, "gpt-oss": EXPERT_SUFFIXES, "granitemoe": EXPERT_SUFFIXES, "qwen3moe": EXPERT_SUFFIXES}

# ggml block formats (ggml-common.h): Q8_0 = 32 elems in 34 bytes, Q4_0 = 32 in 18.
KV_BYTES_PER_ELEM = {"f16": 2.0, "q8": 34 / 32, "q4": 18 / 32}


class Refusal(Exception):
    def __init__(self, reason: str, detail: str, missing=None, nearest=None):
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail, self.missing, self.nearest = reason, detail, missing or [], nearest

    def to_dict(self):
        return {"reason": self.reason, "detail": self.detail, "missing": self.missing, "nearest": self.nearest}


@dataclass
class ModelCard:
    model_id: str
    architecture: str
    is_moe: bool
    n_layer: int
    n_expert: int
    n_expert_used: int
    context_limit: int
    total_bytes: int
    resident_bytes: int          # everything that is not a routed expert
    expert_bytes: int
    active_bytes_per_token: int  # bytes one token's forward pass must touch
    expert_slice_bytes: dict     # per expert-tensor kind: min/max contiguous bytes of ONE expert
    kv_bytes_per_token: dict     # f16 exact; q8/q4 from ggml block sizes
    kv_note: str
    declared: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def derive_card(path: str, model_id: Optional[str] = None) -> ModelCard:
    g = gguf.read(path)
    kv = g["kv"]
    arch = kv.get("general.architecture")
    if arch not in ARCH_REGISTRY:
        raise Refusal("ArchitectureUnsupported",
                      f"architecture {arch!r} has no registry row; a row needs its expert-tensor pattern "
                      f"observed in a real checkpoint (known: {sorted(ARCH_REGISTRY)})")
    n_layer = int(kv[f"{arch}.block_count"])
    n_expert = int(kv[f"{arch}.expert_count"])
    n_used = int(kv[f"{arch}.expert_used_count"])
    suffixes = ARCH_REGISTRY[arch]
    tensors = g["tensors"]
    exp = [t for t in tensors if t["name"].endswith(suffixes)]
    if not exp:
        raise Refusal("ArchitectureUnsupported", f"{arch}: no tensors match the registry's expert pattern")
    slices: dict = {}
    for t in exp:
        if t["dims"][-1] != n_expert:
            raise Refusal("IntegrityFailed", f"{t['name']} last dim {t['dims'][-1]} != expert_count {n_expert}")
        kind = t["name"].split("blk.")[1].split(".", 1)[1]
        s = t["bytes"] // n_expert
        if t["bytes"] % n_expert:
            raise Refusal("IntegrityFailed", f"{t['name']} bytes not divisible by expert_count")
        lo, hi = slices.get(kind, (s, s))
        slices[kind] = (min(lo, s), max(hi, s))
    expert_bytes = sum(t["bytes"] for t in exp)
    total = sum(t["bytes"] for t in tensors)
    resident = total - expert_bytes
    embd = next((t for t in tensors if t["name"] == "token_embd.weight"), None)
    tied_head = not any(t["name"] == "output.weight" for t in tensors)
    # an embedding lookup touches one row; if the head is tied the table is read in full by the head.
    lookup_saving = embd["bytes"] if (embd and not tied_head) else 0
    active = resident - lookup_saving + expert_bytes * n_used // n_expert
    n_head, n_kv = int(kv[f"{arch}.attention.head_count"]), int(kv[f"{arch}.attention.head_count_kv"])
    hd = int(kv.get(f"{arch}.attention.key_length", int(kv[f"{arch}.embedding_length"]) // n_head))
    hdv = int(kv.get(f"{arch}.attention.value_length", hd))
    elems = n_layer * n_kv * (hd + hdv)
    note = "upper bound: sliding-window layers (if any) keep fewer than context_len tokens" \
        if f"{arch}.attention.sliding_window" in kv else "exact for full attention"
    return ModelCard(
        model_id=model_id or path.rsplit("/", 1)[-1], architecture=arch, is_moe=True, n_layer=n_layer,
        n_expert=n_expert, n_expert_used=n_used, context_limit=int(kv[f"{arch}.context_length"]),
        total_bytes=total, resident_bytes=resident, expert_bytes=expert_bytes,
        active_bytes_per_token=active, expert_slice_bytes={k: list(v) for k, v in slices.items()},
        kv_bytes_per_token={k: int(elems * b) for k, b in KV_BYTES_PER_ELEM.items()}, kv_note=note,
        declared={"file_bytes": g["file_bytes"], "n_tensors": len(tensors), "tied_head": tied_head},
    )


def _tier_need(card: ModelCard, tier: str, context: int, kv: str = "f16") -> int:
    """Lower bound on bytes the tier must hold. Excludes the engine working set (PL-E21)."""
    kvb = card.kv_bytes_per_token[kv] * context
    if tier == "resident":
        return card.total_bytes + kvb
    # streamed: non-expert weights + KV + at least one layer's worth of active experts as cache floor
    per_layer_active = sum(hi for _, hi in card.expert_slice_bytes.values()) * card.n_expert_used
    return card.resident_bytes + kvb + per_layer_active


def plan(profile: dict, card: ModelCard, context: int, kv: str = "f16") -> dict:
    """Returns {"tiers": {tier: verdict}} or raises Refusal. verdict is never 'feasible'."""
    gq = profile["memory"]["grantable_quiesced"]
    if gq["value"] is None:
        raise Refusal("NotCalibrated", "grantable_quiesced was not measured", missing=["memory.grantable_quiesced"])
    if context > card.context_limit:
        raise Refusal("OutOfRange", f"context {context} exceeds model limit {card.context_limit}")
    upper = gq["value"]  # A1: foreground grant <= quiesced grant
    missing_common = ["memory.grantable_foreground (PL-S2)", "cpu.clusters[].matmul_gbps (PL-E11)",
                      "thermal.sustained_derate (PL-E14)"]
    if gq["provenance"] != "measured":
        missing_common.insert(0, f"memory.grantable_quiesced measured in-regime (is {gq['provenance']})")
    out = {}
    for tier in ("resident", "streamed"):
        need = _tier_need(card, tier, context, kv)
        if need > upper:
            out[tier] = {"verdict": "Infeasible", "need_lower_bound": need, "grant_upper_bound": upper,
                         "detail": "need (excluding engine working set) exceeds the largest grant ever "
                                   "observed to be keepable, under assumption A1"}
        else:
            miss = list(missing_common) + (["expert-cache warm hit-rate sample (04 sec 3.6)"] if tier == "streamed" else [])
            out[tier] = {"verdict": "NotCalibrated", "need_lower_bound": need, "grant_upper_bound": upper,
                         "detail": "passes the necessary memory condition only; no rate can be promised",
                         "missing": miss}
    if all(v["verdict"] == "Infeasible" for v in out.values()):
        raise Refusal("Infeasible", f"no tier fits context={context}: " + "; ".join(
            f"{t} needs >= {v['need_lower_bound']} B vs grant <= {v['grant_upper_bound']} B" for t, v in out.items()),
            nearest=None)
    return {"context": context, "kv": kv, "tiers": out}
