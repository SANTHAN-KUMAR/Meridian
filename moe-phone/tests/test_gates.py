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


def _synthetic_trace(T=1500, L=6, E=32, k=4, seed=0, locality=0.6):
    """A trace with tunable temporal locality: with probability `locality` a
    token reuses the previous token's expert set, else it draws fresh. Nothing
    here depends on any gate passing."""
    rng = np.random.default_rng(seed)
    out = {}
    for l in range(L):
        rows, prev = [], rng.choice(E, size=k, replace=False)
        for _ in range(T):
            if rng.random() < locality:
                cur = prev.copy()
                j = rng.integers(0, k)
                cur[j] = rng.integers(0, E)
                cur = np.unique(cur)
                while cur.size < k:
                    cur = np.unique(np.append(cur, rng.integers(0, E)))
            else:
                cur = rng.choice(E, size=k, replace=False)
            rows.append(cur[:k])
            prev = cur[:k]
        out[l] = np.array(rows, dtype=np.int16)
    return out, E, L, k


def test_split_capacity_preserves_the_total_slot_budget():
    # Arithmetic, not a result: "10% cache" must mean the same BYTES under
    # per-layer and global scope or the two are not comparable.
    for cap in (0, 1, 7, 51, 102, 128, 1023):
        for L in (1, 3, 16, 61):
            caps = cache_sim.split_capacity(cap, L)
            assert len(caps) == L
            assert sum(caps) == cap
            assert max(caps) - min(caps) <= 1      # even to within one slot


def test_event_stream_and_request_stream_are_the_same_keys():
    # [T, L, k] flattened in C order IS the token-major request stream. If this
    # ever diverges the two replay paths are simulating different workloads.
    tr, E, L, k = _synthetic_trace()
    seq, T1, L1, k1 = cache_sim.request_stream(tr, E)
    ev, T2, L2, k2 = cache_sim.event_stream(tr, E)
    assert (T1, L1, k1) == (T2, L2, k2)
    assert np.array_equal(ev.reshape(-1), seq)


def test_new_replay_reproduces_the_legacy_global_sequential_path():
    # Regression identity. The scope/replay rewrite must not silently change
    # the configuration that was already published; it must ADD configurations.
    tr, E, L, k = _synthetic_trace()
    seq, T, _, _ = cache_sim.request_stream(tr, E)
    ev, _, _, _ = cache_sim.event_stream(tr, E)
    for f in (0.05, 0.2, 0.5):
        cap = max(1, int(round(f * L * E)))
        assert (cache_sim.lru_hits_events(ev, cap, "global", "sequential")
                == cache_sim.lru_hits(seq, cap))
        assert (cache_sim.belady_hits_scoped(ev, cap, "global")
                == cache_sim.belady_hits(seq, cap))


def test_event_atomic_replay_never_scores_below_sequential():
    # A theorem about the two replays, independent of any trace's locality:
    # sequential eviction can discard an expert the CURRENT event still needs
    # and then count it as a miss; atomic decides residency for the whole event
    # first, so it can only ever count more hits.
    for seed in range(4):
        tr, E, L, k = _synthetic_trace(seed=seed, locality=0.3 + 0.2 * seed)
        ev, T, _, _ = cache_sim.event_stream(tr, E)
        for f in (0.03, 0.1, 0.25, 0.6):
            cap = max(1, int(round(f * L * E)))
            for scope in ("per_layer", "global"):
                a = cache_sim.lru_hits_events(ev, cap, scope, "atomic")
                q = cache_sim.lru_hits_events(ev, cap, scope, "sequential")
                assert a >= q, (scope, f, a, q)


def test_lru_on_locality_free_routing_returns_the_cache_fraction():
    # Recovery gate (CLAUDE.md 9.2): with iid uniform routing there is no
    # locality to find, so the correct answer is the cache fraction exactly.
    # A simulator reporting a hit rate ABOVE the floor here is inventing
    # locality, which is the failure this whole rewrite exists to rule out.
    tr, E, L, k = _synthetic_trace(T=4000, locality=0.0)
    ev, T, _, _ = cache_sim.event_stream(tr, E)
    n = T * L * k
    for f in (0.1, 0.25, 0.5):
        cap = max(1, int(round(f * L * E)))
        hit = cache_sim.lru_hits_events(ev, cap, "per_layer", "atomic") / n
        se = math.sqrt(f * (1 - f) / n)
        assert abs(hit - f) < 5 * se + cap / n


def test_belady_is_not_bounded_by_the_cache_fraction():
    # The floor is the null for an ONLINE memoryless policy, not for an offline
    # optimum: Belady exploits the realised sequence and beats the fraction even
    # with no locality at all. Any "Belady minus cache fraction = locality"
    # reading is therefore wrong, and this pins the reason.
    tr, E, L, k = _synthetic_trace(T=4000, locality=0.0)
    ev, T, _, _ = cache_sim.event_stream(tr, E)
    n = T * L * k
    f = 0.1
    cap = max(1, int(round(f * L * E)))
    assert cache_sim.belady_hits_scoped(ev, cap, "per_layer") / n > 1.5 * f


def test_per_layer_belady_never_beats_global_belady():
    # Per-layer quotas are the SAME optimisation problem with an added
    # constraint, so the constrained optimum cannot exceed the free one.
    tr, E, L, k = _synthetic_trace()
    ev, T, _, _ = cache_sim.event_stream(tr, E)
    for f in (0.05, 0.2, 0.5):
        cap = max(1, int(round(f * L * E)))
        assert (cache_sim.belady_hits_scoped(ev, cap, "global")
                >= cache_sim.belady_hits_scoped(ev, cap, "per_layer"))


def test_unbounded_lookahead_is_exactly_belady():
    # An identity, not a result: with the whole future visible, farthest-next-use
    # IS Belady's rule. If these ever diverge the lookahead sweep is measuring
    # something other than an approach to the offline optimum.
    tr, E, L, k = _synthetic_trace()
    ev, T, _, _ = cache_sim.event_stream(tr, E)
    for f in (0.05, 0.2, 0.5):
        cap = max(1, int(round(f * L * E)))
        assert (cache_sim.lookahead_hits(ev, cap, None)
                == cache_sim.belady_hits_scoped(ev, cap, "per_layer"))


def test_lookahead_is_bracketed_by_sequential_lru_and_belady():
    # horizon 0 does not mean "no information": the engine is executing the
    # current token, so it always knows that token's own expert set. So H=0 must
    # beat sequential LRU (which throws that away) and no horizon may beat the
    # offline optimum.
    tr, E, L, k = _synthetic_trace()
    ev, T, _, _ = cache_sim.event_stream(tr, E)
    for f in (0.05, 0.2, 0.5):
        cap = max(1, int(round(f * L * E)))
        seq = cache_sim.lru_hits_events(ev, cap, "per_layer", "sequential")
        bel = cache_sim.belady_hits_scoped(ev, cap, "per_layer")
        for h in (0, 1, 4, 16, None):
            got = cache_sim.lookahead_hits(ev, cap, h)
            assert seq <= got <= bel, (f, h, seq, got, bel)


def test_rank_eviction_is_worse_than_no_lookahead_when_the_predictor_is_wrong():
    # A NEGATIVE result, asserted so it cannot quietly come back. Ranking victims
    # by a predicted next-use discards recency, which is itself information. With
    # a uniformly wrong predictor the rule therefore scores BELOW the
    # no-lookahead rule. Any change that makes this test fail has either fixed
    # the policy or broken the noise model, and both need saying out loud.
    tr, E, L, k = _synthetic_trace(T=2500)
    ev, T, _, _ = cache_sim.event_stream(tr, E)
    for f in (0.1, 0.25):
        cap = max(1, int(round(f * L * E)))
        base = cache_sim.lookahead_hits(ev, cap, 0, mode="rank")
        blind = cache_sim.lookahead_hits(
            ev, cap, 4, ev_hat=cache_sim.corrupt_routing(ev, 0.0, E, seed=1),
            mode="rank")
        assert blind < base, (f, blind, base)


def test_protect_eviction_degrades_gracefully_when_the_predictor_is_wrong():
    # The property that makes a CHEAP predictor shippable: if the predictor may
    # only veto a candidate and recency still ranks, then a wrong prediction
    # removes the veto's value without removing the ranking signal underneath.
    # So protect must stay near the no-lookahead rule where rank collapses, and
    # it must never beat rank when the lookahead is exact.
    tr, E, L, k = _synthetic_trace(T=2500)
    ev, T, _, _ = cache_sim.event_stream(tr, E)
    for f in (0.1, 0.25):
        cap = max(1, int(round(f * L * E)))
        base = cache_sim.lookahead_hits(ev, cap, 0, mode="rank")
        hat0 = cache_sim.corrupt_routing(ev, 0.0, E, seed=1)
        blind_rank = cache_sim.lookahead_hits(ev, cap, 4, ev_hat=hat0, mode="rank")
        blind_prot = cache_sim.lookahead_hits(ev, cap, 4, ev_hat=hat0, mode="protect")
        assert blind_prot > blind_rank, (f, blind_prot, blind_rank)
        assert blind_prot >= 0.7 * base, (f, blind_prot, base)
        exact_rank = cache_sim.lookahead_hits(ev, cap, 4, mode="rank")
        exact_prot = cache_sim.lookahead_hits(ev, cap, 4, mode="protect")
        assert exact_prot <= exact_rank, (f, exact_prot, exact_rank)


def test_protect_mode_is_still_bounded_by_belady():
    # Whatever the mode, no online rule may beat the offline optimum.
    tr, E, L, k = _synthetic_trace()
    ev, T, _, _ = cache_sim.event_stream(tr, E)
    for f in (0.05, 0.2):
        cap = max(1, int(round(f * L * E)))
        bel = cache_sim.belady_hits_scoped(ev, cap, "per_layer")
        for h in (0, 4, None):
            assert cache_sim.lookahead_hits(ev, cap, h, mode="protect") <= bel


def test_corrupt_routing_keeps_every_expert_inside_its_own_layer():
    # A corrupted id that escaped its layer would be a key no layer ever
    # requests, which would silently inflate the measured miss count.
    tr, E, L, k = _synthetic_trace()
    ev, T, _, _ = cache_sim.event_stream(tr, E)
    for acc in (0.0, 0.5, 1.0):
        hat = cache_sim.corrupt_routing(ev, acc, E, seed=2)
        assert hat.shape == ev.shape
        for l in range(L):
            lo, hi = l * E, (l + 1) * E - 1
            assert hat[:, l, :].min() >= lo and hat[:, l, :].max() <= hi
        if acc >= 1.0:
            assert np.array_equal(hat, ev)


def test_unbounded_lookahead_is_belady_under_global_scope_too():
    # The same identity must hold for the shared-pool machine, or the two scopes
    # are not being compared on equal terms. This caught a real defect: the
    # global branch originally built L independent full-size caches instead of
    # one pool, which silently made "global" mean "L times the memory".
    tr, E, L, k = _synthetic_trace()
    ev, T, _, _ = cache_sim.event_stream(tr, E)
    for f in (0.05, 0.2, 0.5):
        cap = max(1, int(round(f * L * E)))
        assert (cache_sim.lookahead_hits(ev, cap, None, scope="global")
                == cache_sim.belady_hits_scoped(ev, cap, "global"))


def test_lookahead_rejects_an_unknown_scope():
    # An unrecognised scope must fail loudly rather than fall through to a
    # default, because silently simulating the other machine is exactly the
    # defect this module was rewritten to remove.
    tr, E, L, k = _synthetic_trace(T=200)
    ev, T, _, _ = cache_sim.event_stream(tr, E)
    with pytest.raises(ValueError):
        cache_sim.lookahead_hits(ev, 16, 4, scope="per-layer")   # note the hyphen


def test_lookahead_is_monotone_in_horizon():
    # Monotonicity gate (CLAUDE.md 9.2): strictly more future information must
    # not produce a worse cache. A non-monotone curve means the eviction rule is
    # mis-ranking victims, not that longer lookahead is unhelpful.
    for seed in range(3):
        tr, E, L, k = _synthetic_trace(seed=seed)
        ev, T, _, _ = cache_sim.event_stream(tr, E)
        for f in (0.05, 0.2):
            cap = max(1, int(round(f * L * E)))
            xs = [cache_sim.lookahead_hits(ev, cap, h) for h in (0, 1, 2, 4, 8, None)]
            assert all(a <= b for a, b in zip(xs, xs[1:])), (seed, f, xs)


def test_window_verification_never_increases_fetches_per_verified_token():
    # Unioning W tokens' expert sets fetches each distinct expert once, so the
    # fetch count per VERIFIED token is non-increasing in W. (Per ACCEPTED token
    # it can increase -- that depends on the acceptance rate and is priced by
    # the caller, not here.)
    tr, E, L, k = _synthetic_trace(T=1200)
    ev, T, _, _ = cache_sim.event_stream(tr, E)
    for f in (0.05, 0.2):
        cap = max(1, int(round(f * L * E)))
        prev = None
        for W in (1, 2, 4, 8):
            fetch, nwin = cache_sim.window_fetches(ev, cap, W)
            per_tok = fetch / (nwin * W)
            if prev is not None:
                assert per_tok <= prev + 1e-9, (f, W, per_tok, prev)
            prev = per_tok


def test_shuffle_control_preserves_expert_popularity_exactly():
    # The control must isolate ONE factor. Shuffling token order has to leave
    # every layer's expert-frequency histogram bit-identical, or the real-minus-
    # shuffled gap conflates recency with popularity.
    tr, E, L, k = _synthetic_trace()
    sh = cache_sim.shuffle_control(tr, seed=3)
    for l in tr:
        a = np.bincount(tr[l].ravel().astype(np.int64), minlength=E)
        b = np.bincount(sh[l].ravel().astype(np.int64), minlength=E)
        assert np.array_equal(a, b)


def test_static_pinning_is_scored_only_on_held_out_tokens():
    # A policy fitted on a warmup slice and scored on the same slice reports its
    # own training fit. Scoring on the remainder must give a DIFFERENT number;
    # equality would mean the split is not being applied.
    import expert_policy   # gates/ is already on sys.path (see the header)
    tr, E, L, k = _synthetic_trace(T=2000)
    caps = cache_sim.split_capacity(max(1, int(round(0.2 * L * E))), L)
    oracle, _ = expert_policy.static_hits(tr, caps)
    heldout, _ = expert_policy.static_hits(tr, caps, warmup_frac=0.25)
    assert heldout != oracle
    assert heldout <= oracle + 1e-9      # no-peeking cannot beat whole-trace fit


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

def test_a_persistence_predictor_adds_nothing_to_what_the_engine_knows():
    """S12's mechanism, as a property rather than a measurement.

    The eviction rule already knows the expert set of the token it is executing.
    A persistence predictor asserts the next token wants that same set, so it
    supplies no information the rule did not have, and must reproduce the
    horizon-0 result exactly. This holds for any trace: it is a statement about
    what the two rules can see, not about how much locality a trace has.

    If this ever fails, either the horizon-0 branch has stopped using the current
    token's true routing, or persistence has stopped being persistence."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gates"))
    import predictor
    tr, E, L, k = _synthetic_trace(T=1200)
    ev, T, _, _ = cache_sim.event_stream(tr, E)
    hat = predictor.predict_persistence(ev, E, split=0)
    for f in (0.05, 0.1, 0.3):
        cap = max(1, int(round(f * L * E)))
        empty = cache_sim.lookahead_hits(ev, cap, 1, ev_hat=hat, mode="protect")
        knows_now = cache_sim.lookahead_hits(ev, cap, 0, mode="protect")
        assert empty == knows_now, (f, empty, knows_now)


def test_uniform_corruption_is_not_a_flattering_noise_model():
    """A REJECTED hypothesis, kept as a test so it cannot be re-adopted.

    It was proposed that the abstract accuracy sweep flatters a predictor,
    because corrupt_routing() replaces a wrong guess with a uniformly random
    expert -- rarely resident, so a wrong veto rarely fires -- whereas a real
    predictor's errors are plausible experts that often ARE resident. If that
    held, corrupting from the trace's own routing would score materially worse
    at matched accuracy. It does not: the two agree closely, so the noise model
    is not what makes measured predictors underperform. (Horizon is; see
    test_horizon_not_accuracy_is_what_a_one_step_predictor_lacks.)"""
    tr, E, L, k = _synthetic_trace(T=1500)
    ev, T, _, _ = cache_sim.event_stream(tr, E)
    rng = np.random.default_rng(5)
    uniform = cache_sim.corrupt_routing(ev, 0.5, E, seed=5)
    plausible = ev.copy()                      # same accuracy, plausible errors
    flip = rng.random(ev.shape) >= 0.5
    donor = ev[rng.permutation(T)]
    plausible[flip] = donor[flip]
    cap = max(1, int(round(0.1 * L * E)))
    u = cache_sim.lookahead_hits(ev, cap, 4, ev_hat=uniform, mode="protect")
    pl = cache_sim.lookahead_hits(ev, cap, 4, ev_hat=plausible, mode="protect")
    assert abs(pl - u) / max(1, u) < 0.05, (pl, u)


def test_horizon_not_accuracy_is_what_a_one_step_predictor_lacks():
    """The supported explanation, as a property.

    A longer horizon must be worth more than a shorter one at the SAME
    predictor accuracy -- more future is more information, whatever its quality.
    That is why a one-step predictor cannot reach the lookahead lever: the lever
    needs horizon ~4, and one step supplies one."""
    tr, E, L, k = _synthetic_trace(T=1500)
    ev, T, _, _ = cache_sim.event_stream(tr, E)
    cap = max(1, int(round(0.1 * L * E)))
    for acc in (1.0, 0.5, 0.3):
        hat = None if acc >= 1.0 else cache_sim.corrupt_routing(ev, acc, E, seed=11)
        h1 = cache_sim.lookahead_hits(ev, cap, 1, ev_hat=hat, mode="protect")
        h4 = cache_sim.lookahead_hits(ev, cap, 4, ev_hat=hat, mode="protect")
        assert h4 >= h1, (acc, h1, h4)


def test_every_documented_number_matches_its_artifact():
    """CLAUDE.md 7.1: one claim, one script, one artifact. gates/claims_check.py
    holds the map; this asserts every entry still agrees with the artifact it
    names. The expected values are what the DOCUMENTS say, so re-running a gate
    and forgetting to update the prose fails here instead of surviving into a
    paper. It caught a hand-typed 0.490 where the artifact said 0.5015."""
    import claims_check
    _, out, bad = claims_check.check()
    assert out, "the claims map is empty"
    assert not bad, "documents disagree with their artifacts: " + repr(bad)


def test_claims_markdown_is_regenerable():
    """CLAIMS.md is generated, not written. If it drifts from what the checker
    emits, it is being edited by hand and has stopped being evidence."""
    import claims_check
    _, out, _ = claims_check.check()
    path = os.path.join(os.path.dirname(HERE), "CLAIMS.md")
    with open(path, encoding="utf-8") as f:
        current = f.read()
    assert current == claims_check.emit_markdown(out), (
        "moe-phone/CLAIMS.md is out of date; regenerate with "
        "python moe-phone/gates/claims_check.py --emit")


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


def test_prefetch_sim_without_prefetch_is_exactly_atomic_per_layer_lru():
    """An identity, not a result: with prefetching off, prefetch_sim's replay IS the
    engine's per-layer, event-atomic LRU, so its hit rate must equal cache_sim's to
    the last access. If this breaks, the prefetch comparison is between two machines."""
    import prefetch_sim
    tr, E, L, k = _synthetic_trace(T=800)
    ev, T, _, _ = cache_sim.event_stream(tr, E)
    for f in (0.1, 0.3):
        cap = max(1, int(round(f * L * E)))
        got = prefetch_sim.replay(ev, E, cap, {}, 0.0, 1e-3, prefetch=False)["hit_rate"]
        ref = cache_sim.lru_hits_events(ev, cap, "per_layer", "atomic") / (T * L * k)
        assert got == ref, (f, got, ref)


def test_prefetch_with_zero_recall_never_reduces_flash_reads():
    """A predictor that is always wrong can only ADD reads (wasted fetches plus the
    evictions they cause); it cannot make demand traffic fall below no prefetch."""
    import prefetch_sim
    tr, E, L, k = _synthetic_trace(T=800)
    ev, T, _, _ = cache_sim.event_stream(tr, E)
    recall0 = {l: 0.0 for l in range(L)}
    cap = max(1, int(round(0.25 * L * E)))
    base = prefetch_sim.replay(ev, E, cap, recall0, 0.0, 1e-3, prefetch=False)
    pf = prefetch_sim.replay(ev, E, cap, recall0, 0.0, 1e-3, prefetch=True)
    assert pf["flash_reads_per_tok"] >= base["flash_reads_per_tok"]


# ---- platform design documents: every number matches its artifact -----------

def test_platform_documents_do_not_drift_from_their_artifacts():
    """CLAUDE.md §7.1 for the platform/ design documents.

    Each number in platform/**.md is written as `value [E:id]` and read from an
    artifact by platform/tools/evidence.py. This asserts the documents still agree
    with the artifacts, so re-running a gate and forgetting to update a design
    document is a test failure rather than a stale number in a handed-off spec.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "platform", "tools"))
    import evidence

    evidence.CLAIMS = evidence._claims()
    table = evidence.build()
    bad, seen = evidence.check(table)
    assert not bad, "platform documents drifted from their artifacts:\n" + "\n".join(bad)
    assert seen, "no [E:id] markers found — the check is not actually running"


def test_platform_evidence_table_is_regenerable_and_current():
    """EVIDENCE.md is generated, so it must match what the generator produces now."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "platform", "tools"))
    import evidence

    evidence.CLAIMS = evidence._claims()
    fresh = evidence.emit(evidence.build())
    with open(os.path.join(os.path.dirname(HERE), "platform", "EVIDENCE.md"),
              encoding="utf-8") as f:
        assert f.read() == fresh, "platform/EVIDENCE.md is stale: re-run evidence.py --emit"


def test_platform_internal_links_resolve():
    """A handed-off document set whose cross-references are broken is not handed off."""
    import re
    root = os.path.join(os.path.dirname(HERE), "platform")
    broken = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".md"):
            continue
        with open(os.path.join(root, name), encoding="utf-8") as f:
            text = f.read()
        for m in re.finditer(r"\]\(([^)#][^)]*)\)", text):
            target = m.group(1)
            if target.startswith("http"):
                continue
            if not os.path.exists(os.path.normpath(os.path.join(root, target))):
                broken.append("%s -> %s" % (name, target))
    assert not broken, "broken links:\n" + "\n".join(broken)
