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
       "errors,first_errno,power_w,temp_start_c,temp_end_c")


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
        "1,direct,rand,4,1,1,1,450,1,1,1,0,0,nan,nan,nan",
        "1,direct,rand,512,1,1,1,3500,1,1,1,0,0,nan,nan,nan",
        "2,direct,rand,512,1,1,1,9000,1,1,1,0,0,nan,nan,nan",  # impossible: cache-served
    ])
    res = g1_analyze.analyse(rows, True)
    assert res["n_invalid_above_interface_ceiling"] == 1
    assert res["bulk_rand_best"]["MBps_median"] == 3500
    assert res["small_4k_rand_best"]["MBps_median"] == 450
    assert res["bulk_to_small_ratio"] == pytest.approx(3500 / 450)


def test_energy_subtracts_idle_power():
    rows = _rows([
        "0,idle,none,0,0,0,5,0,0,0,0,0,0,1.0,30,30",
        "1,direct,rand,512,1,1,1,2000,1,1,1,0,0,3.0,30,31",
    ])
    res = g1_analyze.analyse(rows, True)
    # (3 W - 1 W) / 2 GB/s = 1 J/GB
    assert res["bulk_rand_best"]["J_per_GB_above_idle"] == pytest.approx(1.0)


# ---- G2/G3 instrumentation (needs torch + transformers) --------------------

def test_traces_sparsity_selftest():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import traces_sparsity
    assert traces_sparsity.selftest() == 0
