"""Zufallsstrom: SplitMix64 gegen die veröffentlichten Referenzwerte, Gleichverteilung grob, Reihen bitgleich je Zufalls-Seed."""

import numpy as np

import lfk_scenario as S
from lfk_rng import SplitMix64


def test_splitmix64_reference_values_for_seed_0():
    """Die ersten drei Ausgaben von SplitMix64 mit Zustand 0 (Referenzimplementierung von Vigna, in vielen Bibliotheken als Testwert veröffentlicht)."""
    r = SplitMix64(0)
    assert [r.next_u64() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_random_is_in_the_half_open_unit_interval_and_roughly_uniform():
    r = SplitMix64(123)
    u = np.array([r.random() for _ in range(20000)])
    assert u.min() >= 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 4.5 * np.sqrt(1 / 12 / len(u))


def test_same_seed_same_series_and_different_seed_different_series():
    a = S.make_series(7, 50, 20, 1.0)
    assert np.array_equal(a, S.make_series(7, 50, 20, 1.0))
    assert not np.array_equal(a, S.make_series(8, 50, 20, 1.0))


def test_the_jump_shifts_the_same_draw_after_nu_observations():
    base = S.make_series(9, 60, 0, 0.0)
    jumped = S.make_series(9, 60, 25, 1.5)
    assert np.array_equal(jumped[:25], base[:25])
    np.testing.assert_allclose(jumped[25:], base[25:] + 1.5, rtol=0, atol=1e-15)


def test_simulation_is_reproducible_for_a_seed():
    a = S.simulate_run_lengths(2, 0.0, 30, 5)
    assert np.array_equal(a, S.simulate_run_lengths(2, 0.0, 30, 5))
    assert not np.array_equal(a, S.simulate_run_lengths(2, 0.0, 30, 6))
