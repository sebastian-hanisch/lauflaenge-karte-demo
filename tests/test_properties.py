"""Eigenschaften (je 20 bis 50 Zufallsinstanzen mit festem Zufalls-Seed) und Gegenproben für Absolutaussagen (ORACLE-PLAYBOOK §6)."""

import math

import numpy as np
import pytest

import lfk_chain as H
import lfk_chart as K
import lfk_evaluation as E
from helpers_lfk import reference_first_alarm

RNG = np.random.default_rng(2026)
SETS = (1, 2, 3, 4)


def test_cdf_grows_in_t_and_arl0_grows_in_c():
    for _ in range(40):
        c = float(RNG.uniform(1.5, 3.8))
        two = bool(RNG.integers(0, 2))
        rl = E.run_length(1, two, c, 8, 0.0)
        t = np.sort(RNG.integers(0, 3000, size=12))
        cdf = [rl.cdf(int(v)) for v in t]
        assert all(b >= a - 1e-15 for a, b in zip(cdf, cdf[1:]))
        c2 = c + float(RNG.uniform(0.05, 0.5))
        assert E.run_length(1, two, c2, 8, 0.0).mean > rl.mean
        assert E.run_length(1, two, c2, 8, 0.0).cdf(100) < rl.cdf(100)


def test_arl1_falls_in_delta_and_is_even_in_delta_for_the_two_sided_card():
    for rules in SETS:
        chain = H.build_chain(rules)
        values = [float(H.arl_vector(chain, d)[0]) for d in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)]
        assert all(b < a for a, b in zip(values, values[1:])), rules
        for d in (0.5, 1.0, 2.5):                     # Spiegelung: die Zonen und Regeln sind symmetrisch
            assert H.arl_vector(chain, -d)[0] == pytest.approx(H.arl_vector(chain, d)[0], rel=1e-9)


def test_the_rule_sets_are_nested_arl_does_not_grow_with_more_rules():
    """Kopplung: jeder Alarm des kleineren Satzes ist einer des größeren, also ist die Lauflänge des größeren Satzes nicht länger."""
    for delta in (0.0, 0.5, 1.0, 2.0):
        arl = [float(H.arl_vector(H.build_chain(r), delta)[0]) for r in SETS]
        assert all(b <= a + 1e-12 for a, b in zip(arl, arl[1:])), (delta, arl)
    for delta in (0.0, 1.0):
        a8 = H.arl_vector(H.build_chain(4, 8), delta)[0]
        a9 = H.arl_vector(H.build_chain(4, 9), delta)[0]
        a10 = H.arl_vector(H.build_chain(4, 10), delta)[0]
        a_wo = H.arl_vector(H.build_chain(3), delta)[0]
        assert a8 < a9 < a10 < a_wo                   # weniger oft auslösende Regel 4: größere ARL


def test_the_rule_sets_are_nested_on_random_series():
    for _ in range(300):
        x = RNG.normal(loc=float(RNG.choice([0.0, 1.0])), size=60)
        first = [reference_first_alarm(x, r) for r in SETS]
        times = [f[0] if f else 10**9 for f in first]
        assert all(b <= a for a, b in zip(times, times[1:]))


def test_the_rules_are_mirror_symmetric():
    for rules in SETS:
        for _ in range(60):
            x = RNG.normal(loc=float(RNG.choice([0.0, 1.0, 2.0])), size=50)
            assert K.first_alarm(K.rule_flags(x, rules)) == K.first_alarm(K.rule_flags(-x, rules))


def test_the_probability_function_sums_to_one_and_the_tail_is_negligible():
    for rules in SETS:
        for delta in (0.0, 0.5, 2.0):
            rl = E.run_length(rules, True, 3.0, 8, delta)
            s = rl.s if rl.s is not None else rl.survival_curve(H.tail_length(rl.mean))
            assert s[0] == pytest.approx(1.0, abs=1e-15)
            assert s[-1] < 1e-12                       # Rest der berechneten Verteilung
            assert np.all(np.diff(s) <= 1e-15)         # Überlebensfunktion fällt
            assert float(s.sum()) == pytest.approx(rl.mean, rel=1e-9)


def test_expected_alarm_count_is_close_to_window_over_arl0():
    """Erneuerungsrechnung gegen W/ARL₀: bei W = 1000 höchstens 0,5 % Abweichung, bei W = 200 höchstens 2 % (gemessen: bis 1,5 % bei vier Regeln)."""
    for rules in SETS:
        for window, tol in ((200, 0.02), (1000, 0.005)):
            cd = E.count_distribution(rules, True, 3.0, 8, window)
            assert abs(cd["mean"] - cd["t_over_arl0"]) <= tol * cd["t_over_arl0"] + 1e-12, (rules, window)


def test_median_of_the_simple_card_is_about_ln2_times_arl0():
    for _ in range(40):
        p = float(RNG.uniform(1e-4, 0.05))
        med = K.geometric_quantile(p, 0.5)
        assert abs(med - math.log(2) / p) <= 1.0 + 1e-9    # exakt bis auf die Ganzzahl-Rundung (kleinstes t)


def test_scale_invariance_of_the_simple_card():
    """Nur die einfache Karte: Reihe und Schwelle mit demselben Faktor geben dieselben Alarme (die Zonen der Zusatzregeln sind fest bei 1σ, 2σ, 3σ)."""
    for _ in range(60):
        x = RNG.normal(size=80)
        c = float(RNG.uniform(1.5, 3.5))
        f = float(RNG.uniform(0.2, 5.0))
        a = np.flatnonzero(K.rule_flags(x, 1, c=c)[1])
        b = np.flatnonzero(K.rule_flags(x * f, 1, c=c * f)[1])
        assert np.array_equal(a, b)


def test_state_distribution_is_a_distribution_and_mass_equals_survival():
    chain = H.build_chain(4)
    s = H.survival(chain, 0.0, 40)
    for nu in (0, 1, 7, 40):
        v, mass = H.state_distribution(chain, nu)
        assert v.sum() == pytest.approx(1.0, abs=1e-12) and mass == pytest.approx(s[nu], rel=1e-12)


def test_alarm_probability_properties():
    for _ in range(40):
        c = float(RNG.uniform(1.0, 4.0))
        d = float(RNG.uniform(0.0, 3.0))
        assert K.p_alarm(c, d, True) >= K.p_alarm(c, d, False)           # zweiseitig alarmiert nie seltener
        assert K.p_alarm(c, d + 0.5, False) > K.p_alarm(c, d, False)     # größerer Sprung, mehr Alarme
        assert K.p_alarm(c + 0.3, d, True) < K.p_alarm(c, d, True)       # engere Grenze, mehr Alarme
        assert K.p_alarm(c, d, True) == pytest.approx(K.p_alarm(c, -d, True), rel=1e-12)


def test_geometric_claim_is_exact_and_the_rule_sets_are_only_nearly_geometric():
    """Gegenprobe für die Absolutaussage: exakt geometrisch ist nur die einfache Karte; bei den Zusatzregeln gilt der größte Abstand 0,0025 / 0,0134 / 0,0344."""
    rl1 = E.run_length(1, True, 3.0, 8, 0.0)
    assert H.max_distance_to_geometric(rl1.survival_curve(3000), rl1.mean) < 1e-14
    gaps = []
    for r in (2, 3, 4):
        rl = E.run_length(r, True, 3.0, 8, 0.0)
        gaps.append(H.max_distance_to_geometric(rl.s, rl.mean))
    assert gaps == pytest.approx([0.0025, 0.0134, 0.0344], abs=6e-5)
    assert all(g > 1e-3 for g in gaps)             # nicht geometrisch im strengen Sinn
