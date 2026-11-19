"""Handbeispiele als exakte Tests (Brüche, Ganzzahlen): jede Erwartung ist von Hand nachgerechnet, nicht aus dem Code übernommen (DEMO-PLAYBOOK §5).

Teil 1: einfache Karte mit ARL₀ = 370 gesetzt (p = 1/370). Teil 2 und 3: Mini-Karte mit drei Zuständen, ARL₀ als Bruch. Teil 4: Verzögerungsmaße der
Mini-Karte. Teil 5: Alarme je Fenster mit Neustart. Die Mini-Karte läuft durch die Löser der App (lfk_chain), die Gegenrechnung steht hier im Test
(Bruch-Gauß, Aufzählung aller Folgen, Matrixpotenzen, geschlossene Form)."""

import itertools
import math
from fractions import Fraction as Fr

import numpy as np
import pytest

import lfk_chain as H
import lfk_chart as K
import lfk_renewal as R
from helpers_lfk import mini_chain_normal


# ------------------------------------------------------------------------------------------------------------------------- Teil 1
def test_h1_p_as_fraction_and_first_two_probabilities():
    p = Fr(1, 370)
    assert p * (1 - p) == Fr(369, 370 * 370)
    assert p == Fr(1, 370)


def test_h1_probability_within_the_first_100_observations_as_exact_fraction():
    p = Fr(1, 370)
    q = 1 - p
    f100 = 1 - q**100
    assert abs(float(f100) - 0.237106) < 1e-6
    assert f100 == sum(q ** (i - 1) * p for i in range(1, 101))   # zweiter Weg: Summe der Wahrscheinlichkeiten P(T = 1..100), exakter Bruch
    assert abs(float(1 - q**99) - 0.235039) < 1e-6              # „vor Beobachtung 100“ im engen Sinn: T <= 99
    assert abs(float(K.geometric_sf(1 / 370, 100)) - float(q**100)) < 1e-14   # die App rechnet in Gleitkomma dasselbe


def _smallest_t(num_over: int, den_over: int) -> int:
    """Kleinstes t mit (369/370)^t <= num/den, reine Ganzzahlrechnung (P(T > t) <= num/den)."""
    t = 1
    while den_over * 369**t > num_over * 370**t:
        t += 1
    return t


def test_h1_median_and_quantiles_by_integer_arithmetic():
    assert _smallest_t(1, 2) == 257
    assert _smallest_t(95, 100) == 19      # 5-%-Quantil
    assert _smallest_t(5, 100) == 1107     # 95-%-Quantil
    assert _smallest_t(1, 100) == 1702     # 99-%-Quantil
    for q, expected in ((0.5, 257), (0.05, 19), (0.95, 1107), (0.99, 1702)):
        assert K.geometric_quantile(1 / 370, q) == expected   # die App trifft dieselben Ganzzahlen


def test_h1_quantiles_of_the_3_sigma_card():
    p3 = K.p_alarm(3.0, 0.0, True)
    assert [K.geometric_quantile(p3, q) for q in (0.05, 0.5, 0.95, 0.99)] == [19, 257, 1109, 1704]


def test_h1_expected_value_by_exact_finite_sum():
    p = Fr(1, 370)
    q = 1 - p
    big_t = 50
    direct = sum(Fr(t) * p * q ** (t - 1) for t in range(1, big_t + 1))
    assert direct == (1 - q**big_t * (1 + big_t * p)) / p
    assert float((1 - q**10000 * (1 + 10000 * p)) / p) > 369.99999   # Grenzwert 370


def test_h1_no_alarm_in_1000_observations():
    q = 1 - Fr(1, 370)
    assert abs(float(q**1000) - 0.06678) < 1e-4
    assert abs(float(1 - q**1000) - 0.93322) < 1e-4


@pytest.mark.parametrize("arl0,expected", [(50, 0.8674), (100, 0.6340), (200, 0.3942), (370, 0.2371), (1000, 0.0952)])
def test_h1_dependence_on_arl0(arl0, expected):
    exact = float(1 - (1 - Fr(1, arl0)) ** 100)
    assert abs(exact - expected) < 1e-4
    assert abs(1 - float(K.geometric_sf(1 / arl0, 100)) - expected) < 1e-4


def test_h1_the_real_3_sigma_card():
    p3 = 2 * 0.5 * math.erfc(3.0 / math.sqrt(2.0))      # 2 Phi(-3) ohne scipy
    assert abs(p3 - 0.0026997961) < 1e-9 and abs(1 / p3 - 370.3983) < 1e-3
    assert abs(1 - (1 - p3) ** 100 - 0.23688) < 1e-5
    assert abs(K.p_alarm(3.0, 0.0, True) - p3) < 1e-15
    pd = K.p_alarm(3.0, 1.0, True)
    assert abs(1 / pd - 43.8947) < 1e-3                  # Sprung δ = 1


# ------------------------------------------------------------------------------------------------------------------------- Teil 2
A, B, C = Fr(1, 100), Fr(1, 10), Fr(89, 100)
RMINI = [[C, B, Fr(0)], [Fr(0), Fr(0), C], [C, Fr(0), Fr(0)]]   # S0, S1, S2 -> S0, S1, S2 ohne Alarm


def _solve_fractions(a_mat, rhs):
    n = len(a_mat)
    m = [row[:] + [rhs[i]] for i, row in enumerate(a_mat)]
    for i in range(n):
        piv = next(r for r in range(i, n) if m[r][i] != 0)
        m[i], m[piv] = m[piv], m[i]
        m[i] = [v / m[i][i] for v in m[i]]
        for r in range(n):
            if r != i and m[r][i] != 0:
                f = m[r][i]
                m[r] = [vr - f * vi for vr, vi in zip(m[r], m[i])]
    return [m[i][n] for i in range(n)]


def test_h2_arl_of_the_mini_card_closed_form_and_fraction_gauss():
    closed = (1 + B + B * C) / (1 - C - B * C * C)       # aus x2 = 1 + c x0, x1 = 1 + c x2, x0 = 1 + c x0 + b x1
    ir = [[(1 if i == j else 0) - RMINI[i][j] for j in range(3)] for i in range(3)]
    x = _solve_fractions(ir, [Fr(1)] * 3)
    assert x[0] == closed == Fr(118900, 3079)
    assert abs(float(closed) - 38.6164) < 1e-4


def test_h2_the_apps_solver_gives_the_same_arl(mini_fractions):
    arl = H.arl_vector(mini_fractions, 0.0)
    assert arl[0] == pytest.approx(118900 / 3079, rel=1e-12)


def _first_alarm_in_sequence(seq):
    for t in range(len(seq)):
        if seq[t] == "A":
            return t + 1
        if seq[max(0, t - 2): t + 1].count("B") >= 2:
            return t + 1
    return None


def test_h2_distribution_by_enumeration_matrix_powers_and_by_hand(mini_fractions):
    length = 8
    pr = {"A": A, "B": B, "C": C}
    pmf_enum = [Fr(0)] * (length + 1)
    for seq in itertools.product("ABC", repeat=length):       # alle 3^8 Folgen, Regeln direkt an den Folgen
        t = _first_alarm_in_sequence(seq)
        if t is not None:
            w = Fr(1)
            for ch in seq:
                w *= pr[ch]
            pmf_enum[t] += w
    v = [Fr(1), Fr(0), Fr(0)]
    pmf_mat = [Fr(0)] * (length + 1)
    out = [1 - sum(RMINI[i]) for i in range(3)]
    for t in range(1, length + 1):
        pmf_mat[t] = sum(v[i] * out[i] for i in range(3))
        v = [sum(v[i] * RMINI[i][j] for i in range(3)) for j in range(3)]
    assert pmf_enum == pmf_mat
    assert pmf_mat[1] == Fr(1, 100)
    assert pmf_mat[2] == B * (A + B) + C * A == Fr(199, 10000)
    assert pmf_mat[3] == C * C * A + C * B * (A + B) + B * C * (A + B) == Fr(27501, 1000000)
    s = H.survival(mini_fractions, 0.0, length)                # die Löser der App
    pmf_app = s[:-1] - s[1:]
    for t in range(1, length + 1):
        assert pmf_app[t - 1] == pytest.approx(float(pmf_mat[t]), rel=1e-12)


def test_h2_the_distribution_of_the_mini_card(mini_fractions):
    s = H.survival(mini_fractions, 0.0, 4000)
    f = 1 - s
    assert H.quantile(s, 0.5) == 27 and H.quantile(s, 0.05) == 3 and H.quantile(s, 0.95) == 113
    assert f[100] == pytest.approx(0.9299, abs=5e-5)
    assert s.sum() == pytest.approx(118900 / 3079, rel=1e-9) and s[-1] < 1e-12   # Summe der Überlebenswahrscheinlichkeiten = ARL
    assert 0.65 < H.quantile(s, 0.5) / (118900 / 3079) < 0.72                    # Median/Mittel nahe ln 2 (0,693), hier 0,70
    assert H.max_distance_to_geometric(s, 118900 / 3079) < 0.05                  # nahezu geometrisch


# ------------------------------------------------------------------------------------------------------------------------- Teil 3
def test_h3_mini_card_with_normal_zones():
    a_n, b_n, c_n = K.zone_probs(0.0)[7], K.zone_probs(0.0)[6], K.phi(2.0)
    assert a_n + b_n + c_n == pytest.approx(1.0, abs=1e-15)
    assert 1 / a_n == pytest.approx(740.80, abs=0.01)                             # nur Regel 1, einseitig
    arl = H.arl_vector(mini_chain_normal(), 0.0)[0]
    closed = (1 + b_n + b_n * c_n) / (1 - c_n - b_n * c_n**2)
    assert arl == pytest.approx(closed, rel=1e-12)
    assert arl == pytest.approx(450.72, abs=0.01)                                  # Regel 1+2 einseitig
    assert arl < 1 / a_n                                                           # eine Zusatzregel senkt die ARL₀
    assert arl / 2 == pytest.approx(225.4384, abs=0.2)   # Halbierung trifft die exakte zweiseitige Kette hier auf 0,1: eine Beobachtung, kein Satz
    assert H.arl_vector(H.build_chain(2), 0.0)[0] == pytest.approx(225.4384, abs=1e-4)


# ------------------------------------------------------------------------------------------------------------------------- Teil 4
@pytest.mark.parametrize("delta,zero,steady,states", [(1.0, 20.0189, 19.8926, (20.02, 16.01, 17.84)), (2.0, 3.6464, 3.6044, None)])
def test_h4_delay_measures_of_the_mini_card(delta, zero, steady, states):
    chain = mini_chain_normal()
    m = H.delay_measures(chain, delta)
    assert m.zero == pytest.approx(zero, abs=5e-5)
    assert m.steady == pytest.approx(steady, abs=5e-5)
    assert m.pollak == pytest.approx(zero, abs=5e-5) and m.lorden == pytest.approx(zero, abs=5e-5)
    assert m.pollak <= m.lorden + 1e-9 and m.steady < m.zero
    if states:
        np.testing.assert_allclose(np.round(H.arl_vector(chain, delta), 2), states, atol=0.006)   # Zustandswerte: S0 ist der schlechteste Zustand
    _, lam, _ = H.quasi_stationary(chain)
    assert lam < 1 and lam == pytest.approx(1 - 1 / H.arl_vector(chain, 0.0)[0], abs=2e-3)       # dominanter Eigenwert nahe 1 - 1/ARL₀


def test_h4_rule_1_alone_all_measures_equal():
    assert 1 / K.p_alarm(3.0, 1.0, False) == pytest.approx(1 / (1 - K.phi(2.0)), rel=1e-12)       # einseitig 3σ bei δ = 1 gleich 1/(1 - Φ(2))


# ------------------------------------------------------------------------------------------------------------------------- Teil 5
def test_h5_shewhart_window_enumeration_renewal_and_binomial_agree():
    pp, tn = Fr(1, 10), 10
    enum = [Fr(0)] * (tn + 1)
    for seq in itertools.product((0, 1), repeat=tn):          # alle 2^10 Folgen
        w = Fr(1)
        for s_ in seq:
            w *= pp if s_ else 1 - pp
        enum[sum(seq)] += w
    bino = [Fr(math.comb(tn, k)) * pp**k * (1 - pp) ** (tn - k) for k in range(tn + 1)]
    assert enum == bino
    assert abs(float(bino[0]) - 0.3487) < 5e-5 and sum(k * bino[k] for k in range(tn + 1)) == tn * pp
    f = np.array([0.0] + [float((1 - pp) ** (t - 1) * pp) for t in range(1, tn + 1)])
    pmf, _ = R.renewal_counts(f, tn)
    np.testing.assert_allclose(pmf[: tn + 1], [float(b) for b in bino], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(R.binomial_counts(tn, 0.1), [float(b) for b in bino], rtol=1e-12, atol=1e-15)


def test_h5_mini_card_with_restart_enumeration_equals_renewal(mini_fractions):
    tm = 10
    weights = {"A": 1, "B": 10, "C": 89}                       # Ganzzahlgewichte, Nenner 100^10
    cnt = [0] * (tm + 1)
    for seq in itertools.product("ABC", repeat=tm):            # alle 3^10 Folgen
        w = 1
        for ch in seq:
            w *= weights[ch]
        n_alarm, start = 0, 0
        while start < tm:
            al = _first_alarm_in_sequence(seq[start:])         # Neustart nach Alarm ohne Vorgeschichte
            if al is None:
                break
            n_alarm += 1
            start += al
        cnt[n_alarm] += w
    exact = [Fr(z, 100**tm) for z in cnt]
    assert sum(exact) == 1
    s = H.survival(mini_fractions, 0.0, tm)
    f = np.zeros(tm + 1)
    f[1:] = s[:-1] - s[1:]
    pmf, _ = R.renewal_counts(f, tm)
    np.testing.assert_allclose(pmf[: tm + 1], [float(z) for z in exact], rtol=1e-11, atol=1e-15)
    assert float(exact[0]) == pytest.approx(0.7813, abs=5e-5)
    assert float(exact[1]) == pytest.approx(0.1990, abs=5e-5)
    assert float(exact[2]) == pytest.approx(0.0188, abs=5e-5)
