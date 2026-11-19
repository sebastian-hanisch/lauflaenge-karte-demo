"""Orakel-Tests: die Kernroutinen gegen einen anderen Rechenweg (ORACLE-PLAYBOOK §5, DEMO-PLAYBOOK §12).

  einfache Karte      geschlossene Form 1/p und (1 - p)^t gegen die Kette mit einem Zustand gegen exakte Brüche (p = 1/370)
  Zusatzregeln        Laufzähler-Kette der App gegen die Fensterkette (Testhilfe, ohne Laufzähler, anderer Aufbau) in allen vier Regelsätzen
  Verteilung          Aufzählung aller Zonenfolgen (t <= 6 live, t <= 8 eingefroren in lfk_enum.json) gegen die Kette
  Regeln an Rohwerten lfk_chart.rule_flags (vektorisiert) gegen Schleifen (helpers_lfk.reference_first_alarm); die Regeln hängen nur an den Zonen
  Verzögerungsmaße    Pollak <= Lorden, steady-state <= Lorden, Regel 1 allein: alle gleich 1/p; bedingte Verzögerung gegen Simulation
  Alarmzahl           Erneuerungsrechnung gegen Binomial (einfache Karte, direkte Faltung und FFT)
  Simulation          Lauflängen und Fenster im Band gegen die exakten Werte, zwei Zufalls-Seeds
Zufallsergebnisse nur als Band (4,5 Standardfehler); Ganzzahlen und Brüche exakt."""

import json
import math
from fractions import Fraction as Fr
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

import lfk_chain as H
import lfk_chart as K
import lfk_evaluation as E
import lfk_renewal as R
import lfk_scenario as S
from lfk_rng import SplitMix64
from helpers_lfk import build_window_chain, reference_first_alarm
from lfk_make_enum import enumerate_first_alarm   # tools/lfk_make_enum.py

ROOT = Path(__file__).resolve().parent.parent
SETS = (1, 2, 3, 4)


# ------------------------------------------------------------------------------------------------------------------ einfache Karte
def test_simple_card_closed_form_equals_one_state_chain():
    chain = H.build_chain(1)
    assert chain.n == 1                                             # Brook/Evans mit einem Zustand
    for delta in (0.0, 0.5, 1.0, 2.0):
        p = K.p_alarm(3.0, delta, True)
        assert H.arl_vector(chain, delta)[0] == pytest.approx(1 / p, rel=1e-12)
        s = H.survival(chain, delta, 50)
        np.testing.assert_allclose(s, (1 - p) ** np.arange(51), rtol=1e-12)


def test_simple_card_geometric_distribution_in_exact_fractions():
    p = Fr(1, 370)
    pmf = [(1 - p) ** (t - 1) * p for t in range(1, 2001)]
    assert sum(pmf) == 1 - (1 - p) ** 2000                        # Teleskopsumme
    rl = E.run_length(1, True, K.c_for_arl(370.0, True), 8, 0.0)
    assert rl.mean == pytest.approx(370.0, rel=1e-9)
    assert rl.cdf(100) == pytest.approx(float(1 - (1 - p) ** 100), rel=1e-9)


def test_threshold_for_target_arl_and_arl_for_threshold():
    assert K.c_for_arl(370, True) == pytest.approx(2.9997, abs=5e-5)
    assert K.c_for_arl(370, False) == pytest.approx(2.7818, abs=5e-5)
    assert 1 / K.p_alarm(3.0, 0.0, True) == pytest.approx(370.3983, abs=5e-5)
    assert 1 / K.p_alarm(3.0, 0.0, False) == pytest.approx(740.80, abs=5e-3)
    for c, arl in ((2.0, 21.98), (2.5, 80.52), (3.0, 370.40), (3.5, 2149.34)):
        assert 1 / K.p_alarm(c, 0.0, True) == pytest.approx(arl, abs=0.005)
    for target in (50, 100, 370.398, 1000, 5000):
        assert 1 / K.p_alarm(K.c_for_arl(target, True), 0.0, True) == pytest.approx(target, rel=1e-10)


def test_geometric_quantile_agrees_with_the_definition_on_many_p():
    for p in (0.5, 0.1, 1 / 370, 0.0027, 6.3e-5, 0.9):
        for q in (0.01, 0.05, 0.5, 0.9, 0.95, 0.99):
            t = K.geometric_quantile(p, q)
            assert 1 - (1 - p) ** t >= q and (t == 1 or 1 - (1 - p) ** (t - 1) < q)


# ------------------------------------------------------------------------------------------------------------------ Zusatzregeln: zwei Ketten
def test_state_counts_of_both_chains_are_documented():
    assert [H.build_chain(r).n for r in SETS] == [1, 41, 1221, 1497]
    assert [H.build_chain(4, L).n for L in (8, 9, 10)] == [1497, 1589, 1681]
    assert [build_window_chain(r).n for r in SETS] == [1555, 1233, 1221, 15165]


@pytest.mark.parametrize("rules", SETS)
@pytest.mark.parametrize("delta", [0.0, 1.0, 3.0])
def test_run_counter_chain_equals_window_chain(rules, delta):
    a = H.arl_vector(H.build_chain(rules), delta)[0]
    b = H.arl_vector(build_window_chain(rules), delta)[0]
    assert a == pytest.approx(b, rel=1e-7)


def test_other_version_of_rule_4_agrees_between_chains():
    """9 in Folge (Fensterkette mit 9 Seiten: rund 9 s Aufbau; die Fassung mit 10 in Folge ist nur mit der Laufzähler-Kette gerechnet, die Fensterkette dazu braucht rund 100 s)."""
    a = H.arl_vector(H.build_chain(4, 9), 0.0)[0]
    b = H.arl_vector(build_window_chain(4, 9), 0.0)[0]
    assert a == pytest.approx(b, rel=1e-7)


def test_rule_1_of_a_rule_set_equals_the_simple_card():
    assert H.arl_vector(H.build_chain(1), 0.0)[0] == pytest.approx(1 / (2 * float(K.phi(-3.0))), rel=1e-10)


# ------------------------------------------------------------------------------------------------------------------ Aufzählung aller Zonenfolgen
@pytest.mark.parametrize("delta", [0.0, 1.0])
def test_enumeration_of_all_zone_sequences_t_le_6_equals_the_chain(delta):
    en = enumerate_first_alarm(delta, tlen=6)                      # 8^6 = 262 144 Folgen, Regeln direkt an den Zonen
    for r in SETS:
        s = H.survival(H.build_chain(r), delta, 6)
        pmf = s[:-1] - s[1:]
        np.testing.assert_allclose(en[r][1:7], pmf, rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("delta", [0.0, 1.0])
def test_frozen_enumeration_of_all_8_pow_8_sequences_equals_the_chain(delta):
    data = json.loads((ROOT / "lfk_enum.json").read_text(encoding="utf-8"))
    names = {1: "1", 2: "1+2", 3: "1+2+3", 4: "1+2+3+4"}
    for r in SETS:
        field = data["felder"][f"delta_{delta:g}/{names[r]}"]
        assert field["quelle"] and field["stand"]                  # Quelle und Stand je Feld
        s = H.survival(H.build_chain(r), delta, 8)
        np.testing.assert_allclose(field["werte"], s[:-1] - s[1:], rtol=1e-9, atol=1e-15)


# ------------------------------------------------------------------------------------------------------------------ Regeln an Rohwerten
@pytest.mark.parametrize("rules,run4", [(1, 8), (2, 8), (3, 8), (4, 8), (4, 9), (4, 10)])
def test_rule_flags_equal_the_loop_reference(rules, run4):
    rng = np.random.default_rng(1000 + rules * 10 + run4)
    for i in range(120):
        x = rng.normal(loc=rng.choice([0.0, 0.5, 1.0, 2.0]), scale=1.0, size=int(rng.integers(1, 90)))
        ref = reference_first_alarm(x, rules, run4=run4)
        flags = K.rule_flags(x, rules, run4=run4)
        first = K.first_alarm(flags)
        assert (first, int(K.trigger_rule(flags)[first - 1]) if first else None) == (ref if ref else (None, None)), (i, rules, run4)


def test_rule_1_with_threshold_and_one_sided_equals_the_loop_reference():
    rng = np.random.default_rng(7)
    for i in range(150):
        c = float(rng.uniform(1.5, 3.5))
        two = bool(rng.integers(0, 2))
        x = rng.normal(loc=float(rng.choice([0.0, 1.0, -1.0])), size=int(rng.integers(1, 60)))
        ref = reference_first_alarm(x, 1, c=c, two_sided=two)
        first = K.first_alarm(K.rule_flags(x, 1, c=c, two_sided=two))
        assert first == (ref[0] if ref else None), i


def test_the_rules_depend_only_on_the_zones():
    """Grundlage der Zonenkette: jede Beobachtung durch einen Wert derselben Zone ersetzt, ändert nichts am ersten Alarm."""
    rep = np.array([-4.0, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 4.0])
    rng = np.random.default_rng(11)
    for rules in SETS:
        for _ in range(150):
            x = rng.normal(loc=float(rng.choice([0.0, 1.0])), size=int(rng.integers(5, 40)))
            zones = np.array([K.zone_of(v) for v in x])
            assert reference_first_alarm(x, rules) == reference_first_alarm(rep[zones], rules)


def test_zone_of_boundaries():
    assert [K.zone_of(v) for v in (-3.1, -3.0, -2.5, -1.0, -0.1, 0.0, 0.9, 1.0, 2.0, 2.999, 3.0, 3.5)] == [0, 1, 1, 3, 3, 4, 4, 5, 6, 6, 7, 7]


# ------------------------------------------------------------------------------------------------------------------ Verzögerungsmaße
@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0, 3.0])
def test_delay_measures_order_for_all_rule_sets(delta):
    for r in (2, 3, 4):
        m = H.delay_measures(H.build_chain(r), delta)
        assert m.pollak <= m.lorden + 1e-9
        assert m.steady <= m.lorden + 1e-9
        assert m.zero <= m.lorden + 1e-9
        # In dieser Fassung ist der Start ohne Vorgeschichte der schlechteste Zustand: zero-state = Pollak = Lorden (gemessen in diesen Zellen, nicht bewiesen)
        assert m.pollak == pytest.approx(m.zero, abs=1e-9) and m.lorden == pytest.approx(m.zero, abs=1e-9)
        assert m.steady < m.zero


def test_steady_state_ratio_of_four_rules_at_delta_1():
    m = H.delay_measures(H.build_chain(4), 1.0)
    assert m.zero == pytest.approx(9.2219, abs=5e-5) and m.steady == pytest.approx(8.5391, abs=5e-5)
    assert m.steady / m.zero == pytest.approx(0.926, abs=5e-4)


def test_simple_card_all_four_measures_equal_one_over_p():
    for delta in (0.5, 1.0, 2.0):
        m = E.delay_measures(1, True, 3.0, 8, delta)
        v = 1 / K.p_alarm(3.0, delta, True)
        assert (m.zero, m.steady, m.pollak, m.lorden) == pytest.approx((v, v, v, v), rel=1e-12)
        # auch die Kette mit einem Zustand gibt für Regel 1 alles gleich
        mc = H.delay_measures(H.build_chain(1), delta)
        assert (mc.zero, mc.steady, mc.pollak, mc.lorden) == pytest.approx((v, v, v, v), rel=1e-9)


def test_conditional_delay_limits_zero_state_and_steady_state():
    chain = H.build_chain(4)
    m = H.delay_measures(chain, 1.0)
    assert E.conditional_delay(4, True, 3.0, 8, 1.0, 0).mean == pytest.approx(m.zero, rel=1e-12)
    assert E.conditional_delay(4, True, 3.0, 8, 1.0, 600).mean == pytest.approx(m.steady, rel=1e-9)
    means = [E.conditional_delay(4, True, 3.0, 8, 1.0, nu).mean for nu in (0, 1, 2, 5, 10, 50)]
    assert max(means) == pytest.approx(means[0], abs=1e-12) and means[-1] < means[0]   # kein ν hat eine größere Verzögerung als der Start


def test_conditional_delay_of_the_four_rules_against_simulation_with_the_loop_reference():
    """Sprung nach nu = 30 ruhigen Beobachtungen (nur Läufe ohne Alarm davor): mittlere Verzögerung gegen die exakte bedingte Verteilung. Die Regeln arbeiten
    hier an den Rohwerten des Rollfensters (Schleifen, anderer Code als die Kette); das Fenster der Regeln reicht höchstens 10 Beobachtungen zurück."""
    inv = NormalDist().inv_cdf
    nu, delta, runs = 30, 1.0, 1500
    rng = SplitMix64(31337)
    ds = []
    while len(ds) < runs:
        x = []
        for t in range(1, 5000):
            x.append(inv(rng.random() or 2.0**-54) + (delta if t > nu else 0.0))
            window = x[-10:]
            hit = reference_first_alarm(window, 4)
            if hit is not None and hit[0] == len(window):          # Alarm bei der letzten Beobachtung
                break
        if t > nu:
            ds.append(t - nu)
    exact = E.conditional_delay(4, True, 3.0, 8, delta, nu)
    mean, se = float(np.mean(ds)), float(np.std(ds, ddof=1) / math.sqrt(len(ds)))
    assert abs(mean - exact.mean) < 4.5 * se, (mean, se, exact.mean)


# ------------------------------------------------------------------------------------------------------------------ Alarmzahl je Fenster
@pytest.mark.parametrize("window", [100, 1000, 1499, 1500, 3000, 5000])
def test_renewal_counts_equal_binomial_for_the_simple_card(window):
    p = K.p_alarm(3.0, 0.0, True)
    f = np.zeros(window + 1)
    f[1:] = p * (1 - p) ** np.arange(window)
    pmf, tail = R.renewal_counts(f, window)
    pb = R.binomial_counts(window, p)
    m = min(len(pmf), len(pb))
    assert np.max(np.abs(pmf[:m] - pb[:m])) < 1e-10
    assert abs(pmf.sum() - 1) < 1e-9 and abs(float((np.arange(len(pmf)) * pmf).sum()) - window * p) < 1e-8


def test_renewal_fft_equals_direct_convolution_for_the_four_rules(monkeypatch):
    rl = E.run_length(4, True, 3.0, 8, 0.0)
    f = R.pmf_from_survival(rl.s, 1600)
    monkeypatch.setattr(R, "FFT_FROM", 10**9)
    direct, _ = R.renewal_counts(f, 1600)
    monkeypatch.setattr(R, "FFT_FROM", 1)
    fft, _ = R.renewal_counts(f, 1600)
    m = min(len(direct), len(fft))
    assert np.max(np.abs(direct[:m] - fft[:m])) < 1e-11


def test_window_count_distribution_is_a_probability_distribution_and_mean_near_window_over_arl0():
    for rules in SETS:
        cd = E.count_distribution(rules, True, 3.0, 8, 1000)
        assert abs(cd["pmf"].sum() - 1) < 1e-9
        assert abs(cd["mean"] / cd["t_over_arl0"] - 1) < 0.005          # Erneuerungsrechnung gegen W/ARL₀, Abweichung höchstens 0,5 %


# ------------------------------------------------------------------------------------------------------------------ Simulation im Band
def _band(value, se, exact, label):
    assert abs(value - exact) < 4.5 * se, (label, value, se, exact)


@pytest.mark.parametrize("seed", [10100, 20200])
@pytest.mark.parametrize("delta", [0.0, 1.0])
def test_simulated_run_lengths_of_the_simple_card_are_in_the_band(seed, delta):
    runs = 3000
    lengths = S.simulate_run_lengths(1, delta, runs, seed)
    rl = E.run_length(1, True, 3.0, 8, delta)
    se = lengths.std(ddof=1) / math.sqrt(runs)
    _band(lengths.mean(), se, rl.mean, "Mittel")
    for t in (10, 100, 370):
        share = float(np.mean(lengths <= t))
        _band(share, math.sqrt(rl.cdf(t) * (1 - rl.cdf(t)) / runs), rl.cdf(t), f"P(T<={t})")


@pytest.mark.parametrize("rules,runs", [(2, 1200), (3, 1200), (4, 1500)])
def test_simulated_run_lengths_of_the_rule_sets_are_in_the_band(rules, runs):
    lengths = S.simulate_run_lengths(rules, 0.0, runs, 30000 + rules)
    rl = E.run_length(rules, True, 3.0, 8, 0.0)
    se = lengths.std(ddof=1) / math.sqrt(runs)
    _band(lengths.mean(), se, rl.mean, f"Mittel Regeln {rules}")
    for t in (10, 100):
        _band(float(np.mean(lengths <= t)), math.sqrt(rl.cdf(t) * (1 - rl.cdf(t)) / runs), rl.cdf(t), f"P(T<={t}) Regeln {rules}")


def test_simulated_run_lengths_of_four_rules_after_a_jump_are_in_the_band_and_use_other_code():
    lengths = S.simulate_run_lengths(4, 1.0, 1500, 777)
    rl = E.run_length(4, True, 3.0, 8, 1.0)
    se = lengths.std(ddof=1) / math.sqrt(len(lengths))
    _band(lengths.mean(), se, rl.mean, "Mittel δ = 1")


@pytest.mark.parametrize("seed", [10300, 20300])
def test_simulated_windows_of_the_simple_card_are_in_the_band(seed):
    nwin = 600
    cnt = S.simulate_window_counts(1, 1000, nwin, seed)
    cd = E.count_distribution(1, True, 3.0, 8, 1000)
    _band(cnt.mean(), cnt.std(ddof=1) / math.sqrt(nwin), cd["mean"], "Mittel der Alarmzahl")
    for k in (0, 5):
        share = float(np.mean(cnt == 0)) if k == 0 else float(np.mean(cnt >= 5))
        exact = cd["p0"] if k == 0 else cd["p_ge5"]
        _band(share, math.sqrt(exact * (1 - exact) / nwin), exact, f"Anteil k={k}")


def test_simulated_windows_of_four_rules_are_in_the_band():
    nwin = 150
    cnt = S.simulate_window_counts(4, 1000, nwin, 10301)
    cd = E.count_distribution(4, True, 3.0, 8, 1000)
    _band(cnt.mean(), cnt.std(ddof=1) / math.sqrt(nwin), cd["mean"], "Mittel der Alarmzahl, vier Regeln")


def test_random_stream_marginal_matches_the_normal_tail():
    """Der Zufallsstrom der Demo: Randanteil u < Φ(-3) oder u > 1 - Φ(-3) gegen 2Φ(-3) (Zufalls-Seed 10401, 400 000 Zahlen), Band."""
    rng = S.SplitMix64(10401)
    pt = float(K.phi(-3.0))
    n = 400_000
    k = sum(1 for _ in range(n) if (lambda u: u < pt or u > 1 - pt)(rng.random()))
    exact = 2 * pt
    assert abs(k / n - exact) < 4.5 * math.sqrt(exact * (1 - exact) / n)
