"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import lfk_chain as H
import lfk_chart as K
import lfk_evaluation as E
import lfk_renewal as R
import lfk_scenario as S


def test_threshold_zero_means_every_observation_alarms():
    assert K.p_alarm(0.0, 0.0, True) == pytest.approx(1.0)
    rl = E.run_length(1, True, 0.0, 8, 0.0)
    assert rl.mean == pytest.approx(1.0) and rl.quantile(0.5) == 1 and rl.cdf(1) == pytest.approx(1.0)


def test_negative_threshold_does_not_crash_in_the_closed_form():
    assert K.p_alarm(-1.0, 0.0, True) > 0.99
    assert E.run_length(1, True, -1.0, 8, 0.0).quantile(0.95) == 1


def test_very_large_threshold_gives_a_huge_but_finite_arl_and_quantiles():
    rl = E.run_length(1, True, 8.0, 8, 0.0)
    assert np.isfinite(rl.mean) and rl.mean > 1e14
    assert rl.quantile(0.5) > 1e13 and 0.0 <= rl.cdf(1000) < 1e-9


def test_delta_zero_and_very_large_delta():
    assert E.run_length(1, True, 3.0, 8, 10.0).mean == pytest.approx(1.0, abs=1e-9)
    for rules in (2, 3, 4):
        assert E.run_length(rules, True, 3.0, 8, 10.0).mean == pytest.approx(1.0, abs=1e-6)
        assert E.run_length(rules, True, 3.0, 8, 0.0).mean > 50


def test_one_sided_with_negative_delta_practically_never_alarms():
    assert 1 / K.p_alarm(3.0, -3.0, False) > 1e8
    x = S.make_series(1, 200, 0, -3.0)
    assert K.first_alarm(K.rule_flags(x, 1, 3.0, False)) is None


def test_rule_4_needs_eight_observations():
    assert K.first_alarm(K.rule_flags(np.full(7, 0.5), 4)) is None
    assert K.first_alarm(K.rule_flags(np.full(8, 0.5), 4)) == 8
    assert K.first_alarm(K.rule_flags(np.full(8, -0.5), 4)) == 8
    assert K.first_alarm(K.rule_flags(np.full(8, 0.5), 3)) is None               # ohne Regel 4 kein Alarm
    assert K.first_alarm(K.rule_flags(np.full(9, 0.5), 4, run4=9)) == 9          # Fassung 9 in Folge
    assert K.first_alarm(K.rule_flags(np.full(8, 0.5), 4, run4=9)) is None
    x = np.full(8, 0.5)
    x[3] = -0.1                                                                  # eine Beobachtung auf der anderen Seite unterbricht den Lauf
    assert K.first_alarm(K.rule_flags(x, 4)) is None


def test_rules_two_and_three_on_hand_series():
    assert K.first_alarm(K.rule_flags(np.array([2.5, 0.0, 2.5]), 2)) == 3        # 2 von 3 jenseits 2σ
    assert K.first_alarm(K.rule_flags(np.array([2.5, 0.0, 0.0, 2.5]), 2)) is None
    assert K.first_alarm(K.rule_flags(np.array([2.5, -2.5, 0.0]), 2)) is None    # nicht auf derselben Seite
    assert K.first_alarm(K.rule_flags(np.array([2.5, -2.5, 2.5]), 2)) == 3       # zwei positive unter drei, auch wenn dazwischen die andere Seite liegt
    assert K.first_alarm(K.rule_flags(np.array([1.5, 1.5, 0.0, 1.5, 1.5]), 3)) == 5   # 4 von 5 jenseits 1σ
    assert K.first_alarm(K.rule_flags(np.array([3.5]), 1)) == 1


def test_series_edge_cases():
    assert len(S.make_series(1, 1, 0, 1.0)) == 1                                  # Länge 1
    z = S.make_series(3, 20, 0, 0.0)
    np.testing.assert_allclose(S.make_series(3, 20, 0, 2.0), z + 2.0, atol=1e-15)  # ν = 0: Änderung vor der ersten Beobachtung
    np.testing.assert_allclose(S.make_series(3, 20, 20, 2.0), z, atol=1e-15)      # ν gleich Länge: der Sprung liegt hinter der Reihe
    np.testing.assert_allclose(S.make_series(3, 20, 999, 2.0), z, atol=1e-15)     # ν größer als die Länge


def test_quiet_series_has_no_first_alarm_instead_of_zero():
    assert K.first_alarm(K.rule_flags(np.zeros(30), 4)) is None


def test_window_of_length_one_and_two():
    cd = E.count_distribution(1, True, 3.0, 8, 1)
    p = K.p_alarm(3.0, 0.0, True)
    np.testing.assert_allclose(cd["pmf"][:2], [1 - p, p], atol=1e-14)
    cd4 = E.count_distribution(4, True, 3.0, 8, 1)
    assert cd4["pmf"][0] == pytest.approx(1 - p, abs=1e-12)                       # eine Beobachtung: nur Regel 1 kann anschlagen
    assert E.count_distribution(1, True, 3.0, 8, 2)["pmf"].sum() == pytest.approx(1.0)


def test_single_run_and_single_window_simulation():
    one = S.simulate_run_lengths(1, 0.0, 1, 5)
    assert one.shape == (1,) and one[0] >= 1
    rl = E.run_length(1, True, 3.0, 8, 0.0)
    summary = E.sim_run_summary(one, rl, 100)
    assert summary["runs"] == 1 and summary["z_mean"] == 0.0                     # keine Division durch einen Standardfehler 0 oder NaN
    assert S.simulate_window_counts(1, 1, 1, 5).shape == (1,)


def test_simulation_budget_limits_and_refuses_too_expensive_runs():
    assert S.max_runs(370.4, 1, 5000) == 5000
    assert S.max_runs(31574.0, 1, 5000) == 0                                     # 2 Mio. Beobachtungen reichen nur für 63 Läufe: zu wenig
    assert S.max_runs(1000.0, 1, 5000) == 2000                                   # auf das Budget begrenzt
    assert S.max_runs(91.75, 4, 5000) == 5000 and S.max_runs(225.4, 2, 5000) == 2661


def test_quantile_beyond_the_computed_range_returns_the_length():
    assert H.quantile(np.array([1.0, 0.9, 0.8]), 0.95) == 3


def test_settings_force_two_sided_and_c_3_with_additional_rules():
    s = E.Settings(rules=3, side="ein", c=2.0)
    assert s.two_sided and s.c_eff == 3.0
    s1 = E.Settings(rules=1, side="ein", c=2.5)
    assert not s1.two_sided and s1.c_eff == 2.5


def test_formatting_helpers():
    assert E.de(1234.5678, 2) == "1234,57"
    assert E.de_pct(0.2369, 2) == "23,69 %"
    assert (E.fmt_c(3.0), E.fmt_c(2.9997), E.fmt_c(2.5)) == ("3", "2,9997", "2,5")


def test_chain_input_validation():
    for bad in (0, 5):
        with pytest.raises(ValueError):
            H.build_chain(bad)


def test_renewal_for_an_impossible_alarm():
    pmf, tail = R.renewal_counts(np.zeros(11), 10)
    assert pmf[0] == pytest.approx(1.0) and tail[0] == 1.0
