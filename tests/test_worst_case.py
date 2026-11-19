"""Worst-Case-Läufe an den Reglermaxima (DEMO-PLAYBOOK §1c): fangen Hänger und unerwartetes Aufblähen, prüfen aber keine Zeit (keine Zeit-Assertion, Zeiten sind plattformabhängig)."""

import numpy as np

import lfk_constants as C
import lfk_evaluation as E
import lfk_scenario as S


def test_exact_worst_cases_finish_and_are_consistent():
    """Vier Regeln mit Regel 4 = 10 in Folge, ν und Fenster am Maximum, größter Sprung: Komplexität der Kette O(Zustände · Beobachtungen)."""
    cd = E.count_distribution(4, True, 3.0, 10, C.WINDOW_MAX)
    assert abs(cd["pmf"].sum() - 1) < 1e-9
    d = E.conditional_delay(4, True, 3.0, 10, C.DELTA_MAX, C.NU_MAX)
    assert 1.0 <= d.mean < 3.0 and d.quantile(0.5) >= 1
    cmp = E.equal_arl_comparison(4, 10)
    assert cmp["cross"] is not None and cmp["arl0"] > 91.75
    assert len(E.rule4_versions(C.DELTA_MAX)) == 4


def test_simulation_at_the_budget_limit_finishes():
    runs = S.max_runs(91.75, 4, C.RUNS_MAX)
    assert runs == C.RUNS_MAX
    lengths = S.simulate_run_lengths(4, 0.0, runs, 99)
    assert len(lengths) == runs and lengths.min() >= 1
    nwin = min(C.NWIN_MAX, C.SIM_BUDGET_RULES // C.WINDOW_MIN)
    counts = S.simulate_window_counts(4, C.WINDOW_MIN, min(nwin, 600), 99)
    assert counts.shape == (min(nwin, 600),) and np.all(counts >= 0)


def test_simple_card_simulation_at_the_budget_limit_finishes():
    arl0 = E.run_length(1, True, 3.0, 8, 0.0).mean
    runs = S.max_runs(arl0, 1, C.RUNS_MAX)
    assert runs == C.RUNS_MAX                    # 5000 Läufe bei ARL₀ = 370,4 (1,85 Mio. Beobachtungen) liegen unter dem Budget
    lengths = S.simulate_run_lengths(1, 0.0, runs, 7)
    assert len(lengths) == runs
