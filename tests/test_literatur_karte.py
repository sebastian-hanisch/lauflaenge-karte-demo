"""Literaturwerte, nur aus gelesenen Quellen (Stufe W im Docstring der jeweiligen Prüfung; „Quelle nicht eingesehen heißt: kein Test“).

Nicht getestet, weil nicht gelesen: Shewhart 1924/1931, Montgomery, Western-Electric-Handbuch, Champ/Woodall 1987 (die „etwa 92“ der vier Regeln stehen
nur im Gedächtnis; die Rechnung gibt 91,7508, das ist Rechnung, keine Quellenbestätigung), Tabellenzahlen des NIST-Handbuchs (zwei Zellen der CUSUM-Tabelle
dort waren falsch)."""

import pytest

import lfk_chain as H
import lfk_chart as K


def test_nist_text_limits_3_sigma_one_sided_probability_and_arl_about_371():
    """NIST/SEMATECH e-Handbook 6.3.1 und 6.3.2.1 (Stufe W über die Zusammenfassung des Abrufwerkzeugs, nicht Roh-HTML): Grenzen ±3σ, Einseitenwahrscheinlichkeit 0,00135,
    ARL = 1/p, für die Normalverteilung etwa 371. Nur die Textaussage, keine Tabellenzahl."""
    assert float(K.phi(-3.0)) == pytest.approx(0.00135, abs=1e-5)
    assert 1 / K.p_alarm(3.0, 0.0, True) == pytest.approx(371, abs=1.0)


def test_jones_farmer_fap_of_a_phase_i_chart_with_m_points():
    """Jones-Farmer/Woodall/Steiner/Champ 2014, Journal of Quality Technology 46(3) (Stufe W, Abschnitt 2): FAP = 1 - (1 - p)^m mit p = 0,0027; 0,078 bei m = 30
    und 0,126 bei m = 50 Punkten (gedruckt auf drei Stellen)."""
    p = 0.0027
    assert round(1 - (1 - p) ** 30, 3) == 0.078
    assert round(1 - (1 - p) ** 50, 3) == 0.126
    p3 = K.p_alarm(3.0, 0.0, True)
    assert 1 - (1 - p3) ** 30 == pytest.approx(0.0779, abs=5e-5) and 1 - (1 - p3) ** 50 == pytest.approx(0.1264, abs=5e-5)


@pytest.mark.parametrize("delta,printed,tol", [(0.0, 500, 0.5), (1.0, 54.6, 0.05), (1.5, 17.9, 0.05), (2.0, 7.26, 0.01)])
def test_lucas_saccucci_table_3_column_lambda_1_is_the_shewhart_chart(delta, printed, tol):
    """Lucas/Saccucci 1990, Technometrics 32(1), Tabelle 3, Spalte λ = 1 (Stufe W; Kopie mit Zeichenverlusten): λ = 1 ist die Shewhart-Karte, L = 3,090 gibt ARL₀ = 500
    (gerechnet 499,6) und ARL bei δ = 1; 1,5; 2 gleich 54,6; 17,9; 7,26. Toleranz: eine Einheit der letzten gedruckten Stelle (gerechnet 54,554; 17,883; 7,254; bei δ = 2
    liegt der gedruckte Wert 0,006 über der Rechnung: die Quelle rundet anders, nicht aufgeklärt)."""
    arl = 1 / K.p_alarm(3.090, delta, True)
    assert arl == pytest.approx(printed, abs=tol)


def test_lucas_saccucci_one_chain_state_agrees_with_the_closed_form():
    assert H.arl_vector(H.build_chain(1), 1.0)[0] == pytest.approx(1 / K.p_alarm(3.0, 1.0, True), rel=1e-12)
