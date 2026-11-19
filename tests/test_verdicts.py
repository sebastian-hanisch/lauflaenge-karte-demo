"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zahlen, nicht Seeds. Damit prüft der Test die Grenzen (>= 0,10, >= 0,25) und jede genannte Zahl direkt."""

import lfk_chain as H
import lfk_evaluation as E

LABEL = "zweiseitige Karte mit Grenzen ±3σ"


def _series(**kw):
    base = dict(label=LABEL, first=137, rule=1, n=400, nu=100, delta=1.0, p_before=0.2369, arl0=370.398, delay={"mean": 43.89, "median": 31, "q95": 130}, d_cdf=0.574)
    base.update(kw)
    return E.verdict_series(**base)


def test_series_after_the_jump_is_a_success_and_names_delay_expectation_and_position():
    kind, text = _series()
    assert kind == "success"
    assert "ARL₀ = 370,4 Beobachtungen (zero-state)" in text and "Alarm bei Beobachtung 137 (Regel 1), nach dem Sprung bei 100" in text
    assert "Verzögerung D = 37 Beobachtungen" in text and "P(D ≤ 37) = 57,4 %" in text
    assert "im Mittel 43,9 Beobachtungen" in text and "Median 31" in text and "95 % innerhalb von 130" in text and "Sprung um 1σ" in text


def test_series_with_a_false_alarm_before_the_jump_is_a_warning():
    kind, text = _series(first=60, d_cdf=None)
    assert kind == "warning" and "vor dem Sprung bei 100: ein Fehlalarm" in text and "innerhalb der ersten 100 Beobachtungen (exakt): 23,69 %" in text
    assert "Verzögerung D" not in text
    assert _series(first=100, d_cdf=None)[0] == "warning"          # ein Alarm genau bei ν gehört zu den Beobachtungen vor der Änderung (ν = Zahl der Beobachtungen davor)
    assert _series(first=101, d_cdf=0.01)[0] == "success" and "Verzögerung D = 1 Beobachtungen" in _series(first=101, d_cdf=0.01)[1]


def test_series_without_alarm_after_a_jump_reports_the_undetected_jump():
    kind, text = _series(first=None, d_cdf=None)
    assert kind == "warning" and "unentdeckt" in text and "restlichen 300 Beobachtungen" in text


def test_series_without_change_never_reports_a_delay():
    kind, text = _series(delta=0.0, first=None, delay=None, d_cdf=None)
    assert kind == "info" and "Keine Änderung in dieser Reihe" in text and "Verzögerung" not in text and "ruhig" in text
    kind, text = _series(delta=0.0, first=234, delay=None, d_cdf=None)
    assert kind == "warning" and "Alarm bei Beobachtung 234 (Regel 1): ein Fehlalarm" in text and "Verzögerung D" not in text
    kind, text = _series(nu=400, first=250, delay=None, d_cdf=None)     # Sprung hinter der Reihe: kein Sprung in dieser Reihe
    assert kind == "warning" and "ein Fehlalarm" in text and "Verzögerung D" not in text


def test_series_with_nu_zero_has_no_false_alarm_sentence():
    kind, text = _series(nu=0, p_before=0.0, first=40, d_cdf=0.3)
    assert kind == "success" and "innerhalb der ersten" not in text


def test_runlength_verdict_numbers_and_threshold():
    kind, text = E.verdict_runlength(label=LABEL, arl0=370.398, median=257, t0=100, p_t0=0.2369, p_over_1000=0.0670, sim=None)
    assert kind == "warning"                                            # 23,69 % >= 10 %
    assert "ARL₀ = 370,4" in text and "Median 257" in text and "23,69 % der Reihen" in text and "ersten 100 Beobachtungen" in text and "6,70 % bleiben über 1000" in text
    assert E.verdict_runlength(label=LABEL, arl0=5000.0, median=3466, t0=100, p_t0=0.0198, p_over_1000=0.818, sim=None)[0] == "info"
    assert E.verdict_runlength(label=LABEL, arl0=1000.0, median=693, t0=100, p_t0=0.10, p_over_1000=0.37, sim=None)[0] == "warning"      # Grenze 10 % inklusive
    kind, text = E.verdict_runlength(label=LABEL, arl0=370.398, median=257, t0=100, p_t0=0.2369, p_over_1000=0.067, sim={"runs": 3000, "mean": 365.7, "se": 6.8, "share": 0.2373})
    assert "Simulation mit 3000 Läufen: Mittel 365,7 ± 6,8, 23,73 % der Reihen innerhalb von 100" in text


def test_windows_verdict():
    kind, text = E.verdict_windows(label=LABEL, window=1000, mean=2.6998, p0=0.06697, p_ge5=0.1368, t_over_arl0=2.6998, sim=None)
    assert kind == "info" and "Fenster zu 1000 Beobachtungen: 2,70 Alarme im Mittel (T/ARL₀ = 2,70)" in text and "6,70 % der Fenster ohne Alarm, 13,68 % mit fünf oder mehr" in text
    kind, text = E.verdict_windows(label=LABEL, window=1000, mean=10.866, p0=1.22e-5, p_ge5=0.98633, t_over_arl0=10.899, sim={"n": 300, "mean": 10.72, "se": 0.18, "p0": 0.0})
    assert "0,001 % der Fenster ohne Alarm" in text and "Simulation mit 300 Fenstern: Mittel 10,72 ± 0,18, 0,00 % ohne Alarm" in text


def test_delay_verdict_numbers_and_threshold():
    kind, text = E.verdict_delay(label=LABEL, delta=1.0, arl1=43.8947, median=31, q95=130, p_over_100=0.0998, measures=None)
    assert kind == "info"
    assert "Sprung um 1σ" in text and "ARL₁ = 43,9" in text and "Median 31" in text and "95 % innerhalb von 130" in text and "9,98 %" in text and "sind zero-state, Pollak und Lorden gleich" in text
    kind, text = E.verdict_delay(label=LABEL, delta=0.5, arl1=155.2242, median=108, q95=464, p_over_100=0.524, measures=None)
    assert kind == "warning" and "ARL₁ = 155,2" in text and "52,40 %" in text
    assert E.verdict_delay(label=LABEL, delta=0.5, arl1=100.0, median=70, q95=300, p_over_100=0.25, measures=None)[0] == "warning"        # Grenze 25 % inklusive
    assert E.verdict_delay(label=LABEL, delta=0.5, arl1=100.0, median=70, q95=300, p_over_100=0.2499, measures=None)[0] == "info"
    kind, text = E.verdict_delay(label=LABEL, delta=3.0, arl1=2.0, median=1, q95=5, p_over_100=1e-9, measures=None)
    assert "ARL₁ = 2,00" in text and "< 0,001 %" in text                                                     # nie „0,00 %“ für ein seltenes Ereignis
    m = H.DelayMeasures(9.2219, 8.5391, 9.2219, 0, 9.2219, 0)
    kind, text = E.verdict_delay(label="Regeln", delta=1.0, arl1=9.2219, median=8, q95=22, p_over_100=0.0, measures=m)
    assert "steady-state (quasistationär) 8,54" in text and "Pollak 9,22" in text and "Lorden 9,22" in text and "gleich" not in text


def test_delay_verdict_without_a_jump_reports_no_delay():
    kind, text = E.verdict_delay(label=LABEL, delta=0.0, arl1=370.4, median=257, q95=1109, p_over_100=0.0, measures=None)
    assert kind == "info" and "keine Verzögerung" in text and "ARL₁" not in text


def test_rules_verdict():
    kind, text = E.verdict_rules(rule_count=4, run4=8, delta=1.0, arl0=91.7508, c_eq=2.5459, arl1_rules=9.2219, arl1_simple=16.3244, cross=2.2558, steady=8.5391)
    assert kind == "success"
    assert "ARL₀ = 91,75 Beobachtungen (statt 370,4" in text and "c = 2,5459σ" in text and "δ = 1σ: ARL₁ 9,22 gegen 16,32" in text and "steady-state 8,54 (quasistationär)" in text
    assert "δ* = 2,26σ" in text and "darüber ist die einfache Karte schneller" in text
    kind, text = E.verdict_rules(rule_count=4, run4=9, delta=3.0, arl0=109.049, c_eq=2.6, arl1_rules=1.6666, arl1_simple=1.4812, cross=2.2558, steady=1.628)
    assert kind == "warning" and "Regel 4 mit 9 in Folge" in text and "ARL₁ 1,67 gegen 1,48" in text      # über der Kreuzung ist die einfache Karte schneller
    kind, text = E.verdict_rules(rule_count=2, run4=8, delta=0.0, arl0=225.438, c_eq=2.8454, arl1_rules=None, arl1_simple=None, cross=3.4554, steady=None)
    assert kind == "info" and "δ > 0 wählen" in text and "ARL₁" not in text
    kind, text = E.verdict_rules(rule_count=3, run4=8, delta=1.0, arl0=132.893, c_eq=2.6727, arl1_rules=10.9514, arl1_simple=21.1352, cross=None, steady=None)
    assert "Kreuzung" not in text and "steady-state" not in text


def test_kernfragen_answer_every_question_and_name_the_unit():
    args = dict(first=234, rule=1, n=400, label=LABEL, arl0=370.398, median=257, t0=100, p_t0=0.2369, p_over_1000=0.067, window=1000, win_mean=2.6998, p0=0.067, p_ge5=0.1368,
                delta=1.0, arl1=43.8947, delay_median=31, delay_q95=130, four_arl0=91.7508, four_arl1=9.2219, four_simple=16.3244, four_cross=2.2558)
    q = E.kernfragen(**args)
    assert [t for t, _ in q] == ["Wann schlägt die Karte an?", "Was heißt „ARL₀ = 370,4“ für eine einzelne Reihe?", "Wie viele Fehlalarme gibt es in einem Fenster?",
                                 "Wie lange dauert es nach einem Sprung?", "Was kosten Zusatzregeln?"]
    assert "Beobachtung 234 (Regel 1)" in q[0][1]
    assert "Median 257" in q[1][1] and "23,69 %" in q[1][1] and "6,70 %" in q[1][1] and "zero-state" in q[1][1]
    assert "2,70 Alarme" in q[2][1] and "13,68 %" in q[2][1]
    assert "43,9 Beobachtungen" in q[3][1] and "Median 31" in q[3][1] and "130" in q[3][1]
    assert "von 370,4 auf 91,75" in q[4][1] and "9,22 gegen 16,32" in q[4][1] and "δ* = 2,26σ" in q[4][1]
    q0 = E.kernfragen(**{**args, "first": None, "delta": 0.0, "arl1": None, "delay_median": None, "delay_q95": None, "four_arl1": None, "four_simple": None})
    assert "bleibt die Karte 400 Beobachtungen ruhig" in q0[0][1] and "Ohne Sprung" in q0[3][1] and "gegen" not in q0[4][1]


def test_card_label_names_the_card_and_the_rule_4_version():
    assert E.card_label(1, True, 3.0) == "zweiseitige Karte mit Grenzen ±3σ"
    assert E.card_label(1, False, 2.7818) == "einseitige Karte mit Grenze 2,7818σ"
    assert E.card_label(4, True, 3.0, 9) == "zweiseitige Regeln 1+2+3+4, Regel 4 mit 9 in Folge"
    assert E.card_label(2, True, 3.0) == "zweiseitige Regeln 1+2"


def test_adaptive_percent_never_prints_zero_for_a_possible_event():
    assert E.de_pct(1.2e-5, None) == "0,001 %" and E.de_pct(1e-9, None) == "< 0,001 %" and E.de_pct(0.0, None) == "< 0,001 %" and E.de_pct(0.2369, None) == "23,69 %"
    assert E.de_arl(91.7508) == "91,75" and E.de_arl(370.398) == "370,4"
