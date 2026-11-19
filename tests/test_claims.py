"""Claims: jede Zahl der README (Abschnitte „Warum dieses Problem“ bis G) wird hier nachgerechnet, über dieselben Auswertungsfunktionen wie in der App.

Jede Prüfung hat zwei Teile: (1) die genannte Schreibweise steht in der README (so veraltet keine Zahl still), (2) sie stimmt mit der Rechnung auf eine halbe Einheit der
letzten genannten Stelle. Die Zahlen sind exakt (geschlossene Form, Kette); Simulationszahlen der README stehen als „Messung dieser README, nicht im Test“ und werden
hier nicht geprüft (ihre Bänder prüft test_oracle_karte.py). Die Tabelle „Zahl → Test“ der README nennt die Zuordnung."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import lfk_chain as H
import lfk_chart as K
import lfk_evaluation as E
import lfk_texts as T

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------------------------------ Hilfen
def _num(s: str) -> tuple[float, int]:
    """Erste Zahl in `s` (Dezimalkomma) und ihre Nachkommastellen."""
    m = re.search(r"-?\d+(?:,\d+)?", s)
    assert m, s
    txt = m.group(0)
    return float(txt.replace(",", ".")), (len(txt.split(",")[1]) if "," in txt else 0)


def close(written: str, value: float):
    """`written` (z. B. „23,69 %“ oder „370,398“) stimmt mit `value` überein; Prozentzahlen übergibt der Aufrufer schon mal 100."""
    x, dec = _num(written)
    assert abs(value - x) <= 0.5 * 10 ** (-dec) * (1 + 1e-9) + 1e-12, (written, value)


def says(context: str):
    """Die README enthält diese Schreibweise."""
    assert context in README, context


def claim(context: str, written: str, value: float):
    says(context)
    assert written in context, (context, written)
    close(written, value)


def numbers_in(text: str) -> list[str]:
    return re.findall(r"-?\d+(?:,\d+)?", text)


def claims_seq(context: str, values: list[float], start: str | None = None):
    """Die Zahlen von `context` (in Reihenfolge, ggf. ab `start`) stimmen mit `values` überein; Kontext steht in der README."""
    says(context)
    text = context if start is None else context[context.index(start) + len(start):]
    nums = numbers_in(text)
    assert len(nums) == len(values), (context, nums, values)
    for n_, v in zip(nums, values):
        if v is not None:                      # None: Zahl im Kontext, die kein Messwert ist (Regelnummer, Quantilsstufe)
            close(n_, v)


def table(header_start: str) -> list[list[str]]:
    """Zeilen (Zellen) der Markdown-Tabelle, deren Kopfzeile mit `header_start` beginnt (ohne Kopf und Trennzeile)."""
    lines = README.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith(header_start))
    rows = []
    for ln in lines[i + 2:]:
        if not ln.startswith("|"):
            break
        rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
    return rows


# ------------------------------------------------------------------------------------------------------------------ Rechnung
RL = {r: E.run_length(r, True, 3.0, 8, 0.0) for r in (1, 2, 3, 4)}
P3 = K.p_alarm(3.0, 0.0, True)
CMP = {r: E.equal_arl_comparison(r, 8) for r in (2, 3, 4)}


def _s(rl):
    return rl.s if rl.s is not None else rl.survival_curve(H.tail_length(rl.mean))


def _gap(r):
    return H.max_distance_to_geometric(_s(RL[r]), RL[r].mean)


# ------------------------------------------------------------------------------------------------------------------ „Warum dieses Problem“
def test_warum_dieses_problem():
    rl, rl4 = RL[1], RL[4]
    claim("im Mittel erst nach 370,4 Beobachtungen", "370,4", rl.mean)
    claim("der Median liegt bei 257 Beobachtungen", "257", rl.quantile(0.5))
    claim("23,69 % der Reihen haben schon innerhalb der ersten 100 Beobachtungen", "23,69", 100 * rl.cdf(100))
    claim("6,70 % bleiben über 1000 Beobachtungen ruhig", "6,70", 100 * rl.sf(1000))
    cd = E.count_distribution(1, True, 3.0, 8, 1000)
    claim("im Mittel 2,70 Alarme, in 13,68 % der Fenster fünf oder mehr", "2,70", cd["mean"])
    close("13,68", 100 * cd["p_ge5"])
    d1 = E.run_length(1, True, 3.0, 8, 1.0)
    claim("Ein Sprung um 1σ braucht im Mittel 43,9 Beobachtungen", "43,9", d1.mean)
    claim("Median 31, 95-%-Quantil 130", "31", d1.quantile(0.5))
    close("130", d1.quantile(0.95))
    d05 = E.run_length(1, True, 3.0, 8, 0.5)
    claim("in 52,40 % der Reihen länger als 100 Beobachtungen unentdeckt", "52,40", 100 * d05.sf(100))
    claim("senken die ARL₀ auf 91,75", "91,75", rl4.mean)
    claims_seq("(9,22 gegen 16,32 Beobachtungen, zero-state)", list(E.compare_at(CMP[4], 1.0)))
    claim("ab δ\\* = 2,256 (δ = 3: 1,67 gegen 1,48)", "2,256", CMP[4]["cross"])
    r3, e3 = E.compare_at(CMP[4], 3.0)
    close("1,67", r3)
    close("1,48", e3)


# ------------------------------------------------------------------------------------------------------------------ Befunde und Korrekturen
def test_befunde_und_korrekturen():
    p_eq = 1 / RL[4].mean
    rules10, simple10 = RL[4].cdf(10), 1 - (1 - p_eq) ** 10
    claims_seq("innerhalb der ersten 10 Beobachtungen in 7,47 % der Reihen, die einfache Karte in 10,38 %: das sind 28 % weniger (nicht 33 %)",
               [100 * rules10, 100 * simple10, 100 * (1 - rules10 / simple10), 33], start="Beobachtungen in ")
    # Zustandszahlen
    says("kommt auf 1 / 41 / 1221 / 1497 Zustände")
    assert [H.build_chain(r).n for r in (1, 2, 3, 4)] == [1, 41, 1221, 1497]
    says("mit Regel 4 = 9 und 10 in Folge sind es 1589 und 1681")
    assert [H.build_chain(4, L).n for L in (9, 10)] == [1589, 1681]
    says("der Plan nennt 2041 / 1545 / 1497 / 1497 Zustände")
    says("1555 / 1233 / 1221 / 15 165 Zustände")        # Fensterkette: tests/test_oracle_karte.py::test_state_counts_of_both_chains_are_documented
    # Maße für alle drei Zusatzsätze: 12 Zellen
    says("12 Zellen, Abstand höchstens 1e-9")
    worst = 0.0
    cells = 0
    for r in (2, 3, 4):
        chain = H.build_chain(r)
        for d in (0.5, 1.0, 2.0, 3.0):
            m = H.delay_measures(chain, d)
            worst = max(worst, abs(m.pollak - m.zero), abs(m.lorden - m.zero))
            assert m.steady < m.zero
            cells += 1
    assert cells == 12 and worst <= 1e-9
    # W/ARL0
    c1000 = E.count_distribution(4, True, 3.0, 8, 1000)
    claims_seq("(vier Regeln: 10,899 gegen 10,866, also 0,3 %)", [c1000["t_over_arl0"], c1000["mean"], 100 * abs(c1000["t_over_arl0"] / c1000["mean"] - 1)])
    for r in (1, 2, 3, 4):
        c = E.count_distribution(r, True, 3.0, 8, 1000)
        assert abs(c["mean"] / c["t_over_arl0"] - 1) <= 0.005
    c200 = E.count_distribution(4, True, 3.0, 8, 200)
    claims_seq("(vier Regeln: 2,180 gegen 2,147)", [c200["t_over_arl0"], c200["mean"]])
    says("bei Fenstern zu 200 Beobachtungen sind es bis rund 1,5 %")
    dev200 = max(abs(E.count_distribution(r, True, 3.0, 8, 200)["mean"] / E.count_distribution(r, True, 3.0, 8, 200)["t_over_arl0"] - 1) for r in (1, 2, 3, 4))
    assert 0.0145 <= dev200 <= 0.0155
    # 0,863 des Plans gegen 0,862
    m2 = H.delay_measures(H.build_chain(4), 2.0)
    close("0,862", m2.steady / (1 / K.p_alarm(CMP[4]["c_eq"], 2.0, True)))
    says("der Plan nannte 0,863 statt 0,862")


# ------------------------------------------------------------------------------------------------------------------ Abschnitt A
def test_abschnitt_a_einfache_karte():
    rl = RL[1]
    claim("ARL₀ = 370,398 (ARL₀ = 1/p mit p = 2Φ(−3)); einseitig 740,80", "370,398", rl.mean)
    close("740,80", 1 / K.p_alarm(3.0, 0.0, False))
    claim("c = 2,9997 zweiseitig, c = 2,7818 einseitig", "2,9997", K.c_for_arl(370, True))
    close("2,7818", K.c_for_arl(370, False))
    says("19 / 257 / 1109 / 1704 bei ±3σ, 19 / 257 / 1107 / 1702 bei ARL₀ = 370 gesetzt")
    assert [rl.quantile(q) for q in (0.05, 0.5, 0.95, 0.99)] == [19, 257, 1109, 1704]
    assert [K.geometric_quantile(1 / 370, q) for q in (0.05, 0.5, 0.95, 0.99)] == [19, 257, 1107, 1702]
    claim("P(T ≤ 100) = 23,69 % (±3σ) bzw. 23,71 % (370 gesetzt)", "23,69", 100 * rl.cdf(100))
    close("23,71", 100 * (1 - (1 - 1 / 370) ** 100))
    claims_seq("P(T ≤ 99), ist 23,48 % (±3σ) bzw. 23,50 % (370 gesetzt)", [100 * rl.cdf(99), None, 100 * (1 - (1 - 1 / 370) ** 99), None], start="ist ")
    claim("P(T > 1000) = 6,70 %", "6,70", 100 * rl.sf(1000))


def test_abschnitt_a_tabelle_arl0_und_schwelle():
    rows = table("| ARL₀ (einfache Karte) |")
    arl0 = [50, 100, 200, 370, 500, 1000, 5000]
    header = next(ln for ln in README.splitlines() if ln.startswith("| ARL₀ (einfache Karte) |"))
    assert [int(c) for c in numbers_in(header)] == arl0
    for cell, g in zip(rows[0][1:], arl0):
        close(cell, 100 * (1 - (1 - 1 / g) ** 100))
    for cell, g in zip(rows[1][1:], arl0):
        assert int(cell) == K.geometric_quantile(1 / g, 0.5), g
    says("c = 2,0 gibt ARL₀ = 21,98 (99,05 % der Reihen mit Alarm innerhalb der ersten 100); c = 2,5: 80,52 (71,34 %); c = 3,0: 370,40 (23,69 %); c = 3,5: 2149,34 (4,55 %)")
    for c, arl, share in ((2.0, "21,98", "99,05"), (2.5, "80,52", "71,34"), (3.0, "370,40", "23,69"), (3.5, "2149,34", "4,55")):
        p = K.p_alarm(c, 0.0, True)
        close(arl, 1 / p)
        close(share, 100 * (1 - (1 - p) ** 100))


# ------------------------------------------------------------------------------------------------------------------ Abschnitt B
def test_abschnitt_b_tabelle():
    rows = table("| Regeln | ARL₀ | Median |")
    assert [r[0] for r in rows] == ["1", "1+2", "1+2+3", "1+2+3+4"]
    for row, r in zip(rows, (1, 2, 3, 4)):
        rl = RL[r]
        values = [rl.mean, rl.quantile(0.5), rl.quantile(0.05), rl.quantile(0.95), rl.quantile(0.99), 100 * rl.cdf(10), 100 * rl.cdf(100), 100 * rl.sf(1000), _gap(r)]
        for cell, v in zip(row[1:], values):
            close(cell, v)


def test_abschnitt_b_gleiche_arl0():
    rl4 = RL[4]
    p = 1 / rl4.mean
    says("Schwelle c = 2,5459")
    close("2,5459", K.c_for_arl(rl4.mean, True))
    claims_seq("P(T ≤ 100) 66,68 % gegen 66,58 %, Median 65 gegen 64, 95-%-Quantil 268 gegen 274",
               [100 * rl4.cdf(100), 100 * (1 - (1 - p) ** 100), rl4.quantile(0.5), K.geometric_quantile(p, 0.5), None, rl4.quantile(0.95), K.geometric_quantile(p, 0.95)], start="P(T ≤ 100) ")
    t = np.arange(len(rl4.s))
    diff = np.abs((1 - rl4.s) - (1 - (1 - p) ** t))
    claim("der größte Abstand der Verteilungsfunktionen ist 0,0344 bei t = 7", "0,0344", float(diff.max()))
    assert int(diff.argmax()) == 7
    says("innerhalb der ersten 10 Beobachtungen alarmieren die Regeln in 7,47 % der Reihen gegen 10,38 %, innerhalb der ersten 20 in 17,39 % gegen 19,68 %")
    for w, v in (("7,47", 100 * rl4.cdf(10)), ("10,38", 100 * (1 - (1 - p) ** 10)), ("17,39", 100 * rl4.cdf(20)), ("19,68", 100 * (1 - (1 - p) ** 20))):
        close(w, v)


# ------------------------------------------------------------------------------------------------------------------ Abschnitt C
def test_abschnitt_c_tabelle_und_kreuzung():
    says("c = 2,8454 / 2,6727 / 2,5459 für 1+2 / 1+2+3 / 1+2+3+4")
    for w, r in (("2,8454", 2), ("2,6727", 3), ("2,5459", 4)):
        close(w, CMP[r]["c_eq"])
    rows = table("| δ | 1+2: Regeln / einfach |")
    for row, d in zip(rows, (0.5, 1.0, 2.0, 3.0, 4.0)):
        close(row[0], d)
        for cell, r in zip(row[1:], (2, 3, 4)):
            nums = numbers_in(cell)
            a, b = E.compare_at(CMP[r], d)
            assert len(nums) == 3, cell
            close(nums[0], a)
            close(nums[1], b)
            close(nums[2], a / b)
    says("bei δ\\* = 3,455 / 2,705 / 2,256 (1+2 / 1+2+3 / 1+2+3+4)")
    for w, r in (("3,455", 2), ("2,705", 3), ("2,256", 4)):
        close(w, CMP[r]["cross"])


def test_abschnitt_c_verteilung_der_verzoegerung_bei_delta_1():
    rd = E.run_length(4, True, 3.0, 8, 1.0)
    ps = K.p_alarm(CMP[4]["c_eq"], 1.0, True)
    sf = lambda t: (1 - ps) ** t   # noqa: E731
    cdf = lambda t: 1 - sf(t)      # noqa: E731
    vals = [rd.mean, 1 / ps, rd.quantile(0.5), K.geometric_quantile(ps, 0.5), rd.quantile(0.95), K.geometric_quantile(ps, 0.95), rd.quantile(0.99), K.geometric_quantile(ps, 0.99),
            100 * rd.cdf(10), 100 * cdf(10), 100 * rd.sf(50), 100 * sf(50)]
    says("Mittel 9,22 gegen 16,32, Median 8 gegen 11, 95 % innerhalb von 22 gegen 48, 99 % innerhalb von 32 gegen 73, P(D ≤ 10) 69,79 % gegen 46,85 %, **P(D > 50) 0,05 % gegen 4,24 %**")
    for w, v in zip(("9,22", "16,32", "8", "11", "22", "48", "32", "73", "69,79", "46,85", "0,05", "4,24"), vals):
        close(w, v)


def test_abschnitt_c_einfache_karte_3_sigma():
    line = next(ln for ln in README.splitlines() if ln.startswith("Einfache Karte ±3σ (zweiseitig, geschlossen)"))
    body = line.split("Verzögerung über 100 Beobachtungen: ", 1)[1]
    parts = [p.strip() for p in body.split(";")]
    assert len(parts) == 5
    for part, d in zip(parts, (0.5, 1.0, 1.5, 2.0, 3.0)):
        rl = E.run_length(1, True, 3.0, 8, d)
        part = part.replace("**", "").rstrip(".")
        head, rest = part.split(": ")
        close(head, d)
        cells = [c.strip() for c in rest.split("/")]
        close(cells[0], rl.mean)
        close(cells[1], rl.quantile(0.5))
        close(cells[2], rl.quantile(0.95))
        if cells[3].startswith("unter"):
            assert 100 * rl.sf(100) < 0.001
        else:
            close(cells[3], 100 * rl.sf(100))


# ------------------------------------------------------------------------------------------------------------------ Abschnitt D
def test_abschnitt_d_tabelle():
    rows = table("| δ | einfache Karte (gleiche ARL₀) |")
    chain = H.build_chain(4)
    for row, d in zip(rows, (0.5, 1.0, 2.0, 3.0)):
        m = H.delay_measures(chain, d)
        simple = 1 / K.p_alarm(CMP[4]["c_eq"], d, True)
        close(row[0], d)
        close(row[1], simple)
        close(row[2], m.zero)
        close(row[3], m.steady)
        close(row[4], m.steady / m.zero)
        a, b = row[5].split(";")
        close(a, m.zero / simple)
        close(b, m.steady / simple)
        assert m.pollak == pytest.approx(m.zero, abs=1e-9) and m.lorden == pytest.approx(m.zero, abs=1e-9) and m.pollak_nu == 0
        # dieselbe Seite von 1 in beiden gerechneten Maßen
        assert (m.zero / simple < 1) == (m.steady / simple < 1)
    says("Maximum über ν ≤ 600, erreicht bei ν = 0")


def test_abschnitt_d_einfache_karte_alle_masse_gleich():
    for d in (0.5, 1.0, 2.0, 3.0):
        m = E.delay_measures(1, True, 3.0, 8, d)
        assert m.zero == m.steady == m.pollak == m.lorden == pytest.approx(1 / K.p_alarm(3.0, d, True), rel=1e-12)


# ------------------------------------------------------------------------------------------------------------------ Abschnitt E
def test_abschnitt_e_fassungen():
    rows = E.rule4_versions(1.0)
    says("8 in Folge 91,75 / 9,22; 9 in Folge 109,05 / 9,70; 10 in Folge 120,08 / 10,05; ohne Regel 4 132,89 / 10,95")
    for w, v in zip(("91,75", "9,22", "109,05", "9,70", "120,08", "10,05", "132,89", "10,95"), [x for row in rows for x in (row["arl0"], row["arl1"])]):
        close(w, v)
    claim("zwischen 91,75 und 120,08 liegt (Faktor bis 1,31)", "1,31", rows[2]["arl0"] / rows[0]["arl0"])


# ------------------------------------------------------------------------------------------------------------------ Abschnitt F
def test_abschnitt_f_alarme_je_fenster():
    c1 = E.count_distribution(1, True, 3.0, 8, 1000)
    says("Binomial(1000, p), Mittel 2,70; P(0 Alarme) = 6,70 %, P(≥ 5) = 13,68 %, P(≥ 8) = 0,65 %; Quantile 5 % / 50 % / 95 %: 0 / 3 / 6")
    for w, v in (("2,70", c1["mean"]), ("6,70", 100 * c1["p0"]), ("13,68", 100 * c1["p_ge5"]), ("0,65", 100 * c1["p_ge8"])):
        close(w, v)
    assert (c1["q05"], c1["q50"], c1["q95"]) == (0, 3, 6)
    cs = {r: E.count_distribution(r, True, 3.0, 8, 1000) for r in (2, 3, 4)}
    says("Mittel 4,433 / 7,512 / 10,866 für 1+2 / 1+2+3 / 1+2+3+4 (W/ARL₀ = 4,436 / 7,525 / 10,899)")
    for w, r in (("4,433", 2), ("7,512", 3), ("10,866", 4)):
        close(w, cs[r]["mean"])
    for w, r in (("4,436", 2), ("7,525", 3), ("10,899", 4)):
        close(w, cs[r]["t_over_arl0"])
    says("P(0 Alarme) = 1,163 / 0,048 / 0,001 %; P(≥ 5) = 45,56 / 87,31 / 98,63 %; Median 4 / 7 / 11, 95-%-Quantil 8 / 12 / 16")
    for w, r in (("1,163", 2), ("0,048", 3), ("0,001", 4)):
        close(w, 100 * cs[r]["p0"])
    for w, r in (("45,56", 2), ("87,31", 3), ("98,63", 4)):
        close(w, 100 * cs[r]["p_ge5"])
    assert [cs[r]["q50"] for r in (2, 3, 4)] == [4, 7, 11] and [cs[r]["q95"] for r in (2, 3, 4)] == [8, 12, 16]


# ------------------------------------------------------------------------------------------------------------------ Abschnitt G
def test_abschnitt_g_ausblick_einseitig():
    says("einseitig, ARL₀ = 370, δ = 1: einfache Karte 26,746 Beobachtungen")
    close("26,746", 1 / K.p_alarm(K.c_for_arl(370, False), 1.0, False))
    says("ein CUSUM mit k = 0,5 braucht 8,573")      # Zahl der Reihe, hier nicht gerechnet


# ------------------------------------------------------------------------------------------------------------------ Handbeispiele und Texte
def test_handbeispiele_zahlen_der_readme():
    says("Handbeispiele (38,6164; 118900/3079; 199/10000; 27501/1000000; 0,2371)")
    from fractions import Fraction as Fr
    assert Fr(118900, 3079) == (1 + Fr(1, 10) + Fr(1, 10) * Fr(89, 100)) / (1 - Fr(89, 100) - Fr(1, 10) * Fr(89, 100) ** 2)
    close("38,6164", 118900 / 3079)
    close("0,2371", float(1 - (1 - Fr(1, 370)) ** 100))


def test_annahmentabelle_in_app_und_readme_ist_dieselbe():
    """Zwei Fassungen desselben Textes (App und README) werden verglichen, nicht von Hand synchron gehalten."""
    assert T.ASSUMPTIONS_MARKDOWN.strip() in README


def test_zahlen_der_annahmentabelle():
    rows = E.rule4_versions(1.0)
    claims_seq("Mit 9 in Folge steigt die ARL₀ von 91,75 auf 109,05", [None, rows[0]["arl0"], rows[1]["arl0"]])
    close("1,31", rows[2]["arl0"] / rows[0]["arl0"])
    m = H.delay_measures(H.build_chain(4), 1.0)
    claims_seq("(bei δ = 1: 8,54 gegen 9,22 Beobachtungen)", [None, m.steady, m.zero])


def test_testzahl_der_readme_stimmt_mit_der_suite_ueberein():
    """Die Gesamtzahl der Tests in der README gleicht `pytest --collect-only` (eine Einzeldatei mit „37 Tests“ wäre keine Gesamtzahl: hier steht die Gesamtzahl)."""
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT, timeout=300).stdout
    collected = int(re.search(r"(\d+) tests? collected", out).group(1))
    m = re.search(r"Die Suite hat (\d+) Tests", README)
    assert m and int(m.group(1)) == collected, (m and m.group(1), collected)
