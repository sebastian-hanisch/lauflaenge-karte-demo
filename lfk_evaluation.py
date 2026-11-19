"""Kennzahlen der Karte (exakt aus geschlossener Form oder Kette), Simulationszusammenfassungen und Verdikt-Logik.

Alle Verdikte sind reine Funktionen `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`, `tests/test_verdicts.py` prüft jede Schwelle
und jede Zahl im Text an handgesetzten Werten. Zähleinheiten: Beobachtungen (Lauflänge, Verzögerung), Läufe (Simulation), Reihen (Anteile),
Fenster (Alarmzahl). Das Maß der Verzögerung steht in jedem Satz.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

import lfk_chain as H
import lfk_chart as K
import lfk_constants as C
import lfk_renewal as R


# ---------------------------------------------------------------------------------------------------------------------------- Format
def de(x: float, nd: int = 2) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{nd}f}".replace(".", ",")


def de_pct(share: float, nd: int | None = 2) -> str:
    """Anteil (0 bis 1) als Prozentzahl mit Dezimalkomma und Leerzeichen vor dem Zeichen.

    nd = None: angepasst. Anteile unter 0,01 % bekommen drei Stellen, Anteile unter 0,0005 % und Rauschen der Gleitkommarechnung werden als „< 0,001 %“ gezeigt
    (nie „0,00 %“: ein seltenes Ereignis ist nicht unmöglich, DEMO-PLAYBOOK §11a Punkt 4)."""
    if nd is None:
        pct = 100.0 * share
        if pct < 5e-4:
            return "< 0,001 %"
        nd = 3 if pct < 0.01 else 2
    return f"{100.0 * share:.{nd}f}".replace(".", ",") + " %"


def de_arl(x: float) -> str:
    """ARL: zwei Stellen unter 100 (91,75), sonst eine (370,4)."""
    return de(x, 2 if x < 100 else 1)


def fmt_c(c: float) -> str:
    """Schwelle in σ: ganze Zahlen ohne Nachkommastellen, sonst bis vier Stellen ohne überflüssige Nullen."""
    s = f"{c:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")


# ---------------------------------------------------------------------------------------------------------------------------- Einstellungen
@dataclass(frozen=True)
class Settings:
    """Alle Regler. Immer mit Schlüsselwörtern aufrufen (ein eingefügtes Feld verschiebt sonst still Werte)."""

    rules: int = 1
    side: str = "zwei"
    c: float = C.DEFAULT_C
    run4: int = C.DEFAULT_RULE4_RUN
    seed: int = C.DEFAULT_SEED
    n: int = C.DEFAULT_SERIES_N
    nu: int = C.DEFAULT_NU
    delta: float = 0.0
    t0: int = C.DEFAULT_T0
    runs: int = C.DEFAULT_RUNS
    window: int = C.DEFAULT_WINDOW
    nwin: int = C.DEFAULT_NWIN
    sim_seed: int = C.DEFAULT_SIM_SEED
    simple: bool = True

    @property
    def two_sided(self) -> bool:
        """Zusatzregeln gibt es nur zweiseitig; einseitig ist nur für Regel 1 allein definiert und gemessen."""
        return self.side == "zwei" or self.rules > 1

    @property
    def c_eff(self) -> float:
        """Die Zonen der Zusatzregeln liegen fest bei 1σ, 2σ, 3σ: mit Zusatzregeln gilt c = 3."""
        return self.c if self.rules == 1 else 3.0

    def card_args(self) -> tuple:
        return (self.rules, self.two_sided, self.c_eff, self.run4)


def card_label(rule_count: int, two_sided: bool, c: float, run4: int = 8) -> str:
    """Name der Karte für Texte: „zweiseitige Karte mit Grenzen ±3σ“, „Regeln 1+2+3+4 (Regel 4: 8 in Folge)“."""
    if rule_count == 1:
        return f"{'zweiseitige Karte mit Grenzen ±' if two_sided else 'einseitige Karte mit Grenze '}{fmt_c(c)}σ"
    extra = f", Regel 4 mit {run4} in Folge" if rule_count == 4 else ""
    return f"zweiseitige {C.RULE_NAMES[rule_count]}{extra}"


# ---------------------------------------------------------------------------------------------------------------------------- Lauflänge
@dataclass(eq=False)
class RunLength:
    """Verteilung der Lauflänge T (Beobachtungen bis einschließlich der alarmierenden), zero-state.

    Einfache Karte: exakt geometrisch (p je Beobachtung, gedächtnislos), geschlossene Form. Zusatzregeln: Überlebensfunktion `s` aus der Kette."""

    mean: float
    p: float | None
    s: np.ndarray | None

    def sf(self, t: int) -> float:
        """P(T > t)."""
        if t < 0:
            return 1.0
        if self.p is not None:
            return float(K.geometric_sf(self.p, t))
        return float(self.s[t]) if t < len(self.s) else 0.0

    def cdf(self, t: int) -> float:
        return 1.0 - self.sf(t)

    def quantile(self, q: float) -> int:
        if self.p is not None:
            return K.geometric_quantile(self.p, q)
        return H.quantile(self.s, q)

    def survival_curve(self, tmax: int) -> np.ndarray:
        """P(T > t) für t = 0..tmax."""
        if self.p is not None:
            return K.geometric_sf(self.p, np.arange(tmax + 1))
        out = np.zeros(tmax + 1)
        m = min(tmax + 1, len(self.s))
        out[:m] = self.s[:m]
        return out


def run_length(rule_count: int, two_sided: bool, c: float, run4: int, delta: float) -> RunLength:
    """Lauflängenverteilung der Karte bei einem Mittelwert delta (zero-state); delta = 0 ist die Verteilung bis zum Fehlalarm."""
    if rule_count == 1:
        p = K.p_alarm(c, delta, two_sided)
        return RunLength(1.0 / p, p, None)
    chain = H.build_chain(rule_count, run4)
    mean = float(H.arl_vector(chain, delta)[0])
    return RunLength(mean, None, H.survival(chain, delta, H.tail_length(mean)))


def arl1_curve(rule_count: int, two_sided: bool, c: float, run4: int, deltas) -> list[float]:
    """ARL₁ (zero-state, Beobachtungen) der Karte für jede Sprunggröße in `deltas`."""
    if rule_count == 1:
        return [1.0 / K.p_alarm(c, d, two_sided) for d in deltas]
    chain = H.build_chain(rule_count, run4)
    return [float(H.arl_vector(chain, d)[0]) for d in deltas]


def delay_measures(rule_count: int, two_sided: bool, c: float, run4: int, delta: float) -> H.DelayMeasures:
    """zero-state, steady-state (quasistationär), Pollak und Lorden. Die einfache Karte ist gedächtnislos: alle vier gleich 1/p(delta)."""
    if rule_count == 1:
        v = 1.0 / K.p_alarm(c, delta, two_sided)
        return H.DelayMeasures(v, v, v, 0, v, 0)
    return H.delay_measures(H.build_chain(rule_count, run4), delta)


def conditional_delay(rule_count: int, two_sided: bool, c: float, run4: int, delta: float, nu: int) -> RunLength:
    """Verteilung der Verzögerung D = T - nu, wenn nach nu Beobachtungen ohne Alarm der Mittelwert auf delta springt (exakt, Beobachtungen).

    Einfache Karte: gedächtnislos, also gleich der Lauflänge bei delta. Zusatzregeln: Start in der Zustandsverteilung nach nu ruhigen Beobachtungen."""
    if rule_count == 1 or nu == 0:
        return run_length(rule_count, two_sided, c, run4, delta)
    chain = H.build_chain(rule_count, run4)
    mean = float(H.arl_vector(chain, delta) @ H.state_distribution(chain, nu)[0])
    return RunLength(mean, None, H.conditional_delay_survival(chain, delta, nu, H.tail_length(mean)))


def p_alarm_before(rule_count: int, two_sided: bool, c: float, run4: int, nu: int) -> float:
    """P(T <= nu) ohne Änderung: Wahrscheinlichkeit eines Fehlalarms innerhalb der ersten nu Beobachtungen."""
    if nu <= 0:
        return 0.0
    return run_length(rule_count, two_sided, c, run4, 0.0).cdf(nu)


# ---------------------------------------------------------------------------------------------------------------------------- Alarme je Fenster
def count_distribution(rule_count: int, two_sided: bool, c: float, run4: int, window: int) -> dict:
    """Verteilung der Alarmzahl in einem Fenster von `window` Beobachtungen mit Neustart nach Alarm (Erneuerungsrechnung); einfache Karte zusätzlich Binomial."""
    rl = run_length(rule_count, two_sided, c, run4, 0.0)
    if rule_count == 1:
        f = np.zeros(window + 1)
        f[1:] = rl.p * (1.0 - rl.p) ** np.arange(window)
    else:
        f = R.pmf_from_survival(rl.s, window) if window < len(rl.s) else np.concatenate([R.pmf_from_survival(rl.s, len(rl.s) - 1), np.zeros(window - len(rl.s) + 1)])
    pmf, _ = R.renewal_counts(f, window)
    out = {"pmf": pmf, "arl0": rl.mean, "t_over_arl0": window / rl.mean, **R.count_summary(pmf)}
    if rule_count == 1:
        pb = R.binomial_counts(window, rl.p)
        out["binomial"] = pb
        out["max_gap_binomial"] = float(np.max(np.abs(pb[: len(pmf)] - pmf[: len(pb)]))) if len(pmf) <= len(pb) else float(np.max(np.abs(pb - pmf[: len(pb)])))
    return out


# ---------------------------------------------------------------------------------------------------------------------------- Vergleich bei gleicher ARL₀
def equal_arl_comparison(rule_count: int, run4: int, deltas=C.DELTA_GRID) -> dict:
    """Zusatzregeln gegen die einfache zweiseitige Karte mit gleicher ARL₀ (zero-state): ARL₁ je Sprunggröße, Verhältnis, Kreuzung delta*."""
    chain = H.build_chain(rule_count, run4)
    arl0 = float(H.arl_vector(chain, 0.0)[0])
    c_eq = K.c_for_arl(arl0, True)
    rules_arl1 = [float(H.arl_vector(chain, d)[0]) for d in deltas]
    simple_arl1 = [1.0 / K.p_alarm(c_eq, d, True) for d in deltas]

    def gap(d: float) -> float:
        return float(H.arl_vector(chain, d)[0]) - 1.0 / K.p_alarm(c_eq, d, True)

    lo, hi = C.CROSS_BRACKET
    cross = float(brentq(gap, lo, hi, xtol=1e-6)) if gap(lo) * gap(hi) < 0 else None
    return {
        "rule_count": rule_count, "run4": run4, "arl0": arl0, "c_eq": c_eq, "deltas": list(deltas),
        "rules_arl1": rules_arl1, "simple_arl1": simple_arl1,
        "ratio": [r / s for r, s in zip(rules_arl1, simple_arl1)], "cross": cross,
    }


def rule4_versions(delta: float) -> list[dict]:
    """Fassungsfalle: ARL₀ und ARL₁(delta) der vier Regeln mit Regel 4 = 8, 9, 10 in Folge und ohne Regel 4 (zweiseitig, zero-state)."""
    rows = []
    for label, rc, run4 in (("Regel 4: 8 in Folge", 4, 8), ("Regel 4: 9 in Folge", 4, 9), ("Regel 4: 10 in Folge", 4, 10), ("ohne Regel 4 (Regeln 1+2+3)", 3, 8)):
        chain = H.build_chain(rc, run4)
        rows.append({"label": label, "arl0": float(H.arl_vector(chain, 0.0)[0]), "arl1": float(H.arl_vector(chain, delta)[0]) if delta > 0 else None})
    return rows


# ---------------------------------------------------------------------------------------------------------------------------- Simulation
def sim_run_summary(lengths: np.ndarray, rl: RunLength, t0: int) -> dict:
    """Mittel ± Standardfehler, Median und P(T <= t0) der simulierten Lauflängen neben der exakten Verteilung (alles in Beobachtungen bzw. Reihen)."""
    r = len(lengths)
    mean = float(lengths.mean())
    se = float(lengths.std(ddof=1) / math.sqrt(r)) if r > 1 else float("nan")
    share = float(np.mean(lengths <= t0))
    se_share = math.sqrt(max(share * (1.0 - share), 1e-12) / r)
    exact = rl.cdf(t0)
    return {
        "runs": r, "mean": mean, "se": se, "median": float(np.median(lengths)), "share": share, "se_share": se_share, "exact_share": exact,
        "z_mean": (mean - rl.mean) / se if se > 0 else 0.0,
        "z_share": (share - exact) / math.sqrt(max(exact * (1.0 - exact), 1e-12) / r),
    }


# ---------------------------------------------------------------------------------------------------------------------------- Verdikte
def verdict_series(*, label: str, first: int | None, rule: int, n: int, nu: int, delta: float, p_before: float, arl0: float, delay: dict | None, d_cdf: float | None) -> tuple[str, str]:
    """Verdikt zur Einzelreihe: Alarm vor oder nach dem Sprung, daneben die exakten Größen derselben Karte.

    delay = {mean, median, q95} der Verzögerung D bedingt auf keinen Alarm vor nu (Beobachtungen), d_cdf = P(D <= d) dieser Ziehung."""
    head = f"{label[0].upper()}{label[1:]}, ARL₀ = {de_arl(arl0)} Beobachtungen (zero-state)."
    jump_in = delta > 0 and nu < n
    exact_nu = f" Wahrscheinlichkeit eines Fehlalarms innerhalb der ersten {nu} Beobachtungen (exakt): {de_pct(p_before, None)}." if nu > 0 else ""
    if not jump_in:
        if first is None:
            return "info", f"{head} Keine Änderung in dieser Reihe, und die Karte bleibt in {n} Beobachtungen ruhig. Die Reihe zeigt eine Ziehung, der Mittelwert ARL₀ verteilt sich breit."
        return "warning", f"{head} Keine Änderung in dieser Reihe, aber ein Alarm bei Beobachtung {first} (Regel {rule}): ein Fehlalarm. Die Reihe zeigt eine Ziehung, ARL₀ ist nur der Mittelwert vieler solcher Wartezeiten."
    tail = f" Erwartung bei einem Sprung um {fmt_c(delta)}σ nach {nu} Beobachtungen, bedingt auf keinen Alarm davor (exakt): im Mittel {de(delay['mean'], 1)} Beobachtungen bis zum Alarm, Median {delay['median']}, 95 % innerhalb von {delay['q95']}."
    if first is None:
        return "warning", f"{head}{exact_nu} In dieser Reihe bleibt der Sprung in den restlichen {n - nu} Beobachtungen unentdeckt.{tail}"
    if first <= nu:
        return "warning", f"{head} Alarm bei Beobachtung {first} (Regel {rule}), vor dem Sprung bei {nu}: ein Fehlalarm.{exact_nu}{tail}"
    d = first - nu
    where = f" Diese Ziehung liegt bei P(D ≤ {d}) = {de_pct(d_cdf, 1)}." if d_cdf is not None else ""
    return "success", f"{head} Alarm bei Beobachtung {first} (Regel {rule}), nach dem Sprung bei {nu}: Verzögerung D = {d} Beobachtungen.{where}{tail}"


def verdict_runlength(*, label: str, arl0: float, median: int, t0: int, p_t0: float, p_over_1000: float, sim: dict | None) -> tuple[str, str]:
    """Verdikt zu „ARL₀ ist ein Mittel, die Lauflänge eine Verteilung“ (zero-state, exakt, Simulation als Band)."""
    text = (f"{label[0].upper()}{label[1:]}: ARL₀ = {de_arl(arl0)} Beobachtungen (zero-state), Median {median}; "
            f"{de_pct(p_t0)} der Reihen haben einen Alarm schon innerhalb der ersten {t0} Beobachtungen, {de_pct(p_over_1000, None)} bleiben über 1000 Beobachtungen ruhig (exakt).")
    if sim is not None:
        text += f" Simulation mit {sim['runs']} Läufen: Mittel {de(sim['mean'], 1)} ± {de(sim['se'], 1)}, {de_pct(sim['share'], 2)} der Reihen innerhalb von {t0}."
    kind = "warning" if p_t0 >= 0.10 else "info"
    return kind, text


def verdict_windows(*, label: str, window: int, mean: float, p0: float, p_ge5: float, t_over_arl0: float, sim: dict | None) -> tuple[str, str]:
    """Verdikt zu den Alarmen je Fenster (Neustart nach Alarm, exakt aus der Erneuerungsrechnung; Simulation als Band)."""
    text = (f"{label[0].upper()}{label[1:]}, Fenster zu {window} Beobachtungen: {de(mean)} Alarme im Mittel (T/ARL₀ = {de(t_over_arl0)}); "
            f"{de_pct(p0, None)} der Fenster ohne Alarm, {de_pct(p_ge5, None)} mit fünf oder mehr (exakt).")
    if sim is not None:
        text += f" Simulation mit {sim['n']} Fenstern: Mittel {de(sim['mean'])} ± {de(sim['se'])}, {de_pct(sim['p0'], 2)} ohne Alarm."
    return "info", text


def verdict_delay(*, label: str, delta: float, arl1: float, median: int, q95: int, p_over_100: float, measures: H.DelayMeasures | None) -> tuple[str, str]:
    """Verdikt zur Verzögerung nach einem Sprung (zero-state; bei Zusatzregeln zusätzlich steady-state, Pollak, Lorden)."""
    if delta <= 0:
        return "info", f"{label[0].upper()}{label[1:]}: ohne Sprung (δ = 0) gibt es keine Verzögerung, nur die Wartezeit bis zum Fehlalarm (Experimente 1 bis 3)."
    text = (f"{label[0].upper()}{label[1:]}, Sprung um {fmt_c(delta)}σ: ARL₁ = {de(arl1, 2 if arl1 < 20 else 1)} Beobachtungen (zero-state), Median {median}, "
            f"95 % innerhalb von {q95}; {de_pct(p_over_100, None)} der Reihen länger als 100 Beobachtungen ohne Alarm (exakt).")
    if measures is None:
        text += " Bei dieser gedächtnislosen Karte sind zero-state, Pollak und Lorden gleich."
    else:
        text += (f" Maße bei den Zusatzregeln: steady-state (quasistationär) {de(measures.steady, 2)}, Pollak {de(measures.pollak, 2)}, "
                 f"Lorden {de(measures.lorden, 2)}.")
    return ("warning" if p_over_100 >= 0.25 else "info"), text


def verdict_rules(*, rule_count: int, run4: int, delta: float, arl0: float, c_eq: float, arl1_rules: float | None, arl1_simple: float | None, cross: float | None,
                  steady: float | None) -> tuple[str, str]:
    """Verdikt zum Vergleich Zusatzregeln gegen einfache Karte mit gleicher ARL₀ (zero-state, zweiseitig, δ in σ)."""
    name = C.RULE_NAMES[rule_count] + (f", Regel 4 mit {run4} in Folge" if rule_count == 4 else "")
    text = f"{name[0].upper()}{name[1:]}: ARL₀ = {de(arl0, 2)} Beobachtungen (statt 370,4 der einfachen Karte), verglichen mit der einfachen Karte gleicher ARL₀ (c = {de(c_eq, 4)}σ)."
    if arl1_rules is None or delta <= 0:
        return "info", text + " Für den Vergleich der Verzögerung δ > 0 wählen."
    faster = arl1_rules < arl1_simple
    text += f" Bei δ = {fmt_c(delta)}σ: ARL₁ {de(arl1_rules, 2)} gegen {de(arl1_simple, 2)} Beobachtungen (zero-state)"
    if steady is not None:
        text += f", steady-state {de(steady, 2)} (quasistationär)"
    text += "."
    if cross is not None:
        text += f" Kreuzung bei δ* = {de(cross, 2)}σ: darüber ist die einfache Karte schneller, darunter die Zusatzregeln."
    return ("success" if faster else "warning"), text


def compare_at(cmp: dict, delta: float) -> tuple[float, float]:
    """(ARL₁ des Regelsatzes, ARL₁ der einfachen Karte gleicher ARL₀) bei delta, auch zwischen den Rasterpunkten von `equal_arl_comparison` (zero-state, Beobachtungen)."""
    if delta in cmp["deltas"]:
        k = cmp["deltas"].index(delta)
        return cmp["rules_arl1"][k], cmp["simple_arl1"][k]
    chain = H.build_chain(cmp["rule_count"], cmp["run4"])
    return float(H.arl_vector(chain, delta)[0]), 1.0 / K.p_alarm(cmp["c_eq"], delta, True)


def kernfragen(*, first: int | None, rule: int, n: int, label: str, arl0: float, median: int, t0: int, p_t0: float, p_over_1000: float, window: int, win_mean: float,
               p0: float, p_ge5: float, delta: float, arl1: float | None, delay_median: int | None, delay_q95: int | None, four_arl0: float,
               four_arl1: float | None, four_simple: float | None, four_cross: float | None) -> list[tuple[str, str]]:
    """Die fünf Kernfragen mit je einer Antwortzeile aus den exakten Größen der aktuellen Einstellungen (Beobachtungen, Reihen, Fenster, zero-state)."""
    simple_arl0 = 1.0 / K.p_alarm(3.0, 0.0, True)
    a1 = (f"In der gezeigten Reihe bei Beobachtung {first} (Regel {rule}); das ist eine Ziehung der Lauflänge T, nicht ihr Mittel." if first
          else f"In der gezeigten Reihe bleibt die Karte {n} Beobachtungen ruhig; auch das ist eine Ziehung.")
    a2 = (f"ARL₀ ist das Mittel der Lauflänge, nicht die typische Wartezeit: Median {median} Beobachtungen; {de_pct(p_t0)} der Reihen haben schon innerhalb der ersten {t0} Beobachtungen "
          f"einen Alarm, {de_pct(p_over_1000, None)} bleiben über 1000 Beobachtungen ruhig ({label}, exakt, zero-state).")
    a3 = (f"In einem Fenster zu {window} Beobachtungen gibt es im Mittel {de(win_mean)} Alarme; {de_pct(p0, None)} der Fenster haben keinen, {de_pct(p_ge5)} haben fünf oder mehr "
          f"({label}, Neustart nach Alarm, exakt).")
    if delta > 0 and arl1 is not None:
        a4 = (f"Nach einem Sprung um {fmt_c(delta)}σ im Mittel {de(arl1, 2 if arl1 < 20 else 1)} Beobachtungen bis zum Alarm (zero-state), Median {delay_median}, 95 % innerhalb von {delay_q95} ({label}, exakt).")
    else:
        a4 = "Ohne Sprung (δ = 0) gibt es keine Verzögerung; einen Sprung in der Seitenleiste wählen."
    a5 = f"Die vier Regeln senken die ARL₀ von {de(simple_arl0, 1)} auf {de(four_arl0, 2)} Beobachtungen (zero-state, zweiseitig)"
    if delta > 0 and four_arl1 is not None and four_simple is not None:
        a5 += f"; bei gleicher ARL₀ brauchen sie bei δ = {fmt_c(delta)}σ {de(four_arl1, 2)} gegen {de(four_simple, 2)} Beobachtungen der einfachen Karte"
        if four_cross is not None:
            a5 += f", sind aber oberhalb von δ* = {de(four_cross, 2)}σ langsamer"
    a5 += "."
    return [
        ("Wann schlägt die Karte an?", a1),
        (f"Was heißt „ARL₀ = {de_arl(arl0)}“ für eine einzelne Reihe?", a2),
        ("Wie viele Fehlalarme gibt es in einem Fenster?", a3),
        ("Wie lange dauert es nach einem Sprung?", a4),
        ("Was kosten Zusatzregeln?", a5),
    ]
