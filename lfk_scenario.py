"""Szenario: ausgedachte Normalreihen mit bekanntem Sprung und die Simulation von Lauflängen und Fenstern.

Zufall: SplitMix64 aus lfk_rng (bitgleich auf jeder Plattform). Die Einzelreihe (Experiment 1) nimmt `normal()` (Box-Muller, exakt, 70 bis
110 µs je Zahl, bei n <= 1000 also etwa 0,1 s). Die Simulation von Lauflängen und Fenstern braucht Millionen Zahlen und nimmt `random()` mit
`NormalDist().inv_cdf` (1 µs je Normalzahl); `math.log` und `math.cos` sind nicht plattformfest, darum sind alle Ergebnisse dieser
Simulationen Bänder (nie exakt vergleichen). Alle Simulationen ziehen keine Zahl aus Python-`random`.
"""

from statistics import NormalDist

import numpy as np

import lfk_chart as K
import lfk_constants as C
from lfk_rng import SplitMix64

INV = NormalDist().inv_cdf
_U_MIN = 2.0 ** -54


def make_series(seed: int, n: int, nu: int, delta: float) -> np.ndarray:
    """n Beobachtungen: N(0, 1) für t <= nu, danach N(delta, 1) (nu = Zahl der Beobachtungen vor der Änderung; delta = 0: keine Änderung).

    Die Standardnormalzahlen hängen nicht von nu und delta ab: dieselbe Reihe mit anderem Sprung ist dieselbe Ziehung plus Verschiebung."""
    rng = SplitMix64(seed)
    z = np.array([rng.normal() for _ in range(n)])
    shift = np.where(np.arange(1, n + 1) > nu, delta, 0.0)
    return z + shift


def _u(rng: SplitMix64) -> float:
    u = rng.random()
    return u if u > 0.0 else _U_MIN


def max_runs(arl0: float, rule_count: int, runs: int) -> int:
    """Zahl der Läufe, die das Rechenbudget (Beobachtungen je Klick) erlaubt; 0 heißt: zu teuer, es wird nicht simuliert."""
    budget = C.SIM_BUDGET_SIMPLE if rule_count == 1 else C.SIM_BUDGET_RULES
    allowed = int(budget // max(arl0, 1.0))
    r = min(runs, allowed)
    return r if r >= C.SIM_MIN_RUNS else 0


def _alarm_rules_raw(hist: list, rule_count: int, run4: int) -> bool:
    """Schlägt irgendeine der Regeln 1 bis rule_count an der letzten Beobachtung von `hist` an? (Rohwerte, zweiseitig, c = 3.)"""
    x = hist[-1]
    if abs(x) > 3.0:
        return True
    w = hist[-3:]
    if sum(v > 2.0 for v in w) >= 2 or sum(v < -2.0 for v in w) >= 2:
        return True
    if rule_count >= 3:
        w = hist[-5:]
        if sum(v > 1.0 for v in w) >= 4 or sum(v < -1.0 for v in w) >= 4:
            return True
    if rule_count >= 4 and len(hist) >= run4:
        w = hist[-run4:]
        if all(v > 0 for v in w) or all(v < 0 for v in w):
            return True
    return False


def simulate_run_lengths(rule_count: int, delta: float, runs: int, seed: int, c: float = 3.0, two_sided: bool = True, run4: int = 8) -> np.ndarray:
    """Lauflängen (Beobachtungen bis zum ersten Alarm, zero-state) aus `runs` Läufen.

    Nur Regel 1: Alarm genau dann, wenn die Zufallszahl u in einem der Randbereiche liegt (x = Phi^-1(u) + delta, |x| > c). Zusatzregeln: die
    Regeln arbeiten direkt an den Rohwerten x (anderer Code als die Ketten)."""
    rng = SplitMix64(seed)
    out = np.empty(runs, dtype=np.int64)
    if rule_count == 1:
        hi = float(K.phi(c - delta))                     # u > hi  <=>  x > c
        lo = float(K.phi(-c - delta)) if two_sided else -1.0   # u < lo  <=>  x < -c
        for i in range(runs):
            t = 1
            u = _u(rng)
            while not (u > hi or u < lo):
                t += 1
                u = _u(rng)
            out[i] = t
        return out
    for i in range(runs):
        hist: list = []
        t = 0
        while True:
            t += 1
            hist.append(INV(_u(rng)) + delta)
            if len(hist) > 10:
                hist.pop(0)
            if _alarm_rules_raw(hist, rule_count, run4):
                out[i] = t
                break
    return out


def simulate_window_counts(rule_count: int, window: int, nwin: int, seed: int, c: float = 3.0, two_sided: bool = True, run4: int = 8) -> np.ndarray:
    """Alarmzahl je Fenster von `window` Beobachtungen ohne Änderung (delta = 0); nach jedem Alarm Neustart ohne Vorgeschichte."""
    rng = SplitMix64(seed)
    out = np.zeros(nwin, dtype=np.int64)
    if rule_count == 1:
        hi = float(K.phi(c))
        lo = float(K.phi(-c)) if two_sided else -1.0
        for w in range(nwin):
            k = 0
            for _ in range(window):
                u = _u(rng)
                if u > hi or u < lo:
                    k += 1
            out[w] = k
        return out
    for w in range(nwin):
        hist: list = []
        k = 0
        for _ in range(window):
            hist.append(INV(_u(rng)))
            if len(hist) > 10:
                hist.pop(0)
            if _alarm_rules_raw(hist, rule_count, run4):
                k += 1
                hist = []
        out[w] = k
    return out
