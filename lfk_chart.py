"""Die Kontrollkarte: Zonen, Alarmwahrscheinlichkeit der einfachen Karte, Alarmregeln an einer Reihe von Rohwerten.

Beobachtungen x_t ~ N(delta, 1), unabhängig, Ziel 0 und Streuung 1 bekannt (alles in σ-Einheiten).

Regeln (Fassung dieser Demo „nach Art der Western-Electric-Regeln“, keine Wiedergabe einer gelesenen Quelle):
  Regel 1: |x| > 3 (einfache Karte; mit Schwelle c statt 3 und wahlweise einseitig x > c)
  Regel 2: 2 von 3 aufeinanderfolgenden Beobachtungen jenseits 2σ auf derselben Seite
  Regel 3: 4 von 5 aufeinanderfolgenden Beobachtungen jenseits 1σ auf derselben Seite
  Regel 4: L (8, 9 oder 10) Beobachtungen in Folge auf derselben Seite der Mittellinie
Zonen (Grundlage der Ketten, Grenzen = Regelgrenzen, deshalb kein Diskretisierungsfehler):
  0: x < -3, 1: -3…-2, 2: -2…-1, 3: -1…0, 4: 0…1, 5: 1…2, 6: 2…3, 7: x > 3.
"""

import math

import numpy as np
from scipy.special import ndtr, ndtri


EDGES = np.array([-np.inf, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, np.inf])


def phi(x):
    """Verteilungsfunktion der Standardnormalverteilung."""
    return ndtr(x)


def zone_probs(delta: float) -> np.ndarray:
    """Wahrscheinlichkeiten der acht Zonen für x ~ N(delta, 1)."""
    return np.diff(phi(EDGES - delta))


def p_alarm(c: float, delta: float, two_sided: bool = True) -> float:
    """Alarmwahrscheinlichkeit je Beobachtung der einfachen Karte: P(x > c), bei zweiseitiger Karte plus P(x < -c)."""
    p = float(phi(delta - c))
    if two_sided:
        p += float(phi(-c - delta))
    return p


def c_for_arl(arl0: float, two_sided: bool = True) -> float:
    """Schwelle c der einfachen Karte mit ARL₀ = arl0 (zero-state: ARL₀ = 1/p)."""
    p = 1.0 / arl0
    return float(ndtri(1.0 - (p / 2.0 if two_sided else p)))


def zone_of(x: float) -> int:
    """Zone 0 bis 7 eines Rohwerts (x genau auf einer Grenze gehört zur oberen Zone, ein Ereignis der Wahrscheinlichkeit 0)."""
    return int(np.searchsorted(EDGES[1:-1], x, side="right"))


# ------------------------------------------------------------------------------------------------------------ geometrische Lauflänge
def geometric_sf(p: float, t) -> np.ndarray:
    """P(T > t) = (1 - p)^t für die einfache Karte (exakt, gedächtnislos)."""
    return np.exp(np.asarray(t, dtype=float) * math.log1p(-p)) if p < 1.0 else np.where(np.asarray(t) <= 0, 1.0, 0.0)


def geometric_quantile(p: float, q: float) -> int:
    """Kleinstes t mit P(T <= t) >= q für die geometrische Verteilung (Rechnung in Gleitkomma, an der Grenze nachgeprüft)."""
    if p >= 1.0:
        return 1
    lp = math.log1p(-p)
    t = max(1, int(math.ceil(math.log1p(-q) / lp)))
    for _ in range(4):                                    # höchstens wenige Schritte: der Anfangswert liegt höchstens 1 daneben
        if t > 1 and -math.expm1((t - 1) * lp) >= q:
            t -= 1
        else:
            break
    for _ in range(4):
        if -math.expm1(t * lp) < q:
            t += 1
        else:
            break
    return t


# ------------------------------------------------------------------------------------------------------------ Regeln an Rohwerten
def _window_count(flags: np.ndarray, width: int) -> np.ndarray:
    """Zahl der True-Werte in den letzten `width` Beobachtungen (am Anfang weniger Beobachtungen)."""
    cs = np.concatenate(([0], np.cumsum(flags.astype(int))))
    idx = np.arange(1, len(flags) + 1)
    return cs[idx] - cs[np.maximum(idx - width, 0)]


def rule_flags(x: np.ndarray, rule_count: int, c: float = 3.0, two_sided: bool = True, run4: int = 8) -> dict[int, np.ndarray]:
    """Je Regel (1 bis rule_count) ein Boolesches Feld: schlägt die Regel bei der Beobachtung t an? (Regeln direkt an den Rohwerten.)

    Bei Zusatzregeln gilt immer die zweiseitige Karte mit c = 3; nur Regel 1 allein kennt Schwelle c und einseitig."""
    x = np.asarray(x, dtype=float)
    out: dict[int, np.ndarray] = {}
    if rule_count == 1:
        out[1] = (np.abs(x) > c) if two_sided else (x > c)
        return out
    out[1] = np.abs(x) > 3.0
    out[2] = (_window_count(x > 2.0, 3) >= 2) | (_window_count(x < -2.0, 3) >= 2)
    if rule_count >= 3:
        out[3] = (_window_count(x > 1.0, 5) >= 4) | (_window_count(x < -1.0, 5) >= 4)
    if rule_count >= 4:
        n_pos = _window_count(x > 0.0, run4)
        n_neg = _window_count(x < 0.0, run4)
        long_enough = np.arange(1, len(x) + 1) >= run4
        out[4] = long_enough & ((n_pos == run4) | (n_neg == run4))
    return {r: out[r] for r in range(1, rule_count + 1)}


def trigger_rule(flags: dict[int, np.ndarray]) -> np.ndarray:
    """Pro Beobachtung die kleinste Nummer einer anschlagenden Regel, 0 = kein Alarm."""
    n = len(next(iter(flags.values())))
    code = np.zeros(n, dtype=int)
    for r in sorted(flags, reverse=True):
        code = np.where(flags[r], r, code)
    return code


def first_alarm(flags: dict[int, np.ndarray]) -> int | None:
    """Erster Alarm als Beobachtungsnummer (1-basiert) oder None, wenn die Reihe ruhig bleibt."""
    code = trigger_rule(flags)
    hit = np.flatnonzero(code)
    return int(hit[0]) + 1 if len(hit) else None
