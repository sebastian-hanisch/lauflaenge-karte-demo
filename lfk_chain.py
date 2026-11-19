"""Die Karte als absorbierende Markov-Kette: Lauflängenverteilung, ARL und Verzögerungsmaße exakt.

Zustand = (Zonen der letzten Beobachtungen, Länge des Laufs auf einer Seite der Mittellinie). Die Zonengrenzen sind die Regelgrenzen
der Regeln, deshalb gibt es keinen Diskretisierungsfehler. Eine Beobachtung, bei der keine Regel anschlägt, führt in den Folgezustand;
jede andere ist ein Alarm (absorbierend). R ist die Matrix der Übergänge ohne Alarm, der Start ist der Zustand ohne Vorgeschichte (zero-state).

  P(T > t)   = Start · R^t · 1                     Überlebensfunktion der Lauflänge (Beobachtungen bis einschließlich der alarmierenden)
  ARL        = Start · (I - R)^-1 · 1              Erwartungswert
  ARL₁(s)    = ((I - R_delta)^-1 · 1)[s]           mittlere Verzögerung ab Zustand s nach einem Sprung

Maße der Verzögerung D = T - nu (nu = Beobachtungen vor der Änderung; der Alarm bei der ersten veränderten Beobachtung zählt 1):
  zero-state   Start ohne Vorgeschichte (nu = 0)
  steady-state quasistationär: Start in der Grenzverteilung der alarmfreien Kette (Grenzwert von D_nu für nu → ∞, bedingt auf „kein Alarm davor“);
               nie das Shiryaev-Maß (zyklisch), das hier nicht berechnet wird
  Pollak       Maximum über nu von E[T - nu | T > nu]
  Lorden       Maximum über alle erreichbaren Zustände der mittleren Verzögerung ab diesem Zustand
"""

from functools import lru_cache
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

import lfk_chart as K
import lfk_constants as C


class Chain(NamedTuple):
    rule_count: int
    run4: int
    rows: np.ndarray
    cols: np.ndarray
    zones: np.ndarray
    n: int          # Zahl der Zustände; Zustand 0 ist der Start ohne Vorgeschichte


def _side(z: int) -> int:
    return 0 if z < 4 else 1


@lru_cache(maxsize=64)
def build_chain(rule_count: int, run4: int = 8) -> Chain:
    """Kette der Regelsätze 1, 1+2, 1+2+3, 1+2+3+4 (zweiseitig, Zonen bei ±1, ±2, ±3). Nur die gebrauchte Vorgeschichte wird mitgeführt:
    zwei Zonen für Regel 2, vier für Regel 3 und den Laufzähler (gedeckelt bei run4 - 1) nur mit Regel 4."""
    if rule_count not in C.RULE_COUNTS:
        raise ValueError(f"Regelsatz {rule_count} gibt es nicht")
    use2, use3, use4 = rule_count >= 2, rule_count >= 3, rule_count >= 4
    hist_len = 4 if use3 else (2 if use2 else 0)
    start = ((-1,) * hist_len, 0)
    index = {start: 0}
    queue = [start]
    rows, cols, zs = [], [], []
    qi = 0
    while qi < len(queue):
        state = queue[qi]
        qi += 1
        h, r = state
        for z in range(8):
            if z in (0, 7):                                              # Regel 1 (immer): |x| > 3
                continue
            w3 = (z,) + h[:2]
            if use2 and (sum(v in (0, 1) for v in w3) >= 2 or sum(v in (6, 7) for v in w3) >= 2):
                continue
            w5 = (z,) + h
            if use3 and (sum(v in (0, 1, 2) for v in w5) >= 4 or sum(v in (5, 6, 7) for v in w5) >= 4):
                continue
            r2 = 0
            if use4:
                same = h[0] != -1 and _side(h[0]) == _side(z)
                r2 = r + 1 if same else 1
                if r2 >= run4:
                    continue
                r2 = min(r2, run4 - 1)
            nxt = (((z,) + h[: hist_len - 1]) if hist_len else (), r2)
            if nxt not in index:
                index[nxt] = len(index)
                queue.append(nxt)
            rows.append(index[state])
            cols.append(index[nxt])
            zs.append(z)
    return Chain(rule_count, run4, np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64), np.array(zs, dtype=np.int64), len(index))


def matrix(chain: Chain, delta: float) -> sp.csr_matrix:
    """Übergangsmatrix ohne Alarm für x ~ N(delta, 1)."""
    pc = K.zone_probs(delta)
    return sp.csr_matrix((pc[chain.zones], (chain.rows, chain.cols)), shape=(chain.n, chain.n))


def arl_vector(chain: Chain, delta: float) -> np.ndarray:
    """Mittlere Lauflänge je Startzustand: (I - R)^-1 · 1."""
    n = chain.n
    return spl.spsolve(sp.identity(n, format="csc") - matrix(chain, delta).tocsc(), np.ones(n))


def survival(chain: Chain, delta: float, tmax: int, start: np.ndarray | None = None) -> np.ndarray:
    """S[t] = P(T > t), t = 0..tmax; Start im Zustand 0 oder in der Verteilung `start` (Summe 1)."""
    rt = matrix(chain, delta).T.tocsr()
    v = np.zeros(chain.n)
    if start is None:
        v[0] = 1.0
    else:
        v[:] = start
    s = np.empty(tmax + 1)
    s[0] = v.sum()
    for t in range(1, tmax + 1):
        v = rt @ v
        s[t] = v.sum()
    return s


def state_distribution(chain: Chain, nu: int) -> tuple[np.ndarray, float]:
    """Verteilung der Zustände nach nu Beobachtungen ohne Alarm (bedingt auf „kein Alarm davor“) und P(kein Alarm in den ersten nu)."""
    rt = matrix(chain, 0.0).T.tocsr()
    v = np.zeros(chain.n)
    v[0] = 1.0
    for _ in range(nu):
        v = rt @ v
    mass = float(v.sum())
    return (v / mass if mass > 0 else v), mass


def tail_length(mean: float, factor: float = 45.0, cap: int = C.TMAX_CAP) -> int:
    """Länge, bis zu der die Überlebensfunktion ausgerechnet wird: etwa das 45fache des Mittels (der Rest ist dann kleiner als e^-45)."""
    return int(min(cap, max(200, np.ceil(factor * mean))))


def quasi_stationary(chain: Chain, iters: int = 4000, tol: float = 1e-13) -> tuple[np.ndarray, float, float]:
    """Grenzverteilung der alarmfreien Kette bei delta = 0 (Potenzverfahren): (Verteilung, dominanter Eigenwert, Rest der Eigenvektorgleichung)."""
    rt = matrix(chain, 0.0).T.tocsr()
    v = np.zeros(chain.n)
    v[0] = 1.0
    for _ in range(iters):
        w = rt @ v
        w /= w.sum()
        if np.max(np.abs(w - v)) < tol:
            v = w
            break
        v = w
    w = rt @ v
    lam = float(w.sum())
    return v, lam, float(np.max(np.abs(w / lam - v)))


class DelayMeasures(NamedTuple):
    zero: float
    steady: float
    pollak: float
    pollak_nu: int
    lorden: float
    lorden_state: int


def delay_measures(chain: Chain, delta: float, nu_max: int = C.POLLAK_NU_MAX) -> DelayMeasures:
    """Mittlere Verzögerung nach einem Sprung der Größe delta in den vier Maßen (Beobachtungen)."""
    l1 = arl_vector(chain, delta)
    q, _, _ = quasi_stationary(chain)
    rt = matrix(chain, 0.0).T.tocsr()
    v = np.zeros(chain.n)
    v[0] = 1.0
    best, best_nu = -1.0, 0
    for nu in range(nu_max + 1):
        val = float(v @ l1) / float(v.sum())
        if val > best + 1e-12:
            best, best_nu = val, nu
        v = rt @ v
    return DelayMeasures(float(l1[0]), float(q @ l1), best, best_nu, float(l1.max()), int(np.argmax(l1)))


def conditional_delay_survival(chain: Chain, delta: float, nu: int, tmax: int) -> np.ndarray:
    """P(D > t | kein Alarm vor nu), t = 0..tmax, wenn der Mittelwert nach nu Beobachtungen auf delta springt (exakt)."""
    start, _ = state_distribution(chain, nu)
    return survival(chain, delta, tmax, start=start)


def quantile(s: np.ndarray, q: float) -> int:
    """Kleinstes t mit P(T <= t) >= q aus der Überlebensfunktion s (s[t] = P(T > t))."""
    f = 1.0 - s
    hit = np.flatnonzero(f >= q)
    return int(hit[0]) if len(hit) else len(s)


def max_distance_to_geometric(s: np.ndarray, mean: float) -> float:
    """Größter Betrag der Differenz der Überlebensfunktion zur geometrischen mit gleichem Mittel."""
    p = 1.0 / mean
    t = np.arange(len(s))
    return float(np.max(np.abs(s - (1.0 - p) ** t)))
