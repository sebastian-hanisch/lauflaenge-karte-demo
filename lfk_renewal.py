"""Alarme je Fenster: Verteilung der Alarmzahl in T Beobachtungen, wenn nach jedem Alarm ohne Vorgeschichte neu gestartet wird.

Erneuerungsrechnung aus der Wahrscheinlichkeitsfunktion f der Lauflänge: P(N_T >= k) = P(S_k <= T), S_k = Summe von k unabhängigen Lauflängen.
Das ist die Festlegung dieser Seite (Neustart nach Alarm): „ARL ist nur sinnvoll, wenn bei Alarm angehalten wird“ (Zwetsloot u. a. 2024).
Für die einfache Karte (gedächtnislos) ist die Alarmzahl Binomial(T, p); das ist der zweite Rechenweg.
"""

import numpy as np
from scipy import stats
from scipy.signal import fftconvolve

FFT_FROM = 1500   # ab dieser Fensterlänge faltet die Rechnung mit FFT (Betrag der Rundung höchstens 1e-12), darunter direkt


def renewal_counts(f, T: int, kmax: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """(P(N_T = k), P(N_T >= k)) für k = 0, 1, 2, …; f[t] = P(T = t) für t = 0..T (f[0] = 0)."""
    f = np.asarray(f[: T + 1], dtype=float)
    cur = f.copy()
    tail = [1.0]
    for _ in range(kmax):
        total = float(cur.sum())
        tail.append(total)
        if total < 1e-17:
            break
        cur = (fftconvolve(cur, f)[: T + 1] if T >= FFT_FROM else np.convolve(cur, f)[: T + 1])
        cur = np.clip(cur, 0.0, None)
    tail_a = np.array(tail)
    pmf = tail_a - np.append(tail_a[1:], 0.0)
    return pmf, tail_a


def pmf_from_survival(s: np.ndarray, T: int) -> np.ndarray:
    """P(T = t) für t = 0..T aus der Überlebensfunktion s (s[t] = P(T > t))."""
    f = np.zeros(T + 1)
    f[1:] = s[:T] - s[1 : T + 1]
    return f


def binomial_counts(T: int, p: float) -> np.ndarray:
    """P(N = k), k = 0..T, einfache Karte: Alarmzahl im Fenster ist Binomial(T, p) (zweiter Rechenweg zur Erneuerungsrechnung)."""
    return stats.binom.pmf(np.arange(T + 1), T, p)


def count_summary(pmf: np.ndarray) -> dict:
    """Mittel, P(0), P(>= 5), Quantile 5 % / 50 % / 95 % der Alarmzahl."""
    k = np.arange(len(pmf))
    cum = np.cumsum(pmf)
    return {
        "mean": float((k * pmf).sum()),
        "p0": float(pmf[0]),
        "p_ge5": float(pmf[5:].sum()),
        "p_ge8": float(pmf[8:].sum()),
        "q05": int(np.argmax(cum >= 0.05)),
        "q50": int(np.argmax(cum >= 0.5)),
        "q95": int(np.argmax(cum >= 0.95)),
    }
