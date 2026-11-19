"""Testhilfen: Mini-Karte der Handbeispiele als Kette der App, Fensterkette ohne Laufzähler (zweiter Aufbau), Referenzimplementierung der Regeln mit Schleifen."""

import numpy as np

import lfk_chain as H

# Mini-Karte (Handbeispiel 2): Zonen einer Beobachtung A (Alarmzone, Zone 7), B (Warnzone, Zone 6), C (Rest, Zonen 0 bis 5; hier alles auf Zone 5 gelegt).
MINI_PROBS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.89, 0.1, 0.01])


def mini_chain() -> H.Chain:
    """Drei Zustände: S0 (keine B in den letzten zwei), S1 (letzte war B), S2 (vorletzte B, letzte nicht). Regel 1: A löst aus; Regel 2: zwei B unter drei aufeinanderfolgenden lösen aus.
    Übergänge ohne Alarm: S0 -C-> S0, S0 -B-> S1, S1 -C-> S2, S2 -C-> S0. Gebaut von Hand, nicht mit build_chain: das prüft die Löser der App an einer Handrechnung.
    Die Zone der Einträge ist 5 für C (und für die Normalzonen-Fassung zusätzlich 0 bis 4, siehe `mini_chain_normal`) und 6 für B."""
    rows = np.array([0, 0, 1, 2], dtype=np.int64)
    cols = np.array([0, 1, 2, 0], dtype=np.int64)
    zones = np.array([5, 6, 5, 5], dtype=np.int64)
    return H.Chain(2, 8, rows, cols, zones, 3)


def mini_chain_normal() -> H.Chain:
    """Dieselbe Mini-Karte mit echten Normalzonen: C umfasst die Zonen 0 bis 5 (alle x <= 2σ), B die Zone 6 (2σ bis 3σ), A die Zone 7 (x > 3σ)."""
    rows, cols, zones = [], [], []
    for z in range(6):  # C
        for r, c in ((0, 0), (1, 2), (2, 0)):
            rows.append(r)
            cols.append(c)
            zones.append(z)
    rows.append(0)
    cols.append(1)
    zones.append(6)  # B aus S0 nach S1
    return H.Chain(2, 8, np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64), np.array(zones, dtype=np.int64), 3)


# ------------------------------------------------------------------------------------------------------------- Fensterkette (zweiter Aufbau)
def _side(z):
    return 0 if z < 4 else 1


def build_window_chain(rule_count: int, run4: int = 8):
    """Kette ohne Laufzähler: Zustand = Zonen der letzten vier Beobachtungen plus die Seiten der Beobachtungen 5 bis run4 - 1 davor (nur mit Regel 4).
    Regel 4 liest run4 Seiten direkt aus dem Fenster. Rückgabe wie lfk_chain.Chain (anderer Aufbau, derselbe Zweck)."""
    use4 = rule_count >= 4
    ns = (run4 - 5) if use4 else 0
    start = ((-1,) * 4, (-1,) * ns)
    index = {start: 0}
    queue = [start]
    rows, cols, zs = [], [], []
    qi = 0
    while qi < len(queue):
        st = queue[qi]
        qi += 1
        zv, sv = st
        for c in range(8):
            w5 = (c,) + zv
            if c in (0, 7):
                continue
            if rule_count >= 2:
                w3 = w5[:3]
                if sum(x in (0, 1) for x in w3) >= 2 or sum(x in (6, 7) for x in w3) >= 2:
                    continue
            if rule_count >= 3:
                if sum(x in (0, 1, 2) for x in w5) >= 4 or sum(x in (5, 6, 7) for x in w5) >= 4:
                    continue
            if use4:
                sides = (_side(c),) + tuple(-1 if x == -1 else _side(x) for x in zv) + sv
                if -1 not in sides and len(set(sides)) == 1:
                    continue
            nsv = (((-1 if zv[3] == -1 else _side(zv[3])),) + sv)[:ns] if ns else ()
            nxt = (w5[:4], nsv)
            if nxt not in index:
                index[nxt] = len(index)
                queue.append(nxt)
            rows.append(index[st])
            cols.append(index[nxt])
            zs.append(c)
    return H.Chain(rule_count, run4, np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64), np.array(zs, dtype=np.int64), len(index))


# ------------------------------------------------------------------------------------------------------------- Referenz der Regeln mit Schleifen
def reference_first_alarm(x, rule_count: int, c: float = 3.0, two_sided: bool = True, run4: int = 8):
    """Erster Alarm (1-basiert) und Regelnummer, mit einfachen Schleifen an den Rohwerten (anderer Code als lfk_chart.rule_flags). None, wenn keiner."""
    hist = []
    for t, v in enumerate(x, start=1):
        hist.append(float(v))
        if rule_count == 1:
            if (abs(v) > c) if two_sided else (v > c):
                return t, 1
            continue
        if abs(v) > 3.0:
            return t, 1
        w = hist[-3:]
        if sum(a > 2.0 for a in w) >= 2 or sum(a < -2.0 for a in w) >= 2:
            return t, 2
        if rule_count >= 3:
            w = hist[-5:]
            if sum(a > 1.0 for a in w) >= 4 or sum(a < -1.0 for a in w) >= 4:
                return t, 3
        if rule_count >= 4 and len(hist) >= run4:
            w = hist[-run4:]
            if all(a > 0 for a in w) or all(a < 0 for a in w):
                return t, 4
    return None
