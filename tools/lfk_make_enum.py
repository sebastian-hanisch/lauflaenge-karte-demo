"""Erzeugt lfk_enum.json: P(T = t) für t = 1 … 8 aus der Aufzählung ALLER 8^8 = 16 777 216 Zonenfolgen (Orakel zur Kette).

Die Regeln werden hier direkt an den Zonenfolgen geprüft (Spaltenarithmetik, ohne Zustände und ohne Kette): das ist der unabhängige Rechenweg, gegen den
tests/test_oracle_karte.py die Kette (lfk_chain.py) für t <= 8 vergleicht. Für t <= 6 läuft dieselbe Aufzählung im Test live (8^6 = 262 144 Folgen).

Aufruf (im Demo-Ordner, mit dem Test-venv; rechnet rund 30 bis 60 s):
    python tools/lfk_make_enum.py            schreibt lfk_enum.json
    python tools/lfk_make_enum.py --check    rechnet neu und vergleicht mit lfk_enum.json (relative Toleranz 1e-9), Rückgabewert 1 bei Abweichung

Nur Standardbibliothek und numpy; die Normalverteilungsfunktion über math.erfc (plattformfest bis auf die letzte Stelle, darum die Toleranz).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "lfk_enum.json"
EDGES = [-math.inf, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, math.inf]
SETS = {"1": 1, "1+2": 2, "1+2+3": 3, "1+2+3+4": 4}
DELTAS = (0.0, 1.0)
TLEN = 8
STAND = "2026-11-18"


def _phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def zone_probs(delta: float) -> np.ndarray:
    cdf = [_phi(e - delta) for e in EDGES]
    return np.diff(np.array(cdf))


def enumerate_first_alarm(delta: float, tlen: int = TLEN, run4: int = 8, chunk: int = 1 << 20) -> dict[int, np.ndarray]:
    """P(T = t), t = 1 … tlen, je Regelsatz (Schlüssel 1 bis 4 = Zahl der Regeln), aus allen 8^tlen Zonenfolgen; Index 0 ungenutzt (0)."""
    pc = zone_probs(delta)
    n_total = 8 ** tlen
    res = {r: np.zeros(tlen + 1) for r in SETS.values()}
    for off in range(0, n_total, chunk):
        idx = np.arange(off, min(off + chunk, n_total), dtype=np.int64)
        z = np.empty((len(idx), tlen), dtype=np.int8)
        for k in range(tlen):
            z[:, k] = (idx // (8 ** (tlen - 1 - k))) % 8
        prob = np.prod(pc[z], axis=1)
        low2, high2 = (z <= 1), (z >= 6)
        low3, high3 = (z <= 2), (z >= 5)
        below = z <= 3
        a1 = (z == 0) | (z == 7)
        a2 = np.zeros_like(a1)
        a3 = np.zeros_like(a1)
        a4 = np.zeros_like(a1)
        for t in range(tlen):
            lo = max(0, t - 2)
            a2[:, t] = (low2[:, lo : t + 1].sum(1) >= 2) | (high2[:, lo : t + 1].sum(1) >= 2)
            lo = max(0, t - 4)
            a3[:, t] = (low3[:, lo : t + 1].sum(1) >= 4) | (high3[:, lo : t + 1].sum(1) >= 4)
            if t >= run4 - 1:
                seg = below[:, t - run4 + 1 : t + 1]
                a4[:, t] = seg.all(1) | (~seg).all(1)
        for r in res:
            alarm = a1.copy()
            if r >= 2:
                alarm |= a2
            if r >= 3:
                alarm |= a3
            if r >= 4:
                alarm |= a4
            any_alarm = alarm.any(1)
            first = np.argmax(alarm, axis=1) + 1
            for t in range(1, tlen + 1):
                m = any_alarm & (first == t)
                res[r][t] += prob[m].sum()
    return res


def compute() -> dict:
    out = {"_beschreibung": "P(T = t), t = 1 … 8, aus der Aufzählung aller 8^8 Zonenfolgen; Regeln direkt an den Zonen (tools/lfk_make_enum.py); Regel 4 mit 8 in Folge", "felder": {}}
    for delta in DELTAS:
        res = enumerate_first_alarm(delta)
        for name, r in SETS.items():
            out["felder"][f"delta_{delta:g}/{name}"] = {
                "quelle": f"tools/lfk_make_enum.py, Aufzählung aller 8^8 Zonenfolgen, Mittelwert delta = {delta:g}, Regelsatz {name}",
                "stand": STAND,
                "werte": [float(v) for v in res[r][1:]],
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="neu rechnen und mit lfk_enum.json vergleichen")
    args = ap.parse_args()
    new = compute()
    if not args.check:
        OUT.write_text(json.dumps(new, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"geschrieben: {OUT} ({len(new['felder'])} Felder)")
        return 0
    old = json.loads(OUT.read_text(encoding="utf-8"))
    bad = []
    for key, field in new["felder"].items():
        old_vals = old["felder"].get(key, {}).get("werte")
        if old_vals is None or not np.allclose(old_vals, field["werte"], rtol=1e-9, atol=0.0):
            bad.append(key)
    if set(old["felder"]) != set(new["felder"]):
        bad.append("Feldmenge")
    print("lfk_enum.json stimmt mit der Neurechnung überein" if not bad else f"Abweichung in: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
