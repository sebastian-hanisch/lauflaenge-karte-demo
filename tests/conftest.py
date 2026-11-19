"""Gemeinsame Fixtures: Importpfad, die Mini-Karte der Handbeispiele und eine Testhilfe für zerlegte Ketten (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "tools"))

import lfk_chart as K  # noqa: E402
from helpers_lfk import MINI_PROBS, mini_chain  # noqa: E402


@pytest.fixture
def mini():
    """Die Mini-Karte aus den Handbeispielen (drei Zustände, einseitig, Regel 1 und 2) als Kette der App, mit den ausgedachten Bruch-Wahrscheinlichkeiten
    a = 1/100 (Alarmzone), b = 1/10 (Warnzone), c = 89/100 (Rest): die Zonenwahrscheinlichkeiten werden für den Test ersetzt."""
    return mini_chain()


@pytest.fixture
def mini_fractions(monkeypatch):
    """Wie `mini`, aber `lfk_chart.zone_probs` liefert die ausgedachten Wahrscheinlichkeiten (Alarm 0,01 / Warnung 0,1 / Rest 0,89) statt Normalverteilungswerten."""
    monkeypatch.setattr(K, "zone_probs", lambda delta: MINI_PROBS.copy())
    return mini_chain()
