"""AppTest-Tests dieser Demo über den Smoke-Test der Vorlage hinaus (der steht in test_app_smoke.py: jeder Knopf, jeder Slider an Minimum und Maximum): Zahlenfelder an den Grenzen,
alle Regelsätze und Fassungen von Regel 4, Eckfall-Matrix über die Regler, je Schnellstart-Szenario die erwartete Verdikt-Zeile (ein Szenario ohne Effekt darf keinen Effekt
melden), die Simulationsknöpfe (die der Vorlagen-Test als „▶“ auslässt), einseitig nur bei Regel 1, der Knopf für ARL₀ = 370."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import lfk_constants as C
import lfk_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
WIDGETS = ("slider", "number_input", "selectbox", "radio", "checkbox")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _widget(at, key):
    for art in WIDGETS:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


def _texts(at) -> str:
    parts = []
    for kind in ("success", "info", "warning", "error"):
        parts += [el.value for el in getattr(at, kind)]
    return "\n".join(parts)


def test_app_loads_without_exception():
    at = _fresh_app()
    assert at.title[0].value


@pytest.mark.parametrize("key", ["c_input", "seed_input", "sim_seed_input"])
@pytest.mark.parametrize("edge", ["lo", "hi"])
def test_number_inputs_at_their_limits_do_not_raise(key, edge):
    at = _fresh_app()
    spec = P.SETTING_SPECS[key]
    _widget(at, key).set_value(spec.lo if edge == "lo" else spec.hi).run(timeout=TIMEOUT)
    assert not at.exception, (key, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("rules", list(C.RULE_COUNTS))
@pytest.mark.parametrize("run4", list(C.RULE4_RUNS))
def test_every_rule_set_and_rule_4_version_runs(rules, run4):
    at = _fresh_app()
    _widget(at, "rules_sel").set_value(rules).run(timeout=TIMEOUT)
    _widget(at, "run4_sel").set_value(run4).run(timeout=TIMEOUT)
    _widget(at, "delta_slider").set_value(1.5).run(timeout=TIMEOUT)
    assert not at.exception, (rules, run4, [str(e) for e in at.exception])


@pytest.mark.parametrize("rules", [1, 4])
def test_all_minima_and_all_maxima_together(rules):
    """Eckfall-Matrix: alle Regler zugleich am Minimum bzw. am Maximum (Schwelle nur bei Regel 1 wählbar)."""
    for edge in ("lo", "hi"):
        at = _fresh_app()
        _widget(at, "rules_sel").set_value(rules).run(timeout=TIMEOUT)
        for key, spec in P.SETTING_SPECS.items():
            if spec.lo is None or key in ("rules_sel",):
                continue
            if key == "c_input" and rules != 1:
                continue
            _widget(at, key).set_value(spec.lo if edge == "lo" else spec.hi)
        at.run(timeout=TIMEOUT)
        assert not at.exception, (rules, edge, [str(e) for e in at.exception])


def test_one_sided_is_only_offered_for_rule_1_and_resets_with_additional_rules():
    at = _fresh_app()
    _widget(at, "side_radio").set_value("ein").run(timeout=TIMEOUT)
    assert not at.exception and at.sidebar.radio(key="side_radio").value == "ein" and not at.sidebar.radio(key="side_radio").proto.disabled
    assert "einseitige karte mit grenze 3σ" in _texts(at).lower()
    _widget(at, "rules_sel").set_value(2).run(timeout=TIMEOUT)
    assert at.sidebar.radio(key="side_radio").value == "zwei" and at.sidebar.radio(key="side_radio").proto.disabled
    assert at.sidebar.number_input(key="c_input").proto.disabled
    assert "zweiseitige regeln 1+2" in _texts(at).lower()


def test_the_c_button_sets_the_threshold_for_arl_370():
    at = _fresh_app()
    next(b for b in at.sidebar.button if b.label.startswith("c für ARL₀ = 370")).click().run(timeout=TIMEOUT)
    assert at.sidebar.number_input(key="c_input").value == pytest.approx(2.9997, abs=1e-9)
    assert "ARL₀ = 370,0 Beobachtungen" in _texts(at)
    _widget(at, "side_radio").set_value("ein").run(timeout=TIMEOUT)
    next(b for b in at.sidebar.button if b.label.startswith("c für ARL₀ = 370")).click().run(timeout=TIMEOUT)
    assert at.sidebar.number_input(key="c_input").value == pytest.approx(2.7818, abs=1e-9)


EXPECTED = {   # je Preset: Teile der Verdikt-Zeilen (Zahlen aus der Rechnung, DEMO-PLAYBOOK §11a Punkt 7)
    "Ruhige Reihe 3σ": ["Keine Änderung in dieser Reihe", "ARL₀ = 370,4"],
    "Sprung 1σ nach 100": ["nach dem Sprung bei 100", "im Mittel 43,9 Beobachtungen", "Median 31", "95 % innerhalb von 130"],
    "Sprung 0,5σ (kaum zu sehen)": ["Sprung um 0,5σ", "ARL₁ = 155,2", "52,40 %"],
    "Sprung 3σ": ["Sprung um 3σ", "ARL₁ = 2,00"],
    "Vier Regeln, gleiche ARL₀": ["ARL₀ = 91,75", "ARL₁ 9,22 gegen 16,32", "δ* = 2,26σ"],
    "ARL₀ = 100 (zu enge Grenze)": ["±2,5758σ", "ARL₀ = 99,99", "63,40 %"],
}


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_each_preset_shows_its_expected_verdict(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    text = _texts(at)
    for part in EXPECTED[name]:
        assert part in text, (name, part)


def test_presets_without_a_jump_never_report_a_delay():
    for name in ("Ruhige Reihe 3σ", "ARL₀ = 100 (zu enge Grenze)"):
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert "Verzögerung D =" not in _texts(at) and "erwartet" not in " ".join(m.delta or "" for m in at.metric)
        assert "ohne Sprung (δ = 0) gibt es keine Verzögerung" in _texts(at)


def test_the_simulation_buttons_run_and_show_a_band_only_for_the_clicked_settings():
    at = _fresh_app()
    runs = next(b for b in at.button if b.label.startswith("▶") and "Läufe" in b.label)
    runs.click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "Simulation mit 2000 Läufen" in _texts(at)
    assert any("Rechenzeit dieser Simulation beim Klick" in c.value for c in at.caption)
    _widget(at, "t0_slider").set_value(50).run(timeout=TIMEOUT)             # t0 ändert nur die Anzeige: die Simulation bleibt gültig
    assert "Simulation mit 2000 Läufen" in _texts(at) and "innerhalb von 50" in _texts(at)
    _widget(at, "sim_seed_input").set_value(777).run(timeout=TIMEOUT)       # andere Einstellungen: das Ergebnis gilt nicht mehr
    assert "Simulation mit" not in _texts(at)
    assert any("noch einmal starten" in c.value for c in at.caption)
    win = next(b for b in at.button if b.label.startswith("▶") and "Fenster" in b.label)
    win.click().run(timeout=TIMEOUT)
    assert not at.exception and "Simulation mit 1000 Fenstern" in _texts(at)


def test_simulation_is_refused_when_the_budget_is_too_small():
    at = _fresh_app()
    _widget(at, "c_input").set_value(4.0)
    _widget(at, "side_radio").set_value("ein").run(timeout=TIMEOUT)         # einseitig, c = 4: ARL₀ = 31574
    assert not at.exception
    assert not [b for b in at.button if b.label.startswith("▶") and "Läufe" in b.label]
    assert any("Simulation hier nicht angeboten" in c.value for c in at.caption)
