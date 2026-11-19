"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jede Zahl der Hilfetexte gegen die Rechnung."""

import pytest

import lfk_chart as K
import lfk_constants as C
import lfk_evaluation as E
import lfk_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds_or_choices():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            value = preset[key]
            if spec.choices is not None:
                assert value in spec.choices, (name, key)
            elif spec.lo is not None:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert E.Settings(**C.PRESETS[first]) == E.Settings()
    for state_key, spec in P.SETTING_SPECS.items():
        key = next(k for k, v in P.PRESET_KEYS.items() if v == state_key)
        assert C.PRESETS[first][key] == spec.default


def test_setting_specs_cover_every_preset_key_and_every_settings_field():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert set(P.PRESET_KEYS) == set(E.Settings.__dataclass_fields__)


def test_presets_use_the_right_rule_set_and_jump():
    p = C.PRESETS
    assert p["Ruhige Reihe 3σ"]["delta"] == 0.0 and p["Ruhige Reihe 3σ"]["rules"] == 1
    assert p["Sprung 1σ nach 100"]["delta"] == 1.0 and p["Sprung 1σ nach 100"]["nu"] == 100
    assert p["Sprung 0,5σ (kaum zu sehen)"]["delta"] == 0.5 and p["Sprung 3σ"]["delta"] == 3.0
    assert p["Vier Regeln, gleiche ARL₀"]["rules"] == 4 and p["Vier Regeln, gleiche ARL₀"]["side"] == "zwei"
    assert p["ARL₀ = 100 (zu enge Grenze)"]["c"] == 2.5758 and p["ARL₀ = 100 (zu enge Grenze)"]["delta"] == 0.0


def test_numbers_in_the_help_texts_match_the_calculation():
    rl = E.run_length(1, True, 3.0, 8, 0.0)
    assert "370,4" in C.PRESET_HELP["Ruhige Reihe 3σ"] and rl.mean == pytest.approx(370.4, abs=0.05)
    half = E.run_length(1, True, 3.0, 8, 0.5)
    assert "155,2" in C.PRESET_HELP["Sprung 0,5σ (kaum zu sehen)"] and half.mean == pytest.approx(155.2, abs=0.05) and half.sf(100) > 0.5
    assert "2,0 Beobachtungen" in C.PRESET_HELP["Sprung 3σ"] and E.run_length(1, True, 3.0, 8, 3.0).mean == pytest.approx(2.0, abs=0.05)
    assert "91,75" in C.PRESET_HELP["Vier Regeln, gleiche ARL₀"] and E.run_length(4, True, 3.0, 8, 0.0).mean == pytest.approx(91.75, abs=0.005)
    c = C.PRESETS["ARL₀ = 100 (zu enge Grenze)"]["c"]
    text = C.PRESET_HELP["ARL₀ = 100 (zu enge Grenze)"]
    assert "±2,5758σ" in text and "99,99" in text and "63,4 %" in text
    assert 1 / K.p_alarm(c, 0.0, True) == pytest.approx(99.99, abs=0.005)
    assert E.run_length(1, True, c, 8, 0.0).cdf(100) == pytest.approx(0.634, abs=5e-4)
