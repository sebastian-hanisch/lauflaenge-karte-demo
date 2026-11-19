"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import lfk_chart as K
import lfk_constants as C


def _to_bool(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "ja", "yes", "on")


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None   # bei Auswahlreglern: erlaubte Werte (statt lo/hi)


SETTING_SPECS = {
    "rules_sel": SettingSpec("r", int, 1, choices=C.RULE_COUNTS),
    "side_radio": SettingSpec("s", str, "zwei", choices=C.SIDES),
    "c_input": SettingSpec("c", float, C.DEFAULT_C, C.C_MIN, C.C_MAX),
    "run4_sel": SettingSpec("l4", int, C.DEFAULT_RULE4_RUN, choices=C.RULE4_RUNS),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "n_slider": SettingSpec("n", int, C.DEFAULT_SERIES_N, C.SERIES_N_MIN, C.SERIES_N_MAX),
    "nu_slider": SettingSpec("nu", int, C.DEFAULT_NU, C.NU_MIN, C.NU_MAX),
    "delta_slider": SettingSpec("d", float, 0.0, C.DELTA_MIN, C.DELTA_MAX),
    "t0_slider": SettingSpec("t0", int, C.DEFAULT_T0, C.T0_MIN, C.T0_MAX),
    "runs_slider": SettingSpec("runs", int, C.DEFAULT_RUNS, C.RUNS_MIN, C.RUNS_MAX),
    "window_slider": SettingSpec("w", int, C.DEFAULT_WINDOW, C.WINDOW_MIN, C.WINDOW_MAX),
    "nwin_slider": SettingSpec("nw", int, C.DEFAULT_NWIN, C.NWIN_MIN, C.NWIN_MAX),
    "sim_seed_input": SettingSpec("ss", int, C.DEFAULT_SIM_SEED, 0, C.SIM_SEED_MAX),
    "simple_check": SettingSpec("eq", _to_bool, True),
}
# Preset-Schlüssel (lfk_constants.BASE) -> Reglerschlüssel
PRESET_KEYS = {
    "rules": "rules_sel", "side": "side_radio", "c": "c_input", "run4": "run4_sel", "seed": "seed_input", "n": "n_slider", "nu": "nu_slider",
    "delta": "delta_slider", "t0": "t0_slider", "runs": "runs_slider", "window": "window_slider", "nwin": "nwin_slider", "sim_seed": "sim_seed_input",
    "simple": "simple_check",
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _clean(state_key: str, value):
    """Wert auf die Grenzen bzw. erlaubten Werte des Reglers begrenzen (Permalink und Preset)."""
    spec = SETTING_SPECS[state_key]
    if spec.choices is not None:
        return value if value in spec.choices else spec.default
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def normalize_state():
    """Einseitig gibt es nur für Regel 1 allein: bei Zusatzregeln gilt zweiseitig (der Regler ist dann gesperrt)."""
    if st.session_state.get("rules_sel", 1) != 1:
        st.session_state["side_radio"] = "zwei"


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = _clean(state_key, spec.caster(qp[spec.url_param]))
            except (ValueError, TypeError):
                continue
            st.session_state[state_key] = value
    normalize_state()
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]
    normalize_state()


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def on_rules_change():
    """Wechsel des Regelsatzes: einseitig ist nur bei Regel 1 allein definiert."""
    normalize_state()


def set_c_for_target():
    """Setzt die Schwelle so, dass ARL₀ = 370 (gesetzter Zielwert) gilt: zweiseitig 2,9997, einseitig 2,7818 (die 3σ-Karte hat 370,398)."""
    two = st.session_state.get("side_radio", "zwei") == "zwei" or st.session_state.get("rules_sel", 1) != 1
    st.session_state["c_input"] = round(K.c_for_arl(C.TARGET_ARL, two), 4)


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
