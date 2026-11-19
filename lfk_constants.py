"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Modell (Kurzfassung, Einzelheiten im 📐-Expander der App): unabhängige Beobachtungen x_t ~ N(delta, 1), Ziel 0 und Streuung 1 bekannt,
Alarmregel der Kontrollkarte, Zähleinheit Beobachtungen (Lauflänge bis zum Alarm), Reihen (Anteile) und Fenster (Alarmzahl je Fenster).
"""

# --- Kette und Regeln ---------------------------------------------------------------------------------------------------------------
RULE_COUNTS = (1, 2, 3, 4)                 # Regelsatz = Zahl der Regeln: 1 = nur Regel 1, 2 = Regeln 1+2, 3 = 1+2+3, 4 = 1+2+3+4 (geschachtelt)
RULE_NAMES = {1: "nur Regel 1 (einfache Karte)", 2: "Regeln 1+2", 3: "Regeln 1+2+3", 4: "Regeln 1+2+3+4"}
RULE_SHORT = {1: "1", 2: "1+2", 3: "1+2+3", 4: "1+2+3+4"}
RULE4_RUNS = (8, 9, 10)                    # Fassungen von Regel 4: so viele Beobachtungen in Folge auf derselben Seite der Mittellinie
DEFAULT_RULE4_RUN = 8
SIDES = ("zwei", "ein")
SIDE_NAMES = {"zwei": "zweiseitig (|x| > c)", "ein": "einseitig (x > c)"}
TARGET_ARL = 370                           # „ARL₀ = 370“: gesetzter Zielwert; die 3σ-Karte hat 370,398 (zweiseitig), das sind zwei verschiedene Ziele

# --- Regler ---------------------------------------------------------------------------------------------------------------------------
C_MIN, C_MAX, DEFAULT_C = 2.0, 4.0, 3.0    # Schwelle der einfachen Karte in σ-Einheiten; bei Zusatzregeln fest 3σ
SERIES_N_MIN, SERIES_N_MAX, DEFAULT_SERIES_N = 50, 1000, 400   # Länge der Einzelreihe (normal() ist langsam: höchstens 1000)
NU_MIN, NU_MAX, DEFAULT_NU = 0, 1000, 100  # Beobachtungen vor der Änderung (nu = 0: die Änderung liegt vor der ersten Beobachtung)
DELTA_MIN, DELTA_MAX, DEFAULT_DELTA, DELTA_STEP = 0.0, 4.0, 1.0, 0.25   # Sprung des Mittelwerts in σ (0 = keiner)
T0_MIN, T0_MAX, DEFAULT_T0 = 10, 1000, 100  # Fensterlänge für „Alarm innerhalb der ersten t0 Beobachtungen“
RUNS_MIN, RUNS_MAX, DEFAULT_RUNS = 200, 5000, 2000   # simulierte Läufe
WINDOW_MIN, WINDOW_MAX, DEFAULT_WINDOW = 100, 5000, 1000   # Länge eines Fensters (Beobachtungen)
NWIN_MIN, NWIN_MAX, DEFAULT_NWIN = 100, 3000, 1000  # Zahl simulierter Fenster
SEED_MAX, DEFAULT_SEED = 999999, 5
SIM_SEED_MAX, DEFAULT_SIM_SEED = 99_999_999, 20261118

# Rechenbudget der Simulation (Beobachtungen je Klick); Laufzeit auf der Entwicklungsmaschine: siehe README „Rechenzeit“
SIM_BUDGET_SIMPLE = 2_000_000              # einfache Karte (Vergleich der Zufallszahl mit zwei Schwellen)
SIM_BUDGET_RULES = 600_000                 # Zusatzregeln (Regeln an den Rohwerten, Python-Schleife)
SIM_MIN_RUNS = 100                         # darunter wird nicht simuliert (Meldung statt eines Bands, das nichts aussagt)

# --- Kette ----------------------------------------------------------------------------------------------------------------------------
TMAX_CAP = 60_000                          # Obergrenze der berechneten Verteilung der Lauflänge (Beobachtungen)
POLLAK_NU_MAX = 600                        # Pollak-Maß: Maximum über nu = 0 … POLLAK_NU_MAX
DELTA_GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0)   # ARL₁-Kurve
CROSS_BRACKET = (0.5, 4.0)                 # Suchbereich der Kreuzung (ARL₁ gleich)

# --- Farben ---------------------------------------------------------------------------------------------------------------------------
COLOR_EXACT = "#2F6B65"
COLOR_SIM = "#d9822b"
COLOR_RULES = "#7b4fa3"
COLOR_SIMPLE = "#4c78a8"
COLOR_JUMP = "rgba(190,60,60,0.9)"
COLOR_GRID = "rgba(150,150,150,0.55)"
RULE_COLORS = {1: "#c0392b", 2: "#d9822b", 3: "#b8a000", 4: "#7b4fa3"}

# --- Presets (Schlüssel = Reglernamen in lfk_presets.PRESET_KEYS) ----------------------------------------------------------------------
BASE = {
    "rules": 1, "side": "zwei", "c": DEFAULT_C, "run4": DEFAULT_RULE4_RUN,
    "seed": DEFAULT_SEED, "n": DEFAULT_SERIES_N, "nu": DEFAULT_NU, "delta": 0.0,
    "t0": DEFAULT_T0, "runs": DEFAULT_RUNS, "window": DEFAULT_WINDOW, "nwin": DEFAULT_NWIN, "sim_seed": DEFAULT_SIM_SEED, "simple": True,
}
PRESETS = {
    "Ruhige Reihe 3σ": dict(BASE),
    "Sprung 1σ nach 100": {**BASE, "delta": 1.0, "seed": 43},
    "Sprung 0,5σ (kaum zu sehen)": {**BASE, "delta": 0.5, "n": 600, "seed": 3},
    "Sprung 3σ": {**BASE, "delta": 3.0},
    "Vier Regeln, gleiche ARL₀": {**BASE, "rules": 4, "delta": 1.0, "seed": 23},
    "ARL₀ = 100 (zu enge Grenze)": {**BASE, "c": 2.5758, "seed": 59},
}
PRESET_HELP = {
    "Ruhige Reihe 3σ": "Keine Änderung (δ = 0), zweiseitige Karte mit Grenzen ±3σ: Alarme sind hier Fehlalarme. Die Reihe zeigt eine Ziehung, ARL₀ = 370,4 den Mittelwert.",
    "Sprung 1σ nach 100": "Nach 100 Beobachtungen springt der Mittelwert um 1σ. Wie lange dauert es bis zum Alarm, und wie streut das?",
    "Sprung 0,5σ (kaum zu sehen)": "Ein halbes σ Sprung nach 100 Beobachtungen: die einfache Karte braucht im Mittel 155,2 Beobachtungen, über die Hälfte der Reihen länger als 100.",
    "Sprung 3σ": "Ein Sprung von 3σ nach 100 Beobachtungen: die einfache Karte alarmiert im Mittel nach 2,0 Beobachtungen.",
    "Vier Regeln, gleiche ARL₀": "Alle vier Regeln mit Sprung 1σ: ARL₀ sinkt auf 91,75; verglichen wird mit der einfachen Karte gleicher ARL₀ (Experiment 5).",
    "ARL₀ = 100 (zu enge Grenze)": "Grenzen ±2,5758σ geben ARL₀ = 99,99 (rund 100): 63,4 % der Reihen haben schon in den ersten 100 Beobachtungen einen Alarm, ohne dass sich etwas geändert hat.",
}
