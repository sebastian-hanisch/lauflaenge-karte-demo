"""Karte und Lauflänge: was „ARL = 370“ verschweigt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Erste Seite der Reihe „Änderungserkennung“. Eine Kontrollkarte (Shewhart, mit und ohne Zusatzregeln) auf unabhängigen Normalwerten mit bekanntem Ziel:
Verteilung der Lauflänge bis zum Alarm, Alarme je Fenster, Verzögerung nach einem Sprung. Alles exakt (geschlossene Form bzw. Markov-Kette), Simulation als Band.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import streamlit as st

import lfk_chain as H
import lfk_chart as K
import lfk_constants as C
import lfk_evaluation as E
import lfk_scenario as S
import lfk_texts as T
from lfk_layout import dunkles_design, kopf_css
from lfk_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    on_rules_change,
    randomize_seed,
    set_c_for_target,
    sidebar_presets,
    sync_query_params,
)
from lfk_visualization import build_arl1_curve, build_cdf, build_counts, build_series

st.set_page_config(page_title="Karte und Lauflänge – Sebastian Hanisch", layout="wide")


# ------------------------------------------------------------------------------------------------------------------ gepufferte Rechnungen
@st.cache_data(show_spinner=False)
def _series(seed, n, nu, delta, rules, two, c, run4):
    x = S.make_series(seed, n, nu, delta)
    flags = K.rule_flags(x, rules, c, two, run4)
    return x, K.trigger_rule(flags), K.first_alarm(flags)


@st.cache_data(show_spinner=False)
def _rl(rules, two, c, run4, delta):
    return E.run_length(rules, two, c, run4, delta)


@st.cache_data(show_spinner=False)
def _cond_delay(rules, two, c, run4, delta, nu):
    return E.conditional_delay(rules, two, c, run4, delta, nu)


@st.cache_data(show_spinner=False)
def _measures(rules, two, c, run4, delta):
    return E.delay_measures(rules, two, c, run4, delta)


@st.cache_data(show_spinner=False)
def _counts(rules, two, c, run4, window):
    return E.count_distribution(rules, two, c, run4, window)


@st.cache_data(show_spinner=False)
def _arl1(rules, two, c, run4, deltas):
    return E.arl1_curve(rules, two, c, run4, deltas)


@st.cache_data(show_spinner=False)
def _compare(rule_count, run4):
    return E.equal_arl_comparison(rule_count, run4)


@st.cache_data(show_spinner=False)
def _versions(delta):
    return E.rule4_versions(delta)


@st.cache_data(show_spinner=False)
def _sim_runs(rules, delta, runs, seed, c, two, run4):
    return S.simulate_run_lengths(rules, delta, runs, seed, c, two, run4)


@st.cache_data(show_spinner=False)
def _sim_windows(rules, window, nwin, seed, c, two, run4):
    return S.simulate_window_counts(rules, window, nwin, seed, c, two, run4)


def _verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


def _simulate(name: str, params: tuple, label: str, spinner_text: str, compute):
    """Simulation erst auf Klick; das Ergebnis gilt nur für die Einstellungen, mit denen geklickt wurde (sonst Hinweis). `compute` ist gepuffert: nach dem Klick
    kostet jeder weitere Lauf nichts. Die Rechenzeit des Klicks bleibt als Zahl im Zustand (kein großes Array)."""
    clicked = st.button(label, key=f"{name}_btn")
    if clicked:
        st.session_state[f"{name}_key"] = params
        st.session_state.pop(f"{name}_sec", None)
    if st.session_state.get(f"{name}_key") != params:
        if f"{name}_key" in st.session_state:
            st.caption("Die Einstellungen haben sich seit der letzten Simulation geändert: noch einmal starten.")
        return None
    t_start = time.perf_counter()
    with st.spinner(spinner_text):
        result = compute()
    if clicked:
        st.session_state[f"{name}_sec"] = time.perf_counter() - t_start
    sec = st.session_state.get(f"{name}_sec")
    if sec is not None:
        st.caption(f"Rechenzeit dieser Simulation beim Klick: {E.de(sec, 1)} s (danach aus dem Puffer).")
    return result


# ------------------------------------------------------------------------------------------------------------------ Kopf
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📉 Karte und Lauflänge: was „ARL = 370“ verschweigt")
st.markdown(T.INTRO)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/lauflaenge-karte-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Eine Kontrollkarte prüft jede neue Messung und schlägt Alarm, wenn sie zu weit vom Sollwert abweicht. Nennt man für sie nur die mittlere Zahl der Messungen bis zum Fehlalarm, klingt das nach einem festen Takt. Die Demo zeigt, dass es keiner ist: Die Wartezeit bis zum Fehlalarm streut weit, der Alarm kommt oft viel früher und manchmal erst viel später als der Mittelwert. Die App rechnet diese Verteilung exakt aus und prüft sie gegen Zufallsreihen. Sie zeigt auch, wie lange die Karte nach einem echten Sprung braucht, wie viele Fehlalarme in einem Zeitfenster zu erwarten sind und was Zusatzregeln kosten. Alles gilt für ausgedachte, voneinander unabhängige Messwerte mit bekanntem Sollwert und bekannter Schwankung.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/lauflaenge-karte-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert die Kontrollkarte", expanded=True):
    st.markdown(T.HOW_IT_WORKS)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

# ------------------------------------------------------------------------------------------------------------------ Seitenleiste
with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Karte")
    rules = st.selectbox("Regelsatz", C.RULE_COUNTS, key="rules_sel", format_func=lambda r: C.RULE_NAMES[r], on_change=on_rules_change,
                         help="Regel 1: |x| > Schwelle. Regel 2: 2 von 3 jenseits 2σ. Regel 3: 4 von 5 jenseits 1σ. Regel 4: 8 (9, 10) in Folge auf einer Seite. Die Sätze sind geschachtelt.")
    side = st.radio("Seite der Karte", C.SIDES, key="side_radio", format_func=lambda s: C.SIDE_NAMES[s], disabled=rules != 1,
                    help="Einseitig ist nur für Regel 1 allein definiert und gemessen; die Zusatzregeln gibt es nur zweiseitig.")
    if rules != 1:
        st.caption("Zusatzregeln: zweiseitig, Zonen fest bei 1σ, 2σ, 3σ; die Schwelle c gilt nur für Regel 1 allein.")
    c = st.number_input("Schwelle c (σ)", *bounds("c_input"), key="c_input", step=0.01, format="%.4f", disabled=rules != 1,
                        help="Grenzen ±c in σ-Einheiten (einfache Karte). 3,0 gibt ARL₀ = 370,398; 2,9997 gibt 370.")
    st.button("c für ARL₀ = 370 setzen", width="stretch", on_click=set_c_for_target, disabled=rules != 1,
              help="Zweiseitig c = 2,9997, einseitig c = 2,7818: ARL₀ = 370 gesetzt. Die ±3σ-Karte hat 370,398 (ein anderes Ziel).")
    run4 = st.selectbox("Fassung von Regel 4", C.RULE4_RUNS, key="run4_sel", format_func=lambda n_: f"{n_} in Folge",
                        help="Wie viele Beobachtungen in Folge auf derselben Seite der Mittellinie Regel 4 auslösen. Wirkt auf Regelsatz 1+2+3+4 und auf Experiment 5.")

    st.header("📈 Reihe und Sprung")
    seed = st.number_input("Zufalls-Seed der Reihe", *bounds("seed_input"), key="seed_input", step=1, help="Experiment 1: dieselbe Zahl, dieselbe Reihe.")
    st.button("🎲 Neue Reihe ziehen", width="stretch", on_click=randomize_seed)
    n = st.slider("Länge der Reihe (Beobachtungen)", *bounds("n_slider"), key="n_slider", step=1)
    nu = st.slider("Sprung nach ν Beobachtungen", *bounds("nu_slider"), key="nu_slider", step=1,
                   help="Zahl der Beobachtungen vor der Änderung. ν = 0: die Änderung liegt vor der ersten Beobachtung; ν größer als die Länge: der Sprung liegt hinter der Reihe.")
    delta = st.slider("Sprung δ (σ-Einheiten)", *bounds("delta_slider"), key="delta_slider", step=C.DELTA_STEP,
                      help="Größe des Sprungs des Mittelwerts; 0 = keine Änderung. Gilt für die Experimente 1, 4 und 5.")

    st.header("🎲 Lauflängen und Fenster")
    t0 = st.slider("Fenster t₀ für „Alarm innerhalb der ersten t₀“", *bounds("t0_slider"), key="t0_slider", step=1, help="Experiment 2.")
    runs = st.slider("Simulierte Läufe R", *bounds("runs_slider"), key="runs_slider", step=1,
                     help="Experiment 2. Die Zahl der Läufe wird auf das Rechenbudget begrenzt (R × ARL₀ höchstens 2 Mio. Beobachtungen bei Regel 1 allein, 600 000 mit Zusatzregeln).")
    window = st.slider("Fensterlänge W (Beobachtungen)", *bounds("window_slider"), key="window_slider", step=1, help="Experiment 3: Alarme je Fenster mit Neustart nach Alarm.")
    nwin = st.slider("Simulierte Fenster", *bounds("nwin_slider"), key="nwin_slider", step=1, help="Experiment 3; Fenster × Fensterlänge wird auf das Rechenbudget begrenzt.")
    sim_seed = st.number_input("Zufalls-Seed der Simulation", *bounds("sim_seed_input"), key="sim_seed_input", step=1, help="Experimente 2 und 3.")

    st.header("⚖️ Vergleich")
    simple = st.checkbox("Einfache Karte gleicher ARL₀ mitzeichnen", key="simple_check", help="Experiment 5: gestrichelt die einfache Karte, die dieselbe ARL₀ hat wie der Regelsatz.")

sync_query_params({k: st.session_state[k] for k in ("rules_sel", "side_radio", "c_input", "run4_sel", "seed_input", "n_slider", "nu_slider", "delta_slider",
                                                   "t0_slider", "runs_slider", "window_slider", "nwin_slider", "sim_seed_input", "simple_check")})

s = E.Settings(rules=int(rules), side=str(side), c=float(c), run4=int(run4), seed=int(seed), n=int(n), nu=int(nu), delta=float(delta), t0=int(t0),
               runs=int(runs), window=int(window), nwin=int(nwin), sim_seed=int(sim_seed), simple=bool(simple))
card = s.card_args()                      # (Regelsatz, zweiseitig, Schwelle, Fassung von Regel 4)
label = E.card_label(*card)

# ------------------------------------------------------------------------------------------------------------------ Rechnung (exakt)
rl0 = _rl(*card, 0.0)
x, codes, first = _series(s.seed, s.n, s.nu, s.delta, *card)
jump_in = s.delta > 0 and s.nu < s.n
p_before = rl0.cdf(s.nu) if s.nu > 0 else 0.0
rule_first = int(codes[first - 1]) if first else 0
delay_info = None
d_cdf = None
if jump_in:
    dl = _cond_delay(*card, s.delta, s.nu)
    delay_info = {"mean": dl.mean, "median": dl.quantile(0.5), "q95": dl.quantile(0.95)}
    if first is not None and first > s.nu:
        d_cdf = dl.cdf(first - s.nu)

q_med, q05, q95_, q99 = rl0.quantile(0.5), rl0.quantile(0.05), rl0.quantile(0.95), rl0.quantile(0.99)
p_t0 = rl0.cdf(s.t0)
p_1000 = rl0.sf(1000)
cd = _counts(*card, s.window)

# ------------------------------------------------------------------------------------------------------------------ Experiment 1 (Ergebnis)
st.markdown("## 🎯 Ergebnis – Experiment 1: eine Reihe durch die Karte")
m1, m2, m3, m4 = st.columns(4)
m1.metric("ARL₀ (exakt, zero-state)", f"{E.de_arl(rl0.mean)} Beobachtungen", delta=label, delta_color="off")
m2.metric(f"Fehlalarm in den ersten {s.nu} Beobachtungen" if s.nu > 0 else "Fehlalarm vor dem Sprung", E.de_pct(p_before, None), delta="exakt: P(T ≤ ν)", delta_color="off")
m3.metric("Erster Alarm", f"bei Beobachtung {first} (Regel {rule_first})" if first else f"kein Alarm in {s.n} Beobachtungen",
          delta=("Fehlalarm" if first <= s.nu else "nach dem Sprung") if (first and s.delta > 0 and s.nu < s.n) else None, delta_color="off")
m4.metric("Verzögerung dieser Reihe", f"{first - s.nu} Beobachtungen" if (jump_in and first and first > s.nu) else "–",
          delta=f"erwartet {E.de(delay_info['mean'], 1)}" if delay_info else None, delta_color="off")
st.plotly_chart(build_series(x, codes, first, s.nu, s.delta, s.rules, s.c_eff, s.two_sided), width="stretch", key="series_chart")
st.caption("Grau: Beobachtungen in σ-Einheiten; gestrichelt blau: Grenzen der Karte (bei Zusatzregeln dazu die Zonenlinien bei 1σ und 2σ); rot gepunktet: wahrer Mittelwert. "
           "Farbige Punkte: Beobachtungen, bei denen eine Regel anschlägt (ohne Neustart, die Regeln lesen die Reihe fortlaufend); der goldene Stern ist der erste Alarm, also die Lauflänge T dieser Ziehung. "
           "Zufall: SplitMix64 mit `normal()` (bitgleich auf jeder Plattform).")
_verdict(E.verdict_series(label=label, first=first, rule=rule_first, n=s.n, nu=s.nu, delta=s.delta, p_before=p_before, arl0=rl0.mean, delay=delay_info, d_cdf=d_cdf))

# ------------------------------------------------------------------------------------------------------------------ Kernfragen
st.markdown("## ❓ Kernfragen")
cmp4 = _compare(4, s.run4)
four_now = E.compare_at(cmp4, s.delta) if s.delta > 0 else (None, None)
arl1_now = _rl(*card, s.delta).mean if s.delta > 0 else None
dl_now = _rl(*card, s.delta) if s.delta > 0 else None
for q, a in E.kernfragen(
    first=first, rule=rule_first, n=s.n, label=label, arl0=rl0.mean, median=q_med, t0=s.t0, p_t0=p_t0, p_over_1000=p_1000, window=s.window, win_mean=cd["mean"],
    p0=cd["p0"], p_ge5=cd["p_ge5"], delta=s.delta, arl1=arl1_now, delay_median=dl_now.quantile(0.5) if dl_now else None, delay_q95=dl_now.quantile(0.95) if dl_now else None,
    four_arl0=cmp4["arl0"], four_arl1=four_now[0], four_simple=four_now[1], four_cross=cmp4["cross"],
):
    st.markdown(f"**{q}**  \n{a}")

st.markdown("---")

# ------------------------------------------------------------------------------------------------------------------ Experiment 2
st.subheader("🔬 Experiment 2: ARL₀ ist ein Mittel, die Lauflänge eine Verteilung")
st.caption("Zähleinheit: Reihen (Anteile) und Beobachtungen (Lauflänge), zero-state, ohne Änderung. Die Falle: „ARL = 370“ klingt nach 370 ruhigen Beobachtungen.")
tmax2 = int(min(max(q99, 2 * s.t0), 6000))
sim_len = None
cap_runs = S.max_runs(rl0.mean, s.rules, s.runs)
if cap_runs:
    est = cap_runs * rl0.mean * (0.8 if s.rules == 1 else 3.6) * 1e-6
    key_params = (s.rules, s.two_sided, s.c_eff, s.run4, cap_runs, s.sim_seed)
    sim_len = _simulate("simrun", key_params, f"▶ {cap_runs} Läufe simulieren (grob {E.de(est, 1)} s)" + (f", auf {cap_runs} von {s.runs} begrenzt" if cap_runs < s.runs else ""),
                        f"Simuliere {cap_runs} Läufe …", lambda: _sim_runs(s.rules, 0.0, cap_runs, s.sim_seed, s.c_eff, s.two_sided, s.run4))
else:
    st.caption(f"Simulation hier nicht angeboten: ARL₀ = {E.de(rl0.mean, 0)} Beobachtungen, dafür reicht das Rechenbudget für weniger als {C.SIM_MIN_RUNS} Läufe. Die exakten Werte stehen oben.")
st.plotly_chart(build_cdf(tmax2, rl0.cdf, s.t0, sim_len, title="Verteilungsfunktion der Lauflänge ohne Änderung"), width="stretch", key="cdf_chart")
sim_sum = E.sim_run_summary(sim_len, rl0, s.t0) if sim_len is not None else None
_verdict(E.verdict_runlength(label=label, arl0=rl0.mean, median=q_med, t0=s.t0, p_t0=p_t0, p_over_1000=p_1000, sim=sim_sum))
if sim_sum is not None:
    st.caption(f"Simulation (Zufalls-Seed {s.sim_seed}, {sim_sum['runs']} Läufe): Mittel {E.de(sim_sum['mean'], 1)} ± {E.de(sim_sum['se'], 1)} gegen exakt {E.de_arl(rl0.mean)} (z = {E.de(sim_sum['z_mean'], 1)}), "
               f"Anteil der Reihen mit Alarm innerhalb von {s.t0}: {E.de_pct(sim_sum['share'], 2)} ± {E.de(100 * sim_sum['se_share'], 2)} gegen exakt {E.de_pct(sim_sum['exact_share'], 2)} (z = {E.de(sim_sum['z_share'], 1)}). "
               "Die Simulation ist ein Band, kein exakter Wert.")
cols = st.columns(5)
for col, (lab, val) in zip(cols, (("5-%-Quantil", q05), ("Median", q_med), ("95-%-Quantil", q95_), ("99-%-Quantil", q99), ("Länger als 1000 ruhig", E.de_pct(p_1000, None)))):
    col.metric(lab, f"{val}" if isinstance(val, int) else val)
arl_table = []
for g in (50, 100, 200, 370, 500, 1000, 5000):
    pg = 1.0 / g
    arl_table.append({"ARL₀ (einfache Karte)": g, f"P(T ≤ {s.t0}) in %": 100 * float(1 - K.geometric_sf(pg, s.t0)), "Median": K.geometric_quantile(pg, 0.5)})
st.markdown(f"**Wie „Alarm innerhalb der ersten {s.t0} Beobachtungen“ an der ARL₀ hängt** (einfache Karte, exakt geometrisch; Zähleinheit Reihen):")
st.dataframe(pd.DataFrame(arl_table), hide_index=True, width="stretch",
             column_config={f"P(T ≤ {s.t0}) in %": st.column_config.NumberColumn(format="%.2f %%"), "ARL₀ (einfache Karte)": st.column_config.NumberColumn(format="%d"), "Median": st.column_config.NumberColumn(format="%d")})
if s.rules > 1:
    eq = K.c_for_arl(rl0.mean, True)
    p_eq = 1.0 / rl0.mean
    s_curve = rl0.survival_curve(len(rl0.s) - 1)
    gap = H.max_distance_to_geometric(s_curve, rl0.mean)
    st.markdown(f"**Negativbefund: mehr Regeln lösen das nicht.** Bei gleicher ARL₀ = {E.de_arl(rl0.mean)} (einfache Karte mit c = {E.de(eq, 4)}σ) ist die Lauflänge der Regeln der der einfachen Karte in Median und oberem Rand meist ähnlich, früh aber verschieden: "
                f"P(T ≤ {s.t0}) = {E.de_pct(p_t0, 2)} gegen {E.de_pct(1 - (1 - p_eq) ** s.t0, 2)}, Median {q_med} gegen {K.geometric_quantile(p_eq, 0.5)}; "
                f"der größte Abstand der Verteilungsfunktionen ist {E.de(gap, 4)}, innerhalb der ersten 10 Beobachtungen alarmieren die Regeln in {E.de_pct(rl0.cdf(10), 2)} der Reihen, die einfache Karte in {E.de_pct(1 - (1 - p_eq) ** 10, 2)} "
                "(exakt, Zähleinheit Reihen). Die Verteilung verschiebt sich wesentlich nur mit der ARL₀ (Tabelle oben).")

st.markdown("---")

# ------------------------------------------------------------------------------------------------------------------ Experiment 3
st.subheader("🔬 Experiment 3: wie viele Fehlalarme in einem Fenster?")
st.caption("Zähleinheit: Fenster zu W Beobachtungen ohne Änderung, Neustart nach Alarm ohne Vorgeschichte. Festlegung dieser Seite: ARL ist nur sinnvoll, wenn bei Alarm angehalten wird (Zwetsloot u. a. 2024); "
           "hier wird nach jedem Alarm ohne Vorgeschichte neu gestartet.")
sim_cnt = None
budget_w = C.SIM_BUDGET_SIMPLE if s.rules == 1 else C.SIM_BUDGET_RULES
cap_nwin = int(min(s.nwin, budget_w // s.window))
if cap_nwin >= 20:
    est = cap_nwin * s.window * (0.9 if s.rules == 1 else 4.7) * 1e-6
    key_params = (s.rules, s.two_sided, s.c_eff, s.run4, s.window, cap_nwin, s.sim_seed)
    sim_cnt = _simulate("simwin", key_params, f"▶ {cap_nwin} Fenster simulieren (grob {E.de(est, 1)} s)" + (f", auf {cap_nwin} von {s.nwin} begrenzt" if cap_nwin < s.nwin else ""),
                        f"Simuliere {cap_nwin} Fenster zu {s.window} Beobachtungen …", lambda: _sim_windows(s.rules, s.window, cap_nwin, s.sim_seed, s.c_eff, s.two_sided, s.run4))
else:
    st.caption("Simulation hier nicht angeboten: Fenster × Fensterlänge übersteigt das Rechenbudget; die exakte Verteilung steht unten.")
st.plotly_chart(build_counts(cd["pmf"], sim_cnt, s.window), width="stretch", key="counts_chart")
win_sim = None
if sim_cnt is not None:
    win_sim = {"n": len(sim_cnt), "mean": float(sim_cnt.mean()), "se": float(sim_cnt.std(ddof=1) / np.sqrt(len(sim_cnt))), "p0": float(np.mean(sim_cnt == 0))}
_verdict(E.verdict_windows(label=label, window=s.window, mean=cd["mean"], p0=cd["p0"], p_ge5=cd["p_ge5"], t_over_arl0=cd["t_over_arl0"], sim=win_sim))
cols = st.columns(5)
for col, (lab, val) in zip(cols, (("Mittel", E.de(cd["mean"], 3)), ("P(0 Alarme)", E.de_pct(cd["p0"], None)), ("P(≥ 5 Alarme)", E.de_pct(cd["p_ge5"], 2)), ("P(≥ 8 Alarme)", E.de_pct(cd["p_ge8"], None)),
                               ("Quantile 5 / 50 / 95 %", f"{cd['q05']} / {cd['q50']} / {cd['q95']}"))):
    col.metric(lab, val)
if "binomial" in cd:
    st.caption(f"Einfache Karte: zweiter Rechenweg, Binomialverteilung (W, p): größter Abstand zur Erneuerungsrechnung {cd['max_gap_binomial']:.1e}. Mit Neustart ist die Alarmzahl der gedächtnislosen Karte binomial; T/ARL₀ = {E.de(cd['t_over_arl0'], 3)}.")
else:
    st.caption(f"Erneuerungsrechnung aus der exakten Lauflängenverteilung: Mittel {E.de(cd['mean'], 3)} Alarme; die Näherung W/ARL₀ gibt {E.de(cd['t_over_arl0'], 3)} (Abweichung {E.de(abs(cd['t_over_arl0'] - cd['mean']), 3)}).")
if sim_cnt is not None:
    st.caption(f"Der Standardfehler eines Anteils aus {len(sim_cnt)} Fenstern ist √(p(1−p)/N), hier rund {E.de(100 * (win_sim['p0'] * (1 - win_sim['p0']) / len(sim_cnt)) ** 0.5, 2)} Prozentpunkte beim Anteil ohne Alarm: die Simulation ist ein Band (±2 Standardfehler im Diagramm), kein exakter Wert.")

st.markdown("---")

# ------------------------------------------------------------------------------------------------------------------ Experiment 4
st.subheader("🔬 Experiment 4: wie lange dauert es nach einem Sprung?")
st.caption("Zähleinheit: Beobachtungen (Verzögerung D = T − ν ≥ 1), zero-state; bei Zusatzregeln zusätzlich steady-state (quasistationär), Pollak und Lorden. δ aus der Seitenleiste.")
grid = C.DELTA_GRID
arl1s = _arl1(*card, tuple(grid))
st.plotly_chart(build_arl1_curve(list(grid), {label: arl1s}, marked_delta=s.delta, colors={label: C.COLOR_EXACT}), width="stretch", key="arl1_chart")
if s.delta > 0:
    rl_d = _rl(*card, s.delta)
    meas = _measures(*card, s.delta) if s.rules > 1 else None
    _verdict(E.verdict_delay(label=label, delta=s.delta, arl1=rl_d.mean, median=rl_d.quantile(0.5), q95=rl_d.quantile(0.95), p_over_100=rl_d.sf(100), measures=meas))
    tmax4 = int(min(max(rl_d.quantile(0.995), 30), 3000))
    st.plotly_chart(build_cdf(tmax4, rl_d.cdf, None, None, title=f"Verzögerung nach einem Sprung um {E.fmt_c(s.delta)}σ (zero-state)", xlabel="Verzögerung D in Beobachtungen", ylabel="P(D ≤ d)"),
                    width="stretch", key="delay_chart")
    if meas is not None:
        rows = [("zero-state (Start ohne Vorgeschichte)", meas.zero), ("steady-state (quasistationär)", meas.steady), (f"Pollak (Maximum über ν ≤ {C.POLLAK_NU_MAX}, bei ν = {meas.pollak_nu})", meas.pollak),
                ("Lorden (schlechtester erreichbarer Zustand)", meas.lorden)]
        st.dataframe(pd.DataFrame([{"Maß": a, "mittlere Verzögerung (Beobachtungen)": b} for a, b in rows]), hide_index=True, width="stretch",
                     column_config={"mittlere Verzögerung (Beobachtungen)": st.column_config.NumberColumn(format="%.3f")})
        st.caption(f"Bei diesem Regelsatz und δ = {E.fmt_c(s.delta)}σ liegt das Verhältnis steady-state zu zero-state bei {E.de(meas.steady / meas.zero, 3)}; zero-state, Pollak und Lorden fallen zusammen: "
                   f"{'ja' if max(abs(meas.zero - meas.pollak), abs(meas.zero - meas.lorden)) < 1e-9 else 'nein'}. Gedeutet, nicht bewiesen: eine Vorgeschichte kann den Zählern einer Regel nur etwas hinzufügen, nie einen Alarm verhindern, "
                   "deshalb ist der Start ohne Vorgeschichte hier der schlechteste Zustand.")
    else:
        st.caption("Die einfache Karte ist gedächtnislos: zero-state, steady-state, Pollak und Lorden sind gleich 1/p(δ).")
else:
    _verdict(E.verdict_delay(label=label, delta=0.0, arl1=rl0.mean, median=q_med, q95=q95_, p_over_100=0.0, measures=None))
st.caption("Ausblick, hier nicht gerechnet: ein Gedächtnis ändert die Verzögerung stark (einseitig, ARL₀ = 370, δ = 1: einfache Karte 26,746 Beobachtungen, CUSUM mit k = 0,5 8,573; Zahl aus der Konzepte-Reihe, nicht aus dieser Demo).")

st.markdown("---")

# ------------------------------------------------------------------------------------------------------------------ Experiment 5
st.subheader("🔬 Experiment 5: Zusatzregeln, schneller, aber nicht umsonst")
st.caption("Zähleinheit: Beobachtungen, zero-state, zweiseitig. Verglichen wird bei **gleicher ARL₀**: die einfache Karte bekommt die Schwelle c, die dieselbe ARL₀ gibt wie der Regelsatz.")
cmp = {rc: _compare(rc, s.run4 if rc == 4 else 8) for rc in (2, 3, 4)}
series_plot, colors, dashes = {}, {}, {}
for rc in (2, 3, 4):
    nm = f"Regeln {C.RULE_SHORT[rc]}"
    series_plot[nm] = cmp[rc]["rules_arl1"]
    colors[nm] = {2: C.COLOR_SIMPLE, 3: C.COLOR_SIM, 4: C.COLOR_RULES}[rc]
    if s.simple:
        nm2 = f"einfach, ARL₀ = {E.de_arl(cmp[rc]['arl0'])}"
        series_plot[nm2] = cmp[rc]["simple_arl1"]
        colors[nm2] = colors[nm]
        dashes[nm2] = "dash"
st.plotly_chart(build_arl1_curve(list(C.DELTA_GRID), series_plot, marked_delta=s.delta, cross=cmp[4]["cross"], colors=colors, dashes=dashes), width="stretch", key="compare_chart")
st.caption("Durchgezogen: Regelsätze; gestrichelt: die einfache Karte mit derselben ARL₀ (gleiche Farbe). Die Kreuzung im Bild gehört zu 1+2+3+4.")
rows5 = []
for rc in (2, 3, 4):
    ci = cmp[rc]
    row = {"Regelsatz": C.RULE_NAMES[rc], "ARL₀": ci["arl0"], "c der einfachen Karte": ci["c_eq"], "Kreuzung δ* (σ)": ci["cross"] if ci["cross"] is not None else float("nan")}
    if s.delta > 0:
        r_, e_ = E.compare_at(ci, s.delta)
        row.update({f"ARL₁ Regeln (δ = {E.fmt_c(s.delta)})": r_, "ARL₁ einfach": e_, "Verhältnis": r_ / e_})
    rows5.append(row)
st.dataframe(pd.DataFrame(rows5), hide_index=True, width="stretch",
             column_config={"ARL₀": st.column_config.NumberColumn(format="%.2f"), "c der einfachen Karte": st.column_config.NumberColumn(format="%.4f"),
                            "Kreuzung δ* (σ)": st.column_config.NumberColumn(format="%.2f"), "ARL₁ einfach": st.column_config.NumberColumn(format="%.2f"),
                            "Verhältnis": st.column_config.NumberColumn(format="%.3f"), f"ARL₁ Regeln (δ = {E.fmt_c(s.delta)})": st.column_config.NumberColumn(format="%.2f")})
steady4 = None
if s.delta > 0:
    steady4 = _measures(4, True, 3.0, s.run4, s.delta).steady
r4 = cmp[4]
arl_r, arl_e = E.compare_at(r4, s.delta) if s.delta > 0 else (None, None)
_verdict(E.verdict_rules(rule_count=4, run4=s.run4, delta=s.delta, arl0=r4["arl0"], c_eq=r4["c_eq"], arl1_rules=arl_r, arl1_simple=arl_e, cross=r4["cross"], steady=steady4))
st.markdown("**Fassungsfalle: Regel 4 mit 8, 9 oder 10 in Folge** (zweiseitig, zero-state; Zähleinheit Beobachtungen):")
ver = _versions(s.delta)
st.dataframe(pd.DataFrame([{"Fassung": v["label"], "ARL₀": v["arl0"], f"ARL₁ (δ = {E.fmt_c(s.delta)})": v["arl1"] if v["arl1"] is not None else float("nan")} for v in ver]),
             hide_index=True, width="stretch", column_config={"ARL₀": st.column_config.NumberColumn(format="%.2f"), f"ARL₁ (δ = {E.fmt_c(s.delta)})": st.column_config.NumberColumn(format="%.2f")})
st.caption(f"Wer „die vier Regeln“ sagt, ohne die Fassung zu nennen, nennt eine ARL₀ mit Faktor {E.de(ver[1]['arl0'] / ver[0]['arl0'], 2)} Unsicherheit (9 gegen 8 in Folge). Welche Fassung eine bestimmte Quelle meint, ist nicht gelesen.")
st.markdown("**Negativbefunde.** Das Maß der Verzögerung dreht die Rangfolge hier nicht (vier Regeln dieser Fassung, δ = 0,5; 1; 2; 3): zero-state, Pollak und Lorden fallen zusammen, steady-state ist kleiner, und in jedem dieser δ "
            "liegt das Verhältnis zur einfachen Karte in beiden gerechneten Maßen auf derselben Seite von 1. Und die Regeln gewinnen nicht überall: oberhalb der Kreuzung ist die einfache Karte schneller.")

st.markdown("---")

# ------------------------------------------------------------------------------------------------------------------ Grenzen
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.ASSUMPTIONS_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/lauflaenge-karte-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind unabhängige Beobachtungen $x_1, x_2, \dots$ in Einheiten der Streuung, ein Sprung des Mittelwerts um $\delta$ nach $\nu$ Beobachtungen und eine Alarmregel. Gesucht ist die Verteilung der Lauflänge $T$ bis zum Alarm, ihr Mittel, die Verzögerung nach dem Sprung und die Zahl der Alarme in einem Fenster.

| Symbol | Bedeutung |
|---|---|
| $x_t$ | Beobachtung $t$ in Einheiten der Streuung |
| $\delta$, $\nu$ | Sprung des Mittelwerts; Zahl der Beobachtungen vor der Änderung |
| $c$, $p(\delta)$ | Schwelle der einfachen Karte; Alarmwahrscheinlichkeit je Beobachtung |
| $T$, $D$ | Lauflänge bis einschließlich der alarmierenden Beobachtung; Verzögerung $D = T - \nu \ge 1$ |
| $\text{ARL}_0$, $\text{ARL}_1$ | Mittel von $T$ ohne Änderung und nach einem Sprung |
| $R_\delta$, $\mathbf{e}_0$ | Übergangsmatrix der Karte ohne Alarm; Start ohne Vorgeschichte |
| $W$, $N_W$, $S_k$ | Fensterlänge; Zahl der Alarme im Fenster; Summe von $k$ Lauflängen |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/lauflaenge-karte-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(T.MATH_MARKDOWN)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Änderungserkennung: Alarm und Lauflänge](https://sebastianhanisch.net/konzepte-aenderungserkennung.html)."
)
