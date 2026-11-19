"""Plotly-Abbildungen. Alle Achsen fest (fixedrange), damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt."""

import numpy as np
import plotly.graph_objects as go

import lfk_constants as C


def _lock(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(height=height, margin=dict(l=50, r=20, t=40, b=45), legend=dict(orientation="h", y=-0.22, x=0), hovermode="closest")
    return fig


def build_series(x, codes, first: int | None, nu: int, delta: float, rule_count: int, c: float, two_sided: bool) -> go.Figure:
    """Eine Reihe durch die Karte: Beobachtungen, Grenzen, Zonenlinien, Alarmpunkte mit auslösender Regel, Sprung."""
    n = len(x)
    t = np.arange(1, n + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=np.round(x, 3), mode="lines+markers", name="Beobachtung", line=dict(color="rgba(90,90,90,0.55)", width=1), marker=dict(size=4, color="rgba(90,90,90,0.8)"),
                             hovertemplate="t = %{x}<br>x = %{y:.2f}σ<extra></extra>"))
    # wahrer Mittelwert (Treppe) nur bei Sprung
    if delta > 0 and nu < n:
        fig.add_trace(go.Scatter(x=[1, nu + 0.5, nu + 0.5, n], y=[0, 0, delta, delta], mode="lines", name="wahrer Mittelwert", line=dict(color=C.COLOR_JUMP, width=2, dash="dot"), hoverinfo="skip"))
    lim = [c] if rule_count == 1 else [3.0]
    for lv in lim:
        fig.add_hline(y=lv, line=dict(color=C.COLOR_SIMPLE, width=1.5, dash="dash"), annotation_text=f"+{lv:g}σ", annotation_position="top right")
        if two_sided:
            fig.add_hline(y=-lv, line=dict(color=C.COLOR_SIMPLE, width=1.5, dash="dash"), annotation_text=f"−{lv:g}σ", annotation_position="bottom right")
    if rule_count > 1:
        for lv in (1.0, 2.0):
            for s in (1, -1):
                fig.add_hline(y=s * lv, line=dict(color=C.COLOR_GRID, width=1, dash="dot"))
    fig.add_hline(y=0.0, line=dict(color=C.COLOR_GRID, width=1))
    for rule in sorted(set(int(v) for v in codes if v)):
        idx = np.flatnonzero(np.asarray(codes) == rule)
        fig.add_trace(go.Scatter(x=idx + 1, y=np.round(np.asarray(x)[idx], 3), mode="markers", name=f"Alarm durch Regel {rule}",
                                 marker=dict(size=10, color=C.RULE_COLORS[rule], line=dict(color="white", width=1)),
                                 hovertemplate=f"Regel {rule} schlägt an<br>t = %{{x}}<extra></extra>"))
    if first is not None:
        fig.add_trace(go.Scatter(x=[first], y=[float(np.asarray(x)[first - 1])], mode="markers", name="erster Alarm", marker=dict(size=17, symbol="star", color="gold", line=dict(color="black", width=1.5)),
                                 hovertemplate="erster Alarm bei t = %{x}<extra></extra>"))
    if delta > 0 and 0 < nu < n:
        fig.add_vline(x=nu + 0.5, line=dict(color=C.COLOR_JUMP, width=2, dash="dash"), annotation_text="Sprung", annotation_position="top left")
    fig.update_xaxes(title_text="Beobachtung t", range=[0, n + 1])
    fig.update_yaxes(title_text="x in σ-Einheiten", range=[-5.5, 8.5])
    return _lock(fig, 420)


def build_cdf(tmax: int, exact_cdf, t0: int | None, sim_lengths=None, title: str = "", xlabel: str = "Beobachtungen t", ylabel: str = "P(T ≤ t)") -> go.Figure:
    """Verteilungsfunktion der Lauflänge P(T <= t): exakt (Linie), simulierte Läufe (Punkte), markierte Fensterlänge t0."""
    t = np.unique(np.concatenate([np.arange(0, min(tmax, 300) + 1), np.linspace(0, tmax, 300).astype(int)]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=np.round([exact_cdf(int(v)) for v in t], 6), mode="lines", name="exakt", line=dict(color=C.COLOR_EXACT, width=3),
                             hovertemplate="P(T ≤ %{x}) = %{y:.4f}<extra>exakt</extra>"))
    if sim_lengths is not None and len(sim_lengths):
        ts = np.unique(np.linspace(0, tmax, 60).astype(int))
        srt = np.sort(sim_lengths)
        emp = np.searchsorted(srt, ts, side="right") / len(srt)
        fig.add_trace(go.Scatter(x=ts, y=np.round(emp, 5), mode="markers", name=f"Simulation ({len(srt)} Läufe)", marker=dict(size=7, color=C.COLOR_SIM),
                                 hovertemplate="Anteil der Läufe mit T ≤ %{x}: %{y:.4f}<extra>Simulation</extra>"))
    if t0 is not None:
        fig.add_vline(x=t0, line=dict(color=C.COLOR_JUMP, width=2, dash="dash"), annotation_text=f"t₀ = {t0}", annotation_position="top right")
    fig.update_xaxes(title_text=xlabel, range=[0, tmax])
    fig.update_yaxes(title_text=ylabel, range=[0, 1.02])
    fig.update_layout(title=dict(text=title, font=dict(size=14)))
    return _lock(fig)


def build_counts(pmf, sim_counts=None, window: int = 1000) -> go.Figure:
    """Alarmzahl je Fenster: exakte Verteilung (Balken), Anteile der simulierten Fenster (Punkte mit Band ±2 Standardfehler)."""
    pmf = np.asarray(pmf)
    nz = np.flatnonzero(pmf > 1e-6)
    kmax = int(nz[-1]) if len(nz) else 5
    kmin = int(nz[0]) if len(nz) else 0
    ks = np.arange(max(0, kmin - 1), kmax + 2)
    ks = ks[ks < len(pmf)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ks, y=np.round(pmf[ks], 6), name="exakt (Erneuerungsrechnung)", marker_color=C.COLOR_EXACT, opacity=0.8,
                         hovertemplate="%{x} Alarme: %{y:.4f}<extra>exakt</extra>"))
    if sim_counts is not None and len(sim_counts):
        emp = np.array([np.mean(sim_counts == k) for k in ks])
        se = np.sqrt(np.maximum(emp * (1 - emp), 1e-12) / len(sim_counts))
        fig.add_trace(go.Scatter(x=ks, y=np.round(emp, 5), mode="markers", name=f"Simulation ({len(sim_counts)} Fenster, Band ±2 SE)",
                                 error_y=dict(type="data", array=np.round(2 * se, 5), visible=True), marker=dict(size=8, color=C.COLOR_SIM),
                                 hovertemplate="Anteil der Fenster mit %{x} Alarmen: %{y:.4f}<extra>Simulation</extra>"))
    fig.update_xaxes(title_text=f"Alarme je Fenster zu {window} Beobachtungen", dtick=1 if len(ks) <= 25 else None)
    fig.update_yaxes(title_text="Anteil der Fenster")
    fig.update_layout(barmode="overlay")
    return _lock(fig)


def build_arl1_curve(deltas, series: dict[str, list[float]], marked_delta: float | None = None, cross: float | None = None, colors: dict | None = None, dashes: dict | None = None) -> go.Figure:
    """ARL₁ (zero-state, Beobachtungen) über die Sprunggröße, log-Achse; markiert den gewählten Sprung und die Kreuzung."""
    fig = go.Figure()
    colors = colors or {}
    dashes = dashes or {}
    for name, ys in series.items():
        fig.add_trace(go.Scatter(x=deltas, y=np.round(ys, 4), mode="lines+markers", name=name, line=dict(width=3, color=colors.get(name), dash=dashes.get(name)),
                                 hovertemplate="δ = %{x}σ<br>ARL₁ = %{y:.2f} Beobachtungen<extra>" + name + "</extra>"))
    if marked_delta is not None and marked_delta > 0:
        fig.add_vline(x=marked_delta, line=dict(color=C.COLOR_JUMP, width=2, dash="dash"), annotation_text=f"δ = {marked_delta:g}σ", annotation_position="top right")
    if cross is not None:
        fig.add_vline(x=cross, line=dict(color="rgba(120,120,120,0.9)", width=1.5, dash="dot"), annotation_text=f"Kreuzung δ* = {cross:.2f}σ".replace(".", ","), annotation_position="bottom left")
    fig.update_xaxes(title_text="Sprung δ in σ-Einheiten", range=[0, max(deltas) + 0.2])
    fig.update_yaxes(title_text="ARL₁ (Beobachtungen, zero-state)", type="log")
    return _lock(fig)
