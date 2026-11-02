# -*- coding: utf-8 -*-
"""
Figures du site (plotly). Le JavaScript de plotly n'est PAS recopié dans chaque figure : la page
(``app.page``) le charge une seule fois depuis ``/static/plotly.min.js`` (mis en cache par le
navigateur) ; une page de run passe ainsi de ~19 Mo à quelques centaines de ko.
"""
from __future__ import annotations

import html

import data

FONT = {"family": "system-ui, Segoe UI, sans-serif"}
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1a9850", "#898781"
# légende à droite, au-dessus du tracé : ne recouvre pas le titre, qui est à gauche
LEGEND_TOP = {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1}


def _n(value: float, fmt: str = ",.0f") -> str:
    """Nombre à la française : espace insécable pour les milliers."""
    return format(value, fmt).replace(",", " ")


def _html(fig) -> str:
    """Figure en fragment HTML, sans le JavaScript de plotly (déjà chargé par la page)."""
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})


def section_html(run: data.Run) -> str:
    """Profil de la meilleure section contre la section de départ, puis scores PCA."""
    d = data.section_figure_data(run)
    if d is None:
        return '<p class="warn">Prévisualisation de la section indisponible (PCA introuvable).</p>'
    import plotly.graph_objects as go  # noqa: PLC0415

    x = d["stations"]
    prof = go.Figure()
    if d["start"] is not None:
        for key, show in (("start", True), ("start_lower", False)):
            prof.add_trace(go.Scatter(x=x, y=d[key], name="section de départ", showlegend=show,
                                      legendgroup="start", line={"color": GREY, "width": 2, "dash": "dot"}))
    for key, show in (("best", True), ("best_lower", False)):
        prof.add_trace(go.Scatter(x=x, y=d[key], name="meilleure section", showlegend=show,
                                  legendgroup="best", line={"color": BLUE, "width": 2.5}))
    prof.update_layout(height=320, margin={"t": 60, "b": 40, "l": 50, "r": 20}, font=FONT,
                       title={"text": f"Profil — {run.name}", "font": {"size": 15}},
                       legend=LEGEND_TOP,
                       xaxis={"title": "abscisse [c]"},
                       yaxis={"title": "ordonnée [c]", "scaleanchor": "x", "scaleratio": 1})
    modes = [f"mode {i}" for i in range(1, len(d["scores"]) + 1)]
    bar = go.Figure()
    if d["start_scores"] is not None:
        bar.add_trace(go.Bar(x=modes, y=d["start_scores"], name="section de départ", marker={"color": GREY}))
    bar.add_trace(go.Bar(x=modes, y=d["scores"], name="meilleure section", marker={"color": BLUE}))
    for o in d["optima"]:
        bar.add_trace(go.Bar(x=[f"mode {i}" for i in range(1, len(o["scores"]) + 1)], y=o["scores"],
                             name=f"optimum local (région {o['region']})",
                             marker={"color": ORANGE, "opacity": 0.8}))
    bar.update_layout(height=340, margin={"t": 60, "b": 40, "l": 50, "r": 20}, font=FONT, barmode="group",
                      title={"text": "Scores PCA (centrés réduits) de la section", "font": {"size": 15}},
                      legend=LEGEND_TOP)
    return _html(prof) + _html(bar)


def speeds_html(run: data.Run) -> str:
    """Vitesse de la meilleure section par condition (vent/tack)."""
    speeds = run.best.get("speeds_by_condition")
    if not speeds:
        return ""
    import plotly.graph_objects as go  # noqa: PLC0415

    items = [(k, v) for k, v in speeds.items() if v is not None and v == v]
    missing = [k for k, v in speeds.items() if v is None or v != v]
    if not items:
        return ""
    fig = go.Figure(go.Bar(x=[k for k, _ in items], y=[v for _, v in items], marker={"color": BLUE},
                           text=[f"{v:.1f}" for _, v in items], textposition="outside",
                           textfont={"size": 9}, name="vitesse [nd]"))
    title = f"Vitesses par condition (TWA/TWS) — moyenne {run.mean_speed:.3f} nd" if run.mean_speed else run.name
    if missing:
        title += f" — non résolues : {', '.join(missing[:6])}{'…' if len(missing) > 6 else ''}"
    fig.update_layout(height=420, margin={"t": 60, "b": 70, "l": 50, "r": 20}, font=FONT,
                      title={"text": title, "font": {"size": 15}},
                      xaxis={"title": "condition TWA/TWS", "tickangle": -60},
                      yaxis={"title": "vitesse [nœuds]", "range": [0, max(v for _, v in items) * 1.15]})
    note = ""
    if any(v >= 23.999 for _, v in items):
        note = ('<p class="small">Les barres à 24,0 nd sont le <b>plafond de vitesse de la VPP</b> : '
                "ce n'est pas un équilibre calculé (à lire comme « très rapide »).</p>")
    return _html(fig) + note


def gain_html(run: data.Run, distance_nm: float = data.REFERENCE_DISTANCE_NM,
              eur_per_h: float | None = None) -> str:
    """Gain VALIDÉ de la meilleure section sur la section de départ, et ce qu'il représente.

    Le gain est celui du run lui-même (``summary.txt``) : vitesses validées contre le code de
    référence, moyenne pondérée par la distribution d'exploitation. La conversion en temps (et en
    euros si ``eur_per_h`` est donné) est une simple règle de trois sur ``distance_nm``.
    """
    g = data.validated_gain(run)
    if not g:
        return ('<p class="small">Gain validé indisponible : <span class="mono">validation.csv</span> '
                "n'est pas (encore) écrit ou ne correspond pas à la config du run.</p>")
    import plotly.graph_objects as go  # noqa: PLC0415

    hours = data.time_saved_h(g["start_speed"], g["best_speed"], distance_nm)
    money = ""
    if eur_per_h:
        amount = _n(hours * eur_per_h, "+,.0f")
        money = (f'<div class="vfield"><div class="vlabel">Valeur à {eur_per_h:g} €/h</div>'
                 f'<div class="vval mono">{amount} €</div></div>')
    cards = f"""
<div class="finrow">
  <div class="vfield"><div class="vlabel">Section de départ → meilleure (validées)</div>
    <div class="vval mono">{g['start_speed']:.3f} → {g['best_speed']:.3f} nd</div>
    <span class="vnote">départ : {html.escape(g['start_name'])}</span></div>
  <div class="vfield"><div class="vlabel">Gain de vitesse</div>
    <div class="vval mono">{g['gain_kt']:+.3f} nd ({g['gain_pct']:+.2f} %)</div>
    <span class="vnote">{g['n_conditions']} conditions, {100 * g['probability']:.0f} % de la probabilité</span></div>
  <div class="vfield"><div class="vlabel">Temps gagné sur {_n(distance_nm)} nm</div>
    <div class="vval mono">{hours:+.2f} h</div>
    <span class="vnote">règle de trois à vitesse moyenne constante</span></div>
  {money}
</div>"""
    speeds = data.validated_speeds(run)["speeds"]
    order = sorted(speeds, key=speeds.get)
    colors = [GREEN if n == "best" else (GREY if n == g["start_name"] else BLUE) for n in order]
    fig = go.Figure(go.Bar(y=order, x=[speeds[n] for n in order], orientation="h", marker={"color": colors},
                           text=[f"{speeds[n]:.3f}" for n in order], textposition="outside"))
    lo = min(speeds.values())
    fig.update_layout(height=60 + 34 * len(order), margin={"t": 40, "b": 40, "l": 260, "r": 40}, font=FONT,
                      title={"text": "Vitesse validée de chaque section du run (nd)", "font": {"size": 15}},
                      xaxis={"title": "vitesse moyenne pondérée [nd]", "range": [lo - 0.15, max(speeds.values()) + 0.1]})
    return cards + _html(fig)


def incumbent_html(run: data.Run) -> str:
    """Évolution dans le temps de la vitesse PRÉDITE du meilleur candidat (ou de l'activité du run)."""
    series = data.incumbent_series(run)
    if not series:
        return '<p class="warn">Évolution indisponible (ni updates.json ni evaluations.csv lus).</p>'
    import plotly.graph_objects as go  # noqa: PLC0415

    times = [s.get("elapsed_min") or 0.0 for s in series]
    has_speed = any(s.get("speed") is not None for s in series)
    if has_speed:
        values = [s["speed"] if s.get("speed") is not None else float("nan") for s in series]
        y_title = "vitesse prédite du meilleur candidat [nd]"
    else:
        values = [s.get("cost") or (i + 1) for i, s in enumerate(series)]
        y_title = "calculs réussis (activité du run)"
    fig = go.Figure(go.Scatter(x=times, y=values, mode="lines+markers",
                               line={"color": GREEN, "width": 1.8}, marker={"size": 4, "color": GREEN}))
    fig.update_layout(height=340, margin={"t": 50, "b": 46, "l": 60, "r": 20}, font=FONT, showlegend=False,
                      paper_bgcolor="#fbfdfb", plot_bgcolor="rgba(26,152,80,0.03)",
                      title={"text": f"Évolution du meilleur candidat — {run.name}", "font": {"size": 15}},
                      xaxis={"title": "temps écoulé [min]", "range": [0, (times[-1] or 1.0) * 1.03],
                             "gridcolor": "rgba(26,152,80,0.15)"},
                      yaxis={"title": y_title, "gridcolor": "rgba(26,152,80,0.15)"})
    return _html(fig)


def bounds_html(box: str, scale: float) -> str:
    """La zone où le run peut chercher : sections admissibles, effet de chaque contrainte, bornes."""
    import plotly.graph_objects as go  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    s = data.bounds_summary(box, scale)
    x = list(s["x"])
    env = go.Figure()
    for lo, hi, label in (("upper_min", "upper_max", "extrados"), ("lower_min", "lower_max", "intrados")):
        env.add_trace(go.Scatter(x=x, y=list(s[hi]), line={"width": 0}, showlegend=False, hoverinfo="skip"))
        env.add_trace(go.Scatter(x=x, y=list(s[lo]), fill="tonexty", fillcolor="rgba(42,120,214,0.25)",
                                 line={"width": 0}, name=f"zone explorable ({label})",
                                 showlegend=(label == "extrados"), legendgroup="zone"))
    for key, show in (("current_upper", True), ("current_lower", False)):
        env.add_trace(go.Scatter(x=x, y=list(s[key]), name="section C-foil actuelle", showlegend=show,
                                 legendgroup="cur", line={"color": GREEN, "width": 2.5}))
    env.update_layout(height=350, margin={"t": 60, "b": 40, "l": 50, "r": 20}, font=FONT,
                      title={"text": f"Enveloppe de {s['n_samples']} sections admissibles tirées dans la zone",
                             "font": {"size": 15}},
                      legend=LEGEND_TOP, xaxis={"title": "abscisse [c]"},
                      yaxis={"title": "ordonnée [c]", "scaleanchor": "x", "scaleratio": 1})

    labels = [f"extrados {i}" for i in range(1, 9)] + [f"intrados {i}" for i in range(1, 9)]
    wts = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.12,
                        subplot_titles=("Bornes des 16 poids de Kulfan", "Projection des sections admissibles"))
    wts.add_trace(go.Bar(y=labels, x=list(s["w_hi"] - s["w_lo"]), base=list(s["w_lo"]), orientation="h",
                         marker={"color": "rgba(235,104,52,0.25)", "line": {"color": ORANGE, "width": 1}},
                         name="bornes de la boîte"), row=1, col=1)
    wts.add_trace(go.Bar(y=labels, x=list(s["w_max"] - s["w_min"]), base=list(s["w_min"]), orientation="h",
                         width=0.35, marker={"color": BLUE}, name="étendue des sections admissibles"),
                  row=1, col=1)
    wts.add_trace(go.Scatter(y=labels, x=list(s["w_current"]), mode="markers", name="section actuelle",
                             marker={"color": GREEN, "size": 8, "symbol": "diamond"}), row=1, col=1)
    pts, cur = s["scatter"]
    wts.add_trace(go.Scatter(x=list(pts[:, 0]), y=list(pts[:, 1]), mode="markers", showlegend=False,
                             marker={"size": 4, "color": "rgba(42,120,214,0.45)"}), row=1, col=2)
    wts.add_trace(go.Scatter(x=[float(cur[0])], y=[float(cur[1])], mode="markers", showlegend=False,
                             marker={"size": 13, "symbol": "cross", "color": GREEN, "line": {"width": 3}}),
                  row=1, col=2)
    wts.update_yaxes(autorange="reversed", row=1, col=1)
    wts.update_xaxes(title_text="poids de Kulfan", row=1, col=1)
    wts.update_xaxes(title_text="mode de forme 1", row=1, col=2)
    wts.update_yaxes(title_text="mode de forme 2", row=1, col=2)
    wts.update_layout(height=470, margin={"t": 60, "b": 46, "l": 90, "r": 20}, font=FONT, barmode="overlay",
                      legend={"orientation": "h", "y": -0.18})
    return _html(env) + _html(wts)
