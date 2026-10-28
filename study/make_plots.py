"""Figures of the prompting study (plotly, one HTML file): results/*.json -> figures/etude_prompts.html."""
import glob
import json
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {"A": "A. Énoncé seul", "B": "B. Tests fournis", "C": "C. Formules données", "D": "D. Vérification imposée"}
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
SYMBOLS = {"A": "circle", "B": "square", "C": "diamond", "D": "triangle-up"}
GROUPS = {"Noyaux": "test_kernel_values", "Vraisemblance": "test_lml_fixed", "Prédiction": "test_predict_fixed",
          "Interpolation": "test_interpolation", "LOO": "test_loo_fixed", "Cas singulier": "test_near_singular",
          "fit (Branin)": "test_fit_branin"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load():
    runs = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(HERE, "results", "*.json")))]
    return sorted(runs, key=lambda r: (r["arm"], r["run"]))


def group_scores(run):
    acc = run["acceptance"]
    out = {}
    for label, prefix in GROUPS.items():
        tests = [v for k, v in acc.items() if k.startswith(prefix)]
        out[label] = (sum(tests), len(tests)) if tests else (0, 0)
    return out


def main():
    runs = load()
    fig = make_subplots(rows=4, cols=1, vertical_spacing=0.08, row_heights=[0.27, 0.24, 0.31, 0.18],
                        subplot_titles=("Tests d'acceptance réussis (sur 23), par passe",
                                        "Avant → après correction par Claude (lignes modifiées)",
                                        "Réussite par groupe de tests (avant correction)",
                                        "Durée de chaque passe (min)"))
    # 1. score per run, one trace per arm
    for arm, label in ARMS.items():
        rs = [r for r in runs if r["arm"] == arm]
        if not rs:
            continue
        fig.add_trace(go.Scatter(
            x=[label] * len(rs), y=[r["acceptance_passed"] for r in rs], mode="markers+text",
            text=[r["run"] for r in rs], textposition="middle right", textfont=dict(color=MUTED, size=11),
            marker=dict(size=13, color=COLORS[arm], symbol=SYMBOLS[arm], line=dict(color="#fcfcfb", width=2)),
            name=label, legendgroup=arm,
            customdata=[[r["stop"], r["steps"], r["minutes"], r["own_tests"]] for r in rs],
            hovertemplate="%{text} : %{y}/23<br>arrêt : %{customdata[0]}<br>%{customdata[1]} étapes, "
                          "%{customdata[2]:.0f} min<br>ses tests : %{customdata[3]}<extra></extra>"), row=1, col=1)
    fig.update_yaxes(range=[-1, 24.5], dtick=5, row=1, col=1, title_text="tests réussis")

    # 2. before / after Claude's minimal fix (fixes/fixes.json), one dumbbell per run
    fixes_path = os.path.join(HERE, "fixes", "fixes.json")
    fixes = json.load(open(fixes_path, encoding="utf-8")) if os.path.exists(fixes_path) else {}
    for r in runs:
        fx = fixes.get(r["run"])
        if not fx:
            continue
        arm = r["arm"]
        fig.add_trace(go.Scatter(x=[r["run"]] * 2, y=[r["acceptance_passed"], fx["after"]], mode="lines",
                                 line=dict(color=GRID, width=2), showlegend=False, hoverinfo="skip"), row=2, col=1)
        fig.add_trace(go.Scatter(x=[r["run"]], y=[r["acceptance_passed"]], mode="markers", showlegend=False,
                                 marker=dict(size=12, color=COLORS[arm], symbol=SYMBOLS[arm], line=dict(color="#fcfcfb", width=2)),
                                 hovertemplate=f"{r['run']} avant : %{{y}}/23<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(x=[r["run"]], y=[fx["after"]], mode="markers+text", showlegend=False,
                                 text=[f"{fx['lines']} l." if fx["lines"] else "—"], textposition="top center",
                                 textfont=dict(color=MUTED, size=11),
                                 marker=dict(size=12, color="#fcfcfb", symbol=SYMBOLS[arm], line=dict(color=COLORS[arm], width=2)),
                                 hovertemplate=f"{r['run']} après : %{{y}}/23, {fx['lines']} ligne(s)<br>{fx['cause']}<extra></extra>"),
                      row=2, col=1)
    fig.update_xaxes(categoryorder="array", categoryarray=[r["run"] for r in runs], row=2, col=1)
    fig.update_yaxes(range=[10, 26.5], dtick=5, row=2, col=1, title_text="tests réussis")

    # 3. heatmap runs x test groups (fraction passed), single-hue sequential scale
    names = [r["run"] for r in runs]
    z, text = [], []
    for label in GROUPS:
        row_z, row_t = [], []
        for r in runs:
            ok, n = group_scores(r)[label]
            row_z.append(ok / n if n else None)
            row_t.append(f"{ok}/{n}" if n else "-")
        z.append(row_z)
        text.append(row_t)
    fig.add_trace(go.Heatmap(z=z, x=names, y=list(GROUPS), text=text, texttemplate="%{text}",
                             textfont=dict(color=INK, size=11), xgap=2, ygap=2, zmin=0, zmax=1, showscale=False,
                             colorscale=[[0, "#f0efec"], [1, "#86b6ef"]],
                             hovertemplate="%{x} · %{y} : %{text}<extra></extra>"), row=3, col=1)
    fig.update_yaxes(autorange="reversed", row=3, col=1)

    # 4. duration per run, coloured by arm
    for arm, label in ARMS.items():
        rs = [r for r in runs if r["arm"] == arm]
        if rs:
            fig.add_trace(go.Bar(x=[r["run"] for r in rs], y=[r["minutes"] for r in rs], name=label, legendgroup=arm,
                                 showlegend=False, marker=dict(color=COLORS[arm], line=dict(color="#fcfcfb", width=2)),
                                 customdata=[[r["steps"], r["stop"]] for r in rs],
                                 hovertemplate="%{x} : %{y:.0f} min, %{customdata[0]} étapes<br>arrêt : %{customdata[1]}<extra></extra>"),
                          row=4, col=1)
    fig.update_xaxes(categoryorder="array", categoryarray=names, row=4, col=1)
    fig.update_yaxes(title_text="min", row=4, col=1)

    fig.update_layout(template="plotly_white", height=1350, width=900, font=dict(color=INK, size=13),
                      title=dict(text="Qwen3.8-27B en agent : effet du prompt sur la justesse du code (projet gp_lite)", x=0.02),
                      legend=dict(orientation="h", y=1.06, x=0.02), margin=dict(l=110, r=40, t=120, b=50),
                      paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb", bargap=0.35)
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures", "etude_prompts.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(out)


if __name__ == "__main__":
    main()
