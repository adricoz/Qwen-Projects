# -*- coding: utf-8 -*-
"""
Site « VPP acceleration » : lancer une optimisation de section (``scripts/run_real_vpp.py``), la
suivre en direct, arrêter un run, et relire les runs terminés (gain validé, section, vitesses).

Serveur en Python standard (``http.server``), aucune dépendance web ; les figures utilisent plotly :

    python app.py [--port 8321]

Les runs sont lancés dans leur propre dossier ``vpp_acceleration/runs/<date>_<nom>/``. Le site
n'altère jamais un run existant, et quitter le site ne tue pas les runs en cours.
"""
from __future__ import annotations

import html
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import data
import figures
import launcher

# -----------------------------------------------------------------------------
# Style : tout est scanné sur #page, pour ne jamais entrer en conflit avec le CSS de plotly
# -----------------------------------------------------------------------------
CSS = """
:root{--ink:#1c1b1a;--sub:#6f6d68;--line:#e3e2dd;--bg:#f6f6f4;--card:#fff;
      --acc:#2a78d6;--ok:#199e70;--ko:#c0392b;--wait:#a04a00;
      --mono:ui-monospace,Consolas,monospace}
#page{max-width:1080px;margin:0 auto;padding:28px 20px 60px;font-family:system-ui,'Segoe UI',sans-serif;
      color:var(--ink);background:var(--bg);line-height:1.5;min-height:100vh;box-sizing:border-box}
#page *{box-sizing:border-box}
#page h1{font-size:24px;margin:8px 0 2px}
#page h2{font-size:17px;margin:28px 0 10px}
#page h3{font-size:14px;margin:0 0 8px}
#page .sub{color:var(--sub);margin:0 0 20px}
#page .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:12px 0}
#page table{border-collapse:collapse;width:100%;font-size:14px}
#page td,#page th{padding:7px 10px;border-bottom:1px solid #ecebe7;text-align:left}
#page tr:hover td{background:#fafaf8}
#page a{color:var(--acc);text-decoration:none}
#page a:hover{text-decoration:underline}
#page .mono{font-family:var(--mono);font-size:12px}
#page .small{font-size:12px;color:var(--sub)}
#page .warn{color:var(--wait);background:#fff7ee;border:1px solid #f2d9b8;padding:8px 12px;border-radius:8px}
#page .banner{background:#fff7ee;border:1px solid #f2d9b8;border-radius:10px;padding:12px 16px;margin:12px 0}
#page .ok{color:var(--ok);font-weight:600}
#page .ko{color:var(--ko);font-weight:600}
#page .wait{color:var(--wait);font-weight:600}
#page .log{background:#141312;color:#d8d6cf;border-radius:8px;padding:12px;font-family:var(--mono);
           font-size:12px;white-space:pre-wrap;max-height:340px;overflow:auto;margin:8px 0}
#page form button{padding:8px 16px;border:0;border-radius:6px;background:var(--acc);color:#fff;font-size:14px;cursor:pointer}
#page form button:hover{background:#1f66c0}
#page form button.stop{padding:.45rem 1.2rem;font-size:.95rem;border:2px solid var(--ko);background:#fff;
                  color:var(--ko);border-radius:8px;cursor:pointer}
#page form button.stop:hover{background:var(--ko);color:#fff}
#page form button.ghost{background:transparent;color:var(--acc);border:1px solid var(--acc)}
#page input,#page select{padding:6px 9px;border:1px solid #d5d4cf;border-radius:6px;font-size:14px;
                         max-width:100%;background:#fff;color:var(--ink)}
#page select[multiple]{min-height:120px}
form.vform{display:flex;flex-direction:column;gap:16px;margin-top:6px}
form.vform fieldset.vblock{border:1px solid var(--line);border-radius:10px;padding:12px 16px 16px;margin:0;
                           display:flex;flex-direction:column;gap:14px;background:#fff}
form.vform legend{font-size:14px;font-weight:700;padding:0 8px;color:var(--acc)}
.vfield{display:flex;flex-direction:column;gap:3px;min-width:0}
.vfield .vlabel{font-size:13px;font-weight:600}
.vfield .vnote{font-size:12px;color:var(--sub)}
.vfield input,.vfield select{width:min(100%,420px)}
.vfield input[type=checkbox]{width:auto}
.vfield .vlabel.check{display:flex;flex-direction:row;align-items:center;gap:8px;font-size:14px;font-weight:400}
.vfield.fixed .vval{font-size:14px;font-weight:600;background:#f4f8ff;border:1px solid #cfe0f7;
                    border-radius:8px;padding:8px 12px;width:min(100%,520px)}
.vfield .lock{display:inline-block;background:#24405f;color:#fff;border-radius:20px;padding:1px 10px;
              font-size:11px;font-weight:600;vertical-align:middle}
.submitrow{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.finrow{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0}
.finrow .vfield{flex:1 1 200px;min-width:170px;background:#fbfcfe;border:1px solid var(--line);
                border-radius:8px;padding:10px 14px}
.finrow .vval{font-size:17px;font-weight:600}
#page iframe{width:100%;border:1px solid #d9e2ec;border-radius:8px;background:#fff}
@media (max-width:640px){
  #page{padding:18px 12px 40px}
  #page .card{padding:12px}
  #page table{display:block;overflow-x:auto;white-space:nowrap}
}
"""

_PLOTLY_JS: list = []   # le JavaScript de plotly (~4,8 Mo), lu une fois


def plotly_js() -> bytes:
    """Le JavaScript de plotly, servi une seule fois sur /static/plotly.min.js."""
    if not _PLOTLY_JS:
        from plotly.offline import get_plotlyjs  # noqa: PLC0415
        _PLOTLY_JS.append(get_plotlyjs().encode("utf-8"))
    return _PLOTLY_JS[0]


def page(title: str, body: str, auto_refresh: int = 0) -> str:
    """Page HTML complète (responsive) autour du corps ``body``."""
    refresh = f'<meta http-equiv="refresh" content="{auto_refresh}">' if auto_refresh else ""
    script = '<script src="/static/plotly.min.js"></script>' if "plotly-graph-div" in body else ""
    return f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{refresh}<title>{html.escape(title)}</title><style>{CSS}</style>{script}</head>
<body style="margin:0;background:#f6f6f4"><main id="page">{body}</main></body></html>"""


def _hms(seconds: int) -> str:
    return f"{seconds // 3600} h {seconds % 3600 // 60:02d}"


# -----------------------------------------------------------------------------
# Carte « économies » (journal de statistiques du Qwen local)
# -----------------------------------------------------------------------------
SAVINGS_FILE = data.CODE_ADRIS / "Qwen-Benchmark" / "stats" / "qwen_usage.jsonl"


def savings_card() -> str:
    """Carte « Économies vs Claude » lue dans le journal de statistiques ; vide s'il n'existe pas."""
    try:
        entries = []
        for line in SAVINGS_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not entries:
            return ""
        saved = sum(int(e.get("saved_in", 0)) + int(e.get("saved_out", 0)) for e in entries)
        minutes = sum(float(e.get("minutes", 0)) for e in entries)
        return (f'<div class="card"><h2 style="margin-top:0">Économies vs Claude</h2>'
                f"<p>{len(entries)} tâches confiées au Qwen local : environ <b>{saved / 1e6:.2f} M tokens</b> "
                f"Claude évités, {minutes:.0f} min de calcul Qwen.</p>"
                f'<p class="small">Source : <span class="mono">{html.escape(str(SAVINGS_FILE))}</span> '
                "(estimation basse, voir Qwen-Benchmark/REGLE_DELEGATION.md).</p></div>")
    except OSError:
        return ""


# -----------------------------------------------------------------------------
# Prévisualisations : distribution d'exploitation, zone de recherche
# -----------------------------------------------------------------------------

def profile_preview_page(name: str) -> str:
    """Prévisualisation d'une distribution d'exploitation (TWA × TWS, taille = poids)."""
    path = data.profile_path(name)
    if path is None:
        return _failed_run_page(name, "profil d'exploitation introuvable dans vpp_acceleration/profiles")
    try:
        sys.path.insert(0, str(data.VPP_DIR))
        from src.operating_profile import OperatingProfile  # noqa: PLC0415
        from plotly import express as px  # noqa: PLC0415
        prof = OperatingProfile.from_json(str(path))
        nodes = prof.nodes()
    except Exception as e:  # module absent ou JSON illisible : message clair, pas de crash
        return page(f"Distribution {name}",
                    f"<p class='warn'>Impossible de lire <span class='mono'>{html.escape(name)}</span> : "
                    f"{html.escape(str(e))}</p><p><a href='/'>← retour</a></p>")
    fig = px.scatter(nodes, x="twa", y="tws", size=[n["weight"] for n in nodes], color="mode",
                     title=f"Distribution d'exploitation — {prof.name}",
                     labels={"twa": "angle de vent vrai TWA [°]", "tws": "vitesse de vent vrai TWS [nd]",
                             "size": "poids", "color": "bord"},
                     color_discrete_sequence=["#2a78d6", "#eb6834", "#1a9850", "#898781"])
    fig.update_layout(height=480, margin={"t": 60, "b": 46, "l": 50, "r": 20}, font=figures.FONT)
    others = [p for p in data.profiles_list() if p["name"] != name]
    opts = "\n".join(f'<option value="/profile/{html.escape(p["name"])}">{html.escape(p["name"])}</option>'
                     for p in others)
    body = f"""
<h1>Distribution « {html.escape(name)} »</h1>
<p class="sub">{len(nodes)} condition(s) d'exploitation — taille des pastilles = poids relatif. La recherche
pondère ses critères sur ces conditions (<span class="mono">--search-profile</span>).</p>
{figures._html(fig)}
<p class="sub">Comparer avec une autre distribution de <span class="mono">vpp_acceleration/profiles</span> :
<select onchange="if (this.value) location = this.value">
<option value="">— choisir un profil —</option>
{opts}
</select></p>
<p><a href="/">← retour à l'accueil</a></p>"""
    return page(f"Distribution {name}", body)


def bounds_preview_page(box: str, scale: str) -> str:
    """La zone de recherche réelle : la famille du run (boîte × échelle, Ixx, confiance NeuralFoil)."""
    try:
        scale_f = round(max(0.1, min(5.0, float(scale))), 2)
    except ValueError:
        scale_f = 1.0
    box = "dimred" if box == "dimred" else "current"
    try:
        fam = data.build_family(box, scale_f)
        s = data.bounds_summary(box, scale_f)
        graphs = figures.bounds_html(box, scale_f)
    except Exception as e:  # famille illisible (notebook, étude) : message clair
        return page(f"Zone de recherche ({box})",
                    f"<p class='warn'>Prévisualisation impossible : {html.escape(str(e))}</p>"
                    "<p><a href='/'>← retour</a></p>")
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{100 * v:.1f} %</td></tr>" for k, v in s["rates"].items())
    if box == "current":
        what = (f"Largeurs du notebook Dim-Reduction × <b>{scale_f:g}</b>, centrées sur la section C-foil "
                "actuelle. C'est la boîte que le site lance toujours : c'est la seule qui impose "
                "<b>Ixx ≥ Ixx(actuelle)</b>.")
    else:
        what = ("Boîte du notebook Dim-Reduction, <b>sans contrainte d'Ixx</b> : toutes ses sections sont "
                f"plus souples que la section actuelle (Ixx/Ixx(actuelle) de {s['ixx_ratio'][0]:.2f} à "
                f"{s['ixx_ratio'][1]:.2f}). Le site ne lance pas cette boîte ; elle n'est montrée que pour comparer.")
    body = f"""
<h1>Zone dans laquelle le run peut chercher</h1>
<p class="sub">{what}</p>
<div class="card">
<h3>Quelle part de la boîte est admissible ?</h3>
<table><tr><th>contrainte</th><th>part des tirages qui la passent</th></tr>{rows}</table>
<p class="small">Tirages uniformes dans la boîte (les 16 poids de Kulfan). La dernière ligne cumule les contraintes ;
les autres sont comptées parmi les sections sans croisement. Le run ne cherche que parmi les sections admissibles.</p>
<p>Sections admissibles : épaisseur maximale de {s['thickness'][0]:.1f} à {s['thickness'][2]:.1f} % de la corde
(médiane {s['thickness'][1]:.1f} %), Ixx/Ixx(actuelle) de {s['ixx_ratio'][0]:.2f} à {s['ixx_ratio'][1]:.2f}
(Ixx actuelle = {fam.ixx_current:.4e}, corde 1). Épaisseur de bord de fuite fixée à {data.TE_THICKNESS} c.</p>
</div>
{graphs}
<p><a href="/">← retour à l'accueil</a> ·
<a href="/bounds/current/1.0">boîte actuelle ×1</a> · <a href="/bounds/current/2.0">×2</a> ·
<a href="/bounds/current/3.0">×3</a> · <a href="/bounds/dimred/1.0">dimred (sans Ixx)</a></p>"""
    return page(f"Zone de recherche ({box} × {scale_f:g})", body)


def _failed_run_page(name: str, detail: str = "") -> str:
    """Page d'erreur claire pour un run ou un profil introuvable."""
    extra = f'<p class="small">Détail : {html.escape(detail)}</p>' if detail else ""
    body = (f"<h1>Indisponible</h1><p>« <span class=\"mono\">{html.escape(name)}</span> » est introuvable.</p>"
            f"{extra}<p><a href=\"/\">← Retour à l'accueil</a></p>")
    return page(f"{name} (indisponible)", body)


# -----------------------------------------------------------------------------
# Accueil et formulaire de lancement
# -----------------------------------------------------------------------------

def _field(key: str, label: str, note: str, value: str = "", kind: str = "text") -> str:
    """Un champ du formulaire : libellé au-dessus, aide en dessous."""
    return (f'<div class="vfield"><label class="vlabel" for="f_{key}">{label}</label>'
            f'<input id="f_{key}" name="{key}" value="{html.escape(value)}" inputmode="{kind}">'
            f'<span class="vnote">{note}</span></div>')


def launch_form(start: str = "") -> str:
    """Formulaire de lancement : contraintes fixées, réglages (préréglages + avancé), distribution."""
    sections = data.best_sections()
    start_opts = '<option value="">— section actuelle du C-foil (défaut) —</option>' + "\n".join(
        f'<option value="{html.escape(n["name"])}"{" selected" if n["name"] == start else ""}>'
        f'{html.escape(n["name"])} ({n["speed"]:.3f} nd)</option>' for n in sections)
    compare_opts = "\n".join(
        f'<option value="{html.escape(n["name"])}">{html.escape(n["name"])} ({n["speed"]:.3f} nd)</option>'
        for n in sections)
    profiles = data.profiles_list()
    prof_opts = '<option value="none">— Uniforme : 5 conditions équipondérées —</option>' + "\n".join(
        f'<option value="{html.escape(p["name"])}"{" selected" if p["name"] == launcher.DEFAULT_PROFILE else ""}>'
        f'{html.escape(p["name"])} ({p["n_nodes"]} conditions)</option>' for p in profiles)
    preset_opts = "\n".join(
        f'<option value="{k}"{" selected" if k == launcher.DEFAULT_PRESET else ""}>{html.escape(v["label"])}</option>'
        for k, v in launcher.PRESETS.items())
    ixx = data.ixx_current()
    ixx_txt = f" (Ixx actuelle = {ixx:.4e}, corde 1)" if ixx else ""
    constraints = [
        ("Ixx normalisé ≥ 1", f"Ixx / Ixx(section actuelle) ≥ 1{ixx_txt}",
         "le run écarte toute section plus souple que l'actuelle (vérifié sur les sections décodées). "
         "Cette contrainte n'existe qu'avec la boîte « current », que le site lance toujours"),
        ("Boîte de recherche", "--box current",
         "largeurs du notebook Dim-Reduction × l'échelle ci-dessous, centrées sur la section actuelle"),
        ("Géométrie", "surfaces sans croisement · confiance NeuralFoil · bord de fuite "
                      f"{data.TE_THICKNESS} c fixe", "16 poids de Kulfan libres dans la boîte, tout le reste fixé"),
    ]
    rows = "\n".join(
        f'<div class="vfield fixed"><div class="vlabel">{html.escape(n)} <span class="lock">verrouillé</span></div>'
        f'<div class="vval">{html.escape(v)}</div><span class="vnote">{d}</span></div>' for n, v, d in constraints)
    presets_js = json.dumps({k: v["values"] for k, v in launcher.PRESETS.items()})
    return f"""
<h2>Nouvelle optimisation de section</h2>
<form method="post" action="/" class="vform">
  <fieldset class="vblock"><legend>Contraintes fixées (appliquées par le script, non modifiables ici)</legend>
    {rows}
  </fieldset>

  <fieldset class="vblock"><legend>Réglages</legend>
    <div class="vfield"><label class="vlabel" for="f_preset">Réglages de départ</label>
      <select id="f_preset" onchange="applyPreset(this.value)">{preset_opts}</select>
      <span class="vnote">remplit les champs ci-dessous ; tous restent modifiables</span></div>
    {_field("name", "Nom du run", "dossier créé dans runs/ : lettres, chiffres et _", "mon_run")}
    {_field("time_limit_min", "Durée de la boucle (minutes)", "le run s'arrête à cette durée ; ajoutez ~10 à 20 min de validation", "140", "numeric")}
    {_field("box_scale", "Échelle de la boîte (box-scale)", "largeur des bornes × échelle. 1 = bornes du notebook, 3 = « ne pas restreindre la section » (décision du 29/10)", "3", "decimal")}
    <div class="vfield"><button type="button" class="ghost" onclick="showBounds()">Voir la zone de recherche pour cette échelle</button>
      <iframe id="boundsbox" style="display:none;height:1500px"></iframe></div>

    <div class="vfield"><label class="vlabel" for="f_profile">Distribution d'exploitation</label>
      <select id="f_profile" name="profile" onchange="showProfile()">{prof_opts}</select>
      <span class="vnote">les conditions TWA/TWS sur lesquelles la vitesse est optimisée, avec leurs poids
        (<span class="mono">--search-profile</span>, et analysée à la fin avec <span class="mono">--profile</span>)</span>
      <button type="button" class="ghost" style="align-self:flex-start" onclick="showProfile(true)">Voir la distribution</button>
      <iframe id="profilebox" style="display:none;height:560px"></iframe></div>

    <div class="vfield"><label class="vlabel" for="f_start_section">Section de départ</label>
      <select id="f_start_section" name="start_section">{start_opts}</select>
      <span class="vnote">la recherche part de cette section (<span class="mono">--start-section</span>). Si elle est hors de la
        boîte ou trop souple, le run prend la section admissible la plus proche et l'écrit dans run.log</span></div>

    <div class="vfield"><label class="vlabel" for="f_compare">Comparer à la fin avec (optionnel)</label>
      <select id="f_compare" name="compare" multiple>{compare_opts}</select>
      <span class="vnote">meilleures sections de runs précédents, validées avec la nouvelle (<span class="mono">--compare</span>) ;
        Ctrl+clic pour en choisir plusieurs</span></div>

    <div class="vfield"><label class="vlabel check"><input type="checkbox" name="fsi" checked>
      Foil flexible (FSI) : correction de déformation, mise à l'échelle par Ixx(actuelle)/Ixx (<span class="mono">--fsi</span>)</label></div>

    <details><summary class="vlabel">Réglages avancés</summary>
      <div style="display:flex;flex-direction:column;gap:14px;margin-top:12px">
      {_field("k", "Modes PCA (k)", "dimension de l'espace de recherche de la section (10 = 99,9 % de la variance de forme)", "10", "numeric")}
      {_field("n_low", "Plan de départ, niveau 1 (rapide)", "calculs NPLLT × NeuralFoil avant la boucle", "256", "numeric")}
      {_field("n_high", "Plan de départ, niveau 2 (précis)", "calculs AVL × XFOIL avant la boucle", "64", "numeric")}
      {_field("workers", "Processus de calcul des codes", "la machine est partagée : 16 sur 32 threads laisse de la place", "16", "numeric")}
      {_field("vpp_workers", "Processus de la VPP", "0 = dans le processus principal", "16", "numeric")}
      {_field("batch", "Choix par lot", "1 = séquentiel ; 16 = lots (réglage des runs 4-8)", "16", "numeric")}
      {_field("seed", "Graine aléatoire", "même graine, même plan de départ", "0", "numeric")}
      <div class="vfield"><label class="vlabel" for="f_mode">Niveau de référence</label>
        <select id="f_mode" name="mode">
          <option value="auto">AVL × XFOIL (défaut)</option>
          <option value="npllt">NPLLT × XFOIL (écarté le 28/10 : même précision, plus lent)</option>
        </select><span class="vnote"><span class="mono">--hf</span> ; les niveaux sont NPLLT × NeuralFoil puis ce code</span></div>
      {_field("local_fraction", "Part de recherche locale", "0 = aucune ; 0,5 = la moitié des candidats autour du meilleur (région de confiance)", "0.5", "decimal")}
      {_field("converge_window", "Convergence : fenêtre (mises à jour)", "vide = pas de test de convergence", "10", "numeric")}
      {_field("converge_tol", "Convergence : gain maximal sur la fenêtre (nd)", "la région a convergé quand ses dernières mises à jour gagnent moins", "0.01", "decimal")}
      {_field("restarts", "Redémarrages après convergence", "nouvelles régions de recherche ; exige la convergence ci-dessus", "2", "numeric")}
      {_field("min_gain", "Gain minimal d'un changement de meilleur (nd)", "en dessous, il compte comme « gardé » pour le rayon de la région", "0.002", "decimal")}
      {_field("sensitivity_median", "Sensibilités : médiane de N évaluations", "1 = la dernière", "3", "numeric")}
      <div class="vfield"><label class="vlabel check"><input type="checkbox" name="smoke">
        Smoke test : mini-run de 4 min qui vérifie la chaîne (<span class="mono">--smoke</span>)</label></div>
      </div>
    </details>
  </fieldset>

  <div class="submitrow"><button type="submit">Lancer le run</button>
    <span class="small">le run tourne en tâche de fond ; vous pouvez fermer cette page, il continue.</span></div>
</form>
<script>
const PRESETS = {presets_js};
function applyPreset(id) {{
  for (const [k, v] of Object.entries(PRESETS[id] || {{}})) {{
    const el = document.getElementById('f_' + k); if (el) el.value = v;
  }}
}}
function showBounds() {{
  const s = parseFloat(document.getElementById('f_box_scale').value.replace(',', '.')) || 1;
  const f = document.getElementById('boundsbox'); f.style.display = 'block'; f.src = '/bounds/current/' + s;
}}
function showProfile(force) {{
  const p = document.getElementById('f_profile').value, f = document.getElementById('profilebox');
  if (p === 'none') {{ f.style.display = 'none'; return; }}
  if (force || f.style.display === 'block') {{ f.style.display = 'block'; f.src = '/profile/' + encodeURIComponent(p); }}
}}
applyPreset(document.getElementById('f_preset').value);
</script>
"""


def index_page(start: str = "") -> str:
    """Accueil : runs en cours, formulaire de lancement, runs terminés avec leur gain validé."""
    rows = []
    for name in data.list_runs()[:30]:
        try:
            r = data.load_run(name)
            g = data.validated_gain(r)
            speed = f"{r.mean_speed:.3f} nd" if r.mean_speed is not None else "—"
            gain = f"{g['gain_kt']:+.3f} nd ({g['gain_pct']:+.1f} %)" if g else "—"
            rows.append(f"<tr><td><a href='/run/{html.escape(name)}'>{html.escape(name)}</a></td>"
                        f"<td>{' → '.join(r.levels)}</td><td>{speed}</td><td>{gain}</td></tr>")
        except (OSError, ValueError, KeyError):
            rows.append(f"<tr><td>{html.escape(name)}</td><td colspan=3 class='small'>illisible</td></tr>")
    live = [a.to_dict() for a in list(launcher.ACTIVE.values())]
    live.sort(key=lambda a: a["key"], reverse=True)
    live_rows = "".join(
        f"<tr><td><a href='/live/{html.escape(a['key'])}'>{html.escape(a['name'])}</a></td>"
        f"<td class='{'ok' if a['status'] == 'terminé' else 'wait' if a['status'] == 'en cours' else 'ko'}'>"
        f"{html.escape(a['status'])}</td><td>{_hms(a['elapsed_s'])}</td></tr>" for a in live[:10])
    busy = launcher.running_processes()
    banner = ""
    if busy:
        names = ", ".join(f"<span class='mono'>{html.escape(b['name'])}</span>" for b in busy)
        banner = (f'<div class="banner"><b>Un run tourne déjà sur la machine</b> : {names}. '
                  "Un second run ralentirait les deux (machine partagée) ; le site vous demandera confirmation.</div>")
    body = f"""
<h1>VPP acceleration — optimisation de section</h1>
<p class="sub">Co-krigeage séquentiel multi-fidélité : recherche de la section du C-foil qui maximise la vitesse
du bateau (VPP MDS) sur une distribution d'exploitation, sous la contrainte d'inertie de la section actuelle.</p>
{banner}
{('<div class="card"><h2 style="margin-top:0">Runs lancés depuis ce site</h2><table><tr><th>run</th><th>état</th><th>durée</th></tr>'
  + live_rows + "</table></div>") if live_rows else ""}
{launch_form(start)}
<div class="card"><h2 style="margin-top:0">Runs terminés</h2>
<table><tr><th>run</th><th>niveaux</th><th>vitesse validée</th><th>gain sur le départ</th></tr>
{''.join(rows)}</table>
<p class="small">Gain = vitesse validée de la meilleure section − celle de la section de départ du run, moyennes
pondérées par la distribution d'exploitation (même chiffre que <span class="mono">summary.txt</span>).
Les distributions n'étant pas les mêmes d'un run à l'autre, ces gains ne sont comparables qu'à distribution égale.</p>
</div>
{savings_card()}"""
    return page("VPP acceleration", body)


# -----------------------------------------------------------------------------
# Pages de run
# -----------------------------------------------------------------------------

def incumbent_table(run) -> str:
    """Tableau d'évolution du meilleur candidat (itération, temps, vitesse prédite)."""
    history = data.incumbent_series(run)
    if not history:
        return '<p class="small">Aucun historique d\'évolution disponible (le run vient de commencer ?).</p>'

    def _fmt(value, nd=1):
        return f"{value:.{nd}f}" if isinstance(value, (int, float)) and value == value else "—"

    rows = "".join(f"<tr><td>{h.get('iteration', '—')}</td><td>{_fmt(h.get('elapsed_min'))}</td>"
                   f"<td>{_fmt(h.get('speed'), 3)}</td></tr>" for h in history[-40:])
    return ("<details><summary class='small'>Tableau des mises à jour</summary>"
            "<table><tr><th>itération</th><th>temps [min]</th><th>vitesse prédite [nd]</th></tr>"
            + rows + "</table></details>")


def _settings_summary(run) -> str:
    """Les réglages du run en une ligne lisible."""
    c = run.config
    prof = c.get("search_profile")
    parts = [f"boîte {c.get('box', '?')} × {c.get('box_scale', '?')}", f"k = {run.k}",
             f"conditions : {Path_name(prof) if prof else 'uniforme (5)'}",
             f"lots de {c.get('batch', 1)}", f"{c.get('time_limit_min', '?')} min",
             "FSI" if c.get("fsi") else "sans FSI", f"graine {c.get('seed', '?')}"]
    if c.get("start_section"):
        parts.append("départ : " + Path_name(c["start_section"], 2))
    return " · ".join(parts)


def Path_name(path: str, up: int = 0) -> str:  # noqa: N802
    """Nom du fichier de ``path`` (ou du dossier ``up`` niveaux plus haut)."""
    from pathlib import PurePath  # noqa: PLC0415
    p = PurePath(path)
    return (p.parents[up - 1].name if up else p.stem) or str(path)


def run_page(name: str, query: dict | None = None) -> str:
    """Page d'un run terminé : gain validé, évolution, section, vitesses, figures."""
    try:
        run = data.load_run(name)
    except (OSError, ValueError) as e:
        return _failed_run_page(name, str(e))
    if not run.best:
        return run_progress_page(name)
    query = query or {}
    try:
        distance = max(1.0, float(query.get("dist", [data.REFERENCE_DISTANCE_NM])[0]))
    except ValueError:
        distance = data.REFERENCE_DISTANCE_NM
    try:
        eur = float(query["eur"][0]) if query.get("eur") and query["eur"][0] else None
    except ValueError:
        eur = None
    speeds = run.best.get("speeds_by_condition") or {}
    resolved = [v for v in speeds.values() if v is not None and v == v]
    n = html.escape(name)
    body = f"""
<h1>Run {n}</h1>
<p class="sub">{html.escape(_settings_summary(run))} · niveaux : {' → '.join(run.levels)} ·
{len(run.optima)} autre(s) optimum(s) local(aux)</p>
<p><a href="/?start_section={n}">Lancer un nouveau run qui part de la meilleure section de celui-ci →</a></p>
<div class="card"><h2 style="margin-top:0">Gain validé sur la section de départ</h2>
{figures.gain_html(run, distance, eur)}
<form method="get" class="small" style="margin-top:8px">Convertir en temps gagné sur
<input name="dist" value="{distance:g}" size="6" inputmode="decimal"> nm, valeur du temps
<input name="eur" value="{'' if eur is None else f'{eur:g}'}" size="6" inputmode="decimal" placeholder="—"> €/h
<button type="submit" class="ghost">Recalculer</button>
— estimation indicative (règle de trois à vitesse moyenne constante), à adapter à votre usage.</form>
</div>
<div class="card"><h2 style="margin-top:0">Évolution du meilleur candidat</h2>
{figures.incumbent_html(run)}
{incumbent_table(run)}
</div>
<div class="card"><h2 style="margin-top:0">Meilleure section</h2>
<p>Vitesse moyenne validée : <b>{(f"{run.mean_speed:.3f} nd" if run.mean_speed is not None else "—")}</b>
· {len(resolved)} conditions résolues sur {len(speeds)}</p>
{figures.section_html(run)}
</div>
{('<div class="card"><h2 style="margin-top:0">Vitesses par condition</h2>' + figures.speeds_html(run) + "</div>") if speeds else ""}
<div class="card"><h2 style="margin-top:0">Figures complètes du run</h2>
<ul>
<li><a href="/run/{n}/static/figures/index.html" target="_blank">index des figures</a></li>
<li><a href="/run/{n}/static/figures/sections.html" target="_blank">sections (meilleure + optima locaux)</a></li>
<li><a href="/run/{n}/static/figures/convergence.html" target="_blank">convergence</a></li>
<li><a href="/run/{n}/static/figures/speeds.html" target="_blank">vitesses par condition</a></li>
<li><a href="/run/{n}/static/figures/points.html" target="_blank">points de la boucle / attitudes</a></li>
<li><a href="/run/{n}/static/figures/validation.html" target="_blank">validation (surrogate vs codes)</a></li>
<li><a href="/run/{n}/static/results/sections/best_section.dat" download>Selig .dat de la meilleure section</a></li>
</ul>
<p class="small">Dossier du run : <span class="mono">runs/{n}/</span></p>
</div>
{('<div class="card"><h2 style="margin-top:0">Résumé (summary.txt)</h2><div class="log">'
  + html.escape(run.summary[:6000]) + "</div></div>") if run.summary else ""}
<p><a href="/">← accueil</a></p>"""
    return page(f"Run {name}", body)


def run_progress_page(name: str) -> str:
    """Page d'un run sans résultat : en cours, ou arrêté avant sa première section."""
    if not data.has_config(name):
        return _failed_run_page(name, "aucun dossier runs/<nom> avec config.json")
    run = data.load_run(name)
    live = next((a for a in launcher.ACTIVE.values() if a.folder() == name), None)
    link = f"<p><a href='/live/{html.escape(live.key)}'>Suivi en direct de ce run →</a></p>" if live else ""
    log = launcher.tail(run.root / "run.log", 5000)
    body = f"""
<h1>Run {html.escape(name)}</h1>
<p class="sub">{html.escape(_settings_summary(run))}</p>
<p class="warn">Pas encore de meilleure section : le run est en cours, ou il s'est arrêté avant d'en écrire une.</p>
{link}
<div class="card"><h2 style="margin-top:0">Évolution</h2>{figures.incumbent_html(run)}</div>
<div class="card"><h2 style="margin-top:0">run.log</h2><div class="log">{html.escape(log) or "(vide)"}</div></div>
<p><a href="/">← accueil</a></p>"""
    return page(f"Run {name}", body)


def launch_status_page(key: str) -> str:
    """Suivi en direct d'un run lancé depuis le site, avec le bouton d'arrêt."""
    st = launcher.live_status(key)
    if st is None:
        return page("Run", "<p class='warn'>Run inconnu : il n'a pas été lancé depuis ce site, ou son fichier "
                           "d'état (logs/*.json) a été supprimé.</p><p><a href='/'>← accueil</a></p>")
    icon = {"terminé": ("✓", "ok"), "terminé (échec)": ("✗", "ko"), "arrêté": ("■", "ko")}.get(
        st["status"], ("…", "wait"))
    stop_button = ""
    if st["status"] == "en cours":
        stop_button = (f'<form method="post" action="/live/{html.escape(st["key"])}/stop" style="margin:.5rem 0" '
                       "onsubmit=\"return confirm('Arrêter vraiment ce run ? Les calculs en cours seront perdus.')\">"
                       '<button class="stop" type="submit">⏹ Arrêter le run</button></form>')
    folder = st["folder"]
    logs = st["run_log"] + ("\n--- sortie de la console ---\n" if st["out_log"] and st["run_log"] else "") + st["out_log"]
    log_text = html.escape(logs[-6000:]) or "(aucune sortie pour l'instant)"
    refresh_note = "La page se rafraîchit toute seule toutes les 10 s." if st["status"] == "en cours" else ""
    open_link = ""
    if folder:
        label = "Ouvrir la page du run" if st["status"].startswith("terminé") and not st["error"] else "Page du run (en cours)"
        open_link = f'<div class="card"><p><a href="/run/{html.escape(folder)}">{label} →</a></p></div>'
    body = f"""
<h1>Run {html.escape(st['name'])}</h1>
<p class="sub"><span class="{icon[1]}">{icon[0]}</span> {html.escape(st['status'])} · {_hms(st['elapsed_s'])} ·
dossier : <span class="mono">{('runs/' + html.escape(folder)) if folder else 'pas encore créé'}</span></p>
{stop_button}
<p class="small mono">{html.escape(st['cmd'])}</p>
{('<p class="warn">' + html.escape(st['error']) + '</p>') if st['error'] else ''}
{open_link}
<div class="card"><h2 style="margin-top:0">Journal du run</h2>
<div class="log">{log_text}</div>
<p class="small">{refresh_note}
Le run écrit aussi dans <span class="mono">run.log</span> et <span class="mono">results/history.csv</span> de son dossier.</p></div>
<p><a href="/">← accueil</a></p>"""
    return page(f"Run {st['name']}", body, auto_refresh=10 if st["status"] == "en cours" else 0)


def busy_page(opts: dict, busy: list) -> str:
    """Confirmation avant de lancer un second run sur une machine déjà occupée."""
    names = ", ".join(f"<span class='mono'>{html.escape(b['name'])}</span>" for b in busy)
    hidden = "".join(f'<input type="hidden" name="{html.escape(k)}" value="{html.escape(v)}">'
                     for k, v in opts.items() if isinstance(v, str)) + "".join(
        f'<input type="hidden" name="compare" value="{html.escape(c)}">' for c in opts.get("compare", []))
    body = f"""
<h1>Un run tourne déjà</h1>
<p class="warn">Sur cette machine partagée tourne déjà : {names}. Lancer « {html.escape(opts.get('name', ''))} » en
même temps ralentirait les deux, et fausserait la durée limite de chacun.</p>
<form method="post" action="/" style="display:flex;gap:12px;align-items:center">{hidden}
<input type="hidden" name="force" value="on">
<button type="submit">Lancer quand même</button> <a href="/">Annuler</a></form>"""
    return page("Un run tourne déjà", body)


# -----------------------------------------------------------------------------
# Serveur HTTP
# -----------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "VppSite/2.0"

    def _send(self, status: int, content: bytes, ctype: str, extra: dict | None = None):
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(content)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(content)
        except ConnectionError:   # navigateur ou iframe fermé en cours de route
            pass

    def _page(self, html_text: str, status: int = 200):
        self._send(status, html_text.encode("utf-8"), "text/html; charset=utf-8")

    def _redirect(self, location: str):
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

    def do_GET(self):  # noqa: N802
        url = urlsplit(self.path)
        p, query = url.path, parse_qs(url.query)
        if p == "/static/plotly.min.js":
            try:
                self._send(200, plotly_js(), "application/javascript; charset=utf-8",
                           {"Cache-Control": "public, max-age=86400"})
            except ImportError:
                self._page("<p>plotly n'est pas installé pour cet interpréteur</p>", 500)
            return
        m = re.fullmatch(r"/run/([^/]+)", p)
        if m:
            name = m.group(1)
            if data.has_config(name):
                self._page(run_page(name, query))
            else:
                self._page(_failed_run_page(name, "aucun dossier runs/<nom> avec config.json"), 404)
            return
        m = re.fullmatch(r"/run/([^/]+)/static/(.+)", p)
        if m:
            self._serve_file(m.group(1), m.group(2))
            return
        m = re.fullmatch(r"/live/([^/]+)", p)
        if m:
            self._page(launch_status_page(m.group(1)))
            return
        m = re.fullmatch(r"/profile/([^/]+)", p)
        if m:
            self._page(profile_preview_page(m.group(1)))
            return
        m = re.fullmatch(r"/bounds/([^/]+)/([^/]+)", p)
        if m:
            self._page(bounds_preview_page(m.group(1), m.group(2)))
            return
        if p in ("/", ""):
            self._page(index_page(query.get("start_section", [""])[0]))
            return
        self._page("<p>404</p>", 404)

    def _serve_file(self, name: str, rel: str):
        """Envoie un fichier du dossier du run (jamais hors de runs/<name>/)."""
        root = (data.RUNS_DIR / name).resolve()
        try:
            path = (root / rel).resolve()
        except (OSError, ValueError):
            self._page("<p>400</p>", 400)
            return
        if root.parent != data.RUNS_DIR.resolve() or not path.is_relative_to(root) or not path.is_file():
            self._page(f"<h1>{html.escape(rel)}</h1><p class='warn'>Fichier absent dans "
                       f"<span class='mono'>runs/{html.escape(name)}/</span> (le run a peut-être échoué avant de "
                       f"l'écrire).</p><p><a href='/run/{html.escape(name)}'>← page du run</a> · "
                       "<a href='/'>accueil</a></p>", 404)
            return
        ctype = ("text/html" if path.suffix in (".html", ".htm") else
                 "text/plain" if path.suffix in (".csv", ".txt", ".dat", ".json", ".log") else "application/octet-stream")
        self._send(200, path.read_bytes(), ctype + "; charset=utf-8")

    def do_POST(self):  # noqa: N802
        p = urlsplit(self.path).path
        m = re.fullmatch(r"/live/([^/]+)/stop", p)
        if m:
            if launcher.stop_run(m.group(1)):
                self._redirect(f"/live/{m.group(1)}")
            else:
                self._page("<p class='warn'>Ce run est inconnu ou déjà terminé.</p><p><a href='/'>← accueil</a></p>", 404)
            return
        if p not in ("/run", "/"):
            self._page("<p>404</p>", 404)
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        form = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
        opts = {k: v[0] for k, v in form.items() if k not in ("compare",)}
        opts["compare"] = form.get("compare", [])
        try:
            live = launcher.launch_run(opts, force=opts.get("force") == "on")
        except launcher.RunBusy as e:
            self._page(busy_page(opts, e.args[0]), 409)
            return
        except OSError as e:
            self._page(f"<p class='warn'>Échec du lancement : {html.escape(str(e))}</p>", 500)
            return
        self._redirect(f"/live/{live.key}")

    def log_message(self, fmt, *args):  # journal discret
        sys.stderr.write("[vpp-site] " + (fmt % args) + "\n")


def main() -> None:
    import argparse  # noqa: PLC0415
    parser = argparse.ArgumentParser(description="Site VPP acceleration")
    parser.add_argument("--port", type=int, default=8321)
    parser.add_argument("--python", default=None, help="interpréteur pour lancer les runs")
    args = parser.parse_args()
    if args.python:
        data.PYTHON_CANDIDATES.insert(0, Path(args.python))
    n = launcher.load_persisted()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Site VPP acceleration : http://127.0.0.1:{args.port}   ({n} run(s) retrouvé(s))")
    print(f"Runs : {data.RUNS_DIR}")
    print(f"Interpréteur des runs : {data.available_python()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du site (les runs en cours continuent).")


if __name__ == "__main__":
    main()
