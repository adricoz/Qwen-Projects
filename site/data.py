# -*- coding: utf-8 -*-
"""
Lecture des dossiers de runs de vpp_acceleration (Multifidelity-GP-Clean).

Le site ne fait que *lire* les runs existants (config.json, results/*, figures/*)
et *lancer* de nouvelles optimisations de section via ``scripts/run_real_vpp.py``.
Aucun résultat n'est modifié ; un run lancé par le site est un dossier comme les
autres, dans vpp_acceleration/runs/<date>_<nom>/.
"""
from __future__ import annotations

import csv
import functools
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

# Racine du projet : le site vit dans Code-Adri/Qwen-Projects/site.
_THIS = Path(__file__).resolve()
CODE_ADRIS = _THIS.parents[2]  # .../Code-Adri
MF_ROOT = CODE_ADRIS / "Multifidelity-GP-Clean"
VPP_DIR = MF_ROOT / "vpp_acceleration"
RUNS_DIR = VPP_DIR / "runs"
PROFILES_DIR = VPP_DIR / "profiles"

# Interpréteurs capables de lancer un run : le venv du projet (VPP MDS, Python 3.10) d'abord,
# le Python qui fait tourner le site en dernier recours.
PYTHON_CANDIDATES = [
    MF_ROOT / ".venv-mds" / "Scripts" / "python.exe",
    MF_ROOT / ".venv-mds" / "bin" / "python",
    Path(sys.executable),
]

CODE_NAMES = {
    "nf_llt": "NPLLT – NeuralFoil",
    "xfoil_llt": "NPLLT – XFOIL",
    "avl": "AVL – XFOIL",
}

# Modules dont dépend la chaîne de calcul du run.
RUN_DEPENDENCIES = ("sklearn", "numpy", "numba", "scipy")

# Épaisseur de bord de fuite des sections (en fraction de la corde), fixée par la famille Kulfan.
TE_THICKNESS = 0.00197


def _probe_python(python: str) -> bool:
    """Vrai si le binaire existe et importe les dépendances du run."""
    if not Path(python).exists():
        return False
    try:
        code = ";".join(f"import {n}" for n in RUN_DEPENDENCIES)
        r = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=120)
        return r.returncode == 0
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def available_python() -> str:
    """Interpréteur à utiliser pour lancer un run : le premier qui a toutes les dépendances.

    Le résultat est mémorisé (la vérification importe numba et prend quelques secondes).
    """
    ordered = [p for p in PYTHON_CANDIDATES if p.exists()]
    if not ordered:
        return sys.executable
    for p in ordered:
        if _probe_python(str(p)):
            return str(p)
    return str(ordered[0])


# -----------------------------------------------------------------------------
# Runs existants
# -----------------------------------------------------------------------------

def is_run_folder(name: str) -> bool:
    """Un run terminé a un config.json et des results/best_section.json (ou figures/)."""
    root = RUNS_DIR / name
    return (root / "config.json").is_file() and (
        (root / "results" / "best_section.json").is_file()
        or (root / "figures" / "index.html").is_file()
    )


def list_runs() -> list[str]:
    """Noms des runs terminés, du plus récent au plus ancien."""
    if not RUNS_DIR.is_dir():
        return []
    runs = [n for n in os.listdir(RUNS_DIR) if is_run_folder(n)]
    runs.sort(reverse=True)
    return runs


def has_config(name: str) -> bool:
    """Le dossier existe et a un config.json (run terminé, en cours ou échoué)."""
    return (RUNS_DIR / name / "config.json").is_file()


@dataclass
class Run:
    """Vue d'un run : son nom, sa config, son résumé et le détail de la meilleure section."""

    name: str
    config: dict
    best: dict
    optima: list[dict] = field(default_factory=list)
    summary: str = ""
    has_figures: bool = False

    @property
    def root(self) -> Path:
        return RUNS_DIR / self.name

    @property
    def k(self) -> int:
        return int(self.config.get("k", 10))

    @property
    def mean_speed(self) -> float | None:
        return self.best.get("mean_speed")

    @property
    def levels(self) -> list[str]:
        levels = self.config.get("levels")
        if not isinstance(levels, list):
            levels = ["nf_llt", self.config.get("hf", "avl")]
        return levels


def load_run(name: str) -> Run:
    """Charge config.json, best_section.json, optima.json et summary.txt d'un run."""
    root = RUNS_DIR / name
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    best_path = root / "results" / "best_section.json"
    best = json.loads(best_path.read_text(encoding="utf-8")) if best_path.is_file() else {}
    optima: list[dict] = []
    optima_path = root / "results" / "optima.json"
    if optima_path.is_file():
        optima = [o for o in json.loads(optima_path.read_text(encoding="utf-8")) if not o.get("best")]
    summary_path = root / "results" / "summary.txt"
    summary = summary_path.read_text(encoding="utf-8", errors="replace") if summary_path.is_file() else ""
    has_figures = (root / "figures" / "index.html").is_file() or (root / "figures" / "sections.html").is_file()
    return Run(name, config, best, optima, summary, has_figures)


def best_sections() -> list[dict]:
    """Meilleures sections des runs terminés (du plus récent au plus ancien).

    Chaque entrée : ``{"name", "speed", "k"}``. Sert aux listes « section de départ » et
    « comparer à la fin avec ».
    """
    out: list[dict] = []
    for name in list_runs():
        try:
            run = load_run(name)
        except Exception:
            continue
        speed = run.mean_speed
        if speed is None or not (RUNS_DIR / name / "results" / "best_section.json").is_file():
            continue
        out.append({"name": name, "speed": float(speed), "k": run.k})
    return out


# -----------------------------------------------------------------------------
# Distributions d'exploitation
# -----------------------------------------------------------------------------

def profiles_list() -> list[dict]:
    """Profils d'exploitation existants dans vpp_acceleration/profiles.

    Chaque entrée : ``{"name", "path", "n_modes", "n_nodes"}``. Ordre alphabétique inverse :
    ``profile_v2`` (le profil retenu le 29/10) avant ``profile_v1``, puis les hypothèses ``guess_*``.
    """
    out: list[dict] = []
    if not PROFILES_DIR.is_dir():
        return out
    for path in sorted(PROFILES_DIR.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            modes = raw.get("modes")
            nodes = raw.get("nodes")
            out.append({
                "name": path.stem,
                "path": str(path),
                "n_modes": len(modes) if modes else (1 if nodes else 0),
                "n_nodes": len(nodes) if nodes else (
                    int(raw.get("nodes_per_dim", 3)) ** 2 * len(modes) if modes else 0),
            })
        except Exception:
            continue
    out.sort(key=lambda d: d["name"], reverse=True)   # profile_v2, profile_v1, guess_*
    return out


def profile_path(name: str) -> Path | None:
    """Chemin du profil d'exploitation ``name`` (JSON dans profiles/), ou None."""
    for d in profiles_list():
        if d["name"] == name:
            return Path(d["path"])
    return None


# -----------------------------------------------------------------------------
# Suivi d'un run : évolution du meilleur candidat, gain validé
# -----------------------------------------------------------------------------

def _to_float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def incumbent_series(run: Run) -> list[dict]:
    """Évolution du meilleur score (incumbent) au cours du run, pour le tableau de bord.

    Lecture de ``results/updates.json`` (une entrée par mise à jour du modèle : vitesse
    PRÉDITE par le surrogate pour le meilleur candidat). Pendant un run en cours,
    ``updates.json`` n'existe pas encore : on retombe sur ``data/evaluations.csv`` pour
    tracer l'activité (nombre de calculs réussis). Liste vide si rien n'est lisible.
    """
    path = run.root / "results" / "updates.json"
    if path.is_file():
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
            return [
                {
                    "iteration": e.get("iteration"),
                    "elapsed_min": e.get("elapsed_min"),
                    "speed": e.get("incumbent_speed"),
                    "n_lf": e.get("n_lf"),
                    "n_hf": e.get("n_hf"),
                }
                for e in entries
            ]
        except Exception:
            pass
    csv_path = run.root / "data" / "evaluations.csv"
    if csv_path.is_file():
        try:
            with csv_path.open(encoding="utf-8", errors="replace") as f:
                rows = list(csv.DictReader(f))
            series: list[dict] = []
            n_ok = 0
            for row in rows:
                if str(row.get("ok", row.get("success", ""))).lower() != "true":
                    continue
                n_ok += 1
                series.append({"iteration": n_ok, "elapsed_min": _to_float(row.get("elapsed_min")),
                               "speed": None, "cost": float(n_ok), "n_lf": None, "n_hf": None})
            return series
        except Exception:
            pass
    return []


def validated_speeds(run: Run) -> dict:
    """Vitesse moyenne pondérée de chaque section validée du run (``results/validation.csv``).

    C'est le chiffre que le run lui-même annonce dans ``summary.txt`` : vitesse du surrogate
    corrigée de son écart au code de référence, moyenne pondérée par ``condition_weights`` (la
    distribution d'exploitation), sur les conditions résolues pour TOUTES les sections validées
    (sinon les sections ne seraient pas comparables).

    Returns
    -------
    dict
        ``{"speeds": {section: kt}, "n_conditions": int, "probability": float}`` ; dict vide
        si ``validation.csv`` est absent ou incohérent avec la config.
    """
    path = run.root / "results" / "validation.csv"
    weights = run.config.get("condition_weights")
    if not path.is_file() or not weights:
        return {}
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["section"], []).append(row)
    n = len(weights)
    if not groups or any(len(g) != n for g in groups.values()):
        return {}
    solved = [all(str(g[i]["success"]).lower() == "true" and _to_float(g[i]["speed_kt"]) is not None
                  for g in groups.values()) for i in range(n)]
    total = sum(w for w, ok in zip(weights, solved) if ok)
    if total <= 0:
        return {}
    def corrected(row: dict) -> float:
        # même définition que scripts/run_real_vpp.py (résumé du run) : vitesse du surrogate
        # corrigée au premier ordre de son écart au code de référence
        return float(row["speed_kt"]) - (_to_float(row.get("speed_error_kt")) or 0.0)

    speeds = {name: sum(w * corrected(g[i]) for i, (w, ok) in enumerate(zip(weights, solved)) if ok) / total
              for name, g in groups.items()}
    return {"speeds": speeds, "n_conditions": sum(solved),
            "probability": total / sum(weights)}


def validated_gain(run: Run) -> dict:
    """Gain validé de la meilleure section sur la section de départ du run.

    Returns
    -------
    dict
        ``{"start_name", "start_speed", "best_speed", "gain_kt", "gain_pct", "n_conditions",
        "probability"}`` ; dict vide si la validation n'est pas lisible.
    """
    v = validated_speeds(run)
    start = run.best.get("start")
    if not v or "best" not in v["speeds"] or start not in v["speeds"]:
        return {}
    v0, v1 = v["speeds"][start], v["speeds"]["best"]
    return {"start_name": start, "start_speed": v0, "best_speed": v1, "gain_kt": v1 - v0,
            "gain_pct": (v1 / v0 - 1.0) * 100.0, "n_conditions": v["n_conditions"],
            "probability": v["probability"]}


# Hypothèse de route pour convertir un gain de vitesse en temps gagné (modifiable sur la page).
REFERENCE_DISTANCE_NM = 1000.0


def time_saved_h(v_ref: float, v_best: float, distance_nm: float = REFERENCE_DISTANCE_NM) -> float:
    """Heures gagnées sur ``distance_nm`` milles en passant de ``v_ref`` à ``v_best`` [nœuds]."""
    if not (v_ref and v_best) or v_ref <= 0 or v_best <= 0:
        return 0.0
    return distance_nm / v_ref - distance_nm / v_best


def section_figure_data(run: Run) -> dict | None:
    """Profils (meilleure section, section de départ) et scores, pour la figure de prévisualisation.

    Retourne ``None`` si la PCA du run n'est pas lisible.
    """
    try:
        sys.path.insert(0, str(VPP_DIR))
        from src.section_pca import SectionPCA, section_coordinates  # noqa: PLC0415
    except Exception:
        return None
    pca_path = run.root / "data" / "pca.json"
    if not pca_path.is_file():
        return None
    try:
        pca = SectionPCA.load(pca_path)
    except Exception:
        return None
    import numpy as np  # noqa: PLC0415
    x = pca.stations
    best = run.best
    upper = best.get("kulfan_upper", [])
    lower = best.get("kulfan_lower", [])
    if not upper or not lower or len(upper) != len(lower):
        return None
    # section_coordinates renvoie (m, 2p) : surface supérieure, puis surface inférieure.
    best_pts = section_coordinates(np.array(upper + lower), x, pca.te_thickness)[0]
    best_y = list(best_pts[: len(x)])
    best_low = list(best_pts[len(x):])
    start = start_low = None
    if best.get("start_scores") is not None:
        try:
            w = pca.decode(np.array(best["start_scores"])[None])[0]
            pts = section_coordinates(w, x, pca.te_thickness)[0]
            start, start_low = list(pts[: len(x)]), list(pts[len(x):])
        except Exception:
            start = None
    return {
        "stations": list(np.asarray(x).astype(float)),
        "best": best_y,
        "best_lower": best_low,
        "start": start,
        "start_lower": start_low,
        "scores": list(np.asarray(best.get("scores", [])).astype(float)),
        "start_scores": (list(np.asarray(best["start_scores"]).astype(float))
                         if best.get("start_scores") is not None else None),
        "optima": [
            {"region": o.get("region"), "scores": list(np.asarray(o.get("scores", [])).astype(float))}
            for o in run.optima
        ],
    }


# -----------------------------------------------------------------------------
# Zone de recherche : la même famille que celle du run
# -----------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def build_family(box: str, scale: float):
    """Famille de sections construite EXACTEMENT comme ``scripts/run_real_vpp.py``.

    - ``box="current"`` : largeurs du notebook Dim-Reduction × ``scale``, centrées sur la section
      C-foil actuelle ; contraintes : pas de croisement des surfaces, Ixx >= Ixx(actuelle),
      confiance NeuralFoil (``src.meta_foil.confident``).
    - ``box="dimred"`` : boîte du notebook, sans contrainte d'Ixx (comme le premier run).

    Returns
    -------
    SimpleNamespace
        ``family, current (16,), ixx_current, lower, upper, checks`` où ``checks`` est un dict
        ``{nom: fonction(weights) -> bool array}`` pour mesurer l'effet de chaque contrainte.
    """
    sys.path[:0] = [p for p in (str(VPP_DIR), str(MF_ROOT / "mfego")) if p not in sys.path]
    from src.meta_foil import confident  # noqa: PLC0415
    from src.section_pca import (CFoilFamily, KulfanBoxFamily, notebook_bounds,  # noqa: PLC0415
                                 section_ixx)
    current = CFoilFamily.from_study_config(
        MF_ROOT / "studies" / "cfoil_intens_kulfan_trim_2" / "config_cfoil_trim.json"
    ).reference_weights()
    ixx_current = float(section_ixx(current, TE_THICKNESS)[0])
    lower, upper = notebook_bounds()
    if box == "dimred":
        family = KulfanBoxFamily.from_notebook(te_thickness=TE_THICKNESS, extra_check=confident)
        checks = {"confiance NeuralFoil": confident}
    else:
        def stiff(weights):
            return section_ixx(weights, TE_THICKNESS) >= ixx_current

        def stiff_and_confident(weights):
            return stiff(weights) & confident(weights)
        family = KulfanBoxFamily.centered(current, scale * (upper - lower), TE_THICKNESS,
                                          extra_check=stiff_and_confident)
        checks = {"Ixx ≥ Ixx(actuelle)": stiff, "confiance NeuralFoil": confident}
    return SimpleNamespace(family=family, current=current, ixx_current=ixx_current,
                           lower=family.lower, upper=family.upper, checks=checks)


@functools.lru_cache(maxsize=8)
def bounds_summary(box: str, scale: float, n_samples: int = 400, n_draws: int = 1500) -> dict:
    """Ce que le run peut explorer : sections admissibles, effet de chaque contrainte, enveloppe.

    Returns
    -------
    dict
        ``rates`` (part des tirages uniformes dans la boîte qui passent chaque contrainte),
        ``x`` (stations), ``upper_min/max``, ``lower_min/max`` (enveloppe des sections admissibles),
        ``current_upper/lower``, ``w_lo/w_hi`` (bornes de la boîte), ``w_min/w_max`` (étendue des
        sections admissibles), ``w_current``, ``scatter`` (projection 2D), ``thickness``,
        ``ixx_ratio``.
    """
    import numpy as np  # noqa: PLC0415
    from scipy.stats import qmc  # noqa: PLC0415
    sys.path[:0] = [p for p in (str(VPP_DIR),) if p not in sys.path]
    from src.section_pca import (section_coordinates, section_ixx,  # noqa: PLC0415
                                 thickness_distribution, THICKNESS_STATIONS)
    fam = build_family(box, scale)
    family = fam.family
    # part des tirages uniformes de la boîte qui passent chaque contrainte
    unit = qmc.LatinHypercube(d=len(fam.lower), seed=3).random(n_draws)
    draws = fam.lower + unit * (fam.upper - fam.lower)
    no_cross = family.thickness_ok(draws)
    rates = {"surfaces sans croisement": float(no_cross.mean())}
    for name, check in fam.checks.items():
        ok = np.zeros(len(draws), dtype=bool)
        ok[no_cross] = check(draws[no_cross])        # parmi les sections sans croisement
        rates[name] = float(ok.sum() / max(no_cross.sum(), 1))
    rates["toutes les contraintes"] = float(family.contains(draws).mean())
    samples = family.sample(n_samples, seed=1)
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 121)))
    coords = section_coordinates(samples, x, TE_THICKNESS)
    cur = section_coordinates(fam.current, x, TE_THICKNESS)[0]
    p = len(x)
    # projection 2D sur les deux premiers modes de forme des sections admissibles
    centred = coords - coords.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    scatter = (centred @ vt[:2].T, (cur - coords.mean(axis=0)) @ vt[:2].T)
    thick = thickness_distribution(samples, THICKNESS_STATIONS, TE_THICKNESS).max(axis=1) * 100.0
    return {
        "rates": rates, "x": x,
        "upper_min": coords[:, :p].min(axis=0), "upper_max": coords[:, :p].max(axis=0),
        "lower_min": coords[:, p:].min(axis=0), "lower_max": coords[:, p:].max(axis=0),
        "current_upper": cur[:p], "current_lower": cur[p:],
        "w_lo": fam.lower, "w_hi": fam.upper, "w_min": samples.min(axis=0), "w_max": samples.max(axis=0),
        "w_current": fam.current, "scatter": scatter,
        "thickness": (float(thick.min()), float(np.median(thick)), float(thick.max())),
        "ixx_ratio": (float(section_ixx(samples, TE_THICKNESS).min() / fam.ixx_current),
                      float(section_ixx(samples, TE_THICKNESS).max() / fam.ixx_current)),
        "ixx_current": fam.ixx_current, "n_samples": int(n_samples),
    }


def ixx_current() -> float | None:
    """Ixx de la section C-foil actuelle (corde 1) ; ``None`` si le code du run est illisible."""
    try:
        return float(build_family("current", 1.0).ixx_current)
    except Exception:
        return None


def run_folder_pattern(name: str):
    """Motif des dossiers créés par ``RunDir`` pour ``--name <name>`` : AAAA-MM-JJ_HHMM_<name>."""
    return re.compile(r"^\d{4}-\d{2}-\d{2}_\d{4}_" + re.escape(name) + r"$")
