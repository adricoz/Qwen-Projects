# -*- coding: utf-8 -*-
"""Tests du site : commande de lancement, suivi des runs, gain validé, zone de recherche, pages, serveur.

À lancer avec l'interpréteur du projet (celui qui a numpy, scipy, plotly et pytest) :

    ..\\..\\Multifidelity-GP-Clean\\.venv-mds\\Scripts\\python.exe -m pytest -q tests
"""
import json
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as appmod  # noqa: E402
import data  # noqa: E402
import figures  # noqa: E402
import launcher  # noqa: E402

RUN8 = "2026-10-29_1002_profile_v2_box3_3h"      # run de référence : gain validé +0,256 nd (summary.txt)
needs_run8 = pytest.mark.skipif(not data.is_run_folder(RUN8), reason="run de référence absent")


# --- la commande envoyée à run_real_vpp.py ------------------------------------------------------

def script_flags() -> set:
    """Les options réellement acceptées par le script (lues dans son code source)."""
    text = (data.VPP_DIR / "scripts" / "run_real_vpp.py").read_text(encoding="utf-8")
    return set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', text))


FULL = {"name": "Mon Run!", "time_limit_min": "30", "box_scale": "1.5", "k": "10", "n_low": "100",
        "n_high": "40", "workers": "4", "vpp_workers": "4", "batch": "8", "seed": "7", "mode": "npllt",
        "local_fraction": "0.5", "min_gain": "0.002", "sensitivity_median": "3", "fsi": "on",
        "converge_window": "10", "converge_tol": "0.01", "restarts": "2", "profile": "profile_v2",
        "smoke": "on"}


def test_script_and_site_share_the_same_root():
    assert (data.VPP_DIR / "scripts" / "run_real_vpp.py").is_file()
    assert "--start-section" in script_flags(), "le script doit accepter --start-section"


def test_build_cmd_only_real_flags():
    """Chaque option émise existe dans le script : rien n'est inventé ni silencieusement ignoré."""
    cmd = launcher.build_cmd({**FULL, "start_section": data.list_runs()[0], "compare": data.list_runs()[:2]})
    emitted = {c for c in cmd if c.startswith("--")}
    assert emitted <= script_flags(), f"flags inconnus du script : {emitted - script_flags()}"
    for flag in ("--name", "--box", "--box-scale", "--time-limit-min", "--k", "--n-low", "--n-high", "--workers",
                 "--vpp-workers", "--batch", "--seed", "--hf", "--local-fraction", "--min-gain",
                 "--sensitivity-median", "--fsi", "--converge", "--restarts", "--search-profile", "--profile",
                 "--start-section", "--compare", "--smoke"):
        assert flag in cmd, f"flag manquant : {flag}"


def test_build_cmd_values():
    cmd = launcher.build_cmd(FULL)
    pairs = {cmd[i]: cmd[i + 1] for i, c in enumerate(cmd[:-1]) if c.startswith("--")}
    assert pairs["--name"] == "mon_run"                       # nom nettoyé
    assert pairs["--box"] == "current"
    assert pairs["--box-scale"] == "1.5" and pairs["--batch"] == "8" and pairs["--hf"] == "npllt_xfoil"
    assert cmd[cmd.index("--converge") + 1: cmd.index("--converge") + 3] == ["10", "0.01"]
    assert Path(cmd[cmd.index("--search-profile") + 1]).name == "profile_v2.json"


def test_box_is_always_current_so_ixx_constraint_applies():
    """Seule la boîte « current » impose Ixx >= Ixx(actuelle) : le site ne lance jamais « dimred »."""
    for opts in ({}, {"box": "dimred"}, FULL):
        cmd = launcher.build_cmd(opts)
        assert cmd[cmd.index("--box") + 1] == "current"


def test_build_cmd_clamps_and_cleans():
    cmd = launcher.build_cmd({"name": "../../x y", "workers": "9999", "time_limit_min": "abc", "k": "-3",
                              "compare": ["../evil", "nope"], "start_section": "../evil", "profile": "inexistant"})
    assert cmd[cmd.index("--name") + 1] == "x_y"
    assert cmd[cmd.index("--workers") + 1] == "32"            # borné
    assert "--time-limit-min" not in cmd                      # illisible : le défaut du script
    assert cmd[cmd.index("--k") + 1] == "2"
    for flag in ("--compare", "--start-section", "--search-profile", "--fsi", "--smoke"):
        assert flag not in cmd


def test_restarts_need_converge():
    """--restarts sans --converge ferait échouer le script : le site ne l'émet pas."""
    cmd = launcher.build_cmd({"restarts": "2"})
    assert "--restarts" not in cmd and "--converge" not in cmd
    cmd = launcher.build_cmd({"restarts": "2", "converge_window": "10", "converge_tol": "0.01"})
    assert "--restarts" in cmd and "--converge" in cmd


def test_presets_are_valid_and_match_form_fields():
    form = appmod.launch_form()
    for name, preset in launcher.PRESETS.items():
        for key, value in preset["values"].items():
            assert f'id="f_{key}"' in form, f"{name}: le champ {key} n'existe pas dans le formulaire"
            if key in launcher.NUMERIC and value != "":
                flag, kind, lo, hi = launcher.NUMERIC[key]
                assert lo <= float(value) <= hi, f"{name}: {key}={value} hors bornes"
        assert preset["values"].get("profile") in ("none", *(p["name"] for p in data.profiles_list()))
    assert launcher.DEFAULT_PRESET in launcher.PRESETS and data.profile_path(launcher.DEFAULT_PROFILE)


def test_interpreter_has_run_dependencies():
    py = data.available_python()
    assert Path(py).is_file() and ".venv-mds" in py.replace("\\", "/") or data._probe_python(py)


# --- suivi des runs : lancement, redémarrage du site, arrêt -------------------------------------

@pytest.fixture
def fake_run(tmp_path, monkeypatch):
    """Remplace le script du run par un processus qui dort, dans un dossier d'état temporaire."""
    monkeypatch.setattr(launcher, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(launcher, "running_processes", lambda: [])
    monkeypatch.setattr(launcher, "build_cmd",
                        lambda opts, python=None: [sys.executable, "-c", "import time; time.sleep(300)"])
    launcher.ACTIVE.clear()
    yield
    for live in list(launcher.ACTIVE.values()):
        if live.alive():
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(live.pid)], capture_output=True, check=False)
    launcher.ACTIVE.clear()


def test_launch_persist_restart_and_stop(fake_run):
    live = launcher.launch_run({"name": "essai"})
    assert live.record_path.is_file() and live.status == "en cours" and live.alive()
    # « redémarrage du site » : la mémoire est perdue, le fichier d'état suffit à retrouver le run
    launcher.ACTIVE.clear()
    assert launcher.load_persisted() == 1
    again = launcher.ACTIVE[live.key]
    assert again.proc is None and again.alive() and again.status == "en cours"
    assert launcher.stop_run(live.key) is True
    assert not again.alive() and again.status == "arrêté"
    assert launcher.stop_run(live.key) is False                # déjà arrêté
    assert json.loads(live.record_path.read_text(encoding="utf-8"))["status"] == "arrêté"


def test_elapsed_time_is_per_run(fake_run):
    """Régression : l'heure de départ était évaluée une seule fois pour tous les runs."""
    a = launcher.launch_run({"name": "a"})
    a.started -= 3600
    b = launcher.launch_run({"name": "b"})
    assert a.elapsed_s() - b.elapsed_s() >= 3590


def test_second_run_needs_confirmation(fake_run, monkeypatch):
    monkeypatch.setattr(launcher, "running_processes", lambda: [{"name": "autre", "pids": [1]}])
    with pytest.raises(launcher.RunBusy) as err:
        launcher.launch_run({"name": "second"})
    assert err.value.args[0][0]["name"] == "autre"
    assert not launcher.ACTIVE
    assert "Lancer quand même" in appmod.busy_page({"name": "second"}, err.value.args[0])
    live = launcher.launch_run({"name": "second"}, force=True)    # confirmation explicite
    assert live.alive()


# --- gain validé et zone de recherche -----------------------------------------------------------

@needs_run8
def test_validated_speeds_match_the_run_summary():
    """Les vitesses validées lues dans validation.csv sont exactement celles de summary.txt."""
    run = data.load_run(RUN8)
    wanted = dict(re.findall(r"([\w.-]+) (\d+\.\d+) kt", run.summary.splitlines()[2]))
    speeds = data.validated_speeds(run)["speeds"]
    assert len(speeds) == 6
    for name, value in speeds.items():
        assert value == pytest.approx(float(wanted[name]), abs=6e-5), name
    gain = data.validated_gain(run)
    assert gain["start_name"] == "closest_to_current"
    assert gain["gain_kt"] == pytest.approx(0.256, abs=2e-3)      # le gain annoncé le 29/10


def test_time_saved():
    assert data.time_saved_h(13.0, 13.0) == 0.0
    assert data.time_saved_h(13.0, 13.26, 1000) == pytest.approx(1000 / 13.0 - 1000 / 13.26)
    assert data.time_saved_h(0, 13.0) == 0.0


def test_bounds_summary_is_the_real_family():
    """La zone montrée est celle du run : Ixx >= Ixx(actuelle) en boîte « current », pas en « dimred »."""
    cur = data.bounds_summary("current", 1.0)
    assert cur["ixx_ratio"][0] >= 1.0 - 1e-6
    assert 0.0 < cur["rates"]["toutes les contraintes"] < 1.0
    assert "Ixx ≥ Ixx(actuelle)" in cur["rates"]
    old = data.bounds_summary("dimred", 1.0)                      # la boîte que le site proposait avant
    assert old["ixx_ratio"][1] < 1.0, "aucune section de « dimred » ne respecte Ixx >= Ixx(actuelle)"
    fam = data.build_family("current", 1.0)
    assert fam.ixx_current == pytest.approx(6.717e-5, rel=1e-3)
    # centrée sur la section actuelle (et non sur le centre de la boîte du notebook)
    assert ((fam.lower + fam.upper) / 2 == pytest.approx(fam.current))


# --- pages --------------------------------------------------------------------------------------

def test_index_page():
    idx = appmod.index_page()
    assert "VPP acceleration" in idx and data.list_runs()[0] in idx
    assert "Ixx / Ixx(section actuelle) ≥ 1" in idx             # contrainte affichée telle qu'appliquée
    assert "nombre de saumons" not in idx and "IXX = 1" not in idx
    assert 'value="profile_v2" selected' in idx and 'name="fsi" checked' in idx
    for name in ("name", "time_limit_min", "box_scale", "profile", "start_section", "compare", "fsi", "smoke",
                 "k", "n_low", "n_high", "workers", "vpp_workers", "batch", "seed", "mode", "local_fraction",
                 "converge_window", "converge_tol", "restarts", "min_gain", "sensitivity_median"):
        assert f'name="{name}"' in idx, f"champ manquant : {name}"
    assert 'name="box"' not in idx                                # la boîte n'est pas un choix


def test_form_javascript_is_valid():
    """Le script du formulaire (préréglages, aperçus) est syntaxiquement correct."""
    node = subprocess.run(["node", "--version"], capture_output=True, text=True, check=False)
    if node.returncode != 0:
        pytest.skip("node absent")
    script = re.search(r"<script>(.*?)</script>", appmod.launch_form(), re.S).group(1)
    tmp = Path(__file__).with_name("_form_check.js")
    tmp.write_text(script, encoding="utf-8")
    try:
        out = subprocess.run(["node", "--check", str(tmp)], capture_output=True, text=True, check=False)
    finally:
        tmp.unlink()
    assert out.returncode == 0, out.stderr


@needs_run8
def test_run_page_is_light_and_loads_plotly_once():
    page = appmod.run_page(RUN8)
    assert "Gain validé sur la section de départ" in page and "+0.256" in page
    assert page.count("/static/plotly.min.js") == 1 and "plotly.js v" not in page
    assert len(page) < 1_000_000, "le JavaScript de plotly est de nouveau recopié dans chaque figure"
    assert 'class="warn"' not in page


@needs_run8
def test_run_page_time_conversion_is_editable():
    page = appmod.run_page(RUN8, {"dist": ["2000"], "eur": ["50"]})
    assert "2 000 nm" in page, "distance mise en forme à la française"
    assert "Valeur à 50 €/h" in page


def test_preview_pages_render():
    for html_text in (appmod.bounds_preview_page("current", "1.0"), appmod.bounds_preview_page("dimred", "1"),
                      appmod.profile_preview_page(launcher.DEFAULT_PROFILE)):
        assert "Prévisualisation impossible" not in html_text and "Impossible de lire" not in html_text
        assert "plotly-graph-div" in html_text
    assert "sans contrainte d'Ixx" in appmod.bounds_preview_page("dimred", "1")


def test_figures_have_no_inline_plotly():
    run = data.load_run(data.list_runs()[0])
    for text in (figures.section_html(run), figures.incumbent_html(run)):
        assert "plotly.js v" not in text


def test_savings_path_is_relative_to_the_project():
    assert appmod.SAVINGS_FILE.is_relative_to(data.CODE_ADRIS)


# --- serveur HTTP -------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), appmod.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


def test_server_pages(server):
    status, body, _ = get(server + "/")
    assert status == 200 and b"Nouvelle optimisation" in body
    status, body, headers = get(server + "/static/plotly.min.js")
    assert status == 200 and len(body) > 1_000_000 and "max-age" in headers["Cache-Control"]
    assert get(server + "/run/inexistant")[0] == 404
    assert get(server + "/live/inconnu")[0] == 200            # page claire, pas un plantage
    assert get(server + "/nimporte/quoi")[0] == 404


def test_server_never_serves_files_outside_a_run(server):
    name = data.list_runs()[0]
    assert get(f"{server}/run/{name}/static/config.json")[0] == 200
    for evil in ("../../scripts/run_real_vpp.py", "..%2F..%2Fscripts%2Frun_real_vpp.py"):
        assert get(f"{server}/run/{name}/static/{evil}")[0] == 404
    assert get(f"{server}/run/..%2Fprofiles/static/profile_v1.json")[0] == 404


def test_server_stop_unknown_run(server):
    req = urllib.request.Request(server + "/live/inconnu/stop", method="POST", data=b"")
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req, timeout=30)
    assert err.value.code == 404


def test_server_form_post_launches_and_redirects(server, fake_run):
    """POST du formulaire : redirection vers la page de suivi, qui propose l'arrêt."""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    body = b"name=essai+web&time_limit_min=5&compare=" + data.list_runs()[0].encode()
    with pytest.raises(urllib.error.HTTPError) as err:
        opener.open(urllib.request.Request(server + "/", data=body), timeout=30)
    assert err.value.code == 303
    live_url = server + err.value.headers["Location"]
    status, page, _ = get(live_url)
    assert status == 200 and "Arrêter le run" in page.decode("utf-8") and "onsubmit" in page.decode("utf-8")
    time.sleep(0.2)
