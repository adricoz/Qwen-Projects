# -*- coding: utf-8 -*-
"""
Lancement, suivi et arrêt des runs de ``scripts/run_real_vpp.py`` lancés depuis le site.

Un run dure de quelques minutes à plusieurs heures : il doit survivre au site.
- Chaque run lancé est décrit dans ``logs/<clé>.json`` (pid, commande, heure) ; le site
  relit ces fichiers au démarrage, donc un redémarrage ne fait pas perdre le bouton « arrêter ».
- Quitter le site (Ctrl-C) ne tue JAMAIS un run.
- Un second run n'est lancé que si l'utilisateur le demande explicitement (la machine est partagée,
  deux runs se ralentissent mutuellement).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import data

RUN_SCRIPT = data.VPP_DIR / "scripts" / "run_real_vpp.py"
LOG_DIR = Path(__file__).resolve().parent / "logs"

# -----------------------------------------------------------------------------
# Réglages : tous correspondent à un flag réel de run_real_vpp.py (voir --help)
# -----------------------------------------------------------------------------

# clé du formulaire -> (flag, type, minimum, maximum)
NUMERIC = {
    "time_limit_min": ("--time-limit-min", float, 1, 1440),
    "box_scale": ("--box-scale", float, 0.1, 5.0),
    "k": ("--k", int, 2, 16),
    "n_low": ("--n-low", int, 8, 2000),
    "n_high": ("--n-high", int, 4, 1000),
    "workers": ("--workers", int, 1, 32),
    "vpp_workers": ("--vpp-workers", int, 0, 32),
    "batch": ("--batch", int, 1, 64),
    "seed": ("--seed", int, 0, 2 ** 31 - 1),
    "local_fraction": ("--local-fraction", float, 0.0, 1.0),
    "min_gain": ("--min-gain", float, 0.0, 1.0),
    "sensitivity_median": ("--sensitivity-median", int, 1, 20),
}

# Réglages de départ proposés par le formulaire (tout reste modifiable à la main).
PRESETS = {
    "production": {
        "label": "Production — réglages des runs 7-8 (≈ 2 h 30)",
        "values": {"time_limit_min": "140", "box_scale": "3", "k": "10", "n_low": "256",
                   "n_high": "64", "workers": "16", "vpp_workers": "16", "batch": "16", "seed": "0",
                   "local_fraction": "0.5", "min_gain": "0.002", "sensitivity_median": "3",
                   "converge_window": "10", "converge_tol": "0.01", "restarts": "2",
                   "profile": "profile_v2"},
    },
    "rapide": {
        "label": "Test rapide — vérifier une idée, 5 conditions (≈ 25 min)",
        # 5 conditions : avec profile_v2 (49 conditions) une seule mise à jour du modèle dure déjà
        # plus de 10 min (run test du 30/10 : 767 s), un test de 10 min n'aurait aucune mise à jour
        "values": {"time_limit_min": "10", "box_scale": "1", "k": "10", "n_low": "120",
                   "n_high": "30", "workers": "8", "vpp_workers": "8", "batch": "4", "seed": "0",
                   "local_fraction": "0", "min_gain": "0", "sensitivity_median": "1",
                   "converge_window": "", "converge_tol": "", "restarts": "0", "profile": "none"},
    },
}
DEFAULT_PRESET = "production"
DEFAULT_PROFILE = "profile_v2"


def slugify(name: str | None) -> str:
    """Nom de run sûr pour un dossier : minuscules, chiffres et ``_``, 40 caractères au plus."""
    slug = re.sub(r"[^a-z0-9_]", "_", (name or "").strip().lower())[:40].strip("_")
    return slug or "mon_run"


def _num(value, kind, lo, hi) -> str | None:
    """Valeur numérique du formulaire bornée à [lo, hi] ; ``None`` si vide ou illisible."""
    if value is None or str(value).strip() == "":
        return None
    try:
        f = float(str(value).replace(",", "."))
    except ValueError:
        return None
    if f != f:  # NaN
        return None
    f = min(max(f, lo), hi)
    return str(int(round(f))) if kind is int else f"{f:g}"


def build_cmd(opts: dict, python: str | None = None) -> list[str]:
    """Commande de lancement construite à partir des champs du formulaire.

    ``opts`` est un dict de chaînes (champs du formulaire) ; ``opts["compare"]`` peut être une
    liste de noms de runs. La boîte est TOUJOURS ``--box current`` : c'est la seule qui impose
    Ixx >= Ixx(section actuelle) (la boîte « dimred » du premier run n'a aucune contrainte
    d'Ixx : toutes ses sections sont plus souples que la section actuelle).
    """
    cmd = [python or data.available_python(), str(RUN_SCRIPT), "--name", slugify(opts.get("name")),
           "--box", "current"]
    for key, (flag, kind, lo, hi) in NUMERIC.items():
        value = _num(opts.get(key), kind, lo, hi)
        if value is not None:
            cmd += [flag, value]
    if opts.get("mode") == "npllt":
        cmd += ["--hf", "npllt_xfoil"]
    if opts.get("fsi") == "on":
        cmd.append("--fsi")
    window = _num(opts.get("converge_window"), int, 2, 200)
    tol = _num(opts.get("converge_tol"), float, 0.0, 10.0)
    if window and tol:
        cmd += ["--converge", window, tol]
        restarts = _num(opts.get("restarts"), int, 0, 10)
        if restarts and restarts != "0":
            cmd += ["--restarts", restarts]          # --restarts exige --converge
    prof = opts.get("profile") or ""
    if prof and prof != "none":
        path = data.profile_path(prof)
        if path is not None:
            cmd += ["--search-profile", str(path), "--profile", str(path)]
    known = set(data.list_runs())
    start = opts.get("start_section") or ""
    if start in known:
        cmd += ["--start-section", str(data.RUNS_DIR / start / "results" / "best_section.json")]
    compare = [c for c in (opts.get("compare") or []) if c in known][:8]
    if compare:
        cmd += ["--compare"] + [str(data.RUNS_DIR / c / "results" / "best_section.json") for c in compare]
    if opts.get("smoke") == "on":
        cmd.append("--smoke")
    return cmd


# -----------------------------------------------------------------------------
# Processus
# -----------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    """Le processus ``pid`` existe-t-il encore ?"""
    if not pid:
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True, timeout=20, check=False).stdout
            return f'"{int(pid)}"' in out
        os.kill(int(pid), 0)
        return True
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


def running_processes() -> list[dict]:
    """Runs en cours sur la machine (lancés par le site ou à la main) : ``[{"name", "pids"}]``."""
    try:
        if os.name == "nt":
            script = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "
                      "'*run_real_vpp.py*' -and $_.CommandLine -notlike '*Get-CimInstance*' } | "
                      "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
            out = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                                 capture_output=True, text=True, timeout=40, check=False).stdout
        else:
            out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True,
                                 timeout=20, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    runs: dict[str, list[int]] = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s+(.*)", line.replace("\t", " "))
        if not m or "run_real_vpp.py" not in m.group(2) or "python" not in m.group(2).lower():
            continue
        name = re.search(r"--name\s+(\S+)", m.group(2))
        runs.setdefault(name.group(1) if name else "?", []).append(int(m.group(1)))
    return [{"name": k, "pids": v} for k, v in runs.items()]


class RunBusy(RuntimeError):
    """Un run tourne déjà sur la machine (``args[0]`` : la liste de ces runs)."""


@dataclass
class LiveRun:
    """Un run lancé par le site : son processus, son dossier et l'état connu."""

    key: str
    name: str
    cmd: list
    pid: int
    started: float
    out_log: Path
    proc: subprocess.Popen | None = None
    status: str = "en cours"
    error: str = ""
    ended: float | None = None

    # -- fichier d'état ---------------------------------------------------------
    @property
    def record_path(self) -> Path:
        return LOG_DIR / f"{self.key}.json"

    def save(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        rec = {"key": self.key, "name": self.name, "cmd": self.cmd, "pid": self.pid,
               "started": self.started, "out_log": str(self.out_log), "status": self.status,
               "ended": self.ended}
        self.record_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")

    # -- état -------------------------------------------------------------------
    def folder(self) -> str | None:
        """Nom du dossier créé par le run (AAAA-MM-JJ_HHMM_<nom>), ``None`` s'il n'existe pas encore."""
        pattern = data.run_folder_pattern(self.name)
        if not data.RUNS_DIR.is_dir():
            return None
        found = [d for d in data.RUNS_DIR.iterdir()
                 if d.is_dir() and pattern.match(d.name) and d.stat().st_ctime >= self.started - 30]
        return max(found, key=lambda d: d.stat().st_ctime).name if found else None

    def alive(self) -> bool:
        return self.proc.poll() is None if self.proc is not None else _pid_alive(self.pid)

    def status_now(self) -> None:
        """Met à jour ``status`` (en cours / terminé / terminé (échec) / arrêté)."""
        if self.status != "en cours":
            return
        if self.alive():
            return
        folder = self.folder()
        done = folder is not None and (data.RUNS_DIR / folder / "results" / "best_section.json").is_file()
        self.status = "terminé" if done else "terminé (échec)"
        if self.out_log.is_file():
            self.ended = self.out_log.stat().st_mtime
        if not done:
            self.error = self._error_text(folder)
        self.save()

    def _error_text(self, folder: str | None) -> str:
        """Fin du journal quand le run s'est arrêté sans résultat (la cause de l'échec)."""
        for path in ((data.RUNS_DIR / folder / "run.log") if folder else None, self.out_log):
            text = tail(path, 3000) if path is not None else ""
            if re.search(r"Traceback|Error|Exception", text):
                return text[-1500:]
        return ""

    def elapsed_s(self) -> int:
        return round((self.ended or time.time()) - self.started)

    def to_dict(self) -> dict:
        self.status_now()
        folder = self.folder()
        return {
            "key": self.key, "name": self.name, "folder": folder, "status": self.status,
            "error": self.error, "elapsed_s": self.elapsed_s(), "cmd": subprocess.list2cmdline(self.cmd),
            "run_log": tail(data.RUNS_DIR / folder / "run.log") if folder else "",
            "out_log": tail(self.out_log),
        }


def tail(path: Path | None, n: int = 4000) -> str:
    """Les ``n`` derniers caractères d'un fichier texte (vide s'il n'existe pas)."""
    if path is None or not Path(path).is_file():
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - n * 2))
        raw = f.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:   # console Windows redirigée vers un fichier : cp1252 (runs d'avant le correctif)
        text = raw.decode("cp1252", errors="replace")
    return text[-n:]


ACTIVE: dict[str, LiveRun] = {}
ACTIVE_LOCK = threading.Lock()


def load_persisted(max_age_days: float = 14.0) -> int:
    """Relit ``logs/*.json`` : retrouve les runs lancés avant un redémarrage du site."""
    n = 0
    if not LOG_DIR.is_dir():
        return n
    for path in LOG_DIR.glob("*.json"):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - float(rec["started"]) > max_age_days * 86400:
                continue
            live = LiveRun(rec["key"], rec["name"], rec["cmd"], int(rec["pid"]), float(rec["started"]),
                           Path(rec["out_log"]), None, rec.get("status", "en cours"), "", rec.get("ended"))
        except (OSError, ValueError, KeyError):
            continue
        with ACTIVE_LOCK:
            ACTIVE.setdefault(live.key, live)
        n += 1
    return n


def launch_run(opts: dict, python: str | None = None, force: bool = False) -> LiveRun:
    """Lance ``run_real_vpp.py`` en tâche de fond, détaché du navigateur et du site.

    Raises
    ------
    RunBusy
        Si un run tourne déjà sur la machine et que ``force`` est faux.
    """
    busy = running_processes()
    if busy and not force:
        raise RunBusy(busy)
    slug = slugify(opts.get("name"))
    key = time.strftime("%Y%m%d_%H%M%S") + "_" + slug
    cmd = build_cmd(opts, python)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_log = LOG_DIR / (key + ".out")
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    with open(out_log, "w", encoding="utf-8", errors="replace") as out:
        # sortie console en UTF-8 (sinon cp1252 : accents illisibles dans le journal du site)
        proc = subprocess.Popen(cmd, cwd=str(data.VPP_DIR), stdout=out, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, creationflags=flags,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    live = LiveRun(key, slug, cmd, proc.pid, time.time(), out_log, proc)
    live.save()
    with ACTIVE_LOCK:
        ACTIVE[key] = live
    return live


def live_status(key: str) -> dict | None:
    live = ACTIVE.get(key)
    return live.to_dict() if live is not None else None


def stop_run(key: str) -> bool:
    """Arrête un run lancé depuis le site (kill de toute l'arborescence : workers, AVL, XFOIL).

    Retourne ``False`` si le run est inconnu ou déjà terminé.
    """
    live = ACTIVE.get(key)
    if live is None:
        return False
    live.status_now()
    if live.status != "en cours":
        return False
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(live.pid)],
                           capture_output=True, timeout=30, check=False)
        else:
            import signal  # noqa: PLC0415
            try:
                os.killpg(os.getpgid(live.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                os.kill(live.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        return False
    if live.proc is not None:
        try:
            live.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            return False
    live.status, live.ended = "arrêté", time.time()
    live.save()
    return True
