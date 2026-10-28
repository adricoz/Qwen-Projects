"""Prompting study: run the Qwen agent on the gp_lite task with 4 prompt variants, 2 runs each, and score
every run with Claude's acceptance suite (always the pristine copy in ../acceptance).

    python run_study.py            # runs what is missing in results/, in the order A1 B1 C1 D1 A2 B2 C2 D2
    python run_study.py A1 C2      # only these runs

Variants (prompts/): A = base task, B = base + acceptance tests given, C = base + formulas,
D = base + imposed verification method.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ACCEPTANCE = os.path.join(HERE, "..", "acceptance", "test_acceptance.py")
AGENT = "C:/Users/SIM/.claude/tools/qwen_agent.py"
READ_ROOT = "C:/Users/SIM/Desktop/Code-Adri/Multifidelity-GP-Clean"
ORDER = ["A1", "B1", "C1", "D1", "A2", "B2", "C2", "D2"]
DELIVERABLES = "gp_lite/kernels.py,gp_lite/gp.py,tests/test_gp_lite.py,examples/branin_demo.py,README.md"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(HERE, "study.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def server_ok():
    try:
        return b'"ok"' in urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=5).read()
    except Exception:
        return False


def ensure_server():
    """Restart the scheduled task if llama-server is down; unload any big Ollama model (GPU conflict)."""
    if not server_ok():
        log("llama-server down: schtasks /Run")
        subprocess.run(["schtasks", "/Run", "/TN", "Qwen llama-server"], capture_output=True)
        for _ in range(60):
            time.sleep(5)
            if server_ok():
                break
        else:
            sys.exit("llama-server did not come back")
    try:
        ps = json.load(urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=3))
        for m in ps.get("models", []):
            if m.get("size_vram", m.get("size", 0)) > 4e9:
                log(f"Ollama holds {m['name']} on the GPU: ollama stop")
                subprocess.run(["ollama", "stop", m["name"]], capture_output=True)
    except Exception:
        pass


def write_task(run_dir, arm):
    text = open(os.path.join(HERE, "prompts", "base.md"), encoding="utf-8").read()
    if arm != "A":
        text += open(os.path.join(HERE, "prompts", f"{arm}.md"), encoding="utf-8").read()
    with open(os.path.join(run_dir, "TASK.md"), "w", encoding="utf-8") as f:
        f.write(text)
    if arm == "B":
        os.makedirs(os.path.join(run_dir, "tests_acceptance"), exist_ok=True)
        shutil.copy(ACCEPTANCE, os.path.join(run_dir, "tests_acceptance", "test_acceptance.py"))


def acceptance(run_dir):
    """Pristine acceptance suite on the run's package: {test name: True/False}."""
    xml = os.path.join(run_dir, ".qwen_agent", "acceptance.xml")
    env = {**os.environ, "GP_LITE_DIR": run_dir, "MPLBACKEND": "Agg"}
    cwd = os.path.dirname(os.path.abspath(ACCEPTANCE))
    try:
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}",
                        os.path.abspath(ACCEPTANCE)], cwd=cwd, env=env, capture_output=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {"timeout": False}
    if not os.path.exists(xml):
        return {"no_result": False}
    tests = {}
    for case in ET.parse(xml).getroot().iter("testcase"):
        ok = not any(child.tag in ("failure", "error", "skipped") for child in case)
        tests[case.get("name")] = ok
    if not tests:   # import error: pytest reports a single collection error
        tests["collection_error"] = False
    return tests


def own_tests(run_dir):
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=run_dir,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        return lines[-1] if lines else "(no output)"
    except subprocess.TimeoutExpired:
        return "timeout"


def run_one(run):
    arm = run[0]
    run_dir = os.path.join(HERE, "runs", run)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    write_task(run_dir, arm)
    ensure_server()
    test_cmd = "-m pytest -q --tb=short tests tests_acceptance" if arm == "B" else "-m pytest -q tests"
    cmd = [sys.executable, AGENT, "@" + os.path.join(run_dir, "TASK.md"), "--workdir", run_dir, "--read-root", READ_ROOT,
           "--effort", "medium", "--max-steps", "90", "--max-minutes", "60",
           "--deliverables", DELIVERABLES, "--test-cmd", test_cmd]
    log(f"{run}: agent start")
    t0 = time.time()
    with open(os.path.join(run_dir, "agent_run.log"), "w", encoding="utf-8") as out:
        try:
            subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, timeout=100 * 60)
        except subprocess.TimeoutExpired:
            out.write("\n[run_study] killed after 100 min\n")
    wall = (time.time() - t0) / 60
    transcript = os.path.join(run_dir, ".qwen_agent", "transcript.json")
    meta = {}
    if os.path.exists(transcript):
        t = json.load(open(transcript, encoding="utf-8"))
        meta = {"stop": t["stop"], "steps": t["steps"], "minutes": t["minutes"]}
    agent_out = open(os.path.join(run_dir, "agent_run.log"), encoding="utf-8", errors="replace").read()
    report = agent_out.split("=== REPORT ===", 1)[1].split("[qwen_agent] stop:", 1)[0].strip() if "=== REPORT ===" in agent_out else ""
    acc = acceptance(run_dir)
    res = {"run": run, "arm": arm, "wall_minutes": wall, **meta,
           "deliverables_missing": [d for d in DELIVERABLES.split(",") if not os.path.exists(os.path.join(run_dir, d))],
           "own_tests": own_tests(run_dir), "acceptance": acc,
           "acceptance_passed": sum(acc.values()), "acceptance_total": 23, "report": report}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"{run}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    log(f"{run}: {meta.get('stop')} | {meta.get('steps')} steps | {wall:.0f} min | acceptance {res['acceptance_passed']}/23 "
        f"| own tests: {res['own_tests']} | missing: {res['deliverables_missing'] or 'none'}")


def main():
    runs = sys.argv[1:] or [r for r in ORDER if not os.path.exists(os.path.join(HERE, "results", f"{r}.json"))]
    log(f"study: runs {runs}")
    for run in runs:
        run_one(run)
    log("study: done")


if __name__ == "__main__":
    main()
