"""
Long-term statistics of the tasks delegated to the local Qwen (ask_local.py / qwen_agent.py).

A *task* groups the Qwen calls made while it was open (logged by qwen_savings.log_event into
CALLS_FILE with the task id; calls without "task" are legacy) plus what Claude itself spent on
it (prompt, review, fixes), measured from the Claude Code transcript between start and end.

Money (USD, Claude Opus 5.5 API, $ per million tokens):
    cost = (input*4 + output*20 + cache_read*0.4 + cache_write_5m*5 + cache_write_1h*8) / 1e6
    est_saved per call = (saved_in*4 + saved_out*20) / 1e6    (same estimate as qwen_savings)
    net_usd per task   = est_saved_usd (of its calls) - claude_usd
Biases, both towards a LOWER net: est_saved is the lower bound of qwen_savings (context re-reads
not counted), and claude_usd is everything Claude spent in the window (prompt, tests, review,
fixes, and its own work done meanwhile), cache re-reads of the whole context included. Keep the
window tight around the delegated work.

Usage:
    python qwen_task.py start "write the parser"          open a task (records the transcript)
    python qwen_task.py end --outcome fixed --note "..."  close it, print the one-line report
    python qwen_task.py status                            show the open task
    python qwen_task.py report [--no-html]                summary + interactive HTML page
"""
import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

OUTCOMES = ("ok", "fixed", "failed", "abandoned")
# $ per million tokens, Claude Opus 5.5 API (claude-api skill)
PRICES = {"input": 4.0, "output": 20.0, "cache_read": 0.4, "cache_write_5m": 5.0, "cache_write_1h": 8.0}

ZERO_USAGE = {"turns": 0, "input": 0, "output": 0, "cache_read": 0,
              "cache_write_5m": 0, "cache_write_1h": 0, "tokens": 0, "usd": 0.0}


# --- paths (read from the environment at CALL time so tests can redirect them) ---
def stats_dir() -> Path:
    """Directory of the stats files (env QWEN_STATS_DIR)."""
    return Path(os.environ.get("QWEN_STATS_DIR", "C:/Users/SIM/Desktop/Code-Adri/Qwen-Benchmark/stats"))


def calls_file() -> Path:
    """Per-call log written by qwen_savings.log_event."""
    return stats_dir() / "qwen_usage.jsonl"


def tasks_file() -> Path:
    """One JSON line per closed task."""
    return stats_dir() / "tasks.jsonl"


def current_file() -> Path:
    """File of the task currently open."""
    return stats_dir() / "current_task.json"


def html_file() -> Path:
    """Self-contained plotly page of the long-term statistics."""
    return stats_dir() / "qwen_stats.html"


def transcripts_dir() -> Path:
    """Claude Code transcripts (env CLAUDE_TRANSCRIPTS_DIR)."""
    return Path(os.environ.get("CLAUDE_TRANSCRIPTS_DIR", "~/.claude/projects")).expanduser()


# --- helpers ---
def _parse_time(value) -> dt.datetime:
    """ISO string (with Z / offset, or naive = UTC) or None (now) -> timezone-aware UTC datetime."""
    if value is None:
        return dt.datetime.now(dt.timezone.utc)
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    s = str(value).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    d = dt.datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)


def _iso(d: dt.datetime) -> str:
    """Aware datetime -> UTC ISO string ending in Z."""
    return d.astimezone(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _cost(usage: dict) -> float:
    """USD of one usage dict at PRICES."""
    return sum(usage[k] * p for k, p in PRICES.items()) / 1e6


def _call_saved(c: dict) -> float:
    """Estimated saving of one call line (same formula as qwen_savings)."""
    return (c.get("saved_in", 0) * PRICES["input"] + c.get("saved_out", 0) * PRICES["output"]) / 1e6


def read_jsonl(path) -> list:
    """Lines of a JSON-lines file as dicts, skipping invalid lines; [] if the file is missing."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# --- Claude's real usage, from the transcript ---
def newest_transcript(root=None):
    """Newest (mtime) *.jsonl in the subdirectories of root (pattern root/*/*.jsonl); None if none."""
    root = Path(root) if root else transcripts_dir()
    files = [p for p in root.glob("*/*.jsonl") if p.is_file()]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def claude_usage(transcript, start, end) -> dict:
    """Sum of the assistant usages of [start, end] (aware UTC) in one transcript.

    One message (same message.id on several lines) is counted once; lines of other types,
    without usage, or not valid JSON are skipped. transcript=None or missing -> all zeros.
    """
    u = dict(ZERO_USAGE)
    p = Path(transcript) if transcript else None
    if p is None or not p.exists():
        return u
    seen = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "assistant":
            continue
        msg = obj.get("message") or {}
        usage = msg.get("usage")
        mid = msg.get("id")
        if not usage or (mid is not None and mid in seen):
            continue
        try:
            ts = _parse_time(obj["timestamp"])
        except (KeyError, ValueError):
            continue
        if not (start <= ts <= end):
            continue
        seen.add(mid)
        cc = usage.get("cache_creation") if isinstance(usage.get("cache_creation"), dict) else None
        if cc is not None:
            w5, w1 = cc.get("ephemeral_5m_input_tokens", 0), cc.get("ephemeral_1h_input_tokens", 0)
        else:
            w5, w1 = 0, usage.get("cache_creation_input_tokens", 0)
        vals = {"input": usage.get("input_tokens", 0), "output": usage.get("output_tokens", 0),
                "cache_read": usage.get("cache_read_input_tokens", 0),
                "cache_write_5m": w5, "cache_write_1h": w1}
        u["turns"] += 1
        for k, v in vals.items():
            u[k] += v
    u["tokens"] = sum(v for k, v in u.items() if k not in ("turns", "usd", "tokens"))
    u["usd"] = _cost(u)
    return u


# --- task lifecycle ---
def current_task():
    """Content of CURRENT_FILE (the open task), or None."""
    try:
        return json.loads(current_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def start(title, at=None, transcript=None) -> dict:
    """Open a task (id = start time in UTC), record the transcript to measure Claude on."""
    if current_task() is not None:
        raise RuntimeError("a task is already open")
    t = _parse_time(at)
    task = {"id": t.strftime("%Y%m%d-%H%M%S"), "title": title, "start": _iso(t),
            "transcript": str(transcript or newest_transcript() or "") or None}
    stats_dir().mkdir(parents=True, exist_ok=True)
    current_file().write_text(json.dumps(task, ensure_ascii=False), encoding="utf-8")
    return task


def end(outcome, note="", at=None) -> tuple:
    """Close the open task: aggregate its calls and Claude's usage; return (record, one-line report)."""
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    task = current_task()
    if task is None:
        raise RuntimeError("no task is open")
    end_t = _parse_time(at)
    start_t = _parse_time(task["start"])
    calls = [c for c in read_jsonl(calls_file()) if c.get("task") == task["id"]]
    usage = claude_usage(task.get("transcript"), start_t, end_t)
    est = sum(_call_saved(c) for c in calls)
    record = {
        "id": task["id"], "title": task["title"], "start": task["start"], "end": _iso(end_t),
        "minutes": round((end_t - start_t).total_seconds() / 60.0, 1),
        "outcome": outcome, "note": str(note)[:200],
        "qwen_calls": len(calls), "qwen_minutes": round(sum(c.get("minutes", 0) for c in calls), 2),
        "qwen_tokens": sum(c.get("qwen_in", 0) + c.get("qwen_out", 0) for c in calls),
        "est_saved_tokens": sum(c.get("saved_in", 0) + c.get("saved_out", 0) for c in calls),
        "est_saved_usd": round(est, 4),
        "claude_turns": usage["turns"], "claude_tokens": usage["tokens"], "claude_usd": round(usage["usd"], 4),
        "claude_detail": {k: usage[k] for k in
                          ("input", "output", "cache_read", "cache_write_5m", "cache_write_1h")},
        "net_usd": round(est - usage["usd"], 4),
    }
    line = json.dumps(record, ensure_ascii=False)
    stats_dir().mkdir(parents=True, exist_ok=True)
    with open(tasks_file(), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    current_file().unlink(missing_ok=True)
    all_tasks = read_jsonl(tasks_file())
    cum = round(sum(t.get("net_usd", 0) for t in all_tasks), 4)
    return record, (f"[qwen task] {record['title']} ({outcome}): {record['qwen_calls']} Qwen call(s), "
                    f"{record['qwen_minutes']} min, {record['qwen_tokens']:,} tokens; "
                    f"Claude {record['claude_tokens']:,} tokens, ${record['claude_usd']:.4f}; "
                    f"est. saving ${record['est_saved_usd']:.4f}, net ${record['net_usd']:+.4f} | "
                    f"cumulative: {len(all_tasks)} task(s), net ${cum:+.4f}")


def summary(tasks, calls) -> dict:
    """Aggregate numbers over closed tasks and all calls (including legacy ones)."""
    outcomes = {}
    for t in tasks:
        outcomes[t.get("outcome", "?")] = outcomes.get(t.get("outcome", "?"), 0) + 1
    return {"tasks": len(tasks), "outcomes": outcomes,
            "qwen_calls": len(calls), "legacy_calls": sum("task" not in c for c in calls),
            "qwen_minutes": round(sum(c.get("minutes", 0) for c in calls), 2),
            "qwen_tokens": sum(c.get("qwen_in", 0) + c.get("qwen_out", 0) for c in calls),
            "claude_usd": round(sum(t.get("claude_usd", 0) for t in tasks), 4),
            "est_saved_usd": round(sum(_call_saved(c) for c in calls), 4),
            "net_usd": round(sum(t.get("net_usd", 0) for t in tasks), 4)}


# --- HTML report ---
def write_html(tasks, calls, path=None) -> Path:
    """Self-contained interactive plotly page (French) of the long-term statistics."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    tasks = sorted(tasks, key=lambda t: t.get("start", ""))
    calls = sorted(calls, key=lambda c: c.get("time", ""))
    fig = make_subplots(rows=5, cols=1, vertical_spacing=0.08,
                        row_heights=[0.18, 0.17, 0.17, 0.18, 0.15],
                        subplot_titles=("Tâches et appels Qwen (cumulatif)", "Par tâche : minutes Qwen",
                                        "Par tâche : tokens Claude (empilés)", "Évolution en $ (cumulatif)",
                                        "Tâches par résultat"))
    # 1. cumulative counts over time (merge both streams, sorted by time)
    events = [(t.get("start"), "task") for t in tasks] + [(c.get("time"), "call") for c in calls]
    events.sort(key=lambda p: p[0])
    nt = nc = 0
    x_tasks, y_tasks, x_calls, y_calls = [], [], [], []
    for when, kind in events:
        if kind == "task":
            nt += 1; x_tasks.append(when); y_tasks.append(nt)
        else:
            nc += 1; x_calls.append(when); y_calls.append(nc)
    if x_tasks:
        fig.add_trace(go.Scatter(x=x_tasks, y=y_tasks, mode="lines+markers", name="tâches (cumul)"), row=1, col=1)
    if x_calls:
        fig.add_trace(go.Scatter(x=x_calls, y=y_calls, mode="lines+markers", name="appels Qwen (cumul)"), row=1, col=1)
    # 2. per task: Qwen minutes bar + Claude tokens stacked
    titles = [f"{t['id'][4:8]}-{t['id'][9:13]} {t['title'][:16]}" for t in tasks]   # unique labels
    if tasks:
        fig.add_trace(go.Bar(x=titles, y=[t.get("qwen_minutes", 0) for t in tasks],
                             name="minutes Qwen", marker_color="#2a6fdb"), row=2, col=1)
        colors = {"input": "#2a6fdb", "output": "#e07b39", "cache_read": "#94a3b8",
                  "cache_write_5m": "#6ba583", "cache_write_1h": "#b56576"}
        for key, label in (("input", "input"), ("output", "output"), ("cache_read", "cache lecture"),
                           ("cache_write_5m", "cache écriture 5m"), ("cache_write_1h", "cache écriture 1h")):
            fig.add_trace(go.Bar(x=titles, y=[(t.get("claude_detail") or {}).get(key, 0) for t in tasks],
                                 name=label, marker_color=colors[key]), row=3, col=1)
    # 3. cumulative $ over time
    if tasks:
        saved_t = claude_t = 0.0
        x3, y_saved, y_claude, y_net = [], [], [], []
        for t in tasks:
            saved_t += t.get("est_saved_usd", 0); claude_t += t.get("claude_usd", 0)
            x3.append(t.get("start"))
            y_saved.append(round(saved_t, 4)); y_claude.append(round(claude_t, 4))
            y_net.append(round(saved_t - claude_t, 4))
        fig.add_trace(go.Scatter(x=x3, y=y_saved, mode="lines+markers", name="économie estimée (tâches)"), row=4, col=1)
        fig.add_trace(go.Scatter(x=x3, y=y_claude, mode="lines+markers", name="coût Claude (tâches)"), row=4, col=1)
        fig.add_trace(go.Scatter(x=x3, y=y_net, mode="lines+markers", name="net (tâches)"), row=4, col=1)
    if calls:
        saved_c = 0.0
        y_c = []
        for c in calls:
            saved_c += _call_saved(c)
            y_c.append(round(saved_c, 4))
        fig.add_trace(go.Scatter(x=[c.get("time") for c in calls], y=y_c, mode="lines", name="économie estimée (tous appels)"), row=4, col=1)
    # 4. tasks per outcome
    fig.add_trace(go.Bar(x=[o for o in OUTCOMES], y=[sum(1 for t in tasks if t.get("outcome") == o) for o in OUTCOMES],
                         name="tâches", marker_color=["#2a9d8f", "#e9c46a", "#e76f51", "#8d99ae"]), row=5, col=1)
    # 5. table of every task (plain figure: table traces don't fit an xy subplot grid)
    table = go.Figure(go.Table(
        header=dict(values=["date", "titre", "résultat", "appels Qwen", "min", "tokens Claude", "Claude $",
                            "estimé $", "net $"], fill_color="#f0f4fa"),
        cells=dict(values=[[t["start"][:10], t["title"], t.get("outcome", ""), t.get("qwen_calls", 0),
                            t.get("qwen_minutes", 0), t.get("claude_tokens", 0),
                            round(t.get("claude_usd", 0), 4), round(t.get("est_saved_usd", 0), 4),
                            round(t.get("net_usd", 0), 4)] for t in tasks], fill_color="#ffffff"),
        columnwidth=[70, 160, 80, 90, 60, 100, 80, 80, 80]))
    fig.update_layout(title_text="Statistiques de long terme — tâches déléguées à Qwen",
                      template="plotly_white", height=1050, barmode="stack")
    fig.update_yaxes(title_text="nb", row=1, col=1)
    fig.update_yaxes(title_text="minutes Qwen", row=2, col=1)
    fig.update_yaxes(title_text="tokens Claude", row=3, col=1)
    fig.update_yaxes(title_text="$ (cumulatif)", row=4, col=1)
    fig.update_yaxes(title_text="nb de tâches", row=5, col=1)
    table.update_layout(title_text="Détail des tâches", template="plotly_white", height=300 + 35 * len(tasks))
    p = Path(path) if path else html_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    page = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>Statistiques de long terme — tâches déléguées à Qwen</title></head><body>"
            + fig.to_html(full_html=False, include_plotlyjs="cdn")
            + table.to_html(full_html=False, include_plotlyjs=False)
            + "</body></html>")
    p.write_text(page, encoding="utf-8")
    return p


# --- CLI ---
def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Long-term statistics of the tasks delegated to Qwen.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_start = sub.add_parser("start", help="open a task")
    p_start.add_argument("title"); p_start.add_argument("--at")
    p_end = sub.add_parser("end", help="close the open task")
    p_end.add_argument("--outcome", required=True); p_end.add_argument("--note", default=""); p_end.add_argument("--at")
    sub.add_parser("status", help="show the open task")
    p_rep = sub.add_parser("report", help="summary + HTML page")
    p_rep.add_argument("--no-html", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "start":
        print(start(args.title, at=args.at))
    elif args.cmd == "end":
        _, line = end(args.outcome, note=args.note, at=args.at)
        print(line)
    elif args.cmd == "status":
        task = current_task()
        print(json.dumps(task, ensure_ascii=False) if task else "no open task")
    else:
        tasks, calls = read_jsonl(tasks_file()), read_jsonl(calls_file())
        s = summary(tasks, calls)
        print(f"{s['tasks']} task(s) closed {s['outcomes'] or ''} | {s['qwen_calls']} Qwen call(s) "
              f"({s['legacy_calls']} legacy), {s['qwen_minutes']} min, {s['qwen_tokens']:,} tokens")
        print(f"est. saving ${s['est_saved_usd']:.4f} (all calls) - Claude ${s['claude_usd']:.4f} "
              f"(tasks) = net ${s['net_usd']:+.4f}")
        if not args.no_html:
            print(f"HTML: {write_html(tasks, calls)}")


if __name__ == "__main__":
    main()
