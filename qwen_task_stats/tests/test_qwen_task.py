"""Acceptance tests of qwen_task.py (written by Claude before the delegation, 2026-10-28)."""
import datetime as dt
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen_task  # noqa: E402

UTC = dt.timezone.utc


def usage(i, o, read, w5, w1):
    return {"input_tokens": i, "output_tokens": o, "cache_read_input_tokens": read,
            "cache_creation_input_tokens": w5 + w1,
            "cache_creation": {"ephemeral_5m_input_tokens": w5, "ephemeral_1h_input_tokens": w1}}


def assistant(ts, mid, u):
    return json.dumps({"type": "assistant", "timestamp": ts, "message": {"id": mid, "usage": u}})


@pytest.fixture
def env(tmp_path, monkeypatch):
    stats, projects = tmp_path / "stats", tmp_path / "projects"
    (projects / "p").mkdir(parents=True)
    monkeypatch.setenv("QWEN_STATS_DIR", str(stats))
    monkeypatch.setenv("CLAUDE_TRANSCRIPTS_DIR", str(projects))
    transcript = projects / "p" / "session.jsonl"
    lines = [
        assistant("2026-10-28T07:59:00.000Z", "m0", usage(1, 1, 1, 1, 1)),          # before the task
        json.dumps({"type": "user", "timestamp": "2026-10-28T08:01:00.000Z", "message": {"content": "hi"}}),
        assistant("2026-10-28T08:02:00.000Z", "m1", usage(10, 100, 1000, 200, 300)),
        assistant("2026-10-28T08:02:01.000Z", "m1", usage(10, 100, 1000, 200, 300)),  # same message: once
        "not json",
        assistant("2026-10-28T08:30:00.000Z", "m2", {"input_tokens": 5, "output_tokens": 50,
                                                     "cache_read_input_tokens": 0,
                                                     "cache_creation_input_tokens": 100}),  # no detail: 1h
        assistant("2026-10-28T09:30:00.000Z", "m3", usage(7, 7, 7, 7, 7)),          # after the task
    ]
    transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"stats": stats, "transcript": transcript}


M1_USD = (10 * 4 + 100 * 20 + 1000 * 0.4 + 200 * 5 + 300 * 8) / 1e6
M2_USD = (5 * 4 + 50 * 20 + 100 * 8) / 1e6


def test_newest_transcript_and_usage(env):
    assert qwen_task.newest_transcript() == env["transcript"]
    u = qwen_task.claude_usage(env["transcript"], dt.datetime(2026, 10, 28, 8, 0, tzinfo=UTC),
                               dt.datetime(2026, 10, 28, 9, 0, tzinfo=UTC))
    assert u["turns"] == 2
    assert (u["input"], u["output"], u["cache_read"], u["cache_write_5m"], u["cache_write_1h"]) == \
        (15, 150, 1000, 200, 400)
    assert u["tokens"] == 15 + 150 + 1000 + 200 + 400
    assert u["usd"] == pytest.approx(M1_USD + M2_USD)
    assert qwen_task.claude_usage(None, dt.datetime(2026, 1, 1, tzinfo=UTC),
                                  dt.datetime(2026, 1, 2, tzinfo=UTC))["usd"] == 0


def test_start_end_record(env):
    task = qwen_task.start("stats tool", at="2026-10-28T08:00:00Z")
    assert task["id"] == "20261028-080000" and task["start"].endswith("Z")
    assert qwen_task.current_task()["title"] == "stats tool"
    with pytest.raises(RuntimeError):
        qwen_task.start("again")
    calls = [{"time": "2026-10-28T08:10:00", "tool": "qwen_agent", "saved_in": 1000, "saved_out": 2000,
              "qwen_in": 50000, "qwen_out": 8000, "minutes": 12.5, "task": task["id"]},
             {"time": "2026-10-28T08:40:00", "tool": "ask_local", "saved_in": 500, "saved_out": -50,
              "qwen_in": 9000, "qwen_out": 1000, "minutes": 1.25, "task": task["id"]},
             {"time": "2026-10-27T16:29:31", "tool": "ask_local", "saved_in": 79249, "saved_out": -450,
              "qwen_in": 1, "qwen_out": 1, "minutes": 4.85}]                       # legacy call
    qwen_task.calls_file().write_text("\n".join(json.dumps(c) for c in calls) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        qwen_task.end("great", at="2026-10-28T09:00:00Z")
    record, line = qwen_task.end("fixed", note="x" * 500, at="2026-10-28T09:00:00Z")
    assert list(record)[:7] == ["id", "title", "start", "end", "minutes", "outcome", "note"]
    assert record["minutes"] == 60.0 and record["outcome"] == "fixed" and len(record["note"]) == 200
    assert record["qwen_calls"] == 2 and record["qwen_minutes"] == pytest.approx(13.75)
    assert record["qwen_tokens"] == 68000 and record["est_saved_tokens"] == 3450
    est = (1500 * 4 + 1950 * 20) / 1e6
    assert record["est_saved_usd"] == pytest.approx(est, abs=1e-4)
    assert record["claude_turns"] == 2
    assert record["claude_usd"] == pytest.approx(M1_USD + M2_USD, abs=1e-4)
    assert record["claude_detail"]["cache_write_1h"] == 400
    assert record["net_usd"] == pytest.approx(est - M1_USD - M2_USD, abs=2e-4)
    assert line.startswith("[qwen task]") and "\n" not in line
    assert qwen_task.current_task() is None
    tasks = qwen_task.read_jsonl(qwen_task.tasks_file())
    assert len(tasks) == 1 and tasks[0]["id"] == task["id"]
    assert len(qwen_task.tasks_file().read_text(encoding="utf-8").strip().splitlines()[0]) < 1000
    with pytest.raises(RuntimeError):
        qwen_task.end("ok")
    s = qwen_task.summary(tasks, qwen_task.read_jsonl(qwen_task.calls_file()))
    assert s["tasks"] == 1 and s["outcomes"] == {"fixed": 1}
    assert s["qwen_calls"] == 3 and s["legacy_calls"] == 1
    assert s["qwen_minutes"] == pytest.approx(18.6)


def test_report_html(env):
    path = qwen_task.write_html([], [], env["stats"] / "empty.html")
    assert path.exists()
    qwen_task.start("plot me", at="2026-10-28T08:00:00Z")
    qwen_task.end("ok", at="2026-10-28T09:00:00Z")
    html = qwen_task.write_html(qwen_task.read_jsonl(qwen_task.tasks_file()), []).read_text(encoding="utf-8")
    assert "plot me" in html and "plotly" in html.lower()
    qwen_task.main(["report"])
    assert qwen_task.html_file().exists()
    qwen_task.main(["status"])
