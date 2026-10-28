# Task: write `qwen_task.py`, long-term statistics of the tasks delegated to the local Qwen

Write ONE file, `qwen_task.py`, in the working directory. The acceptance tests are in
`tests/test_qwen_task.py`; do not modify them. Run them with `python -m pytest -q tests`.
Reference code (read only): `C:/Users/SIM/.claude/tools/qwen_savings.py`, the existing per-call logger. Use
the same style: module docstring explaining the purpose, the formulas and the usage; English; small functions;
standard library only, plus plotly imported lazily inside `write_html`.

## Context

Claude (the main assistant) delegates work to Qwen through `ask_local.py` / `qwen_agent.py`. Each Qwen call
appends one JSON line to `CALLS_FILE` (written by `qwen_savings.log_event`, not by you). A call line looks like:

```json
{"time": "2026-10-28T09:30:00", "tool": "qwen_agent", "project": "x", "saved_in": 1000, "saved_out": 2000,
 "qwen_in": 50000, "qwen_out": 8000, "minutes": 12.5, "note": "...", "task": "20261028-083000"}
```

`task` is present only when a task was open; older lines have no `task` key ("legacy calls").

A *task* groups the Qwen calls of one piece of work, plus what Claude itself spent during that work (writing the
prompt, reviewing, fixing). Claude's real token usage is read from the Claude Code transcript: a JSON-lines file
where each line is an object; the lines with `"type": "assistant"` have
`"timestamp": "2026-10-28T08:25:23.322Z"` (UTC, ISO 8601, trailing Z) and `"message": {"id": "...",
"usage": {...}}`. The usage dict has `input_tokens`, `output_tokens`, `cache_read_input_tokens`,
`cache_creation_input_tokens` and, usually, `cache_creation: {"ephemeral_5m_input_tokens": a,
"ephemeral_1h_input_tokens": b}`. **One message is written on several lines with the same `message.id` and the
same usage: count each `message.id` once.** Lines of other types, or without usage, are ignored. Lines that are not
valid JSON are skipped.

## Paths (read at CALL time from the environment, never at import time, so tests can redirect them)

- `stats_dir()`: env `QWEN_STATS_DIR`, default `C:/Users/SIM/Desktop/Code-Adri/Qwen-Benchmark/stats`.
- `CALLS_FILE = stats_dir()/"qwen_usage.jsonl"`, `TASKS_FILE = stats_dir()/"tasks.jsonl"`,
  `CURRENT_FILE = stats_dir()/"current_task.json"`, `HTML_FILE = stats_dir()/"qwen_stats.html"`
  (implement as functions `calls_file()`, `tasks_file()`, `current_file()`, `html_file()`).
- `transcripts_dir()`: env `CLAUDE_TRANSCRIPTS_DIR`, default `~/.claude/projects`.

## Prices ($ per million tokens, Claude Opus 5.5 API)

```python
PRICES = {"input": 4.0, "output": 20.0, "cache_read": 0.4, "cache_write_5m": 5.0, "cache_write_1h": 8.0}
```

If a usage has `cache_creation_input_tokens` but no `cache_creation` dict, count those tokens as `cache_write_1h`.
Estimated savings of a call (same as qwen_savings): `(saved_in * 4.0 + saved_out * 20.0) / 1e6` dollars.

## Functions (exact names and signatures; the tests call them)

- `newest_transcript(root=None) -> Path | None`: newest (mtime) `*.jsonl` file in the subdirectories of
  `root` (default `transcripts_dir()`), pattern `root/*/*.jsonl`; None if there is none.
- `claude_usage(transcript, start, end) -> dict`: sums the usage of the assistant messages whose timestamp is in
  `[start, end]` (both `datetime`, timezone-aware UTC). Returns
  `{"turns": n_messages, "input": .., "output": .., "cache_read": .., "cache_write_5m": .., "cache_write_1h": ..,
  "tokens": sum of the five, "usd": cost with PRICES}`. `transcript=None` or a missing file -> all zeros.
- `current_task() -> dict | None`: content of `CURRENT_FILE`, or None.
- `start(title, at=None, transcript=None) -> dict`: opens a task. `at` = ISO string (with or without Z / offset;
  naive = UTC) or None = now (UTC). Raises `RuntimeError` if a task is already open. Writes `CURRENT_FILE` with
  `{"id": "YYYYMMDD-HHMMSS" of the start in UTC, "title": title, "start": ISO UTC string ending in "Z",
  "transcript": str(transcript or newest_transcript()) or None}` and returns that dict. Creates the stats dir.
- `end(outcome, note="", at=None) -> tuple[dict, str]`: closes the open task (`RuntimeError` if none). `outcome`
  must be one of `OUTCOMES = ("ok", "fixed", "failed", "abandoned")`, else `ValueError`:
  ok = Qwen's work used as is, fixed = used after Claude's corrections, failed = Claude redid it, abandoned.
  It collects the calls of `CALLS_FILE` whose `task` equals the id, measures `claude_usage` on the transcript
  recorded at start between start and `at` (default now), appends ONE short JSON line to `TASKS_FILE`, deletes
  `CURRENT_FILE`, and returns `(record, line)`. The record keys, in this order:
  `id, title, start, end, minutes` (wall clock, 1 decimal), `outcome, note` (note cut to 200 chars),
  `qwen_calls, qwen_minutes` (sum, 2 decimals), `qwen_tokens` (sum of qwen_in + qwen_out),
  `est_saved_tokens` (sum of saved_in + saved_out), `est_saved_usd` (4 decimals),
  `claude_turns, claude_tokens, claude_usd` (4 decimals), `claude_detail` (dict: input, output, cache_read,
  cache_write_5m, cache_write_1h), `net_usd = est_saved_usd - claude_usd` (4 decimals).
  `line` is a one-line text starting with `"[qwen task]"` giving title, outcome, Qwen calls, Qwen minutes, Qwen
  tokens, Claude tokens and $, estimated saving $, net $, and the cumulative net $ and number of tasks over
  `TASKS_FILE`.
- `read_jsonl(path) -> list[dict]`: lines of a JSON-lines file, skipping invalid lines; [] if missing.
- `summary(tasks, calls) -> dict`: `{"tasks": n, "outcomes": {outcome: count}, "qwen_calls": total calls in
  calls, "legacy_calls": calls without "task", "qwen_minutes": total minutes of all calls,
  "qwen_tokens": .., "claude_usd": sum over tasks, "est_saved_usd": sum over ALL calls,
  "net_usd": sum of net_usd over tasks}`.
- `write_html(tasks, calls, path=None) -> Path`: self-contained interactive plotly page (`include_plotlyjs="cdn"`),
  French titles, `template="plotly_white"`. Panels (make_subplots):
  1. cumulative number of tasks and of Qwen calls against time;
  2. per task (x = title, short): Qwen minutes (bar) and Claude tokens split input / output / cache read /
     cache write (stacked bars, secondary axis or separate row: your choice, keep it readable);
  3. cumulative $ against time: estimated saving (all calls), Claude cost (tasks), net (tasks);
  4. number of tasks per outcome;
  then a `go.Table` row listing every task (date, title, outcome, Qwen calls, Qwen min, Claude tokens, Claude $,
  estimated $, net $). Must work with zero tasks (only legacy calls) and with zero calls.
- `main(argv=None)`: argparse CLI with sub-commands
  `start TITLE [--at ISO]`, `end --outcome X [--note TEXT] [--at ISO]`, `status` (prints the open task or
  "no open task"), `report [--no-html]` (prints the summary in 3-6 lines, writes the HTML unless --no-html and
  prints its path). Print with `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` in main.

Keep the file under ~300 lines. When the tests pass, stop and write a short report.
