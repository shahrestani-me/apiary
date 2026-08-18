"""Runs the console did not start, made visible on the page anyway.

The run panel used to show only the jobs this console process spawned, because
that is all `SwarmRuns` can know: its jobs live in its memory. But a run
started from a terminal - or from a console that has since been restarted - is
just as real, and the operator watching the page had no account of it beyond
the board's label changes. The question "is an orchestrator working right now,
and on what?" deserved an answer that does not depend on which shell happened
to exec it.

**The artifacts are that answer, and they already exist.** Every run, however
launched, writes `.swarm/runs/<run-id>/` - `run.json` at startup,
`events.jsonl` per cycle, `summary.json` at a clean exit - and `swarm runs` /
`swarm show` have always read runs back from exactly these files. This module
is those readers pointed at the page: the latest run directory, its identity,
and its cycle log streamed with the same `since`/`next` contract the console's
own jobs use. No IPC, no pidfiles, no second registry - a run that died still
shows what it did, which is precisely when someone is looking.

**Liveness is stated honestly, never guessed.** A `summary.json` proves a run
finished. Without one, the run is either alive or was killed - and the event
log cannot tell those apart, because a cycle blocks on model calls that were
measured at seven minutes. So the page gets the facts and a vocabulary that
admits the ambiguity: `finished`, `active` (an event landed within the last
two cycles' worth of seconds), or `quiet` with the age attached - a long
model call, or a dead process, and the next event settles which.

**What the log shows is the run's own recorded account**, not its stdout: the
per-cycle summary plus the gate verdicts and failures `cli._report_cycle`
files. Startup lines (the provision report, the plan) are not in the event
log and are not invented here.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from .artifacts import EVENT_LOG_NAME, list_runs, read_events

__all__ = ["ACTIVE_WITHIN_S", "latest_external"]

#: An event younger than this means an orchestrator is demonstrably cycling:
#: cycles start every ~15s, so two cycles of silence is the line between "the
#: loop is running" and "something long, or nothing, is happening".
ACTIVE_WITHIN_S = 45

#: Same shape `console_board` validates before building any GitHub URL: these
#: strings end up in hrefs, and `run.json`'s repo field is still input.
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _lines(directory: Path) -> list[str]:
    """The cycle log as the terminal would have shown it, one string per row."""
    rows: list[str] = []
    for event in read_events(directory / EVENT_LOG_NAME):
        if summary := event.get("summary"):
            rows.append(f"· {summary}")
        for verdict in event.get("gate") or ():
            rows.append(f"    {verdict}")
        for failure in event.get("failures") or ():
            rows.append(f"  ! {failure}")
        if goal := event.get("goal"):
            rows.append(f"» {goal}")
    return rows


def latest_external(
    root: str | Path | None = None, *, since: int = 0, run_id: str = ""
) -> dict[str, Any] | None:
    """The newest recorded run under `root`, or None when nothing ever ran.

    Newest by artifacts order (`list_runs` is oldest-first by id, and ids sort
    chronologically by construction - `run.py` put the UTC stamp in them for
    exactly this). The whole answer is built from files, so it is equally true
    for a run that is mid-flight, finished, or twelve hours dead.

    `since` slices the log for incremental polling, and `run_id` names the run
    the caller's counter belongs to: when a newer run has taken the latest
    slot, the offset is someone else's and the answer starts from line zero.
    """
    runs = list_runs(root)
    if not runs:
        return None
    view = runs[-1]
    if run_id and run_id != view.run_id:
        since = 0

    lines = _lines(view.path)
    log = view.path / EVENT_LOG_NAME
    try:
        last_event_s = max(0.0, time.time() - log.stat().st_mtime)
    except OSError:
        last_event_s = None

    if view.complete:
        state = "finished"
    elif last_event_s is not None and last_event_s <= ACTIVE_WITHIN_S:
        state = "active"
    else:
        # Alive inside a long model call, or killed - the files cannot tell,
        # so neither does this. The age lets the reader judge.
        state = "quiet"

    return {
        "run_id": view.run_id,
        "repo": view.repo,
        # Built from the validated slug, never trusted from disk: run.json is
        # written by this system, but the page puts this string in an href.
        "repo_url": f"https://github.com/{view.repo}" if _REPO_RE.match(view.repo) else "",
        "objective": view.objective,
        "state": state,
        "last_event_s": None if last_event_s is None else round(last_event_s, 1),
        "started_at": view.started_at.isoformat() if view.started_at else None,
        "note": view.note,
        "lines": lines[since:],
        "next": len(lines),
    }
