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

## The ending (#134)

A build that stops without saying so reads as a build that hung, and
"finished" has four meanings here - the objective was met, the round cap was
hit, every task reached a terminal state, or a human is needed. `run_outcome`
is that answer: which ending it was, in the sentence the run itself recorded
(`cli._outcome` -> `summary.json`), with the counts, the clock and the tasks
that need a person named beside it.

**Read from the summary, not recomposed.** Every number here has a field in
`summary.json` behind it, which is what makes the panel and `swarm show`
answer the same question the same way - and what makes it survive a page
reload, a console restart, and the process that knew it. The console's other
account of an ending (`console_runs.RunJob.conclude`) is a live one: it reads
three regular expressions off a child's stdout, and can only ever speak for a
run this console process spawned while it is still in memory.

**The inference share is reported only when it was measured.** Nothing
increments `CycleMetrics.inference_s` yet - the field is recorded per cycle
and has no producer - so the honest answer today is that the run did not
measure it, and a page dividing zero by the wall clock to print "0%" would be
saying something false about the model instead.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from .artifacts import (
    EVENT_LOG_NAME,
    ArtifactsError,
    RunView,
    list_runs,
    load_run,
    read_events,
)
from .run import RunError

__all__ = ["ACTIVE_WITHIN_S", "latest_external", "run_outcome"]

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
        "repo_url": _repo_url(view.repo),
        "objective": view.objective,
        "state": state,
        "last_event_s": None if last_event_s is None else round(last_event_s, 1),
        "started_at": view.started_at.isoformat() if view.started_at else None,
        "note": view.note,
        "lines": lines[since:],
        "next": len(lines),
    }


def _repo_url(repo: str) -> str:
    """A GitHub link, or nothing. Built from the validated slug for
    `latest_external`'s reason: `run.json` is this system's file, and the page
    puts this string in an href."""
    return f"https://github.com/{repo}" if _REPO_RE.match(repo) else ""


def run_outcome(run_id: str = "", root: str | Path | None = None) -> dict[str, Any] | None:
    """How one run ended, for the terminal panel. `None` while it has not.

    `run_id` names the run; empty means the newest, which is what the external
    view has in hand. `None` for a run that never wrote a summary - it is
    either still going or it was killed, and both are `latest_external`'s
    `active`/`quiet` to report rather than an ending to draw.

    **Every number is read back, none is derived twice.** The counts come off
    `summary.json` - `outcome.merged` from the merge gate, `needs_human` from
    the result files, `outcome.abandoned` from the goal gate - so "the counts
    match the summary" is true by construction rather than by two computations
    agreeing. The three are kept apart on the page for the same reason they are
    kept apart here: a task that merged and a task waiting for a person are the
    two answers an operator is actually sorting for, and a single "8 tasks"
    hides the second behind the first.
    """
    view = _view(run_id, root)
    if view is None or not view.complete:
        return None

    outcome = view.outcome
    latest = view.results.latest
    needs_human = list(view.needs_human)
    # A pull request per task, opened by the worker and reported by its own
    # result record; the merged ones are a subset, which is why the page reads
    # "N opened, M merged" rather than adding them.
    opened = sum(1 for record in latest.values() if record.outcome == "pr-open")
    wall_s = None
    if view.started_at and view.finished_at:
        wall_s = round(max(0.0, (view.finished_at - view.started_at).total_seconds()), 1)
    metrics = view.metrics
    share = None
    if metrics.inference_calls and wall_s:
        share = round(min(metrics.inference_s / wall_s, 1.0), 3)

    return {
        "run_id": view.run_id,
        "repo": view.repo,
        "repo_url": _repo_url(view.repo),
        "objective": view.objective,
        # Text on the page, never an href: a browser will not follow a `file:`
        # link out of an `http:` document, so a link here would be a link that
        # does nothing. What the reader wants is the path to paste into a
        # terminal, next to `swarm show`.
        "path": str(view.path),
        "outcome": outcome.kind,
        "reason": outcome.reason,
        "note": view.note,
        "cycles": outcome.cycles or len(metrics.cycles),
        "cap": outcome.cap,
        "live": outcome.live,
        "tasks": {
            "merged": list(outcome.merged),
            "needs_human": needs_human,
            "abandoned": list(outcome.abandoned),
        },
        "prs": {"opened": opened, "merged": len(outcome.merged)},
        "attempts": view.results.attempts,
        "wall_s": wall_s,
        # Absent rather than zero when nothing measured it - see the module
        # docstring. `inference_s` rides along for a reader who wants the
        # seconds behind the share.
        "inference_s": round(metrics.inference_s, 1) if metrics.inference_calls else None,
        "inference_share": share,
        "finished_at": view.finished_at.isoformat() if view.finished_at else None,
    }


def _view(run_id: str, root: str | Path | None) -> RunView | None:
    """The run the caller asked for, or the newest when it asked for none.

    One directory read for a named run rather than `list_runs`, which builds a
    `RunView` per run - a results glob and an event-line count each - to answer
    a question about one of them. Both views that ask hold an id, so the
    listing is the exceptional path here rather than the usual one.

    A miss is `None` rather than a fallback to the newest: the caller that
    passes an id is a page holding one run's card, and answering it with a
    different run's ending is the one wrong answer this panel must not give.
    `load_run` refuses a malformed id before it becomes a path (`RunError`) and
    an absent one after (`ArtifactsError`); both are "no such run" here.
    """
    if not run_id:
        runs = list_runs(root)
        return runs[-1] if runs else None
    try:
        return load_run(run_id, root)
    except (ArtifactsError, RunError):
        return None
