"""The external-run view: runs the console did not start, read from artifacts.

What is pinned: the answer is built from the same files `swarm runs` reads
(run.json, events.jsonl, summary.json), so it exists for CLI-launched runs,
for runs that outlived a console restart, and for runs that died. Liveness is
stated honestly - `finished` only on summary.json evidence, `active` only on
a demonstrably fresh event, `quiet` for the ambiguity in between. The
`since`/`run` contract keeps incremental polling correct when a newer run
takes the latest slot. And the console's own jobs record their swarm run id,
because the page must not show one run twice.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from swarm.console import Console
from swarm.console_external import ACTIVE_WITHIN_S, latest_external

HOST = {"Host": "127.0.0.1:8117"}


def write_run(
    root: Path,
    run_id: str,
    *,
    repo: str = "me/thing",
    cycles: int = 2,
    finished: bool = False,
    event_age_s: float = 0.0,
) -> Path:
    d = root / run_id
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({
        "run_id": run_id, "repo": repo, "objective": "make the thing",
        "started_at": "2026-08-18T10:00:00+00:00",
    }))
    events = [json.dumps({"event": "run.started", "run": run_id, "repo": repo})]
    events += [
        json.dumps({
            "event": "cycle.reconciled", "run": run_id, "cycle": i,
            "summary": f"cycle {i}: dispatched 1",
            "gate": [f"#{i}: passed"] if i else [],
            "failures": ["merge refused - checks pending"] if i == 1 else [],
        })
        for i in range(cycles)
    ]
    (d / "events.jsonl").write_text("\n".join(events) + "\n")
    if finished:
        (d / "summary.json").write_text(json.dumps({
            "run_id": run_id, "repo": repo, "objective": "make the thing",
            "started_at": "2026-08-18T10:00:00+00:00",
            "finished_at": "2026-08-18T10:30:00+00:00",
        }))
    if event_age_s:
        stamp = time.time() - event_age_s
        os.utime(d / "events.jsonl", (stamp, stamp))
    return d


def test_nothing_ever_ran_is_none_not_an_error(tmp_path):
    assert latest_external(tmp_path) is None


def test_the_latest_run_is_read_with_its_cycle_log(tmp_path):
    write_run(tmp_path, "demo-20260818-100000-aaaaaa")
    write_run(tmp_path, "demo-20260818-110000-bbbbbb", cycles=2)

    latest = latest_external(tmp_path)

    assert latest["run_id"] == "demo-20260818-110000-bbbbbb"  # newest, by id order
    assert latest["repo"] == "me/thing"
    assert latest["repo_url"] == "https://github.com/me/thing"
    assert latest["lines"][0] == "· cycle 0: dispatched 1"
    assert "    #1: passed" in latest["lines"]                # gate verdicts shown
    assert "  ! merge refused - checks pending" in latest["lines"]
    assert latest["next"] == len(latest["lines"])


def test_liveness_is_stated_not_guessed(tmp_path):
    write_run(tmp_path, "a-20260818-100000-aaaaaa", finished=True)
    assert latest_external(tmp_path)["state"] == "finished"

    write_run(tmp_path, "a-20260818-110000-bbbbbb", event_age_s=1.0)
    assert latest_external(tmp_path)["state"] == "active"

    write_run(tmp_path, "a-20260818-120000-cccccc", event_age_s=ACTIVE_WITHIN_S * 10)
    quiet = latest_external(tmp_path)
    assert quiet["state"] == "quiet"                 # alive-or-dead is not knowable
    assert quiet["last_event_s"] >= ACTIVE_WITHIN_S


def test_since_slices_and_a_stale_run_counter_resets_to_zero(tmp_path):
    write_run(tmp_path, "a-20260818-100000-aaaaaa", cycles=3)

    page = latest_external(tmp_path, since=2)
    assert page["next"] == len(page["lines"]) + 2    # slice, not the whole log

    # A newer run takes the latest slot; the old counter belongs to nobody.
    write_run(tmp_path, "a-20260818-110000-bbbbbb", cycles=1)
    fresh = latest_external(tmp_path, since=99, run_id="a-20260818-100000-aaaaaa")
    assert fresh["run_id"] == "a-20260818-110000-bbbbbb"
    assert fresh["lines"], "a stale counter against a new run must restart from line zero"


def test_an_unslug_repo_gets_no_url(tmp_path):
    """run.json is this system's file, but the page puts the value in an href."""
    write_run(tmp_path, "a-20260818-100000-aaaaaa", repo="not a slug")

    assert latest_external(tmp_path)["repo_url"] == ""


def test_the_route_serves_it_and_404s_when_nothing_ran(tmp_path, monkeypatch):
    monkeypatch.setenv("APIARY_ARTIFACTS_DIR", str(tmp_path / "empty"))
    console = Console()
    assert console.render("GET", "/swarm/external", HOST).status == 404

    write_run(tmp_path / "empty", "demo-20260818-100000-aaaaaa")
    body = json.loads(console.render("GET", "/swarm/external?since=0", HOST).body)
    assert body["run_id"] == "demo-20260818-100000-aaaaaa"


def test_a_console_job_records_its_swarm_run_id_for_deduplication(monkeypatch):
    """The external view reads artifacts, and the console's own runs write
    artifacts too - without the id captured from the log, the page would show
    every console run twice: once from memory, once from disk."""
    from swarm.console_runs import RunJob

    job = RunJob(id="x", command="swarm run", started=0.0)
    job.absorb("» run wallet-20260818-090333-aeek6b  repo me/thing  objective: x")

    assert job.progress["run_id"] == "wallet-20260818-090333-aeek6b"
    assert job.progress["repo"] == "me/thing"
