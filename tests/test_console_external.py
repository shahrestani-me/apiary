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
from swarm.console_external import ACTIVE_WITHIN_S, latest_external, run_outcome

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


# --------------------------------------------------------------------------
# How the run ended (#134)
# --------------------------------------------------------------------------


def finished_run(
    root: Path,
    run_id: str = "demo-20260818-100000-aaaaaa",
    *,
    repo: str = "me/thing",
    outcome: dict | None = None,
    results: tuple[tuple[int, int], ...] = ((7, 0), (8, 1)),
    inference_calls: int = 0,
) -> Path:
    """A run directory with a summary, written the way a real run writes one.

    `results` is `(issue, exit code)` per task: exit 0 is a worker that opened a
    pull request, anything else is a task that stopped needing a person.
    """
    from swarm.worker.result import ResultRecord, write_result

    d = write_run(root, run_id, repo=repo, finished=True)
    for issue, exit_code in results:
        write_result(
            ResultRecord(run_id=run_id, issue=issue, attempt=1, exit_code=exit_code,
                         reason="the verify command failed" if exit_code else "opened",
                         repo=repo),
            d / "results",
        )
    summary = json.loads((d / "summary.json").read_text())
    summary["outcome"] = {
        "kind": "met", "reason": "objective met: every task is verified and merged",
        "cycles": 6, "cap": 40, "live": 0, "merged": [7], "abandoned": [],
        **(outcome or {}),
    }
    summary["metrics"] = {"cycles": [
        {"cycle": i, "inference_calls": inference_calls, "inference_s": 90.0}
        for i in range(summary["outcome"]["cycles"])
    ]}
    (d / "summary.json").write_text(json.dumps(summary))
    return d


def test_the_ending_is_the_sentence_the_run_recorded(tmp_path):
    """Not a rewording of it. `close_the_loop` and the goal gate compose these
    sentences, the run records the one it ended on, and the page quotes it -
    which is what stops "objective met" and "stopped after 40 cycles" from
    both rendering as "done"."""
    finished_run(tmp_path)

    ended = run_outcome(root=tmp_path)

    assert ended["outcome"] == "met"
    assert ended["reason"] == "objective met: every task is verified and merged"
    assert ended["cycles"] == 6 and ended["cap"] == 40


@pytest.mark.parametrize(
    "kind, reason",
    [
        ("met", "objective met: every task is verified and merged"),
        ("capped", "stopped after 40 cycle(s) with 2 live issue(s)"),
        ("exhausted", "stopped after 4 cycle(s) with 0 live issue(s)"),
        ("failed", "stopping without meeting the objective: #8 was abandoned"),
    ],
)
def test_each_of_the_four_endings_arrives_distinctly(tmp_path, kind, reason):
    """The criterion, one ending at a time: the page has to be able to tell
    them apart, and the two that exit 0 - met and capped - are the pair the
    exit code never could."""
    finished_run(tmp_path, outcome={"kind": kind, "reason": reason})

    ended = run_outcome(root=tmp_path)

    assert ended["outcome"] == kind
    assert ended["reason"] == reason


def test_the_counts_match_the_summary_and_keep_merged_apart_from_needs_human(tmp_path):
    """Both halves of the criterion. Every number is read back off
    `summary.json` rather than recomputed, and the task waiting for a person is
    listed separately from the ones that merged - folded into one total, it is
    the first thing to disappear."""
    directory = finished_run(
        tmp_path,
        outcome={"kind": "failed", "reason": "stopping without meeting the objective",
                 "merged": [7], "abandoned": ["add-retry"]},
        results=((7, 0), (8, 1)),
    )
    summary = json.loads((directory / "summary.json").read_text())

    ended = run_outcome(root=tmp_path)

    assert ended["tasks"]["merged"] == summary["outcome"]["merged"] == [7]
    assert ended["tasks"]["needs_human"] == [8]        # exit 1: no pull request open
    assert ended["tasks"]["abandoned"] == ["add-retry"]
    # One pull request per task: #7's is open and merged, #8 never opened one.
    assert ended["prs"] == {"opened": 1, "merged": 1}


def test_the_wall_clock_is_reported_and_the_inference_share_only_when_measured(tmp_path):
    """Nothing increments `CycleMetrics.inference_s` yet. A page dividing an
    unmeasured field by the wall clock to print "0% inference" would be making
    a claim about the model instead of about the recording."""
    finished_run(tmp_path)

    ended = run_outcome(root=tmp_path)

    assert ended["wall_s"] == 1800.0                   # 10:00 -> 10:30
    assert ended["inference_share"] is None
    assert ended["inference_s"] is None

    finished_run(tmp_path, "demo-20260818-110000-bbbbbb", inference_calls=3)

    measured = run_outcome(root=tmp_path)

    assert measured["inference_share"] == round(6 * 90.0 / 1800.0, 3)


def test_a_run_that_has_not_ended_has_no_ending_to_draw(tmp_path):
    """404 on the route, and `None` here. A run still going is
    `latest_external`'s `active`/`quiet` to report - drawing a terminal state
    for it is the "the build hung" reading, backwards."""
    write_run(tmp_path, "demo-20260818-100000-aaaaaa")     # no summary.json

    assert run_outcome(root=tmp_path) is None


def test_a_named_run_is_never_answered_with_a_different_run_s_ending(tmp_path):
    """Both callers hold one run's card. A miss is `None` rather than a
    fallback to the newest, which would put the wrong ending under the right
    run id at exactly the moment a second run starts."""
    finished_run(tmp_path, "demo-20260818-100000-aaaaaa")
    finished_run(tmp_path, "demo-20260818-110000-bbbbbb",
                 outcome={"kind": "capped", "reason": "stopped after 40 cycle(s)"})

    assert run_outcome("demo-20260818-100000-aaaaaa", root=tmp_path)["outcome"] == "met"
    assert run_outcome("demo-20260818-110000-bbbbbb", root=tmp_path)["outcome"] == "capped"
    assert run_outcome("demo-20260818-999999-zzzzzz", root=tmp_path) is None


def test_the_route_serves_the_ending_and_404s_until_there_is_one(tmp_path, monkeypatch):
    monkeypatch.setenv("APIARY_ARTIFACTS_DIR", str(tmp_path))
    console = Console()

    write_run(tmp_path, "demo-20260818-100000-aaaaaa")
    assert console.render("GET", "/swarm/outcome?run=demo-20260818-100000-aaaaaa",
                          HOST).status == 404

    finished_run(tmp_path, "demo-20260818-110000-bbbbbb")
    body = json.loads(console.render(
        "GET", "/swarm/outcome?run=demo-20260818-110000-bbbbbb", HOST).body)

    assert body["outcome"] == "met"
    assert body["repo_url"] == "https://github.com/me/thing"
    assert body["path"].endswith("demo-20260818-110000-bbbbbb")
