"""The four claims #29 makes about a run directory, each provoked.

- **A completed run is understandable from its directory alone.** No GitHub
  call, no daemon, no orchestrator: the tests below build a directory and then
  read it back through `load_run`, which knows nothing but a path and an id.
- **A killed run is understandable too.** The orchestrator that matters most is
  the one that died, and it wrote no summary. Its cycles come back out of the
  event log, which is the whole reason that log is structured.
- **Wall clock does not diagnose a slow run.** Two runs here burn identical
  inference seconds; one was swapping weights and one was not, and the artifacts
  say which.
- **No artifact contains a credential.** Asserted in both directions: the
  auditor finds nothing in a directory this module wrote, and does find the same
  token when it is planted by hand - an empty list from a scanner that never
  matches anything is not a security property.

The orphan-log path is driven through the real `Reaper` and a scripted `docker`
rather than by calling `container_log` directly, because the claim is that
`log_sink` fits the socket #20 left for it. A hand-called writer would pass even
if the signature had drifted.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pytest

from swarm.artifacts import (
    HOST_ROOT_ENV,
    PR_CHECKS,
    PR_MERGED,
    PR_OPENED,
    TASK_CLAIMED,
    TASK_ELIGIBLE,
    TASK_LANDED,
    TASK_NEEDS_HUMAN,
    TASK_RESULT,
    host_path,
    EVENT_LOG_NAME,
    RESULTS_DIR_NAME,
    SUMMARY_FILE_NAME,
    ArtifactsError,
    CycleMetrics,
    RunArtifacts,
    RunMetrics,
    artifacts_root,
    list_runs,
    load_run,
    read_events,
    read_run,
    runs_text,
    show_text,
)
from swarm.containers.manager import DockerCLI, Handle, Redactor
from swarm.containers.reaper import Reaper
from swarm.run import Run, RunError
from swarm.security import assert_unprivileged, scan_artifacts
from swarm.worker.entrypoint import EXIT_INFRASTRUCTURE, EXIT_OK, EXIT_TASK_FAILED
from swarm.worker.result import DEFAULT_RESULT_DIR, RESULT_DIR_ENV, ResultRecord, write_result

REPO = "shahrestani-me/apiary"
OTHER_REPO = "shahrestani-me/other"
RUN_ID = "apiary-20260814-142530-k3f9qz"
DEAD_RUN_ID = "apiary-20260814-140000-aaaaaa"
OBJECTIVE = "add retry to the client"

#: Not token-shaped on purpose. `SECRET_PATTERNS` cannot see this one, so
#: anything that catches it caught it by enrolment - which is what proves the
#: artifacts writer redacts on its own rather than trusting whoever handed it
#: the string.
SECRET = "hunter2-not-a-recognisable-token-shape"

ORCHESTRATOR_MODEL = "gemma4:31b"
WORKER_MODEL = "gemma4:26b"

CONTAINER_ID = "c0ffee" + "0" * 58


# --------------------------------------------------------------------------
# Fixtures and doubles
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def credential(monkeypatch: pytest.MonkeyPatch) -> str:
    """A token in the environment, exactly as the orchestrator runs with one."""
    monkeypatch.setenv("GITHUB_TOKEN", SECRET)
    return SECRET


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def a_run(
    *,
    run_id: str = RUN_ID,
    repo: str = REPO,
    objective: str = OBJECTIVE,
    now: dt.datetime | None = None,
) -> Run:
    return Run.start(repo, objective, run_id=run_id, now=now or _at("14:25:30"))


def _at(clock: str) -> dt.datetime:
    hour, minute, second = (int(part) for part in clock.split(":"))
    return dt.datetime(2026, 8, 14, hour, minute, second, tzinfo=dt.timezone.utc)


def a_record(**overrides: Any) -> ResultRecord:
    fields: dict[str, Any] = {
        "run_id": RUN_ID,
        "issue": 7,
        "attempt": 1,
        "exit_code": EXIT_OK,
        "verify_command": "python -m pytest -q",
        "verify_output": "1 passed",
        "reason": "verified and committed",
        "repo": REPO,
    }
    return ResultRecord(**{**fields, **overrides})


def a_handle(**overrides: Any) -> Handle:
    fields: dict[str, Any] = {
        "id": CONTAINER_ID,
        "run_id": RUN_ID,
        "issue": 7,
        "name": "apiary-worker",
        "image": "apiary-worker",
        "captured": "» issue #7\n  · verifying: python -m pytest -q\n",
    }
    return Handle(**{**fields, **overrides})


@dataclass
class ScriptedDocker:
    """A `Runner` keyed by docker subcommand. Enough daemon for one sweep.

    A local copy rather than an import from `test_container_manager.py`: sharing
    a double couples two suites that should be free to move separately, and only
    `ps`, `logs` and `rm` are exercised here.
    """

    replies: dict[str, Any] = field(default_factory=dict)
    calls: list[list[str]] = field(default_factory=list)

    def __call__(
        self, argv: Sequence[str], *, timeout_s: float | None, merge: bool
    ) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        return subprocess.CompletedProcess(list(argv), 0, self.replies.get(argv[1], ""), "")


def ps_line(container_id: str, run_id: str, issue: int, state: str = "exited") -> str:
    """One row in `_PS_FORMAT`: id, name, image, run label, issue label, state.

    `exited` by default: an orphan is a container nobody was left to wait on,
    and this sink exists to write down what it said before it goes.
    """
    return f"{container_id}\tapiary-worker\tapiary-worker\t{run_id}\t{issue}\t{state}\n"


# --------------------------------------------------------------------------
# A run, understood from its directory
# --------------------------------------------------------------------------


def test_a_completed_run_is_understandable_from_its_directory_alone(root):
    """The ticket, end to end. Nothing below this line consults GitHub."""
    run = a_run()
    with RunArtifacts.open(run, root=root) as artifacts:
        with artifacts.cycle(1) as cycle:
            cycle.api_call(11)
            cycle.inference(WORKER_MODEL, seconds=94.0, load_s=6.7, queued=2)
        artifacts.container_log(a_handle())
        write_result(a_record(issue=7, attempt=1, exit_code=EXIT_OK), artifacts.results_dir)
        write_result(
            a_record(issue=8, attempt=2, exit_code=EXIT_TASK_FAILED, reason="the verify command failed"),
            artifacts.results_dir,
        )

    view = load_run(RUN_ID, root)

    assert view.complete
    assert (view.repo, view.objective) == (REPO, OBJECTIVE)
    assert view.results.attempts == 2 and view.results.consumed == 2
    # The question the ticket says is the interesting one.
    assert view.needs_human == (8,)
    assert view.metrics.api_calls == 11 and view.metrics.model_swaps == 1
    assert [path.name for path in view.container_logs] == ["issue-7-c0ffee000000.log"]
    assert "the verify command failed" in show_text(view)


def test_a_killed_run_still_reports_its_cycles(root):
    """No `summary.json`, because nothing reached the end. The log is the record."""
    artifacts = RunArtifacts.open(a_run(), root=root)
    with artifacts.cycle(1) as cycle:
        cycle.api_call(4)
        cycle.inference(ORCHESTRATOR_MODEL, seconds=8.0, load_s=6.7)
    with artifacts.cycle(2) as cycle:
        cycle.api_call(6)
    # And then the process is gone: no `finish`, no summary.

    view = read_run(artifacts.path)

    assert not (artifacts.path / SUMMARY_FILE_NAME).exists()
    assert not view.complete
    assert [cycle.cycle for cycle in view.metrics.cycles] == [1, 2]
    assert view.metrics.api_calls == 10
    assert "did not reach its end" in show_text(view)


def test_a_cycle_that_raised_still_leaves_its_numbers(root):
    """The cycle that blew up is the one whose cost is worth having."""
    artifacts = RunArtifacts.open(a_run(), root=root)

    with pytest.raises(ZeroDivisionError):
        with artifacts.cycle(1) as cycle:
            cycle.api_call(3)
            raise ZeroDivisionError("the reconciler fell over")

    assert artifacts.metrics().api_calls == 3
    assert RunMetrics.from_events(read_events(artifacts.events_path)).api_calls == 3


def test_leaving_the_context_manager_records_why(root):
    """A summary that says nothing about why the run stopped is the thing to avoid."""
    with pytest.raises(KeyboardInterrupt):
        with RunArtifacts.open(a_run(), root=root):
            raise KeyboardInterrupt

    view = load_run(RUN_ID, root)

    assert view.complete and "KeyboardInterrupt" in view.note


# --------------------------------------------------------------------------
# The event log
# --------------------------------------------------------------------------


def test_every_event_is_one_json_object_on_one_line(root):
    artifacts = RunArtifacts.open(a_run(), root=root)
    artifacts.event("dispatch.claimed", issue=7, files=["src/thing.py"])

    lines = (artifacts.path / EVENT_LOG_NAME).read_text().splitlines()
    events = [json.loads(line) for line in lines]

    assert [event["event"] for event in events] == ["run.started", "dispatch.claimed"]
    assert all(event["run"] == RUN_ID and event["ts"] for event in events)
    # Nested structure survives, because a replay reads fields and not prose.
    assert events[-1]["files"] == ["src/thing.py"]


def test_one_unreadable_line_does_not_cost_the_others(root, capsys):
    """A `SIGKILL`ed orchestrator's last line is expected to be half-written."""
    artifacts = RunArtifacts.open(a_run(), root=root)
    artifacts.event("cycle.started", cycle=1)
    with artifacts.events_path.open("a") as handle:
        handle.write('{"event": "cycle.finish')

    events = read_events(artifacts.events_path)

    assert [event["event"] for event in events] == ["run.started", "cycle.started"]
    assert "skipping events.jsonl:3" in capsys.readouterr().err


# --------------------------------------------------------------------------
# What explains a slow run
# --------------------------------------------------------------------------


def test_swap_time_tells_a_bad_model_from_a_swapping_host(root):
    """Same wall clock, two diagnoses. This is why the counters exist.

    Both runs spend 200 seconds inside Ollama. In one of them the host read
    weights off disk for 80 of those seconds; in the other it generated tokens
    for all 200. A run report that only knows the total says the same thing
    about both, and the fixes are different (`config.py`: stop swapping between
    two models, or change the model).
    """
    swapping = RunArtifacts.open(a_run(), root=root)
    with swapping.cycle(1) as cycle:
        for _ in range(4):
            cycle.inference(ORCHESTRATOR_MODEL, seconds=25.0, load_s=10.0)
            cycle.inference(WORKER_MODEL, seconds=25.0, load_s=10.0)

    thinking = RunArtifacts.open(a_run(run_id="apiary-20260814-150000-b2c3d4"), root=root)
    with thinking.cycle(1) as cycle:
        for _ in range(8):
            cycle.inference(WORKER_MODEL, seconds=25.0)

    assert swapping.metrics().inference_s == thinking.metrics().inference_s == 200.0
    assert (swapping.metrics().model_swaps, swapping.metrics().swap_s) == (8, 80.0)
    assert (thinking.metrics().model_swaps, thinking.metrics().swap_s) == (1, 0.0)
    assert swapping.metrics().swap_share == 0.4 and thinking.metrics().swap_share == 0.0
    assert "the host was swapping, not thinking" in swapping.metrics().text()
    assert "swapping, not thinking" not in thinking.metrics().text()


def test_a_swap_across_a_cycle_boundary_is_still_a_swap(root):
    """`OLLAMA_MAX_LOADED_MODELS=1` does not reset when a cycle does.

    Cycle 1 leaves the worker model resident. Cycle 2's first call asks for the
    orchestrator model, which costs a swap that a per-cycle counter starting
    from nothing would attribute to no one.
    """
    artifacts = RunArtifacts.open(a_run(), root=root)
    with artifacts.cycle(1) as first:
        first.inference(WORKER_MODEL, seconds=30.0)
        first.inference(WORKER_MODEL, seconds=30.0)
    with artifacts.cycle(2) as second:
        second.inference(ORCHESTRATOR_MODEL, seconds=8.0, load_s=6.7)
        second.inference(ORCHESTRATOR_MODEL, seconds=8.0)

    first, second = artifacts.metrics().cycles
    assert (first.model_swaps, second.model_swaps) == (1, 1)
    assert second.swap_s == 6.7


def test_queue_depth_is_the_peak_and_api_calls_are_per_cycle(root):
    """Two containers against two slots is the design; six waiting is the story."""
    artifacts = RunArtifacts.open(a_run(), root=root)
    with artifacts.cycle(1) as cycle:
        cycle.api_call(2)
        cycle.queue(2)
        cycle.queue(6)
        cycle.queue(1)
    with artifacts.cycle(2) as cycle:
        cycle.api_call(40)

    metrics = artifacts.metrics()
    assert metrics.peak_queue_depth == 6
    assert [cycle.api_calls for cycle in metrics.cycles] == [2, 40]
    # The cycle to look at, named rather than left to be spotted in a table.
    assert metrics.busiest_cycle is not None and metrics.busiest_cycle.cycle == 2
    assert "busiest cycle 2" in metrics.text()


def test_a_cycle_record_round_trips_through_the_event_log(root):
    original = CycleMetrics(cycle=3, api_calls=7, model_swaps=2, swap_s=13.4, peak_queue_depth=4)

    restored = CycleMetrics.from_dict(original.to_dict())

    assert restored.to_dict() == original.to_dict()


# --------------------------------------------------------------------------
# Container logs
# --------------------------------------------------------------------------


def test_a_reaped_orphans_log_lands_under_the_run_that_produced_it(root):
    """#20's `sink` is this module's writer, and an orphan is another run's.

    Filing a dead run's evidence under the live run's directory is exactly how
    "which run went wrong" stops being answerable, so the sink routes on the
    container's own `apiary.run` label.
    """
    artifacts = RunArtifacts.open(a_run(), root=root)
    docker = ScriptedDocker(
        replies={
            "ps": ps_line(CONTAINER_ID, DEAD_RUN_ID, 12),
            "logs": "» issue #12\n  · cloning\n",
        }
    )
    reaper = Reaper(
        docker=DockerCLI(redact=Redactor(), runner=docker),
        run=artifacts.run,
        sink=artifacts.log_sink,
    )

    sweep = reaper.sweep()

    assert sweep.ok and sweep.run_ids == (DEAD_RUN_ID,)
    orphan_log = root / DEAD_RUN_ID / "logs" / "issue-12-c0ffee000000.log"
    assert "cloning" in orphan_log.read_text()
    # And this run's own directory is not where it went.
    assert list(artifacts.logs_dir.iterdir()) == []


def test_two_attempts_at_one_issue_keep_both_logs(root):
    """A retry must not overwrite the log that says why there was a retry."""
    artifacts = RunArtifacts.open(a_run(), root=root)

    artifacts.container_log(a_handle(id="a" * 64, captured="first attempt"))
    artifacts.container_log(a_handle(id="b" * 64, captured="second attempt"))

    assert sorted(path.name for path in artifacts.logs_dir.iterdir()) == [
        "issue-7-aaaaaaaaaaaa.log",
        "issue-7-bbbbbbbbbbbb.log",
    ]


def test_a_container_nobody_captured_says_so(root):
    """`captured is None` means the ordering was not followed, which is worse
    than an empty log and must not read like one."""
    artifacts = RunArtifacts.open(a_run(), root=root)

    path = artifacts.container_log(a_handle(captured=None))

    assert "nothing was captured" in path.read_text()


# --------------------------------------------------------------------------
# Redaction, in both directions
# --------------------------------------------------------------------------


def test_no_artifact_carries_a_credential(root, credential):
    """Every writer here, fed the token, and then the whole tree audited.

    The token is not token-shaped, so `SECRET_PATTERNS` cannot save this: only
    enrolment by variable name catches it, which is the mechanism
    `containers/manager.py` already owns and this module reuses rather than
    reimplementing.
    """
    run = Run.start(REPO, f"deploy with {credential}", run_id=RUN_ID)
    with RunArtifacts.open(run, root=root) as artifacts:
        artifacts.event("dispatch.failed", detail=f"remote rejected: {credential}", nested={"t": credential})
        artifacts.container_log(a_handle(captured=f"fatal: authentication failed for {credential}"))
        with artifacts.cycle(1) as cycle:
            cycle.api_call()

    tree = "\n".join(
        path.read_text() for path in sorted(root.rglob("*")) if path.is_file()
    )
    assert credential not in tree
    assert "***" in tree
    assert artifacts.audit() == []


def test_the_audit_is_not_a_scanner_that_never_matches(root, credential):
    """The control for the test above: planted by hand, the same token is found."""
    artifacts = RunArtifacts.open(a_run(), root=root)
    (artifacts.path / "pasted-by-a-human.txt").write_text(f"GITHUB_TOKEN={credential}\n")

    leaks = artifacts.audit()

    assert [leak.path.name for leak in leaks] == ["pasted-by-a-human.txt"]
    # A leak report that quotes the leak is a second copy of it.
    assert credential not in str(leaks[0])


def test_a_worker_written_result_is_audited_with_everything_else(root, credential):
    """Results arrive from inside a container, which is where the token is."""
    artifacts = RunArtifacts.open(a_run(), root=root)
    write_result(a_record(verify_output=f"E   token={credential}"), artifacts.results_dir)

    assert scan_artifacts(artifacts.path, env={"GITHUB_TOKEN": credential})
    # ... and this is the seam that is not this module's to close: the worker
    # writes its own file, so the artifacts writer never sees the string. #15's
    # redaction covers the container's log, not the file it wrote through the
    # mount, and the audit is what notices.


# --------------------------------------------------------------------------
# The mount the results arrive through
# --------------------------------------------------------------------------


def test_the_mount_it_asks_for_is_not_a_privileged_one(root):
    """`ContainerManager.spawn` mounts nothing today; these are the flags for it.

    Checked against `security.assert_unprivileged` because the mount is the one
    part of a worker's argv this module contributes, and a run directory is not
    the Docker socket or the host root.
    """
    artifacts = RunArtifacts.open(a_run(), root=root)

    flags = artifacts.mount_flags()

    assert_unprivileged(["create", *flags, "apiary-worker"])
    assert flags[1].endswith(f":{DEFAULT_RESULT_DIR}")
    assert flags[1].startswith(str(artifacts.results_dir))
    assert artifacts.worker_env() == {RESULT_DIR_ENV: DEFAULT_RESULT_DIR}


def test_only_the_results_directory_is_reachable_from_a_container(root):
    """The event log and every other issue's record stay out of the mount.

    The container runs LLM-generated code with a push token; mounting the run
    directory would put the orchestrator's own log under that code's pen.
    """
    artifacts = RunArtifacts.open(a_run(), root=root)

    mounted = Path(artifacts.mount_flags()[1].split(":" + DEFAULT_RESULT_DIR)[0])

    assert mounted == artifacts.results_dir
    assert artifacts.events_path.parent == artifacts.path
    assert artifacts.events_path.parent != mounted


# --------------------------------------------------------------------------
# Listing past runs
# --------------------------------------------------------------------------


def test_runs_are_listed_oldest_first_and_grouped_by_repository(root):
    for run_id, repo, clock in (
        ("apiary-20260814-150000-bbbbbb", REPO, "15:00:00"),
        ("apiary-20260814-142530-k3f9qz", REPO, "14:25:30"),
        ("other-20260814-143000-cccccc", OTHER_REPO, "14:30:00"),
    ):
        RunArtifacts.open(a_run(run_id=run_id, repo=repo, now=_at(clock)), root=root).finish()

    views = list_runs(root)

    assert [view.run_id for view in views] == [
        "apiary-20260814-142530-k3f9qz",
        "other-20260814-143000-cccccc",
        "apiary-20260814-150000-bbbbbb",
    ]
    text = runs_text(views)
    assert text.index(REPO) < text.index(OTHER_REPO)
    assert text.count(REPO) == 1  # a header, not one per run


def test_a_directory_that_is_not_a_run_is_not_read(root, capsys):
    RunArtifacts.open(a_run(), root=root).finish()
    (root / "Not A Run Id").mkdir()
    (root / "notes.txt").write_text("mine")

    assert [view.run_id for view in list_runs(root)] == [RUN_ID]
    assert list_runs(root / "nowhere") == ()


def test_an_id_from_a_human_cannot_escape_the_root(root):
    """`swarm show <run-id>` takes a string a human typed, and joins it to a path."""
    with pytest.raises(RunError):
        load_run("../../etc", root)


def test_a_missing_run_says_so_rather_than_reporting_an_empty_one(root):
    with pytest.raises(ArtifactsError):
        load_run("apiary-20260814-999999-zzzzzz", root)


def test_the_artifacts_root_comes_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("APIARY_ARTIFACTS_DIR", raising=False)
    assert artifacts_root() == Path(".swarm/runs")

    monkeypatch.setenv("APIARY_ARTIFACTS_DIR", str(tmp_path))
    assert artifacts_root() == tmp_path


def test_the_summary_is_derived_and_the_result_files_are_the_truth(root):
    """A hand-edited `summary.json` cannot make the run look better than it was."""
    artifacts = RunArtifacts.open(a_run(), root=root)
    write_result(a_record(exit_code=EXIT_INFRASTRUCTURE), artifacts.results_dir)
    artifacts.finish()

    summary_path = artifacts.path / SUMMARY_FILE_NAME
    payload = json.loads(summary_path.read_text())
    payload["results"]["issues"] = {"7": "pr-open"}
    summary_path.write_text(json.dumps(payload))

    view = load_run(RUN_ID, root)

    assert view.results.latest[7].outcome == "infrastructure"
    assert view.needs_human == (7,)
    assert (artifacts.path / RESULTS_DIR_NAME / "issue-7-attempt-1.json").exists()


# --------------------------------------------------------------------------
# The daemon's view of the mount
# --------------------------------------------------------------------------


def test_the_mount_source_is_the_hosts_path_when_the_orchestrator_is_a_container(
    tmp_path, monkeypatch
):
    """A containerized orchestrator must not name its own path to the daemon.

    It reaches Docker through a socket, so `--volume <src>:...` is resolved on
    the host. `/var/apiary/runs` is real inside the orchestrator and absent
    outside it, so the daemon would create an empty directory, the worker would
    write its result into that, and the orchestrator would read its own bind
    mount and find nothing — a run that looks healthy and learns nothing about
    what any worker did.
    """
    monkeypatch.setenv(HOST_ROOT_ENV, "/Users/someone/apiary/.swarm/runs")
    artifacts = RunArtifacts.open(a_run(), root=tmp_path / "var" / "apiary" / "runs")

    flags = artifacts.mount_flags()

    assert flags[0] == "--volume"
    source, _, target = flags[1].rpartition(":")
    assert target == DEFAULT_RESULT_DIR
    assert source.startswith("/Users/someone/apiary/.swarm/runs/")
    assert source.endswith("/results")
    # The run id survives the translation: results are filed per run, and a
    # rewrite that lost it would put every run's results in one directory.
    assert artifacts.run.id in source


def test_the_mount_source_is_unchanged_on_the_host(tmp_path, monkeypatch):
    """One filesystem serves both, so the translation is the identity."""
    monkeypatch.delenv(HOST_ROOT_ENV, raising=False)
    artifacts = RunArtifacts.open(a_run(), root=tmp_path)

    assert artifacts.mount_flags() == [
        "--volume", f"{artifacts.results_dir}:{DEFAULT_RESULT_DIR}"
    ]


def test_a_path_outside_the_artifacts_root_is_left_alone(monkeypatch):
    """This translates one mount. Guessing about others would be worse."""
    monkeypatch.setenv(HOST_ROOT_ENV, "/Users/someone/runs")
    assert host_path("/etc/hosts", root="/var/apiary/runs") == Path("/etc/hosts")


# --------------------------------------------------------------------------
# What produced this result (#97)
# --------------------------------------------------------------------------
#
# #87's success signal is a query over `.swarm/runs/*/summary.json` returning a
# non-Python run with merged PRs and its objective met. That query could not be
# written: `run.json` held `{schema, run_id, repo, objective, started_at}` and
# `summary.json` held outcome counts. Neither recorded the stack or the gate.


def test_run_json_records_the_stack_and_the_gate(root):
    artifacts = RunArtifacts.open(a_run(), root=root, stack="node", verify="npm test")

    written = json.loads((artifacts.path / "run.json").read_text())

    assert written["stack"] == "node"
    assert written["verify"] == "npm test"


def test_summary_json_records_them_too(root):
    """Repeated rather than cross-referenced: this is the file people grep, and
    a query that has to open two files to answer one question is a query nobody
    writes."""
    artifacts = RunArtifacts.open(a_run(), root=root, stack="node", verify="npm test")

    artifacts.finish()
    written = json.loads((artifacts.path / "summary.json").read_text())

    assert written["stack"] == "node"
    assert written["verify"] == "npm test"


def test_the_epic_success_query_can_now_be_written(root):
    """The signal itself, as a test: find the non-Python runs.

    Stated this way because "the fields are present" is not the claim - the
    claim is that a `jq` over the summaries can separate a React run from a
    Python one, which is what nobody could do before.
    """
    for run_id, stack in (
        ("apiary-20260814-140000-aaaaaa", "python"),
        ("apiary-20260814-150000-bbbbbb", "node"),
    ):
        RunArtifacts.open(
            a_run(run_id=run_id), root=root, stack=stack, verify="x"
        ).finish()

    non_python = [view.run_id for view in list_runs(root) if view.stack != "python"]

    assert non_python == ["apiary-20260814-150000-bbbbbb"]


def test_a_run_recorded_before_this_field_existed_still_reads(root):
    """Additive: an older directory reads back with empty strings rather than
    failing to load, and `show_text` prints "(unrecorded)" rather than a blank
    column that looks like a bug."""
    artifacts = RunArtifacts.open(a_run(), root=root)
    artifacts.finish()

    view = read_run(artifacts.path)

    assert view.stack == "" and view.verify == ""
    assert "(unrecorded)" in show_text(view)


def test_show_prints_the_stack_and_the_gate(root):
    artifacts = RunArtifacts.open(a_run(), root=root, stack="react", verify="npm test")

    text = show_text(artifacts.finish())

    assert "react" in text
    assert "npm test" in text


def test_a_zero_merge_run_says_what_the_gate_actually_printed(root):
    """The criterion this ticket exists for.

    For a run with zero merges the entire user-visible answer used to be the
    string "the verify command failed", repeated - so "the model wrote bad
    React" and "npm was unreachable" were byte-identically labelled, while
    `verify_output` sat on disk with no code path printing it.
    """
    artifacts = RunArtifacts.open(a_run(), root=root, stack="node", verify="npm test")
    write_result(
        a_record(
            issue=7,
            exit_code=1,
            reason="the verify command failed",
            verify_command="npm test",
            verify_output="FAIL src/calc.test.js\n  ● adds numbers\n  expected 3 to be 4",
        ),
        artifacts.results_dir,
    )
    write_result(
        a_record(
            issue=8,
            exit_code=2,
            reason="the verify command was denied the network",
            verify_command="npm test",
            verify_output="npm error code E403\nnpm error 403 Filtered",
        ),
        artifacts.results_dir,
    )

    text = show_text(artifacts.finish())

    # Both needed a human. The point is that they no longer read the same.
    assert "#7" in text and "#8" in text
    assert "expected 3 to be 4" in text
    assert "403 Filtered" in text
    assert "gate: npm test" in text


def test_the_shown_output_is_bounded_and_says_when_it_elided(root):
    """A summary, not a log. The full tail is in the result file two lines
    below it, and an unbounded dump would bury the run it is summarising."""
    artifacts = RunArtifacts.open(a_run(), root=root)
    write_result(
        a_record(issue=7, exit_code=1, verify_output="x" * 4_000), artifacts.results_dir
    )

    text = show_text(artifacts.finish())

    assert "earlier characters elided" in text
    assert len(text) < 4_000


def test_the_image_reaches_the_run_view(root):
    """#99 makes the image vary per task, at which point "which image produced
    this result" stops being answerable from anything else in the directory."""
    artifacts = RunArtifacts.open(a_run(), root=root)
    write_result(
        a_record(issue=7, exit_code=1, image="apiary-worker-node:latest"),
        artifacts.results_dir,
    )

    assert "apiary-worker-node:latest" in show_text(artifacts.finish())


# --------------------------------------------------------------------------
# The per-task lifecycle (#141)
# --------------------------------------------------------------------------
#
# `events.jsonl` is append-only and read back, so the names below are a
# published vocabulary rather than a log line. Two things are pinned here: that
# they exist and are distinct, and that adding them left the *reader* alone -
# a run recorded before this ticket renders exactly as it did.


def test_the_lifecycle_names_are_distinct_and_carry_no_label_vocabulary():
    """ADR 0001 removes the `swarm:*` labels; these names outlive them.

    Asserted on the constants rather than on a payload, because the constants
    are what a reader of a recorded run joins on - a rename is a broken replay,
    which is exactly why `CYCLE_FINISHED` was named beside them.
    """
    names = [
        TASK_ELIGIBLE,
        TASK_CLAIMED,
        TASK_RESULT,
        PR_OPENED,
        PR_CHECKS,
        PR_MERGED,
        TASK_LANDED,
        TASK_NEEDS_HUMAN,
    ]

    assert len(set(names)) == len(names)
    assert not any("swarm" in name for name in names)
    # The two families a reader groups by, and the reason `task.needs_human`
    # is spelled with an underscore: `name.split(".")` has to stay two parts.
    assert all(len(name.split(".")) == 2 for name in names)


#: One run's event log exactly as an orchestrator wrote it *before* this
#: ticket - four names, no task events. Held as a literal rather than produced
#: by today's writer, because a fixture the current code generates cannot
#: notice the current code changing.
PRE_LIFECYCLE_EVENTS = [
    {"event": "run.started", "run": RUN_ID, "repo": REPO, "objective": OBJECTIVE},
    {"event": "cycle.started", "run": RUN_ID, "cycle": 0},
    {
        "event": "cycle.finished",
        "run": RUN_ID,
        "cycle": 0,
        "api_calls": 3,
        "inference_calls": 1,
        "inference_s": 12.5,
        "model_swaps": 0,
        "swap_s": 0.0,
        "peak_queue_depth": 1,
        "loaded_model": WORKER_MODEL,
    },
    {"event": "cycle.reconciled", "run": RUN_ID, "cycle": 0, "live": 1, "summary": "cycle 0"},
    {"event": "run.finished", "run": RUN_ID, "note": "the plan is exhausted"},
]


def _recorded_before_this_ticket(root: Path) -> Path:
    """A run directory as it was left by an orchestrator that predates #141."""
    directory = root / RUN_ID
    (directory / RESULTS_DIR_NAME).mkdir(parents=True)
    (directory / "run.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "run_id": RUN_ID,
                "repo": REPO,
                "objective": OBJECTIVE,
                "started_at": "2026-08-14T14:25:30+00:00",
                "stack": "python",
                "verify": "python -m pytest -q",
            }
        ),
        encoding="utf-8",
    )
    (directory / EVENT_LOG_NAME).write_text(
        "".join(json.dumps(event) + "\n" for event in PRE_LIFECYCLE_EVENTS), encoding="utf-8"
    )
    write_result(
        a_record(issue=7, exit_code=1, reason="the verify command failed"),
        directory / RESULTS_DIR_NAME,
    )
    return directory


def test_show_is_unchanged_for_a_run_recorded_before_the_lifecycle_existed(root):
    """The acceptance criterion that keeps the log readable in both directions.

    #141 only ever *writes* new names, so a directory that carries none of them
    must render byte-identically to how it always did: same cycle count folded
    out of `cycle.finished`, same event count, same "needed a human" section.
    Pinned as the whole string, because an assertion on a substring would pass
    for a reader that had started skipping a line.
    """
    root.mkdir(parents=True, exist_ok=True)
    directory = _recorded_before_this_ticket(root)

    text = show_text(read_run(directory))

    assert text == "\n".join(
        [
            f"run {RUN_ID}",
            f"  repo       {REPO}",
            f"  objective  {OBJECTIVE}",
            "  stack      python",
            "  verify     python -m pytest -q",
            "  started    2026-08-14T14:25:30+00:00",
            "  finished   (never - this run did not reach its end)",
            "",
            f"run {RUN_ID}: 1 attempt(s), 1 consumed",
            "  1 task-failed",
            "  issue #7 attempt 1: task-failed (exit 1)  the verify command failed",
            "",
            "1 cycle(s), 3 API call(s), 1 inference call(s) in 12.5s",
            "  0 model swap(s) costing 0.0s, peak inference queue depth 1",
            "  busiest cycle 0: 3 API call(s)",
            "",
            "needed a human: #7",
            "  #7: the verify command failed",
            "      gate: python -m pytest -q",
            "      | 1 passed",
            "",
            # #146. A run recorded before the shadow window existed carries no
            # `state.shadow`, and the line says "not run" rather than reporting
            # zero divergences - the distinction the whole tally exists for.
            "derived shadow: not run (APIARY_DERIVED_SHADOW off, or a run from before #146)",
            "",
            f"{len(PRE_LIFECYCLE_EVENTS)} event(s) in {EVENT_LOG_NAME}, "
            "0 container log(s) in logs/",
            f"at {directory}",
        ]
    )
