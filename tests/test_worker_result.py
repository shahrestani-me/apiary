"""Exit codes and result files: the four claims #18 makes, each provoked.

- **All three codes are reachable, and they mean different things.** A verified
  run records `0`, a failed gate records `1` and consumes the attempt, and a
  dead model records `2` and does *not*. That last asymmetry is the ticket: a
  broken Ollama that read as a task failure would burn every issue's retry
  budget before anyone noticed, and here it costs nothing.
- **A timeout kill still yields a record.** The container manager's `wait`
  raises `ContainerTimeout` after stopping the container, so the record for
  that attempt has to be synthesised from outside. Without it the attempt is
  absent from the artifacts and the run summary quietly under-counts.
- **A run summary needs nothing but the directory.** No GitHub, no daemon, no
  orchestrator - which is what makes it answerable after the run is dead.

The three exit codes are provoked through the real `entrypoint.main` against
`fixtures/repo.py`'s scratch repository, not asserted against hand-built
`WorkerResult`s: the claim is about what a worker *does*, and a record built
from a fabricated result would pass even if the worker never produced that
outcome. The timeout is provoked through the real `ContainerManager` with a
scripted `docker`, for the same reason - #15's `wait` is what the synthesised
path has to survive, so it is what the test calls.

`ScriptedRunner` is a local four-reply version of the one in
`test_container_manager.py`; importing a double across test modules couples two
suites that should be free to move separately, and only `wait`, `stop`, `logs`
and `rm` are exercised here.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pytest
from fixtures.github import response
from fixtures.repo import VERIFY_COMMAND, ScratchRepo

from swarm.containers.manager import ContainerManager, ContainerTimeout, Handle
from swarm.github.branches import task_branch
from swarm.github.refs import task_ref
from swarm.run import Run
from swarm.state import FileEdit, WorkerOutput
from swarm.worker.entrypoint import (
    EXIT_INFRASTRUCTURE,
    EXIT_OK,
    EXIT_TASK_FAILED,
    OUTPUT_TAIL_CHARS,
    WorkerResult,
    main,
    run_worker,
)
from swarm.worker.result import (
    RESULT_DIR_ENV,
    SCHEMA_VERSION,
    ResultError,
    ResultRecord,
    from_worker,
    has_result,
    load_results,
    read_result,
    record_path,
    report,
    result_dir,
    summarise,
    summarise_dir,
    synthesise,
    synthesise_timeout,
    write_result,
)

ISSUE = 7
RUN_ID = "scratch-20260814-142530-k3f9qz"
REPO = "shahrestani-me/apiary"
BASE_COMMIT = "9f2c1ab3d4e5f60718293a4b5c6d7e8f90a1b2c3"
CONTAINER_ID = "c0ffee" + "0" * 58

#: A verify command that fails whatever the repository contains.
ALWAYS_FAILS = f'{sys.executable} -c "raise SystemExit(3)"'

GOOD_CALC = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


@dataclass
class FakeEditor:
    """The worker model, minus the model. `propose_edits` only calls `.invoke`."""

    output: Any

    def invoke(self, messages: Sequence[tuple[str, str]]) -> WorkerOutput:
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


@dataclass
class ScriptedRunner:
    """A `Runner` keyed by docker subcommand; an exception reply is raised.

    Enough of a daemon for the timeout path: `wait` blows up, `stop` and `rm`
    succeed, `logs` returns whatever the container had printed before it hung.
    """

    replies: dict[str, Any] = field(default_factory=dict)
    calls: list[list[str]] = field(default_factory=list)

    DEFAULTS = {
        "create": CONTAINER_ID + "\n",
        "start": CONTAINER_ID + "\n",
        "wait": "0\n",
        "logs": "",
        "stop": CONTAINER_ID + "\n",
        "rm": CONTAINER_ID + "\n",
    }

    def __call__(
        self, argv: Sequence[str], *, timeout_s: float | None, merge: bool
    ) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        reply = self.replies.get(argv[1], self.DEFAULTS.get(argv[1], ""))
        if isinstance(reply, BaseException):
            raise reply
        return subprocess.CompletedProcess(list(argv), 0, reply, "")

    @property
    def commands(self) -> list[str]:
        return [call[1] for call in self.calls]


def edits(paths: dict[str, str]) -> WorkerOutput:
    return WorkerOutput(edits=[FileEdit(path=path, content=text) for path, text in paths.items()])


def issue_body(*, verify: str = VERIFY_COMMAND) -> str:
    return (
        "<!-- apiary:task id=add-sub attempt=1 -->\n\n"
        "## Goal\n`calc.sub` subtracts its second argument from its first.\n\n"
        "## Files\n- calc.py\n\n"
        f"## Verify\n{verify}\n\n"
        "## Blocked by\n_none._\n"
    )


def issue(**kwargs: Any):
    return response(
        200, {"number": ISSUE, "title": "Add a sub function", "body": issue_body(**kwargs)}
    )


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def worker_env(scratch_repo: ScratchRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """The scratch repo's git isolation, for the worker's own subprocesses."""
    for key, value in scratch_repo.env.items():
        monkeypatch.setenv(key, value)


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "workspace"


@pytest.fixture()
def artifacts(tmp_path: Path) -> Path:
    """Stands in for the run's mounted artifacts directory."""
    return tmp_path / "artifacts" / RUN_ID


def argv(scratch_repo: ScratchRepo, workspace: Path, *extra: str) -> list[str]:
    return [
        "--repo", str(scratch_repo.remote),
        "--issue", str(ISSUE),
        "--base-commit", scratch_repo.head(),
        "--workspace", str(workspace),
        *extra,
    ]


def handle(issue_number: int | None = ISSUE) -> Handle:
    return Handle(id=CONTAINER_ID, run_id=RUN_ID, issue=issue_number, name="apiary-worker")


def a_record(**overrides: Any) -> ResultRecord:
    fields: dict[str, Any] = {
        "run_id": RUN_ID,
        "issue": ISSUE,
        "attempt": 1,
        "exit_code": EXIT_OK,
        "verify_command": "pytest -q",
        "verify_output": "1 passed",
        "reason": "verified and committed",
        "task_id": "add-sub",
        "repo": REPO,
        "branch": task_branch(task_ref(ISSUE), 0),
        "commit": "abc1234",
        "written": ("calc.py",),
    }
    return ResultRecord(**{**fields, **overrides})


# --------------------------------------------------------------------------
# The three codes, provoked
# --------------------------------------------------------------------------


@pytest.mark.usefixtures("worker_env")
def test_a_verified_run_records_exit_zero(fake_github, scratch_repo, workspace, artifacts):
    # The fixture bodies here carry `attempt=1`, so the worker asks for the
    # previous attempt's feedback comment before it edits. No comments is the
    # ordinary answer for an issue whose first attempt died outside the run.
    gh, _, _ = fake_github(issue(), response(200, []))
    started = dt.datetime.now(dt.timezone.utc)

    result = run_worker(
        repo=gh.repo,
        issue=ISSUE,
        base_commit=scratch_repo.head(),
        workspace=workspace,
        clone_url=str(scratch_repo.remote),
        client=gh,
        editor=FakeEditor(edits({"calc.py": GOOD_CALC})),
    )
    path = report(result, run_id=RUN_ID, attempt=1, directory=artifacts, started_at=started)

    written = read_result(path)
    assert written.exit_code == EXIT_OK and written.outcome == "pr-open"
    assert written.consumes_attempt
    # The three things only this process knew, and the reason the file exists.
    assert written.verify_command == VERIFY_COMMAND
    assert written.written == ("calc.py",)
    assert written.commit == result.commit
    assert written.duration_s is not None and written.duration_s >= 0


@pytest.mark.usefixtures("worker_env")
def test_a_failed_gate_records_exit_one_and_charges_the_attempt(
    fake_github, scratch_repo, workspace, artifacts
):
    gh, _, _ = fake_github(issue(verify=ALWAYS_FAILS), response(200, []))

    result = run_worker(
        repo=gh.repo,
        issue=ISSUE,
        base_commit=scratch_repo.head(),
        workspace=workspace,
        clone_url=str(scratch_repo.remote),
        client=gh,
        editor=FakeEditor(edits({"calc.py": GOOD_CALC})),
    )
    record = read_result(report(result, run_id=RUN_ID, attempt=1, directory=artifacts))

    assert record.exit_code == EXIT_TASK_FAILED and record.outcome == "task-failed"
    assert record.consumes_attempt
    assert record.commit is None
    assert record.verify_command == ALWAYS_FAILS


@pytest.mark.usefixtures("worker_env")
def test_a_dead_model_records_exit_two_and_charges_nothing(
    fake_github, scratch_repo, workspace, artifacts
):
    """The point of the third code. This attempt did no work; it must cost none.

    `main` is what produces the 2 - `run_worker` raises rather than returning -
    so the record for it is synthesised the way the manager would synthesise
    one, from the exit code and the container's output.
    """
    gh, _, _ = fake_github(issue(), response(200, []))
    editor = FakeEditor(RuntimeError("connection refused: host.docker.internal:11434"))

    code = main(argv(scratch_repo, workspace), client=gh, editor=editor)

    assert code == EXIT_INFRASTRUCTURE
    record = synthesise(
        handle(),
        exit_code=code,
        reason="the worker exited 2",
        attempt=1,
        logs="! connection refused: host.docker.internal:11434",
    )
    write_result(record, artifacts)

    reread = read_result(record_path(artifacts, ISSUE, 1))
    assert reread.outcome == "infrastructure"
    assert not reread.consumes_attempt
    assert reread.synthesised


def test_the_codes_disagree_about_the_attempt_budget():
    """Stated once, flatly, because everything downstream turns on it."""
    budget = {code: a_record(exit_code=code).consumes_attempt for code in (0, 1, 2)}

    assert budget == {EXIT_OK: True, EXIT_TASK_FAILED: True, EXIT_INFRASTRUCTURE: False}


def test_an_undefined_exit_code_charges_the_attempt():
    """137 is the OOM killer, and nothing in the protocol claims it.

    Charged rather than forgiven: an unknown failure that costs nothing can
    repeat for free forever, and `docs/issue-contract.md` §5 would rather give
    up early and put a human in front of it.
    """
    record = a_record(exit_code=137)

    assert record.outcome == "unknown"
    assert record.consumes_attempt


# --------------------------------------------------------------------------
# The kill nobody gets to report
# --------------------------------------------------------------------------


def test_a_timeout_kill_still_yields_a_result(artifacts):
    """#19's wall clock fires, so the worker never writes its own record.

    Driven through the real `ContainerManager`, whose `wait` stops the
    container and raises `ContainerTimeout`; the caller is left holding the
    handle, which is exactly what the synthesised path needs.
    """
    runner = ScriptedRunner(
        replies={
            "wait": subprocess.TimeoutExpired(cmd="docker wait", timeout=600),
            "logs": "» issue #7 of shahrestani-me/apiary\n  · verifying: pytest -q\n",
        }
    )
    manager = ContainerManager(run=Run.start(REPO, "add retry logic"), runner=runner, env={})
    spawned = manager.spawn(task_ref(ISSUE), BASE_COMMIT, issue=ISSUE)
    started = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=600)

    with pytest.raises(ContainerTimeout):
        manager.wait(spawned, timeout_s=600)
    logs = manager.dispose(spawned)
    path = write_result(
        synthesise_timeout(spawned, attempt=2, timeout_s=600, started_at=started, logs=logs),
        artifacts,
    )

    record = read_result(path)
    assert record.issue == ISSUE and record.attempt == 2
    # A task that can hang for free hangs forever, so the kill is charged.
    assert record.exit_code == EXIT_TASK_FAILED and record.consumes_attempt
    assert record.synthesised and "wall clock" in record.reason
    # The container's last words are the only evidence of what it was doing.
    assert "verifying" in record.verify_output
    assert record.duration_s is not None and record.duration_s >= 600
    # And the container really was stopped and removed on the way through.
    assert runner.commands[-3:] == ["stop", "logs", "rm"]


def test_a_synthesised_record_never_overwrites_the_workers_own(artifacts):
    """A worker that reported and *then* hung has said something truer."""
    write_result(a_record(attempt=1), artifacts)

    write_result(
        synthesise_timeout(handle(), attempt=1, timeout_s=600, logs="hung after pushing"),
        artifacts,
        overwrite=False,
    )

    kept = read_result(record_path(artifacts, ISSUE, 1))
    assert not kept.synthesised and kept.exit_code == EXIT_OK


def test_a_container_without_an_issue_label_is_not_filed_under_a_guess():
    """It is not one of ours, and inventing an issue number invents an attempt."""
    with pytest.raises(ResultError):
        synthesise_timeout(handle(None), attempt=1, timeout_s=600)


# --------------------------------------------------------------------------
# A summary from artifacts alone
# --------------------------------------------------------------------------


def test_a_run_summary_is_reconstructible_from_the_directory(artifacts):
    """No GitHub, no daemon, no orchestrator - just files somebody kept."""
    write_result(a_record(issue=7, attempt=1, exit_code=EXIT_TASK_FAILED), artifacts)
    write_result(a_record(issue=7, attempt=2, exit_code=EXIT_OK), artifacts)
    write_result(
        a_record(issue=8, attempt=1, exit_code=EXIT_INFRASTRUCTURE, synthesised=True), artifacts
    )
    write_result(a_record(issue=9, attempt=1, exit_code=EXIT_TASK_FAILED), artifacts)

    summary = summarise_dir(artifacts)

    assert summary.run_id == RUN_ID
    assert summary.attempts == 4
    # The infrastructure attempt cost nothing; the other three cost budget.
    assert summary.consumed == 3
    assert summary.counts == {"pr-open": 1, "task-failed": 2, "infrastructure": 1}
    assert {issue: r.outcome for issue, r in summary.latest.items()} == {
        7: "pr-open",
        8: "infrastructure",
        9: "task-failed",
    }
    assert [r.issue for r in summary.infrastructure] == [8]
    assert "4 attempt(s), 3 consumed" in summary.text()


def test_one_unreadable_file_does_not_cost_the_others(artifacts, capsys):
    write_result(a_record(issue=7, attempt=1), artifacts)
    record_path(artifacts, 8, 1).write_text("{ truncated mid-writ")

    records = load_results(artifacts)

    assert [record.issue for record in records] == [7]
    assert "skipping issue-8-attempt-1.json" in capsys.readouterr().err


def test_an_attempt_never_overwrites_the_one_before_it(artifacts):
    """One file per attempt: the retry's reason lives in the previous record."""
    write_result(a_record(attempt=1, exit_code=EXIT_TASK_FAILED), artifacts)
    write_result(a_record(attempt=2, exit_code=EXIT_OK), artifacts)

    assert sorted(p.name for p in artifacts.iterdir()) == [
        f"issue-{ISSUE}-attempt-1.json",
        f"issue-{ISSUE}-attempt-2.json",
    ]
    assert has_result(artifacts, ISSUE, 1) and has_result(artifacts, ISSUE, 2)


def test_an_empty_directory_summarises_to_nothing(tmp_path):
    summary = summarise([])

    assert summary.attempts == 0 and summary.consumed == 0
    assert load_results(tmp_path) == ()


# --------------------------------------------------------------------------
# The file itself
# --------------------------------------------------------------------------


def test_the_record_round_trips(artifacts):
    original = a_record(
        started_at=dt.datetime(2026, 8, 14, 14, 25, 30, tzinfo=dt.timezone.utc),
        finished_at=dt.datetime(2026, 8, 14, 14, 27, 0, tzinfo=dt.timezone.utc),
    )

    assert read_result(write_result(original, artifacts)) == original


def test_the_file_is_json_a_human_can_read(artifacts):
    payload = json.loads(write_result(a_record(), artifacts).read_text())

    assert payload["schema"] == SCHEMA_VERSION
    assert payload["issue"] == ISSUE and payload["attempt"] == 1
    # Derived, and written anyway: `"exit_code": 2` alone tells a reader nothing.
    assert payload["outcome"] == "pr-open" and payload["consumes_attempt"] is True
    assert payload["written"] == ["calc.py"]


def test_unknown_fields_are_ignored_and_derived_ones_are_recomputed(artifacts):
    """A newer worker's record still loads, and a hand-edit cannot lie."""
    path = write_result(a_record(exit_code=EXIT_INFRASTRUCTURE), artifacts)
    payload = json.loads(path.read_text())
    payload["outcome"] = "pr-open"
    payload["consumes_attempt"] = True
    payload["invented_by_a_later_version"] = {"nested": True}
    path.write_text(json.dumps(payload))

    record = read_result(path)

    assert record.outcome == "infrastructure" and not record.consumes_attempt


def test_the_record_lists_deletions_apart_from_writes(artifacts):
    """A deleted file listed under `written` would send a reader of the summary
    looking for contents that no longer exist. The field is additive with a
    default, like `image`: no schema bump, and a record written before it
    existed still loads."""
    record = a_record(deleted=("obsolete.py",))

    payload = record.to_dict()
    assert payload["written"] == ["calc.py"]
    assert payload["deleted"] == ["obsolete.py"]
    assert read_result(write_result(record, artifacts)).deleted == ("obsolete.py",)

    elder = {key: value for key, value in payload.items() if key != "deleted"}
    assert ResultRecord.from_dict(elder).deleted == ()


def test_from_worker_carries_the_deletions(tmp_path):
    result = WorkerResult(
        issue=ISSUE,
        repo=REPO,
        task_id="drop-legacy",
        branch=task_branch(task_ref(ISSUE), 0),
        root=tmp_path,
        verify_command="pytest -q",
        verify_output="1 passed",
        passed=True,
        commit="abc1234",
        deleted=("obsolete.py",),
    )

    record = from_worker(result, run_id=RUN_ID, attempt=1)

    assert record.deleted == ("obsolete.py",)
    # A deletion is an edit that landed: the reason must not claim otherwise
    # when a cleanup-only attempt fails its gate.
    failed = from_worker(
        WorkerResult(**{**vars(result), "passed": False, "commit": None}),
        run_id=RUN_ID,
        attempt=1,
    )
    assert failed.reason == "the verify command failed"


def test_a_naive_timestamp_is_read_as_utc():
    """Container and host need not agree about the local zone."""
    naive = dt.datetime(2026, 8, 14, 14, 25, 30)

    record = a_record(started_at=naive, finished_at=naive + dt.timedelta(seconds=90))

    assert record.started_at == naive.replace(tzinfo=dt.timezone.utc)
    assert record.duration_s == 90


def test_output_is_bounded_before_it_reaches_disk(artifacts):
    """A synthesised record carries a container log, which #15 bounds far higher."""
    record = a_record(verify_output="x" * (OUTPUT_TAIL_CHARS * 3))

    written = read_result(write_result(record, artifacts))
    assert len(written.verify_output) < OUTPUT_TAIL_CHARS * 2
    assert "elided" in written.verify_output


def test_a_missing_file_is_an_error_not_an_empty_record(artifacts):
    with pytest.raises(ResultError):
        read_result(record_path(artifacts, ISSUE, 1))


def test_a_json_array_is_not_a_record(artifacts):
    artifacts.mkdir(parents=True)
    stray = record_path(artifacts, ISSUE, 1)
    stray.write_text("[1, 2, 3]")

    with pytest.raises(ResultError):
        read_result(stray)


def test_the_result_directory_comes_from_the_environment(monkeypatch, tmp_path):
    """The mount point the container is given, with a default for a laptop run."""
    monkeypatch.delenv(RESULT_DIR_ENV, raising=False)
    assert result_dir("/somewhere") == Path("/somewhere")

    monkeypatch.setenv(RESULT_DIR_ENV, str(tmp_path))
    assert result_dir("/somewhere") == tmp_path


def test_from_worker_reports_the_gate_and_the_verdict(tmp_path):
    """#22 reads the code; a human reads the reason. Both come off one object."""
    result = WorkerResult(
        issue=ISSUE,
        repo=REPO,
        task_id="add-sub",
        branch=task_branch(task_ref(ISSUE), 0),
        root=tmp_path,
        verify_command="pytest -q",
        verify_output="1 passed",
        passed=True,
        commit="abc1234",
        written=("calc.py",),
    )

    record = from_worker(result, run_id=RUN_ID, attempt=2)

    assert record.exit_code == EXIT_OK and record.reason == "verified and committed"
    assert record.task_id == "add-sub" and record.branch == task_branch(task_ref(ISSUE), 0)
    assert not record.synthesised

    # Verified but with nothing to commit is a failure, and says why.
    nothing = from_worker(
        WorkerResult(**{**vars(result), "commit": None}), run_id=RUN_ID, attempt=2
    )
    assert nothing.exit_code == EXIT_TASK_FAILED and "changed nothing" in nothing.reason
