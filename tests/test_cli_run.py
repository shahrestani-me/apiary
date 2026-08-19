"""Unit tests for run identity and the v2 entrypoint.

Two properties carry this file, and both are decisions rather than mechanics:

**A run id is safe everywhere it is used.** It is a Docker label value (#15), a
`docker ps --filter` argument (#20), a directory name and a string a human
types back into `swarm show` (#29). Most of the tests below are one consumer's
constraint each, because the id is minted in one place and consumed in four,
and the consumer that discovers a bad id discovers it as a container that
cannot be reaped or a path outside the artifacts root.

**Re-invoking after a crash resumes without replanning.** That is #33's
acceptance criterion, and it decomposes into two assertions that must both
hold: the second invocation gets a *different* run id (`run.py` records why),
and it attaches to the *same* issues rather than planning new ones.

No network and no token. `FakeClient` replays issue payloads shaped like
GitHub's and records every write, and the ledgers are built by running the real
parser over real issue bodies, so a body that would not parse cannot become a
fixture here. #31 will lift the double into the shared fixture set.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from types import SimpleNamespace

import pytest

import swarm.cli as cli
from swarm.cli import main
from swarm.config import SETTINGS
from swarm.containers.manager import DockerCLI
from swarm.github.ledger import KNOWN_STACKS
from swarm.github.client import GitHubHTTPError
from swarm.github.ledger import load_ledger, render_marker
from swarm.greenfield.provision import (
    CHECK_NAME,
    CI_WORKFLOW_PATH,
    PLACEHOLDER_VERIFY,
    ProvisionReport,
)
from swarm.run import (
    MAX_RUN_ID_LENGTH,
    RUN_LABEL,
    SUFFIX_ALPHABET,
    Run,
    RunError,
    attach,
    live_entries,
    new_run_id,
    repo_prefix,
    start_run,
    validate_run_id,
)


def workflow_command(workflow: str) -> str:
    """The command the generated workflow actually runs, read back through YAML.

    A helper rather than a substring check, because #96 made the command a
    block scalar: `run: <command>` no longer appears in the text, and a
    substring assertion could not catch the indent bug the block scalar exists
    to prevent either.
    """
    yaml = pytest.importorskip("yaml")
    steps = yaml.safe_load(workflow)["jobs"][CHECK_NAME]["steps"]
    return str(steps[-1]["run"]).strip()

@pytest.fixture(autouse=True)
def no_stack_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub #101's stack classification for every test in this file.

    `Bootstrap.for_prompt` asks the orchestrator model which stack a prompt
    implies, and leaving that live turned this suite into a 118-second run
    against the host's Ollama - the definition of a test that does not run.
    `choose_stack`'s own behaviour is asserted in `test_bootstrap.py` with an
    injected oracle.
    """
    monkeypatch.setattr(
        "swarm.greenfield.bootstrap.choose_stack", lambda prompt, llm=None: "python"
    )


@pytest.fixture(autouse=True)
def no_image_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub `swarm run`'s image preflight for every test in this file.

    It shells out to `docker image inspect`, so leaving it live would make this
    suite's result depend on which worker images the developer happens to have
    built - the definition of a test that does not run. The preflight's own
    behaviour is asserted directly in "The image preflight" below, where the
    doctor is injected rather than the daemon consulted.
    """
    monkeypatch.setattr(cli, "_refuse_unrunnable_stacks", lambda stack: None)


REPO = "shahrestani-me/apiary"
OBJECTIVE = "add retry with exponential backoff to the http client"
NOW = dt.datetime(2026, 8, 14, 14, 25, 30, tzinfo=dt.timezone.utc)


# --------------------------------------------------------------------------
# Fixtures and helpers (#31 will lift these into the shared set)
# --------------------------------------------------------------------------


def body(
    *,
    marker: str | None = "add-retry-logic",
    attempt: int = 0,
    goal: str = "the client retries idempotent requests",
    files: Sequence[str] = ("src/swarm/github/client.py",),
    verify: str = "python -m pytest -q tests/test_client_retry.py",
    blocked: Sequence[int] = (),
) -> str:
    lines: list[str] = []
    if marker is not None:
        lines += [render_marker(marker, attempt), ""]
    lines += ["## Goal", goal, "", "## Files"]
    lines += [f"- {path}" for path in files]
    lines += ["", "## Verify", verify, "", "## Blocked by"]
    lines += [f"- #{number}" for number in blocked] or ["_none._"]
    return "\n".join(lines)


def issue(
    number: int,
    *,
    labels: Sequence[str] = ("swarm:ready",),
    state: str = "open",
    state_reason: str | None = None,
    title: str | None = None,
    **body_kwargs: Any,
) -> dict[str, Any]:
    return {
        "number": number,
        "title": title if title is not None else f"task {number}",
        "state": state,
        "state_reason": state_reason,
        "labels": [{"name": name} for name in labels],
        "body": body(**body_kwargs),
    }


def done_issue(number: int, *, marker: str, **body_kwargs: Any) -> dict[str, Any]:
    """A finished task: `swarm:done` *and* closed as completed.

    Both facts, always. Readiness reads `state_reason` and not the label, so a
    fixture that carries only one of them is a task that reads as finished to
    this module and as unmet to the one downstream of it.
    """
    return issue(
        number,
        marker=marker,
        labels=("swarm:done",),
        state="closed",
        state_reason="completed",
        **body_kwargs,
    )


@dataclass
class FakeClient:
    """The five calls the entrypoint makes, and a record of every write.

    Body patches are applied to the stored payloads rather than only recorded,
    so marker adoption is visible to a second `load_ledger` in the same test -
    which is exactly the "did the second invocation duplicate anything" question.
    """

    issues: list[dict[str, Any]] = field(default_factory=list)
    writes: list[tuple[str, int, str]] = field(default_factory=list)
    repo: str = REPO

    def list_issues(self, *, state: str = "open", **_: Any) -> list[dict[str, Any]]:
        return [dict(payload) for payload in self.issues]

    def get_issue(self, number: int) -> dict[str, Any]:
        for payload in self.issues:
            if payload["number"] == number:
                return dict(payload)
        raise GitHubHTTPError(404, "GET", f"/issues/{number}", b'{"message": "Not Found"}')

    def update_issue(self, number: int, **fields: Any) -> dict[str, Any]:
        for payload in self.issues:
            if payload["number"] == number:
                payload.update(fields)
                self.writes.append(("update", number, ",".join(sorted(fields))))
                return dict(payload)
        raise GitHubHTTPError(404, "PATCH", f"/issues/{number}", b'{"message": "Not Found"}')

    def add_labels(self, number: int, labels: Iterable[str]) -> list[dict[str, Any]]:
        names = list(labels)
        for payload in self.issues:
            if payload["number"] == number:
                payload["labels"] = payload["labels"] + [{"name": name} for name in names]
        self.writes.extend(("add", number, name) for name in names)
        return [{"name": name} for name in names]

    def remove_label(self, number: int, label: str) -> bool:
        self.writes.append(("remove", number, label))
        for payload in self.issues:
            if payload["number"] == number:
                payload["labels"] = [
                    entry for entry in payload["labels"] if entry["name"] != label
                ]
        return True


def ledger_of(*payloads: dict[str, Any]):
    return load_ledger(FakeClient(list(payloads)))


# --------------------------------------------------------------------------
# The id's shape
# --------------------------------------------------------------------------


def test_a_run_id_is_legible_and_carries_the_repo_and_the_time():
    run_id = new_run_id(REPO, now=NOW, suffix="k3f9qz")

    assert run_id == "apiary-20260814-142530-k3f9qz"


def test_a_run_id_is_lowercase_hyphenated_and_bounded():
    run_id = new_run_id("Shahrestani-ME/Receipt_Scanner.API", now=NOW)

    # The host filesystem is case-insensitive, so two ids differing only in
    # case would be one directory while being two container labels.
    assert run_id == run_id.lower()
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", run_id)
    assert len(run_id) <= MAX_RUN_ID_LENGTH


def test_a_very_long_repo_name_still_yields_a_bounded_id():
    long_name = "a-repository-with-an-unreasonably-long-and-descriptive-name-indeed"
    run_id = new_run_id(f"owner/{long_name}", now=NOW)

    assert len(run_id) <= MAX_RUN_ID_LENGTH
    assert validate_run_id(run_id) == run_id
    # Truncation must not leave the prefix ending in a hyphen against the
    # timestamp's, which would read as an empty component.
    assert "--" not in run_id


def test_a_repo_with_no_alphanumerics_in_its_name_still_yields_an_id():
    # `___` is a legal repository name, and a prefix of nothing would produce
    # an id starting with the timestamp's hyphen.
    assert new_run_id("owner/___", now=NOW, suffix="aaaaaa") == "run-20260814-142530-aaaaaa"


def test_a_run_id_is_usable_as_a_docker_label_value():
    run_id = new_run_id(REPO, now=NOW)

    # `docker ps --filter label=apiary.run=<id>` is parsed on `=` and `,`, and
    # an id carrying either would filter for something else entirely.
    assert not set(run_id) & set(" \t\n\"'=,;:/\\")


def test_a_run_id_cannot_escape_the_artifacts_root(tmp_path):
    run = Run.start(REPO, OBJECTIVE, now=NOW)

    directory = run.artifacts_dir(tmp_path)

    assert directory.parent == tmp_path
    assert directory.resolve().is_relative_to(tmp_path.resolve())


def test_run_ids_of_one_repo_sort_chronologically():
    earlier = new_run_id(REPO, now=NOW, suffix="zzzzzz")
    later = new_run_id(REPO, now=NOW + dt.timedelta(seconds=1), suffix="aaaaaa")

    # Lexicographic order is chronological order, which is what makes #29's
    # `swarm runs` listing sortable without parsing anything.
    assert earlier < later


def test_two_runs_started_in_the_same_second_get_different_ids():
    # The suffix is the whole reason this holds: a crash-restart loop tight
    # enough to fit inside one second would otherwise share a run id, which is
    # the one thing minting a new id per process is supposed to prevent.
    minted = {new_run_id(REPO, now=NOW) for _ in range(25)}

    assert len(minted) == 25
    assert all(set(run_id.rsplit("-", 1)[1]) <= set(SUFFIX_ALPHABET) for run_id in minted)


def test_a_run_id_from_outside_is_validated_before_it_becomes_a_path():
    for hostile in ("..", "../../etc", "a/b", "run id", "Run-1", "-lead", "trail-", "", "a" * 65):
        with pytest.raises(RunError):
            validate_run_id(hostile)


def test_a_repo_that_is_not_owner_slash_name_is_refused():
    for bad in ("apiary", "a/b/c", "", "owner/"):
        with pytest.raises(RunError):
            repo_prefix(bad)


def test_a_run_needs_an_objective():
    with pytest.raises(RunError):
        Run.start(REPO, "   ", now=NOW)


def test_a_run_labels_its_containers_with_its_id():
    run = Run.start(REPO, OBJECTIVE, now=NOW)

    # #15 adds `apiary.issue=<n>` per container; this is the part that is
    # constant for the whole run and the one #20 reaps on.
    assert run.container_labels == {RUN_LABEL: run.id}


# --------------------------------------------------------------------------
# Resume semantics
# --------------------------------------------------------------------------


def test_a_ledger_with_live_issues_is_attached_to_not_replanned():
    ledger = ledger_of(
        issue(1, marker="task-one", labels=("swarm:claimed",)),
        issue(2, marker="task-two", labels=("swarm:review",)),
    )

    attachment = attach(Run.start(REPO, OBJECTIVE, now=NOW), ledger)

    assert attachment.mode == "attach"
    assert attachment.resumed is True
    assert [entry.number for entry in attachment.live] == [1, 2]
    assert attachment.counts == {"swarm:claimed": 1, "swarm:review": 1}


def test_an_empty_ledger_plans():
    attachment = attach(Run.start(REPO, OBJECTIVE, now=NOW), ledger_of())

    assert attachment.mode == "plan"
    assert attachment.live == ()


def test_a_ledger_of_only_finished_work_plans_rather_than_attaching():
    # `done` and `failed` are terminal. A new objective against a repo whose
    # last run completed is new work, not a resumption - attaching to it would
    # leave the run with nothing to dispatch and no reason to plan.
    ledger = ledger_of(
        done_issue(1, marker="task-one"),
        issue(2, marker="task-two", labels=("swarm:failed",)),
    )

    assert attach(Run.start(REPO, OBJECTIVE, now=NOW), ledger).mode == "plan"


def test_live_entries_ignores_terminal_issues_but_keeps_blocked_ones():
    ledger = ledger_of(
        done_issue(1, marker="task-one"),
        issue(2, marker="task-two", labels=("swarm:blocked",), blocked=[1]),
        issue(3, marker="task-three", labels=("swarm:ready",)),
    )

    assert [entry.number for entry in live_entries(ledger)] == [2, 3]


def test_reinvoking_mints_a_new_id_and_adopts_the_same_ledger():
    """#33's acceptance criterion, in the two halves it decomposes into."""
    client = FakeClient([
        issue(1, marker="task-one", labels=("swarm:claimed",)),
        issue(2, marker="task-two", labels=("swarm:ready",)),
    ])

    first = start_run(REPO, OBJECTIVE, source=client, now=NOW)
    second = start_run(REPO, OBJECTIVE, source=client, now=NOW + dt.timedelta(seconds=90))

    # A new id, because the containers and artifacts of the dead process must
    # stay distinguishable from this one's (#20, #29).
    assert first.run.id != second.run.id
    # And the same work, because the issues are the checkpoint.
    assert second.mode == "attach"
    assert [entry.task_id for entry in second.live] == [entry.task_id for entry in first.live]
    # Nothing was written either time: both issues already carry markers, so
    # there was nothing to adopt and certainly nothing to create.
    assert client.writes == []


def test_a_dry_run_does_not_even_adopt_a_hand_written_issue():
    # Adoption is a real write to somebody's issue body. A command that
    # promised to change nothing must not make that one either.
    client = FakeClient([issue(1, marker=None, title="Add retry logic")])

    attachment = start_run(REPO, OBJECTIVE, source=client, now=NOW, adopt=False)

    assert attachment.mode == "attach"
    assert client.writes == []


def test_attaching_adopts_a_hand_written_issue_under_a_stable_id():
    client = FakeClient([issue(1, marker=None, title="Add retry logic")])

    first = start_run(REPO, OBJECTIVE, source=client, now=NOW)
    second = start_run(REPO, OBJECTIVE, source=client, now=NOW + dt.timedelta(seconds=90))

    assert [entry.task_id for entry in first.live] == ["add-retry-logic"]
    # The marker was persisted on the first pass, so the second reads it back
    # rather than deriving a second identity for the same work.
    assert [entry.task_id for entry in second.live] == ["add-retry-logic"]
    assert [write[0] for write in client.writes] == ["update"]


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------


def test_run_against_a_live_ledger_reports_the_run_and_reconciles_readiness(capsys):
    client = FakeClient([
        done_issue(1, marker="task-one"),
        issue(2, marker="task-two", labels=("swarm:blocked",), blocked=[1]),
    ])

    code = main(["run", "--repo", REPO, "--objective", OBJECTIVE, "--plan-only"], client=client)

    assert code == 0
    out = capsys.readouterr().out
    assert "attached to 1 live issue(s)" in out
    assert "no issues were replanned" in out
    # #1 is closed as completed, so #2's only dependency is discharged and the
    # first act of the resumed run is to unblock it.
    assert ("add", 2, "swarm:ready") in client.writes
    assert ("remove", 2, "swarm:blocked") in client.writes


def test_a_dry_run_reports_readiness_without_writing_a_single_label(capsys):
    client = FakeClient([
        done_issue(1, marker="task-one"),
        issue(2, marker="task-two", labels=("swarm:blocked",), blocked=[1]),
    ])

    code = main(["run", "--repo", REPO, "--objective", OBJECTIVE, "--dry-run", "--plan-only"],
                client=client)

    assert code == 0
    assert "nothing will be written" in capsys.readouterr().out
    assert client.writes == []


def test_a_dry_run_refuses_to_plan_an_empty_ledger(capsys):
    """Planning writes issues, and a dry run promised to write nothing.

    Saying so beats a command that silently does nothing on a fresh repo -
    which reads as a bug in GitHub rather than as the choice it is.
    """
    code = main(
        ["run", "--repo", REPO, "--objective", OBJECTIVE, "--dry-run"],
        client=FakeClient([]),
    )

    assert code == 0
    assert "writes no plan" in capsys.readouterr().err


def test_a_dependency_cycle_fails_the_command_rather_than_waiting_forever(capsys):
    client = FakeClient([
        issue(1, marker="task-one", labels=("swarm:blocked",), blocked=[2]),
        issue(2, marker="task-two", labels=("swarm:blocked",), blocked=[1]),
    ])

    code = main(["run", "--repo", REPO, "--objective", OBJECTIVE, "--plan-only"], client=client)

    assert code == 1
    assert "cycle" in capsys.readouterr().err
    assert client.writes == []


def test_a_repo_that_is_not_owner_slash_name_fails_before_any_request(capsys):
    client = FakeClient([issue(1, marker="task-one")])

    code = main(["run", "--repo", "apiary", "--objective", OBJECTIVE], client=client)

    assert code == 1
    assert "owner/name" in capsys.readouterr().err
    assert client.writes == []


@pytest.mark.parametrize(
    "argv",
    [
        ["run"],
        ["run", "--repo", REPO],
        ["run", "--new", "a markdown to CSV tool", "--repo", REPO],
        ["run", "--new", "a markdown to CSV tool"],
        ["run", "--new", "a markdown to CSV tool", "--owner", "me", "--objective", OBJECTIVE],
        ["run", "--new", "a markdown to CSV tool", "--owner", "me", "--dry-run"],
    ],
)
def test_ambiguous_or_incomplete_invocations_are_refused(argv):
    # Every one of these is refused *before* anything is created or read;
    # `--new` in particular creates a repository, which is not undoable from
    # here, so an ambiguous command must never reach it.
    with pytest.raises(SystemExit) as excinfo:
        main(argv, client=FakeClient([]))

    assert excinfo.value.code == 2


PROMPT = "a markdown to CSV tool"
#: What `--new` names the repository, slugified from the prompt as it really is.
NEW_REPO = "me/a-markdown-to-csv-tool"


class NoLabels:
    """`LabelReport`'s one method, which is all `ProvisionReport.summary` calls."""

    def summary(self) -> str:
        return "labels: unchanged"


@dataclass
class Provisioning:
    """Records what would have been created, and answers as `provision` does.

    No repository is created anywhere in this file. The plan is the interesting
    half anyway: it decides the initial commit, the workflow and the verify
    command before a single request is sent, which is the property `provision`
    exists to preserve.
    """

    calls: list[Any] = field(default_factory=list)
    assume_yes: bool | None = None

    def __call__(self, plan: Any, target: Any = None, **kwargs: Any) -> Any:
        self.calls.append(plan)
        self.assume_yes = kwargs.get("assume_yes")
        # A real report, built as `provision` builds it: the command it reports
        # is the one in the commit, not one the caller kept its own copy of.
        return ProvisionReport(
            repo=plan.full_name,
            html_url=f"https://github.com/{plan.full_name}",
            default_branch=plan.default_branch,
            commit_sha="0" * 40,
            labels=NoLabels(),
            protection=("required_status_checks",),
            verify_command=plan.verify_command,
        )

    @property
    def plan(self) -> Any:
        assert len(self.calls) == 1, self.calls
        return self.calls[0]


@dataclass
class Planning:
    """`plan_node`, minus the model. Records the verify command it was handed.

    It also appends an issue, because the run judges planning by re-reading the
    ledger rather than by trusting the planner's return value - a read taken
    straight after a write can be served stale, and trusting it once printed
    "produced nothing" under a list of the issues just created.
    """

    client: FakeClient
    verify: str | None = None
    #: What `--stack` sent, so the flag is asserted where it is handed over
    #: rather than only where it is parsed.
    stack: str | None = None
    #: The bootstrap task #101 prepends, or None for a `--repo` run.
    bootstrap: Any = None

    def __call__(
        self,
        state: Any,
        source: Any = None,
        verify: str | None = None,
        stack: str | None = None,
        bootstrap: Any = None,
    ) -> dict:
        self.verify = verify
        self.stack = stack
        self.bootstrap = bootstrap
        self.client.issues.append(issue(1, marker="seed", labels=("swarm:ready",)))
        return {"tasks": {"seed": {}}, "events": ["planned 1 task(s)"]}


def greenfield(monkeypatch, client: FakeClient, *argv: str) -> tuple[int, Provisioning, Planning]:
    """Run `--new` with provisioning and planning stubbed, and nothing created."""
    provisioning, planning = Provisioning(), Planning(client)
    monkeypatch.setattr("swarm.cli.provision", provisioning)
    monkeypatch.setattr("swarm.cli.plan_node", planning)
    # --plan-only throughout: these tests are about provisioning and the
    # hand-off to the run, not about dispatching containers into a repository
    # that exists only as a fake.
    code = main(
        ["run", "--new", PROMPT, "--owner", "me", "--yes", "--plan-only", *argv],
        client=client,
    )
    return code, provisioning, planning


def test_greenfield_runs_against_the_repository_it_just_created(monkeypatch, capsys):
    client = FakeClient([])

    code, provisioning, _ = greenfield(monkeypatch, client, "--public")

    assert code == 0
    assert provisioning.plan.owner == "me"
    assert provisioning.plan.private is False
    assert provisioning.assume_yes is True
    out = capsys.readouterr().out
    # The prompt is the objective, and the run targets the new repo.
    assert f"repo {NEW_REPO}" in out
    assert PROMPT in out


def test_greenfield_provisions_an_empty_repository_the_placeholder_can_gate(monkeypatch):
    """This test's premise inverted with #104, and the inversion is the point.

    It used to assert the opposite: that `--new` must *not* build a plain
    `ProvisionPlan`, because the repository that produced had no test suite and
    the planner then wrote issues verified with a runner it could not run.

    #101 answered that a different way. The project is now the first *issue* of
    the plan, generated by the model in a worker container, so the initial
    commit is deliberately almost empty - and `test -f README.md` is the only
    command that can gate it. What used to be the bug is now the design, and
    what makes it safe is that the bootstrap's PR replaces both the files and
    the gate before any other task runs.
    """
    code, provisioning, _ = greenfield(monkeypatch, FakeClient([]))

    assert code == 0
    assert provisioning.plan.verify_command == PLACEHOLDER_VERIFY
    # Three files, and no project: the README the placeholder gates, the
    # LICENSE, and the workflow that runs it.
    assert set(provisioning.plan.files()) == {"README.md", "LICENSE", CI_WORKFLOW_PATH}


def test_the_workflow_and_every_issue_carry_the_same_one_command(monkeypatch):
    """One string, from one place. Three copies that can disagree is the bug.

    The required status check runs what the workflow says, the worker runs what
    the issue says, and a run where those two differ has a gate that is red
    before anyone has touched the task.
    """
    code, provisioning, planning = greenfield(monkeypatch, FakeClient([]))

    assert code == 0
    verify = provisioning.plan.verify_command
    assert workflow_command(provisioning.plan.files()[CI_WORKFLOW_PATH]) == verify
    assert verify in provisioning.plan.files()["README.md"]
    assert planning.verify == verify


def test_the_repository_this_would_create_passes_the_command_its_issues_carry(
    monkeypatch, tmp_path
):
    """The defect stated as the property that was untrue.

    Lay out the initial commit exactly as it would be pushed, then run the
    string the planner puts in every `## Verify` - from the repository root,
    with nothing installed, believing only the exit code, which is how CI and
    the worker container both run it. Asserting on the generated files instead
    would pass for a project that does not import.
    """
    code, provisioning, planning = greenfield(monkeypatch, FakeClient([]))
    for path, content in provisioning.plan.files().items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    result = subprocess.run(
        planning.verify, shell=True, cwd=tmp_path, capture_output=True, text=True
    )

    assert code == 0
    assert result.returncode == 0, result.stderr


def test_an_explicit_verify_command_overrides_the_scaffolds(monkeypatch):
    # The override reaches the workflow as well as the issues, which is the
    # whole point: it stays one string, it is simply the operator's string.
    code, provisioning, planning = greenfield(monkeypatch, FakeClient([]), "--verify", "make check")

    assert code == 0
    assert planning.verify == "make check"
    assert workflow_command(provisioning.plan.files()[CI_WORKFLOW_PATH]) == "make check"


def test_a_stack_this_host_cannot_run_is_refused_before_anything_is_created(
    monkeypatch, capsys
):
    """#103 inverted this: the refusal is a fact about the machine, not a word
    in the prompt. What did not change is *when* it happens - while a refusal
    is still free, before a real repository with a URL somebody may already
    have seen.

    The autouse `no_image_preflight` fixture is overridden here, because the
    preflight is the thing under test.
    """
    monkeypatch.undo()  # drop the autouse preflight stub for this test only
    monkeypatch.setattr(
        "swarm.greenfield.bootstrap.choose_stack", lambda prompt, llm=None: "react"
    )
    provisioning = Provisioning()
    monkeypatch.setattr("swarm.cli.provision", provisioning)
    monkeypatch.setattr(
        cli, "preflight", lambda stacks, **kw: _Diagnosis(tuple(stacks))
    )

    code = main(
        ["run", "--new", "a dashboard for build metrics", "--owner", "me", "--yes"],
        client=FakeClient([]),
    )

    assert code == 1
    assert provisioning.calls == []
    err = capsys.readouterr().err
    assert "react" in err


@dataclass
class _Diagnosis:
    """A preflight that fails for whatever stacks it was asked about."""

    stacks: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return False

    @property
    def failures(self):
        from swarm.doctor import Check, stack_check

        return tuple(
            Check.failed(
                stack_check(stack),
                f"no image for {stack}",
                fix="docker build -f Dockerfile.worker.node -t apiary-worker-node .",
            )
            for stack in self.stacks
        )


def test_a_stack_this_host_can_run_is_provisioned(monkeypatch):
    """The other half, and the one the epic is actually for: a React prompt
    now reaches provisioning instead of being refused by a word list."""
    monkeypatch.setattr(
        "swarm.greenfield.bootstrap.choose_stack", lambda prompt, llm=None: "react"
    )
    provisioning = Provisioning()
    monkeypatch.setattr("swarm.cli.provision", provisioning)
    planning = Planning(FakeClient([]))
    monkeypatch.setattr("swarm.cli.plan_node", planning)

    main(
        ["run", "--new", "a dashboard for build metrics", "--owner", "me", "--yes",
         "--plan-only"],
        client=planning.client,
    )

    assert provisioning.calls
    assert provisioning.plan.stack == "react"


@pytest.mark.parametrize(
    "argv, expected",
    [
        ([], SETTINGS.verify_command),
        (["--verify", "make check"], "make check"),
    ],
)
def test_an_existing_repo_verifies_with_the_operators_command(monkeypatch, argv, expected):
    """There is no scaffold to read it off, so the operator is the source.

    Inferring it from the repository's own CI is the tempting alternative and
    the wrong one: a workflow with a matrix and four setup steps has no single
    line to lift, and a command inferred wrong is a gate that was red before a
    worker touched the task.
    """
    client = FakeClient([])
    planning = Planning(client)
    monkeypatch.setattr("swarm.cli.plan_node", planning)

    code = main(
        ["run", "--repo", REPO, "--objective", OBJECTIVE, "--plan-only", *argv], client=client
    )

    assert code == 0
    assert planning.verify == expected


# --------------------------------------------------------------------------
# The sqlite checkpointer is gone
# --------------------------------------------------------------------------


def test_the_entrypoint_holds_no_local_durable_state():
    """The v1 checkpointer is not merely unused - it is not reachable.

    `docs/architecture-v2.md` says the orchestrator holds no irreplaceable
    state and that GitHub wins on any disagreement, which is only a rule you
    need if two stores can disagree. Checking the import graph in a fresh
    interpreter rather than reading the source is what makes this survive
    somebody re-adding the checkpointer three modules deep.

    The assertion is about `langgraph.checkpoint.sqlite` - the package that
    persisted v1's state, and the dependency this decision makes droppable -
    rather than about LangGraph as a whole. `swarm/__init__.py` still imports
    `build_graph` eagerly, so importing anything under `swarm` loads the graph;
    that import is outside this issue's file contract and is named as a
    follow-up rather than reached around from here.
    """
    probe = (
        "import sys, swarm.cli; "
        "print('langgraph.checkpoint.sqlite' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", "the v2 entrypoint loaded the sqlite checkpointer"
    assert not hasattr(sys.modules["swarm.cli"], "_checkpointer")


def test_resume_is_no_longer_a_flag():
    # Resume is the default, not an option: `run.py` mints a new id and adopts
    # the ledger. A `--resume <thread-id>` that quietly did nothing would be
    # worse than one that is gone, because the thread id it names no longer
    # refers to anything.
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--repo", REPO, "--objective", OBJECTIVE, "--resume", "1a2b3c4d"])

    assert excinfo.value.code == 2


# --------------------------------------------------------------------------
# A brief long enough to plan from
# --------------------------------------------------------------------------


def test_an_objective_can_be_read_from_a_file(tmp_path, capsys):
    """`--objective "a trip planner"` gives the planner nothing to decompose.

    A useful objective is paragraphs - constraints, the shape of the thing,
    what done looks like - and that is miserable to quote on a command line and
    impossible to keep in version control. `@path` reads it from a file.
    """
    brief = tmp_path / "brief.md"
    brief.write_text(
        "Build a trip planner.\n\n"
        "It stores trips, each with a destination and dates, and warns when two\n"
        "trips overlap. Persistence is a JSON file; no database.\n",
        encoding="utf-8",
    )
    client = FakeClient([issue(1, marker="task-one", labels=("swarm:ready",))])

    code = main(
        ["run", "--repo", REPO, "--objective", f"@{brief}", "--plan-only"],
        client=client,
    )

    assert code == 0
    assert "trips overlap" in brief.read_text()


def test_a_missing_brief_is_a_command_line_error_not_a_created_repository(tmp_path):
    """The greenfield path would otherwise find the typo after creating a repo."""
    with pytest.raises(SystemExit) as caught:
        main(["run", "--new", f"@{tmp_path / 'nope.md'}", "--owner", "me", "--yes"])
    assert caught.value.code == 2


def test_an_empty_brief_is_refused(tmp_path):
    empty = tmp_path / "empty.md"
    empty.write_text("   \n", encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        main(["run", "--repo", REPO, "--objective", f"@{empty}"])
    assert caught.value.code == 2


def test_the_loop_hands_every_collaborator_the_same_client(monkeypatch):
    """`source` is a client when injected and a repository slug otherwise.

    Converting it at each use site is how `Recovery` ended up holding the
    string while `Reconciler` held a client, and the run died three frames
    inside `apply_plan` asking a `str` for `get_issue`. One conversion, at the
    top of `_loop`, so a new collaborator cannot reintroduce it.
    """
    import swarm.cli as cli

    seen: dict = {}

    class StopHere(Exception):
        pass

    def spy_reconciler(**kwargs):
        seen["reconciler"] = kwargs["client"]
        seen["recovery"] = kwargs["recovery"].client
        raise StopHere

    monkeypatch.setattr("swarm.orchestrator.reconcile.Reconciler", spy_reconciler)
    def spy_fleet(**kwargs):
        seen["fleet_env"] = kwargs["env"]
        return SimpleNamespace(docker=None)

    monkeypatch.setattr("swarm.containers.manager.ContainerManager", spy_fleet)
    monkeypatch.setattr("swarm.containers.reaper.Reaper", lambda **k: SimpleNamespace())
    monkeypatch.setattr(cli.RunArtifacts, "open", classmethod(
        # `**_` absorbs `stack=` and `verify=`: #97 records both in
        # `run.json`, and this double is about the client, not the artifacts.
        lambda cls, run, **_: SimpleNamespace(
            worker_env=lambda: {},
            mount_flags=lambda: [],
            log_sink=lambda handle: None,
            # Where the workers' records land. `_loop` hands it to the
            # reconciler, because a reconciler without it never observes a
            # worker's exit code.
            results_dir="/var/apiary/results",
            event=lambda name, **fields: {},
            # #146's recorder seam. `_loop` hands it to the reconciler, so a
            # double without it stops resembling the object under test.
            observed=lambda payload: {},
        )))

    client = FakeClient([issue(1, marker="task-one", labels=("swarm:ready",))])
    # `_loop` resolves the base commit through the client now, because a
    # worker clones at a commit and an empty one sent it nowhere.
    client.head_sha = lambda ref=None: "a" * 40
    args = SimpleNamespace(
        base_commit="", no_merge=True, no_goal_check=False, dry_run=False, max_cycles=1
    )
    attachment = SimpleNamespace(
        run=SimpleNamespace(id="r", repo=REPO, objective="make the thing work")
    )

    with pytest.raises(StopHere):
        cli._loop(args, attachment, source=client)

    assert seen["reconciler"] is seen["recovery"] is client
    # And the fleet gets what a worker needs to reach anything at all: a token,
    # a model host, somewhere to write, and the proxy that is its only route
    # off an internal network.
    env = seen["fleet_env"]
    assert "HTTP_PROXY" in env and "http_proxy" in env
    assert "NO_PROXY" in env


# --------------------------------------------------------------------------
# What the command's exit code means
# --------------------------------------------------------------------------


def _cycle(goal=None, *, live: int = 0, index: int = 0) -> SimpleNamespace:
    return SimpleNamespace(index=index, live=live, goal=goal)


def _goal(met: bool, *, missing: tuple[str, ...] = (), summary: str = "s") -> SimpleNamespace:
    return SimpleNamespace(
        met=met,
        summary=lambda: summary,
        assessment=SimpleNamespace(missing=missing),
    )


def test_a_run_that_met_its_objective_exits_zero(capsys):
    assert cli._report_outcome([_cycle(_goal(True, summary="objective met"))]) == 0
    assert "objective met" in capsys.readouterr().out


def test_a_run_that_stopped_short_exits_non_zero_and_says_what_is_missing(capsys):
    """A shell script chaining `swarm run` must not read "I abandoned one of the
    three things you asked for and stopped" as success."""
    report = _goal(False, missing=("there is no CLI",), summary="stopping without meeting it")

    code = cli._report_outcome([_cycle(report)])

    assert code == 1
    out = capsys.readouterr().out
    assert "stopping without meeting it" in out
    assert "still missing: there is no CLI" in out


def test_a_run_the_cap_ended_is_not_a_failure(capsys):
    """`--max-cycles` stopping a healthy run says nothing about the objective,
    and the next invocation attaches to whatever is still open."""
    code = cli._report_outcome([_cycle(None, live=2)])

    assert code == 0
    assert "2 live issue(s)" in capsys.readouterr().out


# --------------------------------------------------------------------------
# The three subcommands that were functions before they were commands
# --------------------------------------------------------------------------
#
# `doctor.py:62-64` and `artifacts.py:57-61` both left this wiring to `cli.py`
# and said so. So the tests here are about *dispatch and transport* - that the
# right callable is reached with the right arguments and that its exit code
# survives - and not about what any of them prints. What they print is tested
# where it is decided, in `test_doctor.py` and `test_artifacts.py`, and #97
# changes it there.


import swarm.doctor as doctor_module
from swarm.artifacts import RunArtifacts
from swarm.run import Run


def test_swarm_with_no_subcommand_still_errors(capsys):
    """`required=True` predates this ticket and outlives it: `swarm` alone is
    not a synonym for `swarm run`, and quietly making it one would mean a typo'd
    subcommand starting a swarm."""
    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code == 2
    assert "required" in capsys.readouterr().err


def test_swarm_doctor_runs_the_preflight_and_passes_its_exit_code_through(monkeypatch):
    """Zero and non-zero both come from `doctor.main`. This command adds no
    verdict of its own - the preflight already has one."""
    seen: list[list[str]] = []

    def fake_doctor_main(argv):
        seen.append(list(argv))
        return 0

    monkeypatch.setattr(cli, "doctor_main", fake_doctor_main)
    assert main(["doctor", REPO]) == 0
    assert seen == [[REPO, "--ci-ref", "main"]]


def test_swarm_doctor_exits_non_zero_when_a_check_fails(monkeypatch):
    monkeypatch.setattr(cli, "doctor_main", lambda argv: 1)

    assert main(["doctor", REPO]) == 1


def test_swarm_doctor_forwards_every_argument_it_was_given(monkeypatch):
    """Including the ones with defaults, and *excluding* a repo that was
    omitted - `doctor` falls back to $GITHUB_REPOSITORY for that, and forwarding
    an empty string would defeat it."""
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "doctor_main", lambda argv: seen.append(list(argv)) or 0)

    main(["doctor", "--ci-ref", "develop", "--skip-schema"])

    assert seen == [["--ci-ref", "develop", "--skip-schema"]]


def test_the_doctor_subcommand_accepts_every_option_doctor_itself_does():
    """The drift pin.

    `swarm doctor` re-declares its options rather than sharing a parent parser,
    because `doctor.build_parser` sets its own `prog` and is a complete parser.
    Re-declaration is only safe while something notices a flag added on one side
    and not the other - two commands with the same name and different options is
    the worst of both.
    """
    subparsers = [
        action for action in cli.build_parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    ][0]
    wired = {action.dest for action in subparsers.choices["doctor"]._actions}
    own = {action.dest for action in doctor_module.build_parser()._actions}

    assert own <= wired


def test_swarm_runs_lists_newest_first(tmp_path, capsys):
    """The opposite order to `list_runs`, which is documented oldest-first and
    stays that way. A human typing `swarm runs` wants the one they just ran."""
    root = tmp_path / "runs"
    for run_id, clock in (
        ("apiary-20260814-142530-k3f9qz", "14:25:30"),
        ("apiary-20260814-150000-bbbbbb", "15:00:00"),
    ):
        RunArtifacts.open(
            Run.start(REPO, OBJECTIVE, run_id=run_id, now=_utc_at(clock)), root=root
        ).finish()

    assert main(["runs", "--root", str(root)]) == 0

    out = capsys.readouterr().out
    assert out.index("150000-bbbbbb") < out.index("142530-k3f9qz")


def test_swarm_runs_says_so_when_there_are_none(tmp_path, capsys):
    assert main(["runs", "--root", str(tmp_path / "never-used")]) == 0
    assert "no runs recorded" in capsys.readouterr().out


def test_swarm_show_prints_the_run_summary(tmp_path, capsys):
    root = tmp_path / "runs"
    run_id = "apiary-20260814-142530-k3f9qz"
    RunArtifacts.open(
        Run.start(REPO, OBJECTIVE, run_id=run_id, now=_utc_at("14:25:30")), root=root
    ).finish()

    assert main(["show", run_id, "--root", str(root)]) == 0

    out = capsys.readouterr().out
    assert run_id in out
    assert OBJECTIVE in out


def test_swarm_show_of_an_unknown_id_names_it_rather_than_raising(tmp_path, capsys):
    """A run id is a string a human types back in from a previous line of
    output, so getting it wrong is the common case, not the exotic one."""
    unknown = "apiary-20260814-999999-zzzzzz"

    assert main(["show", unknown, "--root", str(tmp_path)]) == 1

    err = capsys.readouterr().err
    assert unknown in err
    assert "Traceback" not in err


def test_swarm_show_of_a_malformed_id_is_refused_before_it_becomes_a_path(tmp_path, capsys):
    """`load_run` validates first. This asserts the refusal reaches the operator
    as a line rather than as a `RunError` traceback."""
    assert main(["show", "../../etc", "--root", str(tmp_path)]) == 1
    assert "Traceback" not in capsys.readouterr().err


def _utc_at(clock: str) -> dt.datetime:
    hour, minute, second = (int(part) for part in clock.split(":"))
    return dt.datetime(2026, 8, 14, hour, minute, second, tzinfo=dt.timezone.utc)


# --------------------------------------------------------------------------
# The two clocks, at the point of no return
# --------------------------------------------------------------------------


def test_an_inverted_timeout_pair_stops_the_run_before_it_starts(monkeypatch, capsys):
    """`swarm doctor` explains this one; `swarm run` refuses it.

    Checked before `_target`, which is what creates a repository for `--new`:
    provisioning a repo and then dying in the first container is a worse way to
    learn that two environment variables contradict each other.
    """
    monkeypatch.setattr(
        cli, "SETTINGS", SETTINGS.__class__(worker_timeout_s=600, verify_timeout_s=900)
    )

    code = main(["run", "--repo", REPO, "--objective", OBJECTIVE])

    assert code == 1
    err = capsys.readouterr().err
    assert "SWARM_WORKER_TIMEOUT" in err and "SWARM_VERIFY_TIMEOUT" in err
    assert "Traceback" not in err


def test_swarm_doctor_still_runs_with_an_inverted_pair(monkeypatch):
    """The command that diagnoses the problem must not be stopped by it."""
    monkeypatch.setattr(
        cli, "SETTINGS", SETTINGS.__class__(worker_timeout_s=600, verify_timeout_s=900)
    )
    monkeypatch.setattr(cli, "doctor_main", lambda argv: 1)

    assert main(["doctor", REPO]) == 1


# --------------------------------------------------------------------------
# The image preflight (#100)
# --------------------------------------------------------------------------
#
# `IMAGES=0` and `BUILD=0` mean the orchestrator can neither pull nor build a
# worker image, so a missing one is not a runtime inconvenience - it is a
# guaranteed all-infrastructure run, discovered mid-cycle, after a real
# repository already exists. That is worth stopping for.
#
# These bypass the autouse stub above by driving `_refuse_unrunnable_stacks`
# with an injected `Doctor`, so nothing here touches a daemon either.


def a_doctor(images: dict[str, str], **kwargs):
    """A `Doctor` whose only live collaborator is a scripted docker runner."""
    from swarm.doctor import Doctor

    subject = Doctor(
        repo=REPO,
        env={},
        which=lambda name: "/usr/local/bin/docker",
        in_container=False,
        probe_schema=False,
        **kwargs,
    )
    assert subject.docker is not None
    subject.docker = DockerCLI(
        redact=subject.docker.redact, runner=ImageRunner(images=images)
    )
    return subject


@dataclass
class ImageRunner:
    """`docker version` and `docker image inspect`, and nothing else."""

    images: dict[str, str] = field(default_factory=dict)

    def __call__(self, argv, *, timeout_s=None, merge=False):
        args = list(argv)[1:]
        if args[:1] == ["version"]:
            return subprocess.CompletedProcess([], 0, "29.2.1\n", "")
        if args[:2] == ["image", "inspect"]:
            if args[2] in self.images:
                config = json.dumps({"Labels": {"org.apiary.stack": self.images[args[2]]}})
                return subprocess.CompletedProcess([], 0, f"sha256:abc|{config}\n", "")
            return subprocess.CompletedProcess([], 1, "", f"No such image: {args[2]}\n")
        raise AssertionError(f"unexpected docker call: {argv}")


def test_a_run_starts_when_every_stack_it_needs_has_an_image():
    from swarm.containers.manager import DEFAULT_STACK_IMAGES
    from swarm.doctor import preflight

    present = {image: stack for stack, image in DEFAULT_STACK_IMAGES.items()}

    assert preflight(sorted(KNOWN_STACKS), doctor=a_doctor(present)).ok


def test_a_run_is_refused_when_a_stack_it_needs_has_no_image(capsys):
    """The message names the image and the build line, because the orchestrator
    cannot fix this itself and a human has to."""
    from swarm.doctor import preflight

    diagnosis = preflight(["node"], doctor=a_doctor({}, stacks=("node",)))

    assert not diagnosis.ok
    report = diagnosis.report()
    assert "apiary-worker-node" in report
    assert "docker build -f Dockerfile.worker.node" in report
    assert "IMAGES=0" in report


def test_only_the_stack_the_flag_named_is_required():
    """`--stack python` on a host with no Node image is a run that can proceed:
    nothing it dispatches will ever need one."""
    from swarm.containers.manager import WORKER_IMAGE
    from swarm.doctor import preflight

    diagnosis = preflight(["python"], doctor=a_doctor({WORKER_IMAGE: "python"}, stacks=("python",)))

    assert diagnosis.ok


def test_a_denied_probe_does_not_stop_a_run():
    """Through the socket proxy the probe is denied by design, and doctor's
    inability to look is not evidence about the host. Refusing here would make
    a containerised orchestrator unstartable."""
    from swarm.doctor import preflight

    @dataclass
    class Denying(ImageRunner):
        def __call__(self, argv, *, timeout_s=None, merge=False):
            args = list(argv)[1:]
            if args[:1] == ["version"]:
                return subprocess.CompletedProcess([], 0, "29.2.1\n", "")
            return subprocess.CompletedProcess(
                [], 1, "", "Error response from daemon: 403 Forbidden\n"
            )

    subject = a_doctor({}, stacks=("node",))
    subject.docker = DockerCLI(redact=subject.docker.redact, runner=Denying())

    diagnosis = preflight(["node"], doctor=subject)

    assert diagnosis.ok
    assert diagnosis.skipped


# --------------------------------------------------------------------------
# `swarm local` (#UI local mode)
# --------------------------------------------------------------------------


def test_ensure_local_repo_creates_a_repo_a_worker_can_branch_from(tmp_path):
    """A fresh directory gets `git init` and one commit, because `base_branch`
    needs a commit to name; an existing repository is returned untouched."""
    import subprocess

    from swarm.cli import ensure_local_repo

    repo = ensure_local_repo(tmp_path / "demo")

    assert (repo / ".git").is_dir()
    head = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                          cwd=repo, capture_output=True, text=True, check=True)
    assert head.stdout.strip() == "main"
    marker = repo / "untouched.txt"
    marker.write_text("mine")
    assert ensure_local_repo(repo) == repo  # idempotent
    assert marker.exists()                  # and it edited nothing


def test_the_local_subcommand_parses_and_helps():
    import subprocess
    import sys

    probe = subprocess.run([sys.executable, "-m", "swarm.cli", "local", "--help"],
                           capture_output=True, text=True, timeout=30)

    assert probe.returncode == 0, probe.stderr
    assert "--max-rounds" in probe.stdout


def test_the_local_help_names_the_capability_it_gives_up():
    """The help is where a runner is chosen, so it is where the missing
    sandbox is stated - not in a document the chooser has no reason to open.

    The assertions are on the capabilities by name rather than on prose,
    because the prose is allowed to be rewritten and the four rows are not.
    """
    import subprocess
    import sys

    probe = subprocess.run([sys.executable, "-m", "swarm.cli", "local", "--help"],
                           capture_output=True, text=True, timeout=30)

    assert probe.returncode == 0, probe.stderr
    help_text = probe.stdout.lower()
    for capability in ("container sandbox", "egress policy",
                       "pull request + ci", "merge queue"):
        assert capability in help_text, capability
    assert "--unsandboxed" in help_text
    # And the top-line help, which is all `swarm --help` shows.
    top = subprocess.run([sys.executable, "-m", "swarm.cli", "--help"],
                         capture_output=True, text=True, timeout=30)
    assert "no sandbox" in top.stdout.lower(), top.stdout


def test_local_refuses_to_start_without_the_unsandboxed_flag(tmp_path, capsys):
    """The gate is a refusal rather than a warning: a warning is read after
    the model's code has already run, which is the moment it stops helping.

    `--repo` names a directory that does not exist, so reaching
    `ensure_local_repo` would create it - and the absence of that directory is
    the assertion that nothing ran.
    """
    target = tmp_path / "never-created"

    code = main(["local", "--repo", str(target), "--objective", "anything"])

    assert code == 1
    assert not target.exists()
    err = capsys.readouterr().err
    assert "--unsandboxed" in err
    assert "no sandbox" in err.lower()


def test_the_flag_is_the_only_thing_in_front_of_a_local_run(tmp_path, monkeypatch):
    """Passing it gets straight to the run, so the gate cannot quietly become
    a second precondition that refuses for some unrelated reason."""
    reached = []

    def _sentinel(path):
        reached.append(path)
        raise RuntimeError("stop here")

    monkeypatch.setattr(cli, "ensure_local_repo", _sentinel)

    with pytest.raises(RuntimeError, match="stop here"):
        cli._local(
            argparse.Namespace(
                unsandboxed=True,
                repo=str(tmp_path / "demo"),
                objective="anything",
                verify=None,
                max_rounds=None,
            ),
            cli.build_parser(),
        )

    assert reached
