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

import datetime as dt
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from types import SimpleNamespace

import pytest

from swarm.cli import main
from swarm.config import SETTINGS
from swarm.github.client import GitHubHTTPError
from swarm.github.ledger import load_ledger, render_marker
from swarm.greenfield.provision import (
    CI_WORKFLOW_PATH,
    PLACEHOLDER_VERIFY,
    ProvisionReport,
)
from swarm.greenfield.scaffold import PYTHON_VERIFY, ScaffoldedPlan, scaffold_for
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

    def __call__(self, state: Any, source: Any = None, verify: str | None = None) -> dict:
        self.verify = verify
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


def test_greenfield_provisions_a_scaffold_rather_than_an_empty_repository(monkeypatch):
    """The live bug: `--new` built a plain `ProvisionPlan`.

    The repository that produced had a README, a LICENSE and a workflow running
    `test -f README.md`, and the planner then wrote issues verified with a test
    runner that repository had no way to run. A worker's first act was to clone
    a project with no project in it.
    """
    code, provisioning, _ = greenfield(monkeypatch, FakeClient([]))
    files = provisioning.plan.files()

    assert code == 0
    assert isinstance(provisioning.plan, ScaffoldedPlan)
    # The scaffold rides in the *initial* commit, so the first check run - and
    # the first worker's clone - already has a test suite in front of it.
    assert set(scaffold_for(PROMPT).files()) <= set(files)
    assert provisioning.plan.verify_command == PYTHON_VERIFY != PLACEHOLDER_VERIFY


def test_the_workflow_and_every_issue_carry_the_same_one_command(monkeypatch):
    """One string, from one place. Three copies that can disagree is the bug.

    The required status check runs what the workflow says, the worker runs what
    the issue says, and a run where those two differ has a gate that is red
    before anyone has touched the task.
    """
    code, provisioning, planning = greenfield(monkeypatch, FakeClient([]))

    assert code == 0
    verify = provisioning.plan.verify_command
    assert f"run: {verify}" in provisioning.plan.files()[CI_WORKFLOW_PATH]
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
    assert "run: make check" in provisioning.plan.files()[CI_WORKFLOW_PATH]


def test_a_prompt_naming_another_stack_is_refused_before_anything_is_created(
    monkeypatch, capsys
):
    # `choose_stack` declines rather than silently generating Python, and the
    # decision is taken while a refusal is still free - afterwards there is a
    # real repository with a URL somebody may already have seen.
    provisioning = Provisioning()
    monkeypatch.setattr("swarm.cli.provision", provisioning)

    code = main(
        ["run", "--new", "a dashboard in TypeScript", "--owner", "me", "--yes"],
        client=FakeClient([]),
    )

    assert code == 1
    assert provisioning.calls == []
    assert "TypeScript" in capsys.readouterr().err


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
    monkeypatch.setattr("swarm.containers.manager.ContainerManager", lambda **k: SimpleNamespace(docker=None))
    monkeypatch.setattr("swarm.containers.reaper.Reaper", lambda **k: SimpleNamespace())
    monkeypatch.setattr(cli.RunArtifacts, "open", classmethod(
        lambda cls, run: SimpleNamespace(
            worker_env=lambda: {},
            mount_flags=lambda: [],
            log_sink=lambda handle: None,
        )))

    client = FakeClient([issue(1, marker="task-one", labels=("swarm:ready",))])
    args = SimpleNamespace(base_commit="", no_merge=True, dry_run=False, max_cycles=1)
    attachment = SimpleNamespace(run=SimpleNamespace(id="r", repo=REPO))

    with pytest.raises(StopHere):
        cli._loop(args, attachment, source=client)

    assert seen["reconciler"] is seen["recovery"] is client
