"""Unit tests for the planner's write half.

The acceptance criterion is a round trip, so the suite is built as one: a
`Tracker` keeps issues in a dict behind #31's `FakeTransport`, the planner
writes through a real `GitHubClient`, and every assertion about what was
written is made by reading it back with `ledger.load_ledger`. Nothing here
inspects a body string to decide whether a write was correct - the loader is
the only thing that says what an issue means, and a test that agreed with the
planner while the loader disagreed would prove nothing.

The second theme is replanning, which is the fragile half of #10. A replan
re-invokes the model, so the tests below drive `write_plan` twice against the
same store and assert on the *issue numbers*: the same ids must land on the
same issues, and a second set is the failure this whole design exists to
prevent (`docs/issue-contract.md` §2).

No network and no token: the transport is the shared fake from
`tests/fixtures/github.py`, and nothing in this file reaches the real tracker.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Sequence

import pytest

from fixtures.github import REPO, response
from swarm.config import SETTINGS
from swarm.github.ledger import load_ledger, render_marker
from swarm.nodes import planner
from swarm.nodes.planner import (
    NO_DEPENDENCIES,
    IssueAction,
    PlanReport,
    Draft,
    PlanError,
    area_label,
    normalise,
    order_drafts,
    plan_node,
    render_body,
    size_label,
    write_plan,
)
from swarm.state import Plan, PlannedTask

BASE = f"/repos/{REPO}/issues"
VERIFY = "python -m pytest -q"


# --------------------------------------------------------------------------
# A tracker behind the shared transport
# --------------------------------------------------------------------------


@dataclass
class Tracker:
    """An issue store, in the seam `FakeTransport.handler` exists for.

    Deliberately the smallest thing that can answer the six calls the planner
    and the loader make between them. It serves back what it was told to
    create, which is the whole point: a double that only replayed a script
    could not tell "updated the issue" from "opened a second one", and that is
    the distinction every replan test turns on.
    """

    issues: dict[int, dict[str, Any]] = field(default_factory=dict)
    next_number: int = 1

    def add(
        self,
        *,
        body: str,
        labels: Sequence[str] = ("swarm:ready",),
        title: str = "a task",
        state: str = "open",
        state_reason: str | None = None,
    ) -> int:
        number = self.next_number
        self.next_number += 1
        self.issues[number] = {
            "number": number,
            "title": title,
            "body": body,
            "labels": [{"name": name} for name in labels],
            "state": state,
            "state_reason": state_reason,
        }
        return number

    def label_names(self, number: int) -> set[str]:
        return {label["name"] for label in self.issues[number]["labels"]}

    def __call__(self, request: Any) -> Any:
        assert request.path.startswith(BASE), f"unexpected path {request.path}"
        tail = request.path[len(BASE):]
        payload = request.json()

        if not tail:
            if request.method == "GET":
                wanted = request.query.get("state", "open")
                return response(
                    200,
                    [
                        dict(issue)
                        for issue in self.issues.values()
                        if wanted == "all" or issue["state"] == wanted
                    ],
                )
            if request.method == "POST":
                number = self.add(
                    title=payload["title"],
                    body=payload.get("body", ""),
                    labels=payload.get("labels", ()),
                )
                return response(201, dict(self.issues[number]))

        parts = [urllib.parse.unquote(part) for part in tail.strip("/").split("/")]
        issue = self.issues[int(parts[0])]

        if len(parts) == 1:
            if request.method == "GET":
                return response(200, dict(issue))
            if request.method == "PATCH":
                issue.update(payload)
                return response(200, dict(issue))
        elif parts[1] == "labels":
            if request.method == "POST":
                for name in payload["labels"]:
                    if name not in self.label_names(issue["number"]):
                        issue["labels"].append({"name": name})
                return response(200, list(issue["labels"]))
            if request.method == "DELETE":
                issue["labels"] = [
                    label for label in issue["labels"] if label["name"] != parts[2]
                ]
                return response(200, list(issue["labels"]))

        raise AssertionError(f"unhandled {request.method} {request.path}")


@pytest.fixture()
def github(fake_github):
    """A `GitHubClient` on the shared transport, backed by a `Tracker`."""

    def build():
        store = Tracker()
        client, transport, _ = fake_github(handler=store)
        return client, store, transport

    return build


# --------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------


def task(
    task_id: str,
    *,
    goal: str | None = None,
    files: Sequence[str] | None = None,
    depends_on: Sequence[str] = (),
) -> PlannedTask:
    return PlannedTask(
        id=task_id,
        goal=goal if goal is not None else f"{task_id} is done",
        files=list(files) if files is not None else [f"src/swarm/{task_id}.py"],
        depends_on=list(depends_on),
    )


def plan(*tasks: PlannedTask, reasoning: str = "one file each") -> Plan:
    return Plan(tasks=list(tasks), reasoning=reasoning)


def write(client, *tasks: PlannedTask, **kwargs: Any):
    return write_plan(client, plan(*tasks), verify=VERIFY, **kwargs)


def numbers_of(store: Tracker) -> list[int]:
    return sorted(store.issues)


# --------------------------------------------------------------------------
# The round trip - the acceptance criterion
# --------------------------------------------------------------------------


def test_a_plan_becomes_issues_the_loader_reads_back_identically(github):
    client, store, _ = github()

    report = write(
        client,
        task("parse-headers", goal="Headers are parsed", files=["src/swarm/github/headers.py"]),
        task(
            "retry-requests",
            goal="Idempotent requests retry three times",
            files=["src/swarm/github/retry.py", "tests/test_retry.py"],
            depends_on=["parse-headers"],
        ),
    )

    assert [action.kind for action in report.actions] == ["created", "created"]
    ledger = load_ledger(client)
    assert set(ledger.entries) == {"parse-headers", "retry-requests"}

    entry = ledger.entries["retry-requests"]
    assert entry.goal == "Idempotent requests retry three times"
    assert entry.files == ("src/swarm/github/retry.py", "tests/test_retry.py")
    assert entry.verify == VERIFY
    assert entry.attempt == 0
    # The dependency is written as an issue *number* and read back as one, and
    # the ledger turns it into a task id: identity and addressing, both intact.
    assert entry.blocked_by == (ledger.entries["parse-headers"].number,)
    assert entry.depends_on == ("parse-headers",)
    assert ledger.errors == () and ledger.repairs == ()


def test_creation_labels_ready_or_blocked_by_the_dependencies_it_wrote(github):
    client, store, _ = github()

    write(client, task("root"), task("leaf", depends_on=["root"]))

    ledger = load_ledger(client)
    assert ledger.entries["root"].state_label == "swarm:ready"
    # Its dependency exists and is open, so `blocked` is the only honest label;
    # readiness (#11) owns every transition after this one.
    assert ledger.entries["leaf"].state_label == "swarm:blocked"


def test_routing_labels_are_applied_at_creation(github):
    client, store, _ = github()

    write(
        client,
        task(
            "parse-headers",
            files=["src/swarm/github/headers.py", "tests/test_headers.py"],
        ),
    )

    assert store.label_names(1) == {"swarm:ready", "area/github", "size/S"}


@pytest.mark.parametrize(
    "files, expected",
    [
        (["src/swarm/github/client.py", "tests/test_client.py"], "area/github"),
        # Only the tests carry a directory, and "every task has tests" is not
        # an area.
        (["README.md", "tests/test_readme.py"], None),
        # Two areas, so no honest single answer.
        (["src/swarm/github/client.py", "src/swarm/nodes/planner.py"], None),
    ],
)
def test_area_label_names_a_directory_or_nothing(files, expected):
    assert area_label(files) == expected


def test_size_label_follows_the_file_count():
    assert size_label(["a.py"]) == "size/S"
    assert size_label([f"{n}.py" for n in range(4)]) == "size/M"
    assert size_label([f"{n}.py" for n in range(9)]) == "size/L"


def test_a_goal_quoting_a_section_heading_round_trips(github):
    """#11's trap, written rather than read: the body must survive its own goal."""
    client, store, _ = github()
    goal = "An issue is ready exactly when every `## Blocked by` reference is closed"

    write(client, task("readiness", goal=goal), task("later", depends_on=["readiness"]))

    ledger = load_ledger(client)
    assert ledger.entries["readiness"].goal == goal
    assert ledger.entries["later"].blocked_by == (ledger.entries["readiness"].number,)


def test_a_multi_line_goal_is_collapsed_onto_one_line(github):
    client, store, _ = github()

    write(client, task("wrapped", goal="first line\n## Files\nsecond line"))

    ledger = load_ledger(client)
    # Collapsed, so the stray heading cannot open a section and the task keeps
    # the files it was planned with.
    assert ledger.entries["wrapped"].goal == "first line ## Files second line"
    assert ledger.entries["wrapped"].files == ("src/swarm/wrapped.py",)


def test_a_task_with_no_dependencies_says_so_in_words(github):
    client, store, _ = github()

    write(client, task("solo"))

    assert NO_DEPENDENCIES in store.issues[1]["body"]
    assert load_ledger(client).entries["solo"].blocked_by == ()


# --------------------------------------------------------------------------
# Replanning
# --------------------------------------------------------------------------


def test_a_replan_updates_the_same_issues_instead_of_opening_new_ones(github):
    client, store, transport = github()
    write(client, task("root"), task("leaf", depends_on=["root"]))
    before = numbers_of(store)

    report = write(
        client,
        task("root", goal="the root does something else now"),
        task("leaf", depends_on=["root"]),
    )

    assert numbers_of(store) == before, "a replan opened a second set of issues"
    assert [action.kind for action in report.updated] == ["updated"]
    assert report.numbers == tuple(before)
    ledger = load_ledger(client)
    assert set(ledger.entries) == {"root", "leaf"}
    assert ledger.entries["root"].goal == "the root does something else now"


def test_a_replan_that_changed_nothing_writes_nothing(github):
    client, store, transport = github()
    write(client, task("root"), task("leaf", depends_on=["root"]))
    written = len(transport.sent)

    report = write(client, task("root"), task("leaf", depends_on=["root"]))

    assert [action.kind for action in report.actions] == ["unchanged", "unchanged"]
    assert not report.changed
    assert [call for call in transport.calls[written:] if call[0] in {"POST", "PATCH"}] == []


def test_a_replan_preserves_the_marker_id_and_the_attempt_counter(github):
    client, store, _ = github()
    number = store.add(
        body=render_body(
            "retry-requests",
            goal="the old goal",
            files=["src/swarm/github/retry.py"],
            verify=VERIFY,
            attempt=2,
        )
    )

    write(client, task("retry-requests", goal="a new goal", files=["src/swarm/retry.py"]))

    assert numbers_of(store) == [number]
    assert render_marker("retry-requests", 2) in store.issues[number]["body"]
    entry = load_ledger(client).entries["retry-requests"]
    assert entry.goal == "a new goal"
    # The counter is the retry budget (§5). A replan that reset it would hand
    # every stalled task an unbounded supply of attempts.
    assert entry.attempt == 2


def test_a_replan_does_not_retitle(github):
    client, store, _ = github()
    number = store.add(
        title="Make the retries actually work",
        body=render_body(
            "retry-requests",
            goal="the old goal",
            files=["src/swarm/github/retry.py"],
            verify=VERIFY,
        ),
    )

    write(client, task("retry-requests", goal="a new goal", files=["src/swarm/retry.py"]))

    # §2 chose a body marker over a title prefix precisely so a human may
    # retitle mid-run; re-asserting a generated title takes that back.
    assert store.issues[number]["title"] == "Make the retries actually work"


def test_a_replan_never_writes_two_issues_under_one_id(github):
    client, store, _ = github()

    report = write(
        client,
        task("retry-requests", files=["src/a.py"]),
        task("Retry Requests", files=["src/b.py"]),
    )

    # Both ids slugify to the same thing, and two issues carrying one id abort
    # the *next* cycle with `DuplicateTaskIdError`.
    assert [action.task_id for action in report.created] == ["retry-requests"]
    assert "id" in report.rejected[0].reason
    assert numbers_of(store) == [1]
    assert set(load_ledger(client).entries) == {"retry-requests"}


def test_a_dropped_task_that_never_started_is_closed_as_not_planned(github):
    client, store, _ = github()
    write(client, task("root"), task("leaf"))

    report = write(client, task("root"))

    assert [(action.task_id, action.kind) for action in report.actions] == [
        ("root", "unchanged"),
        ("leaf", "retired"),
    ]
    dropped = store.issues[report.retired[0].number]
    assert dropped["state"] == "closed"
    # `not_planned`, never `completed`: readiness satisfies a dependency only on
    # `completed`, so nothing downstream is unblocked by a cancellation.
    assert dropped["state_reason"] == "not_planned"


def test_a_dropped_task_with_an_open_pr_is_left_alone(github):
    client, store, transport = github()
    number = store.add(
        body=render_body("in-review", goal="a goal", files=["src/a.py"], verify=VERIFY),
        labels=("swarm:review",),
    )
    written = len(transport.sent)

    report = write(client, task("something-else"))

    retained = report.retained
    assert [(action.task_id, action.number) for action in retained] == [("in-review", number)]
    assert "swarm:review" in retained[0].reason
    assert store.issues[number]["state"] == "open"
    assert store.label_names(number) == {"swarm:review"}
    assert [call for call in transport.calls[written:] if call[1].endswith(f"/{number}")] == []


def test_a_dropped_task_stays_open_when_retiring_is_switched_off(github):
    client, store, _ = github()
    write(client, task("root"), task("leaf"))

    report = write(client, task("root"), retire_dropped=False)

    assert [action.task_id for action in report.retained] == ["leaf"]
    assert store.issues[report.retained[0].number]["state"] == "open"


def test_a_task_in_flight_is_not_rewritten(github):
    client, store, _ = github()
    body = render_body(
        "claimed-task",
        goal="the goal a container is working to",
        files=["src/a.py"],
        verify=VERIFY,
    )
    number = store.add(body=body, labels=("swarm:claimed",))

    report = write(client, task("claimed-task", goal="a different goal", files=["src/a.py"]))

    assert [action.kind for action in report.actions] == ["retained"]
    assert store.issues[number]["body"] == body


def test_a_task_whose_issue_a_human_closed_is_not_resurrected(github):
    client, store, _ = github()
    number = store.add(
        body=render_body("cancelled", goal="the goal", files=["src/a.py"], verify=VERIFY),
        state="closed",
        state_reason="not_planned",
    )

    report = write(client, task("cancelled", goal="a new goal", files=["src/a.py"]))

    assert [action.kind for action in report.actions] == ["retained"]
    assert store.issues[number]["state"] == "closed"
    # Reopening would be the planner overruling a human, and creating a second
    # issue for the same id would corrupt the ledger outright.
    assert numbers_of(store) == [number]


def test_a_replan_prompt_carries_the_existing_ids(monkeypatch):
    """The mitigation for slug churn: the model is told what to re-use."""
    prompts: list[str] = []

    class Stub:
        def invoke(self, messages):
            prompts.append(messages[0][1])
            return plan(task("root"))

    monkeypatch.setattr(planner, "orchestrator_llm", lambda: None)
    monkeypatch.setattr(planner, "structured", lambda _llm, _schema: Stub())

    plan_node(
        {
            "objective": "make it work",
            "tasks": {"retry-requests": {"id": "retry-requests", "status": "failed",
                                         "goal": "retry things", "last_error": "boom"}},
        }
    )

    assert "retry-requests" in prompts[0]
    assert "EXACT existing id" in prompts[0]


# --------------------------------------------------------------------------
# Refusals - what is never written
# --------------------------------------------------------------------------


def test_a_dependency_cycle_raises_before_anything_is_written(github):
    client, store, transport = github()

    with pytest.raises(PlanError, match="cycle"):
        write(client, task("a", depends_on=["b"]), task("b", depends_on=["a"]))

    assert store.issues == {}
    assert [call for call in transport.calls if call[0] == "POST"] == []


def test_a_task_depending_on_itself_raises(github):
    client, store, _ = github()

    with pytest.raises(PlanError, match="themselves"):
        write(client, task("a", depends_on=["a"]))

    assert store.issues == {}


def test_a_body_the_loader_would_refuse_is_never_written(github):
    client, store, _ = github()

    report = write(
        client,
        task("globbed", files=["src/swarm/*.py"]),
        task("fine", files=["src/swarm/fine.py"]),
    )

    rejected = report.rejected
    assert [action.task_id for action in rejected] == ["globbed"]
    assert "glob" in rejected[0].reason
    # The point of rejecting rather than writing: everything in the tracker is
    # in the ledger, so nothing is left present-but-undispatchable.
    assert set(load_ledger(client).entries) == {"fine"}


@pytest.mark.parametrize(
    "broken, reason",
    [
        (PlannedTask(id="", goal="", files=[]), "task id"),
        (PlannedTask(id="empty-goal", goal="   ", files=["src/a.py"]), "Goal"),
        (PlannedTask(id="no-files", goal="a goal", files=[]), "Files"),
    ],
)
def test_a_task_that_cannot_be_a_contract_is_rejected_with_a_reason(github, broken, reason):
    client, store, _ = github()

    report = write(client, broken)

    assert [action.kind for action in report.actions] == ["rejected"]
    assert reason in report.rejected[0].reason
    assert store.issues == {}


def test_an_unresolvable_dependency_is_dropped_and_reported(github):
    client, store, _ = github()

    report = write(client, task("leaf", depends_on=["a-task-nobody-planned"]))

    assert [action.kind for action in report.actions] == ["created"]
    assert report.warnings and "a-task-nobody-planned" in report.warnings[0]
    # Dropped from the body because the contract has no way to write a ref that
    # is not `#N` - and said out loud, because a silently dropped dependency is
    # a task that runs before its prerequisite.
    assert load_ledger(client).entries["leaf"].blocked_by == ()


def test_normalise_strips_the_decoration_a_model_adds():
    drafts, rejected = normalise(
        [PlannedTask(id="Retry Requests!", goal=" a  goal ", files=["`./src/a.py`", "  "])],
        verify=f" {VERIFY} ",
    )

    assert rejected == ()
    assert drafts == (
        Draft(task_id="retry-requests", goal="a goal", files=("src/a.py",), verify=VERIFY),
    )


def test_order_drafts_puts_dependencies_first():
    drafts, _ = normalise(
        [task("c", depends_on=["b"]), task("a"), task("b", depends_on=["a"])], verify=VERIFY
    )

    assert [draft.task_id for draft in order_drafts(drafts)] == ["a", "b", "c"]


# --------------------------------------------------------------------------
# The node
# --------------------------------------------------------------------------


def _stub_model(monkeypatch, produced: Plan) -> None:
    class Stub:
        def invoke(self, _messages):
            return produced

    monkeypatch.setattr(planner, "orchestrator_llm", lambda: None)
    monkeypatch.setattr(planner, "structured", lambda _llm, _schema: Stub())


def test_plan_node_returns_what_the_loader_says_and_not_what_it_sent(github, monkeypatch):
    client, store, _ = github()
    _stub_model(monkeypatch, plan(task("root"), task("leaf", depends_on=["root"])))

    result = plan_node({"objective": "make it work"}, source=client)

    tasks = result["tasks"]
    assert set(tasks) == {"root", "leaf"}
    # `branch` is derived from the issue number by the loader, so its presence
    # is proof the graph is being handed GitHub's answer rather than the plan.
    assert tasks["root"]["branch"] == f"swarm/issue-{store.issues[1]['number']}"
    assert tasks["leaf"]["depends_on"] == ["root"]
    assert tasks["leaf"]["status"] == "pending"
    assert any("created" in event for event in result["events"])


def test_plan_node_writes_the_verify_command_it_was_given(github, monkeypatch):
    """The caller knows the repository; the planner and the model do not.

    A greenfield run's command is whatever the scaffold committed, and it is
    passed down rather than defaulted here, because the alternative - the v1
    `SETTINGS.verify_command` - is a pytest invocation the generated repository
    has no way to run. That is the live failure this parameter closes.
    """
    client, store, _ = github()
    _stub_model(monkeypatch, plan(task("root")))

    result = plan_node(
        {"objective": "make it work"},
        source=client,
        verify="python3 -m unittest discover -q",
    )

    assert result["tasks"]["root"]["status"] == "pending"
    # Read back through the loader, which is the only thing that says what an
    # issue means.
    entry = load_ledger(client, adopt=False).entries["root"]
    assert entry.verify == "python3 -m unittest discover -q"
    assert store.issues[1]["body"].count("## Verify") == 1


def test_plan_node_falls_back_to_the_setting_when_nobody_says(github, monkeypatch):
    # v1's default, and the only answer available to a graph run with no CLI
    # behind it to resolve one.
    client, _, _ = github()
    _stub_model(monkeypatch, plan(task("root")))

    plan_node({"objective": "make it work"}, source=client)

    assert load_ledger(client, adopt=False).entries["root"].verify == SETTINGS.verify_command


def test_plan_node_without_a_target_keeps_the_v1_in_memory_ledger(monkeypatch):
    _stub_model(monkeypatch, plan(task("root")))

    result = plan_node({"objective": "make it work"})

    assert set(result["tasks"]) == {"root"}
    assert result["tasks"]["root"]["status"] == "pending"
    # Silence would read as "the planner did nothing"; the fallback says which
    # of the two happened.
    assert any("not written to the ledger" in event for event in result["events"])


def test_plan_node_takes_its_target_from_the_run_state(github, monkeypatch):
    client, store, _ = github()
    _stub_model(monkeypatch, plan(task("root")))

    monkeypatch.setattr(planner, "_as_client", lambda source: client)
    result = plan_node({"objective": "make it work", "repo": REPO})

    assert set(result["tasks"]) == {"root"}
    assert numbers_of(store) == [1]


# --------------------------------------------------------------------------
# Reading back a plan GitHub has not finished showing
# --------------------------------------------------------------------------


def test_the_read_back_waits_for_issues_github_has_not_surfaced_yet():
    """A write is not immediately visible to the list endpoint.

    Seen twice on a real repository: the planner created two issues and the
    very next read returned the ledger as it was before them, so the run
    printed "the planner wrote nothing" directly beneath a line naming what it
    had written. Dropping the conditional cache was necessary and not
    sufficient - GitHub's own replication is the rest of it.
    """
    from swarm.nodes.planner import _read_back

    calls = {"n": 0}
    def payload(number: int, task_id: str) -> dict:
        return {
            "number": number,
            "title": task_id,
            "state": "open",
            "labels": [{"name": "swarm:ready"}],
            "body": (
                f"<!-- apiary:task id={task_id} attempt=0 -->\n\n"
                "## Goal\ndo the thing\n\n## Files\n- a.py\n\n"
                f"## Verify\n{VERIFY}\n\n## Blocked by\n_none._\n"
            ),
        }

    late = [payload(1, "one"), payload(2, "two")]

    class LaggingClient:
        repo = REPO

        def list_issues(self, **_):
            calls["n"] += 1
            # Empty until the third read, which is the shape of the real thing.
            return late if calls["n"] >= 3 else []

        def invalidate_cache(self):
            pass

    report = PlanReport(repo=REPO, actions=(
        IssueAction(kind="created", task_id="one", number=1),
        IssueAction(kind="created", task_id="two", number=2),
    ))

    ledger = _read_back(LaggingClient(), report, sleep=lambda _: None)

    assert set(ledger.entries) == {"one", "two"}
    assert calls["n"] == 3


def test_the_read_back_gives_up_rather_than_hanging():
    """Bounded: a caller deciding what an empty ledger means beats a run that
    never returns."""
    from swarm.nodes.planner import READ_BACK_ATTEMPTS, _read_back

    calls = {"n": 0}

    class NeverClient:
        repo = REPO

        def list_issues(self, **_):
            calls["n"] += 1
            return []

    report = PlanReport(repo=REPO, actions=(IssueAction(kind="created", task_id="one", number=1),))

    ledger = _read_back(NeverClient(), report, sleep=lambda _: None)

    assert ledger.entries == {}
    assert calls["n"] == READ_BACK_ATTEMPTS
