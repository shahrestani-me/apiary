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

The third is #212, finished by #152: what a task's state *is* comes from the
cycle's `Belief` and there is no longer a label it could come from instead.
`authority.state_of` raises without a belief, so a test that is *about* a state
declares it with `believing()` below rather than by typing a `swarm:*` label onto
an issue. Each of those tests asserts on the **selection**, because the failure
they cover produces no error and no transition: a revival that should not happen
just puts a worker back onto merged code.

No network and no token: the transport is the shared fake from
`tests/fixtures/github.py`, and nothing in this file reaches the real tracker.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from fixtures.github import REPO, response
from fixtures.markers import legacy_marker
from swarm.config import SETTINGS
from swarm.github.branches import task_branch
from swarm.github.client import GitHubError
from swarm.github.ledger import (
    DEFAULT_STACK,
    LedgerEntry,
    load_ledger,
    parse_contract,
    render_marker,
)
from swarm.github.refs import pull_ref, task_ref
from swarm.greenfield.bootstrap import STACK_VERIFY
from swarm.nodes import planner
from swarm.nodes.planner import (
    NO_DEPENDENCIES,
    IssueAction,
    PlanReport,
    Draft,
    PlanError,
    area_label,
    format_listing,
    human_prompt,
    normalise,
    order_drafts,
    plan_node,
    prompt_for,
    render_body,
    repository_files,
    size_label,
    write_plan,
)
from swarm.orchestrator.authority import WAITING, Belief, believe
from swarm.orchestrator.derived import (
    ELIGIBLE,
    LANDED,
    NEEDS_HUMAN,
    PullFact,
    observe,
)
from swarm.orchestrator.derived import BLOCKED as BLOCKED_STATE
from swarm.orchestrator.derived import CLAIMED as CLAIMED_STATE
from swarm.orchestrator.derived import REVIEW as REVIEW_STATE
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
    comments: list[tuple[int, str]] = field(default_factory=list)

    def add(
        self,
        *,
        body: str,
        labels: Sequence[str] = (),
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
        elif parts[1] == "comments" and request.method == "POST":
            self.comments.append((issue["number"], payload["body"]))
            return response(201, {"id": len(self.comments)})

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


def believing(book: Any, states: Mapping[str, str] | None = None) -> Belief:
    """The cycle's belief about `book`, with `states` declared on top of it.

    Every test below used to declare a task's state by typing a `swarm:*` label
    onto its issue, and `write_plan` read it back off `LedgerEntry.state_label`.
    #152 removed that field - a state is derived per cycle and lives on the
    cycle's `Belief`, and `authority.state_of` raises rather than falling back to
    a label that no longer exists - so the declaration travels here instead.

    Anything in the ledger the test does not name is `eligible`: a planned task
    nothing has started, which is what those issues wore `swarm:ready` for.
    """
    declared = {task_id: ELIGIBLE for task_id in book.entries}
    declared.update(states or {})
    return Belief(
        states=declared,
        refs={task_id: entry.ref for task_id, entry in book.entries.items()},
    )


def write(client, *tasks: PlannedTask, states: Mapping[str, str] | None = None, **kwargs: Any):
    """Write a plan, with the belief the cycle would have been holding.

    `states` is what this test says the tasks already in the ledger are in; see
    `believing`. A caller passing `believed=` outright is saying something more
    specific and is left alone.
    """
    if "believed" not in kwargs:
        book = kwargs.setdefault("ledger", load_ledger(client))
        kwargs["believed"] = believing(book, states)
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
    assert entry.blocked_by == (ledger.entries["parse-headers"].ref,)
    assert entry.depends_on == ("parse-headers",)
    # `ledger.repairs` stood here until #152 - there is no second state label to
    # repair. `ignored` is the surviving question of the same shape: every issue
    # the planner wrote carries the marker that puts it in the ledger, so nothing
    # it just created may fall out of its own read.
    assert ledger.errors == () and ledger.ignored == ()


def test_creation_says_eligible_or_blocked_by_the_dependencies_it_wrote(github):
    """The same decision `state_label` used to be read back for, in ADR 0001's
    own vocabulary: there is no label on the issue to assert on since #152, and
    the report is where `_create` says which of the two it resolved."""
    client, store, _ = github()

    report = write(client, task("root"), task("leaf", depends_on=["root"]))

    assert [(action.task_id, action.reason) for action in report.created] == [
        ("root", ELIGIBLE),
        # Its dependency exists and is open, so `blocked` is the only honest
        # answer; readiness (#11) owns every transition after this one.
        ("leaf", BLOCKED_STATE),
    ]
    assert set(load_ledger(client).entries) == {"root", "leaf"}


def test_routing_labels_are_applied_at_creation(github):
    client, store, _ = github()

    write(
        client,
        task(
            "parse-headers",
            files=["src/swarm/github/headers.py", "tests/test_headers.py"],
        ),
    )

    # `area/*` and `size/*` are the planner's routing hints, and they are now the
    # *only* labels anything writes. `_create` still resolves the state a fresh
    # task starts in - it is the `reason` the previous test reads off the report
    # - but it no longer stamps it onto the issue: a state apiary invented has no
    # business in a customer's tracker (ADR 0001), and membership is the identity
    # marker in the body, so nothing needs it there.
    assert store.label_names(1) == {"area/github", "size/S"}


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
    assert ledger.entries["later"].blocked_by == (ledger.entries["readiness"].ref,)


def test_a_heading_inside_a_goal_cannot_open_a_section(github):
    """The safety property, which outlived the collapse that used to provide it.

    Goals are multi-line now - they are the worker's whole brief - so they can
    no longer be flattened to keep a stray `## Files` from sectioning the body.
    The heading is defused on the way in instead, and the task keeps the files
    it was planned with. This is issue #11's trap approached from inside the
    Goal section.
    """
    client, store, _ = github()

    write(client, task("wrapped", goal="first line\n## Files\nsecond line"))

    ledger = load_ledger(client)
    assert ledger.entries["wrapped"].goal == "first line\nFiles\nsecond line"
    assert ledger.entries["wrapped"].files == ("src/swarm/wrapped.py", "src/swarm/test_wrapped.py")


def test_a_fence_inside_a_goal_cannot_swallow_the_contract(github):
    """The other half of the same trap: `_scan` treats a fence as opaque, so an
    unclosed one in a goal would hide every section after it."""
    client, store, _ = github()

    write(client, task("fenced", goal="first line\n```python\nsecond line"))

    ledger = load_ledger(client)
    assert ledger.entries["fenced"].files == ("src/swarm/fenced.py", "src/swarm/test_fenced.py")
    assert "```" not in ledger.entries["fenced"].goal


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
        body=render_body("in-review", goal="a goal", files=["src/a.py"], verify=VERIFY)
    )
    written = len(transport.sent)

    report = write(client, task("something-else"), states={"in-review": REVIEW_STATE})

    retained = report.retained
    assert [(action.task_id, action.number) for action in retained] == [("in-review", number)]
    # `review`, not `swarm:review`: the state is the belief's and the report
    # speaks the run's own vocabulary.
    assert "review" in retained[0].reason
    assert "swarm:" not in retained[0].reason
    assert store.issues[number]["state"] == "open"
    assert store.label_names(number) == set()
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
    number = store.add(body=body)

    report = write(
        client,
        task("claimed-task", goal="a different goal", files=["src/a.py"]),
        states={"claimed-task": CLAIMED_STATE},
    )

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


# --------------------------------------------------------------------------
# Revival: a kept `needs-human` task goes back to `eligible`
# --------------------------------------------------------------------------


def failed_body(
    task_id: str,
    *,
    goal: str = "the goal",
    files: Sequence[str] = ("src/a.py",),
    attempt: int = 3,
    blocker: str = "",
    streak: int | None = None,
) -> str:
    """A failed task's body: the marker carries the give-up's budget record.

    The body is the whole record now. `swarm:failed` on the issue said the task
    had given up until #152; the state is the cycle's belief, so every test below
    declares `needs-human` through `states=` and the marker still carries the
    budget the revival arithmetic runs on.
    """
    body = render_body(task_id, goal=goal, files=list(files), verify=VERIFY, attempt=attempt)
    return body.replace(
        render_marker(task_id, attempt),
        legacy_marker(task_id, attempt, blocker=blocker, streak=streak),
    )


def test_a_kept_failed_task_is_revived_with_its_budget_intact(github):
    """The live gap: a replan reported "0 created, 3 updated, 11 left alone"
    and the failed task blocking the whole chain was among the left-alone, so
    the run stayed 0-ready and re-stalled until a human relabelled it."""
    client, store, _ = github()
    body = failed_body("stuck", blocker="ab12cd34ef", streak=3)
    number = store.add(body=body)

    # The plan keeps the task unchanged - same goal, same files.
    report = write(
        client,
        task("stuck", goal="the goal", files=["src/a.py"]),
        states={"stuck": NEEDS_HUMAN},
        max_attempts=3,
        max_total_attempts=9,
    )

    assert [action.kind for action in report.actions] == ["revived"]
    # The revival is the comment and the report; #152 took the relabel off it,
    # so a revived issue's labels are exactly the ones it already had.
    assert store.label_names(number) == set()
    # Nothing is reset: attempt, blocker and streak survive verbatim, so a
    # retry that fails the same way again gives up on its first observation.
    assert store.issues[number]["body"] == body
    assert number in report.numbers
    assert report.summary().endswith("1 revived")
    issue_number, text = store.comments[0]
    assert issue_number == number
    assert text.startswith("apiary: the replan retained this task")
    assert f"returned to `{ELIGIBLE}`" in text
    assert "streak 3 of 3, total 3 of 9" in text


def test_a_kept_failed_task_is_revived_even_when_the_plan_updated_it(github):
    client, store, _ = github()
    body = failed_body("stuck", blocker="ab12cd34ef", streak=3)
    number = store.add(body=body)

    report = write(
        client,
        task("stuck", goal="a different decomposition of the same work", files=["src/a.py"]),
        states={"stuck": NEEDS_HUMAN},
        max_attempts=3,
        max_total_attempts=9,
    )

    assert [action.kind for action in report.actions] == ["revived"]
    # The body is deliberately not rewritten: rendering it fresh would drop
    # the marker's signature record, which is the whole guard.
    assert store.issues[number]["body"] == body


def test_a_failed_task_with_the_total_budget_spent_stays_failed(github):
    client, store, _ = github()
    body = failed_body("spent", attempt=9, blocker="ab12cd34ef", streak=1)
    number = store.add(body=body)

    report = write(
        client,
        task("spent", goal="the goal", files=["src/a.py"]),
        states={"spent": NEEDS_HUMAN},
        max_attempts=3,
        max_total_attempts=9,
    )

    assert [action.kind for action in report.actions] == ["retained"]
    assert "total retry budget spent" in report.actions[0].reason
    assert store.issues[number]["body"] == body
    # No comment either: the give-up comment already told a human what to do,
    # and a replan that keeps refusing must not bury it under repetition.
    assert store.comments == []


def test_a_failed_marker_without_a_signature_record_still_revives(github):
    # Back-compat: an issue failed before blocker= existed. The streak falls
    # back to the attempt counter, which is what it was then.
    client, store, _ = github()
    body = failed_body("old-style", attempt=3)
    number = store.add(body=body)

    report = write(
        client,
        task("old-style", goal="the goal", files=["src/a.py"]),
        states={"old-style": NEEDS_HUMAN},
        max_attempts=3,
        max_total_attempts=9,
    )

    assert [action.kind for action in report.actions] == ["revived"]
    assert store.issues[number]["body"] == body
    assert "streak 3 of 3, total 3 of 9" in store.comments[0][1]


def test_a_dropped_failed_task_is_retired_as_superseded(github):
    """The user's rule: "if plan changed we can cancel old tickets" - a plan
    that dropped a failed task has already decided the work is not wanted, so
    the ticket closes instead of sitting `needs-human` on the board forever.
    Revival is for work the plan still wants; this is its complement."""
    client, store, _ = github()
    body = failed_body("abandoned", attempt=3, blocker="ab12cd34ef", streak=3)
    number = store.add(body=body)

    report = write(client, task("something-else"), states={"abandoned": NEEDS_HUMAN})

    retired = [action for action in report.actions if action.kind == "retired"]
    assert [action.task_id for action in retired] == ["abandoned"]
    assert "closed as superseded" in retired[0].reason
    assert store.issues[number]["state"] == "closed"
    # `not_planned`, so readiness never reads this cancellation as a
    # dependency met - the same rule every other retirement follows.
    assert store.issues[number]["state_reason"] == "not_planned"
    # The marker is the record and stays exactly as the give-up left them:
    # reopening the issue resumes the budget arithmetic mid-count.
    assert store.issues[number]["body"] == body
    issue_number, text = store.comments[0]
    assert issue_number == number
    assert text.startswith("apiary: the plan no longer contains this task")
    assert "closed as superseded" in text
    assert "reopen the issue" in text


def test_a_dropped_failed_task_a_human_closed_is_left_alone(github):
    # GitHub wins: the closure is their record, and a second comment onto a
    # finished conversation explains nothing.
    client, store, _ = github()
    number = store.add(
        body=failed_body("abandoned"),
        state="closed",
        state_reason="completed",
    )

    report = write(client, task("something-else"), states={"abandoned": NEEDS_HUMAN})

    retained = [action for action in report.actions if action.kind == "retained"]
    assert [action.task_id for action in retained] == ["abandoned"]
    assert store.issues[number]["state_reason"] == "completed"
    assert store.comments == []


def test_a_dropped_failed_task_stays_open_when_retiring_is_switched_off(github):
    # `retire_dropped=False` is the goal gate's whole safety argument for its
    # follow-up rounds; the failed case must not become an exception to it.
    client, store, _ = github()
    number = store.add(body=failed_body("abandoned"))

    report = write(
        client,
        task("something-else"),
        states={"abandoned": NEEDS_HUMAN},
        retire_dropped=False,
    )

    retained = [action for action in report.actions if action.kind == "retained"]
    assert [action.task_id for action in retained] == ["abandoned"]
    assert store.issues[number]["state"] == "open"
    assert store.comments == []


# --------------------------------------------------------------------------
# The state is the authority's, and there is no label left to be it (#212, #152)
# --------------------------------------------------------------------------
#
# `_update` and the drop path read `entry.state_label` and `write_plan` took no
# belief at all, so a label a human edited mid-run decided a revival, a
# retirement and a rewrite - which is the one thing #147's criterion says in as
# many words must not happen. #212 moved both decisions onto `authority.state_of`
# and #152 removed the label they used to read, so the hand edit these tests make
# is now a label that reaches no decision by construction.
#
# The tests stay, and what they pin has got stronger rather than weaker: an issue
# can still be labelled by hand - `area/*`, `size/*`, `agent:*` and anything a
# human types - and `LedgerEntry.labels` still carries every one of them. That a
# label sitting on an issue changes nothing the planner does is exactly the
# property, and it is a property somebody can still break.
#
# The failing direction produces no error and no transition. A revival that
# should not happen simply puts a worker back onto merged code, so every
# assertion here is on the *selection* - the action, the labels on the issue and
# the calls the client made - and never on a plan having succeeded.


def a_task(github: Any, label: str, *, was: str = LANDED, **facts: Any) -> SimpleNamespace:
    """One task wearing `label`, a world that disagrees, and the belief.

    The label is the hand edit and everything else is held fixed: one body, one
    observation, one plan. The belief comes from that single observation, which
    is what makes the comparisons below about what is believed rather than about
    two different runs (`shadow.py`'s rule, and `test_authority`'s §7).

    `was` is what this process believed last cycle, and passing it is what makes
    these edits made *mid-run* rather than seeds: with no memory a task apiary
    has not seen this process is simply unknown to it (`believe`'s `previous`),
    and there is no label left for it to be seeded from. A `was` of `landed` goes
    into the ratchet's own set as well as into `remembered`, which is #214's split
    - only what this process decided may enter that set.
    """
    client, store, transport = github()
    number = store.add(
        body=failed_body("landed-work", blocker="ab12cd34ef", streak=3), labels=(label,)
    )
    book = load_ledger(client)
    ref = task_ref(number)
    seen = observe(cycle=0, entries=list(book.entries.values()), **facts)
    return SimpleNamespace(
        client=client,
        store=store,
        transport=transport,
        number=number,
        ledger=book,
        held=believe(
            book,
            seen,
            remembered={ref: was},
            landed=frozenset({ref}) if was == LANDED else frozenset(),
        ),
        mark=len(transport.sent),
    )


def a_merged_task(github: Any, label: str) -> SimpleNamespace:
    """`a_task` over a world where the task's pull request merged."""
    return a_task(
        github,
        label,
        pulls=(PullFact(number=pull_ref(101), ref=task_ref(1), merged=True),),
    )


def decided(one: SimpleNamespace, report: PlanReport) -> tuple[Any, ...]:
    """What the write actually did, as a value two runs can be compared on.

    Deliberately **not** the labels the issue wears. The two runs differ there by
    construction - that is the edit - and `revive` writes one, so a comparison
    including them could only ever fail. What has to match is the actions, the
    issue as the loader will read it back, the comments and the calls made.
    """
    return (
        tuple((action.kind, action.task_id, action.number, action.reason) for action in report.actions),
        tuple(
            (number, issue["state"], issue.get("state_reason"), issue["body"])
            for number, issue in sorted(one.store.issues.items())
        ),
        tuple(one.store.comments),
        tuple(one.transport.calls[one.mark:]),
    )


def test_the_planner_decides_in_the_authoritys_own_states():
    """The sets are internal states now, and this is what keeps them honest.

    They are spelled by hand rather than imported, because `orchestrator` imports
    this module and the dependency must not point back up (`planner.FAILED`'s
    reason, one constant along). A duplicated constant is a constant that can
    drift, so the duplication is pinned to its source here instead.
    """
    assert planner.WRITABLE == WAITING == {ELIGIBLE, BLOCKED_STATE}
    assert planner.IN_FLIGHT == {CLAIMED_STATE, REVIEW_STATE}
    assert (planner.NEEDS_HUMAN, planner.ELIGIBLE) == (NEEDS_HUMAN, ELIGIBLE)
    # And nothing is left spelling a state as a label: `FAILED` was
    # `swarm:failed` until #152 and is the internal state now, like every other
    # constant here.
    assert planner.FAILED == NEEDS_HUMAN
    assert not any(
        name.startswith("swarm:")
        for name in (planner.FAILED, planner.NEEDS_HUMAN, planner.ELIGIBLE, *planner.WRITABLE, *planner.IN_FLIGHT)
    )


def test_a_done_issue_relabelled_failed_mid_run_is_not_revived(github):
    """#201's route into a cleared ratchet, closed where it started.

    A `swarm:done` issue a human relabels `swarm:failed` mid-run is still
    believed `landed` - #147 ignores the relabel and #201's ratchet holds it -
    and its issue is *open*, so `_update`'s closed-issue guard does not catch it
    either. The revival branch fired on the label alone: a worker back onto code
    that is already on the default branch, and before #210 the revival overlay
    cleared the ratchet on its way past.
    """
    correct = a_merged_task(github, "swarm:done")
    edited = a_merged_task(github, "swarm:failed")
    assert edited.held.state("landed-work") == LANDED
    kept = task("landed-work", goal="the goal", files=["src/a.py"])

    reports = [
        write(one.client, kept, ledger=one.ledger, believed=one.held, max_attempts=3, max_total_attempts=9)
        for one in (correct, edited)
    ]

    # Byte-identical: the hand edit moved the label and decided nothing.
    assert decided(correct, reports[0]) == decided(edited, reports[1])
    assert [action.kind for action in reports[1].actions] == ["retained"]
    assert LANDED in reports[1].actions[0].reason
    # The selection, which is the whole assertion: no relabel, no comment and no
    # write of any kind against the issue. A revival here produces no error and
    # no transition, so nothing else here would notice it.
    assert edited.store.label_names(edited.number) == {"swarm:failed"}
    assert edited.store.comments == []
    assert edited.transport.calls[edited.mark:] == [("GET", BASE)]


# `test_under_labels_that_relabel_revives_merged_work_again` stood here. It was
# the escape-hatch half of the pair above: `APIARY_STATE_SOURCE=labels` restored
# the pre-#212 behaviour, defect included, by believing the `swarm:*` label a
# human had typed. #152 removed the flag, the arm in `authority.believe` behind
# it and the labels themselves, so there is no longer a way to ask for the old
# behaviour and nothing left for this test to select. Its surviving half - that
# the relabel decides nothing - is the test directly above, and "a caller with no
# belief at all" is `test_a_task_the_belief_never_saw_is_retained_rather_than_rewritten`.


def test_a_dropped_landed_task_relabelled_failed_is_not_retired(github):
    """The drop path's own version, and its cost falls on the tasks behind it.

    Retiring closes the issue `not_planned`, which is precisely the state reason
    readiness (#11) refuses to read as a dependency met - so a `swarm:failed`
    typed onto merged work and dropped by the same replan left every task waiting
    on that landed one blocked for the rest of the run.
    """
    correct = a_merged_task(github, "swarm:done")
    edited = a_merged_task(github, "swarm:failed")
    dropped = task("something-else")

    reports = [
        write(one.client, dropped, ledger=one.ledger, believed=one.held)
        for one in (correct, edited)
    ]

    assert decided(correct, reports[0]) == decided(edited, reports[1])
    retained = [action for action in reports[1].actions if action.task_id == "landed-work"]
    assert [(action.kind, action.reason) for action in retained] == [
        ("retained", f"dropped, {LANDED}")
    ]
    assert edited.store.issues[edited.number]["state"] == "open"
    assert edited.store.comments == []
    # The edit is still on the issue and still says the opposite of what is
    # believed - it simply reaches nothing.
    assert edited.store.label_names(edited.number) == {"swarm:failed"}


def test_a_spent_task_a_human_asked_for_back_is_revived_rather_than_rewritten(github):
    """The direction #152 made silent, which is why the belief has to decide it.

    A human who reads apiary's own give-up comment does what it asks and moves
    the issue back. Selecting on the label, the planner read that as ordinary
    waiting work and rewrote its body - dropping the marker's signature record,
    which is the arithmetic's whole input. The belief reads apiary's own store
    instead - `authority.BUDGET_SPENT`, a streak already at its cap - and revives
    it, which is the decision that actually grants the attempt that was asked for.

    Since #152 there is no `swarm:ready` for the human to type and no
    `entry.state_label` for the planner to read, so the label arm this test was
    the `derived` half of cannot select anything at all - with no error and no
    transition to show it. The state below is declared where a state now lives.
    """
    spent = a_task(github, "swarm:ready", was=NEEDS_HUMAN)
    assert spent.held.state("landed-work") == NEEDS_HUMAN
    kept = task("landed-work", goal="the goal", files=["src/a.py"])

    revived = write(
        spent.client, kept, ledger=spent.ledger, believed=spent.held, max_attempts=3, max_total_attempts=9
    )

    body = spent.store.issues[spent.number]["body"]
    assert [action.kind for action in revived.actions] == ["revived"]
    assert "streak 3 of 3, total 3 of 9" in spent.store.comments[0][1]
    # Nothing is reset and nothing is rewritten: the marker's signature record is
    # the whole guard, and a revival that re-rendered the body would drop it.
    assert body == failed_body("landed-work", blocker="ab12cd34ef", streak=3)
    # And the hand-typed label is left exactly where the human put it. `revive`
    # relabelled the issue until #152; a revival is a comment and a report now.
    assert spent.store.label_names(spent.number) == {"swarm:ready"}

    # The complement, and the reason the belief is asked at all: the same issue
    # believed `eligible` is ordinary waiting work, so the body *is* rewritten -
    # and the signature record the arithmetic runs on goes with it.
    waiting = a_task(github, "swarm:ready", was=ELIGIBLE)
    assert [
        action.kind
        for action in write(
            waiting.client,
            kept,
            ledger=waiting.ledger,
            believed=Belief(states={"landed-work": ELIGIBLE}),
        ).actions
    ] == ["updated"]
    assert waiting.store.issues[waiting.number]["body"] != body
    assert waiting.store.comments == []


def test_a_task_the_belief_never_saw_is_retained_rather_than_rewritten(github):
    """A caller holding two reads of the tracker, and the direction to fail in.

    Every caller in the loop passes the belief built from the very ledger
    `write_plan` was handed, so this is a shape nothing produces today. That is
    precisely why it may not be a write: the alternative to refusing is patching
    an issue on the strength of a disagreement about which read is current.
    """
    one = a_merged_task(github, "swarm:failed")
    kept = task("landed-work", goal="the goal", files=["src/a.py"])

    report = write(one.client, kept, ledger=one.ledger, believed=Belief())

    assert [(action.kind, action.reason) for action in report.actions] == [
        (
            "retained",
            f"{planner.NO_BELIEF}; the planner rewrites neither work in flight "
            "nor work that has landed",
        )
    ]
    assert one.store.label_names(one.number) == {"swarm:failed"}
    assert one.store.comments == []
    assert one.transport.calls[one.mark:] == [("GET", BASE)]


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
        # `src/test_a.py` is the pytest-gate repair: a lone module under a pytest
        # verify gains its test file, or the task is exit 5 by construction.
        Draft(task_id="retry-requests", goal="a goal", files=("src/a.py", "src/test_a.py"),
              verify=VERIFY),
    )


def test_order_drafts_puts_dependencies_first():
    drafts, _ = normalise(
        [task("c", depends_on=["b"]), task("a"), task("b", depends_on=["a"])], verify=VERIFY
    )

    assert [draft.task_id for draft in order_drafts(drafts)] == ["a", "b", "c"]


# --------------------------------------------------------------------------
# The repository listing
# --------------------------------------------------------------------------


def test_the_planner_is_shown_the_repositorys_files():
    """The defect the listing exists for: planned from the objective alone, a
    real run implemented the same domain three times because the model had no
    way to know the first implementation existed. The listing is a fact about
    this run, so it rides in the *human* turn, next to the objective."""
    system, human = prompt_for(
        "an objective", files=["src/wallet.py", "src/expense.py"], verify=VERIFY
    )

    assert "The repository currently contains these files" in human
    assert "src/wallet.py" in human and "src/expense.py" in human
    assert "parallel implementation" in human
    assert "src/wallet.py" not in system


def test_an_absent_listing_reproduces_the_prompt_byte_for_byte():
    """Pinned: the listing is advisory, and the callers that cannot obtain one
    (the console has no repository; a tree read may 502) must send exactly the
    prompt that has been working all along - not a variant of it."""
    plain = prompt_for("an objective", verify=VERIFY)

    assert prompt_for("an objective", verify=VERIFY, files=None) == plain
    assert prompt_for("an objective", verify=VERIFY, files=()) == plain
    # A listing of nothing but machinery filters down to nothing and must not
    # leave a header announcing no files.
    assert prompt_for("an objective", verify=VERIFY, files=[".git/HEAD"]) == plain
    assert plain[1] == "Objective:\nan objective"


def test_the_listing_is_sorted_filtered_and_capped():
    """A big repository must not drown the objective under its own tree: the
    machinery is filtered out, the rest is sorted so siblings sit together, and
    everything past the cap is summarised as an honest count."""
    files = [f"src/module_{index:03}.py" for index in range(250)] + [
        ".git/config",
        "node_modules/left-pad/index.js",
        "assets/logo.png",
        "package-lock.json",
        "poetry.lock",
        "dist/app.whl",
    ]

    text = format_listing(files)

    for machinery in (".git/", "node_modules", "logo.png", "-lock", "poetry", "dist/"):
        assert machinery not in text, f"machinery survived the filter: {machinery}"
    assert text.count("src/module_") == 200
    assert "… and 50 more files" in text
    lines = text.splitlines()
    assert lines[-1] == "… and 50 more files"
    shown = [line for line in lines if line.startswith("src/")]
    assert shown == sorted(shown)


def test_the_local_checkout_is_listed_from_the_filesystem(tmp_path, monkeypatch):
    """`swarm local` has no GitHub to ask, so the walk is the source - pruned,
    because the point of skipping node_modules is also not to enumerate it."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# r\n", encoding="utf-8")
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("x", encoding="utf-8")
    monkeypatch.setattr(planner, "SETTINGS", SimpleNamespace(repo_path=str(tmp_path)))

    assert repository_files(None) == ("README.md", "src/app.py")


def test_a_directory_that_is_not_a_checkout_yields_no_listing(tmp_path, monkeypatch):
    # `repo_path` defaults to the working directory, and presenting whatever
    # happens to be there as "the project" would be worse than saying nothing.
    monkeypatch.setattr(planner, "SETTINGS", SimpleNamespace(repo_path=str(tmp_path)))

    assert repository_files(None) is None


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
    # `branch` is derived from the task ref and the attempt by the loader
    # (#144), so its presence is proof the graph is being handed GitHub's
    # answer rather than the plan - the plan has neither.
    assert tasks["root"]["branch"] == task_branch(task_ref(store.issues[1]["number"]), 0)
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
    entry = load_ledger(client).entries["root"]
    assert entry.verify == "python3 -m unittest discover -q"
    assert store.issues[1]["body"].count("## Verify") == 1


def test_plan_node_falls_back_to_the_setting_when_nobody_says(github, monkeypatch):
    # v1's default, and the only answer available to a graph run with no CLI
    # behind it to resolve one.
    client, _, _ = github()
    _stub_model(monkeypatch, plan(task("root")))

    plan_node({"objective": "make it work"}, source=client)

    assert load_ledger(client).entries["root"].verify == SETTINGS.verify_command


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


def _recording_model(monkeypatch, produced: Plan) -> list[str]:
    """Like `_stub_model`, but hands back the human turns the model was shown."""
    humans: list[str] = []

    class Stub:
        def invoke(self, messages):
            humans.append(dict(messages)["human"])
            return produced

    monkeypatch.setattr(planner, "orchestrator_llm", lambda: None)
    monkeypatch.setattr(planner, "structured", lambda _llm, _schema: Stub())
    return humans


def test_plan_node_shows_the_model_the_targets_tree(github, monkeypatch):
    client, store, _ = github()
    client.list_tree = lambda ref=None: ["src/wallet.py", "README.md"]
    humans = _recording_model(monkeypatch, plan(task("root")))

    result = plan_node({"objective": "make it work"}, source=client)

    assert set(result["tasks"]) == {"root"}
    assert "src/wallet.py" in humans[0] and "README.md" in humans[0]


def test_a_failed_tree_read_does_not_fail_the_plan(github, monkeypatch):
    """Pinned: the listing is advisory context, never a blocker. A 502 from the
    trees API costs the prompt its listing - nothing else - because a planner
    that refused to plan over a transient read error would be a regression."""
    client, store, _ = github()

    def boom(ref=None):
        raise GitHubError("GET /git/trees/main -> 502")

    client.list_tree = boom
    humans = _recording_model(monkeypatch, plan(task("root")))

    result = plan_node({"objective": "make it work"}, source=client)

    assert set(result["tasks"]) == {"root"}
    assert numbers_of(store) == [1]
    assert humans[0] == "Objective:\nmake it work"


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
            "labels": [],
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


# --------------------------------------------------------------------------
# A task can declare its stack (#98)
# --------------------------------------------------------------------------


def entry_for(draft: Draft, *, stack: str = DEFAULT_STACK) -> LedgerEntry:
    """The ledger entry an issue written from `draft` reads back as.

    Built rather than round-tripped through `parse_contract` so the stack under
    test is the one named here, not one derived from the body the draft would
    render - which is the value `matches` is being asked about.
    """
    return LedgerEntry(
        number=7,
        title="t",
        task_id=draft.task_id,
        attempt=0,
        goal=draft.goal,
        files=draft.files,
        verify=draft.verify,
        blocked_by=(),
        stack=stack,
        # A state is not a label any more (#152) and is not on the entry at all.
        # `matches` compares the contract; the labels are the routing hints an
        # issue happens to carry, and an empty set is what a `Draft` implies.
        labels=frozenset(),
    )


def test_a_draft_that_says_nothing_matches_an_issue_that_says_nothing():
    """A Python plan against Python issues must still report "unchanged", or
    every replan would rewrite every body for no reason."""
    draft = Draft(task_id="t", goal="g", files=("src/a.py",), verify="pytest -q")
    entry = entry_for(draft)

    assert draft.matches(entry, ())


def test_a_replan_that_changes_a_tasks_stack_is_not_unchanged():
    """Without this row the replan writes nothing, reports the task unchanged,
    and the task keeps running on the old toolchain while the plan says
    otherwise — the kind of divergence that surfaces two stacks later."""
    draft = Draft(task_id="t", goal="g", files=("src/a.py",), verify="pytest -q", stack="node")
    entry = entry_for(draft, stack="python")

    assert not draft.matches(entry, ())


def test_a_draft_declaring_python_matches_an_issue_that_declared_nothing():
    """`None` and `"python"` are the same target, and the comparison is against
    the *resolved* entry stack. Reading them as different would rewrite every
    body once, on the first replan after this shipped."""
    draft = Draft(task_id="t", goal="g", files=("src/a.py",), verify="pytest -q", stack="python")
    entry = entry_for(draft, stack="python")

    assert draft.matches(entry, ())


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("node", "node"),
        ("Node", "node"),
        ("  react  ", "react"),
        ("`python`", "python"),
        ("rust", None),        # a stack this vocabulary does not have
        ("javascript", None),  # a language, not one of the ids
        ("", None),
        (None, None),
    ],
)
def test_a_models_stack_answer_is_normalised_or_dropped(answer, expected):
    """Dropped rather than raised.

    The alternative is one hallucinated word in one task failing
    `parse_contract` for the whole plan *after* the issues have been written. A
    task with no `## Stack` runs on the default, which is exactly what it did
    before the field existed.
    """
    drafts, _ = normalise(
        [PlannedTask(id="t", goal="g", files=["src/a.py"], stack=answer)],
        verify="pytest -q",
    )

    assert drafts[0].stack == expected


def test_a_dropped_stack_answer_still_yields_a_body_that_parses():
    """The reason dropping is safe: `render_body` omits the section, so a body
    never carries a value the parser would refuse to read back."""
    drafts, _ = normalise(
        [PlannedTask(id="t", goal="g", files=["src/a.py"], stack="rust")],
        verify="pytest -q",
    )

    body = drafts[0].body()

    assert "## Stack" not in body
    assert parse_contract(7, body).stack is None


def test_a_declared_stack_round_trips_through_the_body():
    """The round-trip is the planner's acceptance criterion, and this field is
    no exception: whatever is written must read back identically."""
    drafts, _ = normalise(
        [PlannedTask(id="t", goal="g", files=["app/page.tsx"], stack="react")],
        verify="npm test",
    )

    assert parse_contract(7, drafts[0].body()).stack == "react"


# --------------------------------------------------------------------------
# The pytest-gate repair
# --------------------------------------------------------------------------


def test_a_pytest_gate_with_no_test_file_gains_one_beside_its_module():
    """Observed on the first greenfield plan: a Goal demanding "the test must
    assert ..." over a `## Files` of one module. A worker may only write the
    listed files, so pytest collected nothing (exit 5) and the task burned its
    whole retry budget on a contract that was unwinnable when written."""
    drafts, _ = normalise(
        [PlannedTask(id="record-expense", goal="store expenses", files=["expenses.py"])],
        verify=VERIFY,
    )

    assert drafts[0].files == ("expenses.py", "test_expenses.py")


def test_the_repair_keeps_the_test_beside_a_nested_module():
    drafts, _ = normalise(
        [PlannedTask(id="t", goal="g", files=["src/pkg/budget.py"])], verify=VERIFY
    )

    assert drafts[0].files == ("src/pkg/budget.py", "src/pkg/test_budget.py")


@pytest.mark.parametrize(
    "files",
    [
        ["expenses.py", "test_expenses.py"],      # already collectable
        ["tests/expenses_test.py", "expenses.py"],  # suffix convention counts too
    ],
)
def test_a_task_that_already_carries_a_test_file_is_left_alone(files):
    drafts, _ = normalise([PlannedTask(id="t", goal="g", files=files)], verify=VERIFY)

    assert drafts[0].files == tuple(files)


def test_a_lookalike_module_name_does_not_count_as_a_test_file():
    """`contest_rules.py` neither starts with test_ nor ends with _test."""
    drafts, _ = normalise(
        [PlannedTask(id="t", goal="g", files=["contest_rules.py"])], verify=VERIFY
    )

    assert drafts[0].files == ("contest_rules.py", "test_contest_rules.py")


def test_a_non_pytest_gate_is_not_repaired():
    drafts, _ = normalise(
        [PlannedTask(id="t", goal="g", files=["app.js"])], verify="npm test"
    )

    assert drafts[0].files == ("app.js",)


def test_a_react_gate_keeps_the_modules_own_extension():
    """`.test.jsx`, not `.test.js` (#293). A `.jsx` component tested from a `.js`
    file is not transformed by `@vitejs/plugin-react`, so a repair that dropped
    the `x` would add a file whose first JSX token is a syntax error - a task
    made unwinnable by the thing meant to rescue it."""
    drafts, _ = normalise(
        [PlannedTask(id="t", goal="g", files=["src/TodoList.jsx"])],
        verify=STACK_VERIFY["react"],
    )

    assert drafts[0].files == ("src/TodoList.jsx", "test/TodoList.test.jsx")


def test_a_node_gate_gains_a_test_its_guard_can_actually_find():
    """`test/`, not beside the module, and the node gate is the reason.

    It reads `test -n "$(ls test/*.test.js)" && node --test`, so a test written
    anywhere else does not merely go uncollected - the guard finds no test files,
    short-circuits, and the gate exits non-zero having run nothing. Asserting the
    directory is asserting that the repaired task can pass.
    """
    drafts, _ = normalise(
        [PlannedTask(id="t", goal="g", files=["src/todo.js"])],
        verify=STACK_VERIFY["node"],
    )

    assert drafts[0].files == ("src/todo.js", "test/todo.test.js")


@pytest.mark.parametrize("stack", ["python", "node", "react"])
def test_every_shipped_stacks_gate_reaches_this_repair(stack):
    """The wiring assertion, generalised. The repair keyed on the substring
    `pytest`, so it was inert for two of the three stacks it shipped with - and
    the JS half fails quietly: `vitest run` collects the scaffold's suite,
    passes, and merges an untested component."""
    source = {"python": "mod.py", "node": "mod.js", "react": "mod.jsx"}[stack]

    drafts, _ = normalise(
        [PlannedTask(id="t", goal="g", files=[source])], verify=STACK_VERIFY[stack]
    )

    assert len(drafts[0].files) == 2, f"{stack} was not repaired"


def test_the_planner_prompt_carries_the_stacks_own_constraints():
    """"Plausible for a react project" is a statement about paths, and the model
    read it as one: it planned "Define a TypeScript interface" against an image
    with no TypeScript in it and no network to fetch one. The bootstrap task was
    already told; the planner deciding what the tasks *are* was not."""
    prompt = planner.system_prompt(verify=STACK_VERIFY["react"], stack="react")

    assert "TypeScript" in prompt
    assert ".tsx" in prompt
    assert "@testing-library/react" in prompt


def test_the_planner_prompt_says_nothing_extra_when_the_stack_is_unknown():
    """A `--repo` run against someone else's repository has no stack table entry
    and must not be told about one."""
    prompt = planner.system_prompt(verify="make check")

    assert "TypeScript" not in prompt
    assert "the project's existing stack" in prompt


def test_the_greenfield_python_gate_reaches_this_repair(monkeypatch):
    """The repair above was correct and, for two months, unreachable (#293).

    Every test in this section passes a pytest command in by hand, and the
    repair keys on the literal substring `pytest`. The gate a *greenfield python
    run* actually shipped was `python3 -m unittest discover -q`, which contains
    no such substring - so on the runs that needed the repair most, it was dead
    code. `create-bestiary-module` was planned live with a Goal demanding
    comprehensive unit tests over a `## Files` of one module, got no test file,
    and burned its budget exactly as the docstring above predicts.

    So this asserts the wiring rather than the behaviour: the table and the
    repair have to keep agreeing, and the two live one import apart.
    """
    drafts, _ = normalise(
        [PlannedTask(id="create-bestiary-module", goal="store creatures", files=["bestiary.py"])],
        verify=STACK_VERIFY["python"],
    )

    assert drafts[0].files == ("bestiary.py", "test_bestiary.py")


def test_a_task_with_no_python_file_is_left_alone():
    """Nothing sane to derive: docs and configs are covered by other tasks'
    tests, and inventing `test_README.py` would be worse than the gap."""
    drafts, _ = normalise(
        [PlannedTask(id="t", goal="g", files=["README.md"])], verify=VERIFY
    )

    assert drafts[0].files == ("README.md",)
