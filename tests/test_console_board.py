"""The board: a read-only projection of the derived resolver into columns.

What is pinned: the columns come from `derived.resolve` and from nothing else -
a `swarm:*` label places not a single card (#158, the criterion #147's cutover
makes checkable); the column keys are the internal vocabulary; `landed` is
promoted to Verified only on post-merge CI evidence, and every weaker verdict
(pending, red, no checks, no merged PR) stays in Landed saying so; positive
verdicts are cached because a merge commit is immutable; needs-human is a
strip, not a column; the reader adopts nothing; a poll costs one issue listing
and one pull request listing, as it did under the labels; and every URL the
page will put into an href is built from the validated slug, never lifted from
a payload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from swarm.console import Console
from swarm.console_board import BoardError, BoardReader, COLUMNS
from swarm.github.branches import task_branch
from swarm.github.refs import task_ref
from swarm.orchestrator.derived import ContainerFact

HOST = {"Host": "127.0.0.1:8117"}
MARKER = "<!-- apiary:task id={tid} attempt={attempt} -->"


def issue(number: int, label: str, *, tid: str = "", attempt: int = 0,
          title: str = "", closed: bool = False, state_reason: str | None = None,
          blocked_by: str = "(none)") -> dict[str, Any]:
    """One ledger issue. `label` is still required - a state label is what makes
    an issue part of the ledger at all - but nothing in these tests may expect
    it to place the card: the world facts do."""
    tid = tid or f"task-{number}"
    body = (
        f"{MARKER.format(tid=tid, attempt=attempt)}\n\n"
        f"## Goal\nDo the thing for {tid}.\n\n"
        f"## Files\n- src/{tid}.py\n\n"
        f"## Verify\npython -m pytest -q\n\n"
        f"## Blocked by\n{blocked_by}\n"
    )
    payload: dict[str, Any] = {
        "number": number,
        "title": title or f"Task {number}",
        "body": body,
        "labels": [{"name": label}],
        "state": "closed" if closed else "open",
    }
    if state_reason is not None:
        payload["state_reason"] = state_reason
    return payload


def branch(number: int, attempt: int = 0) -> str:
    """The head ref a worker for `(number, attempt)` pushed - built, not spelled."""
    return task_branch(task_ref(number), attempt)


def pr(number: int, branch: str, *, merged: bool = False, closed: bool = False,
       sha: str = "") -> dict[str, Any]:
    return {
        "number": number,
        "head": {"ref": branch},
        "state": "closed" if (closed or merged) else "open",
        "merged_at": "2026-08-17T12:00:00Z" if merged else None,
        "merge_commit_sha": sha or None,
    }


def container(number: int, *, running: bool = True, run_id: str = "r1") -> ContainerFact:
    return ContainerFact(id=f"c{number}" * 6, run_id=run_id,
                         ref=task_ref(number), running=running)


def check(conclusion: str | None, status: str = "completed") -> dict[str, Any]:
    return {"status": status, "conclusion": conclusion}


@dataclass
class FakeClient:
    issues: list[dict[str, Any]] = field(default_factory=list)
    prs: list[dict[str, Any]] = field(default_factory=list)
    checks: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    issue_reads: int = 0
    pull_reads: int = 0
    check_reads: list[str] = field(default_factory=list)
    patched: list[Any] = field(default_factory=list)

    def list_issues(self, **_: Any) -> list[dict[str, Any]]:
        self.issue_reads += 1
        return [dict(i) for i in self.issues]

    def update_issue(self, number: int, **kwargs: Any) -> dict[str, Any]:
        self.patched.append((number, kwargs))
        return {"number": number}

    def list_pull_requests(self, **_: Any) -> list[dict[str, Any]]:
        self.pull_reads += 1
        return [dict(p) for p in self.prs]

    def list_check_runs(self, ref: str) -> list[dict[str, Any]]:
        self.check_reads.append(ref)
        return list(self.checks.get(ref, []))


def reader(client: FakeClient, containers: list[ContainerFact] | None = None) -> BoardReader:
    return BoardReader(client_for=lambda repo: client,
                       containers_for=lambda: list(containers or []))


def cards(board: dict[str, Any], column: str) -> list[int]:
    return [c["number"] for c in board["columns"][column]]


# --------------------------------------------------------------------------
# The projection: columns come from the resolver
# --------------------------------------------------------------------------


def test_every_derived_state_lands_in_its_lifecycle_column():
    """Five columns, five world facts - and every label deliberately wrong, so
    a projection that read even one of them would misplace a card."""
    client = FakeClient(
        issues=[
            # Waits on #2, which has not landed: blocked.
            issue(1, "swarm:done", blocked_by="- #2"),
            # No dependency, no container, no PR: eligible.
            issue(2, "swarm:claimed"),
            # A running container carries it: claimed.
            issue(3, "swarm:blocked"),
            # An open pull request on its branch: review.
            issue(4, "swarm:ready"),
            # A merged pull request: landed (post-merge CI unread here: Landed).
            issue(5, "swarm:blocked"),
        ],
        prs=[pr(11, branch(4)), pr(12, branch(5), merged=True, sha="abc123")],
    )

    board = reader(client, containers=[container(3)]).read("me/thing")

    assert cards(board, "blocked") == [1]
    assert cards(board, "eligible") == [2]
    assert cards(board, "claimed") == [3]
    assert cards(board, "review") == [4]
    assert cards(board, "landed") == [5]
    assert cards(board, "verified") == []


def test_a_hand_edited_label_moves_no_card():
    """#147's criterion, on pixels: a task whose pull request is open sits in
    review whatever a human typed onto the issue - the board projects the
    resolver, and the resolver cannot see a label at all."""
    client = FakeClient(
        issues=[issue(4, "swarm:failed")],
        prs=[pr(11, branch(4))],
    )

    board = reader(client).read("me/thing")

    assert cards(board, "review") == [4]
    assert board["needs_human"] == []


def test_a_work_item_closed_as_completed_lands_without_a_pull_request():
    """Half of what a real plan waits on is a hand-finished issue no worker
    ever touched - `derived._landed`'s second way in."""
    client = FakeClient(issues=[issue(5, "swarm:ready", closed=True,
                                      state_reason="completed")])

    board = reader(client).read("me/thing")

    assert cards(board, "landed") == [5]
    assert board["columns"]["landed"][0]["ci"] == "none"  # no evidence, no pass


def test_a_spent_attempt_budget_is_the_strip():
    """Three attempts, three pull requests closed unmerged: needs a human. The
    evidence is the attempt number in the branch names - code host, not label."""
    client = FakeClient(
        issues=[issue(9, "swarm:ready")],
        prs=[pr(13, branch(9, attempt=3), closed=True)],
    )

    board = reader(client).read("me/thing")

    assert [c["number"] for c in board["needs_human"]] == [9]
    assert all(not board["columns"][key] for key, _ in COLUMNS)


def test_a_work_item_closed_as_not_planned_leaves_the_strip():
    """Closed-not-planned is `needs-human` to the resolver - but a *closed*
    ticket was already answered by the human it was waiting for, and a board
    that showed it forever would put a permanent red badge on every finished
    project. Observed live: #21 and #22 of the wallet demo, closed as
    superseded, still haunting the strip."""
    client = FakeClient(issues=[
        issue(9, "swarm:failed", closed=True, state_reason="not_planned"),
        issue(10, "swarm:failed"),  # open, budget spent: still wants its human
    ], prs=[pr(13, branch(10, attempt=3), closed=True)])

    board = reader(client).read("me/thing")

    assert [c["number"] for c in board["needs_human"]] == [10]
    assert all(not board["columns"][key] for key, _ in COLUMNS)


def test_an_exited_container_does_not_claim():
    """`docker ps --all` lists the worker that finished this cycle; a board
    reading it as a claim would hold every ticket in Claimed until the reaper
    arrived. Liveness, not existence (#187)."""
    client = FakeClient(issues=[issue(4, "swarm:ready")], prs=[pr(11, branch(4))])

    board = reader(client, containers=[container(4, running=False)]).read("me/thing")

    assert cards(board, "review") == [4]
    assert cards(board, "claimed") == []


def test_a_running_container_outranks_the_open_pull_request():
    """`worker/pr.py` reuses one pull request across retries, so a re-claimed
    task still has one open - and a running container is a claim about *now*."""
    client = FakeClient(issues=[issue(4, "swarm:ready")], prs=[pr(11, branch(4))])

    board = reader(client, containers=[container(4)]).read("me/thing")

    assert cards(board, "claimed") == [4]


def test_an_unreachable_container_daemon_degrades_and_says_so():
    def explode() -> list[ContainerFact]:
        raise RuntimeError("Cannot connect to the Docker daemon")

    client = FakeClient(issues=[issue(4, "swarm:ready")], prs=[pr(11, branch(4))])
    board = BoardReader(client_for=lambda repo: client,
                        containers_for=explode).read("me/thing")

    assert cards(board, "review") == [4]  # the code-host evidence still places it
    assert any("container daemon" in note for note in board["notes"])


def test_every_card_carries_the_fact_that_placed_it():
    client = FakeClient(issues=[issue(4, "swarm:ready")], prs=[pr(11, branch(4))])

    card = reader(client).read("me/thing")["columns"]["review"][0]

    assert "pull request #11 is open" in card["because"]


# --------------------------------------------------------------------------
# PR matching (unchanged rules: by the ref inside the branch, never the name)
# --------------------------------------------------------------------------


def test_a_ticket_is_matched_to_its_pr_by_the_worker_branch():
    client = FakeClient(
        issues=[issue(4, "swarm:review")],
        prs=[pr(11, branch(4)), pr(12, branch(99))],
    )

    card = reader(client).read("me/thing")["columns"]["review"][0]

    assert card["pr"] == 11
    assert card["pr_url"] == "https://github.com/me/thing/pull/11"
    assert card["url"] == "https://github.com/me/thing/issues/4"


def test_a_ticket_keeps_its_pr_link_after_the_attempt_counter_moves():
    """The board matches on the ref inside the head branch, not on
    `LedgerEntry.branch` (#144). The entry names the ticket's *current* attempt,
    so a board that compared names would drop the PR link the moment a task was
    retried - and a card that loses its link looks like a task nobody worked on."""
    client = FakeClient(
        issues=[issue(4, "swarm:review", attempt=2)],
        prs=[pr(11, branch(4, attempt=1))],
    )

    card = reader(client).read("me/thing")["columns"]["review"][0]

    assert card["pr"] == 11


def test_the_newest_pull_request_wins_when_a_ticket_has_had_several_attempts():
    """One branch per attempt means one pull request per attempt, and the board
    wants the live one. GitHub lists newest first."""
    client = FakeClient(
        issues=[issue(4, "swarm:review", attempt=1)],
        prs=[pr(12, branch(4, attempt=1)), pr(11, branch(4, attempt=0))],
    )

    card = reader(client).read("me/thing")["columns"]["review"][0]

    assert card["pr"] == 12


def test_a_pull_request_on_a_pre_144_branch_is_named_rather_than_silently_dropped():
    """A repository mid-run when the naming changed. The resolver cannot join
    the pull request to the task, so the ticket shows in Eligible with no PR
    link - honest, but "no link" and "no pull request" look identical on a
    card, so the reason is put where the operator will see it."""
    client = FakeClient(
        issues=[issue(4, "swarm:review")],
        prs=[pr(11, "swarm/issue-4"), pr(12, "renovate/urllib3-2.x")],
    )

    board = reader(client).read("me/thing")

    assert cards(board, "eligible") == [4]
    assert "pr" not in board["columns"]["eligible"][0]
    # One note, for the apiary branch. A human's branch is somebody's work, not
    # a degradation, and a permanent note about it would be noise every poll.
    assert len(board["notes"]) == 1
    assert "1 pull request(s)" in board["notes"][0]


def test_the_reader_adopts_nothing():
    """A board polling every few seconds must never edit somebody's backlog."""
    client = FakeClient(issues=[{
        "number": 7, "title": "hand-written", "body": "## Goal\nx\n\n## Files\n- a.py\n\n"
        "## Verify\npytest\n\n## Blocked by\n(none)\n",
        "labels": [{"name": "swarm:ready"}], "state": "open",
    }])

    reader(client).read("me/thing")

    assert client.patched == []


def test_a_poll_costs_one_issue_listing_and_one_pull_listing():
    """The rate budget the module docstring promises: reading the resolver
    added no API call - its other inputs are local or already in these two."""
    client = FakeClient(
        issues=[issue(1, "swarm:ready"), issue(5, "swarm:done")],
        prs=[pr(12, branch(5), merged=True, sha="abc123")],
    )

    reader(client, containers=[container(1)]).read("me/thing")

    assert client.issue_reads == 1
    assert client.pull_reads == 1
    assert client.check_reads == ["abc123"]  # one per unverified landed ticket


# --------------------------------------------------------------------------
# Verified: post-merge CI, derived and cached (unchanged)
# --------------------------------------------------------------------------


def test_green_post_merge_checks_promote_landed_to_verified():
    client = FakeClient(
        issues=[issue(5, "swarm:done")],
        prs=[pr(11, branch(5), merged=True, sha="abc123")],
        checks={"abc123": [check("success"), check("skipped")]},
    )

    board = reader(client).read("me/thing")

    assert cards(board, "verified") == [5]
    assert cards(board, "landed") == []


@pytest.mark.parametrize(
    ("checks", "said"),
    [
        ({"abc123": [check(None, status="in_progress")]}, "pending"),
        ({"abc123": [check("failure")]}, "red"),
        ({}, "none"),
    ],
)
def test_anything_short_of_green_stays_landed_and_says_why(checks, said):
    client = FakeClient(
        issues=[issue(5, "swarm:done")],
        prs=[pr(11, branch(5), merged=True, sha="abc123")],
        checks=checks,
    )

    board = reader(client).read("me/thing")

    assert cards(board, "verified") == []
    assert board["columns"]["landed"][0]["ci"] == said


def test_the_merged_pull_request_carries_the_verdict_even_after_a_newer_one():
    """The Verified column reads the *merged* pull request's merge commit. A
    newer pull request on the same ticket - a crash-orphaned retry a human has
    not closed yet - must not blank the evidence."""
    client = FakeClient(
        issues=[issue(5, "swarm:done", attempt=1)],
        prs=[pr(12, branch(5, attempt=1)),
             pr(11, branch(5, attempt=0), merged=True, sha="abc123")],
        checks={"abc123": [check("success")]},
    )

    board = reader(client).read("me/thing")

    assert cards(board, "verified") == [5]


def test_a_verified_verdict_is_cached_because_a_merge_commit_is_immutable():
    client = FakeClient(
        issues=[issue(5, "swarm:done")],
        prs=[pr(11, branch(5), merged=True, sha="abc123")],
        checks={"abc123": [check("success")]},
    )
    board_reader = reader(client)

    board_reader.read("me/thing")
    board_reader.read("me/thing")

    assert client.check_reads == ["abc123"]  # the second poll paid nothing


def test_a_pending_verdict_is_polled_again():
    client = FakeClient(
        issues=[issue(5, "swarm:done")],
        prs=[pr(11, branch(5), merged=True, sha="abc123")],
        checks={"abc123": [check(None, status="queued")]},
    )
    board_reader = reader(client)

    board_reader.read("me/thing")
    client.checks["abc123"] = [check("success")]
    board = board_reader.read("me/thing")

    assert cards(board, "verified") == [5]


def test_unreadable_pull_requests_leave_the_board_blind_not_broken():
    """A PAT minted with issues but not pull requests answers 403 on /pulls
    while the ledger read succeeds - the same 'blind, decide nothing' shape
    `orchestrator/checks.py` chose. The columns the issues alone can place
    still render; the degradation is said out loud; nothing is guessed
    Verified."""

    class NoPulls(FakeClient):
        def list_pull_requests(self, **_: Any) -> list[dict[str, Any]]:
            raise RuntimeError("403: Resource not accessible by personal access token")

    client = NoPulls(issues=[
        issue(2, "swarm:ready"),
        issue(5, "swarm:done", closed=True, state_reason="completed"),
    ])

    board = reader(client).read("me/thing")

    assert cards(board, "eligible") == [2]
    assert cards(board, "landed") == [5]
    assert cards(board, "verified") == []
    assert "pr" not in board["columns"]["landed"][0]
    assert any("Pull requests: read" in note for note in board["notes"])
    assert client.check_reads == []  # blind means asking nothing further


# --------------------------------------------------------------------------
# Refusals and the route
# --------------------------------------------------------------------------


def test_a_bad_repo_shape_is_refused_with_the_fix():
    with pytest.raises(BoardError) as caught:
        reader(FakeClient()).read("not-a-slug")

    assert "owner/name" in str(caught.value)
    assert caught.value.fix


def test_the_board_route_serves_the_projection():
    client = FakeClient(issues=[issue(2, "swarm:ready")])
    console = Console(board=reader(client))

    body = json.loads(console.render("GET", "/swarm/board?repo=me%2Fthing", HOST).body)

    assert body["repo"] == "me/thing"
    assert body["repo_url"] == "https://github.com/me/thing"
    assert [c["number"] for c in body["columns"]["eligible"]] == [2]


def test_a_github_failure_is_a_502_with_a_fix_not_a_traceback():
    def explode(repo: str) -> Any:
        raise RuntimeError("boom")

    console = Console(board=BoardReader(client_for=explode))
    response = console.render("GET", "/swarm/board?repo=me%2Fthing", HOST)

    assert response.status == 502
    body = json.loads(response.body)
    assert "boom" in body["error"] and body["fix"]


def test_the_board_route_checks_the_host_like_every_other_route():
    console = Console()

    assert console.render("GET", "/swarm/board?repo=me%2Fthing",
                          {"Host": "attacker.example"}).status == 403
