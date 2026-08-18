"""The board: a read-only projection of the ledger into lifecycle columns.

What is pinned: the label→column mapping stays §3's label set; `swarm:done`
is promoted to Verified only on post-merge CI evidence, and every weaker
verdict (pending, red, no checks, no merged PR) stays in Merged saying so;
positive verdicts are cached because a merge commit is immutable; failed is a
strip, not a column; the reader adopts nothing; and every URL the page will
put into an href is built from the validated slug, never lifted from a
payload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

import pytest

from swarm.console import Console
from swarm.console_board import BoardError, BoardReader, COLUMNS

HOST = {"Host": "127.0.0.1:8117"}
MARKER = "<!-- apiary:task id={tid} attempt={attempt} -->"


def issue(number: int, label: str, *, tid: str = "", attempt: int = 0,
          title: str = "") -> dict[str, Any]:
    tid = tid or f"task-{number}"
    body = (
        f"{MARKER.format(tid=tid, attempt=attempt)}\n\n"
        f"## Goal\nDo the thing for {tid}.\n\n"
        f"## Files\n- src/{tid}.py\n\n"
        f"## Verify\npython -m pytest -q\n\n"
        f"## Blocked by\n(none)\n"
    )
    return {
        "number": number,
        "title": title or f"Task {number}",
        "body": body,
        "labels": [{"name": label}],
    }


def pr(number: int, branch: str, *, merged: bool = False,
       sha: str = "") -> dict[str, Any]:
    return {
        "number": number,
        "head": {"ref": branch},
        "merged_at": "2026-08-17T12:00:00Z" if merged else None,
        "merge_commit_sha": sha or None,
    }


def check(conclusion: str | None, status: str = "completed") -> dict[str, Any]:
    return {"status": status, "conclusion": conclusion}


@dataclass
class FakeClient:
    issues: list[dict[str, Any]] = field(default_factory=list)
    prs: list[dict[str, Any]] = field(default_factory=list)
    checks: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    check_reads: list[str] = field(default_factory=list)
    patched: list[Any] = field(default_factory=list)

    def list_issues(self, **_: Any) -> list[dict[str, Any]]:
        return [dict(i) for i in self.issues]

    def update_issue(self, number: int, **kwargs: Any) -> dict[str, Any]:
        self.patched.append((number, kwargs))
        return {"number": number}

    def list_pull_requests(self, **_: Any) -> list[dict[str, Any]]:
        return [dict(p) for p in self.prs]

    def list_check_runs(self, ref: str) -> list[dict[str, Any]]:
        self.check_reads.append(ref)
        return list(self.checks.get(ref, []))


def reader(client: FakeClient) -> BoardReader:
    return BoardReader(client_for=lambda repo: client)


def cards(board: dict[str, Any], column: str) -> list[int]:
    return [c["number"] for c in board["columns"][column]]


# --------------------------------------------------------------------------
# The projection
# --------------------------------------------------------------------------


def test_every_state_label_lands_in_its_lifecycle_column():
    client = FakeClient(issues=[
        issue(1, "swarm:blocked"),
        issue(2, "swarm:ready"),
        issue(3, "swarm:claimed"),
        issue(4, "swarm:review"),
        issue(5, "swarm:done"),
    ])

    board = reader(client).read("me/thing")

    assert cards(board, "backlog") == [1]
    assert cards(board, "ready") == [2]
    assert cards(board, "in_progress") == [3]
    assert cards(board, "review") == [4]
    assert cards(board, "merged") == [5]
    assert cards(board, "verified") == []


def test_a_closed_failed_issue_leaves_the_strip():
    """The strip's title is "needs a human". A failed issue that was *closed* -
    superseded by a changed plan, or resolved by the human it waited for - was
    already answered, and a board that showed it forever would put a permanent
    red badge on every finished project. Observed live: #21 and #22 of the
    wallet demo, closed as superseded, still haunting the strip."""
    closed = issue(9, "swarm:failed")
    closed["state"] = "closed"
    client = FakeClient(issues=[closed, issue(10, "swarm:failed")])

    board = reader(client).read("me/thing")

    assert [c["number"] for c in board["failed"]] == [10]


def test_failed_is_a_strip_not_a_column():
    client = FakeClient(issues=[issue(9, "swarm:failed")])

    board = reader(client).read("me/thing")

    assert [c["number"] for c in board["failed"]] == [9]
    assert all(not board["columns"][key] for key, _ in COLUMNS)


def test_a_ticket_is_matched_to_its_pr_by_the_worker_branch():
    client = FakeClient(
        issues=[issue(4, "swarm:review")],
        prs=[pr(11, "swarm/issue-4"), pr(12, "swarm/issue-99")],
    )

    card = reader(client).read("me/thing")["columns"]["review"][0]

    assert card["pr"] == 11
    assert card["pr_url"] == "https://github.com/me/thing/pull/11"
    assert card["url"] == "https://github.com/me/thing/issues/4"


def test_the_reader_adopts_nothing():
    """A board polling every few seconds must never edit somebody's backlog."""
    client = FakeClient(issues=[{
        "number": 7, "title": "hand-written", "body": "## Goal\nx\n\n## Files\n- a.py\n\n"
        "## Verify\npytest\n\n## Blocked by\n(none)\n",
        "labels": [{"name": "swarm:ready"}],
    }])

    reader(client).read("me/thing")

    assert client.patched == []


# --------------------------------------------------------------------------
# Verified: post-merge CI, derived and cached
# --------------------------------------------------------------------------


def test_green_post_merge_checks_promote_merged_to_verified():
    client = FakeClient(
        issues=[issue(5, "swarm:done")],
        prs=[pr(11, "swarm/issue-5", merged=True, sha="abc123")],
        checks={"abc123": [check("success"), check("skipped")]},
    )

    board = reader(client).read("me/thing")

    assert cards(board, "verified") == [5]
    assert cards(board, "merged") == []


@pytest.mark.parametrize(
    ("checks", "said"),
    [
        ({"abc123": [check(None, status="in_progress")]}, "pending"),
        ({"abc123": [check("failure")]}, "red"),
        ({}, "none"),
    ],
)
def test_anything_short_of_green_stays_merged_and_says_why(checks, said):
    client = FakeClient(
        issues=[issue(5, "swarm:done")],
        prs=[pr(11, "swarm/issue-5", merged=True, sha="abc123")],
        checks=checks,
    )

    board = reader(client).read("me/thing")

    assert cards(board, "verified") == []
    assert board["columns"]["merged"][0]["ci"] == said


def test_done_without_a_merged_pr_is_not_verified():
    """Merged by hand, or the branch is gone: no evidence is not a pass."""
    client = FakeClient(issues=[issue(5, "swarm:done")], prs=[pr(11, "swarm/issue-5")])

    board = reader(client).read("me/thing")

    assert cards(board, "merged") == [5]
    assert board["columns"]["merged"][0]["ci"] == "none"


def test_a_verified_verdict_is_cached_because_a_merge_commit_is_immutable():
    client = FakeClient(
        issues=[issue(5, "swarm:done")],
        prs=[pr(11, "swarm/issue-5", merged=True, sha="abc123")],
        checks={"abc123": [check("success")]},
    )
    board_reader = reader(client)

    board_reader.read("me/thing")
    board_reader.read("me/thing")

    assert client.check_reads == ["abc123"]  # the second poll paid nothing


def test_a_pending_verdict_is_polled_again():
    client = FakeClient(
        issues=[issue(5, "swarm:done")],
        prs=[pr(11, "swarm/issue-5", merged=True, sha="abc123")],
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
    `orchestrator/checks.py` chose. The columns still render; the degradation
    is said out loud; nothing is guessed Verified."""

    class NoPulls(FakeClient):
        def list_pull_requests(self, **_: Any) -> list[dict[str, Any]]:
            raise RuntimeError("403: Resource not accessible by personal access token")

    client = NoPulls(issues=[issue(2, "swarm:ready"), issue(5, "swarm:done")])

    board = reader(client).read("me/thing")

    assert cards(board, "ready") == [2]
    assert cards(board, "merged") == [5]
    assert cards(board, "verified") == []
    assert "pr" not in board["columns"]["merged"][0]
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
    console = Console(board=BoardReader(client_for=lambda repo: client))

    body = json.loads(console.render("GET", "/swarm/board?repo=me%2Fthing", HOST).body)

    assert body["repo"] == "me/thing"
    assert body["repo_url"] == "https://github.com/me/thing"
    assert [c["number"] for c in body["columns"]["ready"]] == [2]


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
