"""The board: one repository's swarm tickets, in the columns of their lifecycle.

The swarm tab's log answers "what is the run doing right now"; it does not
answer "where is every ticket". That question already has an authoritative
answer - the `swarm:*` state labels, which `docs/issue-contract.md` §3 makes
the protocol and which every orchestrator transition writes to GitHub before
anything else believes it. So the board is a *projection of the ledger*, read
back from GitHub per poll, never an account the console keeps for itself:
"on any disagreement, GitHub wins" applies to pixels too. A board built from
parsed log lines would drift the first time a run was started from a
terminal, or a human relabelled an issue - and it would be empty after a
console restart, which is exactly when an operator wants to know where things
stand.

**The columns are the labels, plus one derived column.** Five of the six
columns are `swarm:blocked` (backlog), `swarm:ready`, `swarm:claimed` (in
progress), `swarm:review` and `swarm:done` (merged) read straight off the
ledger. **Verified** is the one thing the label set does not say: after a
merge, the repository's CI runs once more on the merge commit on the base
branch, and a merged ticket whose merge commit's check runs all succeeded has
been tested *post-merge*. That verdict is derived here, read-only, from
`list_check_runs(merge_commit_sha)` - no seventh label is written, because a
projection must not feed state back into the thing it projects. `swarm:failed`
is not a column but a strip: a ticket needing a human is the one thing a
board must never hide behind ordering.

**Verified verdicts are cached, positively only.** A merge commit is
immutable, so "its checks succeeded" can never become false; caching it saves
one `check-runs` request per merged ticket per poll, which is what keeps a
board polling every few seconds inside a PAT's rate budget. "Pending" and
"failing" are never cached - both are exactly the states a poll exists to
watch change.

**PRs are matched by branch, not guessed.** A worker's branch is
`swarm/issue-<n>` (`LedgerEntry.branch`), so one `list_pull_requests` per
poll associates every ticket with its pull request by head ref - the same
addressing `worker/pr.py` uses to find its own PR.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .github.client import GitHubClient
from .github.ledger import LedgerEntry, load_ledger

__all__ = ["BoardError", "BoardReader", "COLUMNS", "COLUMN_BY_LABEL"]

#: `owner/name` - the same shape `run.py` and `console_runs.py` accept.
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

#: Column order is lifecycle order, and the keys are what the page renders.
COLUMNS: tuple[tuple[str, str], ...] = (
    ("backlog", "Backlog"),
    ("ready", "Ready"),
    ("in_progress", "In progress"),
    ("review", "Review"),
    ("merged", "Merged"),
    ("verified", "Verified"),
)

#: Straight off `docs/issue-contract.md` §3. `swarm:done` starts as "merged"
#: and is promoted to "verified" by the post-merge CI verdict below;
#: `swarm:failed` goes to the strip, not a column.
COLUMN_BY_LABEL: dict[str, str] = {
    "swarm:blocked": "backlog",
    "swarm:ready": "ready",
    "swarm:claimed": "in_progress",
    "swarm:review": "review",
    "swarm:done": "merged",
}

#: A check run that concluded any of these did not fail. `neutral` and
#: `skipped` follow GitHub's own branch-protection semantics for required
#: checks; treating a skipped lint as a failed verification would park every
#: ticket in "merged" forever.
_PASSING = frozenset({"success", "neutral", "skipped"})


class BoardError(ValueError):
    """A refusal an operator can fix, with the fix attached."""

    def __init__(self, message: str, *, fix: str = "") -> None:
        super().__init__(message)
        self.fix = fix


@dataclass
class BoardReader:
    """Read one repository's board. `client_for` is the test seam."""

    client_for: Callable[[str], Any] = GitHubClient.from_env
    _verified: set[tuple[str, int]] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def read(self, repo: str) -> dict[str, Any]:
        if not REPO_RE.match(repo or ""):
            raise BoardError(f"repo must be 'owner/name', got {repo!r}",
                             fix="e.g. kamyar-finlex/apiary-sandbox")
        client = self.client_for(repo)
        # `adopt=False`: a board is a reader. Adoption writes markers onto
        # hand-written issues, and a page polling every few seconds must not
        # be the thing that edits somebody's backlog.
        ledger = load_ledger(client, adopt=False)
        # Blind, not broken, when the token cannot list PRs - the same refusal
        # shape `orchestrator/checks.py` uses. A fine-grained PAT minted with
        # issues but not pull requests answers 403 here while the ledger read
        # succeeds, and a board that errored outright would hide the five
        # columns the ledger alone can still fill.
        notes: list[str] = []
        try:
            pulls = client.list_pull_requests(state="all")
        except Exception as exc:  # noqa: BLE001 - degrade and say so, never guess
            pulls = []
            notes.append(
                f"pull requests are unreadable ({type(exc).__name__}); PR links and the "
                f"Verified column are blind - grant the token 'Pull requests: read'"
            )
        by_head = {pr["head"]["ref"]: pr for pr in reversed(pulls)}

        columns: dict[str, list[dict[str, Any]]] = {key: [] for key, _ in COLUMNS}
        failed: list[dict[str, Any]] = []
        for entry in sorted(ledger.entries.values(), key=lambda e: e.number):
            pr = by_head.get(entry.branch)
            card = self._card(repo, entry, pr)
            if entry.state_label == "swarm:failed":
                failed.append(card)
                continue
            column = COLUMN_BY_LABEL.get(entry.state_label, "backlog")
            if column == "merged":
                ci = "none" if notes else self._post_merge_ci(client, repo, entry.number, pr)
                if ci == "green":
                    column = "verified"
                else:
                    card["ci"] = ci  # pending | red | none - worth showing
            columns[column].append(card)

        return {
            "repo": repo,
            "repo_url": f"https://github.com/{repo}",
            "columns": columns,
            "failed": failed,
            # Malformed hand-written issues are skipped by the loader, never
            # dispatched, and would otherwise vanish; a count keeps them real.
            "errors": [str(error) for error in ledger.errors],
            # Degradations, said out loud: what this read could not see.
            "notes": notes,
        }

    def _card(self, repo: str, entry: LedgerEntry, pr: Mapping[str, Any] | None) -> dict[str, Any]:
        # Every URL is built from the validated slug and an integer, never
        # lifted from API payloads: titles are model-written text, and the
        # page puts these strings into hrefs.
        card: dict[str, Any] = {
            "number": entry.number,
            "title": entry.title,
            "task_id": entry.task_id,
            "attempt": entry.attempt,
            "url": f"https://github.com/{repo}/issues/{entry.number}",
        }
        if pr is not None:
            card["pr"] = int(pr["number"])
            card["pr_url"] = f"https://github.com/{repo}/pull/{int(pr['number'])}"
        return card

    def _post_merge_ci(
        self, client: Any, repo: str, number: int, pr: Mapping[str, Any] | None
    ) -> str:
        """What CI said about the merge commit: green | pending | red | none."""
        with self._lock:
            if (repo, number) in self._verified:
                return "green"
        sha = (pr or {}).get("merge_commit_sha") if (pr or {}).get("merged_at") else None
        if not sha:
            # `swarm:done` with no merged PR on its branch: merged by hand, or
            # the branch was deleted and the PR list no longer names it. "No
            # evidence" is not "verified".
            return "none"
        try:
            runs = client.list_check_runs(str(sha))
        except Exception:  # noqa: BLE001 - a token without 'Checks: read' answers 403
            return "none"
        if not runs:
            return "none"  # no checks created is not the same as all passed
        if any(run.get("status") != "completed" for run in runs):
            return "pending"
        if all((run.get("conclusion") or "") in _PASSING for run in runs):
            with self._lock:
                self._verified.add((repo, number))
            return "green"
        return "red"
