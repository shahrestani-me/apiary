"""The board: one repository's swarm tickets, in the columns of their lifecycle.

The swarm tab's log answers "what is the run doing right now"; it does not
answer "where is every ticket". Since #147 that question's authoritative answer
is the derived-state resolver (`orchestrator/derived.py`): the orchestrator
computes each task's state from the code host, the containers and the run
artifacts, and the `swarm:*` labels are written but no longer believed. So the
board projects the resolver too. A board that kept reading the labels would be
rendering the one thing the orchestrator explicitly stopped acting on - and it
would go blank the day #152 removes them.

**The columns are ADR 0001's internal vocabulary, plus one derived column.**
`blocked`, `eligible`, `claimed`, `review` and `landed` are the resolver's own
verdicts, straight out of `derived.resolve`. "Backlog" and "in progress" were
the customer's words for the customer's process and are gone from the fleet
view. **Verified** is unchanged: after a merge, the repository's CI runs once
more on the merge commit on the base branch, and a landed ticket whose merge
commit's check runs all succeeded has been tested *post-merge*. That verdict is
derived here, read-only, from `list_check_runs(merge_commit_sha)` - no state is
written anywhere, because a projection must not feed state back into the thing
it projects. `needs-human` is not a column but a strip: a ticket needing a
human is the one thing a board must never hide behind ordering.

**The poll costs what it cost.** One issue listing (the ledger), one pull
request listing, and one `check-runs` read per unverified landed ticket -
exactly the calls the label-driven board made. The resolver adds none: its
remaining inputs are a `docker ps` (local, no rate budget) and fields the two
listings already carry. `state_reason` rides on `LedgerEntry` for this reason,
and the pull request listing is `state="all"` as it always was, so merged and
closed pull requests - the evidence for `landed` and for spent attempts - are
in the read the board already paid for.

**What this board cannot see, said out loud.** The resolver reads containers
from the local daemon, so a repository being run by an orchestrator on another
machine shows its claimed tasks wherever their code-host evidence puts them -
usually `review`, since the worker's pull request opens early. A daemon that
cannot be reached at all degrades the same way and lands in `notes`, the same
"blind, not broken" shape the pull request listing already has. And the
resolver's `needs-human` is arithmetic over code-host evidence - apiary's store
can renew a budget the board cannot see (`orchestrator/authority.py` is where
that join lives, run-scoped, inside the orchestrator). The board shows the
world's answer; the orchestrator's own overlays are its own.

**Verified verdicts are cached, positively only.** A merge commit is
immutable, so "its checks succeeded" can never become false; caching it saves
one `check-runs` request per landed ticket per poll, which is what keeps a
board polling every few seconds inside a PAT's rate budget. "Pending" and
"failing" are never cached - both are exactly the states a poll exists to
watch change.

**PRs are matched by the task ref inside the branch name, not by the name.**
One `list_pull_requests` per poll associates every ticket with its pull request
by head ref, and since #144 that head ref is `apiary/<ref>-attempt-<n>`
(`github/branches.py`), which the board parses back into a `(ref, attempt)`
pair. Matching the pair's ref rather than comparing against `LedgerEntry.branch`
is the difference between a board that keeps working and one that blanks:
`LedgerEntry.branch` names the ticket's *current* attempt, so the moment the
counter moves - a pull request closed unmerged, a claim recovered - a
string comparison stops finding the PR that is right there. The ref is the half
of the name that does not move.

Branches from before #144 do not parse, and a ticket whose only pull request is
on one shows no PR link *and no review evidence* - the resolver cannot see a
pull request it cannot join to a task. That is said out loud in `notes` rather
than left to look like a ticket nobody opened a PR for.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .config import SETTINGS
from .github.branches import BRANCH_PREFIX, parse_task_branch
from .github.client import GitHubClient
from .github.ledger import LedgerEntry, load_ledger
from .github.refs import task_ref
from .orchestrator.derived import (
    LANDED,
    NEEDS_HUMAN,
    Budget,
    ContainerFact,
    PullFact,
    Verdict,
    observe,
    resolve,
)
from .taskref import TaskRef

__all__ = ["BoardError", "BoardReader", "COLUMNS", "local_containers"]

#: `owner/name` - the same shape `run.py` and `console_runs.py` accept.
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

#: Column order is lifecycle order, and the keys are what the page renders.
#: The keys **are** `derived.py`'s state strings - the projection from verdict
#: to column is the identity, plus the Verified promotion below. `needs-human`
#: is deliberately absent: it is the strip, never a column.
COLUMNS: tuple[tuple[str, str], ...] = (
    ("blocked", "Blocked"),
    ("eligible", "Eligible"),
    ("claimed", "Claimed"),
    ("review", "Review"),
    ("landed", "Landed"),
    ("verified", "Verified"),
)

#: A check run that concluded any of these did not fail. `neutral` and
#: `skipped` follow GitHub's own branch-protection semantics for required
#: checks; treating a skipped lint as a failed verification would park every
#: ticket in "landed" forever.
_PASSING = frozenset({"success", "neutral", "skipped"})


class BoardError(ValueError):
    """A refusal an operator can fix, with the fix attached."""

    def __init__(self, message: str, *, fix: str = "") -> None:
        super().__init__(message)
        self.fix = fix


def local_containers() -> list[ContainerFact]:
    """The running apiary containers this machine's daemon can name.

    The resolver's `claimed` is "a running container labelled with this task",
    and the daemon is local - a `docker ps` per poll costs no API budget, which
    is why this is not counted against the poll's rate spend. `running=True` at
    the listing, so an exited worker is never fetched and discarded: the board
    only ever asks about liveness, and `derived._claiming_container` would drop
    a stopped one anyway.

    Raises whatever the daemon raises. The caller degrades and says so; this
    function deciding "no docker means no containers" would make an unreachable
    daemon indistinguishable from an idle one.
    """
    from .containers.manager import DockerCLI, find_containers

    return [
        ContainerFact(
            id=handle.id,
            run_id=handle.run_id,
            ref=task_ref(int(handle.issue)),
            running=True,
        )
        for handle in find_containers(DockerCLI(), running=True)
        if handle.issue is not None
    ]


@dataclass
class BoardReader:
    """Read one repository's board. `client_for` and `containers_for` are the
    test seams - the second also being how a console that knows it has no
    daemon (a hosted deployment, say) would hand in an empty listing."""

    client_for: Callable[[str], Any] = GitHubClient.from_env
    containers_for: Callable[[], Iterable[ContainerFact]] = local_containers
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
        # succeeds, and a board that errored outright would hide the columns
        # the ledger alone can still fill. The resolver runs either way; with
        # no pull facts a ticket in review shows as eligible or claimed, which
        # the note owns up to.
        notes: list[str] = []
        blind = False
        try:
            pulls = client.list_pull_requests(state="all")
        except Exception as exc:  # noqa: BLE001 - degrade and say so, never guess
            pulls = []
            blind = True
            notes.append(
                f"pull requests are unreadable ({type(exc).__name__}); PR links and the "
                f"Verified column are blind and tickets in review show without their "
                f"evidence - grant the token 'Pull requests: read'"
            )
        # The container daemon is the resolver's evidence for `claimed`. A
        # machine without one - or a repository run from another machine -
        # still gets a board; the claimed tickets show where their code-host
        # evidence puts them, and the note says why.
        try:
            containers = tuple(self.containers_for())
        except Exception as exc:  # noqa: BLE001 - same shape: blind, not broken
            containers = ()
            notes.append(
                f"the container daemon is unreachable ({type(exc).__name__}); running "
                f"claims are invisible, so a claimed ticket shows where its code-host "
                f"evidence puts it"
            )

        facts, by_ref, merged_by_ref, legacy = _pull_facts(pulls)
        if legacy:
            notes.append(
                f"{legacy} pull request(s) are on pre-#144 `swarm/issue-<n>` branches; "
                f"the resolver cannot join them to a task, so their tickets show no PR "
                f"link and no review evidence until they are reopened on a "
                f"`{BRANCH_PREFIX}` branch"
            )

        entries = sorted(ledger.entries.values(), key=lambda e: e.ref)
        resolution = resolve(observe(
            # A poll is not a cycle; the board has no cycle counter and the
            # resolver only echoes the number back.
            cycle=0,
            entries=entries,
            branch_names=[str((p.get("head") or {}).get("ref") or "") for p in pulls],
            containers=containers,
            pulls=facts,
            # The board reads no run directory: results are per run and a board
            # outlives every run. `attempts_spent` is a maximum of lower
            # bounds, so the missing source lowers the bound, never corrupts it.
            budget=Budget(
                max_attempts=SETTINGS.max_attempts_per_task,
                max_total_attempts=SETTINGS.max_total_attempts_per_task,
            ),
            # Empty believes every container - the right reading for a page
            # that watches whatever this machine is running.
            live_run_ids=(),
            state_reasons={entry.ref: entry.state_reason for entry in entries},
        ))
        verdicts = resolution.by_ref

        columns: dict[str, list[dict[str, Any]]] = {key: [] for key, _ in COLUMNS}
        needs_human: list[dict[str, Any]] = []
        for entry in entries:
            verdict = verdicts[entry.ref]
            card = self._card(repo, entry, verdict, by_ref.get(entry.ref))
            if verdict.state == NEEDS_HUMAN:
                #: Open only. The strip's title is "needs a human", and a
                #: closed ticket was already answered - closed as not planned
                #: by the human it was waiting for, or superseded by a changed
                #: plan. Showing it forever would make every finished project
                #: wear a red badge for work nobody intends to do.
                if not entry.closed:
                    needs_human.append(card)
                continue
            column = verdict.state
            if column == LANDED:
                # `blind`, not "are there any notes": the Verified column is
                # unreachable only when the pull request list could not be read
                # at all. A note about something else - a ticket still on a
                # pre-#144 branch - says nothing about the merge commits of the
                # tickets that did match, and parking all of them in "landed"
                # for it would hide a whole repository's post-merge CI.
                ci = "none" if blind else self._post_merge_ci(
                    client, repo, entry.number,
                    merged_by_ref.get(entry.ref) or by_ref.get(entry.ref),
                )
                if ci == "green":
                    column = "verified"
                else:
                    card["ci"] = ci  # pending | red | none - worth showing
            columns[column].append(card)

        return {
            "repo": repo,
            "repo_url": f"https://github.com/{repo}",
            "columns": columns,
            "needs_human": needs_human,
            # Malformed hand-written issues are skipped by the loader, never
            # dispatched, and would otherwise vanish; a count keeps them real.
            "errors": [str(error) for error in ledger.errors],
            # Degradations, said out loud: what this read could not see.
            "notes": notes,
        }

    def _card(
        self,
        repo: str,
        entry: LedgerEntry,
        verdict: Verdict,
        pr: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        # Every URL is built from the validated slug and an integer, never
        # lifted from API payloads: titles are model-written text, and the
        # page puts these strings into hrefs.
        card: dict[str, Any] = {
            "number": entry.number,
            "title": entry.title,
            "task_id": entry.task_id,
            "attempt": entry.attempt,
            "url": f"https://github.com/{repo}/issues/{entry.number}",
            # The fact that decided the column, in the resolver's own sentence.
            # `Verdict.because` is written for a human from the start; a card
            # that names its state without its evidence sends the operator to
            # the issue to find out why.
            "because": verdict.because,
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
            # Landed with no merged PR on its branch: closed as completed by
            # hand, or the branch was deleted and the PR list no longer names
            # it. "No evidence" is not "verified".
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


def _pull_facts(
    pulls: list[Mapping[str, Any]],
) -> tuple[
    list[PullFact],
    dict[TaskRef, Mapping[str, Any]],
    dict[TaskRef, Mapping[str, Any]],
    int,
]:
    """Every apiary pull request, as resolver facts and as card links.

    One walk over the one listing, three views out of it: the `PullFact`s the
    resolver reads (all of them - `derived._open_pull` picks the newest attempt
    itself, and `_landed` scans for merges), the newest payload per ref for the
    card's PR link, and the newest *merged* payload per ref for the Verified
    column's `merge_commit_sha`. The two maps are built reversed so the first
    pull request GitHub listed wins, which is its newest: a ticket that has
    been through more than one attempt has more than one branch, hence more
    than one pull request, and the board wants the live one.
    """
    facts: list[PullFact] = []
    by_ref: dict[TaskRef, Mapping[str, Any]] = {}
    merged_by_ref: dict[TaskRef, Mapping[str, Any]] = {}
    legacy = 0
    for pull in pulls:
        head = str((pull.get("head") or {}).get("ref") or "")
        parsed = parse_task_branch(head)
        if parsed is None:
            # Everything else on the remote: a human's branch, and the
            # `swarm/issue-<n>` branches #144 stopped minting. Only the
            # second kind is worth a note - the first is somebody's work.
            legacy += head.startswith("swarm/issue-")
            continue
        merged = bool(pull.get("merged_at"))
        facts.append(
            PullFact(
                number=int(pull["number"]),
                ref=parsed.ref,
                attempt=parsed.attempt,
                merged=merged,
                # The listing's `state` is `open` or `closed` and a merged PR
                # is closed; `PullFact.open` checks both fields on purpose.
                closed=str(pull.get("state") or "open") != "open",
                draft=bool(pull.get("draft")),
                head_sha=str((pull.get("head") or {}).get("sha") or ""),
            )
        )
    for pull in reversed(pulls):
        head = str((pull.get("head") or {}).get("ref") or "")
        parsed = parse_task_branch(head)
        if parsed is None:
            continue
        by_ref[parsed.ref] = pull
        if pull.get("merged_at"):
            merged_by_ref[parsed.ref] = pull
    return facts, by_ref, merged_by_ref, legacy
