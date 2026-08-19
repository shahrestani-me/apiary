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

**The resolver is not the whole authority, and the strip is where that bites.**
`orchestrator/authority.py` joins the resolver to apiary's own store, because
the resolver's `needs-human` is arithmetic over code-host evidence and apiary
can have *renewed* the budget that arithmetic is counting against. On a board
that skipped the join, the needs-human strip showed tasks the orchestrator had
already stopped considering failed - a verdict the machine has withdrawn, on
the one element this board must never let hide. So the board reads the store
too (`load_ledger(store=...)`, ADR 0002) and applies `authority.budget_spent` -
that function rather than a second copy of the give-up arithmetic, for the
reason its own docstring gives.

Two of the orchestrator's overlays are *not* reachable here, and the difference
from the renewal above is the whole point of naming them separately:

- **A revival** (`nodes/planner.revive`) grants one attempt and "resets
  nothing", so it lives only in `Reconciler`'s memory for the length of a run.
  A revived task therefore still reads capped on this board, which is the
  direction a projection should fail in - it over-reports a task wanting a
  human rather than hiding one.
- **The infrastructure streak** is not derivable at all, by
  `reconcile.infrastructure_streaks`' own argument: exit 2 does not bump the
  attempt, so N mechanical failures write one result filename and no artifact
  can count them. A task escalated on that ceiling reads here as whatever it
  would otherwise be.

**What this board cannot see, said out loud.** The resolver reads containers
from the local daemon, so a repository being run by an orchestrator on another
machine shows its claimed tasks wherever their code-host evidence puts them -
usually `review`, since the worker's pull request opens early. A daemon that
cannot be reached degrades the same way and lands in `notes`, the same "blind,
not broken" shape the pull request listing already has. The store is found the
way `console_projects.py` finds `.swarm/projects.sqlite` - a relative path from
the process's own directory - so a console started somewhere else, or watching a
repository this machine has never run, simply has no store to read. That is the
ordinary case rather than a degradation and gets no note: a project with no
store has no renewals either, so the resolver's arithmetic is the whole answer.
A store that exists and cannot be trusted is the opposite and says so.

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
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

from .config import SETTINGS
from .github.branches import BRANCH_PREFIX, parse_task_branch
from .github.client import GitHubClient
from .github.ledger import LedgerEntry, load_ledger
from .github.refs import pull_ref, task_ref
from .orchestrator.authority import WAITING, budget_spent
from .orchestrator.derived import (
    LANDED,
    NEEDS_HUMAN,
    Budget,
    ContainerFact,
    Observation,
    PullFact,
    Verdict,
    observe,
    resolve,
)
from .store import StoreMissing, TaskStore
from .taskref import TaskRef

__all__ = [
    "BoardError",
    "BoardReader",
    "COLUMNS",
    "local_containers",
    "project_store",
]

#: `authority.WAITING`, imported rather than respelled: it is the set that
#: module bounds the same store overlay to, and two copies of it would be two
#: answers to "may a budget row outrank a running container".

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


def project_store(repo: str) -> TaskStore | None:
    """This project's judgment store, or `None` when this machine has none.

    **`create=False`, and that is the load-bearing argument.** A board is a
    reader - the same rule `adopt=False` enforces one call along - and creating
    a store is a write. It would also be a write with consequences: an empty
    store reads as "every task is on attempt 0 with no blocker", so a console
    that created one beside a run in flight would hand that run a fresh retry
    budget for every task it had already given up on.

    `StoreMissing` becomes `None` rather than a refusal because it is the
    ordinary case, not a fault: a project this machine has never run has no
    store, and a project with no store has no renewals for the strip to be
    wrong about. Every other failure - a file that is not a database, a schema
    this build does not know, another project's store - propagates, because
    ADR 0002's "failing loudly is a feature" is about exactly those and the
    caller turns them into a note the operator can act on.
    """
    from .store import SqliteTaskStore

    try:
        return SqliteTaskStore.open(repo, create=False)
    except StoreMissing:
        return None


@dataclass
class BoardReader:
    """Read one repository's board. `client_for`, `containers_for` and
    `store_for` are the test seams - the last two also being how a console that
    knows it has neither daemon nor store (a hosted deployment, say) hands in
    the empty answer rather than paying for the failure every poll."""

    client_for: Callable[[str], Any] = GitHubClient.from_env
    containers_for: Callable[[], Iterable[ContainerFact]] = local_containers
    store_for: Callable[[str], TaskStore | None] = project_store
    _verified: set[tuple[str, int]] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def read(self, repo: str) -> dict[str, Any]:
        if not REPO_RE.match(repo or ""):
            raise BoardError(f"repo must be 'owner/name', got {repo!r}",
                             fix="e.g. kamyar-finlex/apiary-sandbox")
        client = self.client_for(repo)
        notes: list[str] = []
        # apiary's own judgments about its own execution: the renewal that keeps
        # a task off the strip, and the streak that keeps a given-up one on it
        # after its run directory is gone. Read before the ledger because
        # `load_ledger` is where the join happens (ADR 0002: the store holds
        # only fields the tracker never had, so it is an assignment rather than
        # a reconciliation).
        store = None
        try:
            store = self.store_for(repo)
        except Exception as exc:  # noqa: BLE001 - blind, not broken; say which
            notes.append(
                f"apiary's task store could not be read ({type(exc).__name__}: {exc}); "
                f"a task whose retry budget was renewed may show on the strip as though "
                f"it still wanted a human"
            )
        try:
            # `adopt=False`: a board is a reader. Adoption writes markers onto
            # hand-written issues, and a page polling every few seconds must not
            # be the thing that edits somebody's backlog.
            ledger = load_ledger(client, adopt=False, store=store)
        finally:
            # The connection is this poll's, not the page's: a console holding
            # one open across every poll of every project would keep a lock on
            # a file a live run is writing.
            if store is not None:
                store.close()
        # Blind, not broken, when the token cannot list PRs - the same refusal
        # shape `orchestrator/checks.py` uses. A fine-grained PAT minted with
        # issues but not pull requests answers 403 here while the ledger read
        # succeeds, and a board that errored outright would hide the columns
        # the ledger alone can still fill. The resolver runs either way; with
        # no pull facts a ticket in review shows as eligible or claimed, which
        # the note owns up to.
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
        observation = observe(
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
        )
        verdicts = _with_store(observation, entries)

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
        if entry.renewals:
            # `store.TaskJudgement.renewals`: "the one thing a human reading a
            # capped task wants to know, and the only place it was ever written
            # down was prose in a comment". Never zero-filled - a card wearing
            # "renewed 0x" says nothing and costs a line on every ticket.
            card["renewals"] = entry.renewals
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


#: Suppressing the budget rule means resolving against a cap nothing can reach,
#: rather than teaching `derived.resolve` a second mode. `authority.py` spells
#: the same sentinel for the same reason: the resolver stays a pure function of
#: an observation, the suppression is expressed in the observation, and the two
#: answers are directly comparable because they came from one input.
_UNBOUNDED = 1_000_000_000


def _with_store(
    observation: Observation, entries: Sequence[LedgerEntry]
) -> dict[TaskRef, Verdict]:
    """The resolver's verdicts, with apiary's own budget judgment applied.

    `authority.believe`'s budget overlay, and only that one: the board has no
    run memory, so `landed-stands` (which needs what this process believed last
    cycle) and the infrastructure ceiling (which is not derivable at all) are
    not its to apply. What is left is the half that lives on disk, and it runs
    in both directions:

    - The store says the budget is **spent** and the world shows nothing -
      a task whose run is over and whose pull requests were deleted resolves to
      `eligible` from scratch. Without this it would sit in Eligible looking
      like work about to start, when in fact apiary abandoned it and a human is
      the only thing that will move it.
    - The store says the budget is **not** spent and the world says it is - the
      renewal case, which is #158's review finding. `_retry_or_give_up` gives up
      on the *streak*, and a renewal resets the streak while the attempt counter
      keeps climbing, so code-host arithmetic reads a cap the orchestrator does
      not. The lenient resolution is what the task would have been without the
      budget rule - `review` while its pull request is open, `claimed` while its
      worker runs - which is a better answer than a bare "not needs-human".

    Bounded to `WAITING` in the spent direction for `authority.believe`'s
    reason: a live container or an open pull request is stronger evidence about
    now than a budget row, and `landed` outranks everything. With no store the
    entries carry the marker's legacy fields and this is very nearly the
    identity - which is why a project this machine never ran needs no note.
    """
    strict = resolve(observation).by_ref
    # Two resolutions over one input, which is the only comparison that says
    # anything about either.
    lenient = resolve(replace(
        observation,
        budget=Budget(max_attempts=_UNBOUNDED, max_total_attempts=_UNBOUNDED),
    )).by_ref
    cap = SETTINGS.max_attempts_per_task
    total_cap = SETTINGS.max_total_attempts_per_task

    verdicts: dict[TaskRef, Verdict] = {}
    for entry in entries:
        verdict = strict[entry.ref]
        # `grant=None`: an `authority.Grant` is `Reconciler` state that lapses
        # with the process, so a board cannot know a revival happened. It
        # therefore over-reports a revived task as wanting a human, which is the
        # direction a projection should fail in.
        spent = budget_spent(
            entry,
            verdict.attempts_spent,
            None,
            max_attempts=cap,
            max_total_attempts=total_cap,
        )
        if verdict.state in WAITING and spent:
            verdicts[entry.ref] = replace(
                verdict,
                state=NEEDS_HUMAN,
                because=(
                    f"apiary gave up on this task (streak={entry.streak}, "
                    f"attempt={entry.attempt}) against a cap of {cap}, and the code "
                    f"host accounts for only {verdict.attempts_spent} attempt(s) - a "
                    f"task whose run is over and whose branches are gone leaves none"
                ),
            )
        elif verdict.state == NEEDS_HUMAN and not spent:
            relaxed = lenient.get(entry.ref, verdict)
            verdicts[entry.ref] = replace(
                relaxed,
                because=(
                    f"the code host accounts for {verdict.attempts_spent} attempt(s) "
                    f"against a cap of {cap}, but apiary's own record says the budget "
                    f"is not spent (streak={entry.streak}, attempt={entry.attempt}, "
                    f"renewed {entry.renewals}x), so {relaxed.state} stands"
                ),
            )
        else:
            verdicts[entry.ref] = verdict
    return verdicts


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
                # Minted here because here is the edge: `pulls` is the listing
                # payload as GitHub sent it, and `github/refs.pull_ref` is the
                # one place a number becomes a `PullRef` (#185, #208). The card's
                # link is built from the payload further down rather than from
                # this field - a URL wants the number, not the ref.
                number=pull_ref(int(pull["number"])),
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
