"""Which issues may be dispatched, and which are still waiting.

`docs/issue-contract.md` §4 gives this module exactly two rows of the label
state machine - `blocked -> ready` when every `## Blocked by` reference is
closed as completed, and `ready -> blocked` when one of them is not. This
replaces v1's in-memory set arithmetic in
[`graph.py`](../graph.py)'s `fan_out`, where "done" was a task id in a Python
set. In v2 the answer lives on GitHub, and every way of getting it wrong has
the same shape: an issue that looks fine and never runs, or one that runs
before its prerequisite landed.

Four decisions carry the module, and each of them is one of those failure
modes:

**A cycle raises, immediately.** Two issues waiting on each other means neither
is ever ready, the dispatcher finds nothing to do, the judge sees no progress
and the run sits there looking healthy - the worst failure this system has,
because nothing in the labels or the logs says anything is wrong. So
`compute_readiness` refuses to return a plan at all: `DependencyCycleError`
names the ring and the cycle aborts. Only *live* edges count, though - an edge
onto a dependency that is already closed as completed is discharged and cannot
hang anything, and failing a run over a ring of finished work would be a false
alarm.

**Closed is not done.** GitHub's `state_reason` distinguishes "closed as
completed" from "closed as not planned", and reading only `state == "closed"`
lets a cancelled prerequisite unblock everything downstream - work dispatched
against a foundation somebody explicitly decided not to build. Only
`completed` (and the `null` of a pre-`state_reason` closure) satisfies a
dependency; `not_planned`, `duplicate` and any reason GitHub adds later do not.

**An unresolvable reference is an error, never a satisfied dependency.** A
`#404` that does not exist, or a number that turns out to be a pull request,
is a fact nobody can check. Treating it as met dispatches the dependant;
treating it as merely unmet hides a typo behind a plausible `swarm:blocked`.
It does both: the issue is held at `swarm:blocked` *and* the error is
reported, because §4 gives `any -> failed` to the reconciler and not to this
module. (Cross-repo refs never reach here - `ledger.parse_contract` rejects
them at the parse.)

**Only `ready` and `blocked` are ever written.** An issue that is claimed, in
review, done or failed is somebody else's row in the state machine; relabelling
one because its dependency graph changed would yank an issue out from under a
running container. Those entries still take part in the graph - they are what
everything else is waiting on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .client import GitHubClient, GitHubHTTPError
from .ledger import Ledger, load_ledger

READY = "swarm:ready"
BLOCKED = "swarm:blocked"

# The two labels this module is allowed to write, which is also the set it is
# allowed to overwrite: everything else is another component's row of §4.
TRANSITIONABLE = frozenset({READY, BLOCKED})

# `state_reason` values that mean the work actually happened. `None` is in here
# because issues closed before GitHub shipped the field carry no reason at all,
# and reading those as unfinished would block their dependants forever. Every
# other value - `not_planned`, `duplicate`, anything added later - is treated as
# unsatisfied, so an unfamiliar reason fails towards "wait", not "go".
SATISFYING_STATE_REASONS = frozenset({"completed", None})


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ReadinessError(RuntimeError):
    """Base for everything this module raises."""


class DependencyCycleError(ReadinessError):
    """The dependency graph has a ring in it, so nothing in that ring can run.

    Aborts the cycle rather than joining `ReadinessPlan.errors`, because there
    is no partial answer worth acting on: every issue in the ring reads as
    legitimately blocked, and a run that dispatches the rest of the backlog and
    then waits forever on this is precisely the silent hang.

    `cycle` is the ring as issue numbers, first node repeated at the end.
    """

    def __init__(self, cycle: Sequence[int]) -> None:
        self.cycle = tuple(cycle)
        path = " -> ".join(f"#{number}" for number in self.cycle)
        super().__init__(f"dependency cycle in ## Blocked by: {path}")


class UnresolvableReferenceError(ReadinessError):
    """A `## Blocked by` ref that cannot be resolved to an issue in this repo.

    Collected rather than raised: one issue naming a dead number must not stop
    the other twenty from running. Carries the dependant's number because the
    reconciler posts this back as a comment on that issue and labels it
    `swarm:failed` (`docs/issue-contract.md` §1.4, §4).
    """

    def __init__(self, number: int, ref: int, reason: str) -> None:
        self.number = number
        self.ref = ref
        self.reason = reason
        super().__init__(f"issue #{number}: [Blocked by] {reason}")


# --------------------------------------------------------------------------
# Issue state
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IssueState:
    """Everything readiness needs to know about one referenced issue.

    Deliberately not `LedgerEntry`: a dependency need not be a task at all.
    Half of what this system waits on is hand-written issues carrying no
    `swarm:*` label, which are outside the ledger entirely but perfectly able
    to be open or closed.
    """

    number: int
    state: str = "open"                  # "open" | "closed"
    state_reason: str | None = None
    exists: bool = True
    is_pull_request: bool = False

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> IssueState:
        return cls(
            number=int(payload["number"]),
            state=str(payload.get("state") or "open"),
            state_reason=payload.get("state_reason"),
            is_pull_request="pull_request" in payload,
        )

    @classmethod
    def missing(cls, number: int) -> IssueState:
        return cls(number=number, exists=False)

    @property
    def closed(self) -> bool:
        return self.exists and self.state == "closed"

    @property
    def resolvable(self) -> bool:
        """False when there is nothing here that could ever satisfy anything."""
        return self.exists and not self.is_pull_request

    @property
    def satisfied(self) -> bool:
        return (
            self.resolvable
            and self.state == "closed"
            and self.state_reason in SATISFYING_STATE_REASONS
        )

    @property
    def reason(self) -> str:
        """Why this reference is unmet, in words a human reads on the issue."""
        if not self.exists:
            return f"#{self.number} does not exist"
        if self.is_pull_request:
            return f"#{self.number} is a pull request, not an issue"
        if self.state != "closed":
            return f"#{self.number} is open"
        # `not_planned` reads as "not planned": this string ends up in a comment
        # on somebody's issue, not in a log line.
        reason = (self.state_reason or "unknown").replace("_", " ")
        return f"#{self.number} was closed as {reason}"


@dataclass(frozen=True)
class UnmetRef:
    """One reference that is not satisfied, and whether that is an error.

    Every ref here blocks; only some are errors. Waiting on an open issue is
    the system working, and a reference to an issue that does not exist is not.
    """

    number: int
    reason: str
    error: bool = False


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """What one issue's label should be, and what it currently is."""

    number: int
    task_id: str
    current_label: str
    label: str
    unmet: tuple[UnmetRef, ...] = ()

    @property
    def ready(self) -> bool:
        return self.label == READY

    @property
    def changed(self) -> bool:
        return self.label != self.current_label

    @property
    def errors(self) -> tuple[UnmetRef, ...]:
        return tuple(ref for ref in self.unmet if ref.error)

    def __str__(self) -> str:
        if not self.unmet:
            return f"#{self.number} {self.task_id}: ready"
        return f"#{self.number} {self.task_id}: blocked - " + "; ".join(
            ref.reason for ref in self.unmet
        )


@dataclass(frozen=True)
class ReadinessPlan:
    """The whole answer for one pass: a verdict per transitionable entry.

    Entries that are claimed, in review, done or failed carry no verdict at
    all. Their absence is the statement that this module has nothing to say
    about them - not that they are blocked.
    """

    verdicts: tuple[Verdict, ...] = ()
    errors: tuple[UnresolvableReferenceError, ...] = ()

    @property
    def transitions(self) -> tuple[Verdict, ...]:
        """The verdicts that disagree with the label already on the issue."""
        return tuple(verdict for verdict in self.verdicts if verdict.changed)

    @property
    def ready(self) -> tuple[int, ...]:
        """Issue numbers the dispatcher (#21) may pick up."""
        return tuple(verdict.number for verdict in self.verdicts if verdict.ready)

    @property
    def blocked(self) -> tuple[int, ...]:
        return tuple(verdict.number for verdict in self.verdicts if not verdict.ready)

    def summary(self) -> str:
        return (
            f"{len(self.ready)} ready, {len(self.blocked)} blocked, "
            f"{len(self.transitions)} relabelled, {len(self.errors)} unresolvable refs"
        )


# --------------------------------------------------------------------------
# Cycle detection
# --------------------------------------------------------------------------

_WHITE, _GREY, _BLACK = 0, 1, 2


def find_cycle(edges: Mapping[int, Sequence[int]]) -> tuple[int, ...] | None:
    """Return one cycle as `(a, b, …, a)`, or None if the graph is acyclic.

    An explicit stack rather than recursion: a backlog is a fine place for a
    long chain, and blowing the interpreter's stack on one would report a
    dependency problem as a `RecursionError`. Nodes and successors are walked
    in ascending order so the same graph always names the same ring - a cycle
    that reports a different path every run is much harder to fix.
    """
    colour: dict[int, int] = {}
    for root in sorted(edges):
        if colour.get(root, _WHITE) != _WHITE:
            continue
        colour[root] = _GREY
        path = [root]
        stack = [(root, iter(sorted(edges.get(root, ()))))]
        while stack:
            node, successors = stack[-1]
            following = next(successors, None)
            if following is None:
                colour[node] = _BLACK
                stack.pop()
                path.pop()
                continue
            shade = colour.get(following, _WHITE)
            if shade == _GREY:
                # Grey means "on the current path", so the ring is the tail of
                # that path from this node back round to itself. A self-edge
                # lands here too, with a path of one.
                return tuple(path[path.index(following):] + [following])
            if shade == _BLACK or following not in edges:
                # Already finished, or a leaf we never expand: neither can
                # close a ring, and skipping leaves keeps the walk to the part
                # of the graph this repo actually owns.
                continue
            colour[following] = _GREY
            path.append(following)
            stack.append((following, iter(sorted(edges.get(following, ())))))
    return None


def _live_edges(
    ledger: Ledger, states: Mapping[int, IssueState]
) -> dict[int, tuple[int, ...]]:
    """The dependency graph restricted to edges that can still block something.

    Two prunings, and both are about not crying wolf. An edge onto a satisfied
    dependency is discharged - a ring of merged work hangs nothing, and raising
    over it would fail a run that is in fact fine. An edge onto an issue
    outside the ledger is a leaf: its body was never parsed (a hand-written
    issue need not carry a contract at all), so it has no outgoing edges we are
    entitled to invent.
    """
    numbers = {entry.number for entry in ledger.entries.values()}
    return {
        entry.number: tuple(
            ref
            for ref in entry.blocked_by
            if ref in numbers and not _state(states, ref).satisfied
        )
        for entry in ledger.entries.values()
    }


def _state(states: Mapping[int, IssueState], number: int) -> IssueState:
    """A ref nobody resolved is missing, not satisfied - the safe default."""
    return states.get(number) or IssueState.missing(number)


# --------------------------------------------------------------------------
# Computing
# --------------------------------------------------------------------------


def compute_readiness(ledger: Ledger, states: Mapping[int, IssueState]) -> ReadinessPlan:
    """Decide each transitionable entry's label. Raises on a dependency cycle.

    Pure: `states` is every referenced issue's open/closed fact, already
    resolved. Keeping the I/O out of here is what lets the interesting graphs -
    a diamond, a ring, a ref to a cancelled issue - be tested as data.
    """
    cycle = find_cycle(_live_edges(ledger, states))
    if cycle is not None:
        # Before any verdict, and instead of all of them. Every entry in the
        # ring would otherwise be reported as ordinarily blocked, which is
        # exactly how this failure hides.
        raise DependencyCycleError(cycle)

    verdicts: list[Verdict] = []
    errors: list[UnresolvableReferenceError] = []
    for entry in sorted(ledger.entries.values(), key=lambda entry: entry.number):
        unmet: list[UnmetRef] = []
        for ref in entry.blocked_by:
            state = _state(states, ref)
            if state.satisfied:
                continue
            unresolvable = not state.resolvable
            unmet.append(UnmetRef(ref, state.reason, error=unresolvable))
            if unresolvable:
                errors.append(UnresolvableReferenceError(entry.number, ref, state.reason))
        if entry.state_label not in TRANSITIONABLE or _state(states, entry.number).closed:
            # Claimed, in review, done or failed - somebody else's row of §4.
            # Or closed: `docs/architecture-v2.md` makes "a human can close a
            # task mid-run and have the swarm respect it" a feature of putting
            # the ledger on GitHub, and marking a cancelled issue ready is how
            # that feature turns into resurrected work. Either way the refs
            # were still walked above, so a dangling one is reported wherever
            # it lives.
            continue
        verdicts.append(
            Verdict(
                number=entry.number,
                task_id=entry.task_id,
                current_label=entry.state_label,
                label=BLOCKED if unmet else READY,
                unmet=tuple(unmet),
            )
        )

    return ReadinessPlan(tuple(verdicts), tuple(errors))


# --------------------------------------------------------------------------
# Resolving state from GitHub
# --------------------------------------------------------------------------


def _as_client(source: GitHubClient | str) -> GitHubClient:
    """A repo name builds a client from the environment; anything else is one."""
    return GitHubClient.from_env(source) if isinstance(source, str) else source


def referenced_numbers(ledger: Ledger) -> tuple[int, ...]:
    """Every issue number the ledger waits on, ascending and deduplicated."""
    refs = {ref for entry in ledger.entries.values() for ref in entry.blocked_by}
    return tuple(sorted(refs))


def resolve_states(
    source: GitHubClient | str,
    numbers: Iterable[int],
    *,
    issues: Iterable[Mapping[str, Any]] | None = None,
) -> dict[int, IssueState]:
    """Resolve each referenced number to an `IssueState`, missing ones included.

    One list call covers almost every ref, because the things a ledger waits on
    are overwhelmingly its own siblings; `get_issue` is the fallback for the
    rest. The list is the cheap direction - the client's conditional requests
    make a repeat read a 304 that costs no rate-limit budget, and the reconcile
    loop (#22) runs this every cycle.

    A 404 becomes `IssueState.missing` rather than an exception, because "that
    issue is not there" is an answer about one reference and not a failure of
    the read.
    """
    client = _as_client(source)
    wanted = sorted(set(numbers))
    if not wanted:
        return {}

    # `state="all"`: a dependency is satisfied precisely by being closed, so a
    # read that only saw open issues would report every met dependency as unmet
    # and block the entire backlog.
    listing = issues if issues is not None else client.list_issues(state="all")
    known = {int(payload["number"]): payload for payload in listing}

    states: dict[int, IssueState] = {}
    for number in wanted:
        payload = known.get(number)
        if payload is None:
            try:
                # `list_issues` drops pull requests, so a ref that names one
                # arrives here and is identified by the fetch, not mistaken for
                # a missing issue.
                payload = client.get_issue(number)
            except GitHubHTTPError as exc:
                if exc.status != 404:
                    raise
                states[number] = IssueState.missing(number)
                continue
        states[number] = IssueState.from_payload(payload)
    return states


# --------------------------------------------------------------------------
# Applying
# --------------------------------------------------------------------------


def apply_readiness(
    source: GitHubClient | str,
    *,
    ledger: Ledger | None = None,
    dry_run: bool = False,
) -> ReadinessPlan:
    """Compute readiness against the live tracker and write the two labels.

    Returns the plan whether or not it wrote anything; `dry_run=True` computes
    and reports without touching the repository, which is what the CLI's
    preflight and any "what would this cycle do" call wants. A raised
    `DependencyCycleError` means nothing was written - the cycle is detected
    before the first API call.
    """
    client = _as_client(source)
    if ledger is None:
        ledger = load_ledger(client)
    # The entries' own numbers as well as the ones they reference: a task a
    # human closed mid-run must not be relabelled ready, and that fact lives on
    # the issue rather than in the ledger.
    entries = tuple(entry.number for entry in ledger.entries.values())
    states = resolve_states(client, (*referenced_numbers(ledger), *entries))
    plan = compute_readiness(ledger, states)

    if not dry_run:
        for verdict in plan.transitions:
            _relabel(client, verdict)
    return plan


def _relabel(client: GitHubClient, verdict: Verdict) -> None:
    """Move one issue between `swarm:ready` and `swarm:blocked`, adding first.

    The order is the whole point. GitHub has no transaction across two label
    calls, and a crash between them leaves either two state labels or none.
    Two is repairable - §3's precedence puts `blocked` above `ready`, so the
    conservative reading wins and the reconciler cleans it up. None puts the
    issue outside the ledger entirely, where nothing ever looks at it again.
    """
    client.add_labels(verdict.number, [verdict.label])
    client.remove_label(verdict.number, verdict.current_label)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import os
    import sys

    repo = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GITHUB_REPOSITORY", "")
    report = apply_readiness(repo, dry_run=True)
    for line in report.verdicts:
        print(line)
    for problem in report.errors:
        print(f"error: {problem}")
    print(report.summary())
