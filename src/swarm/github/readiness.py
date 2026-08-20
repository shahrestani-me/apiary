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

**The graph is keyed on `TaskRef`, not on issue numbers.** Every mapping here -
edges, resolved states, verdicts - is keyed on the opaque ref the adapter
minted (`swarm/taskref.py`), and the two numeric spots left are the two that
address the API: the `get_issue` fallback in `resolve_states` and the label
write in `_relabel`. Nothing in the decision path can read a ref, which is what
makes the same graph run over a tracker whose ids are `ENG-123`.

**Only `ready` and `blocked` are ever written.** An issue that is claimed, in
review, done or failed is somebody else's row in the state machine; relabelling
one because its dependency graph changed would yank an issue out from under a
running container. Those entries still take part in the graph - they are what
everything else is waiting on.

**Which entries those are is no longer read off the label (#147).** The one
thing this module used a `swarm:*` label to decide was that set, and
`compute_readiness` now takes it as `transitionable`: the task ids the caller's
authority says are waiting rather than in flight or finished
(`orchestrator/authority.py`). Nothing else here changes, because nothing else
here was ever reading a state - the dependency arithmetic has always been a
question about the code host, and `Verdict.current_label` stays the label that
is really on the issue because it is the one `_relabel` has to remove.

The division of labour is in `authority.py`'s docstring and is worth repeating
from this side: the resolver decides whether a task is *waiting*, and this
module decides which of the two waiting states it is in. It keeps that half
because it is the half it is better at - it sees a ring in the graph, it tells
an unresolvable `#404` from an open issue, and it resolves dependencies that are
not tasks in the plan at all, which the resolver's own `_landed` cannot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Iterable, Mapping, Sequence

from ..mcp.tracker import INTAKE_IS_AUTHORITATIVE
from ..taskref import TaskRef
from .client import GitHubClient, GitHubHTTPError
from .ledger import Ledger, load_ledger
from .refs import issue_number, task_ref

#: ADR 0001's two waiting states, in the internal vocabulary. They were the two
#: `swarm:*` labels this module moved an issue between until #152; the strings
#: changed, the rows of `docs/issue-contract.md` §4 did not.
READY = "eligible"
BLOCKED = "blocked"

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

    `cycle` is the ring as task refs, first node repeated at the end.
    """

    def __init__(self, cycle: Sequence[TaskRef]) -> None:
        self.cycle = tuple(cycle)
        path = " -> ".join(str(ref) for ref in self.cycle)
        super().__init__(f"dependency cycle in ## Blocked by: {path}")


class UnresolvableReferenceError(ReadinessError):
    """A `## Blocked by` ref that cannot be resolved to an issue in this repo.

    Collected rather than raised: one issue naming a dead reference must not
    stop the other twenty from running.

    Two refs, and the names are the whole distinction: `task` is the dependant,
    carried because the reconciler posts this back as a comment on *that* issue
    and labels it `swarm:failed` (`docs/issue-contract.md` §1.4, §4), and `ref`
    is the unresolvable thing it named.
    """

    def __init__(self, task: TaskRef, ref: TaskRef, reason: str) -> None:
        self.task = task
        self.ref = ref
        self.reason = reason
        super().__init__(f"issue {task}: [Blocked by] {reason}")


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

    ref: TaskRef
    state: str = "open"                  # "open" | "closed"
    state_reason: str | None = None
    exists: bool = True
    is_pull_request: bool = False

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> IssueState:
        return cls(
            ref=task_ref(int(payload["number"])),
            state=str(payload.get("state") or "open"),
            state_reason=payload.get("state_reason"),
            is_pull_request="pull_request" in payload,
        )

    @classmethod
    def missing(cls, ref: TaskRef) -> IssueState:
        return cls(ref=ref, exists=False)

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
            return f"{self.ref} does not exist"
        if self.is_pull_request:
            return f"{self.ref} is a pull request, not an issue"
        if self.state != "closed":
            return f"{self.ref} is open"
        # `not_planned` reads as "not planned": this string ends up in a comment
        # on somebody's issue, not in a log line.
        reason = (self.state_reason or "unknown").replace("_", " ")
        return f"{self.ref} was closed as {reason}"


@dataclass(frozen=True)
class UnmetRef:
    """One reference that is not satisfied, and whether that is an error.

    Every ref here blocks; only some are errors. Waiting on an open issue is
    the system working, and a reference to an issue that does not exist is not.
    """

    ref: TaskRef
    reason: str
    error: bool = False


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """What state one issue should be waiting in, and what it is waiting in now.

    States rather than labels since #152. `current_state` is what the cycle's
    belief holds, not what an issue carries - there is nothing on an issue to
    carry it any more - so `changed` is "this readiness pass moved the task"
    rather than "the label on GitHub is wrong".
    """

    ref: TaskRef
    task_id: str
    current_state: str
    state: str
    unmet: tuple[UnmetRef, ...] = ()

    @property
    def ready(self) -> bool:
        return self.state == READY

    @property
    def changed(self) -> bool:
        return self.state != self.current_state

    @property
    def errors(self) -> tuple[UnmetRef, ...]:
        return tuple(ref for ref in self.unmet if ref.error)

    def __str__(self) -> str:
        if not self.unmet:
            return f"{self.ref} {self.task_id}: ready"
        return f"{self.ref} {self.task_id}: blocked - " + "; ".join(
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
    def ready(self) -> tuple[TaskRef, ...]:
        """The tasks the dispatcher (#21) may pick up."""
        return tuple(verdict.ref for verdict in self.verdicts if verdict.ready)

    @property
    def blocked(self) -> tuple[TaskRef, ...]:
        return tuple(verdict.ref for verdict in self.verdicts if not verdict.ready)

    def summary(self) -> str:
        return (
            f"{len(self.ready)} ready, {len(self.blocked)} blocked, "
            f"{len(self.transitions)} relabelled, {len(self.errors)} unresolvable refs"
        )


# --------------------------------------------------------------------------
# Cycle detection
# --------------------------------------------------------------------------

_WHITE, _GREY, _BLACK = 0, 1, 2


def find_cycle(edges: Mapping[TaskRef, Sequence[TaskRef]]) -> tuple[TaskRef, ...] | None:
    """Return one cycle as `(a, b, …, a)`, or None if the graph is acyclic.

    An explicit stack rather than recursion: a backlog is a fine place for a
    long chain, and blowing the interpreter's stack on one would report a
    dependency problem as a `RecursionError`. Nodes and successors are walked
    in ascending order so the same graph always names the same ring - a cycle
    that reports a different path every run is much harder to fix. `TaskRef`
    carries that order itself, so this walk never has to know what it is
    sorting.
    """
    colour: dict[TaskRef, int] = {}
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
    ledger: Ledger, states: Mapping[TaskRef, IssueState]
) -> dict[TaskRef, tuple[TaskRef, ...]]:
    """The dependency graph restricted to edges that can still block something.

    Two prunings, and both are about not crying wolf. An edge onto a satisfied
    dependency is discharged - a ring of merged work hangs nothing, and raising
    over it would fail a run that is in fact fine. An edge onto an issue
    outside the ledger is a leaf: its body was never parsed (a hand-written
    issue need not carry a contract at all), so it has no outgoing edges we are
    entitled to invent.
    """
    known = {entry.ref for entry in ledger.entries.values()}
    return {
        entry.ref: tuple(
            ref
            for ref in entry.blocked_by
            if ref in known and not _state(states, ref).satisfied
        )
        for entry in ledger.entries.values()
    }


def _state(states: Mapping[TaskRef, IssueState], ref: TaskRef) -> IssueState:
    """A ref nobody resolved is missing, not satisfied - the safe default."""
    return states.get(ref) or IssueState.missing(ref)


# --------------------------------------------------------------------------
# Computing
# --------------------------------------------------------------------------


def compute_readiness(
    ledger: Ledger,
    states: Mapping[TaskRef, IssueState],
    *,
    transitionable: Collection[str] | None = None,
    current: Mapping[str, str] | None = None,
) -> ReadinessPlan:
    """Decide each waiting entry's state. Raises on a dependency cycle.

    Pure: `states` is every referenced issue's open/closed fact, already
    resolved. Keeping the I/O out of here is what lets the interesting graphs -
    a diamond, a ring, a ref to a cancelled issue - be tested as data.

    `transitionable` is the task ids this pass may speak about, as decided by
    whoever holds the authority on state (#147). **Required in practice since
    #152**: it used to fall back to the issue's label, and no label is written
    any more, so `None` now speaks about nothing rather than about everything.
    Task ids rather than refs, because that is the key `Belief` is built on and
    the join to a ref happens once, where the belief is built.

    `current` is what those tasks are believed to be waiting in, so a verdict can
    say whether this pass *moved* anything. It was `entry.state_label` until
    #152.
    """
    cycle = find_cycle(_live_edges(ledger, states))
    if cycle is not None:
        # Before any verdict, and instead of all of them. Every entry in the
        # ring would otherwise be reported as ordinarily blocked, which is
        # exactly how this failure hides.
        raise DependencyCycleError(cycle)

    verdicts: list[Verdict] = []
    errors: list[UnresolvableReferenceError] = []
    current = current or {}
    for entry in sorted(ledger.entries.values(), key=lambda entry: entry.ref):
        unmet: list[UnmetRef] = []
        for ref in entry.blocked_by:
            state = _state(states, ref)
            if state.satisfied:
                continue
            unresolvable = not state.resolvable
            unmet.append(UnmetRef(ref, state.reason, error=unresolvable))
            if unresolvable:
                errors.append(UnresolvableReferenceError(entry.ref, ref, state.reason))
        # Which tasks are this module's row of §4 at all. The set is the
        # caller's since #152: it used to fall back to `entry.state_label in
        # TRANSITIONABLE`, and there is no label to read now, so a caller that
        # does not say gets nothing rather than a silent everything.
        mine = entry.task_id in (transitionable or ())
        if not mine or _state(states, entry.ref).closed:
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
                ref=entry.ref,
                task_id=entry.task_id,
                current_state=current.get(entry.task_id, ""),
                state=BLOCKED if unmet else READY,
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


def referenced_refs(ledger: Ledger) -> tuple[TaskRef, ...]:
    """Every task the ledger waits on, ascending and deduplicated."""
    refs = {ref for entry in ledger.entries.values() for ref in entry.blocked_by}
    return tuple(sorted(refs))


def resolve_states(
    source: GitHubClient | str,
    refs: Iterable[TaskRef],
    *,
    issues: Iterable[Mapping[str, Any]] | None = None,
) -> dict[TaskRef, IssueState]:
    """Resolve each referenced task to an `IssueState`, missing ones included.

    One list call covers almost every ref, because the things a ledger waits on
    are overwhelmingly its own siblings; `get_issue` is the fallback for the
    rest. The list is the cheap direction - the client's conditional requests
    make a repeat read a 304 that costs no rate-limit budget, and the reconcile
    loop (#22) runs this every cycle.

    A 404 becomes `IssueState.missing` rather than an exception, because "that
    issue is not there" is an answer about one reference and not a failure of
    the read.

    **A source may say its listing is the whole answer, and then there is no
    fallback** (#151). ADR 0004 closes the tracker capability set at intake,
    comment and create, so a tracker reached over MCP has nothing to fetch one
    item *with* - and the honest response to that is a different rule rather
    than a different call: intake's answer is authoritative, and a ref it did
    not carry is one apiary does not act on. `mcp.TrackerView` sets the
    property; a plain `GitHubClient` does not have it and fetches as before.

    Probed rather than typed, `Snapshot`'s idiom, and for its reason: this
    module knows nothing about MCP and must keep knowing nothing. The cost is
    the case the fallback existed for - a `## Blocked by` line naming a *pull
    request* was identified by the fetch on the direct path and reads as missing
    here, so the task waiting on it stays blocked. That is the conservative
    direction, and a dependency on a pull request rather than on a task is
    outside `docs/issue-contract.md` §1.2 anyway.
    """
    client = _as_client(source)
    wanted = sorted(set(refs))
    if not wanted:
        return {}

    # `state="all"`: a dependency is satisfied precisely by being closed, so a
    # read that only saw open issues would report every met dependency as unmet
    # and block the entire backlog.
    listing = issues if issues is not None else client.list_issues(state="all")
    known = {task_ref(int(payload["number"])): payload for payload in listing}
    authoritative = bool(getattr(client, INTAKE_IS_AUTHORITATIVE, False))

    states: dict[TaskRef, IssueState] = {}
    for ref in wanted:
        payload = known.get(ref)
        if payload is None and authoritative:
            # See the docstring: there is no second call to make, and "not in
            # intake" is the whole answer this source has about the ref.
            states[ref] = IssueState.missing(ref)
            continue
        if payload is None:
            try:
                # `list_issues` drops pull requests, so a ref that names one
                # arrives here and is identified by the fetch, not mistaken for
                # a missing issue. This is an API call, so the ref becomes a
                # number here and nowhere above.
                payload = client.get_issue(issue_number(ref))
            except GitHubHTTPError as exc:
                if exc.status != 404:
                    raise
                states[ref] = IssueState.missing(ref)
                continue
        states[ref] = IssueState.from_payload(payload)
    return states


# --------------------------------------------------------------------------
# Applying
# --------------------------------------------------------------------------


def apply_readiness(
    source: GitHubClient | str,
    *,
    ledger: Ledger | None = None,
    transitionable: Collection[str] | None = None,
    current: Mapping[str, str] | None = None,
) -> ReadinessPlan:
    """Compute readiness against the live tracker. **Writes nothing** since #152.

    `dry_run` is gone with the write it guarded: this module moved an issue
    between `swarm:ready` and `swarm:blocked`, and those labels no longer exist.
    Keeping the parameter would have been worse than removing it - a caller
    passing `dry_run=True` would have been promised something the function no
    longer has any way to violate, and the next reader would have to prove that
    for themselves. The verdicts are the output; the cycle carries them into its
    belief.

    A raised `DependencyCycleError` still means nothing happened - the cycle is
    detected before the first API call.
    """
    client = _as_client(source)
    if ledger is None:
        ledger = load_ledger(client)
    # The entries' own numbers as well as the ones they reference: a task a
    # human closed mid-run must not be relabelled ready, and that fact lives on
    # the issue rather than in the ledger.
    entries = tuple(entry.ref for entry in ledger.entries.values())
    states = resolve_states(client, (*referenced_refs(ledger), *entries))
    return compute_readiness(
        ledger, states, transitionable=transitionable, current=current
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import os
    import sys

    repo = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GITHUB_REPOSITORY", "")
    # Writes nothing since #152, so there is no dry run to ask for.
    report = apply_readiness(repo)
    for line in report.verdicts:
        print(line)
    for problem in report.errors:
        print(f"error: {problem}")
    print(report.summary())
