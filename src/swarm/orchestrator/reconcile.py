"""One cycle: what GitHub says should be true, what is true, and the gap.

`docs/architecture-v2.md`'s orchestration loop is five steps - read, compute
readiness, dispatch, observe, judge - and this module is the body that runs
them in order and reacts to what the first step found. Readiness (#11) decides
which issues may run, the dispatcher (#21) decides which of those run now, and
neither of them looks at a container or a pull request. This one does, which is
what makes it the place where "the ledger lives on GitHub" stops being a
storage decision and becomes a behaviour.

**GitHub wins, every cycle, on every disagreement.** Not as a slogan: the
in-process state is one read of the issue list plus one `docker ps`, both taken
at the top of the cycle and both thrown away at the bottom. A human who closes
an issue mid-run, relabels one, retitles one, or edits a body while a worker is
running is not an edge case to survive - it is the feature the whole control
plane was moved onto GitHub for, and the reconciler's job is to notice within
one cycle and act. Closing an issue by hand disposes its container and the run
carries on; that is `#22`'s acceptance criterion and it falls out of taking the
issue's own `state` as authoritative over anything this process believed.

**The polling budget is the design constraint, not a footnote.** Every request
this loop makes comes out of a primary rate limit shared with every worker
pushing branches and opening PRs, and a loop that re-derives its world from
scratch per issue would spend the run's budget on looking rather than on
working. So a cycle reads the issue list **once** - `Snapshot` is that once, and
the loader, the readiness pass and this module's own rules all take their facts
from it - and the client's conditional requests (#7) make the repeat read a 304
that costs no budget at all. Everything else this module writes is proportional
to what actually changed: a label move per finished worker, not per issue. A run
of N cycles therefore costs O(N) requests, not O(N x issues), and
`tests/test_reconcile.py` asserts exactly that by counting them.

**Two facts a cycle reads, and only one of them is GitHub's.** Everything
above is about the tracker winning, and it still does - for everything the
tracker owns. It does not own the failure signature, the streak of it or the
renewal count: those are apiary's judgments about its own execution, they
cannot be derived from anything external, and #159 moved them out of the
customer's issue body into apiary's own store
(`docs/adr/0002-apiary-owns-a-thin-task-store.md`). So a cycle joins two
sources - the issue listing for what each task *is*, `Reconciler.store` for
what apiary decided about running it - and the join needs no reconciliation at
all, because the store holds only fields the tracker never had. The attempt
counter stays in the issue marker, because the worker reads it from there and
cannot reach the store; the store keeps a *stamp* of which attempt each
judgment belongs to, never a second copy of the counter.

**Two things this module needs and cannot have.** `docs/issue-contract.md` §1.4
requires the orchestrator to post a `ContractError` back as a comment on the
offending issue, and the whole loop wants to read pull requests; `GitHubClient`
has neither an issue-comment method nor `list_pull_requests`, and `client.py`
is outside #22's file set. Both are therefore **probed for** - the same shape,
and the same reason, as `worker/pr.py`'s probe for `list_pull_requests` and
`entrypoint._publish`'s probe for `pr.py`: two tickets whose file sets cannot
reach each other must not deadlock. Until they land, a comment is printed
instead of posted and `ReconcilePlan.blind` says the cycle could not see PR
state, so no rule that needs it fires. Degrading is safe; guessing is not - a
`swarm:review` issue whose PR merely could not be listed must never be read as
a PR that was closed.

**Step 5 of the loop - judge, and act on the judgement - lives here too, and
is the half that decides when a run may stop.** A cycle that changed nothing
while nothing is in flight is the only cycle worth a model swap
(`CycleReport.needs_judgement`), and the judge's answer feeds two different
consumers. A *stall* goes to `replan.py`, which rewrites the backlog. An
*exhausted ledger* goes to `goal.py`, which asks whether the objective was met
and appends follow-up tasks when it was not. Without the second one the loop
stops at plan exhaustion, which is a statement about the planner's first guess
rather than about the objective, and the operator gets a repository that is
missing whatever the planner did not think of and a run that reports success.

The three counters this needs - the previous observation, the stall count and
the rounds already spent - are held on the `Reconciler` for the length of the
run, exactly as `update_budget` is and for the same stated reason: "did it move"
is a question about two readings, GitHub only ever shows the current one, and a
restart granting a fresh budget is the safe direction. Nothing durable is keyed
to them.

**What this module deliberately does not decide.** Check runs, the merge and
the retry-with-failure-in-context are #23; stale `swarm:claimed` labels with no
container behind them are #35; `claimed -> review` belongs to the worker (#17),
which knows the PR exists at the instant it does. §4's table splits the
`review` rows between #22 and #23, and #22's own scope names two of them, so the
split implemented here is: this module reacts to a PR that has already merged or
already closed, and #23 owns everything that requires reading a check run.

**Order within a cycle: reconcile, then readiness, then dispatch.** Reconciling
first is what frees capacity - a finished worker's issue leaves `swarm:claimed`
before the dispatcher counts the cap - and the transitions this cycle wrote are
folded into the in-memory ledger (`fold`) rather than re-read, because a second
listing to observe our own writes is the one API call that buys nothing. Only
transitions that actually landed are folded; a label call that failed leaves the
ledger saying what GitHub still says.

Manual dry run against a real repo - reads only, writes nothing, spawns nothing:

    GITHUB_TOKEN=... python -m swarm.orchestrator.reconcile shahrestani-me/apiary
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Mapping, Protocol

from ..config import SETTINGS
from ..containers.manager import ContainerError, Handle, StackImages
from ..github.client import GitHubClient, GitHubError
from ..github.ledger import (
    ContractError,
    LabelRepair,
    Ledger,
    LedgerEntry,
    load_ledger,
    render_marker,
    resolve_state_label,
)
from ..github.readiness import (
    READY,
    SATISFYING_STATE_REASONS,
    DependencyCycleError,
    IssueState,
    ReadinessPlan,
    apply_readiness,
)
from ..github.refs import issue_number, task_ref
from ..run import TERMINAL_LABELS, Run, live_entries
from ..store import StoreError, TaskStore, record_judgement
from ..taskref import TaskRef
from ..worker.entrypoint import EXIT_OK
from ..worker.result import ResultRecord, summarise_dir, tail
from .dispatcher import CLAIMED, REVIEW, Capacity, DispatchReport, Spawner, dispatch

#: The two labels no other module owns, and therefore the only two spelled out
#: here: `READY` comes from readiness (#11) and `CLAIMED`/`REVIEW` from the
#: dispatcher (#21), imported rather than respelled so a rename cannot leave
#: two modules disagreeing about what a state is called. `TERMINAL_LABELS` is
#: imported for the same reason and not rebuilt from these two - `run.py`
#: decides what "finished" means, and resumption depends on that answer.
DONE = "swarm:done"
FAILED = "swarm:failed"

#: Seconds between the *starts* of two cycles, not between one ending and the
#: next beginning. A cycle that took longer than this does not then sleep on top
#: of it - the interval is a floor on the polling rate, and pacing off the end
#: of a slow cycle would quietly halve it. Deliberately not in `Settings`:
#: `config.py` is outside #22's file set, and the number belongs to this loop.
DEFAULT_INTERVAL_S = 15.0

#: The method names this module probes for on the client. Named constants
#: because the probe and the "here is what is missing" message must not drift,
#: and because grepping for either name should find this line.
COMMENT_METHOD = "create_issue_comment"
PULLS_METHOD = "list_pull_requests"

#: How many consecutive infrastructure verdicts one task may collect before it
#: stops being free and reaches a human instead.
#:
#: Three, matching `SWARM_MAX_ATTEMPTS`, and for a similar reason: one is a data
#: point, two could be a restarting daemon, and the third having the identical
#: shape is the evidence. It is not the *same* number - these are the failures
#: that never consumed an attempt, so the two budgets are independent - but a
#: task that has burned six containers on one machine fault has told a human
#: everything a seventh would.
DEFAULT_INFRASTRUCTURE_CAP = 3
INFRASTRUCTURE_CAP_ENV = "APIARY_MAX_INFRASTRUCTURE"


# --------------------------------------------------------------------------
# The read
# --------------------------------------------------------------------------


class Snapshot:
    """One cycle's view of the control plane, read at most once per endpoint.

    Not a cache with an expiry - a cycle's worth of facts, deliberately
    discarded at the end of it. Every collaborator in a cycle wants the issue
    list: `load_ledger` reads it, `readiness.resolve_states` reads it again to
    resolve `## Blocked by` refs, and the rules below read it a third time for
    each issue's open/closed state. Three listings per cycle is three times the
    budget for one set of facts, so this object is passed where a `GitHubClient`
    is expected and answers all three from one request.

    Duck-typed rather than a subclass, which `ledger.load_ledger` explicitly
    sanctions ("a test double only has to provide `list_issues` and
    `update_issue`"). Only the calls this object *changes* are written out;
    everything else falls through to the client untouched. The fall-through is
    not laziness - it is what makes a probe for a method the client has not
    grown yet (`list_pull_requests`, `create_issue_comment`) see the client's
    real answer instead of this wrapper's silence, which would turn a missing
    method into a permanently missing one.
    """

    def __init__(self, client: GitHubClient) -> None:
        self.client = client
        self._issues: list[dict[str, Any]] | None = None
        self._pulls: tuple[dict[str, Any], ...] | None = None
        self._pulls_readable = True

    # --- the cached read -------------------------------------------------

    def list_issues(self, *, state: str = "open", **kwargs: Any) -> list[dict[str, Any]]:
        """`state="all"` with no filters is the cycle's one listing; the rest delegate.

        The filtered forms are not cached because nothing in a cycle asks for
        one, and a cache keyed on "whatever arguments came first" is the kind
        that answers a `labels=` query with an unfiltered list.
        """
        if state != "all" or kwargs:
            return self.client.list_issues(state=state, **kwargs)
        if self._issues is None:
            self._issues = self.client.list_issues(state="all")
        return self._issues

    @property
    def issues(self) -> list[dict[str, Any]]:
        """Every issue in the repository, as GitHub returned it."""
        return self.list_issues(state="all")

    def states(self) -> dict[TaskRef, IssueState]:
        """Open/closed and `state_reason` per issue, at no additional cost.

        The same shape `readiness.resolve_states` produces, built from the
        listing this snapshot already holds rather than by asking again. Pull
        requests are absent because `list_issues` drops them, which is what
        readiness's own resolver relies on too.
        """
        return {
            state.ref: state
            for state in (IssueState.from_payload(payload) for payload in self.issues)
        }

    def labels(self) -> dict[TaskRef, frozenset[str]]:
        """Label names per issue, including issues the ledger refused to parse.

        A malformed issue never reaches `Ledger.entries`, so its current state
        label is not available anywhere else - and §1.4's policy is to move it
        to `swarm:failed`, which needs to know what it is moving from.
        """
        return {
            task_ref(int(payload["number"])): _label_names(payload)
            for payload in self.issues
        }

    def pull_requests(self) -> tuple[dict[str, Any], ...]:
        """Every open PR, or nothing at all if this client cannot list them.

        Open only. The merge signal does not come from here - a merged PR
        closes its issue through `Closes #<n>`, so the issue listing already
        carries it - and asking for `state="all"` would page through every pull
        request the repository has ever had, once per cycle, to learn something
        the cheaper read already said.
        """
        if self._pulls is not None:
            return self._pulls
        lister = getattr(self.client, PULLS_METHOD, None)
        if lister is None:
            self._pulls_readable = False
            self._pulls = ()
            return self._pulls
        # Only `state` is passed, for `worker/pr.py`'s reason: a future
        # `list_pull_requests` is certain to take it and may or may not take
        # `head`, and guessing wrong would be a `TypeError` in the loop body.
        self._pulls = tuple(lister(state="open") or ())
        return self._pulls

    def open_branches(self) -> frozenset[str] | None:
        """Head refs with an open PR, or `None` meaning "this cycle cannot see".

        `None` is not an empty set and the difference decides a label: an empty
        set means every `swarm:review` issue's PR is gone, and `None` means we
        did not look. Conflating them relabels the whole review queue.
        """
        pulls = self.pull_requests()
        if not self._pulls_readable:
            return None
        return frozenset(
            str((payload.get("head") or {}).get("ref") or "") for payload in pulls
        )

    # --- delegation ------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Everything this object does not shape belongs to the client.

        Guarded against the one recursion that matters: a lookup of `client`
        itself before `__init__` has set it - during `copy` or `pickle` - would
        otherwise call this method forever.
        """
        if name.startswith("_") or name == "client":
            raise AttributeError(name)
        if name == PULLS_METHOD:
            # Served from the cache, but only when the client really has it.
            # Defining this as an ordinary method would make every client look
            # capable of listing pull requests, and `pull_requests()` uses the
            # absence of this very attribute to tell "cannot see" from "nothing
            # open" - a distinction that decides whether the whole review queue
            # gets relabelled. So the wrapper appears if and only if the thing
            # it wraps does.
            inner = getattr(self.client, PULLS_METHOD)

            def cached(*, state: str = "open", **kwargs: Any) -> list[dict[str, Any]]:
                """One listing per cycle, shared by everything in it (#22)."""
                if state != "open" or kwargs:
                    return inner(state=state, **kwargs)
                return list(self.pull_requests())

            return cached
        return getattr(self.client, name)


def _label_names(issue: Mapping[str, Any]) -> frozenset[str]:
    """GitHub returns label objects; some fixtures and webhooks return strings."""
    names: list[str] = []
    for label in issue.get("labels") or ():
        names.append(label.get("name", "") if isinstance(label, Mapping) else str(label))
    return frozenset(name for name in names if name)


# --------------------------------------------------------------------------
# Collaborators
# --------------------------------------------------------------------------


class Fleet(Spawner, Protocol):
    """The three container calls a cycle makes. `ContainerManager` satisfies it.

    `Spawner` is inherited rather than restated so the dispatcher and this
    module cannot end up wanting two different `spawn`. A `Protocol` for the
    same reason every seam here is one: a reconcile test that needs a Docker
    daemon to reach the "a closed issue disposes its worker" rule is a test that
    does not run.
    """

    def find(self, *, ref: TaskRef | None = None) -> list[Handle]: ...

    def dispose(self, handle: Handle) -> str: ...


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Transition:
    """One issue's label move, with the counter write that must precede it.

    `attempt=None` means "leave the counter alone", which is not the same as
    writing the value it already has: `docs/issue-contract.md` §5 makes the
    counter a single `PATCH` of the body, and a `PATCH` nobody needed is a write
    that can lose a human's concurrent edit for nothing.
    """

    ref: TaskRef
    from_label: str
    to_label: str
    reason: str
    task_id: str = ""
    attempt: int | None = None
    comment: str = ""
    #: The failure-signature record this transition is charging for. Written
    #: to apiary's own store (#159), not to the issue body: a signature is
    #: apiary's judgment about its own execution and
    #: `docs/adr/0002-apiary-owns-a-thin-task-store.md` is where those live.
    #: `blocker` is the signature of the failure and `streak` how many
    #: consecutive attempts have now failed with it; both are only meaningful
    #: when `attempt` is written, and a transition that consumes an attempt
    #: without setting them - a stale claim, a failed check run, anything whose
    #: failure has no verify output to sign - deliberately clears the record,
    #: which downstream reads as "no previous blocker" and falls back to the
    #: pre-signature arithmetic.
    blocker: str = ""
    streak: int | None = None
    #: How many times this task's per-blocker budget has been renewed, this
    #: transition included. Carried so the store can record it; nothing here
    #: branches on it, and `store.TaskJudgement.renewals` says why adding a
    #: third input to the give-up arithmetic would be a behaviour change.
    #:
    #: A transition with no signature to record leaves this at 0, and that is
    #: the same "clears the record" direction the two fields above take: a
    #: channel that consumed an attempt without seeing why (a stale claim, a
    #: failed check run) has no basis for any part of the record, and the old
    #: marker rewrite dropped the whole of it too. Nothing decides on the
    #: count, so the cost is a number in a store somebody is reading rather
    #: than a retry granted or refused.
    renewals: int = 0
    #: This move was caused by an infrastructure verdict (exit 2), whether it
    #: re-readied the issue or escalated it at the cap. Carried as a flag
    #: rather than sniffed back out of `reason`, because `infrastructure_streaks`
    #: counts these and a counter keyed on prose is a counter that stops
    #: counting the day somebody rewords a sentence.
    infrastructure: bool = False

    def __str__(self) -> str:
        counter = "" if self.attempt is None else f", attempt {self.attempt}"
        return f"{self.ref}: {self.from_label} -> {self.to_label}{counter} ({self.reason})"


@dataclass(frozen=True)
class InfrastructurePolicy:
    """When a task that only ever fails mechanically stops being free.

    `docs/issue-contract.md` §4 makes exit 2 not consume an attempt, so a
    broken host cannot burn every issue's retry budget before a human notices.
    That rule is right and unchanged; this is the ceiling on it. Without one, a
    purely mechanical fault - a missing worker image, a denied registry -
    retries forever at no cost, and #90 widened what counts as mechanical.

    The only backstop before this was round-based stall detection, which routes
    to the **replanner**: a model, handed a broken socket as though it were a
    planning problem.
    """

    cap: int = DEFAULT_INFRASTRUCTURE_CAP

    @classmethod
    def from_env(cls) -> InfrastructurePolicy:
        """The policy this process was started with. Read once, at the call site."""
        raw = os.environ.get(INFRASTRUCTURE_CAP_ENV)
        if raw is None or not raw.strip():
            return cls()
        try:
            cap = int(raw)
        except ValueError as exc:
            # Loud on garbage, like `dispatcher._env_int` and `checks._env_flag`.
            # A mistyped cap that silently fell back would leave a run looping
            # on a machine fault while somebody believed they had bounded it.
            raise ValueError(f"{INFRASTRUCTURE_CAP_ENV}={raw!r} is not an integer") from exc
        return cls(cap=cap)

    def summary(self) -> str:
        if self.cap <= 0:
            return (
                "infrastructure policy: no ceiling; a task failing mechanically "
                f"retries forever ({INFRASTRUCTURE_CAP_ENV}={self.cap})"
            )
        return (
            f"infrastructure policy: {self.cap} consecutive infrastructure failures "
            f"escalate to a human ({INFRASTRUCTURE_CAP_ENV})"
        )


def infrastructure_streaks(
    previous: Mapping[TaskRef, int], transitions: Iterable[Transition]
) -> dict[TaskRef, int]:
    """The streak map after a cycle, folded forward from the one before it.

    Pure arithmetic, so the ceiling is testable as data and invariant 2 holds -
    nothing here reaches a model, a daemon or GitHub.

    **Counting transitions rather than results is what makes this correct.** A
    result file is one per *attempt*, and an infrastructure verdict does not
    bump the attempt - so two mechanical failures in a row write the same
    filename and the artifacts cannot tell them apart. A transition, by
    contrast, only fires when a claimed issue has a finished container to
    account for, so one infrastructure transition is exactly one infrastructure
    verdict, and re-reading an unchanged results directory produces none.

    Any other transition on an issue clears its streak: a real task failure
    after a mechanical one means the machine recovered and the run is back to
    arguing about code, which is the case the counter must not carry over.
    """
    streaks = dict(previous)
    for transition in transitions:
        if transition.infrastructure:
            streaks[transition.ref] = streaks.get(transition.ref, 0) + 1
        else:
            streaks.pop(transition.ref, None)
    return streaks


@dataclass(frozen=True)
class Disposal:
    """A container this cycle decided is finished with, and why.

    Carried separately from the transition that caused it because the two fail
    independently: GitHub being unreachable must not leak a container, and a
    daemon that will not remove one must not stop the label from moving.
    """

    ref: TaskRef
    reason: str

    def __str__(self) -> str:
        return f"{self.ref}: dispose ({self.reason})"


@dataclass(frozen=True)
class ReconcilePlan:
    """Everything one cycle would change, computed without writing anything.

    Pure, so every interesting case - a human closing an issue under a running
    worker, a worker exiting 2, a PR closed unmerged, an issue carrying two
    state labels - is testable as data rather than as a mocked API.
    """

    transitions: tuple[Transition, ...] = ()
    disposals: tuple[Disposal, ...] = ()
    repairs: tuple[LabelRepair, ...] = ()
    errors: tuple[ContractError, ...] = ()
    #: True when PR state could not be read at all this cycle - see the module
    #: docstring. Rules that need it did not run; they did not fail.
    blind: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.transitions or self.disposals or self.repairs)

    @property
    def refs(self) -> tuple[TaskRef, ...]:
        return tuple(transition.ref for transition in self.transitions)

    def summary(self) -> str:
        parts = [
            f"{len(self.transitions)} transition(s)",
            f"{len(self.disposals)} disposal(s)",
            f"{len(self.repairs)} label repair(s)",
            f"{len(self.errors)} malformed issue(s)",
        ]
        if self.blind:
            parts.append("pull request state unreadable")
        return ", ".join(parts)


def _closed_verdict(state: IssueState) -> tuple[str, str]:
    """What a closed issue becomes, and the sentence a human reads for it.

    `SATISFYING_STATE_REASONS` is imported from readiness rather than respelled
    because the two modules must agree about what "closed" means: a dependency
    that counts as met and an issue that counts as done are the same judgement,
    and a `not_planned` closure satisfies neither.
    """
    if state.state_reason in SATISFYING_STATE_REASONS:
        return DONE, "closed as completed on GitHub"
    reason = (state.state_reason or "unknown").replace("_", " ")
    return FAILED, f"closed as {reason} on GitHub"


#: How much of a failed attempt's verify output travels into the retry comment.
#: Half of `OUTPUT_TAIL_CHARS`, the bound the record itself carries: the comment
#: is prompt fodder for the next attempt, whose whole context window is 16K
#: tokens, and the useful line - the assertion, the traceback's last frame - is
#: at the tail anyway.
COMMENT_TAIL_CHARS = 2_000

#: The two shapes of "this code imports something the container does not have",
#: as CPython prints them. Deliberately this small: `diagnose` speaks only when
#: it is sure, because a wrong diagnosis is repeated verbatim to the next
#: attempt as though it were a fact.
_MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError: No module named '([^']+)'")
_IMPORT_NAME_RE = re.compile(r"ImportError: cannot import name '[^']+' from '([^']+)'")

#: The worker's own pinned syntax-failure line, exactly as
#: `worker.edit.syntax_failure` writes it. This is not a guess at CPython's
#: traceback formatting - the honesty rule holds because the worker authored
#: the sentence being matched, so recognising it is certain. Raw
#: `SyntaxError:` tracebacks from a suite's own output stay unrecognised:
#: attributing one to a file would mean parsing pytest's surrounding lines,
#: which is a guess. The line number is matched but deliberately left out of
#: the diagnosis: `signature` uses the diagnosis as the failure's identity,
#: and a syntax error that moved lines in the same file is the same blocker,
#: not budget-renewing progress.
_SYNTAX_FAILURE_RE = re.compile(r"python syntax error in ([^\s,:]+)")

#: The worker's own pinned overflow sentence, exactly as `worker.edit.
#: fit_context` writes it when the goal and the writable set alone exceed the
#: context window. Same honesty rule as the syntax line: the worker authored
#: the sentence, so recognising it is certain. The token counts are matched
#: but deliberately left out of the diagnosis - `signature` uses the diagnosis
#: as the failure's identity, and a retry whose folded-in feedback nudges the
#: estimate is the same blocker, not budget-renewing progress.
_TOO_LARGE_RE = re.compile(
    r"too large for the worker's context window "
    r"\(~\d+ tokens against a budget of \d+; SWARM_WORKER_CTX=\d+\)"
)


def diagnose(verify_output: str) -> str:
    """Classify a failed verify output into one actionable sentence, or say nothing.

    The retry comment already carries the raw tail; this is the line above it
    that tells the next attempt what to *do* about it. Only failures whose fix
    is mechanical and certain are recognised - a missing Python module, the
    failure observed to burn three identical attempts on one issue, and the
    worker's own two pinned lines (the syntax-failure line and the
    context-overflow sentence), whose shapes are certain because the worker
    wrote them - and anything else returns the empty string, because a
    guessed diagnosis would be repeated to the model as truth.

    Pure, like the rest of the planning half of this module, so every
    classification is testable as a string in and a string out.
    """
    for pattern in (_MISSING_MODULE_RE, _IMPORT_NAME_RE):
        found = pattern.findall(verify_output or "")
        if found:
            # The import may name a submodule (`sqlalchemy.orm`); the thing to
            # declare is the distribution's top-level package.
            package = found[-1].split(".")[0]
            return (
                f"missing dependency {package!r}: declare it in requirements.txt "
                "(installed before the verify runs), or use the standard library "
                "instead"
            )
    if _TOO_LARGE_RE.search(verify_output or ""):
        return (
            "the task is too large for the worker's context window: no retry with "
            "the same file set can fit, so split it into tasks with smaller "
            "`## Files` sets - or narrow this one's - and keep each task's files "
            "to what one focused change needs"
        )
    broken = _SYNTAX_FAILURE_RE.findall(verify_output or "")
    if broken:
        # `dict.fromkeys` deduplicates while keeping the worker's order; one
        # file can carry several findings and should be named once.
        files = ", ".join(dict.fromkeys(broken))
        return (
            f"syntax error in {files}: the file must parse before any test can "
            "run - rewrite the exact line quoted in the verify output"
        )
    return ""


#: The signature of an attempt whose verify output was empty. A fixed word
#: rather than a hash of the empty string, so the marker a human reads says
#: what happened - and so the PR-closed-unmerged path, which has no output at
#: all, signs consistently with itself and differently from any real failure.
EMPTY_SIGNATURE = "no-output"

#: How many hex characters of the digest a signature keeps. Enough that two
#: different failures colliding is not a practical concern for a counter whose
#: worst collision cost is giving up one retry early - §5's safe direction -
#: and short enough to live in an HTML comment a human scrolls past.
SIGNATURE_LENGTH = 10

#: The line the failure's identity is read from, scanned from the tail: an
#: exception line as CPython or pytest prints one (`ValueError: ...`,
#: `E   AssertionError: ...`), matched loosely on the type's conventional
#: suffixes because the alternative - hashing the whole tail - would make
#: every signature unique the moment a timestamp or a tmp path appears in it.
_EXCEPTION_LINE_RE = re.compile(
    r"^\s*(?:E\s+)?[A-Za-z_][A-Za-z0-9_.]*"
    r"(?:Error|Exception|Interrupt|Failure|Warning|Exit)\b.*$"
)

#: What gets stripped before hashing, in order: memory addresses, `line N`
#: references, `file.py:N` locations, and any path-shaped token. Line numbers
#: move when a model rewrites a file and paths carry attempt-specific tmp
#: directories; neither changes *what* failed, and a signature that changed
#: with them would renew a budget the blocker never released.
_NORMALISERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"0x[0-9a-fA-F]+"), "0xADDR"),
    (re.compile(r"\bline \d+\b"), "line N"),
    (re.compile(r":\d+"), ":N"),
    (re.compile(r"\S*[/\\]\S*"), "<path>"),
    (re.compile(r"\s+"), " "),
)


def signature(verify_output: str) -> str:
    """One failed attempt's identity, as a short deterministic string.

    This is what makes "the same failure again" a checkable fact rather than a
    guess. Two attempts that fail for the same reason must sign identically
    even when the incidentals move - a `ModuleNotFoundError` is the same
    missing module whichever line imports it - and two attempts that fail for
    different reasons must sign differently, because a *changed* failure is
    the evidence that the previous blocker is gone and the retry budget
    deserves renewing.

    Three tiers, most confident first:

    - `diagnose` recognised the failure: the diagnosis is the identity. It
      already names the mechanical fact (`missing dependency 'sqlalchemy'`)
      and nothing else, so line numbers and paths never enter the hash.
    - Otherwise, the last exception-shaped line of the output, normalised:
      addresses, line numbers and paths stripped, whitespace collapsed. The
      tail because that is where CPython and pytest put the verdict.
    - No such line: the last non-empty line, normalised the same way.

    Empty output signs as `EMPTY_SIGNATURE`. Pure and string-in/string-out,
    like `diagnose` above it and for the same reason: every classification is
    testable as data.
    """
    text = (verify_output or "").strip()
    if not text:
        return EMPTY_SIGNATURE
    identity = diagnose(text) or _failure_identity(text)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:SIGNATURE_LENGTH]


def _failure_identity(text: str) -> str:
    """The normalised line that names the failure. See `signature`."""
    lines = [line for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if _EXCEPTION_LINE_RE.match(line):
            return _normalise(line)
    return _normalise(lines[-1])


def _normalise(line: str) -> str:
    """Strip the incidentals that move between attempts without the failure changing."""
    line = re.sub(r"^\s*E\s+", "", line).strip()
    for pattern, replacement in _NORMALISERS:
        line = pattern.sub(replacement, line)
    return line.strip()


def retry_comment(attempt: int, reason: str, verify_output: str = "", *, renewal: str = "") -> str:
    """The feedback a retry leaves behind, structured so a worker can find it.

    The first line is the contract: it begins `apiary: attempt N failed`, which
    is what `worker.entrypoint` greps the comment stream for before a retry.
    Below it, the renewal notice when this failure differed from the last one
    (`_retry_or_give_up` writes that sentence, because only it knows the
    budgets), the diagnosis when `diagnose` recognised one, and the bounded
    tail of the verify output in a fence - the same tail discipline the result
    record follows, tightened to `COMMENT_TAIL_CHARS` because this text is
    destined for a prompt rather than a directory.
    """
    parts = [f"apiary: attempt {attempt} failed. {reason}"]
    if renewal:
        parts.append(renewal)
    finding = diagnose(verify_output)
    if finding:
        parts.append(f"Diagnosis: {finding}")
    if (verify_output or "").strip():
        clipped = tail(verify_output, COMMENT_TAIL_CHARS)
        parts.append(f"Verify output (tail):\n```\n{clipped}\n```")
    return "\n\n".join(parts)


def _retry_or_give_up(
    entry: LedgerEntry,
    reason: str,
    max_attempts: int,
    *,
    verify_output: str = "",
    max_total_attempts: int | None = None,
) -> Transition:
    """Consume an attempt, and decide whether any remain. **The budget is per
    blocker, not per task.**

    The increment is carried on the transition so it is persisted *before* the
    label goes back to `swarm:ready` (§5): a crash between the two costs an
    attempt rather than granting a free one, and a counter that can fail to
    bound retries is worse than one that over-counts. The failure signature
    and its streak ride the same transition and therefore the same body
    `PATCH`, so the record cannot say one thing while the counter says another.

    **Why the budget renews.** Issue #21 of the first live run failed three
    times on the identical `ModuleNotFoundError` and was rightly capped - but
    after a human fixed the environment and reset the counter, attempt four
    failed on a *brand-new* SyntaxError and the orchestrator gave up again,
    because the counter was back at its cap and could not see that the new
    failure proved the old blocker gone. A different signature than the last
    recorded one is progress by definition, so the consecutive-failure test
    restarts. Same signature - or no signature recorded, which is every marker
    written before the field existed - burns the budget down exactly as it
    always did.

    **The representation: `attempt` stays monotonic, `streak` counts the
    blocker.** The alternative - resetting `attempt` on renewal and keeping a
    separate total - was rejected because `attempt` is load-bearing far beyond
    this function: it names result files, orders records against the ledger
    (`record.attempt >= entry.attempt`), tells the worker it is a retry, and
    §5's whole crash-ordering argument assumes it only ever goes up. So
    `attempt` keeps counting every attempt honestly, the give-up test runs on
    `streak`, and the hard bound runs on `attempt` itself: renewal cannot make
    a task immortal, because a task that keeps failing *differently* still
    spends from `max_total_attempts` (default three full per-blocker budgets)
    and is given up whatever its latest signature says.

    Both branches carry a comment. The give-up one is for the human the issue
    just reached, and says *which* budget ran out - the same failure repeating
    and the total being spent call for different humans doing different
    things. The retry one is for the *next worker*; a renewal says so out
    loud, so the run's transcript records that the previous blocker is gone.
    `verify_output` is what the attempt's result record captured; the caller
    that has none (a PR closed unmerged) leaves the comment to carry the
    reason alone, and its signature is the fixed `EMPTY_SIGNATURE`.
    """
    attempt = entry.attempt + 1
    cap = max(int(max_attempts), 1)
    total = SETTINGS.max_total_attempts_per_task if max_total_attempts is None else max_total_attempts
    total_cap = max(int(total), cap)
    sig = signature(verify_output)
    renewed = bool(entry.blocker) and sig != entry.blocker
    # An old marker carries no streak; the attempt counter is what the streak
    # was before failures had signatures, so falling back to it preserves the
    # pre-signature arithmetic exactly (the back-compat the tests pin).
    previous_streak = entry.attempt if entry.streak is None else entry.streak
    streak = 1 if renewed else previous_streak + 1
    # Recorded, never read here. The give-up tests below run on `streak` and on
    # `attempt`, exactly as #154-#156 left them; this is the transcript of how
    # often the budget was renewed, which was previously written down only as
    # prose in a comment nobody could count.
    renewals = entry.renewals + 1 if renewed else entry.renewals

    if attempt >= total_cap:
        return Transition(
            ref=entry.ref,
            from_label=entry.state_label,
            to_label=FAILED,
            reason=f"{reason}; {attempt} attempt(s) made against a total cap of {total_cap}",
            task_id=entry.task_id,
            attempt=attempt,
            blocker=sig,
            streak=streak,
            renewals=renewals,
            comment=(
                f"apiary: giving up after {attempt} attempt(s). {reason}\n\n"
                f"The total retry budget is spent ({attempt} of {total_cap}, "
                "SWARM_MAX_TOTAL_ATTEMPTS): the failures changed along the way - each "
                "change renewed the per-blocker budget - but a task that keeps failing "
                "in new ways is not converging, and a human should look at the whole "
                "history rather than the latest error."
            ),
        )
    # Not gated on `renewed`: a renewal restarts the streak at 1, so with a cap
    # of 1 - the operator saying "never retry" - even a brand-new failure gives
    # up here, exactly as it did before failures had signatures.
    if streak >= cap:
        return Transition(
            ref=entry.ref,
            from_label=entry.state_label,
            to_label=FAILED,
            reason=(
                f"{reason}; {attempt} attempt(s) made, the last {streak} failing the "
                f"same way against a cap of {cap}"
            ),
            task_id=entry.task_id,
            attempt=attempt,
            blocker=sig,
            streak=streak,
            renewals=renewals,
            comment=(
                f"apiary: giving up after {attempt} attempt(s). {reason}\n\n"
                f"The last {streak} attempt(s) failed the same way, so another retry "
                "would buy the same failure again. If the blocker is fixed outside the "
                f"task, move this back to `{READY}` - a retry that then fails "
                "*differently* renews its own budget."
            ),
        )
    renewal = ""
    if renewed:
        renewal = (
            "A different failure than the last attempt - the previous blocker is "
            f"gone, which is progress, so the retry budget is renewed (streak "
            f"{streak} of {cap}, total {attempt} of {total_cap})."
        )
    return Transition(
        ref=entry.ref,
        from_label=entry.state_label,
        to_label=READY,
        reason=reason,
        task_id=entry.task_id,
        attempt=attempt,
        blocker=sig,
        streak=streak,
        renewals=renewals,
        comment=retry_comment(attempt, reason, verify_output, renewal=renewal),
    )


def plan_reconcile(
    ledger: Ledger,
    *,
    states: Mapping[TaskRef, IssueState] | None = None,
    open_branches: Collection[str] | None = None,
    results: Mapping[TaskRef, ResultRecord] | None = None,
    running: Collection[TaskRef] = (),
    labels: Mapping[TaskRef, frozenset[str]] | None = None,
    max_attempts: int = SETTINGS.max_attempts_per_task,
    max_total_attempts: int = SETTINGS.max_total_attempts_per_task,
    infrastructure: Mapping[TaskRef, int] | None = None,
    infrastructure_policy: InfrastructurePolicy = InfrastructurePolicy(),
) -> ReconcilePlan:
    """Compare desired state with actual state. Pure - no API call, no daemon.

    `states` is each issue's own open/closed fact, `open_branches` the head refs
    carrying an open PR (`None` when this cycle could not look), `results` the
    latest artifact record per issue and `running` the issues with a live
    container. Every one of them is a fact somebody else read; keeping the I/O
    out of here is what makes the rules assertable.

    The rules are ordered, and the order is the priority: a closed issue is out
    of the run whatever else is true of it, because a human closing an issue is
    the one input this system may never argue with.
    """
    states = states or {}
    results = results or {}
    labels = labels or {}
    # How many infrastructure verdicts each issue has collected in a row. Cross-
    # attempt state, so it cannot live in the worker - that container is one
    # attempt and then exits - and it is passed in rather than held here,
    # because this function stays pure. `infrastructure_streaks` folds the
    # answer forward and `Reconciler` carries it between cycles.
    infrastructure = infrastructure or {}
    live = set(running)

    transitions: list[Transition] = []
    disposals: list[Disposal] = []

    for entry in sorted(ledger.entries.values(), key=lambda entry: entry.ref):
        state = states.get(entry.ref)
        terminal = entry.state_label in TERMINAL_LABELS

        # 1. GitHub wins. A closed issue leaves the run, and its container goes
        #    with it - that is #22's acceptance criterion. The same rule reads
        #    the merge, because `Closes #<n>` closes the issue as completed, so
        #    "the PR merged" and "a human ticked it off" are one fact here.
        if state is not None and state.closed and not terminal:
            label, reason = _closed_verdict(state)
            transitions.append(
                Transition(
                    ref=entry.ref,
                    from_label=entry.state_label,
                    to_label=label,
                    reason=reason,
                    task_id=entry.task_id,
                )
            )
            if entry.ref in live:
                disposals.append(Disposal(entry.ref, reason))
            continue

        # 2. Terminal work keeps no container. Reached by an issue a previous
        #    cycle finished, or one a human labelled `swarm:done` by hand.
        if terminal:
            if entry.ref in live:
                disposals.append(Disposal(entry.ref, f"{entry.state_label} is terminal"))
            continue

        # 3. A worker that finished and said so. The record is written last by
        #    the worker process (`worker/result.py`), so its existence means the
        #    container is done; `attempt >= entry.attempt` is what stops a
        #    record being acted on twice, since the counter moves and the file
        #    does not.
        record = results.get(entry.ref)
        finished = record is not None and record.attempt >= entry.attempt
        if entry.state_label == CLAIMED and finished and record is not None:
            transition, disposal = _observe(
                entry,
                record,
                max_attempts,
                max_total_attempts=max_total_attempts,
                infrastructure_streak=infrastructure.get(entry.ref, 0),
                policy=infrastructure_policy,
            )
            if transition is not None:
                transitions.append(transition)
            if entry.ref in live:
                disposals.append(disposal)
            continue

        if entry.state_label != REVIEW:
            continue

        # 4. A PR that is no longer open, on an issue that is not closed - so it
        #    was closed unmerged, by a human or by #23. The attempt is consumed:
        #    the work was done and rejected, and a retry that costs nothing can
        #    be rejected forever. Skipped entirely when `open_branches` is None,
        #    because "we could not list PRs" must never read as "the PR is gone".
        if open_branches is not None and entry.branch not in open_branches:
            transitions.append(
                _retry_or_give_up(
                    entry,
                    "its pull request was closed without merging",
                    max_attempts,
                    max_total_attempts=max_total_attempts,
                )
            )
            if entry.ref in live:
                disposals.append(Disposal(entry.ref, "its pull request was closed"))
        elif finished and entry.ref in live:
            # The PR is open, so the label is right and nothing moves - but the
            # worker wrote its record, which it does last, so the container is
            # only holding a clone. Waiting for the merge to dispose it would
            # leak one container per task for the length of the review.
            disposals.append(Disposal(entry.ref, "the worker published its pull request"))

    # 5. A malformed body is labelled `swarm:failed` and the reason is posted on
    #    the issue (§1.4), once - an issue already carrying the label is left
    #    alone, or every cycle would comment on it again. These issues are not
    #    in `entries` at all, which is why their label comes from `labels`.
    for error in ledger.errors:
        # `ContractError` is a parse failure and carries the issue number the
        # parse was handed; the ref is minted here rather than there, so the
        # plan speaks one vocabulary.
        error_ref = task_ref(error.number)
        carried = labels.get(error_ref, frozenset())
        current, _ = resolve_state_label(error.number, carried)
        if current is None or current in TERMINAL_LABELS:
            continue
        transitions.append(
            Transition(
                ref=error_ref,
                from_label=current,
                to_label=FAILED,
                reason=f"malformed contract: {error.reason}",
                comment=f"apiary: this issue does not satisfy `docs/issue-contract.md`.\n\n{error}",
            )
        )

    # 6. Containers whose issue is no longer in the ledger at all - a human
    #    deleted it, or stripped its `swarm:*` label mid-run. Nothing will ever
    #    look at that work again, so the container is holding a clone for
    #    nobody.
    known = {entry.ref for entry in ledger.entries.values()}
    planned = {disposal.ref for disposal in disposals}
    for ref in sorted(live - known - planned):
        disposals.append(Disposal(ref, "its issue is no longer in the ledger"))

    return ReconcilePlan(
        transitions=tuple(transitions),
        disposals=tuple(disposals),
        repairs=ledger.repairs,
        errors=ledger.errors,
        blind=open_branches is None,
    )


def _observe(
    entry: LedgerEntry,
    record: ResultRecord,
    max_attempts: int,
    *,
    max_total_attempts: int | None = None,
    infrastructure_streak: int = 0,
    policy: InfrastructurePolicy = InfrastructurePolicy(),
) -> tuple[Transition | None, Disposal]:
    """Turn one finished worker's exit code into a label move (§4).

    Exit 0 moves no label. `claimed -> review` belongs to the worker, which
    writes it at the instant the PR exists; an exit 0 whose label did not stick
    leaves a claimed issue with an open PR, and that is #35's row, not this
    one's. Taking it here would race the worker for the same write.
    """
    if record.exit_code == EXIT_OK:
        return None, Disposal(entry.ref, "the worker published its pull request")

    detail = record.reason or record.outcome
    if record.consumes_attempt:
        return (
            _retry_or_give_up(
                entry,
                f"worker exit {record.exit_code}: {detail}",
                max_attempts,
                # The record is the only place the gate's own words survive the
                # container, and they are what the retry comment - and through
                # it, the next attempt's prompt - is made of. The failure
                # signature is computed from the same text, so "the same
                # failure again" is judged on what the gate actually said.
                verify_output=record.verify_output,
                max_total_attempts=max_total_attempts,
            ),
            Disposal(entry.ref, f"worker exit {record.exit_code}"),
        )
    # Exit 2. The task never really ran, so the attempt is not consumed - a
    # broken Ollama would otherwise burn every task's budget before anyone
    # noticed (`docs/issue-contract.md` §4).
    #
    # That rule is unchanged, and this is the ceiling on it. Not consuming an
    # attempt means a purely mechanical fault - a missing image, a denied
    # registry - retries for free, forever, and #90 widened what counts as
    # mechanical. The only backstop before this was round-based stall
    # detection, which routes to the *replanner*: a model, handed a broken
    # socket as though it were a planning problem.
    streak = infrastructure_streak + 1
    if streak >= policy.cap:
        return (
            Transition(
                ref=entry.ref,
                from_label=entry.state_label,
                to_label=FAILED,
                # The counter is deliberately left alone. The attempts were
                # never consumed and saying otherwise now would rewrite history
                # to make the escalation look like an exhausted budget.
                reason=(
                    f"{streak} consecutive infrastructure failures, most recently "
                    f"{detail!r}; this is not a coding problem and no attempt was "
                    f"ever consumed for it ({INFRASTRUCTURE_CAP_ENV} to change)"
                ),
                task_id=entry.task_id,
                comment=(
                    f"Escalated after {streak} consecutive infrastructure failures.\n\n"
                    f"The last one was: {detail}\n\n"
                    "An infrastructure verdict means the task never really ran, so no "
                    "attempt was consumed for any of them - which is why this issue "
                    "would otherwise retry forever. Fix the host, then move this back "
                    f"to `{READY}`."
                ),
                infrastructure=True,
            ),
            Disposal(entry.ref, f"worker exit 2 x{streak} (infrastructure, escalated)"),
        )
    return (
        Transition(
            ref=entry.ref,
            from_label=entry.state_label,
            to_label=READY,
            reason=f"infrastructure failure: {detail}; the attempt was not consumed",
            task_id=entry.task_id,
            infrastructure=True,
        ),
        Disposal(entry.ref, "worker exit 2 (infrastructure)"),
    )


def fold(ledger: Ledger, transitions: Iterable[Transition]) -> Ledger:
    """Apply transitions to an in-memory ledger, so the cycle need not re-read.

    Only ever called with the transitions that actually landed. Folding a write
    that failed would hand the dispatcher a ledger disagreeing with GitHub,
    which is the one thing this whole module exists to prevent - and the
    disagreement would last exactly until the next read, i.e. long enough to
    dispatch a container against it.
    """
    by_ref = {transition.ref: transition for transition in transitions}
    if not by_ref:
        return ledger
    entries: dict[str, LedgerEntry] = {}
    for task_id, entry in ledger.entries.items():
        transition = by_ref.get(entry.ref)
        if transition is None:
            entries[task_id] = entry
            continue
        entries[task_id] = replace(
            entry,
            state_label=transition.to_label,
            attempt=entry.attempt if transition.attempt is None else transition.attempt,
            # The signature record mirrors the store write exactly: a
            # transition that wrote the counter wrote (or cleared) the
            # judgment in the same act, and one that left the counter alone
            # left the judgment alone too. Folding anything else would hand
            # the rest of the cycle a ledger disagreeing with what was just
            # persisted.
            blocker=entry.blocker if transition.attempt is None else transition.blocker,
            streak=entry.streak if transition.attempt is None else transition.streak,
            renewals=entry.renewals if transition.attempt is None else transition.renewals,
            labels=frozenset(entry.labels - {transition.from_label} | {transition.to_label}),
        )
    return replace(ledger, entries=entries)


# --------------------------------------------------------------------------
# The attempt counter
# --------------------------------------------------------------------------


def rewrite_marker(body: str, task_id: str, attempt: int) -> str:
    """Set the counter in the identity marker, preserving every other byte.

    §5's write rule, and the reason it is a body `PATCH` rather than a label: it
    either applied or it did not. Every line other than the marker is returned
    untouched, because a human editing prose while the orchestrator bumps a
    counter must not lose their edit.

    The marker is located, not parsed: the first line that looks like one and is
    not inside a fence. A body with no marker at all - or one whose only marker
    is a fenced example, which `ledger._parse_marker` correctly ignores - gets a
    fresh marker prepended, which is where the loader's own adoption puts it and
    which the parser reads first either way.

    **The counter and nothing else.** The marker used to carry the failure
    signature too, and #159 moved that into apiary's own store
    (`docs/adr/0002-apiary-owns-a-thin-task-store.md`) - so this rewrite drops
    `blocker=` and `streak=` from any body that still has them, which is how a
    repository upgraded mid-flight sheds them. The counter itself stays,
    because the worker reads it: it is a container with no view of the host and
    it derives its branch name and its result filename from that number
    (`render_marker`).
    """
    marker = render_marker(task_id, attempt)
    lines = (body or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    fenced = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fenced = not fenced
            continue
        if fenced:
            continue
        if stripped.startswith("<!--") and "apiary:task" in stripped:
            lines[index] = marker
            return "\n".join(lines)
    return f"{marker}\n\n{body}" if body else marker


def bump_attempt(client: Any, number: int, task_id: str, attempt: int) -> None:
    """Persist the counter, re-reading the body immediately before the patch.

    Two API calls, and the re-read is the point: the body in the ledger was
    fetched at the top of the cycle, and a human editing in between would have
    their edit overwritten by a patch built from the stale copy. §5 requires the
    fresh read, and it is cheap because it happens only for an issue that just
    finished an attempt.

    The failure signature no longer travels with it - it is written to
    apiary's own store, immediately before this call, by the same `apply_plan`
    loop (#159). §5's crash-ordering argument survives the split intact and
    reads the same way: judgment, then counter, then the label that re-readies
    the task, so a crash anywhere in the sequence costs an attempt with its
    signature recorded rather than granting a retry whose streak forgot what it
    was retrying. The two writes are not atomic together and never were - the
    counter and the label were already two calls - and the order is what buys
    the guarantee, not a transaction.
    """
    issue = client.get_issue(number)
    body = issue.get("body") or ""
    client.update_issue(number, body=rewrite_marker(body, task_id, attempt))


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Failure:
    """One thing a cycle could not do, and to which issue.

    Collected rather than raised, exactly as `Reaper.sweep` collects: one issue
    GitHub will not relabel must not stop the other nineteen from being
    reconciled, and a run whose cycle died halfway through leaves a ledger in a
    state nobody planned.
    """

    ref: TaskRef
    reason: str

    def __str__(self) -> str:
        return f"{self.ref}: {self.reason}"


@dataclass(frozen=True)
class ReconcileReport:
    """What one cycle's writes actually achieved."""

    plan: ReconcilePlan
    applied: tuple[Transition, ...] = ()
    disposed: tuple[TaskRef, ...] = ()
    repaired: tuple[TaskRef, ...] = ()
    failures: tuple[Failure, ...] = ()
    #: Comments §1.4 wanted posted and this client had no method for. Not a
    #: failure - the text was printed - but the gap is worth reporting rather
    #: than swallowing, because the reason an issue was failed is otherwise
    #: nowhere a human will look.
    uncommented: tuple[TaskRef, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        parts = [
            f"applied {len(self.applied)}/{len(self.plan.transitions)} transition(s)",
            f"disposed {len(self.disposed)}",
            f"repaired {len(self.repaired)}",
        ]
        if self.failures:
            detail = "; ".join(str(failure) for failure in self.failures)
            parts.append(f"{len(self.failures)} failed: {detail}")
        if self.uncommented:
            names = ", ".join(str(ref) for ref in self.uncommented)
            parts.append(f"could not comment on {names} - the client has no {COMMENT_METHOD}")
        return ", ".join(parts)


def post_comment(client: Any, number: int, text: str) -> bool:
    """Post one comment, or print it and say it did not land.

    `docs/issue-contract.md` §1.4 requires the `ContractError` text to reach the
    issue, and `GitHubClient` has no method for it (module docstring). Printing
    is not a substitute for the comment; it is what keeps the reason for a
    `swarm:failed` label from being lost entirely in the meantime.
    """
    poster = getattr(client, COMMENT_METHOD, None)
    if poster is None:
        print(f"! no {COMMENT_METHOD}; comment for #{number} not posted:\n{text}", file=sys.stderr)
        return False
    try:
        poster(number, text)
    except GitHubError as exc:
        print(f"! comment on #{number} failed: {exc}", file=sys.stderr)
        return False
    return True


def apply_plan(
    client: Any,
    plan: ReconcilePlan,
    *,
    fleet: Fleet | None = None,
    handles: Mapping[TaskRef, Handle] | None = None,
    store: TaskStore | None = None,
    dry_run: bool = False,
) -> ReconcileReport:
    """Write the plan. Never raises for one issue; see `Failure`.

    Per transition the order is judgment, then counter, then add, then remove.
    Judgment and counter first is §5 - both are persisted before the issue can
    be re-dispatched, so a crash costs an attempt rather than granting a free
    one, and the judgment leads because a counter that moved without one would
    be an attempt charged against a blocker nobody recorded. Add-before-remove
    is `readiness._relabel`'s rule and load-bearing for the same reason: a
    crash between two label calls leaves either two state labels or none, and
    two is repairable by §3's precedence while none puts the issue outside the
    ledger where nothing looks at it again.

    `store` is where the judgment goes (#159). `None` writes none, which is for
    a caller that is only exercising the label half; `Reconciler` requires a
    store and always passes it, because a run that consumed attempts without
    recording their signatures would let every task renew its budget forever.
    """
    handles = handles or {}
    applied: list[Transition] = []
    disposed: list[TaskRef] = []
    repaired: list[TaskRef] = []
    failures: list[Failure] = []
    uncommented: list[TaskRef] = []

    if dry_run:
        return ReconcileReport(plan=plan)

    for repair in plan.repairs:
        try:
            for label in repair.removed:
                client.remove_label(repair.number, label)
        except GitHubError as exc:
            failures.append(Failure(repair.ref, f"repairing labels: {exc}"))
            continue
        repaired.append(repair.ref)
        if not post_comment(client, repair.number, f"apiary: {repair}"):
            uncommented.append(repair.ref)

    for transition in plan.transitions:
        # The one place this loop needs the tracker's own spelling: every call
        # below addresses the GitHub API, which takes issue numbers.
        number = issue_number(transition.ref)
        try:
            if transition.attempt is not None and transition.task_id:
                # apiary's own judgment first, the tracker's counter second.
                # A store write that fails raises out of this `try` like a
                # GitHub error does and lands in `failures` for that one issue,
                # which is the right blast radius: one task whose judgment
                # could not be recorded must not cost the other nineteen their
                # transition, and the un-bumped counter leaves that task
                # exactly where it was for the next cycle to try again.
                record_judgement(
                    store,
                    transition.ref,
                    transition.attempt,
                    blocker=transition.blocker,
                    streak=transition.streak,
                    renewals=transition.renewals,
                )
                bump_attempt(client, number, transition.task_id, transition.attempt)
            client.add_labels(number, [transition.to_label])
            if transition.from_label and transition.from_label != transition.to_label:
                client.remove_label(number, transition.from_label)
        except (GitHubError, StoreError) as exc:
            # A human deleting or relabelling this issue between the read and
            # the write lands here, and so does a store that will not take the
            # judgment. Either is a fact about one issue, not a reason to
            # abandon the cycle - the next read sees whatever they did.
            failures.append(Failure(transition.ref, f"{transition.to_label}: {exc}"))
            continue
        applied.append(transition)
        if transition.comment and not post_comment(client, number, transition.comment):
            uncommented.append(transition.ref)

    for disposal in plan.disposals:
        handle = handles.get(disposal.ref)
        if handle is None or fleet is None:
            continue
        try:
            fleet.dispose(handle)
        except ContainerError as exc:
            failures.append(Failure(disposal.ref, f"disposing {handle}: {exc}"))
            continue
        disposed.append(disposal.ref)

    return ReconcileReport(
        plan=plan,
        applied=tuple(applied),
        disposed=tuple(disposed),
        repaired=tuple(repaired),
        failures=tuple(failures),
        uncommented=tuple(dict.fromkeys(uncommented)),
    )


# --------------------------------------------------------------------------
# The cycle
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CycleReport:
    """One turn of the loop, in enough detail to print and to assert on."""

    index: int
    ledger: Ledger
    result: ReconcileReport
    readiness: ReadinessPlan | None = None
    dispatched: DispatchReport | None = None
    #: The merge gate, in the order it runs: mergeability decides what may be
    #: merged *against the base as it is now*, and hands the surviving plan to
    #: checks. Both are `None` when the cycle never got that far.
    mergeability: Any | None = None
    checks: Any | None = None
    #: Claims released this cycle because nothing was running behind them.
    recovered: Any | None = None
    #: Step 5. `verdict` is `judge.Verdict` on the cycles that earned one and
    #: `None` on the cycles the arithmetic settled without asking; `replanned`
    #: is `replan.ReplanReport` on the cycles that acted on a stall.
    verdict: Any | None = None
    replanned: Any | None = None
    #: `goal.GoalReport`, set only on the cycle where the ledger ran dry.
    goal: Any | None = None
    #: The one fault a cycle records instead of raising. Two reach it: a
    #: `DependencyCycleError` from readiness - the one readiness failure that
    #: aborts a pass rather than joining its errors - and a
    #: `checks.UnresolvedJoin` from the merge gate (#174).
    #:
    #: Recorded rather than raised so the loop reports it every cycle until a
    #: human fixes it, instead of the run dying and taking its containers with
    #: it. For the merge gate there is a second reason: it runs after this
    #: cycle's labels are already written, so an escape would lose the report
    #: that says they were.
    #:
    #: **Set means this cycle dispatched nothing.** Readiness and dispatch are
    #: skipped, because both faults say the same thing - the machinery that
    #: decides what may land is not answering, and adding work to a queue that
    #: cannot drain is how a stuck run goes on looking busy.
    cycle_error: str = ""
    live: int = 0

    @property
    def plan(self) -> ReconcilePlan:
        return self.result.plan

    @property
    def changed(self) -> bool:
        """Did this cycle move anything at all? The stall signal."""
        return bool(
            self.result.applied
            or self.result.disposed
            or self.result.repaired
            or (self.readiness is not None and self.readiness.transitions)
            or (self.dispatched is not None and self.dispatched.dispatched)
        )

    @property
    def exhausted(self) -> bool:
        """Nothing left that is not `swarm:done` or `swarm:failed`.

        The plan is finished. Whether the *objective* is finished is `goal.py`'s
        question, and `finished` below is the answer to that one.
        """
        return self.live == 0

    @property
    def finished(self) -> bool:
        """May the loop stop after this cycle?

        An exhausted ledger that the goal gate then extended is not finished:
        there are new issues on the tracker and the next cycle dispatches them.
        A cycle that never reached the gate - because the ledger still had live
        work - is not finished either.
        """
        if not self.exhausted:
            return False
        return self.goal is None or self.goal.done

    @property
    def needs_judgement(self) -> bool:
        """Is this cycle worth paying the judge's model swap for?

        The dispatcher's own answer, narrowed: a cycle that reconciled something
        learned plenty without a model, so a swap would buy news the arithmetic
        already had (`dispatcher.DispatchReport.needs_judgement`).

        An exhausted ledger is excluded because it has its own question and its
        own model call: the goal gate. Judging it as well would spend two swaps
        to be told twice that there is nothing left to move.
        """
        if self.changed or self.exhausted:
            return False
        return self.dispatched is None or self.dispatched.needs_judgement

    def summary(self) -> str:
        parts = [f"cycle {self.index}: {self.result.summary()}"]
        if self.readiness is not None:
            parts.append(self.readiness.summary())
        if self.dispatched is not None:
            parts.append(self.dispatched.summary())
        if self.checks is not None:
            parts.append(self.checks.summary())
        if self.verdict is not None:
            parts.append(f"judged: {self.verdict.summary()}")
        if self.replanned is not None:
            parts.append(self.replanned.summary())
        if self.goal is not None:
            parts.append(self.goal.summary())
        if self.cycle_error:
            parts.append(f"error: {self.cycle_error}")
        parts.append(f"{self.live} live issue(s)")
        return "; ".join(parts)


def _lifecycle_log() -> Any:
    """`lifecycle.LifecycleLog`, imported at call time.

    `lifecycle` imports this module - it projects a `CycleReport` and reads the
    two terminal labels - so the dependency points that way and a top-level
    import here would be a cycle. Same shape, and the same reason, as `checks`
    and `mergeability` inside `cycle`.
    """
    from .lifecycle import LifecycleLog

    return LifecycleLog()


@dataclass
class Reconciler:
    """The loop body, and the thing that paces it.

    Holds a run, a client and a fleet, and no state that survives a cycle. That
    is the property `docs/architecture-v2.md` promises - "the orchestrator is
    restartable at any point and holds no irreplaceable state" - and it is
    testable: two `Reconciler`s over one repository converge on the same
    answers, because both of them ask GitHub.
    """

    run: Run
    client: GitHubClient
    #: Where apiary's own judgments live (#159): the failure signature, its
    #: streak and the renewal count for every task this run touches.
    #: **Required, and deliberately without a default.** Every other seam here
    #: is optional because its absence disables a rule that is then plainly
    #: not running; a missing store is the opposite - the retry arithmetic
    #: still runs, still consumes attempts, and simply forgets what it decided,
    #: which reads as a fresh budget for every task on the next cycle and lets
    #: a failing task retry forever. That failure is silent, so it is made
    #: impossible instead: there is no `Reconciler` without somewhere to write
    #: its judgments, and mypy says so at every construction site.
    store: TaskStore
    base_commit: str = ""
    fleet: Fleet | None = None
    #: Where the workers' result records land - `RunArtifacts.results_dir`, not
    #: the run directory: `worker.result.load_results` globs the directory it is
    #: given. `None` means this reconciler never observes a worker's exit code,
    #: which disables §4's retry rows entirely, so a real run must pass it.
    artifacts: str | Path | None = None
    capacity: Capacity | None = None
    interval_s: float = DEFAULT_INTERVAL_S
    max_attempts: int = SETTINGS.max_attempts_per_task
    #: The hard ceiling across every failure mode. `max_attempts` bounds one
    #: blocker and renews when the failure changes; this bounds the task, so a
    #: run whose failures keep changing still ends (`SWARM_MAX_TOTAL_ATTEMPTS`).
    max_total_attempts: int = SETTINGS.max_total_attempts_per_task
    dry_run: bool = False
    #: Injection points for the tests, and for nothing else. A loop that slept
    #: for real would make the pacing untestable, which is the half of the
    #: budget that the request count does not cover.
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic

    #: Releases claims whose container is gone. `None` disables the mid-cycle
    #: sweep; the startup sweep is the caller's to run either way.
    recovery: Any | None = None

    #: Whether this cycle merges. Off leaves every `swarm:review` PR alone,
    #: which is what a run wants when a human is doing the merging.
    merge_gate: bool = True

    #: `checks.MergePolicy` and `mergeability.UpdatePolicy`. Typed loosely for
    #: the same reason they are imported inside `cycle`: both modules import
    #: this one. `None` takes each module's own default, which merges on green
    #: - the environment's answer belongs to whoever started the run, and
    #: `cli._loop` reads it there so that one line can report what it bypasses.
    merge_policy: "Any | None" = None
    update_policy: "Any | None" = None

    #: Carried across cycles so an unlucky PR cannot be updated forever. It is
    #: in-process by design: a restart grants a fresh budget, which is the
    #: right failure for a counter whose whole job is bounding one run.
    update_budget: "Any | None" = None

    #: When a task that only ever fails mechanically stops being free. Read at
    #: the call site like the other two policies, so the line that reports it at
    #: startup is the same line that chose it.
    infrastructure_policy: InfrastructurePolicy = field(default_factory=InfrastructurePolicy)

    #: Which image carries which stack's toolchain (#99). Read at the call site
    #: like the policies, so the line that reports it at startup is the line
    #: that chose it.
    images: StackImages = field(default_factory=StackImages)

    #: Step 5. The objective is what the goal gate assesses against, so a
    #: reconciler without one cannot close its own loop and says so rather than
    #: assessing against an empty string; `verify` is the run's repo-wide
    #: command, carried so a replanned or followed-up issue inherits the gate
    #: the original carried rather than `SETTINGS.verify_command`.
    objective: str = ""
    verify: str = ""
    #: Off makes the loop stop at plan exhaustion, which is what `--no-goal-check`
    #: asks for: a run that does exactly the plan a human read and approved.
    goal_gate: bool = True
    #: The three model seams step 5 owns, `None` meaning "the real one". They
    #: exist for `replan.replan`'s stated reason and one more: a reconcile test
    #: that reached Ollama would be a test whose result depends on which model
    #: is pulled, and - because this host has one running - would silently spend
    #: a 31B inference per quiet cycle in a suite that is meant to be hermetic.
    #: `tests/test_reconcile.py` passes an oracle that raises, so an unintended
    #: model call is a failure rather than a slow pass.
    oracle: "Any | None" = None
    assessor: "Any | None" = None
    proposer: "Any | None" = None
    #: Called with every finished `CycleReport`, before the loop paces itself.
    #: The seam `cli` prints through: a run whose cycles are only reported at
    #: the end is a run nobody can watch, and it is also how the merge gate's
    #: verdicts reach the operator while there is still time to act on them.
    on_cycle: Callable[[CycleReport], None] | None = None
    #: Where the per-task lifecycle is announced (#141) - `RunArtifacts.event`
    #: in a real run, so the events land in `events.jsonl` and are redacted like
    #: everything else. `None` announces nothing, which is what a reconciler
    #: with no run directory behind it should do. Separate from `on_cycle`
    #: because the two answer different questions: `on_cycle` is one line per
    #: cycle for whoever is watching, this is one event per task transition for
    #: whoever reads the run back afterwards.
    events: Callable[..., Any] | None = None

    _cycles: int = field(default=0, repr=False)
    #: The progress ledger, run-scoped. See the module docstring.
    _previous: "Any | None" = field(default=None, repr=False)
    _stalls: int = field(default=0, repr=False)
    _replans: int = field(default=0, repr=False)
    _goal_rounds: int = field(default=0, repr=False)
    #: Consecutive infrastructure verdicts per task. In-process for the
    #: same reason `update_budget` and `_stalls` are: it is a question about a
    #: sequence, GitHub only ever shows the current state, and a restart
    #: granting a clean slate is the safe direction for a counter whose job is
    #: bounding one run.
    _infrastructure: dict[TaskRef, int] = field(default_factory=dict, repr=False)
    #: Which task events have already been announced (#141). Run-scoped for the
    #: same reason, and holding nothing a decision reads. Typed loosely and
    #: built through a local import, for `merge_policy`'s reason: `lifecycle`
    #: imports this module.
    _lifecycle: Any = field(default_factory=_lifecycle_log, repr=False)

    # --- one cycle -------------------------------------------------------

    def cycle(self) -> CycleReport:
        """Read, reconcile, compute readiness, dispatch. One pass, in that order.

        Every step after the first takes its facts from `Snapshot`, so the whole
        cycle costs one issue listing plus whatever it decided to change.
        """
        index = self._cycles
        self._cycles += 1

        snapshot = Snapshot(self.client)
        # `adopt` writes a marker onto every hand-written issue (§2), which a
        # dry run promised not to do.
        ledger = load_ledger(
            snapshot,  # type: ignore[arg-type]
            adopt=not self.dry_run,
            store=self.store,
        )

        handles = self._handles()
        # Read once and shared with step 5: the judge's observation carries each
        # task's latest failure text, and that text is what a replan is written
        # from (`replan.brief`). A second read here would be a second directory
        # listing for facts this cycle already has.
        results = self._results()
        plan = plan_reconcile(
            ledger,
            states=snapshot.states(),
            open_branches=snapshot.open_branches(),
            results=results,
            running=tuple(handles),
            labels=snapshot.labels(),
            max_attempts=self.max_attempts,
            max_total_attempts=self.max_total_attempts,
            infrastructure=self._infrastructure,
            infrastructure_policy=self.infrastructure_policy,
        )
        result = apply_plan(
            snapshot,
            plan,
            fleet=self.fleet,
            handles=handles,
            store=self.store,
            dry_run=self.dry_run,
        )
        ledger = fold(ledger, result.applied)
        # Folded from what actually **landed**, not from what was planned: a
        # label write GitHub refused left the issue where it was, so counting
        # it would escalate a task on the strength of a move that never
        # happened. Same rule `fold` follows one line up, for the same reason.
        self._infrastructure = infrastructure_streaks(self._infrastructure, result.applied)

        # A claim with no container behind it is undispatchable forever, and the
        # window that produces one is the dispatcher's own claim-then-spawn gap.
        # Swept here rather than only at startup because that gap opens
        # mid-run: the facts are the ones this cycle has already read, so the
        # sweep costs nothing extra.
        recovered = None
        if self.recovery is not None:
            recovered = self.recovery.sweep(
                ledger,
                # `.values()`, because `_handles()` is keyed by task ref and
                # iterating the mapping yields the keys - `holders` would then
                # ask a `TaskRef` for its `.issue`.
                containers=handles.values(),
                states=snapshot.states(),
                open_branches=snapshot.open_branches(),
            )
            ledger = fold(ledger, recovered.result.applied)

        # Local, because `checks` imports this module: it is the policy over the
        # state this one folds, so the dependency points this way and a
        # top-level import would be a cycle.
        from .checks import read_pulls

        # Read outside the merge gate, and free: `open_branches()` above already
        # forced `Snapshot`'s one pull-request listing, so this is a fold over
        # payloads the cycle is holding. Outside the gate because a run merging
        # by hand still wants `pr.opened` in its event log - the gate being off
        # is not a reason for the run directory to stop recording that a task
        # reached review.
        pulls = read_pulls(snapshot)

        # The merge gate. Mergeability runs first and *subtracts* from the plan
        # checks built: a PR that is green against a base that has since moved
        # is not mergeable, and merging it would land work that never ran
        # against what it is landing on. `plan.admitted` is what survives.
        mergeability = None
        checks = None
        cycle_error = ""
        check_runs: dict[TaskRef, Any] = {}
        if self.merge_gate:
            from .checks import UnresolvedJoin, apply_checks, plan_checks, read_checks
            from .mergeability import run_mergeability

            if pulls is not None:
                for entry in ledger.entries.values():
                    pull = pulls.get(entry.branch) if entry.state_label == REVIEW else None
                    if pull is not None:
                        check_runs[entry.ref] = read_checks(snapshot, pull.ref)
            try:
                checks_plan = plan_checks(
                    ledger,
                    pulls=pulls,
                    checks=check_runs,
                    # Without this the whole `MergePolicy` is whatever the
                    # dataclass defaults to, and `APIARY_MERGE_ADMIN_OVERRIDE=0`
                    # - the one setting that decides whether a human presses
                    # merge - silently does nothing.
                    policy=self.merge_policy,
                    max_attempts=self.max_attempts,
                )
                mergeability = run_mergeability(
                    snapshot,
                    ledger,
                    checks_plan,
                    pulls=pulls,
                    policy=self.update_policy,
                    budget=self.update_budget,
                    max_attempts=self.max_attempts,
                    # Both gates consume attempts of their own - a PR that will
                    # not rebase, a check run that failed - so both need
                    # somewhere to record the judgment that goes with the
                    # counter they bump.
                    store=self.store,
                    dry_run=self.dry_run,
                )
                checks = apply_checks(
                    snapshot, mergeability.plan.admitted, store=self.store, dry_run=self.dry_run
                )
            except UnresolvedJoin as exc:
                # Recorded rather than allowed to escape, for
                # `DependencyCycleError`'s reason and one more of its own.
                #
                # This gate runs *after* `apply_plan` wrote this cycle's labels
                # and after the recovery sweep. An exception leaving `cycle`
                # here is thrown before `CycleReport` exists, so `on_cycle`
                # never fires and the run directory never learns that those
                # writes happened - a loud failure that erases its own
                # evidence, which is a strange thing for #174 of all tickets to
                # ship. It also takes the fleet's containers with it.
                #
                # Nothing of this gate's own is lost by catching it: the join
                # fails while the plan is still being *computed*, so no merge
                # was issued and no label of this gate's was written.
                # `mergeability` and `checks` stay `None`, which is the same
                # shape a cycle with the gate switched off reports, and the
                # ledger is left unfolded because there is nothing to fold.
                mergeability = None
                checks = None
                cycle_error = str(exc)
                print(f"! the merge gate could not resolve a join: {exc}", file=sys.stderr)
            else:
                ledger = fold(ledger, checks.applied)

        readiness: ReadinessPlan | None = None
        dispatched: DispatchReport | None = None
        if not cycle_error:
            try:
                readiness = apply_readiness(
                    snapshot,  # type: ignore[arg-type]
                    ledger=ledger,
                    dry_run=self.dry_run,
                )
            except DependencyCycleError as exc:
                # Nothing was written - readiness detects the ring before its
                # first call - and dispatching over an unresolved graph would
                # run work whose prerequisites can never land.
                cycle_error = str(exc)
            else:
                if self.fleet is not None:
                    dispatched = dispatch(
                        snapshot,
                        self.fleet,
                        ledger,
                        self.base_commit,
                        capacity=self.capacity,
                        ready=readiness.ready,
                        dry_run=self.dry_run,
                        images=self.images,
                    )

        report = CycleReport(
            index=index,
            ledger=ledger,
            result=result,
            readiness=readiness,
            dispatched=dispatched,
            mergeability=mergeability,
            checks=checks,
            recovered=recovered,
            cycle_error=cycle_error,
            live=len(live_entries(ledger)),
        )
        judged = self._judge(snapshot, report, results=results)
        # Last, and on the grown report: the announcement (#141) is a projection
        # of a cycle that has already decided, already written and already been
        # judged, which is what makes "this changes no behaviour" a structural
        # claim rather than a promise. The three facts it needs are passed
        # rather than hung on `CycleReport`, so there is nothing on the report
        # for a future rule to read - see `lifecycle.lifecycle_events`.
        self._lifecycle.announce(
            judged,
            emit=self.events,
            results=results,
            pulls=pulls or {},
            checks=check_runs,
        )
        return judged

    # --- step 5 ----------------------------------------------------------

    def _judge(
        self, client: Any, report: CycleReport, *, results: Mapping[TaskRef, ResultRecord]
    ) -> CycleReport:
        """Judge this cycle, and act on the judgement. Returns the report, grown.

        Three outcomes, and the order is the priority:

        1. **The ledger ran dry.** The goal gate asks whether the objective was
           met and appends follow-up work if it was not (`goal.py`). This runs
           *before* the stall rules, because an exhausted ledger trivially
           satisfies "nothing moved" and would otherwise be replanned - which
           would rewrite a backlog whose every task merged.
        2. **The cycle earned a judgement.** Nothing changed, nothing is in
           flight; `judge.judge` answers, from arithmetic where it can.
        3. **The judgement says the run is stuck.** `replan.replan` decides
           whether that has been true for long enough to be worth rewriting the
           plan, and refuses on its own five grounds.

        Nothing here runs on a dry run: the gate and the replan both write
        issues, and a command that promised to change nothing must not.
        """
        if self.dry_run:
            return report

        from ..nodes.judge import Observation, judge

        observation = Observation.of(report.ledger, results=results)

        if report.exhausted:
            # The observation is still recorded: a follow-up round leaves the
            # ledger non-empty again, and the next stall check needs a previous
            # reading that includes the issues this gate just wrote.
            self._previous = observation
            return replace(report, goal=self._close(client, report))

        verdict = None
        replanned = None
        if report.needs_judgement:
            verdict = judge(
                observation,
                self._previous,
                objective=self.objective,
                stalls=self._stalls,
                round_index=report.index,
                oracle=self.oracle,
            )
            self._stalls = verdict.stalls
            if verdict.stalled:
                replanned = self._replan(client, report, verdict)
        self._previous = observation
        return replace(report, verdict=verdict, replanned=replanned)

    def _close(self, client: Any, report: CycleReport) -> Any:
        """The goal gate. Local import for `checks`' reason - `goal` imports the
        planner, which is a heavier graph than a cycle that never ends should
        pay for on every import of this module."""
        from .goal import close_the_loop

        # `None`, not a negative report: an exhausted ledger that was never
        # assessed must read as "the plan is done" rather than as "the objective
        # was missed", because the caller's exit code is the difference between
        # those two and a run nobody asked to assess has not failed at anything.
        if not self.goal_gate:
            return None
        if not self.objective.strip():
            # Assessing against an empty objective asks a model whether nothing
            # was delivered, which it cannot answer and which would extend the
            # plan from the answer it invented.
            print("! this run carries no objective; the goal gate is skipped", file=sys.stderr)
            return None
        goal = close_the_loop(
            client,
            report.ledger,
            self.objective,
            rounds=self._goal_rounds,
            verify=self.verify or None,
            oracle=self.assessor,
            proposer=self.proposer,
        )
        self._goal_rounds = goal.rounds
        wrote = (
            getattr(goal, "extended", False)
            or getattr(goal, "revived", ())
            # Superseded closures land on a *met* report, which ends the loop -
            # but a caller driving cycles past `finished` (tests, `until`)
            # must still read back what was closed rather than a 304 of it.
            or getattr(goal, "superseded", ())
        )
        if wrote and hasattr(client, "invalidate_cache"):
            # The gate just wrote through this client - follow-up issues, or a
            # revival's label move - and the next cycle's first act is to read
            # the result back. GitHub's conditional cache lags its writes
            # (`invalidate_cache`'s own docstring records the planner hitting
            # this), so without the flush the next read is answered 304 from
            # the pre-write body, the ledger still looks exhausted, and the
            # gate runs AGAIN - for an extension that is a second seven-minute
            # assessment and near-duplicate follow-ups (observed live: cycle 22
            # planned #15-#17, cycle 23 re-planned #18-#20 against the same
            # gap); for a revival it is the same issue read as failed again and
            # a duplicate revival comment posted onto it.
            client.invalidate_cache()
        return goal

    def _replan(self, client: Any, report: CycleReport, verdict: Any) -> Any:
        from .replan import replan

        result = replan(
            client,
            report.ledger,
            self.objective,
            verdict,
            replans=self._replans,
            verify=self.verify or None,
            proposer=self.proposer,
        )
        if result.replanned:
            # `replan` zeroes the stall count on a successful rewrite, because
            # the run that follows is a different plan and its progress is not
            # this one's.
            self._replans = result.replans
            self._stalls = 0
        return result

    # --- the loop --------------------------------------------------------

    def loop(
        self,
        *,
        cycles: int | None = None,
        until: Callable[[CycleReport], bool] | None = None,
    ) -> tuple[CycleReport, ...]:
        """Run cycles until the run is finished, `until` says so, or `cycles` run out.

        **Finished means the objective, not the plan.** `CycleReport.finished`
        is false on a cycle whose goal gate appended follow-up work, so the loop
        carries straight on and dispatches it - which is the whole point of the
        gate, and the difference between a swarm that stops when its first plan
        runs out and one that stops when the objective is met or it has run out
        of ways to get there.

        The interval is a floor between cycle *starts*: a cycle that took longer
        than `interval_s` starts the next one immediately rather than sleeping
        on top of its own duration. Nothing sleeps after the last cycle, because
        the only thing that would wait for is the caller's return.
        """
        reports: list[CycleReport] = []
        remaining = cycles
        while remaining is None or remaining > 0:
            started = self.clock()
            report = self.cycle()
            reports.append(report)
            if self.on_cycle is not None:
                self.on_cycle(report)
            if remaining is not None:
                remaining -= 1
            done = report.finished if until is None else until(report)
            if done or remaining == 0:
                break
            self._pace(started)
        return tuple(reports)

    def _pace(self, started: float) -> None:
        elapsed = self.clock() - started
        delay = self.interval_s - elapsed
        if delay > 0:
            self.sleep(delay)

    # --- what the cycle reads --------------------------------------------

    def _handles(self) -> dict[TaskRef, Handle]:
        """This run's containers, by task. One `docker ps`, whatever the count.

        A container with no issue label is not this run's worker - the label is
        written at `docker create` - so it is left for the reaper (#20), which
        is the module allowed to remove things it did not spawn.

        The label is an issue number, because that is what a container name and
        a docker label may contain; it is minted into a ref here so the plan
        above keys on the same identity everything else does.
        """
        if self.fleet is None:
            return {}
        found: dict[TaskRef, Handle] = {}
        for handle in self.fleet.find():
            if handle.issue is not None:
                found.setdefault(task_ref(int(handle.issue)), handle)
        return found

    def _results(self) -> dict[TaskRef, ResultRecord]:
        """The latest artifact record per issue, or nothing if there is no directory.

        The worker writes its record last (`worker/result.py`), so a record is
        the only evidence this process has that a container finished and what it
        decided - and reading it costs a directory listing rather than an API
        call or a blocking `docker wait`.
        """
        if self.artifacts is None:
            return {}
        # `summarise_dir` keys on the issue number the worker wrote into the
        # record's filename; minted here for `_handles`' reason.
        return {
            task_ref(number): record
            for number, record in summarise_dir(self.artifacts, run_id=self.run.id).latest.items()
        }


if __name__ == "__main__":  # pragma: no cover - manual dry run, see module docstring
    import os

    from ..store import SqliteTaskStore

    repo = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GITHUB_REPOSITORY", "")
    # Read-only on every path: no label, no comment, no adoption, no container.
    reconciler = Reconciler(
        run=Run.start(repo, "dry run", run_id="apiary-dry-run-000000-aaaaaa"),
        client=GitHubClient.from_env(repo),
        # Opened read-only in effect: a dry run plans and writes nothing, but
        # it still has to *read* the judgments, or every task would print as
        # though it had never failed.
        store=SqliteTaskStore.open(repo),
        dry_run=True,
    )
    dry = reconciler.cycle()
    for planned in dry.plan.transitions:
        print(f"would {planned}")
    for gone in dry.plan.disposals:
        print(f"would {gone}")
    for repair in dry.plan.repairs:
        print(f"would repair {repair}")
    print(dry.summary())
