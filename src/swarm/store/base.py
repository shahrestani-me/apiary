"""The record, the seam, and the three ways reading one can fail.

Everything in this module is backend-agnostic on purpose. `sqlite.py` is one
implementation of `TaskStore`; the point of writing the protocol down
separately is that the organization database ADR 0002 defers can arrive as a
second implementation of *this* file rather than as a rewrite of its callers.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from typing import Mapping, Protocol, runtime_checkable

from ..taskref import TaskRef

#: Bumped when a stored field changes meaning, never when one is added. A
#: reader that finds a *higher* number is looking at rows it may misread, and
#: misreading a retry budget is how a task retries forever - so that case
#: raises rather than degrading, which is the opposite of the tolerance
#: `worker/result.py` applies to its own records. The asymmetry is deliberate:
#: a result record is evidence and a partial read costs a line of a summary; a
#: judgment is a bound on retries and a partial read costs the bound.
SCHEMA_VERSION = 1


class StoreError(RuntimeError):
    """The task store could not be opened, read, or written."""


class StoreMissing(StoreError):
    """A store was required to exist and does not.

    Separate from `StoreCorrupt` because the fixes differ and a human reading
    one line should not have to work out which they are looking at: a missing
    store is usually a moved directory or a wrong `APIARY_STORE_DIR`, and a
    corrupt one is a file that has to be looked at.
    """


class StoreCorrupt(StoreError):
    """A store exists and cannot be trusted.

    Raised rather than recovered from, and this is the single most important
    decision in the package. Every plausible recovery - start empty, drop the
    unreadable rows, rebuild from the tracker - answers "what is this task's
    retry budget?" with "a fresh one", for every task at once, at the moment
    something is already wrong. That failure is silent: the run looks healthy,
    the tasks look young, and a task that has already burned its budget retries
    until somebody notices by hand. A run that stops with a path and a fix is
    strictly better than a run that quietly unbounds itself.
    """


@dataclass(frozen=True)
class TaskJudgement:
    """What apiary decided about one task's own execution. Five fields, closed.

    Every field here is something apiary computed about its own run and no
    external system is authoritative for. Nothing the tracker owns appears -
    no goal, no files, no verify command, no title, no dependencies - and the
    record is deliberately a closed dataclass rather than a mapping so that
    adding one is a visible change rather than a key somebody starts writing.

    **`attempt` is a stamp, not a counter.** It records which attempt the
    signature below was taken from. The counter itself lives in the issue
    marker, because the worker reads it there and cannot reach this store (see
    the package docstring). Keeping the stamp is what lets `matches` recognise
    a counter that moved underneath the judgment - a human resetting it after
    fixing the environment is the documented workflow ADR 0002 quotes - and
    treat the judgment as stale rather than applying a signature to an attempt
    it was never about.
    """

    ref: TaskRef
    #: Which attempt this judgment was recorded at. See above: a stamp.
    attempt: int = 0
    #: `reconcile.signature` of the failure that attempt died on, or `""` for
    #: "nothing recorded" - a task that has never failed, an attempt consumed
    #: through a channel with no verify output to sign (a stale claim, a failed
    #: check run), or a row seeded from a marker written before signatures
    #: existed. Empty is read downstream as "no previous blocker", which falls
    #: back to the pre-signature arithmetic exactly.
    blocker: str = ""
    #: How many consecutive attempts have failed with `blocker`. `None` means
    #: the row does not say, which downstream falls back to the attempt counter
    #: - what the streak *was* before failures had signatures.
    streak: int | None = None
    #: How many times this task's per-blocker budget has been renewed, i.e. how
    #: many times a failure differed from the one before it. Recorded, reported
    #: and never branched on: the give-up tests run on `streak` against
    #: `max_attempts_per_task` and on the marker's counter against
    #: `max_total_attempts_per_task`, exactly as #154-#156 left them, and a
    #: third input would be a behaviour change wearing an accounting field's
    #: clothes. It is here because "the budget was renewed four times" is the
    #: one thing a human reading a capped task wants to know and the only place
    #: it was ever written down was prose in a comment.
    renewals: int = 0
    #: When this row was last written. Never read by a decision; it is what
    #: makes a store somebody is debugging legible at all.
    updated_at: dt.datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "attempt", max(0, int(self.attempt)))
        object.__setattr__(self, "renewals", max(0, int(self.renewals)))
        if self.streak is not None:
            object.__setattr__(self, "streak", max(0, int(self.streak)))

    def matches(self, attempt: int) -> bool:
        """Is this judgment about the attempt the tracker's counter now says?

        The counter and the judgment have one writer and are written in one
        act, so they agree unless something outside apiary moved the counter.
        Something outside apiary moving it is not an error - "GitHub wins,
        every cycle, on every disagreement" is the oldest rule in
        `orchestrator/reconcile.py`, and a human resetting a counter after
        fixing the environment is the workflow ADR 0002 quotes. It does mean
        the signature is about an attempt that no longer exists, so `stale`
        below drops it rather than charging a fresh attempt against an old
        blocker.
        """
        return self.attempt == max(0, int(attempt))

    def stale(self) -> TaskJudgement:
        """This row with its signature dropped and its stamp left alone.

        Not an empty judgment: the renewal count is a history of the task and
        survives a counter edit, where the signature is a statement about one
        attempt and does not.
        """
        return replace(self, blocker="", streak=None)


@runtime_checkable
class TaskStore(Protocol):
    """Where apiary's judgments live. Three methods, and no fourth.

    Narrow on purpose. ADR 0002 defers the organization database until
    something needs it and asks that the seam be designed now, and a seam is
    only worth designing if the remote implementation can satisfy it without
    the callers noticing - which rules out anything that leaks a file path, a
    cursor, a transaction or a query language.

    `read()` returns the whole project rather than one row because that is the
    call the reconcile loop actually makes: a cycle wants every task's judgment
    at once, and a per-row API would turn one round trip into one per issue
    against a database that is remote in the case this seam exists for. It is
    the same argument `reconcile.Snapshot` makes about the issue listing, and
    it is why `read()` has no key parameter to grow one.

    `write()` takes a whole `TaskJudgement` rather than the fields, so a
    backend can persist it in one statement and a caller cannot half-write one.

    A `Protocol` rather than an ABC for the reason every seam in this codebase
    is one: a test that needed a real database to reach a retry rule is a test
    that does not run.
    """

    def read(self) -> Mapping[TaskRef, TaskJudgement]:
        """Every judgment this store holds, keyed by task ref.

        Raises `StoreCorrupt` rather than returning an empty mapping when the
        store cannot be trusted - see `StoreCorrupt` for why that is not
        pedantry.
        """
        ...

    def write(self, judgement: TaskJudgement) -> None:
        """Persist one judgment, replacing any the store held for that ref.

        Durable before it returns. Callers rely on that: the write happens
        *before* the label that re-readies the task, so a crash between the two
        costs an attempt rather than granting a free one
        (`docs/issue-contract.md` §5).
        """
        ...

    def close(self) -> None:
        """Release whatever the backend holds. Safe to call twice."""
        ...


def record_judgement(
    store: TaskStore | None,
    ref: TaskRef,
    attempt: int,
    *,
    blocker: str = "",
    streak: int | None = None,
    renewals: int = 0,
) -> None:
    """Write one attempt's judgment, from callers that hold fields not a record.

    Three modules consume an attempt - the reconciler, the check gate and the
    mergeability gate - and all three hold a `Transition`, which this package
    must not import: `store/` sits below `orchestrator/` and a dependency
    pointing back up is how a seam stops being swappable. So the fields arrive
    loose and the record is built here, in one place, rather than three times
    with three chances to forget one.

    `store=None` writes nothing. That is for the callers whose whole subject is
    planning - `plan_checks` and friends decide transitions without applying
    them - and never for a real run: a run whose gates consumed attempts
    without recording them would let a task retry forever, which is the failure
    this package exists to bound. `Reconciler` therefore requires a store and
    passes it into both gates, so the only way to reach the `None` branch is to
    call the gate directly.
    """
    if store is None:
        return
    store.write(
        TaskJudgement(
            ref=ref,
            attempt=attempt,
            blocker=blocker,
            streak=streak,
            renewals=renewals,
            updated_at=dt.datetime.now(dt.timezone.utc),
        )
    )
