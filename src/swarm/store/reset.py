"""Giving one task its retry budget back, as a value rather than a print.

**Why this is a module and not a branch of `cli._reset`.** ADR 0002 quotes the
workflow it replaces: a person fixes whatever the environment was doing wrong
and edits the issue marker's `attempt=` so a stuck task gets another go. ADR
0005 moved that counter into apiary's own store and decision 4 says the
affordance "moves rather than being lost". `swarm reset` was that move, and it
moved the affordance to *a terminal* - which is not where the operator is when
they are looking at a board telling them a task needs a human. #293 added the
run's decision report, and it ends by printing a command the reader has to go
somewhere else to type.

So the decision belongs in one place and the two front doors belong to the
callers: `cli._reset` prints and asks for confirmation, the console posts a
button. Both call `reset_budget`, so they cannot disagree about what a reset
*is* - and the thing they must not disagree about is subtle, which is the rest
of this docstring.

**Why the row is written rather than deleted.** A row saying `attempt=0,
blocker='', streak=None` is arithmetically identical to no row at all -
`_retry_or_give_up` falls back to the counter when the streak is absent, and a
miss cannot renew because renewal is gated on a blocker being present. Deleting
is *not* identical, twice over:

- `ledger._judged` reads the issue marker when the store has no row, and that
  marker is a fossil nothing has maintained since ADR 0005. Deleting would
  resurrect whatever number it happens to carry.
- `seed_attempt_floor` seeds only tasks the store has never judged, so a deleted
  row would be refilled from the branch listing at the next run's startup -
  which is exactly the number the human is trying to get out from under.

Writing the row is what makes a reset survive the floor.

`renewals` is preserved, because it is a history of the task rather than a claim
about one attempt (`store.TaskJudgement`), and nothing branches on it.

**Tracker-agnostic, and `tests/test_framework_boundary.py` enforces it.** The ref
arrives as the store already spells it - `#12` on GitHub, `ENG-123` on Linear -
and nothing here mints one or parses one. A module that reached
`github.refs.task_ref` would bake the GitHub adapter into the one gesture that
has no business knowing which tracker is configured, which is the thing epic
#140 exists to remove.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from ..taskref import TaskRef
from .base import TaskJudgement
from .sqlite import SqliteTaskStore

__all__ = ["Reset", "reset_budget"]


@dataclass(frozen=True)
class Reset:
    """What a reset did, or why it did nothing.

    A value rather than a bool, because both callers need to *say* what
    happened: the CLI prints the before-and-after so a human can see the numbers
    they are changing, and the console shows the same sentence on the page.
    """

    ref: TaskRef
    repo: str
    #: False when the store has never judged this ref. Not an error: a task with
    #: no row already has a full budget, so there is nothing to give back.
    found: bool
    #: What the row said before. Zeroes when there was none.
    attempt_was: int = 0
    blocker_was: str = ""
    streak_was: int | None = None
    attempt_now: int = 0
    #: Every ref the store does hold, when this one was not among them.
    #: "Nothing to reset" and "you typed the wrong spelling" look identical
    #: otherwise, and the second is the likely one the first time somebody uses
    #: this against a tracker whose refs are not numbers.
    known: tuple[TaskRef, ...] = ()

    @property
    def changed(self) -> bool:
        return self.found

    def sentence(self) -> str:
        """One line, for a terminal or a page."""
        if not self.found:
            said = (
                f"{self.ref} has no judgment in {self.repo}'s store, so it already has "
                f"a full budget - or it is spelled differently here."
            )
            if self.known:
                said += f" The store holds: {', '.join(str(one) for one in self.known)}."
            return said
        streak = self.streak_was if self.streak_was is not None else "none"
        return (
            f"{self.ref} in {self.repo}: attempt {self.attempt_was} -> {self.attempt_now}, "
            f"blocker {self.blocker_was or 'none'} -> none, streak {streak} -> none. "
            f"The next run may dispatch it again."
        )


def reset_budget(repo: str, ref: TaskRef, *, attempt: int = 0, apply: bool = True) -> Reset:
    """Zero one task's per-blocker budget in `repo`'s store.

    `apply=False` reads without writing, which is what lets a caller show the
    before-and-after and *then* ask - `cli._reset`'s confirmation prompt would
    otherwise have to describe a write it had already made.
    """
    if attempt < 0:
        raise ValueError(f"attempt {attempt} is negative; a counter starts at 0")

    with SqliteTaskStore.open(repo) as store:
        held = store.read()
        judgement = held.get(ref)
        if judgement is None:
            return Reset(
                ref=ref,
                repo=repo,
                found=False,
                known=tuple(sorted(held)),
            )
        outcome = Reset(
            ref=ref,
            repo=repo,
            found=True,
            attempt_was=judgement.attempt,
            blocker_was=judgement.blocker,
            streak_was=judgement.streak,
            attempt_now=attempt,
        )
        if apply:
            store.write(
                TaskJudgement(
                    ref=ref,
                    attempt=attempt,
                    blocker="",
                    streak=None,
                    renewals=judgement.renewals,
                    updated_at=dt.datetime.now(dt.timezone.utc),
                )
            )
        return outcome
