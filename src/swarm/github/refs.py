"""Minting and un-minting GitHub's task refs. **The only module that may.**

`TaskRef` is opaque above this line (`swarm/taskref.py`), which is only true if
exactly one place knows that this tracker spells a ref `#42`. That place is
here. Everything else - readiness, the reconciler, the container lookup - holds
refs, compares them, keys dictionaries on them and prints them, and asks this
module the two questions that need the spelling:

- `task_ref(42)` when a GitHub payload, a `## Blocked by` line, a docker label
  or an artifact filename has just produced a number, and
- `issue_number(ref)` when something that is addressed by issue number is about
  to be called.

**Pull requests mint here too, and for a different reason (#185).** `pull_ref` /
`pull_number` are the same pair for the other numbering GitHub hands out. A
pull request is not a task and gets no `TaskRef` - it has no identity the
internal model keys anything on - but while its number was a bare `int` it sat
beside an issue's on `Merge`, `Mergeability` and `Decision` with nothing but the
field name keeping them apart, and a record built with the two the wrong way
round type-checked. `PullRef` is nominally distinct from `TaskRef`, so it no
longer does.

The asymmetry between the two pairs is worth reading off the docstrings rather
than guessing: `issue_number` is a last resort, because a caller who un-mints a
task ref to decide something has assumed refs are numeric. `pull_number` is
not - a pull ref's whole purpose is to reach `PUT /pulls/{n}/merge` eventually,
and the type exists to make that hand-back one written-out call at the API
boundary instead of an int drifting through four records to get there.

That second list is the GitHub API in every case but one: `ContainerManager.find`
turns a ref back into the `apiary.issue` label value, because a docker label and
a container name were written with a number at `docker create` and changing that
is a behaviour change (`docs/adr/0001-task-systems-are-integrations.md` wants the
container to carry the ref itself; that is a container-layer ticket).

What no caller may do is reach for `issue_number` to *decide* something - to
sort, to compare, or to derive a name - because that is a module assuming refs
are numeric, which is exactly what the ADR says no core module may do. `#42` is
not a branch-safe token either, and this module once expected to grow a third
function for that; #144 put it in `github/branches.py` instead, because the
encoding there escapes bytes rather than knowing a spelling, and it carries an
attempt counter - which is not a property of a ref and has no business here.
"""

from __future__ import annotations

import re

from ..taskref import PullRef, TaskRef

#: How this adapter spells a ref. Anchored, because `issue_number` is the
#: inverse of `task_ref` and nothing else: a ref another adapter minted must
#: fail loudly here rather than yield some plausible number.
_ISSUE_REF_RE = re.compile(r"^#(\d+)$")

#: How this adapter spells a pull request. The same shape as `_ISSUE_REF_RE` on
#: purpose - `#101` is what a human reads in every log line this repository has
#: ever printed, and #185 is a change to the *types*, not to the output. The
#: spellings may coincide because the classes do not: `PullRef("#101")` is not
#: a `TaskRef` and no annotation accepts both.
_PULL_REF_RE = re.compile(r"^#(\d+)$")


def task_ref(number: int) -> TaskRef:
    """The ref for one GitHub issue number."""
    return TaskRef(f"#{int(number)}")


def issue_number(ref: TaskRef) -> int:
    """The issue number inside a ref this adapter minted.

    Raises on anything else. A ref from another tracker reaching a GitHub API
    call is a wiring bug, and the failure mode of guessing - addressing some
    unrelated issue in this repository - is worse than the exception.
    """
    match = _ISSUE_REF_RE.match(ref.value)
    if match is None:
        raise ValueError(f"task ref {ref.value!r} was not minted by the GitHub adapter")
    return int(match.group(1))


def pull_ref(number: int) -> PullRef:
    """The ref for one GitHub pull request number.

    Called wherever a payload has just produced one - `PullState.from_payload`,
    `Mergeability.from_payload` - so that the number is typed from the moment
    it is read rather than several records later.
    """
    return PullRef(f"#{int(number)}")


def pull_number(ref: PullRef) -> int:
    """The pull request number inside a ref this adapter minted.

    Raises on anything else, for `issue_number`'s reason: guessing would
    address some unrelated pull request in this repository.

    Unlike `issue_number`, calling this is not a smell. Every use is an endpoint
    that takes `{n}` in its path - `GET /pulls/{n}`, `PUT /pulls/{n}/merge`,
    `PUT /pulls/{n}/update-branch` - or an artifact field that has to be JSON.
    What the type forbids is the middle of the pipeline, not its ends.
    """
    match = _PULL_REF_RE.match(ref.value)
    if match is None:
        raise ValueError(f"pull ref {ref.value!r} was not minted by the GitHub adapter")
    return int(match.group(1))
