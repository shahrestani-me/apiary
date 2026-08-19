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

from ..taskref import TaskRef

#: How this adapter spells a ref. Anchored, because `issue_number` is the
#: inverse of `task_ref` and nothing else: a ref another adapter minted must
#: fail loudly here rather than yield some plausible number.
_ISSUE_REF_RE = re.compile(r"^#(\d+)$")


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
