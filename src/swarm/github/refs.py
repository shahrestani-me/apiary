"""Minting and un-minting GitHub's task refs. **The only module that may.**

`TaskRef` is opaque above this line (`swarm/taskref.py`), which is only true if
exactly one place knows that this tracker spells a ref `#42`. That place is
here. Everything else - readiness, the reconciler, the container lookup - holds
refs, compares them, keys dictionaries on them and prints them, and asks this
module the two questions that need the spelling:

- `task_ref(42)` when a GitHub payload, a `## Blocked by` line or a docker
  label has just produced a number, and
- `issue_number(ref)` immediately before a `GitHubClient` call, which addresses
  issues by number and always will.

Callers of `issue_number` are therefore the API call sites and nothing else. A
module reaching for it to *decide* something - to sort, to compare, to derive a
branch name - is a module assuming refs are numeric, which is exactly what
`docs/adr/0001-task-systems-are-integrations.md` says no core module may do.
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
