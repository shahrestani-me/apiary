"""The identity of a task, as the internal model is allowed to know it.

`github/ledger.py` already says the important half out loud: **identity is the
marker id, not the issue number**, because replanning matches on the slug and
keying on the number would fork the ledger on every replan. The dependency
graph never got the message - it was keyed on `int`, and so were readiness,
`Transition` and the container lookup. So identity was a slug everywhere except
the one place that decides what may run.

`TaskRef` is that missing half: a handle to a work item in whatever tracker
minted it, **opaque to everything above the adapter that minted it**. GitHub
mints `#42`; a Linear adapter would mint `ENG-123`, a Jira one `PROJ-7`. Core
holds them, compares them, uses them as dictionary keys and prints them, and
nothing in core may take one apart. `docs/adr/0001-task-systems-are-integrations.md`
is why: apiary ships no tracker integration code, so a core module that can
read a number out of a ref is a core module that only works for one tracker.

Three properties, and each of them is one of those uses:

**Opaque.** The value is a string the adapter chose, and the only two functions
allowed to build one from - or recover one back to - a tracker's own id live
next to that adapter (`github/refs.py`). Nothing here parses `value`.

**Ordered.** `find_cycle` walks nodes and successors in sorted order so the same
graph always names the same ring, and a cycle that reports a different path
every run is much harder to fix. Ordering is therefore part of the type rather
than something a caller derives - which is what keeps `sorted()` in core from
having to know the shape it is sorting. The order is *natural*: digit runs
compare numerically, so `#9` sorts before `#10` exactly as an `int` key did,
and `ENG-9` before `ENG-10` for the tracker that has not been written yet.

**Printable.** `str(ref)` is the value, so `f"{ref}"` renders `#42` and every
message a human reads is unchanged by the retype. That is a property of the
adapter's chosen spelling, not a promise core may rely on: core prints refs, it
does not build them.

**What this does not yet buy.** `LedgerEntry` carries a `number` beside its
`ref`, and the modules above still read that number wherever they address the
GitHub API - a branch name, a label write, a comment. `TaskRef` is opaque; the
*ledger record* is still GitHub-shaped, and swapping the tracker means dealing
with those sites too. What this type finishes is narrower and was the actual
latent bug: the dependency graph, which decides what may run, no longer assumes
identity is an integer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache, total_ordering

#: Splits a value into alternating text and digit runs. Used *only* for
#: ordering - it never tells a caller what the parts mean, which is the line
#: between sorting an opaque value and parsing one.
_DIGITS = re.compile(r"(\d+)")


# Cached because `__lt__` recomputes both operands' keys on every comparison,
# and a sort of n refs makes O(n log n) of them over a set of values that is
# small, fixed per cycle and reused every cycle.
@lru_cache(maxsize=4096)
def _natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    """A sort key where digit runs compare as numbers, not as text.

    Each chunk is tagged, so an int is never compared against a str: two
    numeric chunks compare numerically, two textual chunks lexicographically,
    and a numeric chunk sorts before a textual one.

    The value itself is the last element, and it is not decoration. Without it
    the key is not injective - `#042` and `#42` share one - so two unequal refs
    would compare neither equal nor less-than in either direction, and
    `sorted()` would fall back to insertion order. That is precisely the
    non-determinism the ordering exists to remove: `find_cycle` promises the
    same graph names the same ring every run.
    """
    chunks: tuple[tuple[int, int | str], ...] = tuple(
        # `isdecimal`, not `isdigit`: the latter is true of superscripts and
        # other digit-like codepoints that `int()` then refuses.
        (0, int(chunk)) if chunk.isdecimal() else (1, chunk)
        for chunk in _DIGITS.split(value)
        if chunk
    )
    return (*chunks, (2, value))


@total_ordering
@dataclass(frozen=True)
class TaskRef:
    """One task's identity, in the spelling its tracker uses.

    Construct these through the adapter (`github.refs.task_ref`), not here: the
    constructor is public only because a dataclass's is, and a core module
    calling it would be a core module inventing a tracker's id format.
    """

    value: str

    def __post_init__(self) -> None:
        # A blank ref is a bug that would otherwise surface much later as a
        # graph node nothing can resolve, or as a `dict` key that silently
        # collides with another blank one.
        if not self.value or self.value.strip() != self.value:
            raise ValueError(f"task ref {self.value!r} is empty or padded")

    def __str__(self) -> str:
        return self.value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, TaskRef):
            return NotImplemented
        return _natural_key(self.value) < _natural_key(other.value)
