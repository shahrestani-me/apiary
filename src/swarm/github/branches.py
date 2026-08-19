"""The worker's branch name, and the pair that can be read back out of it.

`docs/adr/0001-task-systems-are-integrations.md` makes agent execution state
**derived, not stored**: `claimed` is a live container, `review` is an open pull
request, `landed` is a merge. That table is only affordable if a crash costs
nothing, and the thing a crash threatens is the association between a work item
and the work already done for it. Today that association lives in a
`swarm:claimed` label - a durable write into a customer's tracker, which is the
one thing decision 2 forbids.

A branch is the natural place for it to live instead. It is already durable,
already on the code host, already apiary's own artefact, and it already exists
for every task that got as far as a worker. `swarm/issue-<n>` simply did not
carry enough: the number is only meaningful to one tracker, and nothing in the
name said which attempt produced it. `apiary/<ref>-attempt-<n>` carries both, so
an orchestrator that has lost every scrap of local memory can list the remote's
branches and know what was in flight and how much budget it had spent.

## Why the encoding is reversible, unlike `TaskRef.label_value`

#142 already made a safe token out of a ref, and deliberately made it lossy: the
Docker label is a *filter* narrowing a listing that is already scoped to one
run, the ledger is the authority on what a task is, and a human greps `docker
ps` with the value, so readable-and-collapsing beat reversible-and-unreadable.

Here the same trade lands the other way, because the branch name is not
narrowing anything - it *is* the record. Recovery reads a name off the remote
with no ledger in hand to disambiguate it, so two refs that collapsed to one
token would be two tasks recovery could not tell apart. So the ref is
percent-encoded: every byte outside `[A-Za-z0-9_-]` becomes `%XX` over UTF-8,
which is injective, and which git accepts in a ref name. `ENG-123` survives
verbatim and reads as itself; GitHub's `#42` becomes `%2342`, which is uglier
than the old `issue-42` and is the price of a name that a second tracker cannot
alias.

Left out of the safe set on purpose: `.`, because git forbids `..`, a trailing
`.` and a `.lock` suffix inside a ref name, and none of that reasoning is worth
carrying for a character no tracker needs; and `/`, because a ref containing one
would create a directory in `refs/heads/` that collides with `apiary/` itself.

## Why here and not in `github/refs.py`

`refs.py` predicted this ticket - "`#42` is not a branch-safe token, so the
ticket that puts the ref in a branch name will need a third function here" - and
that is the one part of its note this module declines. `refs.py` is the *tracker*
adapter, the single place that knows a ref is spelled `#42`; a Linear adapter
would replace it wholesale, and a branch encoder living inside it would be
replaced along with it for no reason. Nothing below knows the spelling of
anything: it escapes bytes. The attempt counter settles it - an attempt is not a
property of a ref, and `refs.py` has no business holding one.

The other candidate was `worker/pr.py`, where the ticket puts it. That one is
not a judgement call: `github/ledger.py` mints this name for every entry it
loads, and `worker/pr.py` imports `worker/entrypoint.py`, which imports
`github/ledger.py`. The pair has to sit below both of them or the import cycles.

## Parsing never raises

Every name this module is shown comes off a remote, and a remote holds whatever
anybody pushed: branches from before this ticket, a human's `fix/typo`, a name
somebody hand-edited. `parse_task_branch` answers `None` for all of them, and
callers report the count rather than acting on it. A parser that raised would
turn one stray branch into a crashed recovery sweep, which is the failure this
whole module exists to prevent.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

from ..taskref import TaskRef

__all__ = [
    "ATTEMPT_SEPARATOR",
    "BRANCH_PREFIX",
    "TaskBranch",
    "parse_task_branch",
    "task_branch",
]

#: Everything apiary pushes lives under this, so a `git branch --list
#: 'apiary/*'` is the whole fleet and nothing of the human's. The old
#: `swarm/` prefix is deliberately not reused: a name under it means a branch
#: from before this ticket, which is exactly the thing the parser must refuse.
BRANCH_PREFIX = "apiary/"

#: Spelled out rather than a bare `-`, because the ref is the free-form half:
#: a ref may itself end in `-3`, and a one-character separator would make
#: `apiary/ENG-3` ambiguous with a ref `ENG` on attempt 3.
ATTEMPT_SEPARATOR = "-attempt-"

#: The bytes that survive encoding unescaped. See the module docstring for what
#: is left out and why.
_UNRESERVED = frozenset(string.ascii_letters + string.digits + "_-")

#: `%XX`, as produced by `_encode`. One capture group, because `_decode` splits
#: on this and reads the odd-numbered pieces.
_ESCAPE_RE = re.compile("%([0-9A-Fa-f]{2})")

#: The whole name. The ref group is greedy and the attempt is anchored to the
#: end, so a ref containing `-attempt-` splits at the *last* one - which is what
#: makes `apiary/ENG-attempt-3-attempt-5` read back as `(ENG-attempt-3, 5)`
#: rather than as anything else.
_NAME_RE = re.compile(
    f"^{re.escape(BRANCH_PREFIX)}"
    f"(?P<ref>[A-Za-z0-9_%-]+)"
    f"{re.escape(ATTEMPT_SEPARATOR)}"
    r"(?P<attempt>\d+)$"
)

#: Git stores a ref as a path, so each component is a filename, and 255 bytes is
#: the ceiling every filesystem apiary runs on shares. Checked at mint time so a
#: ref long enough to break the push is reported by the module that built the
#: name, not by a git error three layers away at the end of a finished task.
MAX_COMPONENT_BYTES = 255


@dataclass(frozen=True)
class TaskBranch:
    """What a branch name says: which task, and which attempt at it.

    Frozen and two-field on purpose - it is the return type of a parse, and a
    caller that wants more has to go and read the thing it wants, rather than
    trusting a name anybody could have pushed.
    """

    ref: TaskRef
    attempt: int

    def __str__(self) -> str:
        return task_branch(self.ref, self.attempt)


def task_branch(ref: TaskRef, attempt: int) -> str:
    """The branch one worker pushes for `(ref, attempt)`.

    The only place this name is built. `parse_task_branch` is its inverse and
    the round trip is the property both of them exist for: whatever a tracker
    calls its work items, the pair that went in comes back out.
    """
    attempt = int(attempt)
    if attempt < 0:
        # A negative attempt would encode as `-1` and put a second `-` next to
        # the separator, which parses back as a different pair. It is a caller
        # bug either way; failing here names the caller.
        raise ValueError(f"attempt {attempt} is negative")
    component = f"{_encode(ref.value)}{ATTEMPT_SEPARATOR}{attempt}"
    if len(component.encode("utf-8")) > MAX_COMPONENT_BYTES:
        raise ValueError(
            f"the branch name for task ref {ref.value!r} is "
            f"{len(component.encode('utf-8'))} bytes, over git's {MAX_COMPONENT_BYTES}"
        )
    return f"{BRANCH_PREFIX}{component}"


def parse_task_branch(name: str) -> TaskBranch | None:
    """The `(ref, attempt)` inside a branch name, or None if it is not one of ours.

    None covers every branch apiary did not mint under the current scheme -
    `main`, a human's, and every `swarm/issue-<n>` from before this ticket. See
    the module docstring: callers count these, they do not act on them.
    """
    match = _NAME_RE.match(name or "")
    if match is None:
        return None
    try:
        ref = TaskRef(_decode(match.group("ref")))
    except (UnicodeDecodeError, ValueError):
        # `_decode` on bytes that are not UTF-8, or a token that decodes to
        # something `TaskRef` refuses - a name hand-crafted to look like ours.
        return None
    return TaskBranch(ref=ref, attempt=int(match.group("attempt")))


def _encode(value: str) -> str:
    """A ref as a git-safe, reversible token. See the module docstring."""
    token = "".join(
        chr(byte) if chr(byte) in _UNRESERVED else f"%{byte:02X}"
        for byte in value.encode("utf-8")
    )
    if token.startswith("-"):
        # Only the first character, and only because a leading `-` makes the
        # name look like an option to every git command a human types by hand.
        # Escaping it changes nothing about the round trip.
        token = f"%2D{token[1:]}"
    return token


def _decode(token: str) -> str:
    """The inverse of `_encode`, byte-wise so multi-byte UTF-8 reassembles."""
    # `re.split` on a pattern with one group alternates literal, capture,
    # literal - so the odd indices are the escapes and the even ones are text.
    raw = bytearray()
    for index, part in enumerate(_ESCAPE_RE.split(token)):
        if index % 2:
            raw.append(int(part, 16))
        else:
            raw += part.encode("utf-8")
    return raw.decode("utf-8")
