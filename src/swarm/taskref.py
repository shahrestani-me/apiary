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

**Labellable.** `label_value` is the ref reduced to the characters Docker
permits in a container name, and it exists so that `swarm/containers/` can
label, name and find a worker's container without importing an adapter to
un-mint the ref first. A tracker id is not a safe token - `#42` cannot be a
container name - so somebody has to make one, and the choice is between the
type that owns the value and every consumer inventing its own rule.

**A pull request is not a task, and now says so in the type (#185).** `PullRef`
sits below `TaskRef` in this module and shares nothing with it but the shape.
It exists because `Merge` carries a task's number beside a pull request's, and
while both were `int` the two could be filled in the wrong order and still
type-check: a `Merge` built with the pull request in the issue's place minted a
perfectly valid ref, just one addressing a pull request - so the refusal was
filed under an identity no outcome answered to, and `swarm:done` went out for a
merge that never happened. The type gate caught a *retype* and could not catch
a *mis-sourcing*, because `task_ref()` never knew which of the two ints it had
been handed.

Two nominal types is the whole fix: `TaskRef` and `PullRef` are unrelated
classes, so mypy rejects each in the other's place and the swap stops being
expressible rather than being caught a layer later by a guard. They are
deliberately *not* siblings under a common base - a base class is a hole,
because anything annotated with it accepts both again and the two vocabularies
re-merge at the first helper that takes one.

## Why `PullRef` is here and not in `github/refs.py`

Its minters are there, so the pair is split across the line and that deserves an
answer rather than an accident. `refs.py` is the **tracker** adapter -
`branches.py` says so outright, "a Linear adapter would replace it wholesale" -
and a code-host type parked inside it would be deleted by a tracker swap that
has nothing to do with pull requests. ADR 0001 §4 keeps the code host GitHub
permanently; the tracker is the half that moves. So the type outlives `refs.py`
and lives where things that outlive adapters live, while the minting - which
*is* GitHub's spelling and *would* move - stays down there with the rest of it.

The cost is that this module's summary line overstates by one word: it holds the
identity of a task, and one address that is not a task at all. Read the next
paragraph before assuming the two are the same kind of thing.

`PullRef` is *not* opaque in `TaskRef`'s sense, and the asymmetry is ADR 0001's.
A task system is pluggable, so core may not know how its ids are spelled; the
*code host* is GitHub and stays GitHub-shaped, so a pull request's number is an
address rather than an identity. `github/refs.py` mints and un-mints it -
`pull_ref` / `pull_number`, beside `task_ref` / `issue_number` - and un-minting
it is ordinary rather than a smell: a `PullRef` exists precisely so that it can
be handed back to `PUT /pulls/{n}/merge` in the end. What the type buys is that
the hand-back is a written-out call at the API boundary rather than an int
drifting through four records on its way there.

**`PullRef` sorts too, over the same `_natural_key` (#208).** It shipped in #185
without an ordering because nothing in the orchestrator ordered pull requests by
number at the time; `derived.py` does, so `PullFact.number` was stuck as an `int`
until the type could be sorted. One key rather than two is the whole of the
choice: two orderings over values that look this much alike is how they diverge,
and a divergence between them would be a sort that reads correctly at both call
sites and disagrees between them.

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

#: The punctuation Docker allows in a container name; `str.isalnum` covers the
#: rest. Anything else is dropped by `TaskRef.label_value`.
_SAFE = frozenset("_.-")


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
    non-determinism both orderings exist to remove: `find_cycle` promises the
    same graph names the same ring every run, and `derived.py` promises the same
    world names the same pull request.
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

    @property
    def label_value(self) -> str:
        """This ref as a Docker-safe token: a label value and a name fragment.

        Docker names match `[a-zA-Z0-9][a-zA-Z0-9_.-]*`, so the rule is to keep
        exactly those characters and drop the rest. `#42` becomes `42` and
        `ENG-123` stays `ENG-123` - which is why moving the container layer onto
        this changed no label and no container name for the tracker apiary
        currently speaks to. That is a consequence, not the design: the point is
        that `containers/` derives the token from the ref it was handed, so a
        tracker whose ids are `ENG-123` labels its containers correctly rather
        than by whatever integer happened to be passed alongside.

        **Not injective, deliberately.** Two ids differing only in characters
        Docker forbids collapse to one token. Within a run that cannot happen -
        one run reads one tracker, whose ids share a format - and the label is a
        *filter* narrowing a listing that is already scoped to `apiary.run`,
        never the authority on what a task is. The ledger is that. A tracker
        that needed the distinction would need a reversible encoding here, and
        an unreadable one: this stays readable because a human greps
        `docker ps` with it.

        Which is why this is not the token a branch name uses.
        `github/branches.py` needs the reversible one - recovery reads a name
        off a remote with no ledger in hand to disambiguate it - and pays the
        readability for it. Two safe tokens is the right number here: the trade
        genuinely lands the other way in the two places.
        """
        token = "".join(char for char in self.value if char in _SAFE or char.isalnum())
        token = token.lstrip("_.-")
        if not token:
            # No real ref reaches this - every tracker's ids carry something
            # alphanumeric - but a token that is empty or starts with a
            # separator is one Docker refuses at `create`, and finding that out
            # at spawn time would blame the container layer for a bad ref.
            raise ValueError(f"task ref {self.value!r} has no Docker-safe form")
        return token


@total_ordering
@dataclass(frozen=True)
class PullRef:
    """One pull request's address on the code host, in that host's spelling.

    A separate class from `TaskRef` rather than a flag on it, and that is the
    entire mechanism: mypy rejects one where the other is expected, so the two
    numberings that `Merge`, `Mergeability` and `Decision` carry side by side
    can no longer be filled in the wrong order (#185). See this module's
    docstring for why they share no base class.

    Construct these through the adapter (`github.refs.pull_ref`) and read the
    number back out through `github.refs.pull_number` when an endpoint needs
    it. Unlike `TaskRef`, un-minting is the expected end of the value's life,
    not a leak - a pull request number is an API address and never an identity
    the internal model keys anything on.

    **Ordered, and for `TaskRef`'s reason rather than its caller (#208).**
    `TaskRef` sorts so `find_cycle` names the same ring every run; this sorts so
    `derived.py` names the same *pull request* every run. A task can have more
    than one - `orchestrator/recovery.py` documents the retry that opens a
    second rather than updating the first - and three sites there pick one of
    them to report: `_landed` and `_merged_pull` take the lowest-numbered merge,
    `_open_pull` breaks a tie between two open pull requests on one attempt.
    Each was an `int` comparison, so each was already deterministic, and the
    ordering is what let `PullFact.number` become this type without any of them
    silently falling back to insertion order. The order is the same *natural*
    one, over the same `_natural_key`: `#9` before `#10`.

    It is deliberately **not** an ordering against `TaskRef`. Comparing the two
    raises, exactly as assigning one to the other is rejected - a total order
    spanning both would be the shared vocabulary the no-base-class choice exists
    to refuse, and `sorted()` over a mixed list is never a thing a caller wants.
    """

    value: str

    def __post_init__(self) -> None:
        # Same rule as `TaskRef`, for the same reason: a blank address surfaces
        # much later, as a request to an endpoint that is missing its path
        # segment rather than as a bad record.
        if not self.value or self.value.strip() != self.value:
            raise ValueError(f"pull ref {self.value!r} is empty or padded")

    def __str__(self) -> str:
        return self.value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, PullRef):
            return NotImplemented
        return _natural_key(self.value) < _natural_key(other.value)
