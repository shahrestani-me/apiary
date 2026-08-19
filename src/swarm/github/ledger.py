"""The read half of the ledger: GitHub issues in, `TaskRecord`s out.

`docs/architecture-v2.md` says GitHub is the database, and
`docs/issue-contract.md` is the schema for it. This module implements that
schema and nothing else - it parses, it maps, it never decides policy. When an
issue is malformed it says so and hands the problem back; labelling the
offender `swarm:failed` and commenting the reason is the orchestrator's call
(`docs/issue-contract.md` §1.4).

Three things here are easy to get subtly wrong, and all three are load-bearing:

**A section heading is a whole line, and fences are opaque.** `body.find("##
Blocked by")` mis-sections issue #11, whose `## Goal` sentence carries that
exact string inside a code span, and reports a blocked task as having no
dependencies. It looked ready when it was not. Line anchoring fixes the code
span; fence tracking fixes the meta-issue that quotes the schema. They are two
problems, and `_scan` solves both before anything else runs.

**Identity is the marker id, not the issue number.** The number addresses an
issue; the slug identifies the work. Replanning (#10) matches on the slug,
which the planner regenerates and which can never equal an integer - keying on
the number would fork the whole ledger on every replan. The dependency graph
now says the same thing: `blocked_by` holds `TaskRef`s, minted here and opaque
above this package (`swarm/taskref.py`, #142), so the one place that decides
what may run no longer assumes identity is an integer.

**The status mapping is lossy in both directions.** `ready`/`blocked` collapse
into `pending` and `claimed`/`review` collapse into `running`, so `TaskStatus`
is a projection maintained for v1 code and never the thing v2 reasons about.
The label set stays on `LedgerEntry`, and anything needing to know whether a PR
is open reads `swarm:review`, not `status == "running"`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from ..state import TaskRecord, TaskStatus
from ..taskref import TaskRef
from .branches import task_branch
from .client import GitHubClient
from .refs import task_ref

# The sections a body must have. Order matters only for reporting the first
# missing one, so the error a human sees names the section they would have
# written first.
REQUIRED_SECTIONS = ("Goal", "Files", "Verify", "Blocked by")

# Sections the parser recognises, required or not. `SECTIONS` used to be one
# tuple doing both jobs - it built `_KNOWN_HEADING_RE` *and* it was the
# required-section loop - which is the entire reason "exactly four sections"
# read as immovable. There was no way to express "recognised but optional".
#
# `## Stack` is the first optional one, and it is emitted **last**, after
# `## Blocked by`. See `render_body` for why - and for what turned out not to
# be a reason.
OPTIONAL_SECTIONS = ("Stack",)
KNOWN_SECTIONS = REQUIRED_SECTIONS + OPTIONAL_SECTIONS

#: Stack ids a `## Stack` section may name. The contract's vocabulary, which is
#: not the same list as `containers.manager.DEFAULT_STACK_IMAGES` and must not
#: import it: that map is about what *this host* can run, this is about what an
#: issue may declare, and `github/` depending on `containers/` would be the
#: wrong way round. `test_every_generatable_stack_is_a_declarable_one` pins that
#: this set covers that map, so the two cannot drift apart silently.
#:
#: An id outside this set is a `ContractError`, never a silent default. A task
#: declaring `## Stack` `rust` in a repository with no Rust image should fail
#: at parse time, where the message names the issue - not at spawn time, where
#: it costs a claim, or worse at verify time, where it costs an attempt.
KNOWN_STACKS = frozenset({"python", "node", "react"})

#: Files a task may **commit** but never **authors**, per stack.
#:
#: `commit_edits` stages exactly the declared `## Files`, and that rule is
#: right: `git add -A` after a verify run would sweep `node_modules`, every
#: cache the command wrote and whatever else it dropped in the tree straight
#: into the pull request. But a lockfile is neither declarable nor writable:
#:
#: - the model cannot author one. A measured Expo lockfile is 16,347 lines,
#:   roughly 180k tokens against a 16,384 window, and it carries SHA-512
#:   integrity hashes that cannot be produced by generation at all;
#: - `edit.py`'s protocol demands "the COMPLETE new contents of every file you
#:   change", so it can never arrive as an ordinary edit;
#: - without it the PR carries a `package.json` change and no lock, CI re-runs
#:   the command on neutral ground, and `npm ci` fails. "Add a dependency" is
#:   unimplementable.
#:
#: So there is a third category between "the task's files" and "everything
#: else": paths the *gate* produces, which the worker commits if they appear
#: and ignores if they do not.
#:
#: **A per-stack constant rather than a sixth contract section.** #105 left
#: that call open and preferred the constant unless something concrete needed
#: the generality. Nothing in #87 does: the set is a property of the toolchain,
#: not of the task, and every case the epic has is one row of this table. A
#: section would also hand the model a way to widen what gets committed, which
#: is the one thing the staging rule exists to prevent.
GENERATED_FILES: Mapping[str, tuple[str, ...]] = {
    "python": (),
    # `node --test` needs no dependencies, so nothing is generated today. The
    # entry is here because the moment a Node task adds one, this is where the
    # lockfile has to be named, and an empty tuple says "considered" where a
    # missing key says "forgotten".
    # Both JS rows are a permission, not a prediction. Since #106 nothing a
    # *generated* project's gate runs can produce a lockfile - a worker reaches
    # no registry, React's toolchain comes from its image, and the generated
    # workflow uses `npm install` rather than `npm ci`. The entries stay for a
    # repository that brings its own installing gate.
    "node": ("package-lock.json",),
    "react": ("package-lock.json",),
}

#: What a task targets when its body does not say. Every issue written before
#: `## Stack` existed is a Python one, and `choose_stack` returns Python
#: unconditionally today, so this default is a description of the world rather
#: than a guess about it.
DEFAULT_STACK = "python"

# Labels → TaskStatus, straight from `docs/issue-contract.md` §3. Note the two
# collapses and the two counter-intuitive rows: `review` is `running` because
# completion is the merge, and `failed` is `abandoned` because v2 has no label
# for "this attempt failed, another is available" - that state is persisted as
# `swarm:ready` with a bumped counter and never sits in the ledger.
STATUS_BY_LABEL: Mapping[str, TaskStatus] = {
    "swarm:ready": "pending",
    "swarm:blocked": "pending",
    "swarm:claimed": "running",
    "swarm:review": "running",
    "swarm:done": "verified",
    "swarm:failed": "abandoned",
}

# Furthest-along-wins, for the two-state-labels fault. A human adding
# `swarm:done` or `swarm:failed` to a claimed issue means "stop", and stopping
# is the safe reading of an ambiguous ledger.
LABEL_PRECEDENCE = (
    "swarm:done",
    "swarm:failed",
    "swarm:review",
    "swarm:claimed",
    "swarm:blocked",
    "swarm:ready",
)

MAX_ID_LENGTH = 64

_TASK_ID_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_MARKER_RE = re.compile(r"^\s*<!--\s*apiary:task\b(?P<fields>[^>]*?)-->\s*$")
_MARKER_FIELD_RE = re.compile(r"(?P<key>[a-z_]+)=(?P<value>[^\s]+)")

# `^## (Goal|Files|Verify|Blocked by|Stack)[ \t]*$` - no leading whitespace,
# no trailing content. Anything looser re-opens the #11 trap.
_KNOWN_HEADING_RE = re.compile(rf"^## ({'|'.join(KNOWN_SECTIONS)})[ \t]*$")
# Any ATX heading ends the previous section, including one the contract does
# not know: an unrecognised `## Notes` must terminate `## Verify` rather than
# being swallowed into it.
_ATX_RE = re.compile(r"^ {0,3}#{1,6}(?:[ \t].*)?$")
_BREAK_RE = re.compile(r"^ {0,3}(?:(?:-[ \t]*){3,}|(?:\*[ \t]*){3,}|(?:_[ \t]*){3,})$")
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>.*)$")
_FENCE_CLOSE_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})[ \t]*$")
_LIST_ITEM_RE = re.compile(r"^ {0,3}(?:[-*+]|\d+[.)])[ \t]+(?P<text>.*)$")
_ISSUE_REF_RE = re.compile(r"(?P<repo>[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)?#(?P<number>\d+)")
_GLOB_CHARS = "*?[]{}"


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class LedgerError(RuntimeError):
    """Base for everything this module raises."""


class ContractError(LedgerError):
    """One issue body does not satisfy the contract.

    Carries the issue number, the offending section and the reason as fields
    rather than only in the message, because the orchestrator posts this back
    as a comment on that issue and needs to know which one.
    """

    def __init__(self, number: int, section: str, reason: str) -> None:
        self.number = number
        self.section = section
        self.reason = reason
        super().__init__(f"issue #{number}: [{section}] {reason}")


class DuplicateTaskIdError(LedgerError):
    """Two issues claim the same task id.

    Not a contract failure of either issue - it is control-plane corruption,
    and there is no safe way to pick a winner: dispatching both would put two
    containers on the same file set. The cycle aborts and names both numbers.
    """

    def __init__(self, task_id: str, first: int, second: int) -> None:
        self.task_id = task_id
        self.numbers = (first, second)
        super().__init__(f"task id {task_id!r} claimed by both issue #{first} and issue #{second}")


# --------------------------------------------------------------------------
# Parsed shapes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskContract:
    """What one issue body says, with nothing derived from labels or siblings.

    `task_id` is `None` when the body carries no marker - a hand-written issue,
    which the loader adopts. `blocked_by` holds `TaskRef`s: the body contains
    issue numbers, and this is the boundary at which they stop being numbers -
    the parse mints one ref per reference and nothing above the adapter reads
    them back (`swarm/taskref.py`). The translation to task ids needs the rest
    of the ledger and happens in `load_ledger`.
    """

    task_id: str | None
    attempt: int
    goal: str
    files: tuple[str, ...]
    verify: str
    blocked_by: tuple[TaskRef, ...]
    #: What the body's optional `## Stack` said, or `None` when it said
    #: nothing. `LedgerEntry.stack` is the resolved value; this one is the
    #: parse, and keeping them distinct is what makes "the body did not
    #: declare" answerable at all.
    stack: str | None = None
    #: The failure signature the last consumed attempt recorded, from the
    #: marker's optional `blocker=` field, or `""` when the marker carries
    #: none - which is every marker written before the field existed, and
    #: every attempt whose failure was never observed. Empty means "no
    #: previous blocker recorded", and the reconciler treats that exactly as
    #: it treated every failure before the field existed: same-failure
    #: arithmetic on the attempt counter.
    blocker: str = ""
    #: How many consecutive attempts have failed with `blocker`'s signature,
    #: from the marker's optional `streak=` field. `None` when the marker does
    #: not say - the reconciler then falls back to the attempt counter, which
    #: is what the streak *was* before failures had signatures, so an old
    #: marker behaves exactly as it always did.
    streak: int | None = None


@dataclass(frozen=True)
class LabelRepair:
    """An issue carrying more than one state label, and the winner.

    The loader only reports it. Removing the losing labels and commenting on
    the issue is a write, and writes belong to the reconciler (#22).
    """

    number: int
    kept: str
    removed: tuple[str, ...]

    @property
    def ref(self) -> TaskRef:
        """This repair's task, as the internal model names it."""
        return task_ref(self.number)

    def __str__(self) -> str:
        return (
            f"issue #{self.number}: {len(self.removed) + 1} state labels, kept {self.kept}, "
            f"drop {', '.join(self.removed)}"
        )


@dataclass(frozen=True)
class LedgerEntry:
    """One issue, fully resolved: contract, labels, identity and addressing.

    This is the lossless record. `TaskRecord` is the projection of it that v1's
    graph consumes, and every field the projection drops - the verify command,
    the issue number, the exact state label - stays here.
    """

    number: int
    title: str
    task_id: str
    attempt: int
    goal: str
    files: tuple[str, ...]
    verify: str
    blocked_by: tuple[TaskRef, ...]
    state_label: str
    labels: frozenset[str]
    #: The stack this task targets, resolved. `TaskContract.stack` is optional;
    #: this one is not, because every consumer downstream - the image #99
    #: chooses, the CI setup #96 emits - needs an answer rather than a maybe.
    stack: str = DEFAULT_STACK
    depends_on: tuple[str, ...] = ()
    adopted: bool = False
    #: The last consumed attempt's failure signature and the length of the
    #: consecutive run of it, straight from the marker's optional `blocker=`
    #: and `streak=` fields. See `TaskContract` for what absent means; the
    #: reconciler's `_retry_or_give_up` is the only consumer, and both fields
    #: exist so a *different* failure can be recognised as the previous
    #: blocker being gone.
    blocker: str = ""
    streak: int | None = None
    #: Whether the issue is closed on GitHub. The ledger reads `state="all"`
    #: because closed `swarm:done` issues anchor the dependency graph - which
    #: means a closed issue can still wear any state label, and a projection
    #: (the console board's failed strip is the live example) needs to tell a
    #: failed task that still wants a human from one that was already closed
    #: as superseded.
    closed: bool = False

    @property
    def ref(self) -> TaskRef:
        """This task's identity for the internal model - the graph, readiness,
        the reconciler's transitions. `number` stays alongside it because the
        GitHub API addresses issues by number and always will; the two are the
        same fact in the two vocabularies, and `github/refs.py` is the only
        module allowed to convert between them."""
        return task_ref(self.number)

    @property
    def generated(self) -> tuple[str, ...]:
        """Paths the gate may produce that this task may commit. See
        `GENERATED_FILES` for why they are not simply more `## Files`."""
        return generated_for(self.stack)

    @property
    def status(self) -> TaskStatus:
        return STATUS_BY_LABEL[self.state_label]

    @property
    def branch(self) -> str:
        """The branch this task's *current* attempt pushes (`github/branches.py`).

        Attempt-scoped, so it moves when the counter does. That is safe rather
        than fragile because of one rule in `reconcile._observe`: an exit 0
        moves no label and writes no counter, so a task sitting in
        `swarm:review` still carries the attempt whose pull request is open.
        The counter only moves on a path that has already decided the previous
        attempt is over - a consumed failure, or a pull request closed unmerged
        - and the next attempt is then supposed to get a name of its own rather
        than force-push over work somebody may still be reading.

        Consumers that have to survive a restart read the pair back out of the
        name instead of rebuilding it from an entry (`console_board.py`,
        `orchestrator/recovery.py`): after a crash there is a remote and no
        ledger, which is the whole point of putting it in the name.
        """
        return task_branch(self.ref, self.attempt)

    def to_task_record(self) -> TaskRecord:
        """Project onto v1's `TaskRecord`, which is all the graph ever sees."""
        return TaskRecord(
            id=self.task_id,
            goal=self.goal,
            files=list(self.files),
            depends_on=list(self.depends_on),
            status=self.status,
            attempts=self.attempt,
            branch=self.branch,
        )


@dataclass(frozen=True)
class Ledger:
    """Everything one read of the tracker found, including what it refused.

    `errors` is not an afterthought: §1.4's policy is that a malformed issue is
    labelled `swarm:failed` and the cycle *continues with the remaining
    issues*, so the loader has to be able to say "these parsed, that one did
    not" in one pass. What it must never do is return a partial record for the
    one that failed.
    """

    entries: dict[str, LedgerEntry] = field(default_factory=dict)
    errors: tuple[ContractError, ...] = ()
    repairs: tuple[LabelRepair, ...] = ()
    ignored: tuple[int, ...] = ()

    @property
    def tasks(self) -> dict[str, TaskRecord]:
        """The v1-shaped view: exactly what the planner used to build in memory."""
        return {task_id: entry.to_task_record() for task_id, entry in self.entries.items()}

    @property
    def by_ref(self) -> dict[TaskRef, str]:
        """Task ref → task id, for resolving what a `## Blocked by` ref names.

        Keyed on the ref rather than the number, like everything else the graph
        touches (#142). A caller holding an issue *number* - a pull request's
        `Closes #<n>`, say - has to mint one first (`github/refs.task_ref`)
        rather than indexing this with an int, which would miss in silence.
        """
        return {entry.ref: task_id for task_id, entry in self.entries.items()}


# --------------------------------------------------------------------------
# Body parsing
# --------------------------------------------------------------------------


def _scan(body: str) -> list[tuple[str, bool]]:
    """Split a body into `(line, fenced)` pairs.

    Fence delimiters count as fenced themselves, so no caller has to remember
    to skip them. Fence length and info strings are respected because a nested
    ```` ``` ```` inside a ```` ```` ```` block is content, not a terminator.
    """
    lines: list[tuple[str, bool]] = []
    fence: tuple[str, int] | None = None
    for line in body.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if fence is None:
            opened = _FENCE_OPEN_RE.match(line)
            # A backtick fence's info string may not contain backticks - that
            # rule is what stops an inline `` `code` `` line from opening one.
            if opened and not (opened["fence"][0] == "`" and "`" in opened["info"]):
                fence = (opened["fence"][0], len(opened["fence"]))
                lines.append((line, True))
                continue
            lines.append((line, False))
            continue
        closed = _FENCE_CLOSE_RE.match(line)
        if closed and closed["fence"][0] == fence[0] and len(closed["fence"]) >= fence[1]:
            fence = None
        lines.append((line, True))
    return lines


def _split_sections(
    number: int, scanned: Sequence[tuple[str, bool]]
) -> dict[str, list[tuple[str, bool]]]:
    """Locate the four known sections. Everything else is tolerated and dropped.

    Tolerating the untracked text is not laxity - the `> Design: …` blockquote
    above `## Goal` and the paragraphs of rationale below the thematic break are
    what let a human write a real issue instead of filling in a form.
    """
    sections: dict[str, list[tuple[str, bool]]] = {}
    current: str | None = None
    for line, fenced in scanned:
        if fenced:
            if current is not None:
                sections[current].append((line, fenced))
            continue
        known = _KNOWN_HEADING_RE.match(line)
        if known:
            name = known[1]
            if name in sections:
                # Two `## Verify` sections are two candidate commands with no
                # rule for choosing, so there is nothing to do but refuse.
                raise ContractError(number, name, "section appears more than once")
            sections[name] = []
            current = name
            continue
        if _ATX_RE.match(line) or _BREAK_RE.match(line):
            current = None
            continue
        if current is not None:
            sections[current].append((line, fenced))
    return sections


def _parse_marker(
    number: int, scanned: Sequence[tuple[str, bool]]
) -> tuple[str | None, int, str, int | None]:
    """Read `<!-- apiary:task id=… attempt=… -->`, ignoring fenced copies of it.

    A body quoting the canonical example - the contract doc's own §6, a planner
    explaining itself - must not have its example's identity adopted as its own.

    Returns `(task_id, attempt, blocker, streak)`. The last two are the
    optional failure-signature fields: a marker without them - every marker
    written before they existed - reads as `("", None)`, and an unknown field
    is ignored entirely, so a body written by a newer orchestrator parses
    cleanly under an older one and vice versa. The tolerance is not an
    accident: the worker reads this marker too (`worker.entrypoint`), and a
    field only the reconciler consumes must never fail a container over it.
    """
    for line, fenced in scanned:
        if fenced:
            continue
        marker = _MARKER_RE.match(line)
        if not marker:
            continue
        fields = {m["key"]: m["value"] for m in _MARKER_FIELD_RE.finditer(marker["fields"])}
        task_id = fields.get("id")
        if task_id is None:
            raise ContractError(number, "marker", "apiary:task marker has no id= field")
        if len(task_id) > MAX_ID_LENGTH or not _TASK_ID_RE.match(task_id):
            raise ContractError(number, "marker", f"task id {task_id!r} is not a kebab-case slug")
        # A missing or unparseable counter reads as 0: that is the hand-written
        # issue the loader just adopted, which has made no attempts.
        try:
            attempt = max(0, int(fields.get("attempt", "0")))
        except ValueError:
            attempt = 0
        # Same tolerance for the streak: unparseable reads as absent, and
        # absent falls back to the attempt counter downstream, which is the
        # pre-signature arithmetic exactly.
        streak: int | None
        try:
            streak = max(0, int(fields["streak"])) if "streak" in fields else None
        except ValueError:
            streak = None
        return task_id, attempt, fields.get("blocker", ""), streak
    return None, 0, "", None


def _parse_goal(number: int, lines: Sequence[tuple[str, bool]]) -> str:
    """The goal as written, line structure preserved.

    This used to join every line with a space and collapse runs of whitespace,
    which made a goal exactly one line however it was written. That was fine
    while the planner also wrote one line, and wrong once the goal became the
    worker's whole brief: a specification reads as a paragraph or a short list,
    and flattening it to a single line is the difference between instructions
    and a run-on sentence.

    A body written before this change parses identically - one line joins to
    itself - so an existing issue reads the same either way.
    """
    kept = [line.strip() for line, _ in lines if line.strip()]
    goal = "\n".join(kept)
    if not goal:
        raise ContractError(number, "Goal", "section is empty")
    return goal


def _parse_files(number: int, lines: Sequence[tuple[str, bool]]) -> tuple[str, ...]:
    """One repo-relative path per list item; globs and escapes are refused.

    A glob is rejected rather than expanded because the dispatcher (#21)
    decides concurrency by intersecting these sets, and a glob has no set
    semantics without a filesystem to resolve it against. The wrong place to
    discover that two tasks overlap is inside a running container.
    """
    paths: list[str] = []
    for line, fenced in lines:
        if fenced:
            continue
        item = _LIST_ITEM_RE.match(line)
        if not item:
            continue
        path = item["text"].strip().strip("`").strip()
        while path.startswith("./"):
            path = path[2:]
        if not path:
            continue
        if path.startswith("/"):
            raise ContractError(number, "Files", f"path {path!r} is absolute")
        if ".." in path:
            raise ContractError(number, "Files", f"path {path!r} escapes the repository")
        if any(char in path for char in _GLOB_CHARS):
            raise ContractError(number, "Files", f"path {path!r} contains a glob metacharacter")
        paths.append(path)
    if not paths:
        raise ContractError(number, "Files", "section lists no files")
    return tuple(paths)


def _parse_verify(number: int, lines: Sequence[tuple[str, bool]]) -> str:
    """One shell command, bare or fenced.

    Multi-line is refused because "run these in order, stopping on failure" is
    a semantics nobody agreed to - write `&&`. As in v1's `verifier.py`, only
    the exit code of whatever this returns is ever believed.
    """
    commands = [
        line.strip()
        for line, _ in lines
        if line.strip() and not _FENCE_OPEN_RE.match(line)
    ]
    if not commands:
        raise ContractError(number, "Verify", "section is empty")
    if len(commands) > 1:
        raise ContractError(
            number, "Verify", f"expected one command, found {len(commands)}: {commands!r}"
        )
    return commands[0]


def generated_for(stack: str) -> tuple[str, ...]:
    """The generated set for one stack. Unknown stacks generate nothing.

    Empty rather than an error: an unknown stack is already a `ContractError`
    at parse time, and this is also called on a `LedgerEntry` whose stack was
    defaulted, where "nothing is generated" is the correct and safe answer.
    """
    return GENERATED_FILES.get((stack or DEFAULT_STACK).casefold(), ())


def _parse_stack(number: int, lines: Sequence[tuple[str, bool]] | None) -> str | None:
    """The optional `## Stack`: one known id, or `None` when there is no section.

    An unknown id is a `ContractError` and never a silent default. That is the
    whole reason for a closed vocabulary: `## Stack` `rust` in a repository
    with no Rust image is a mistake that should be reported against the issue
    that made it, at parse time, rather than becoming a Python container that
    fails its gate three times.

    A *present but empty* section is also an error, for the same reason it is
    one under `## Verify` - somebody wrote the heading meaning to say
    something, and reading it as "no opinion" discards their intent silently.
    """
    if lines is None:
        return None
    names = [
        line.strip()
        for line, _ in lines
        if line.strip() and not _FENCE_OPEN_RE.match(line)
    ]
    if not names:
        raise ContractError(number, "Stack", "section is empty")
    if len(names) > 1:
        raise ContractError(number, "Stack", f"expected one stack, found {names!r}")
    stack = names[0].strip("`").casefold()
    if stack not in KNOWN_STACKS:
        raise ContractError(
            number,
            "Stack",
            f"unknown stack {stack!r}; known: {', '.join(sorted(KNOWN_STACKS))}",
        )
    return stack


def _refuse_generated_files(number: int, files: Sequence[str], stack: str | None) -> None:
    """The two sets are disjoint, and overlap is a `ContractError`.

    A model that lists `package-lock.json` under `## Files` has declared it will
    author a file it cannot author - 180k tokens of SHA-512 hashes - and the
    attempt would burn on a truncated generation rather than on the task. It is
    also the one way the generated set could be widened by model output, which
    is exactly what `commit_edits` staging only the declared paths exists to
    prevent.
    """
    generated = set(generated_for(stack or DEFAULT_STACK))
    overlap = sorted(path for path in files if path.casefold() in generated)
    if overlap:
        raise ContractError(
            number,
            "Files",
            f"{', '.join(overlap)} is generated by the verify command, not written by a "
            "task; remove it from ## Files and it will be committed if it appears",
        )


def _parse_blocked_by(number: int, lines: Sequence[tuple[str, bool]]) -> tuple[TaskRef, ...]:
    """Task refs from `- #N` items. No items at all means no dependencies.

    This is where the wire format stops. The section still says `#N` - nothing
    about the contract changes - and every number it names is minted into a
    `TaskRef` here, so the graph downstream never sees an integer.

    A `#N` on a non-list line is malformed rather than ignored, because that is
    exactly the shape of a dependency that gets silently dropped - and a task
    that runs before its prerequisite is the failure this whole contract
    exists to prevent.
    """
    refs: list[TaskRef] = []
    for line, fenced in lines:
        if fenced or not line.strip():
            continue
        item = _LIST_ITEM_RE.match(line)
        text = item["text"] if item else line
        for ref in _ISSUE_REF_RE.finditer(text):
            if not item:
                raise ContractError(
                    number, "Blocked by", f"reference #{ref['number']} is not in a list item"
                )
            if ref["repo"]:
                # One task, one repo: a cross-repo ref is a dependency this
                # system cannot observe, so treating it as satisfied is a lie.
                raise ContractError(
                    number,
                    "Blocked by",
                    f"cross-repository reference {ref['repo']}#{ref['number']}",
                )
            refs.append(task_ref(int(ref["number"])))
    return tuple(dict.fromkeys(refs))


def parse_contract(number: int, body: str | None) -> TaskContract:
    """Parse one issue body, or raise `ContractError`.

    Never returns a partial record. No defaulting a missing `## Verify` to the
    repo-wide command, no reading an unparseable `## Blocked by` as empty: a
    silently mis-parsed contract produces a task that runs, passes the wrong
    gate and merges.
    """
    scanned = _scan(body or "")
    sections = _split_sections(number, scanned)
    for name in REQUIRED_SECTIONS:
        if name not in sections:
            raise ContractError(number, name, "section is missing")
    task_id, attempt, blocker, streak = _parse_marker(number, scanned)
    stack = _parse_stack(number, sections.get("Stack"))
    files = _parse_files(number, sections["Files"])
    _refuse_generated_files(number, files, stack)
    return TaskContract(
        task_id=task_id,
        attempt=attempt,
        blocker=blocker,
        streak=streak,
        goal=_parse_goal(number, sections["Goal"]),
        files=files,
        verify=_parse_verify(number, sections["Verify"]),
        blocked_by=_parse_blocked_by(number, sections["Blocked by"]),
        # Absent is not the same as `python`: `None` records that the body did
        # not say, which is what lets `## Stack` be added to an issue later
        # without the loader having to guess whether it was always there.
        stack=stack,
    )


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------


def render_marker(
    task_id: str, attempt: int = 0, *, blocker: str = "", streak: int | None = None
) -> str:
    """The identity line, in the one form every writer must emit.

    `blocker` and `streak` are the optional failure-signature fields, and they
    are emitted only when set: a caller that does not pass them - which is
    every caller that predates them, and every writer that consumed an attempt
    through a channel where the signature has no meaning (a stale claim, a
    failed check run) - produces byte-for-byte the marker it always did, so an
    old body round-trips unchanged and dropping the record deliberately falls
    back to the pre-signature arithmetic downstream.
    """
    fields = [f"id={task_id}", f"attempt={attempt}"]
    if blocker:
        fields.append(f"blocker={blocker}")
    if streak is not None:
        fields.append(f"streak={streak}")
    return f"<!-- apiary:task {' '.join(fields)} -->"


def slugify(title: str, *, limit: int = MAX_ID_LENGTH) -> str:
    """Derive a task id from an issue title, in `PlannedTask.id`'s shape."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if len(slug) > limit:
        slug = slug[:limit].rstrip("-")
    return slug


def _adopted_id(number: int, title: str, taken: Iterable[str]) -> str:
    """An id for a hand-written issue, unique against the ids already in use.

    Deterministic on purpose: if the write that persists the marker fails, the
    next cycle derives the same id from the same title rather than inventing a
    second one for the same work.
    """
    taken = set(taken)
    slug = slugify(title) or f"issue-{number}"
    if slug not in taken:
        return slug
    suffix = f"-{number}"
    return f"{slug[:MAX_ID_LENGTH - len(suffix)].rstrip('-')}{suffix}"


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def _label_names(issue: Mapping[str, Any]) -> frozenset[str]:
    """GitHub returns label objects; some fixtures and webhooks return strings."""
    names = []
    for label in issue.get("labels") or ():
        names.append(label.get("name", "") if isinstance(label, Mapping) else str(label))
    return frozenset(name for name in names if name)


def resolve_state_label(
    number: int, labels: Iterable[str]
) -> tuple[str | None, LabelRepair | None]:
    """Pick the one state label, repairing zero-or-two by precedence.

    Zero is not "ready" - an issue with no state label is outside the ledger,
    and defaulting it into the ledger would dispatch work nobody scheduled.
    """
    carried = set(labels)
    present = [label for label in LABEL_PRECEDENCE if label in carried]
    if not present:
        return None, None
    if len(present) == 1:
        return present[0], None
    return present[0], LabelRepair(number, present[0], tuple(present[1:]))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _as_client(source: GitHubClient | str) -> GitHubClient:
    """A repo name builds a client from the environment; anything else is one.

    Duck-typed rather than `isinstance`-checked so a test double only has to
    provide `list_issues` and `update_issue`, which is the whole surface the
    loader touches.
    """
    return GitHubClient.from_env(source) if isinstance(source, str) else source


def _adopt(client: GitHubClient, issue: Mapping[str, Any], task_id: str) -> None:
    """Persist a marker on a human's issue, preserving every byte below it.

    Prepended rather than appended so it stays above the prose a human will
    keep editing - §2's "survives … body edits below it". One invisible line is
    the entire cost of making a hand-written backlog a real ledger.
    """
    body = issue.get("body") or ""
    marker = render_marker(task_id, 0)
    client.update_issue(int(issue["number"]), body=f"{marker}\n\n{body}" if body else marker)


def load_ledger(
    source: GitHubClient | str,
    *,
    state: str = "all",
    adopt: bool = True,
) -> Ledger:
    """Read the tracker and build the ledger.

    `state="all"` by default: a `swarm:done` issue is closed, and a ledger that
    forgot its finished work would report a run as unfinished forever and
    lose the dependency edges pointing at it.

    Malformed issues land in `Ledger.errors` rather than aborting the read, so
    one bad hand-written issue cannot stop a cycle - but they are never
    dispatched either, because they are not in `entries`. Duplicate ids *do*
    abort: there is no safe reading of that one.
    """
    client = _as_client(source)
    issues = sorted(client.list_issues(state=state), key=lambda issue: int(issue["number"]))

    entries: dict[str, LedgerEntry] = {}
    numbers: dict[str, int] = {}
    errors: list[ContractError] = []
    repairs: list[LabelRepair] = []
    ignored: list[int] = []
    pending_adoption: list[tuple[Mapping[str, Any], LedgerEntry]] = []

    for issue in issues:
        number = int(issue["number"])
        labels = _label_names(issue)
        state_label, repair = resolve_state_label(number, labels)
        if state_label is None:
            # Humans use the tracker too, and this repository's own backlog is
            # the live example: no state label means not part of the ledger.
            ignored.append(number)
            continue
        if repair is not None:
            repairs.append(repair)

        title = issue.get("title") or ""
        try:
            contract = parse_contract(number, issue.get("body"))
        except ContractError as exc:
            errors.append(exc)
            continue

        adopted = contract.task_id is None
        task_id = contract.task_id or _adopted_id(number, title, entries.keys())
        if task_id in entries:
            raise DuplicateTaskIdError(task_id, numbers[task_id], number)

        entry = LedgerEntry(
            number=number,
            title=title,
            task_id=task_id,
            attempt=contract.attempt,
            blocker=contract.blocker,
            streak=contract.streak,
            goal=contract.goal,
            files=contract.files,
            verify=contract.verify,
            blocked_by=contract.blocked_by,
            # Resolved here, so nothing downstream has to remember that `None`
            # means Python. The contract keeps the unresolved answer.
            stack=contract.stack or DEFAULT_STACK,
            state_label=state_label,
            labels=labels,
            adopted=adopted,
            closed=(issue.get("state") or "open") != "open",
        )
        entries[task_id] = entry
        numbers[task_id] = number
        if adopted:
            pending_adoption.append((issue, entry))

    # depends_on is a task-id graph, so a ref to an issue outside the ledger has
    # no id to name and is dropped from the projection. The refs themselves
    # survive verbatim on `LedgerEntry.blocked_by`, which is what readiness
    # (#11) reads - it resolves each ref's issue state, and "closed" is a fact
    # the projection cannot represent.
    by_ref = {entry.ref: task_id for task_id, entry in entries.items()}
    resolved = {
        task_id: replace(
            entry,
            depends_on=tuple(by_ref[ref] for ref in entry.blocked_by if ref in by_ref),
        )
        for task_id, entry in entries.items()
    }

    if adopt:
        for issue, entry in pending_adoption:
            _adopt(client, issue, entry.task_id)

    return Ledger(
        entries=resolved,
        errors=tuple(errors),
        repairs=tuple(repairs),
        ignored=tuple(ignored),
    )


def load_tasks(
    repo: GitHubClient | str,
    *,
    state: str = "all",
    adopt: bool = True,
    strict: bool = True,
) -> dict[str, TaskRecord]:
    """The v1-shaped read: issues in, the task ledger the graph expects out.

    `strict=True` re-raises the first `ContractError` the read collected, which
    is the right default for anything that is not the reconcile loop: a caller
    that only wanted `TaskRecord`s has nowhere to put the errors, and dropping
    them silently would make a mis-written issue disappear from the ledger with
    nobody told. The orchestrator passes `strict=False` and applies §1.4's
    policy from `Ledger.errors` itself.
    """
    ledger = load_ledger(repo, state=state, adopt=adopt)
    if strict and ledger.errors:
        raise ledger.errors[0]
    return ledger.tasks
