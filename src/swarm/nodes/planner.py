"""Planner node - the write half of the ledger.

`docs/architecture-v2.md` gives this module one line in its survival table:
"kept, now writes issues". So the plan stops being a dict that dies with the
process and becomes issues on GitHub, which `ledger.load_ledger` (#9) reads
back. This module renders `PlannedTask`s into the body schema of
`docs/issue-contract.md` §1, creates or updates one issue each, and reports
what it did.

**Round-trip is the acceptance criterion, so it is checked before every
write.** Whatever is written here, the loader must read back identically - and
the loader refuses a malformed body outright (§1.4), which would turn a
model's stray glob into an issue nobody can dispatch. Every rendered body is
therefore run through `parse_contract` *before* it is sent, and a task whose
body does not parse is rejected and reported rather than written. The parser
is the only definition of what an issue means; re-implementing half of it here
to pre-validate would create a second one.

**Replanning matches on the marker id, never on the issue number** (§2). A
replan re-invokes the model and gets fresh output; the id is what says "this
is the same work". The number cannot be the key - the model emits kebab-case
slugs and can never emit an integer, so matching on the number would find
nothing every time and fork a complete second set of issues on every stall.
An id that survives is updated in place; the marker itself is never rewritten,
so the identity *and* the attempt counter (§5) both survive a replan.

**What a replan does to a task it dropped** depends entirely on how far that
task got, because the planner owns exactly two rows of the §4 state machine -
creating an issue `ready` and creating one `blocked` - and nothing else:

- `swarm:ready` / `swarm:blocked`, still open: closed as `not_planned`. Nothing
  was built, and leaving it open would have the dispatcher run work the current
  plan does not contain. `not_planned` rather than `completed` is load-bearing:
  readiness (#11) treats only `completed` as satisfying, so a task waiting on a
  retired one stays blocked instead of being unblocked by a cancellation.
- **`swarm:claimed` / `swarm:review` - a live container, or an open PR: left
  exactly as it is, and reported.** This is the case the ticket asks about. The
  PR says `Closes #N`; closing that issue as not-planned strands work that may
  be one merge away from landing, and relabelling it would yank an issue out
  from under a running container. Whether to abandon in-flight work is a
  decision for the reconciler (#22, #23) or a human, and the planner declines it
  loudly instead of making it silently.
- `swarm:done` / `swarm:failed`: terminal. `done` is history; `failed` needs a
  human. Both untouched when *dropped*.

The same reasoning governs a task the replan *keeps*: an entry that is claimed,
in review or done is not rewritten either, because its body is the contract a
container is working against right now (or history, for `done`).

**A `swarm:failed` task the plan keeps is the one exception: it is revived.**
Observed live: a replan fired on a stalled run, reported "0 created, 3 updated,
0 retired, 11 left alone" - and the failed task whose dependency chain blocked
everything was among the ones left alone, so the run was still 0-ready and
re-stalled until a human relabelled the issue by hand. A replan that can
rewrite the whole tracker but cannot revive the one task blocking it is a
replan that cannot fix a stall. Reviving used to be unsafe - a fresh dispatch
against a spent counter would fail once and re-fail the task, or worse, a
human-style counter reset would grant three more rolls at the identical
blocker. The failure signature (`docs/issue-contract.md` §5) is what makes it
safe now, so the revival deliberately resets **nothing**: the marker keeps its
`attempt`, `blocker` and `streak`, the label moves `failed -> ready`, and the
arithmetic at the next failure is the guard - the same failure again gives up
immediately (the streak is already at its cap), a *different* failure
legitimately renews. A failed task whose hard total budget is already spent
(`attempt >= SWARM_MAX_TOTAL_ATTEMPTS`) stays failed and is only reported: the
give-up comment on the issue already told a human what to do, and re-saying it
every replan would bury it. The body is not rewritten either way - rendering it
fresh would drop the marker's signature record, and the retry-feedback comments
already carry what the next attempt needs.

The revival lives on this write path rather than in `replan.py`, so **any**
caller of `write_plan` whose plan retains a failed task revives it - the goal
gate included. In practice the gate never exercises it: `goal.assess` refuses
to act while any task is `swarm:failed` (an abandoned task is a question for a
human, not for more planning), so the stall-replan is the path that revives.

**State labels are written once, at creation.** After that `ready` <-> `blocked`
belongs to readiness (#11), which resolves each `## Blocked by` ref's real state
instead of guessing from a ledger read. The label this module picks at creation
is the best answer available before the issue exists, and readiness corrects it
on the next pass. Routing labels (`area/*`, `size/*`) are written at creation
too and never re-asserted: they are hints for humans that the state machine
never reads (§3), and re-applying them every replan would overrule every human
who re-triaged an issue.

**There is deliberately no `dry_run`**, where `apply_readiness` has one. A
task's `## Blocked by` refs are the issue numbers GitHub assigns at creation,
so a run that writes nothing cannot render the bodies it would have written,
and a report showing them would be showing numbers that will never exist.

`plan_node` still answers with `TaskRecord`s because the graph consumes those,
but on the GitHub path it gets them by *re-reading the ledger* rather than by
projecting the plan it just sent. Anything else would be a second ledger, and
`docs/architecture-v2.md` is explicit that on disagreement GitHub wins.
"""

from __future__ import annotations

import re
import sys
import time

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

from ..config import SETTINGS
from ..github.client import GitHubClient, GitHubError
from ..github.ledger import (
    DEFAULT_STACK,
    KNOWN_STACKS,
    ContractError,
    Ledger,
    LedgerEntry,
    load_ledger,
    parse_contract,
    render_marker,
    slugify,
)
from ..github.readiness import BLOCKED, READY, IssueState, resolve_states
from ..llm import orchestrator_llm, structured
from ..state import Plan, PlannedTask, SwarmState, TaskRecord

#: The planning prompt, minus the two facts that vary per run.
#:
#: What was here before was a list of five constraints and no craft: never share
#: a file, prefer 2-4 tasks, edit only what you list, use depends_on sparingly,
#: kebab-case the ids. Every line restricted the shape of the answer and not one
#: said what a *good* decomposition is. Two consequences, both observed:
#:
#: The gate went unmentioned. `$SWARM_VERIFY`'s exit code is the only authority
#: on whether a task is done - the repository's own invariant - and the prompt
#: that creates tasks never said so. Asked for a trip planner "platform", the
#: planner emitted `client/src/pages/Login.tsx` against a Python repository gated
#: by `python3 -m unittest discover -q`: three tasks, 123 seconds, none of which
#: could ever go green. The command was already in `plan_node`'s hand at the
#: time, being written into each issue's `## Verify` and printed to the terminal
#: one line above the call.
#:
#: And file-disjointness became the cutting principle by default, because it was
#: the only rule about shape. That is a constraint of the worktree model, not a
#: theory of decomposition, and following it produces layers: the first recorded
#: run split `trip-planner-implementation` from `trip-planner-tests`, and the
#: tests half - which cannot pass without the half it was severed from - burned
#: all three attempts and stalled the run.
#:
#: So the gate leads, the slice rule is stated with the example that failed, and
#: the constraints stay constraints.
SYSTEM_RULES = """You decompose a software objective into independent coding tasks.

Each task becomes one GitHub issue and is given to one worker in its own
checkout. A task is DONE when, and only when, this command exits 0:

    {verify}

Nothing else decides. No human reviews the code, and no model is asked whether
it looks right.

So every task must be a slice that command can see:
- A task must leave the project passing that command BY ITSELF, without the
  other tasks. If task B only passes once task A has landed, they are one task.
- Prefer a thin end-to-end slice over a layer. "Store and list trips, with the
  test that proves it" is one task. "Add the data model" and "add the tests"
  are two halves of one, and the half holding the tests cannot pass alone.
- If part of the objective is something that command cannot exercise, say so in
  reasoning and plan only the part it can.

Constraints:
- Two tasks must NEVER list the same file. If they would, merge them into one task.
- Each task must be completable by editing only the files it lists.
- Every path must be plausible for {stack}. Do not invent a stack the project
  does not use.
- Use depends_on only when a task literally cannot start before another finishes.
- There is no target number of tasks, no minimum and no maximum. The count is
  whatever falls out of the rule below.
- One task is ONE behaviour: a single thing the tests can assert, described
  without using "and". If a task's requirements read as a list of independent
  assertions, that is a list of tasks. Creating a record, editing it, listing
  it and searching it are four behaviours, not one "management" task.
- A task is worked by ONE model call that writes every one of its files in
  full, in a single answer. If you would not expect one answer to contain all
  of that code, the task is too big and does not come back half-done - it comes
  back truncated mid-file. Split until each task fits in one answer.
- Every task must list the files it creates or modifies. A worker may only
  write what its task declares, so a task that declares nothing can do nothing.
- Third-party packages reach a worker ONLY if declared in requirements.txt,
  which is installed before {verify} runs (when the operator has allowed the
  package index). Prefer the standard library when it suffices.

Each goal is the entire brief the worker gets. It sees the goal, the files, and
nothing else - not this objective, not the other tasks, not your reasoning. So
write the goal as a real specification, several sentences or a short list:
what to build, the names and signatures that matter, the behaviour at the edges,
and what the test must assert for {verify} to pass. A one-line goal is a worker
guessing.
- Ids are lowercase kebab-case and describe the work, not the order; an id is
  how the issue is recognised again later.

Return JSON only."""


def system_prompt(*, verify: str | None = None, stack: str | None = None) -> str:
    """The planning system prompt, grounded in this run's gate and stack.

    Both arguments are things `plan_node` has already been given and used to
    pass straight through to the writer. Putting them in front of the model
    costs nothing and is the difference between a plan whose tasks can pass and
    one whose tasks cannot.
    """
    # The same collapse `_one_line` does, inlined: that helper is defined far
    # below, and `SYSTEM` is built at import time.
    command = " ".join((verify or SETTINGS.verify_command or "").split())
    return SYSTEM_RULES.format(
        verify=command,
        stack=f"a {stack} project" if stack else "the project's existing stack",
    )


#: The ungrounded default, for the two callers that have no run to ground it in.
#: `orchestrator/goal.py` and `orchestrator/replan.py` import this to build their
#: follow-up and replan prompts, and neither is handed a verify command; the
#: configured default is a better answer than silence about the gate.
SYSTEM = system_prompt()

REPLAN_SUFFIX = """

This is a REPLAN. The previous attempt stalled. Here is what happened:
{failures}

These tasks are already on the tracker:
{existing}

Re-emit a task under its EXACT existing id when it is the same work: the id is
how the tracker recognises it, and a new id for old work opens a second issue
for something that already has one. Use a new id only for genuinely new work,
and simply omit a task you are dropping.

Produce a different decomposition. Do not repeat the approach that failed."""

FOLLOWUP_SUFFIX = """

This is a FOLLOW-UP round. Every task already planned has landed on the default
branch, and the objective is still not met. Here is what shipped:
{shipped}

Here is what is judged to be missing:
{missing}

Emit ONLY the additional tasks needed to close that gap, under new ids. Do not
re-emit a task from the shipped list: it is done, its issue is closed, and
naming it again cannot reopen it. If the gap needs nothing further, emit no
tasks at all rather than inventing work.

The shipped files are on the branch you are extending, so a follow-up task may
list a file an earlier task created - the hard rule about non-overlapping files
applies among the tasks you emit now, not against work that is already in."""

# `## Blocked by` with no list items parses to no dependencies (§1.3). Written
# out rather than left blank so a human reading the issue sees an answer.
NO_DEPENDENCIES = "_none._"

# The only two state labels whose issues this module may rewrite. Everything
# else is another component's row of §4, and an issue in one of them is either
# being worked on right now or finished.
WRITABLE = frozenset({READY, BLOCKED})

# In-flight: a container holds it, or a PR is open against it. Dropping one of
# these is a decision the planner refuses to make - see the module docstring.
IN_FLIGHT = frozenset({"swarm:claimed", "swarm:review"})

# The label a revival takes an issue out of. The same spelling as
# `reconcile.FAILED`; spelled here rather than imported because `orchestrator`
# modules import this one, and the dependency must not point back up.
FAILED = "swarm:failed"

# Path segments that are packaging, not subject matter. `src/swarm/github/…` is
# work on the github area; the `src` says nothing about what the task is.
AREA_CONTAINERS = frozenset(
    {"src", "lib", "app", "pkg", "internal", "cmd", "tests", "test", "spec", "docs"}
)

# Size is the model's sense of scope, and the only quantity a file list actually
# carries is how many files there are. A proxy, labelled as one.
SIZE_SMALL_MAX = 2
SIZE_MEDIUM_MAX = 5

# GitHub truncates nothing, but a title is read in a list view.
MAX_TITLE = 72

# The number `parse_contract` reports a self-check failure under. It never
# reaches a human: the section and the reason are lifted off the error and the
# number is dropped, because the issue does not exist yet.
_SELF_CHECK_NUMBER = 0

ActionKind = Literal[
    "created", "updated", "unchanged", "retired", "retained", "rejected", "revived"
]


class PlanError(RuntimeError):
    """A plan cannot be written as it stands, and nothing was written."""


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_body(
    task_id: str,
    *,
    goal: str,
    files: Sequence[str],
    verify: str,
    blocked_by: Sequence[int] = (),
    attempt: int = 0,
    stack: str | None = None,
) -> str:
    """One issue body in the form of `docs/issue-contract.md` §6.

    The marker goes first, above everything, so it stays above the prose a
    human will keep editing (§2). Sections are emitted in the documented order
    even though the parser does not care, because the audience for the order is
    the person reading the issue.

    **`## Stack` is emitted last**, and the compatibility property that makes
    this section safe is stronger than the plan assumed. `_split_sections`
    treats any unrecognised ATX heading as a section terminator, and any later
    *known* heading opens a new section - so a pre-change parser reads all four
    required sections identically **wherever** `## Stack` sits, not only when
    it is last. `tests/test_ledger.py` pins that for three placements, because
    it is the one thing making this deployable against a repository whose
    issues were written by another version.

    Last is still the right position, for the ordinary reason: it keeps the
    four canonical sections contiguous and in the order §6 documents, so the
    optional one reads as an appendix to a human rather than as an interruption
    of the contract.

    Omitted entirely when `stack` is None, so a Python plan's bodies are
    byte-for-byte what they were.
    """
    lines = [
        render_marker(task_id, attempt),
        "",
        "## Goal",
        goal,
        "",
        "## Files",
        *(f"- {path}" for path in files),
        "",
        "## Verify",
        verify,
        "",
        "## Blocked by",
        *([f"- #{number}" for number in blocked_by] or [NO_DEPENDENCIES]),
    ]
    if stack:
        lines += ["", "## Stack", stack]
    return "\n".join(lines) + "\n"


def issue_title(goal: str, task_id: str) -> str:
    """A human-legible title from the goal, falling back to the id.

    Titles are not identity (§2 chose the marker precisely so a human may
    retitle freely), which is why this only ever runs at creation.
    """
    sentence = goal.split(". ")[0].strip().rstrip(".")
    if len(sentence) > MAX_TITLE:
        head = sentence[:MAX_TITLE].rsplit(" ", 1)[0]
        sentence = f"{head or sentence[:MAX_TITLE]}…"
    return sentence or task_id


def size_label(files: Sequence[str]) -> str:
    """`size/S|M|L` from the file count - a proxy, and the only one available."""
    if len(files) <= SIZE_SMALL_MAX:
        return "size/S"
    return "size/M" if len(files) <= SIZE_MEDIUM_MAX else "size/L"


def area_label(files: Sequence[str]) -> str | None:
    """`area/<dir>` when every file agrees on one, else nothing.

    The directory the work lives in is the closest thing to an area a file list
    can honestly yield. A task spanning two of them gets no area label rather
    than an invented one: the state machine never reads these (§3), so a
    missing hint costs nothing and a wrong one misroutes a human.
    """
    areas = set()
    for path in files:
        parts = [part for part in path.split("/") if part]
        while parts and parts[0].casefold() in AREA_CONTAINERS:
            parts.pop(0)
        # `len(parts) < 2` means the remainder is the file itself, so there is
        # no directory left to name - `tests/test_planner_issues.py` says only
        # that a task has tests, which is true of every task.
        if len(parts) >= 2 and (area := slugify(parts[-2])):
            areas.add(area)
    return f"area/{areas.pop()}" if len(areas) == 1 else None


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IssueAction:
    """What happened to one task, and why when the answer is "nothing"."""

    kind: ActionKind
    task_id: str
    number: int | None = None
    reason: str = ""

    def __str__(self) -> str:
        where = f"#{self.number} " if self.number is not None else ""
        because = f" - {self.reason}" if self.reason else ""
        return f"{where}{self.task_id}: {self.kind}{because}"


@dataclass(frozen=True)
class PlanReport:
    """Everything one `write_plan` did, in the shape a caller can print.

    `warnings` is separate from the actions because a dropped dependency is a
    fact about an issue that was still written - folding it into the action
    would make a successful create look like a failure, and omitting it would
    lose the one line explaining why a task came out unblocked.
    """

    repo: str
    actions: tuple[IssueAction, ...] = ()
    warnings: tuple[str, ...] = ()

    def of(self, *kinds: ActionKind) -> tuple[IssueAction, ...]:
        return tuple(action for action in self.actions if action.kind in kinds)

    @property
    def created(self) -> tuple[IssueAction, ...]:
        return self.of("created")

    @property
    def updated(self) -> tuple[IssueAction, ...]:
        return self.of("updated")

    @property
    def unchanged(self) -> tuple[IssueAction, ...]:
        return self.of("unchanged")

    @property
    def retired(self) -> tuple[IssueAction, ...]:
        return self.of("retired")

    @property
    def retained(self) -> tuple[IssueAction, ...]:
        return self.of("retained")

    @property
    def rejected(self) -> tuple[IssueAction, ...]:
        return self.of("rejected")

    @property
    def revived(self) -> tuple[IssueAction, ...]:
        """Failed tasks the plan kept and therefore returned to `swarm:ready`.

        Reported as their own kind rather than folded into `updated`, because
        the operator watching a stalled run needs the difference: an update
        rewrote a contract, a revival un-stuck the chain a human used to have
        to relabel by hand. See the module docstring's revival rule.
        """
        return self.of("revived")

    @property
    def numbers(self) -> tuple[int, ...]:
        """Issues this plan is now made of, ascending."""
        return tuple(
            sorted(
                action.number
                for action in self.of("created", "updated", "unchanged", "revived")
                if action.number is not None
            )
        )

    @property
    def changed(self) -> bool:
        return bool(self.created or self.updated or self.retired or self.revived)

    def summary(self) -> str:
        head = (
            f"{self.repo}: {len(self.created)} created, {len(self.updated)} updated, "
            f"{len(self.unchanged)} unchanged, {len(self.retired)} retired, "
            f"{len(self.retained)} left alone, {len(self.rejected)} rejected"
        )
        # Appended rather than always present so every existing reading of the
        # summary line stays byte-identical on the plans that revive nothing.
        return f"{head}, {len(self.revived)} revived" if self.revived else head


# --------------------------------------------------------------------------
# Drafts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Draft:
    """One `PlannedTask`, normalised into the shapes the contract accepts.

    Normalising here rather than trusting the model is what keeps the parser's
    verdict predictable: ids go through `slugify` so they match §2's shape,
    and prose is collapsed onto one line so a goal can never contain something
    that looks like a section heading at the start of a line.
    """

    task_id: str
    goal: str
    files: tuple[str, ...]
    verify: str
    depends_on: tuple[str, ...] = ()
    #: `None` means "did not say", which renders no `## Stack` section at all
    #: and reads back as the default. Distinct from the string "python", which
    #: renders the section and says so out loud.
    stack: str | None = None

    @property
    def labels(self) -> tuple[str, ...]:
        """Routing labels, written at creation and never re-asserted."""
        area = area_label(self.files)
        return (size_label(self.files),) if area is None else (area, size_label(self.files))

    @property
    def title(self) -> str:
        return issue_title(self.goal, self.task_id)

    def body(self, *, blocked_by: Sequence[int] = (), attempt: int = 0) -> str:
        return render_body(
            self.task_id,
            goal=self.goal,
            files=self.files,
            verify=self.verify,
            blocked_by=blocked_by,
            attempt=attempt,
            stack=self.stack,
        )

    def matches(self, entry: LedgerEntry, blocked_by: Sequence[int]) -> bool:
        """True when the issue already says exactly this.

        Compared through the *parsed* fields rather than by rendering the old
        body and diffing strings: what matters is whether the loader would read
        anything different, and a human who reflowed a line did not change the
        contract.
        """
        return (
            entry.goal == self.goal
            and entry.files == self.files
            and entry.verify == self.verify
            and entry.blocked_by == tuple(blocked_by)
            # Compared against the *resolved* entry stack, so a draft that says
            # nothing matches an issue that says nothing. Without this row a
            # replan that changed a task's stack would report the issue
            # unchanged and write nothing - the task would keep running on the
            # old toolchain with a plan that says otherwise.
            and entry.stack == (self.stack or DEFAULT_STACK)
        )


def _stack_of(value: str | None) -> str | None:
    """A model's stack answer, normalised, or `None` if it is not one we know.

    Dropped rather than raised. The alternative is that one hallucinated word
    in one task fails `parse_contract` for the whole plan after the issues have
    already been written - and a task with no `## Stack` is a task that runs on
    the default, which is exactly what it did before this field existed.
    `render_body` then omits the section, so a body never carries a value the
    parser would refuse to read back.
    """
    stack = (value or "").strip().strip("`").casefold()
    return stack if stack in KNOWN_STACKS else None


#: A goal line that would be read as document structure rather than prose.
#: `_split_sections` ends a section at any ATX heading and `_scan` treats a
#: fence as opaque, so a goal containing `## Files` or a code fence would
#: truncate the contract or swallow every section after it. That is issue #11's
#: trap, reached from inside the Goal section instead of from a code span.
_GOAL_ATX_RE = re.compile(r"^#{1,6}(?=[ \t]|$)")
_GOAL_FENCE_RE = re.compile(r"^(?:`{3,}|~{3,})")
_GOAL_BREAK_RE = re.compile(r"^(?:(?:-[ \t]*){3,}|(?:\*[ \t]*){3,}|(?:_[ \t]*){3,})$")


def _goal_text(text: str) -> str:
    """A multi-line goal, normalised to exactly what the parser reads back.

    Goals used to be collapsed to one line, which was the cheap way to make the
    `## Goal` section unable to open a `## Files` section inside itself. It also
    capped the worker's entire brief at one sentence: the worker is handed the
    goal and the file list and nothing else - not the objective, not the sibling
    tasks - so the collapse was throwing away the only place detail could live.

    The round trip has to be exact. `Draft.matches` compares the goal it would
    write against the one parsed off the live issue, so a write/read pair
    disagreeing by a single newline would rewrite every issue on every cycle.
    This produces precisely what `_parse_goal` returns: non-empty lines,
    stripped, defused, joined with newlines.
    """
    kept: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        line = _GOAL_ATX_RE.sub("", line).strip()
        line = _GOAL_FENCE_RE.sub("", line).strip()
        if not line or _GOAL_BREAK_RE.match(line):
            continue
        # Runs of space inside a line are decoration, exactly as they were when
        # goals were one line. Only the line breaks are content now.
        kept.append(" ".join(line.split()))
    return "\n".join(kept)


def _one_line(text: str) -> str:
    """Collapse to a single line. Both a tidy-up and a guard.

    A goal arriving as three lines is fine - the parser joins them. A goal
    whose second line begins `## Files` is not, and this is what stops the
    model from opening a section inside one.
    """
    return " ".join((text or "").split())


def _path(raw: str) -> str:
    """Strip the decoration the model adds; leave the verdict to the parser."""
    path = (raw or "").strip().strip("`").strip()
    while path.startswith("./"):
        path = path[2:]
    return path


def with_bootstrap(
    tasks: Sequence[PlannedTask], bootstrap: PlannedTask
) -> tuple[PlannedTask, ...]:
    """Put the bootstrap first and block every other task on it.

    Every task the model planned edits files that do not exist yet, so all of
    them depend on the bootstrap whether or not the model said so. Blocking
    them explicitly is what readiness (#11) reads: without it the dispatcher
    would run three workers against an empty repository in the first cycle,
    each generating its own idea of the project.

    A task that *is* the bootstrap - a replan re-emitting it under its own id -
    is left alone rather than made to depend on itself, which `order_drafts`
    would refuse.
    """
    others = [task for task in tasks if task.id != bootstrap.id]
    blocked = [
        task.model_copy(
            update={"depends_on": sorted({*task.depends_on, bootstrap.id})}
        )
        for task in others
    ]
    return (bootstrap, *blocked)


#: What a pytest collection will pick up, by its default conventions. Anchored
#: to the basename: `contest_rules.py` must not count as a test file.
_TEST_FILE = re.compile(r"(^|/)(test_[^/]+\.py|[^/]+_test\.py)$")


def _with_test_file(files: tuple[str, ...], command: str) -> tuple[str, ...]:
    """A pytest gate with nothing to collect is exit 5, forever.

    The prompt tells the model every task must pass the gate alone, and the
    first greenfield plan violated it anyway: a Goal demanding "the test must
    assert ..." over a `## Files` of one module. A worker may only write the
    files the contract lists, so that task cannot create the test its own goal
    demands, pytest collects nothing, and the whole retry budget burns on a
    contract that was unwinnable when it was written. Three attempts, one
    human reset, and a second identical failure - this repair is cheaper.

    Deterministic and minimal: when the gate runs pytest and no listed file
    matches pytest's default collection conventions, add `test_<module>.py`
    beside the first listed module. A task with no `.py` file at all is left
    alone - there is nothing sane to derive, and `parse_contract` will still
    accept it, so refusing here would reject tasks (docs, configs) that some
    other task's tests already cover.
    """
    if "pytest" not in command or any(_TEST_FILE.search(path) for path in files):
        return files
    module = next((path for path in files if path.endswith(".py")), None)
    if module is None:
        return files
    directory, _, name = module.rpartition("/")
    prefix = f"{directory}/" if directory else ""
    return (*files, f"{prefix}test_{name[:-3]}.py")


def normalise(
    tasks: Iterable[PlannedTask], *, verify: str, stack: str | None = None
) -> tuple[tuple[Draft, ...], tuple[IssueAction, ...]]:
    """Turn planned tasks into drafts, rejecting the ones that cannot be written.

    A rejection is reported, never guessed at. The one rejection that is not
    about a single task is the duplicate id: two issues carrying one id is
    control-plane corruption that aborts the *next* cycle
    (`DuplicateTaskIdError`), so the second claimant is refused here rather
    than being allowed to poison a read nobody has made yet.
    """
    drafts: list[Draft] = []
    rejected: list[IssueAction] = []
    seen: set[str] = set()
    command = _one_line(verify)

    for task in tasks:
        task_id = slugify(task.id or "") or slugify(task.goal or "")
        goal = _goal_text(task.goal)
        files = tuple(dict.fromkeys(path for path in map(_path, task.files) if path))
        # In the order a reader would check them, so the reason names the first
        # thing wrong rather than the last.
        problem = (
            "no usable task id" if not task_id
            else "a second task claims this id" if task_id in seen
            else "[Goal] section is empty" if not goal
            else "[Files] lists no files" if not files
            else "[Verify] section is empty" if not command
            else None
        )
        if problem is not None:
            rejected.append(IssueAction("rejected", task_id or task.id or "?", reason=problem))
            continue

        seen.add(task_id)
        files = _with_test_file(files, command)
        # Dependencies are slugified the same way ids are, or a model that
        # wrote `Parse Headers` in one place and `parse-headers` in the other
        # would name a task that exists and still not match it.
        deps = (slugify(dep or "") for dep in task.depends_on)
        drafts.append(
            Draft(
                task_id=task_id,
                goal=goal,
                files=files,
                verify=command,
                depends_on=tuple(dict.fromkeys(dep for dep in deps if dep)),
                # Normalised here rather than trusted, exactly as the id and
                # the prose are: a model that answers "Python" or " node "
                # meant a stack this vocabulary has, and an unknown answer is
                # dropped to `None` so the task defaults rather than failing
                # the whole plan at `parse_contract` time. The section is only
                # written when the answer is one the parser will accept back.
                # An explicit `--stack` overrides the model's answer for every
                # task. It is the operator saying what this repository *is*,
                # which is knowledge the planner does not have and cannot
                # infer from an objective; the model's own answer is the
                # fallback, not the authority.
                stack=_stack_of(stack) or _stack_of(task.stack),
            )
        )

    return tuple(drafts), tuple(rejected)


def order_drafts(drafts: Sequence[Draft]) -> tuple[Draft, ...]:
    """Dependencies first, so a `## Blocked by` ref names a number that exists.

    Raises `PlanError` on a ring, before any write. A cycle would be caught by
    readiness (#11) anyway - but only after the issues exist, at which point
    every cycle of the run aborts until a human edits a body. Refusing to write
    it is the cheaper failure by a wide margin.
    """
    remaining = {
        draft.task_id: {dep for dep in draft.depends_on if dep != draft.task_id}
        & {other.task_id for other in drafts}
        for draft in drafts
    }
    # A self-edge is dropped from `remaining` above, so it cannot deadlock the
    # walk; it is caught here instead, where the message can name it.
    self_edges = sorted(draft.task_id for draft in drafts if draft.task_id in draft.depends_on)
    if self_edges:
        raise PlanError(f"task(s) depend on themselves: {', '.join(self_edges)}")

    by_id = {draft.task_id: draft for draft in drafts}
    ordered: list[Draft] = []
    while remaining:
        # Sorted, so the same plan always produces the same issue order and a
        # diff of two runs is readable.
        free = sorted(task_id for task_id, deps in remaining.items() if not deps)
        if not free:
            raise PlanError(
                "dependency cycle among planned tasks: " + ", ".join(sorted(remaining))
            )
        for task_id in free:
            ordered.append(by_id[task_id])
            del remaining[task_id]
        for deps in remaining.values():
            deps.difference_update(free)
    return tuple(ordered)


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def _as_client(source: GitHubClient | str) -> GitHubClient:
    """A repo name builds a client from the environment; anything else is one."""
    return GitHubClient.from_env(source) if isinstance(source, str) else source


def _self_check(draft: Draft, body: str) -> str | None:
    """Read the body back with the loader's own parser. None means it round-trips.

    This is the acceptance criterion applied one issue at a time. A body that
    fails here would become an issue the loader refuses (§1.4) - present in the
    tracker, absent from the ledger, dispatched by nobody and explained to
    no one.
    """
    try:
        contract = parse_contract(_SELF_CHECK_NUMBER, body)
    except ContractError as exc:
        return f"[{exc.section}] {exc.reason}"
    if contract.task_id != draft.task_id:
        return f"[marker] identity did not round-trip: read back {contract.task_id!r}"
    return None


def _met(states: Mapping[int, IssueState], number: int) -> bool:
    """Whether a dependency is discharged. Unknown reads as unmet, never as met.

    An issue created moments ago is unknown here, and is also open, so the two
    agree. Erring the other way would label a task ready on the strength of not
    having looked.
    """
    state = states.get(number)
    return state is not None and state.satisfied


def _closed(states: Mapping[int, IssueState], number: int) -> bool:
    return bool((state := states.get(number)) and state.closed)


def write_plan(
    source: GitHubClient | str,
    plan: Plan,
    *,
    ledger: Ledger | None = None,
    verify: str | None = None,
    retire_dropped: bool = True,
    stack: str | None = None,
    bootstrap: PlannedTask | None = None,
    max_attempts: int = SETTINGS.max_attempts_per_task,
    max_total_attempts: int = SETTINGS.max_total_attempts_per_task,
) -> PlanReport:
    """Write a plan to the tracker: create what is new, update what is not.

    `ledger` is the read the caller already made - `start_run` holds one, and
    re-listing the issues to hand this function one would double the
    rate-limit cost of a cycle's first act.

    `verify` is the repo-wide command every issue's `## Verify` carries, and it
    belongs to the caller because the caller is what knows the repository:
    `cli._target` reads it off the scaffold it just committed, or off
    `--verify`. The planner invents no per-task command and lets the model
    choose none, because a command guessed wrong is a gate that was red before
    any worker touched the task. `None` falls back to `SETTINGS.verify_command`
    - v1's `SWARM_VERIFY`, and the right answer only for a repository that
    really is verified that way.

    `max_attempts` and `max_total_attempts` exist for the revival rule (module
    docstring): a kept `swarm:failed` task returns to `swarm:ready` unless its
    hard total budget is spent, and the comment the revival leaves names the
    budget state in the caps the run is actually bounded by.

    Raises `PlanError` - having written nothing - when the plan's own
    dependency graph has a ring in it.
    """
    client = _as_client(source)
    ledger = load_ledger(client) if ledger is None else ledger

    tasks = list(plan.tasks)
    if bootstrap is not None:
        # First, and everything else blocked on it. The stack the bootstrap
        # resolved is also the plan's stack, because a repository is one stack
        # (#87's non-goals: "One repo, one stack").
        tasks = list(with_bootstrap(tasks, bootstrap))
        stack = stack or bootstrap.stack
    drafts, actions = normalise(
        tasks, verify=verify or SETTINGS.verify_command, stack=stack
    )
    ordered = order_drafts(drafts)  # before the first write, or not at all

    actions = list(actions)
    warnings: list[str] = []
    # Every entry's own state, so a closed issue is recognised as one: the
    # ledger reads issues in any state and `LedgerEntry` records the label, not
    # whether GitHub has the issue open.
    states = resolve_states(client, [entry.number for entry in ledger.entries.values()])
    numbers = {task_id: entry.number for task_id, entry in ledger.entries.items()}

    for draft in ordered:
        refs: list[int] = []
        for dep in draft.depends_on:
            number = numbers.get(dep)
            if number is None:
                # The dependency was rejected, or the model named a task it did
                # not emit. Either way there is no number to write, and the
                # contract has no way to express a ref that is not `#N`.
                warnings.append(
                    f"{draft.task_id}: dropped dependency on {dep!r} - no issue for it"
                )
            elif number not in refs:
                refs.append(number)

        entry = ledger.entries.get(draft.task_id)
        if entry is None:
            actions.append(_create(client, draft, refs, states))
            if actions[-1].number is not None:
                numbers[draft.task_id] = actions[-1].number
            continue
        actions.append(
            _update(
                client,
                draft,
                entry,
                refs,
                states,
                max_attempts=max_attempts,
                max_total_attempts=max_total_attempts,
            )
        )

    planned = {draft.task_id for draft in ordered}
    for task_id, entry in sorted(ledger.entries.items(), key=lambda item: item[1].number):
        if task_id not in planned:
            actions.append(_drop(client, entry, states, retire=retire_dropped))

    return PlanReport(client.repo, tuple(actions), tuple(warnings))


def _create(
    client: GitHubClient,
    draft: Draft,
    refs: Sequence[int],
    states: Mapping[int, IssueState],
) -> IssueAction:
    """One new issue, with its body and every label, in a single POST.

    One call rather than create-then-label on purpose: a crash between the two
    would leave an issue with no state label, which §3 reads as outside the
    ledger entirely - work that exists in the tracker and that nothing will
    ever look at again.
    """
    body = draft.body(blocked_by=refs)
    problem = _self_check(draft, body)
    if problem is not None:
        return IssueAction("rejected", draft.task_id, reason=problem)

    label = READY if all(_met(states, ref) for ref in refs) else BLOCKED
    issue = client.create_issue(draft.title, body=body, labels=[label, *draft.labels])
    return IssueAction("created", draft.task_id, int(issue["number"]), reason=label)


def _update(
    client: GitHubClient,
    draft: Draft,
    entry: LedgerEntry,
    refs: Sequence[int],
    states: Mapping[int, IssueState],
    *,
    max_attempts: int = SETTINGS.max_attempts_per_task,
    max_total_attempts: int = SETTINGS.max_total_attempts_per_task,
) -> IssueAction:
    """Rewrite an existing issue's contract, or explain why this one is not touched.

    The marker is carried over verbatim - same id, same attempt - so a replan
    neither reassigns identity nor hands a task a free retry. Everything else
    in the body is replaced, including prose a human wrote outside the four
    sections; preserving that would need a section-level rewrite the loader
    does not expose, and inventing one here would be a second parser.

    The title is never patched. §2 chose a marker over a title prefix precisely
    so a human can retitle mid-run, and re-asserting a generated title every
    replan takes that back.
    """
    if _closed(states, entry.number):
        return IssueAction(
            "retained",
            draft.task_id,
            entry.number,
            reason="issue is closed; reopening it is a human's call",
        )
    if entry.state_label == FAILED:
        # The plan still contains this task - updated or unchanged, it is the
        # same decision - so the failure is not abandonment: the replan wants
        # this work done, and a failed task it cannot revive re-stalls the run
        # until a human relabels it by hand. See the module docstring for why
        # the signature budget makes this safe and what is deliberately kept.
        return revive(
            client, entry, max_attempts=max_attempts, max_total_attempts=max_total_attempts
        )
    if entry.state_label not in WRITABLE:
        return IssueAction(
            "retained",
            draft.task_id,
            entry.number,
            reason=f"{entry.state_label}; the planner does not rewrite work in flight",
        )
    if draft.matches(entry, refs):
        # Nothing to say that the issue does not already say. Worth the check:
        # a stall replans over an objective that mostly did not change, and a
        # PATCH per task per replan is rate-limit budget the workers share.
        return IssueAction("unchanged", draft.task_id, entry.number)

    body = draft.body(blocked_by=refs, attempt=entry.attempt)
    problem = _self_check(draft, body)
    if problem is not None:
        return IssueAction("rejected", draft.task_id, entry.number, reason=problem)

    client.update_issue(entry.number, body=body)
    return IssueAction("updated", draft.task_id, entry.number)


def revive(
    client: GitHubClient,
    entry: LedgerEntry,
    *,
    max_attempts: int,
    max_total_attempts: int,
    because: str = "the replan retained this task",
) -> IssueAction:
    """Return a `swarm:failed` task to `swarm:ready`, resetting nothing.

    This is the orchestrator doing exactly what the human used to do - relabel
    the failed issue so the chain behind it can move - except it does *not*
    reset the counter, because it no longer has to: the marker keeps its
    `attempt`, `blocker` and `streak` verbatim, so the budget arithmetic at the
    next failure is the guard. A retry that fails the same way finds its streak
    already at the cap and gives up immediately; one that fails differently has
    proven the old blocker gone and legitimately renews (§5's signature rule).
    That asymmetry is precisely why reviving is safe now and was not before.

    Two callers, one rule. `_update` revives a failed task a replan *kept*, and
    `goal._revive_abandoned` revives one the goal gate found blocking an unmet
    objective - the second caller is why this is public and why `because`
    exists: the comment's first sentence must say which decision put the task
    back, and everything after it is the shared arithmetic that must not fork.
    Neither caller can revive the same issue twice without a failure in
    between: a revived issue is `swarm:ready`, not `swarm:failed`, so it is
    invisible to both branches until a fresh give-up - which burned at least
    one more attempt - puts it back. The hard total cap therefore bounds the
    revive-fail-revive cycle without any counter of its own.

    The one refusal: a task whose hard total budget is spent stays failed. The
    give-up comment on the issue already says so and names the remedy, and a
    revival there would grant nothing - the very next observation would fail it
    again on the same arithmetic. Retained-and-reported costs no write, so a
    caller that keeps refusing does not spam the issue either.

    Label order is add-before-remove, `readiness._relabel`'s rule: a crash
    between the two calls leaves two state labels, which §3's precedence
    repairs, where zero labels puts the issue outside the ledger entirely. The
    comment is an aid and never a prerequisite - a client that cannot post one
    prints it instead, `reconcile.post_comment`'s discipline.
    """
    total_cap = max(int(max_total_attempts), 1)
    if entry.attempt >= total_cap:
        return IssueAction(
            "retained",
            entry.task_id,
            entry.number,
            reason=(
                f"{FAILED} with the total retry budget spent "
                f"({entry.attempt} of {total_cap}); reviving it is a human's call"
            ),
        )

    cap = max(int(max_attempts), 1)
    # An old marker carries no streak; the attempt counter is what the streak
    # was before failures had signatures (`ledger.LedgerEntry.streak`).
    streak = entry.attempt if entry.streak is None else entry.streak
    budget = f"streak {streak} of {cap}, total {entry.attempt} of {total_cap}"

    client.add_labels(entry.number, [READY])
    client.remove_label(entry.number, FAILED)

    comment = (
        f"apiary: {because}, so it is returned to `{READY}`. "
        f"The retry budget stands as it was - {budget} - so a retry that fails "
        "the same way as the last attempt gives up immediately, and one that "
        "fails differently renews its own budget."
    )
    poster = getattr(client, "create_issue_comment", None)
    if poster is None:
        print(
            f"! no create_issue_comment; revival comment for #{entry.number} not posted:\n"
            f"{comment}",
            file=sys.stderr,
        )
    else:
        try:
            poster(entry.number, comment)
        except GitHubError as exc:
            print(f"! revival comment on #{entry.number} failed: {exc}", file=sys.stderr)

    return IssueAction("revived", entry.task_id, entry.number, reason=budget)


def _drop(
    client: GitHubClient,
    entry: LedgerEntry,
    states: Mapping[int, IssueState],
    *,
    retire: bool,
) -> IssueAction:
    """Deal with a ledger entry the new plan no longer contains.

    The three cases are the module docstring's, and the one that matters is the
    middle one: an issue with a live container or an open PR is left alone.
    """
    if entry.state_label in IN_FLIGHT:
        return IssueAction(
            "retained",
            entry.task_id,
            entry.number,
            reason=f"dropped by the replan but {entry.state_label}; "
            "in-flight work is the reconciler's call, not the planner's",
        )
    if entry.state_label not in WRITABLE:
        return IssueAction(
            "retained", entry.task_id, entry.number, reason=f"dropped, {entry.state_label}"
        )
    if _closed(states, entry.number):
        return IssueAction(
            "retained", entry.task_id, entry.number, reason="dropped, already closed"
        )
    if not retire:
        return IssueAction("retained", entry.task_id, entry.number, reason="dropped, left open")

    # `not_planned`, never `completed`: readiness (#11) satisfies a dependency
    # only on `completed`, so a task waiting on this one stays blocked rather
    # than being unblocked by a cancellation.
    client.update_issue(entry.number, state="closed", state_reason="not_planned")
    return IssueAction("retired", entry.task_id, entry.number, reason="not in the new plan")


# --------------------------------------------------------------------------
# The node
# --------------------------------------------------------------------------


def _source(state: SwarmState, source: GitHubClient | str | None) -> GitHubClient | str | None:
    """Where the issues go, or None for v1's in-memory ledger.

    Deliberately *not* read from `GITHUB_REPOSITORY`. An ambient environment
    variable must never be enough to make a graph run write issues into
    whatever repository the shell happened to be pointing at; the target is
    named by the caller (`write_plan`'s `source`, which is how v2 drives this)
    or by the run state, or there is no target.
    """
    if source is not None:
        return source
    # `repo` is not a declared `SwarmState` field yet - `state.py` belongs to
    # another ticket - so it is read defensively rather than indexed.
    repo = state.get("repo") if isinstance(state, Mapping) else None
    return repo or None


def _replan_prompt(existing: Mapping[str, TaskRecord]) -> str:
    failures = "\n".join(
        f"- {task['id']} ({task.get('status')}): "
        f"{task.get('last_error', 'no error recorded')[:300]}"
        for task in existing.values()
        if task.get("status") in {"failed", "abandoned"}
    ) or "- no specific errors recorded; work simply did not converge"
    # Every id, not only the failed ones: the model needs the whole set to
    # re-emit under the right id, and an id it never sees is an id it invents a
    # replacement for.
    tracked = "\n".join(
        f"- {task['id']} ({task.get('status', 'pending')}): {task.get('goal', '')[:120]}"
        for task in existing.values()
    ) or "- none"
    return SYSTEM + REPLAN_SUFFIX.format(failures=failures, existing=tracked)


#: How long to keep asking GitHub for issues it has just accepted. Writes are
#: not immediately visible to the list endpoint - a plan can be created and the
#: very next read return the ledger as it was before it - and an orchestrator
#: that gave up there would report "the planner wrote nothing" directly beneath
#: a line naming what it wrote. Observed on a real repository twice.
READ_BACK_ATTEMPTS = 6
READ_BACK_DELAY_S = 1.0


def _read_back(client: GitHubClient, report: PlanReport, *, sleep=time.sleep) -> Ledger:
    """Re-read the tracker until it shows the plan that was just written.

    Re-reading at all is the point: `docs/architecture-v2.md` says GitHub wins
    on any disagreement, and that only means something if nothing keeps a
    second copy of the plan. But the read has to be able to *see* the write,
    and two things stop it - this client's own conditional cache, dropped here,
    and GitHub's replication, which is what the retries are for.

    Bounded, and it returns whatever it has when the budget runs out rather
    than raising: a partially visible ledger is a real state the reconciler
    handles every cycle, and the caller decides what an empty one means.
    """
    expected = {action.task_id for action in report.actions if action.number is not None}
    invalidate = getattr(client, "invalidate_cache", None)
    ledger = Ledger()
    for attempt in range(READ_BACK_ATTEMPTS):
        if invalidate is not None:
            invalidate()
        ledger = load_ledger(client, adopt=False)
        if expected <= set(ledger.entries):
            return ledger
        if attempt + 1 < READ_BACK_ATTEMPTS:
            sleep(READ_BACK_DELAY_S)
    return ledger


def prompt_for(
    objective: str,
    existing: Mapping[str, TaskRecord] | None = None,
    *,
    verify: str | None = None,
    stack: str | None = None,
) -> tuple[str, str]:
    """The exact `(system, human)` pair the planner sends.

    Note that the *system* half varies - with ledger state, because a replan
    carries the failure history and the existing ids, and now with the run's
    gate and stack. So there is no single "the planner prompt" to show.
    `swarm console` exposes fresh-plan mode only, and says so, rather than
    showing a replan prompt that would be right for one round and wrong for
    every other.
    """
    system = _replan_prompt(existing) if existing else system_prompt(verify=verify, stack=stack)
    return system, f"Objective:\n{objective}"


def draft_plan(
    objective: str,
    *,
    existing: Mapping[str, TaskRecord] | None = None,
    verify: str | None = None,
    stack: str | None = None,
    llm=None,
) -> Plan:
    """One planning call, and nothing else - no ledger, no issues, no writes.

    Split out of `plan_node` so the decomposition can be asked for on its own.
    `plan_node`'s job is to plan *and write*, which needs a repository, a token
    and a live ledger; the question "what would the planner do with this
    objective" needs none of those, and answering it used to cost all three.

    `llm` is the seam the other five call sites already have and this one did
    not, which is why every test of it has to monkeypatch two module globals.
    """
    system, human = prompt_for(objective, existing, verify=verify, stack=stack)
    model = structured(orchestrator_llm(), Plan) if llm is None else llm
    return model.invoke([("system", system), ("human", human)])


def plan_node(
    state: SwarmState,
    *,
    source: GitHubClient | str | None = None,
    verify: str | None = None,
    stack: str | None = None,
    bootstrap: PlannedTask | None = None,
) -> dict:
    """Plan (or replan) the objective, and write the result to the ledger.

    Returns the v1-shaped `tasks` dict either way. With a target repository
    that dict is *read back from GitHub* after the write, so what the graph
    sees is what the loader says the issues mean - not a projection of the plan
    that was sent, which would be a second ledger disagreeing with the first.

    Without one it falls back to v1's in-memory ledger and says so in the
    events, because a planner that silently wrote nothing reads as a planner
    that did nothing.

    `verify` is `write_plan`'s, passed straight through: the graph's node
    signature is the only reason it has to be named here at all, and defaulting
    it in this function instead of forwarding it would put a second answer next
    to the one the caller was holding.
    """
    objective = state["objective"]
    existing = state.get("tasks") or {}

    # `verify` and `stack` were already parameters of this function, used only
    # to stamp the issues after the model had answered. The model is now told
    # them first, which is what makes a task's gate knowable while it is still
    # being invented rather than after.
    plan: Plan = draft_plan(objective, existing=existing, verify=verify, stack=stack)

    target = _source(state, source)
    if target is None:
        tasks: dict[str, TaskRecord] = {
            task.id: TaskRecord(
                id=task.id,
                goal=task.goal,
                files=task.files,
                depends_on=task.depends_on,
                status="pending",
                attempts=0,
            )
            for task in plan.tasks
        }
        return {
            "tasks": tasks,
            "plan_reasoning": plan.reasoning,
            "round": state.get("round", 0),
            "events": [
                f"planned {len(tasks)} task(s) in memory: {', '.join(tasks)}",
                "no target repository; the plan was not written to the ledger",
            ],
        }

    client = _as_client(target)
    report = write_plan(client, plan, verify=verify, stack=stack, bootstrap=bootstrap)
    # Re-read rather than project: `docs/architecture-v2.md`'s "on any
    # disagreement, GitHub wins" is only a rule that means anything if nothing
    # keeps a second copy. `adopt=False` because the write above just adopted
    # everything it touched.
    ledger = _read_back(client, report)

    events = [f"planned onto {report.summary()}"]
    events += [f"  · {action}" for action in report.actions]
    events += [f"  ! {warning}" for warning in report.warnings]
    return {
        "tasks": ledger.tasks,
        "plan_reasoning": plan.reasoning,
        "round": state.get("round", 0),
        "events": events,
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import os
    import sys

    # Reads only. Writing issues from a smoke test would put model output in
    # somebody's tracker, which is exactly what `provision.py` refuses to do
    # for repositories and for the same reason.
    repo = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GITHUB_REPOSITORY", "")
    for task_id, record in sorted(load_ledger(repo, adopt=False).tasks.items()):
        print(f"{task_id:<32} {record.get('status'):<10} {record.get('branch', '')}")
