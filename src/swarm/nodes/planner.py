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
  human. Both untouched.

The same reasoning governs a task the replan *keeps*: an entry that is claimed,
in review, done or failed is not rewritten either, because its body is the
contract a container is working against right now.

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

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

from ..config import SETTINGS
from ..github.client import GitHubClient
from ..github.ledger import (
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

SYSTEM = """You decompose a software objective into independent coding tasks.

HARD RULES:
- Two tasks must NEVER list the same file. If they would, merge them into one task.
- Prefer 2-4 tasks. Fewer, larger tasks beat many tiny ones.
- Each task must be completable by editing only the files it lists.
- Use depends_on only when a task literally cannot start before another finishes.
- Each task becomes one GitHub issue, and its id is how that issue is recognised
  again later. Ids are lowercase kebab-case and describe the work, not the order.

Return JSON only."""

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

ActionKind = Literal["created", "updated", "unchanged", "retired", "retained", "rejected"]


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
) -> str:
    """One issue body in the form of `docs/issue-contract.md` §6.

    The marker goes first, above everything, so it stays above the prose a
    human will keep editing (§2). Sections are emitted in the documented order
    even though the parser does not care, because the audience for the order is
    the person reading the issue.
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
    def numbers(self) -> tuple[int, ...]:
        """Issues this plan is now made of, ascending."""
        return tuple(
            sorted(
                action.number
                for action in self.of("created", "updated", "unchanged")
                if action.number is not None
            )
        )

    @property
    def changed(self) -> bool:
        return bool(self.created or self.updated or self.retired)

    def summary(self) -> str:
        return (
            f"{self.repo}: {len(self.created)} created, {len(self.updated)} updated, "
            f"{len(self.unchanged)} unchanged, {len(self.retired)} retired, "
            f"{len(self.retained)} left alone, {len(self.rejected)} rejected"
        )


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
        )


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


def normalise(
    tasks: Iterable[PlannedTask], *, verify: str
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
        goal = _one_line(task.goal)
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
) -> PlanReport:
    """Write a plan to the tracker: create what is new, update what is not.

    `ledger` is the read the caller already made - `start_run` holds one, and
    re-listing the issues to hand this function one would double the
    rate-limit cost of a cycle's first act. `verify` overrides the repo-wide
    verify command that every issue's `## Verify` carries; the planner does not
    invent a per-task command, because a command it guessed wrong is a gate
    that was red before any worker touched the task.

    Raises `PlanError` - having written nothing - when the plan's own
    dependency graph has a ring in it.
    """
    client = _as_client(source)
    ledger = load_ledger(client) if ledger is None else ledger

    drafts, actions = normalise(plan.tasks, verify=verify or SETTINGS.verify_command)
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
        actions.append(_update(client, draft, entry, refs, states))

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


def plan_node(state: SwarmState, *, source: GitHubClient | str | None = None) -> dict:
    """Plan (or replan) the objective, and write the result to the ledger.

    Returns the v1-shaped `tasks` dict either way. With a target repository
    that dict is *read back from GitHub* after the write, so what the graph
    sees is what the loader says the issues mean - not a projection of the plan
    that was sent, which would be a second ledger disagreeing with the first.

    Without one it falls back to v1's in-memory ledger and says so in the
    events, because a planner that silently wrote nothing reads as a planner
    that did nothing.
    """
    objective = state["objective"]
    existing = state.get("tasks") or {}

    prompt = _replan_prompt(existing) if existing else SYSTEM
    llm = structured(orchestrator_llm(), Plan)
    plan: Plan = llm.invoke([("system", prompt), ("human", f"Objective:\n{objective}")])

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
    report = write_plan(client, plan)
    # Re-read rather than project: `docs/architecture-v2.md`'s "on any
    # disagreement, GitHub wins" is only a rule that means anything if nothing
    # keeps a second copy. `adopt=False` because the write above just adopted
    # everything it touched.
    # Unconditional: GitHub answers a conditional re-read of a list it has just
    # accepted writes to with a 304 for a short window, and the cached body is
    # the ledger as it was *before* the plan. Re-reading is only worth doing if
    # it can see what was written.
    invalidate = getattr(client, "invalidate_cache", None)
    if invalidate is not None:
        invalidate()
    ledger = load_ledger(client, adopt=False)

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
