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
- `swarm:done`: history. Untouched when dropped - the work landed, and the
  issue is the record of it.
- `swarm:failed`, still open, dropped: closed as `not_planned`, with a comment
  saying the plan superseded it. This used to be retained under "`failed`
  needs a human", and that rationale is honestly out of date twice over: the
  budget machinery (§5's signatures, the hard total cap, revival) now *is* the
  human's decision within bounds - a failed task the plan still wants gets
  revived, not held for a person - and a plan that dropped the task has
  already decided the work is not wanted, so the only thing keeping the ticket
  open bought was a board that ends every run wearing a `swarm:failed` badge
  for work nobody intends to do. The marker keeps its attempt/budget record,
  so a human who disagrees reopens the issue and gets the arithmetic exactly
  where it stopped. A failed issue a human *closed* stays exactly as they left
  it: GitHub wins.

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

**Which state a task is in is the authority's answer, not the label's** (#212).
Every decision above - revive, retain, retire - read `entry.state_label`, and
#147's criterion is that a label a human edits mid-run must not change what the
orchestrator does. The case that bit: a `swarm:done` issue somebody relabels
`swarm:failed` mid-run is still believed `landed` - #147 ignores the relabel and
#201's ratchet holds it - and its issue is *open*, so the closed-issue guard in
`_update` does not catch it either. The revival branch then fired on the label
alone and put a worker back onto merged code, and before #210 the revival overlay
cleared the ratchet on its way past. `write_plan` therefore takes the cycle's
`believed` and both decision paths ask `authority.state_of`, exactly as
`recovery.py`, `goal.py` and `replan.py` have since #198. `believed=None` reads
the label, which is `plan_node` and `APIARY_STATE_SOURCE=labels`.

The states those paths compare against are ADR 0001's own rather than the six
`swarm:*` strings, and that is also what makes this a prerequisite for #152
rather than a tidy-up. That ticket removes the label writes, and a module still
*reading* them would not error: it would read absence as a state,
`entry.state_label == "swarm:failed"` would simply never be true again, and
revivals would stop happening in silence.

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

import os
import re
import sys
import time

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Mapping, Sequence

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
from ..github.refs import task_ref
from ..mcp.tracker import TrackerError
from ..taskref import TaskRef
from ..llm import orchestrator_llm, structured
from ..state import Plan, PlannedTask, SwarmState, TaskRecord

if TYPE_CHECKING:  # pragma: no cover - the annotation, never the module
    # `orchestrator/authority.py` holds the cutover (#147) and this module is
    # the fifth reader of it (#212), but the dependency must not point back up:
    # `orchestrator/goal.py` and `orchestrator/replan.py` import *this* module,
    # and `FAILED` below is spelled by hand for the same reason. So the type is
    # imported for the checker and `state_of` is reached through a local import
    # in `_state_of`, which is exactly the dance `authority._internal` does in
    # the other direction to keep `lifecycle` out of its own ring.
    from ..orchestrator.authority import Belief

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

When the tasks build components that call each other - a controller and a
service, a service and a repository, any two modules that share a boundary -
the plan must end with an integration task, depending (via depends_on) on the
tasks that built those components. Its goal is to exercise the real wiring end
to end: real objects on both sides of every boundary under test, no mocks or
stubs at those boundaries, and tests that fail if the interfaces do not
actually meet. Each component's own tests pass with its neighbour mocked as
imagined, so this task is the only one that can notice the imagined interfaces
never met. A plan whose tasks share no boundary - a single module, or genuinely
independent slices - needs no integration task; do not invent one for it.

Constraints:
- Two tasks must NEVER list the same file. If they would, merge them into one task.
- Each task must be completable by editing only the files it lists.
- Every path must be plausible for {stack}. Do not invent a stack the project
  does not use.{stack_rule}
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

    **The stack's own constraints go in too, and #293 is why.** "Every path must
    be plausible for a react project" is a statement about paths, and the model
    read it as one: asked for a to-do app it planned "Define a TypeScript
    interface for Todo items" and "Initialize a React project with Vite,
    TypeScript, and Vitest". There is no TypeScript in the react image, no
    network to fetch one, and no way for any worker to satisfy either task - a
    whole plan that was unbuildable before the first container started.

    The bootstrap task was already told exactly this (`STACK_RULE`), because it
    is the task that writes `package.json`. Nothing told the planner, which is
    the one deciding what the tasks *are*.
    """
    # From `greenfield.stacks`, which imports nothing at all - the table moved
    # there when the worker started reading it too (#293), because
    # `greenfield.bootstrap` imports the LLM module and `worker/` may not.
    # Lazy for the reason `reconcile` gives about `checks` and `goal`: an
    # importer of this module should not pay for a graph it will not use.
    from ..greenfield.stacks import STACK_RULE  # noqa: PLC0415

    # The same collapse `_one_line` does, inlined: that helper is defined far
    # below, and `SYSTEM` is built at import time.
    command = " ".join((verify or SETTINGS.verify_command or "").split())
    rule = STACK_RULE.get(stack or "", "") if stack else ""
    return SYSTEM_RULES.format(
        verify=command,
        stack=f"a {stack} project" if stack else "the project's existing stack",
        # Indented onto its own line so it reads as part of the bullet it
        # qualifies rather than running on from the sentence above it.
        stack_rule=f"\n  {rule}" if rule else "",
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


# --------------------------------------------------------------------------
# The repository listing
# --------------------------------------------------------------------------

#: What the listing *is*, said before the paths, because a bare list of files
#: is ambiguous: shown without a verb, a model reads it as inventory and plans
#: around it. The observed failure this exists for: planned blind, a run
#: implemented the same spending-tracker domain three times - root modules, a
#: class in src/main.py, and a DB layer - and invented an "Entries" entity
#: when the domain's real one already sat in the tree.
LISTING_HEADER = """The repository currently contains these files. Extend and reuse them: do not
create a parallel implementation of something that already exists, and when the
work belongs in an existing file, name that existing file in the task's files
rather than inventing a new module beside it."""

#: The most paths a prompt will carry. The listing is advisory context, and a
#: big repository must not be able to drown the objective and the rules under
#: thousands of paths; whatever is cut is summarised as a count, so the model
#: knows the list is a sample rather than the whole truth.
LISTING_CAP = 200

#: Directories that are machinery, not subject matter. A path with one of
#: these as a segment says nothing about what the project *is*, and the big
#: ones (node_modules, .venv) are exactly what would blow the cap and push the
#: real sources off the end of it.
_LISTING_SKIP_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", ".idea", ".vscode",
        ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        "__pycache__", ".eggs", "node_modules", "bower_components",
        "vendor", "vendored", "third_party", "dist", "build", "target",
        "site-packages",
    }
)

#: Files a model cannot read anything useful out of a *name* of: binaries,
#: media, archives, compiled artefacts. Suffix-matched, casefolded.
_LISTING_SKIP_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".webp", ".svg",
    ".pdf", ".zip", ".gz", ".tgz", ".tar", ".bz2", ".xz", ".7z",
    ".whl", ".egg", ".so", ".dylib", ".dll", ".exe", ".bin", ".o", ".a",
    ".pyc", ".pyo", ".class", ".jar",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".mp3", ".mp4", ".mov", ".webm", ".wav", ".avi",
    ".sqlite", ".sqlite3", ".db", ".lock",
)

#: Lockfiles whose names do not end in `.lock`. Machine-written, enormous, and
#: their existence is implied by the manifest sitting next to them.
_LISTING_SKIP_NAMES = frozenset(
    {"package-lock.json", "pnpm-lock.yaml", "npm-shrinkwrap.json"}
)


def _listable(path: str) -> bool:
    """Whether one path earns a line in the prompt. See the filters above."""
    clean = path.strip().strip("/")
    if not clean:
        return False
    parts = clean.split("/")
    if any(part.casefold() in _LISTING_SKIP_DIRS for part in parts[:-1]):
        return False
    name = parts[-1].casefold()
    return name not in _LISTING_SKIP_NAMES and not name.endswith(_LISTING_SKIP_SUFFIXES)


def format_listing(files: Sequence[str]) -> str:
    """The repository listing as the prompt carries it, or "" for nothing.

    Sorted so the same tree always renders the same block (and so siblings sit
    together, which is what lets a model see that `src/main.py` already has a
    neighbour), filtered so machinery cannot crowd out sources, and capped with
    an honest count of what was cut. Empty in, empty out: the caller appends
    nothing rather than a header announcing no files.
    """
    kept = sorted({clean for path in files if _listable(path) and (clean := path.strip().strip("/"))})
    if not kept:
        return ""
    shown = kept[:LISTING_CAP]
    lines = [LISTING_HEADER, "", *shown]
    if len(kept) > len(shown):
        lines.append(f"… and {len(kept) - len(shown)} more files")
    return "\n".join(lines)


def human_prompt(objective: str, files: Sequence[str] | None = None) -> str:
    """The human turn every planning call sends: the objective, then the tree.

    One builder for all four callers (fresh plan, replan, the goal gate's
    follow-up, the console) because the console's founding rule is that the
    prompt it shows is byte-identical to the one production sends - a second
    assembly of this string anywhere would quietly break that. With no listing
    the result is exactly the pre-listing prompt, byte for byte, which is what
    keeps the listing advisory: a caller that could not obtain one sends the
    prompt that has been working all along.
    """
    human = f"Objective:\n{objective}"
    listing = format_listing(files) if files else ""
    return f"{human}\n\n{listing}" if listing else human


def _walk(root: Path) -> tuple[str, ...]:
    """Relative POSIX paths of every file under `root`, skip-dirs pruned.

    Pruned during the walk, not just filtered afterwards, because the point of
    skipping `.venv` and `node_modules` is not only to keep them out of the
    prompt - it is to not spend seconds enumerating tens of thousands of files
    that were never going to be shown.
    """
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name for name in dirnames if name.casefold() not in _LISTING_SKIP_DIRS
        )
        rel = Path(dirpath).relative_to(root)
        found.extend((rel / name).as_posix() for name in filenames)
    return tuple(sorted(found))


def repository_files(source: GitHubClient | str | None) -> tuple[str, ...] | None:
    """What the target repository contains, or None when that cannot be known.

    The seam every caller of the planner resolves its listing through. A repo
    name or client asks the trees API; no target at all is the local/v1 path,
    which walks the checkout `SETTINGS.repo_path` names (guarded on `.git`, so
    an unrelated working directory is not presented as the project).

    Any failure - a 502 from the trees API, an empty repository the endpoint
    404s on, a client double with no `list_tree` - degrades to None, and None
    degrades to the prompt as it was before listings existed. The listing is
    advisory context, never a blocker: a planner that refused to plan because
    a tree read failed would turn a transient read error into a failed run.
    """
    try:
        if source is None:
            root = Path(SETTINGS.repo_path or "").expanduser()
            if not (root / ".git").exists():
                return None
            return _walk(root)
        return tuple(_as_client(source).list_tree())
    except Exception:  # noqa: BLE001 - every failure reads the same: no listing
        return None

# `## Blocked by` with no list items parses to no dependencies (§1.3). Written
# out rather than left blank so a human reading the issue sees an answer.
NO_DEPENDENCIES = "_none._"

# ADR 0001's internal states this module *decides* on, in the vocabulary
# `authority.state_of` answers in (#212). Spelled here rather than imported from
# `orchestrator/derived.py` for `FAILED`'s reason below - `orchestrator` modules
# import this one - and held to that module's spelling by a test, because a
# constant duplicated for a layering rule is a constant that can drift.
ELIGIBLE = "eligible"
BLOCKED_STATE = "blocked"
CLAIMED_STATE = "claimed"
REVIEW_STATE = "review"
NEEDS_HUMAN = "needs-human"

# What `_state_of` answers for a task the belief has no opinion about. A phrase
# rather than the bare `""` `Belief.state` returns, because it is in neither set
# below either way and the only thing the empty string changes is a report line
# that reads "dropped, " and looks like a defect.
NO_BELIEF = "no state this cycle believed"

# The only two states whose issues this module may rewrite, and the same set
# `authority.WAITING` names from the other side. Everything else is another
# component's row of §4, and an issue in one of them is either being worked on
# right now or finished.
WRITABLE = frozenset({ELIGIBLE, BLOCKED_STATE})

# In-flight: a container holds it, or a PR is open against it. Dropping one of
# these is a decision the planner refuses to make - see the module docstring.
IN_FLIGHT = frozenset({CLAIMED_STATE, REVIEW_STATE})

# The label a revival takes an issue out of - a *write*, which is why this one
# stays label-shaped where the sets above no longer are: the state that selects
# a revival is `NEEDS_HUMAN`, and this is the string §3 stores it in until #152
# stops storing it at all. The same spelling as `reconcile.FAILED`; spelled here
# rather than imported because `orchestrator` modules import this one, and the
# dependency must not point back up.
FAILED = "needs-human"

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
    blocked_by: Sequence[TaskRef] = (),
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
        # The ref *is* GitHub's `#N` spelling, so this renders exactly what it
        # always did; `github/refs.py` is what makes that true, not this line.
        *([f"- {ref}" for ref in blocked_by] or [NO_DEPENDENCIES]),
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

    def body(self, *, blocked_by: Sequence[TaskRef] = (), attempt: int = 0) -> str:
        return render_body(
            self.task_id,
            goal=self.goal,
            files=self.files,
            verify=self.verify,
            blocked_by=blocked_by,
            attempt=attempt,
            stack=self.stack,
        )

    def matches(self, entry: LedgerEntry, blocked_by: Sequence[TaskRef]) -> bool:
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

#: The same question for the JavaScript stacks: what `node --test` and `vitest
#: run` collect. Both take `*.test.js`; vitest also takes `.jsx`, and the react
#: scaffold's own suite is `test/App.test.jsx`.
_TEST_FILE_JS = re.compile(r"(^|/)[^/]+\.test\.(js|jsx|mjs)$")

#: How a repaired test file is named, per gate family: the marker that says the
#: gate belongs to this family, the source extensions it tests, and how to build
#: the test path from a module path.
#:
#: **A table rather than three branches, because the pytest-only version of this
#: repair was silently doing nothing for two of three stacks** (#293). It keyed
#: on the substring `pytest`, which is right for python and matches neither
#: `node --test` nor `vitest run` - so a react task declaring one component and
#: no test got no repair, and `vitest run` then passed on the *bootstrap's*
#: suite and reported green without testing the new component. That is the false
#: green, which is worse than the red it replaced.
#: Each entry is (gate markers, source extensions, collection pattern, namer).
#: The namer keeps the module's own extension for the JS stacks on purpose: a
#: `.jsx` component tested from a `.js` file is not transformed by
#: `@vitejs/plugin-react`, so the repair would add a file whose first JSX token
#: is a syntax error.
_REPAIRS: tuple[tuple[tuple[str, ...], tuple[str, ...], Any, Any], ...] = (
    (
        ("pytest",),
        (".py",),
        _TEST_FILE,
        lambda directory, stem, ext: f"{directory}test_{stem}{ext}",
    ),
    (
        ("node --test", "vitest"),
        (".js", ".jsx", ".mjs"),
        _TEST_FILE_JS,
        # `test/`, not beside the module, and the node gate is why: it reads
        # `test -n "$(ls test/*.test.js)" && node --test`, so a test written
        # anywhere else is not merely uncollected - the guard sees no test files
        # at all and the gate exits non-zero having run nothing. `vitest run`
        # collects either location, and both scaffolds already write `test/`
        # (`test/index.test.js`, `test/App.test.jsx`), so one directory serves
        # both stacks and matches what the bootstrap laid down.
        lambda directory, stem, ext: f"test/{stem}.test{ext}",
    ),
)


def _with_test_file(files: tuple[str, ...], command: str) -> tuple[str, ...]:
    """A pytest gate with nothing to collect is exit 5, forever.

    The prompt tells the model every task must pass the gate alone, and the
    first greenfield plan violated it anyway: a Goal demanding "the test must
    assert ..." over a `## Files` of one module. A worker may only write the
    files the contract lists, so that task cannot create the test its own goal
    demands, pytest collects nothing, and the whole retry budget burns on a
    contract that was unwinnable when it was written. Three attempts, one
    human reset, and a second identical failure - this repair is cheaper.

    Deterministic and minimal: when the gate is one this understands and no
    listed file matches that gate's collection conventions, add one test file
    beside the first listed module. A task with no source file the gate could
    collect against is left alone - there is nothing sane to derive, and
    `parse_contract` will still accept it, so refusing here would reject tasks
    (docs, configs) that some other task's tests already cover.

    **All three stacks, since #293.** This keyed on the substring `pytest` and
    was therefore inert for `node --test` and `vitest run`. The python half of
    that was already load-bearing - 57 of 66 recorded gate failures were a task
    whose `## Files` could not satisfy its own gate - and the JS half fails
    quieter and worse: `vitest run` collects the scaffold's suite, passes, and
    merges an untested component. `_REPAIRS` is the table; a gate this table
    does not recognise is still left alone.
    """
    for markers, extensions, pattern, name_for in _REPAIRS:
        if not any(marker in command for marker in markers):
            continue
        if any(pattern.search(path) for path in files):
            return files
        module = next(
            (path for path in files if path.endswith(extensions)), None
        )
        if module is None:
            return files
        directory, _, name = module.rpartition("/")
        stem, _, extension = name.rpartition(".")
        prefix = f"{directory}/" if directory else ""
        return (*files, name_for(prefix, stem, f".{extension}"))
    return files


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


def _met(states: Mapping[TaskRef, IssueState], ref: TaskRef) -> bool:
    """Whether a dependency is discharged. Unknown reads as unmet, never as met.

    An issue created moments ago is unknown here, and is also open, so the two
    agree. Erring the other way would label a task ready on the strength of not
    having looked.
    """
    state = states.get(ref)
    return state is not None and state.satisfied


def _closed(states: Mapping[TaskRef, IssueState], ref: TaskRef) -> bool:
    return bool((state := states.get(ref)) and state.closed)


def _state_of(entry: LedgerEntry, believed: Belief | None) -> str:
    """What state is this task in, as the cycle's authority has it (#212)?

    `authority.state_of` and nothing else - "there is exactly one function in
    that package that answers this, and every decision path calls it" - reached
    through a local import for the reason the `TYPE_CHECKING` block at the top
    of this module gives. It **raises** without a belief since #152 - there is no
    label left to fall back to - so every caller passes one, and `plan_node`
    passes an empty `Belief` deliberately: it has no cycle and no observation, so
    it has no opinion about any existing task and says so rather than inventing
    one.

    A task the belief has no opinion about answers `NO_BELIEF`, which is in
    neither `WRITABLE` nor `IN_FLIGHT` and is therefore retained rather than
    rewritten. That is the safe direction and it costs nothing in the loop: every
    caller that passes a belief passes the one built from the very ledger
    `write_plan` was handed, so an entry missing from it is a caller holding two
    different reads of the tracker - and writing to an issue on the strength of
    the disagreement is the one thing worth refusing there.
    """
    from ..orchestrator.authority import state_of

    return state_of(entry, believed) or NO_BELIEF


def write_plan(
    source: GitHubClient | str,
    plan: Plan,
    *,
    ledger: Ledger | None = None,
    verify: str | None = None,
    retire_dropped: bool = True,
    stack: str | None = None,
    bootstrap: PlannedTask | None = None,
    believed: Belief | None = None,
    max_attempts: int = SETTINGS.max_attempts_per_task,
    max_total_attempts: int = SETTINGS.max_total_attempts_per_task,
) -> PlanReport:
    """Write a plan to the tracker: create what is new, update what is not.

    `ledger` is the read the caller already made - `start_run` holds one, and
    re-listing the issues to hand this function one would double the
    rate-limit cost of a cycle's first act.

    `believed` is the cycle's authority on what state each of those entries is
    in (#212), and it belongs to the caller for the same reason `ledger` does:
    it was built from that very read, and a belief rebuilt here would be a
    second opinion about the run this function is writing for. `None` reads the
    labels, which is `plan_node` and `APIARY_STATE_SOURCE=labels`. The two
    decisions it reaches are `_update`'s revival and `_drop`'s retirement; see
    the module docstring for the mid-run relabel that made it matter.

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

    actions = list(actions)  # type: ignore[assignment]
    warnings: list[str] = []
    # Every entry's own state, so a closed issue is recognised as one: the
    # ledger reads issues in any state and `LedgerEntry` records the label, not
    # whether GitHub has the issue open.
    states = resolve_states(client, [entry.ref for entry in ledger.entries.values()])
    known = {task_id: entry.ref for task_id, entry in ledger.entries.items()}

    for draft in ordered:
        refs: list[TaskRef] = []
        for dep in draft.depends_on:
            dependency = known.get(dep)
            if dependency is None:
                # The dependency was rejected, or the model named a task it did
                # not emit. Either way there is no issue to reference, and the
                # contract has no way to express a ref to work that does not
                # exist yet.
                warnings.append(
                    f"{draft.task_id}: dropped dependency on {dep!r} - no issue for it"
                )
            elif dependency not in refs:
                refs.append(dependency)

        entry = ledger.entries.get(draft.task_id)
        if entry is None:
            actions.append(_create(client, draft, refs, states))  # type: ignore[attr-defined]
            if actions[-1].number is not None:
                known[draft.task_id] = task_ref(actions[-1].number)
            continue
        actions.append(  # type: ignore[attr-defined]
            _update(
                client,
                draft,
                entry,
                refs,
                states,
                believed=believed,
                max_attempts=max_attempts,
                max_total_attempts=max_total_attempts,
            )
        )

    planned = {draft.task_id for draft in ordered}
    for task_id, entry in sorted(ledger.entries.items(), key=lambda item: item[1].ref):
        if task_id not in planned:
            actions.append(  # type: ignore[attr-defined]
                _drop(client, entry, states, retire=retire_dropped, believed=believed)
            )

    return PlanReport(client.repo, tuple(actions), tuple(warnings))


def _create(
    client: GitHubClient,
    draft: Draft,
    refs: Sequence[TaskRef],
    states: Mapping[TaskRef, IssueState],
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

    # The state a fresh task starts in, computed and *reported* - not written.
    # This used to be `swarm:ready` or `swarm:blocked` applied to the issue, and
    # it was the last apiary-owned label anything created (#152). Membership is
    # the identity marker in the body now, so nothing is lost by not writing it;
    # what would be lost by writing it is ADR 0001's whole point, since a state
    # apiary invented would be sitting in a customer's tracker again.
    #
    # `draft.labels` still go on, and they are a different kind of thing: the
    # planner's own routing hints (`area/*`, `size/*`) that a human asked for
    # and that decide nothing here.
    state = READY if all(_met(states, ref) for ref in refs) else BLOCKED
    issue = client.create_issue(draft.title, body=body, labels=list(draft.labels))
    return IssueAction("created", draft.task_id, int(issue["number"]), reason=state)


def _update(
    client: GitHubClient,
    draft: Draft,
    entry: LedgerEntry,
    refs: Sequence[TaskRef],
    states: Mapping[TaskRef, IssueState],
    *,
    believed: Belief | None = None,
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

    **The state is the authority's, not the label's** (#212). `believed=None`
    reads the label, which is every caller outside a cycle; `_state_of` says
    what a belief with no opinion about this entry means. The `_closed` guard
    above is not made redundant by it and runs first: it is GitHub's own answer
    about the issue rather than a state, and it is the one input this module
    never argues with.
    """
    if _closed(states, entry.ref):
        return IssueAction(
            "retained",
            draft.task_id,
            entry.number,
            reason="issue is closed; reopening it is a human's call",
        )
    state = _state_of(entry, believed)
    if state == NEEDS_HUMAN:
        # The plan still contains this task - updated or unchanged, it is the
        # same decision - so the failure is not abandonment: the replan wants
        # this work done, and a failed task it cannot revive re-stalls the run
        # until a human relabels it by hand. See the module docstring for why
        # the signature budget makes this safe and what is deliberately kept.
        #
        # On the state and not on `swarm:failed`, which is #212's whole point: a
        # `swarm:done` issue somebody relabels mid-run is still believed
        # `landed`, its issue is open so the guard above does not catch it, and
        # this branch used to revive it - a worker back onto merged code.
        return revive(
            client, entry, max_attempts=max_attempts, max_total_attempts=max_total_attempts
        )
    if state not in WRITABLE:
        return IssueAction(
            "retained",
            draft.task_id,
            entry.number,
            reason=f"{state}; the planner rewrites neither work in flight nor work that has landed",
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

    comment = (
        f"apiary: {because}, so it is returned to `{READY}`. "
        f"The retry budget stands as it was - {budget} - so a retry that fails "
        "the same way as the last attempt gives up immediately, and one that "
        "fails differently renews its own budget."
    )
    _post_comment(client, entry.number, comment)

    return IssueAction("revived", entry.task_id, entry.number, reason=budget)


def _post_comment(client: GitHubClient, number: int, text: str) -> None:
    """Post one comment, or print it and carry on - never a prerequisite.

    `reconcile.post_comment`'s discipline, held locally because the dependency
    must not point from `nodes` up into `orchestrator`: a client that has no
    comment method (or a comment GitHub refused) costs the explanation its
    ideal home, not the write it was explaining.
    """
    poster = getattr(client, "create_issue_comment", None)
    if poster is None:
        print(f"! no create_issue_comment; comment for #{number} not posted:\n{text}", file=sys.stderr)
        return
    try:
        poster(number, text)
    except (GitHubError, TrackerError) as exc:
        print(f"! comment on #{number} failed: {exc}", file=sys.stderr)


def retire_superseded(client: GitHubClient, entry: LedgerEntry, *, because: str, reason: str) -> IssueAction:
    """Close a `swarm:failed` task whose work is no longer wanted, and say so.

    The second shared verb between the replan and the goal gate (`revive` is
    the first, and `because` plays the same role here): `_drop` retires a
    failed task the plan no longer contains, and `goal._retire_superseded`
    retires the failed leftovers of an objective that was met without them.
    Both are the same statement - this work is superseded - so the closure,
    the state reason and the comment's shape must not fork.

    `not_planned`, never `completed`, exactly as for every other retirement:
    readiness (#11) satisfies a dependency only on `completed`, and a task
    waiting on this one must stay blocked rather than be unblocked by a
    cancellation. The marker is untouched - its `attempt`, `blocker` and
    `streak` are the record of what was tried, and a human reopening the issue
    (relabelling it `swarm:ready`) gets the budget arithmetic exactly where it
    left off. The close lands before the comment, so a crash between the two
    leaves a closed issue whose `state_reason` already says `not_planned`
    rather than an open failed issue carrying a comment that claims it closed.
    """
    client.update_issue(entry.number, state="closed", state_reason="not_planned")
    _post_comment(
        client,
        entry.number,
        f"apiary: {because}, so it is closed as superseded. The marker keeps its "
        "attempt and budget record; reopen the issue and relabel it "
        f"`{READY}` to run it again.",
    )
    return IssueAction("retired", entry.task_id, entry.number, reason=reason)


def _drop(
    client: GitHubClient,
    entry: LedgerEntry,
    states: Mapping[TaskRef, IssueState],
    *,
    retire: bool,
    believed: Belief | None = None,
) -> IssueAction:
    """Deal with a ledger entry the new plan no longer contains.

    The three cases are the module docstring's, and the one that matters is the
    middle one: an issue with a live container or an open PR is left alone.

    Every one of them is decided by `believed` since #212 rather than by the
    label, and the drop path has its own version of the mid-run relabel: a
    `swarm:done` issue somebody typed `swarm:failed` onto, dropped by the same
    replan, was closed `not_planned` with a comment calling merged work
    superseded - and `not_planned` is precisely the state reason readiness (#11)
    refuses to read as a dependency met, so every task waiting on that landed
    one would have stayed blocked for the rest of the run.
    """
    state = _state_of(entry, believed)
    if state in IN_FLIGHT:
        return IssueAction(
            "retained",
            entry.task_id,
            entry.number,
            reason=f"dropped by the replan but {state}; "
            "in-flight work is the reconciler's call, not the planner's",
        )
    if state == NEEDS_HUMAN:
        # Dropped and failed is retired like dropped and never-started, not
        # retained like it used to be - see the module docstring for the
        # rewritten rationale. Human-closed stays untouched (GitHub wins), and
        # `retire=False` keeps the promise the flag makes everywhere else.
        if _closed(states, entry.ref):
            return IssueAction(
                "retained", entry.task_id, entry.number, reason="dropped, already closed"
            )
        if not retire:
            return IssueAction(
                "retained", entry.task_id, entry.number, reason="dropped, left open"
            )
        return retire_superseded(
            client,
            entry,
            because="the plan no longer contains this task",
            reason=f"{NEEDS_HUMAN} and not in the new plan; closed as superseded",
        )
    if state not in WRITABLE:
        return IssueAction(
            "retained", entry.task_id, entry.number, reason=f"dropped, {state}"
        )
    if _closed(states, entry.ref):
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
    return repo or None  # type: ignore[return-value]


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


def _read_back(
    client: GitHubClient, report: PlanReport, *, sleep: Callable[[float], object] = time.sleep
) -> Ledger:
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
        ledger = load_ledger(client)
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
    files: Sequence[str] | None = None,
) -> tuple[str, str]:
    """The exact `(system, human)` pair the planner sends.

    Note that the *system* half varies - with ledger state, because a replan
    carries the failure history and the existing ids, and now with the run's
    gate and stack. So there is no single "the planner prompt" to show.
    `swarm console` exposes fresh-plan mode only, and says so, rather than
    showing a replan prompt that would be right for one round and wrong for
    every other.

    `files` is the repository's current listing when the caller could obtain
    one (`repository_files` is the resolver), rendered into the *human* turn:
    it is a fact about this run, like the objective, not a rule. Absent or
    empty, the pair is byte-identical to what it was before listings existed -
    which is what the console relies on, having no repository to list.
    """
    system = _replan_prompt(existing) if existing else system_prompt(verify=verify, stack=stack)
    return system, human_prompt(objective, files)


def draft_plan(
    objective: str,
    *,
    existing: Mapping[str, TaskRecord] | None = None,
    verify: str | None = None,
    stack: str | None = None,
    files: Sequence[str] | None = None,
    llm: Any = None,
) -> Plan:
    """One planning call, and nothing else - no ledger, no issues, no writes.

    Split out of `plan_node` so the decomposition can be asked for on its own.
    `plan_node`'s job is to plan *and write*, which needs a repository, a token
    and a live ledger; the question "what would the planner do with this
    objective" needs none of those, and answering it used to cost all three.

    `llm` is the seam the other five call sites already have and this one did
    not, which is why every test of it has to monkeypatch two module globals.
    """
    system, human = prompt_for(objective, existing, verify=verify, stack=stack, files=files)
    model = structured(orchestrator_llm(), Plan) if llm is None else llm
    return model.invoke([("system", system), ("human", human)])


def plan_node(
    state: SwarmState,
    *,
    source: GitHubClient | str | None = None,
    verify: str | None = None,
    stack: str | None = None,
    bootstrap: PlannedTask | None = None,
) -> dict[str, Any]:
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
    # Resolved before the model call because the model is the consumer: the
    # listing of what the target already contains has to be in the prompt, not
    # merely in this function's hands. `repository_files` degrades to None on
    # any failure, so a tree read that 502s costs the plan its listing and
    # nothing else.
    target = _source(state, source)

    # `verify` and `stack` were already parameters of this function, used only
    # to stamp the issues after the model had answered. The model is now told
    # them first, which is what makes a task's gate knowable while it is still
    # being invented rather than after - and it is shown the repository's own
    # files for the same reason, so it extends what exists instead of planning
    # a parallel implementation beside it.
    plan: Plan = draft_plan(
        objective,
        existing=existing,
        verify=verify,
        stack=stack,
        files=repository_files(target),
    )
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
    # An empty belief, explicitly. `plan_node` is the v1 path: it has no cycle
    # behind it and therefore no observation to resolve states from, so every
    # existing entry answers `NO_BELIEF` and is *retained* rather than revived or
    # retired. That is the safe direction and it is the one this path could
    # honestly take even before #152 - it simply used to reach it by reading a
    # label instead of by admitting it did not know.
    # Local import: the module-level one is under `TYPE_CHECKING` to keep this
    # module out of `authority`'s import ring (see the block at the top).
    from ..orchestrator.authority import Belief as _Belief

    report = write_plan(
        client, plan, verify=verify, stack=stack, bootstrap=bootstrap, believed=_Belief()
    )
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
    for task_id, record in sorted(load_ledger(repo).tasks.items()):
        print(f"{task_id:<32} {record.get('status'):<10} {record.get('branch', '')}")
