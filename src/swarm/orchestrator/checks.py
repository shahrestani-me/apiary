"""What CI said about a `swarm:review` PR, and what that costs the issue.

The four rows of `docs/issue-contract.md` §4 that #22 deliberately left alone -
`review -> done` on a merge, `review -> ready` on a failure with attempts left,
`review -> failed` at the cap - are this module. #22 reacts to a pull request
that has *already* merged or closed; everything that needs a check run is here,
which is also the one place in the system that merges anything.

**This module merges with an admin override, and that bypasses the review gate
the target repository declares.** Not a footnote: this repository's ruleset
requires one approving review from a code owner, GitHub refuses to let an author
approve their own pull request, and every pull request the swarm opens is
authored by the swarm. Without the override nothing it produces could ever
merge and the demo stops at the first PR. So the override is real, and its
consequence is stated here rather than discovered: **the `## Verify` command and
CI are the only quality gate this system actually has.** It is therefore an
explicit, logged and configurable action - `MergePolicy.admin_override`, off by
`APIARY_MERGE_ADMIN_OVERRIDE=0` - and a target repository that wants a human in
the loop turns it off and gets `review` issues that wait for a person to press
merge, which the next cycle then observes as a closed issue (#22's row).

**Zero checks is a third answer, not a quiet "passed".** An empty check set
happens on a repo with no workflow, on a PR that touches no triggering path, and
in the seconds after a push before GitHub has created anything. Reading it as
"no failures" merges code nothing verified; reading it as pending parks the
issue in `swarm:review` for the rest of the run. So it is neither: an empty set
is pending *until* `MergePolicy.zero_check_grace_s` has passed since the PR last
moved, and then it goes to a human with `swarm:failed` saying that nothing ever
gated it. Retrying instead would be theatre - the next attempt touches the same
paths and gets the same empty set. #32's doctor refuses to start against a repo
with no workflow at all (`doctor.check_ci`), so what reaches the grace period
here is the narrower case of a PR nothing was configured to run against.

**A retry is only worth an attempt if the failure reaches the next one.** A
re-dispatch with identical context reproduces the identical result - that is the
loop #24's stall detector exists to catch, and it is cheaper not to create it.
The failure output used to be persisted onto the issue body in a delimited block
before the label went back to `swarm:ready`. **#152 removed that write**, and
what it removed is worth stating because the block looked load-bearing and was
not: `read_feedback` was exported for a worker call site that was never written,
and the worker ended up reading *comments* instead
(`worker/entrypoint.fetch_feedback`), so nothing ever read the block. The failing
check itself is on the pull request, which is where a human looking for it goes.

**A retry after a red check now carries its reason as a comment** (#248). It did
not before, and the gap was older and larger than the block: the retry transition
carried no `comment`, and `fetch_feedback` matches only the
`apiary: attempt N failed` line that `reconcile._retry_or_give_up` posts on the
*worker-result* path. So a worker re-dispatched by this module had never had the
CI output, block or no block - charged an attempt and told nothing, which is the
exact case the paragraph above says is not worth creating.

**And a comment is a write ADR 0001 sanctions, which is the question #248 asked
to be answered rather than assumed.** ADR 0001 forbids apiary writing its own
*vocabulary and workflow* into a customer's tracker - the `swarm:*` labels, the
state machine, the counter in the body. A comment is not that: `comment` is one
of the three capabilities the ADR defines ("post the PR link or flag
needs-human", `mcp/contract.CAPABILITIES`), it goes over the MCP path like every
other tracker write since #151, and it *appends* rather than overwriting, which
was the specific sin of the body `PATCH` #152 removed - a comment cannot lose a
human's edit. The give-up branch has always commented and nobody thought that
needed an argument; the retry branch differs only in who reads it next.

**Some failures are not the worker's to fix.** A PR can pass its own
`## Verify` and still break CI, because another ticket's tests assert behaviour
this change made false - in a file the issue's `## Files` does not include and a
worker is forbidden to edit (`worker/edit.apply_edits` refuses). Feeding that
log back produces three identical failures and a `swarm:failed` for a problem
that was never in the worker's code. So the failing test paths are extracted
from the check output and compared against the declared set: a failure whose
paths are *all* outside it is routed straight to a human, with the paths named,
without spending the attempt budget on proving it three times. The extraction is
a heuristic over four failure shapes - node id, summary line, location, and
`go test`'s indented one - and is allowed to find nothing, in which case the
ordinary retry stands. The escalation is an optimisation on top of a correct
default, never a precondition for one.

It was pytest-shaped until #93, which was the most dangerous silent degradation
this repository has had: a hardcoded `.py` extension in both patterns meant
`foreign_failure` returned `()` for every other stack, so `_decide` never took
the escalate-without-consuming-attempts branch - a whole human-escalation safety
valve switched off for every new stack, with a green suite and a docstring
blessing it.

**Two things this module needs and cannot have**, probed for rather than added,
exactly as `reconcile.py` probes for `create_issue_comment` and `worker/pr.py`
for `list_pull_requests`: `GitHubClient` has no way to list pull requests (so
the issue -> PR mapping is unavailable and, without it, this module decides
nothing at all rather than guessing) and no way to delete a ref (so the
`apiary/<ref>-attempt-<n>` branch survives its merge and is reported). `client.py` is
outside this ticket's file set; both gaps degrade and neither is silent.

Manual dry run against a real repo - reads only, merges nothing, writes nothing:

    GITHUB_TOKEN=... python -m swarm.orchestrator.checks shahrestani-me/apiary

**`from_state` is the label the issue carries, not `review` (#243).** Every
transition built here used to name `review` as a constant, on the reasoning
that this gate only ever fires on a task in review - which is true of what the
gate *believes* and not of what the issue is *wearing*. A human who relabels a
task mid-review leaves the constant naming a label that is not there:
`write_labels` then adds the new one and removes nothing, the issue ends up
with two state labels, and §3's precedence reads the furthest-along of them.
The rule is `reconcile.Transition`'s and this module now follows it, which also
makes the property one rule rather than two.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..config import SETTINGS
from ..github.client import GitHubClient, GitHubError
from ..github.ledger import Ledger, LedgerEntry, load_ledger
from ..github.refs import issue_number, pull_number, pull_ref, task_ref
from ..store import StoreError, TaskStore, record_judgement
from ..taskref import PullRef, TaskRef
from ..worker.result import tail
from .authority import Belief, in_review, label_state
from .derived import ELIGIBLE, LANDED, NEEDS_HUMAN
from .dispatcher import normalise
from .reconcile import (
    COMMENT_METHOD,
    PULLS_METHOD,
    Transition,
    post_comment,
    retry_comment,
    bump_attempt,
    write_labels,
)

#: The client methods this module probes for. Named because the probe and the
#: sentence explaining what is missing must not drift, and because grepping for
#: either name should land on this line. `PULLS_METHOD` is imported from #22
#: rather than restated, so a rename cannot leave two modules disagreeing.
BRANCH_METHODS: tuple[str, ...] = ("delete_branch", "delete_ref")

#: The four answers a check set has. Strings rather than an enum because they
#: are printed, logged and asserted on far more often than they are matched.
PASSED = "passed"
FAILING = "failing"
PENDING = "pending"
EMPTY = "empty"

#: Conclusions that do not stand in the way of a merge. `neutral` and `skipped`
#: are successes for this purpose: a workflow that decided it had nothing to do
#: on this PR has not found a fault, and treating it as one would make a
#: path-filtered job unmergeable forever.
SUCCEEDING = frozenset({"success", "neutral", "skipped"})

#: Everything else a *completed* run can conclude with. Spelled out rather than
#: derived as "not succeeding", so a conclusion GitHub adds later is a value
#: this module has not seen rather than one it silently treats as a failure.
#: `stale` and `action_required` are here because both mean the same thing to a
#: merge: this commit is not cleared.
FAILING_CONCLUSIONS = frozenset(
    {"failure", "timed_out", "cancelled", "action_required", "startup_failure", "stale"}
)

#: How long an empty check set stays "pending" before it becomes a human's
#: problem. Five minutes is a queue delay; an hour is a repository whose CI does
#: not run on this PR. Not in `Settings` - `config.py` is outside this ticket's
#: file set, and the number belongs to this decision.
DEFAULT_ZERO_CHECK_GRACE_S = 300.0

#: The override, and the variable that turns it off. `1/true/yes/on` and
#: `0/false/no/off`, matching nothing else in the codebase because nothing else
#: in the codebase reads a boolean from the environment yet.
ADMIN_OVERRIDE_ENV = "APIARY_MERGE_ADMIN_OVERRIDE"
MERGE_METHOD_ENV = "APIARY_MERGE_METHOD"

#: Every line of quoted CI output is indented by this much. Still not decoration,
#: but **no longer for the reason it was written for**: it said a check log with
#: `## Verify` at column 0 would add a second section to the body and make the
#: issue malformed (`docs/issue-contract.md` §1.1). The contract is parsed from
#: the issue *body* only - `ledger.parse_contract(number, issue.get("body"))` -
#: and since #249 nothing here writes a body. Every remaining caller of `_quote`
#: writes a comment, which the parser never reads, so that hazard is gone.
#:
#: What survives is the second half of the original sentence, and it is enough on
#: its own: indenting makes the block render as code **without a fence the log
#: could itself close**. A log containing three backticks at column 0 would end a
#: fence early and spill the rest of itself into the comment as markdown.
QUOTE_INDENT = "    "

#: How much of a failure to carry. The same bound the worker's own record uses
#: (`worker/result.tail`), because this text lands in a **comment** a human reads
#: and a model is meant to be given, and a megabyte of CI log is neither. It used
#: to land in the issue body; #249 stopped writing there and #250 gave it the
#: channel the worker actually reads (`reconcile.retry_comment`, whose first line
#: `worker.entrypoint.fetch_feedback` greps for). The bound is unchanged - the
#: reason for it never depended on which of the two the text travelled in.
FEEDBACK_CHARS = 4000

#: Extensions a failing path may carry. An allow-list, because the shapes below
#: are matched in running text and "a word with a dot in it" is not a file.
#:
#: `.py` was the only one of these until #93. That was true when every
#: `## Verify` this repository wrote was a pytest invocation, and it turned off
#: `foreign_failure` - and with it the whole escalate-without-consuming-attempts
#: valve - for every other stack, silently, with a green suite.
_SOURCE_EXT = (
    "py|pyi|js|jsx|mjs|cjs|ts|tsx|mts|cts|go|rs|rb|java|kt|kts|cs|php|swift"
    "|c|cc|cpp|h|hpp|scala|ex|exs"
)

#: A path with a directory component, which is `judge._PATH_RE`'s requirement
#: copied deliberately: requiring a slash is what stops an English sentence
#: parsing as a file. The two modules answer the same question about the same
#: text, and `test_the_two_extractors_agree_on_every_language` keeps them in
#: step.
_QUALIFIED_PATH = rf"(?<![\w./+-])(?:[\w.@+-]+/)+[\w.@+-]*\.(?:{_SOURCE_EXT})(?!\w)"

#: Paths that are not the repository's, spelled as `judge._FOREIGN` spells them.
#: A traceback through `site-packages` says nothing about whether this task
#: could have reached its own fix.
_FOREIGN = ("site-packages", "dist-packages", ".venv/", "node_modules/", "/usr/", "/opt/")

#: Where a failing test path is looked for. Three shapes, none of them tied to
#: one language. Finding nothing is still a supported outcome - see the module
#: docstring - and the shapes stay narrow, because a false positive here
#: escalates an issue a worker could have fixed.
#:
#: A **node id**: `tests/test_x.py::test_y`. Pytest's, and nobody else's -
#: generalising the extension costs nothing and catches the pytest-alike
#: runners.
_NODE_ID = re.compile(rf"(?P<path>[\w./+-]+\.(?:{_SOURCE_EXT}))::")

#: A **summary line**: pytest's `FAILED tests/test_x.py`, jest's
#: `FAIL src/calc.test.js`, vitest's `FAIL src/calc.test.ts > adds`. `FAIL` and
#: leading whitespace are new here; go's `FAIL\texample.com/m\t0.002s` names a
#: package rather than a file and so carries no extension to match.
_SUMMARY_LINE = re.compile(
    rf"(?mi)^\s*(?:FAILED|FAIL|ERROR)\s+(?P<path>[\w./+-]+\.(?:{_SOURCE_EXT}))(?!\w)"
)

#: A **location**: `path:line`, optionally `:col`. This is how cargo reports a
#: panic (`panicked at src/lib.rs:12:9`), how vitest points at an assertion
#: (`❯ src/calc.test.ts:5:19`), how a `node --test` stack frame reads, and how
#: `go vet` and a failed compile report. Qualified - a directory is required -
#: because unlike the two above this shape is not introduced by a keyword.
_LOCATION = re.compile(rf"(?P<path>{_QUALIFIED_PATH}):\d+")

#: `go test`'s own failure location, which is the one shape here that has no
#: directory: a package's tests run in the package's directory, so the line
#: under `--- FAIL:` reads `    calc_test.go:12: expected 4, got 3`. The
#: `_test.go` suffix is mandatory in Go and is what makes a bare filename safe
#: to accept; `judge.mentioned_paths` still cannot see it, which is why
#: `test_go_test_output_names_a_file_judge_cannot_see` exists to say so out
#: loud rather than leaving it to be discovered.
_GO_TEST_LINE = re.compile(r"(?m)^\s+(?P<path>[\w.+-]+_test\.go):\d+:")


# --------------------------------------------------------------------------
# The one fault the merge gate may not swallow
# --------------------------------------------------------------------------


class UnresolvedJoin(RuntimeError):
    """A key this gate had to resolve was not in the map it was handed.

    Every join here and in `mergeability.py` is between two facts *one cycle
    read about one task* - a check set and the ledger entry it was read for, an
    outcome and the entry it was decided from, a held merge and the outcome it
    holds. Both sides are built in the same pass over the same entries, so a
    miss is never data. It is the two sides having drifted apart, and there is
    no neutral value to stand in for the answer:

    - **an absent check set is not an empty one.** `CheckSet()` has verdict
      `EMPTY`, and `EMPTY` past its grace period is `swarm:failed` - so a
      lookup that missed would escalate a healthy issue to a human, and the
      ledger cannot tell that from attempts genuinely exhausted;
    - **an absent ledger entry is not "no such issue".** It drops the issue out
      of mergeability, and an issue the staleness gate never decided is a merge
      it never inspected. The module's entire purpose, switched off with no log
      line saying so.

    So the miss raises, where both fail-open alternatives write a label nobody
    afterwards can tell from a real one. #174 is the ticket; #142 is why the
    shape recurs, because a `dict.get` default is how a retype ships green.

    **Raised here, recorded by the caller.** `Reconciler.cycle` catches this
    one and puts it on `CycleReport.cycle_error`, exactly as it does a
    `DependencyCycleError`, and skips dispatch for the cycle. That is not the
    catch these gates exist to prevent - it is the opposite. The merge gate
    runs *after* a cycle's labels are written, so an exception escaping `cycle`
    would be thrown before `CycleReport` exists: `on_cycle` would never fire
    and the run directory would never learn that those writes happened. A loud
    failure that erases its own evidence is not an improvement on a silent one.
    So the rule is that this must never be *defaulted*, not that it must never
    be handled - which is also what #174's acceptance criterion asks for:
    "raises **or** is explicitly handled".

    **`RuntimeError`, deliberately not `LookupError`.** The failure it reports
    *is* a failed lookup, so `LookupError` reads as the natural base - and that
    is the trap: `KeyError` is a `LookupError`, so any caller that ever wraps a
    dict access in `except LookupError` would swallow this one and silently
    restore the fail-open behaviour it exists to remove. `store.StoreError`
    picked `RuntimeError` over the same argument, and for the same reason: an
    exception that must not be recovered from should not inherit from the one
    people routinely recover from.
    """



def render_keys(values: Iterable[str], limit: int = 20) -> str:
    """The first `limit` of `values`, sorted, with the remainder counted.

    An `UnresolvedJoin` names the keys it did hold, because that is what tells
    an operator whether the map was empty, stale or keyed on the wrong thing.
    A review queue is not bounded, though, and an exception message is not the
    place to render one - so this caps what the message carries and says how
    much it left out.
    """
    ordered = sorted(values)
    if len(ordered) <= limit:
        return repr(ordered)
    return f"{ordered[:limit]!r} (+{len(ordered) - limit} more)"

# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


def _env_flag(name: str, default: bool) -> bool:
    """Loud on garbage, like `dispatcher._env_int` and for the same reason.

    A mistyped `APIARY_MERGE_ADMIN_OVERRIDE=flase` that quietly fell back to the
    default would leave a repository merging without review when somebody
    believed they had turned that off, which is the one setting here nobody may
    get by accident.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name}={raw!r} is not a boolean")


@dataclass(frozen=True)
class MergePolicy:
    """Who is allowed to merge, how, and what happens to the branch after.

    Every field exists because a target repository could reasonably want the
    other value, and the default is what this repository needs: an override that
    works, a squash because `greenfield/provision.py` configures squash-only,
    and a branch that does not outlive its PR.
    """

    #: Merge on green. False leaves a passing PR for a human and the issue in
    #: `swarm:review` until they press the button - the opt-in this module's
    #: docstring promises a repository that wants human approval enforced.
    admin_override: bool = True
    merge_method: str = "squash"
    delete_branch: bool = True
    zero_check_grace_s: float = DEFAULT_ZERO_CHECK_GRACE_S

    @classmethod
    def from_env(cls) -> MergePolicy:
        """The policy this process was started with. Read once, at the call site."""
        return cls(
            admin_override=_env_flag(ADMIN_OVERRIDE_ENV, True),
            merge_method=os.environ.get(MERGE_METHOD_ENV) or "squash",
        )

    def summary(self) -> str:
        """The line worth logging at startup, because it names what is bypassed."""
        if not self.admin_override:
            return (
                "merge policy: the swarm will not merge; a human presses the button "
                f"({ADMIN_OVERRIDE_ENV}=0)"
            )
        return (
            f"merge policy: {self.merge_method} with an admin override, which bypasses "
            f"this repository's required review - verify and CI are the only gate "
            f"({ADMIN_OVERRIDE_ENV}=0 to require a human)"
        )


# --------------------------------------------------------------------------
# What CI said
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckSet:
    """One commit's check runs, folded into the only question a merge asks.

    Kept as three name lists rather than one verdict, because "which check is
    still running" and "which one failed" are the two things a human asks next
    and neither is recoverable from a single string.
    """

    total: int = 0
    succeeded: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    pending: tuple[str, ...] = ()
    output: str = ""
    #: True when the checks could not be read at all this cycle. Distinct from
    #: an empty set for `Snapshot.open_branches`'s reason: "we did not look" and
    #: "there is nothing there" decide different labels.
    unreadable: bool = False

    @property
    def verdict(self) -> str:
        """`failing` beats `pending` beats `passed`; nothing at all is `empty`.

        A failure alongside a still-running check is reported as a failure: the
        PR cannot pass, and waiting for the rest of the matrix to finish before
        starting the retry buys nothing but wall clock.
        """
        if self.unreadable:
            return PENDING
        if self.failed:
            return FAILING
        if self.pending:
            return PENDING
        return PASSED if self.total else EMPTY

    def summary(self) -> str:
        if self.unreadable:
            return "check runs could not be read"
        if not self.total:
            return "no check runs on this commit"
        parts = [f"{len(self.succeeded)} passed"]
        if self.failed:
            parts.append(f"{len(self.failed)} failed ({', '.join(self.failed)})")
        if self.pending:
            parts.append(f"{len(self.pending)} still running ({', '.join(self.pending)})")
        return ", ".join(parts)


def summarise_checks(runs: Iterable[Mapping[str, Any]]) -> CheckSet:
    """Fold GitHub's check-run payloads. Pure; `read_checks` is the half with I/O.

    A run is pending unless `status` is `completed`, which covers `queued`,
    `in_progress`, `waiting`, `requested` and anything GitHub adds - the
    complement is the safe direction here, because an unknown status treated as
    complete is a merge nobody checked.
    """
    succeeded: list[str] = []
    failed: list[str] = []
    pending: list[str] = []
    evidence: list[str] = []
    total = 0

    for run in runs:
        total += 1
        name = str(run.get("name") or "?")
        if str(run.get("status") or "") != "completed":
            pending.append(name)
            continue
        conclusion = str(run.get("conclusion") or "").lower()
        if conclusion in SUCCEEDING:
            succeeded.append(name)
            continue
        failed.append(name)
        evidence.append(_evidence(name, conclusion, run))

    return CheckSet(
        total=total,
        succeeded=tuple(succeeded),
        failed=tuple(failed),
        pending=tuple(pending),
        output="\n\n".join(evidence),
    )


def _evidence(name: str, conclusion: str, run: Mapping[str, Any]) -> str:
    """One failing check, as much of it as the checks API carries.

    The API returns a summary and a text blob, not the job log; `details_url` is
    included because for a genuinely opaque failure it is the only thing that
    leads anywhere, and a retry prompt that says "it failed, ask GitHub" is at
    least honest about what is known.
    """
    output = run.get("output") if isinstance(run.get("output"), Mapping) else {}
    lines = [f"{name}: {conclusion or 'no conclusion'}"]
    for key in ("title", "summary", "text"):
        value = str(output.get(key) or "").strip()  # type: ignore[union-attr]
        if value:
            lines.append(value)
    url = str(run.get("details_url") or run.get("html_url") or "").strip()
    if url:
        lines.append(url)
    return "\n".join(lines)


def failing_paths(text: str) -> tuple[str, ...]:
    """Every source path a failure names, in the order it first appears.

    Four shapes, none of them tied to one language: a node id
    (`tests/test_x.py::test_y`), a summary line (`FAILED tests/test_x.py`,
    `FAIL src/calc.test.ts`), a location (`src/lib.rs:12:9`), and `go test`'s
    indented `calc_test.go:12:`. Deliberately conservative - a bare filename
    with none of those around it is as likely to be prose about the change as
    evidence about the failure, and a false positive here escalates an issue a
    worker could have fixed.

    **The location shape widens what pytest output yields**, since a pytest
    traceback line is `tests/test_x.py:12: AssertionError` and nothing looked at
    those before. That is the safe direction: `foreign_failure` escalates only
    when *every* named path is outside the declared set, so finding more paths
    can only make it escalate less often and retry more often. Escalation is an
    optimisation on top of a correct default.

    Paths outside the repository are dropped, on the same list `judge` drops
    them on: a frame through `site-packages` or `node_modules` says nothing
    about whether this task could have reached its own fix.
    """
    found: dict[str, None] = {}
    for pattern in (_SUMMARY_LINE, _NODE_ID, _LOCATION, _GO_TEST_LINE):
        for match in pattern.finditer(text or ""):
            path = match.group("path")
            if path.startswith("/") or any(part in path for part in _FOREIGN):
                continue
            found.setdefault(path.lstrip("./"), None)
    return tuple(found)


def foreign_failure(entry: LedgerEntry, text: str) -> tuple[str, ...]:
    """The failing paths this issue is not allowed to touch, or `()`.

    Empty when the failure names no path at all *or* names at least one inside
    the declared set: in the second case the worker has something it can act on,
    and a retry is the right answer even if other files failed too. Only a
    failure lying entirely outside `## Files` is one no attempt can fix, and
    that is the one that goes to a human (module docstring).
    """
    named = failing_paths(text)
    if not named:
        return ()
    declared = {normalise(path) for path in entry.files}
    outside = tuple(path for path in named if normalise(path) not in declared)
    return () if len(outside) < len(named) else outside


# --------------------------------------------------------------------------
# The pull request
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PullState:
    """The four things about an open PR that a merge decision needs.

    `sha` may be empty, and everything here tolerates that: a listing that does
    not carry the head sha still identifies the branch, `list_check_runs`
    accepts a branch name as a ref, and a merge without one is simply not
    conditional on the head this cycle inspected.
    """

    #: A `PullRef`, not an `int` (#185). This is the *source* of the pull
    #: request numbers that end up on `Merge`, `Mergeability` and `Decision`,
    #: so typing it here is what stops a bare pull request number ever being in
    #: scope next to an issue's - retyping only the destinations would leave
    #: `Merge(number=pull.number, ...)` well-typed, which is the bug itself.
    number: PullRef
    branch: str
    sha: str = ""
    updated_at: dt.datetime | None = None
    draft: bool = False

    @property
    def ref(self) -> str:
        """What to ask for check runs about. The sha when there is one."""
        return self.sha or self.branch

    def age_s(self, now: dt.datetime) -> float:
        """Seconds since this PR last moved. Zero when GitHub said nothing.

        Zero rather than infinity on purpose: an unparseable timestamp must not
        age a PR straight past the grace period and into a human's queue, and
        the alternative failure - waiting one more cycle - costs an interval.
        """
        if self.updated_at is None:
            return 0.0
        return max((now - self.updated_at).total_seconds(), 0.0)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PullState:
        head = payload.get("head") if isinstance(payload.get("head"), Mapping) else {}
        return cls(
            number=pull_ref(int(payload.get("number") or 0)),
            branch=str(head.get("ref") or ""),  # type: ignore[union-attr]
            sha=str(head.get("sha") or ""),  # type: ignore[union-attr]
            updated_at=_parse(payload.get("updated_at") or payload.get("created_at")),
            draft=bool(payload.get("draft", False)),
        )


def _parse(value: Any) -> dt.datetime | None:
    """GitHub's `2026-08-14T14:25:30Z`, or nothing. Never raises.

    Forgiving because the only decision this feeds is a grace period, and
    `artifacts._parse` already set the precedent: an unreadable timestamp costs
    a judgement, not a cycle.
    """
    if not value:
        return None
    try:
        moment = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=dt.timezone.utc)


def read_pulls(client: Any) -> dict[str, PullState] | None:
    """Head ref -> open PR, or `None` when this client cannot list them.

    `None` is not an empty mapping, and #22 already paid for the distinction:
    an empty mapping means every `swarm:review` issue's PR is gone, and `None`
    means we did not look. Conflating them would merge nothing and fail
    everything.
    """
    lister = getattr(client, PULLS_METHOD, None)
    if lister is None:
        return None
    # Only `state` is passed, for `worker/pr.py`'s reason: a future
    # `list_pull_requests` is certain to take it and may or may not take `head`,
    # and guessing wrong would be a `TypeError` in the loop body.
    pulls: dict[str, PullState] = {}
    for payload in lister(state="open") or ():
        state = PullState.from_payload(payload)
        if state.branch:
            pulls.setdefault(state.branch, state)
    return pulls


def read_checks(client: Any, ref: str) -> CheckSet:
    """One commit's checks. A read that failed is `unreadable`, never `empty`.

    One request per `swarm:review` issue per cycle, which is the whole API cost
    this module adds. It is proportional to the review queue rather than to the
    ledger, and the queue is bounded by the dispatch cap plus whatever is
    waiting on a human.
    """
    try:
        return summarise_checks(client.list_check_runs(ref))
    except GitHubError as check_error:
        # A fine-grained PAT cannot read check runs: the `checks` permission is
        # not offered when minting one, so this is 403 for every least-privilege
        # token rather than a rare failure. Actions runs carry the same three
        # fields the fold needs and `actions:read` is grantable, so the gate is
        # readable after all - just not by the obvious call.
        lister = getattr(client, "list_workflow_runs", None)
        if lister is not None:
            try:
                return summarise_checks(lister(ref))
            except GitHubError as actions_error:
                print(
                    f"! neither check runs nor workflow runs for {ref} could be read: "
                    f"{actions_error}",
                    file=sys.stderr,
                )
                return CheckSet(unreadable=True)
        print(f"! check runs for {ref} could not be read: {check_error}", file=sys.stderr)
        return CheckSet(unreadable=True)


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Merge:
    """One pull request this cycle would merge, and under whose authority.

    `admin_override` is carried on the record rather than read from the policy
    at the call site so that the log line and the merge cannot disagree about
    which authority was used - which is the whole point of making the override
    explicit rather than a default nobody sees.
    """

    #: The issue this merge closes - a GitHub issue number, and the same one
    #: the `Outcome` carrying this record holds. Kept a number (#174) because
    #: what still reads it is the *reporting* surface, whose migration is not
    #: this gate's: it keys `ChecksReport.merged` and
    #: `ChecksReport.merge_commits`, and `lifecycle.py` joins it against the
    #: issue-numbered half of its own index.
    #:
    #: **`ref` below is the half that joins**, as it is on `Outcome` and for the
    #: same reason. `apply_checks` collects the merges GitHub refused and tests
    #: each outcome against that set, to stop a `swarm:done` being written for a
    #: merge that did not happen; #174 left that join int-against-int and #181
    #: moved it, because "safe by construction" was the argument and this record
    #: is the one place in the module where two numberings sit side by side.
    number: int
    #: The pull request GitHub is asked to merge. A *different* numbering from
    #: `number`, which is why both are spelled out: `#23: merge PR #101`.
    #:
    #: **This is the field the two numberings used to collide on, and #185 is
    #: why they cannot any more.** A `Merge` built with this number in
    #: `number`'s place used to be well-typed, read fine, and mint a ref for a
    #: pull request - so the refusal was filed under an identity no outcome
    #: carried and the `swarm:done` went out anyway. `PullRef` is nominally
    #: distinct from the `int` beside it, so mypy now rejects the swap in both
    #: directions and it is not expressible rather than merely detected.
    #:
    #: #184's `UnresolvedJoin` guard in `apply_checks` stays regardless: it
    #: catches the same consequence arising from a *human* error - two records
    #: that genuinely disagree about which task a merge belongs to - which no
    #: type can rule out.
    pull: PullRef
    branch: str
    sha: str = ""
    merge_method: str = "squash"
    delete_branch: bool = True
    admin_override: bool = True
    commit_title: str = ""

    @property
    def ref(self) -> TaskRef:
        """The task this merge closes, in the internal model's vocabulary.

        The same ref the `Outcome` carrying this record answers, which is the
        whole point: `apply_checks` joins the two.
        """
        return task_ref(self.number)

    def __str__(self) -> str:
        how = "admin override" if self.admin_override else "no override"
        after = f", deleting {self.branch}" if self.delete_branch else ""
        # `{self.pull}` renders `#101` on its own - the ref carries the `#` -
        # so the literal one is gone and the printed line is byte-identical.
        return f"#{self.number}: merge PR {self.pull} ({self.merge_method}, {how}{after})"


@dataclass(frozen=True)
class Outcome:
    """What one `swarm:review` issue's checks mean for it. One row per issue.

    A row rather than four parallel lists, because every interesting assertion -
    and every interesting log line - is about one issue: what its checks said,
    what label that moves, whether a human is now needed and what the next
    attempt must be told.
    """

    #: The issue this row is about, as GitHub numbers it. `ChecksPlan.escalated`
    #: prints it, `apply_checks` addresses the API with it, and `lifecycle.py`
    #: joins it against the issue-numbered half of its index - all three are the
    #: code-host vocabulary, which ADR 0001 says stays GitHub-shaped.
    #:
    #: **`ref` below is the half that joins.** #142 retyped the task-identity
    #: model and stopped here, and #174 is what that cost: `plan_mergeability`
    #: looked its entry up by this number, and a miss returned `None` and
    #: dropped the issue out of the staleness gate in silence. The lookup is on
    #: the ref now and raises `UnresolvedJoin` instead.
    number: int
    verdict: str
    detail: str = ""
    transition: Transition | None = None
    merge: Merge | None = None
    #: The failure output that must reach the next attempt, persisted onto the
    #: issue before the label goes back to `swarm:ready`. Empty for every
    #: outcome that is not a retry.
    feedback: str = ""
    #: A failure no attempt of this issue could fix - see the module docstring.
    escalated: bool = False

    def __str__(self) -> str:
        return f"#{self.number}: {self.verdict} - {self.detail}"

    @property
    def ref(self) -> TaskRef:
        """This row's task, in the internal model's vocabulary. See `number`."""
        return task_ref(self.number)


@dataclass(frozen=True)
class ChecksPlan:
    """Everything one pass would change, computed without writing anything.

    Pure, so every case that matters - a green PR, a red one at the cap, an
    empty check set inside and outside its grace period, a failure in somebody
    else's test file - is testable as data rather than as a mocked API.
    """

    outcomes: tuple[Outcome, ...] = ()
    #: True when pull requests could not be listed at all. Nothing was decided;
    #: nothing failed. See `read_pulls`.
    blind: bool = False
    policy: MergePolicy = MergePolicy()

    @property
    def transitions(self) -> tuple[Transition, ...]:
        return tuple(o.transition for o in self.outcomes if o.transition is not None)

    @property
    def merges(self) -> tuple[Merge, ...]:
        return tuple(o.merge for o in self.outcomes if o.merge is not None)

    @property
    def escalated(self) -> tuple[int, ...]:
        """Issues handed to a human rather than retried. The list worth printing."""
        return tuple(o.number for o in self.outcomes if o.escalated)

    @property
    def changed(self) -> bool:
        return bool(self.transitions or self.merges)

    def summary(self) -> str:
        parts = [
            f"{len(self.merges)} merge(s)",
            f"{len(self.transitions)} transition(s)",
            f"{len(self.outcomes)} review issue(s)",
        ]
        if self.escalated:
            parts.append("needs a human: " + ", ".join(f"#{n}" for n in self.escalated))
        if self.blind:
            parts.append(f"pull requests unreadable - the client has no {PULLS_METHOD}")
        return ", ".join(parts)


def plan_checks(
    ledger: Ledger,
    *,
    pulls: Mapping[str, PullState] | None,
    checks: Mapping[TaskRef, CheckSet],
    policy: MergePolicy | None = None,
    max_attempts: int = SETTINGS.max_attempts_per_task,
    now: dt.datetime | None = None,
    believed: Belief | None = None,
) -> ChecksPlan:
    """Decide every task in review. Pure - no API call, no daemon, no model.

    "In review" is the cycle's authority since #147 (`authority.in_review`), not
    the `swarm:review` label: a gate that still read the label would merge, or
    refuse to merge, on a label a human edited mid-run - and not changing what
    the orchestrator does on such an edit is the whole of that ticket. Both
    sides say the same thing in the ordinary case; `believed=None` reads the
    label, which is every caller outside `Reconciler.cycle`.

    `pulls` is the open pull requests by head ref (`None` when this cycle could
    not look), `checks` the folded check set per task ref, and `now` the moment
    the grace period is measured against. All three are facts somebody else
    read; keeping the I/O out of here is what makes the rules assertable.

    **`checks` must carry every issue this loop reaches, and a miss raises.**
    The caller reads a check set for exactly the entries selected below - in
    `swarm:review`, with an open pull request - so the two sets are the same set
    or somebody has broken the wiring.

    The default this replaced was `CheckSet()`, which reads as `EMPTY`, and
    `EMPTY` past its grace period escalates the issue to `swarm:failed` - #174's
    headline, a healthy task marked as needing a human because its check runs
    could not be looked up. A *neutral* stand-in does exist in the type, and it
    is worth saying why it is not used: `CheckSet(unreadable=True)` reads as
    `PENDING`, which is what `read_checks` returns when GitHub could not be
    asked, and it escalates nothing. But it is the right answer to a different
    question. "GitHub did not answer" is a fact about this cycle that the next
    cycle re-reads; "the map I was handed has no key for this task" is a fact
    about the code, and mapping it to `PENDING` parks the task in
    `swarm:review` every cycle for the rest of the run with nothing anywhere
    saying why. That is the same disease as the escalation, only quieter - so
    the miss is raised and `Reconciler.cycle` records it where a human sees it.

    Issues with no open PR are skipped rather than decided: that is #22's row -
    it reads a `swarm:review` issue whose branch has no open PR as a PR closed
    unmerged - and two modules writing one transition is how a label ends up
    moving twice in one cycle.
    """
    rules = policy or MergePolicy()
    moment = now or dt.datetime.now(dt.timezone.utc)
    outcomes: list[Outcome] = []

    for entry in sorted(ledger.entries.values(), key=lambda entry: entry.ref):
        if not in_review(entry, believed) or pulls is None:
            continue
        pull = pulls.get(entry.branch)
        if pull is None:
            continue
        try:
            found = checks[entry.ref]
        except KeyError:
            raise UnresolvedJoin(
                f"no check set for {entry.ref} ({entry.task_id}), whose pull request "
                f"{pull.ref} this cycle listed as open. The caller reads one check set per "
                f"reviewable entry, so this is the two halves having drifted apart - and "
                f"the miss cannot be defaulted: an absent check set reads as an empty one, "
                f"which escalates this issue to swarm:failed. Keys held: "
                f"{render_keys(str(key) for key in checks)}"
            ) from None
        outcomes.append(_decide(entry, pull, found, rules, max_attempts, moment))

    return ChecksPlan(outcomes=tuple(outcomes), blind=pulls is None, policy=rules)


def _decide(
    entry: LedgerEntry,
    pull: PullState,
    checks: CheckSet,
    policy: MergePolicy,
    max_attempts: int,
    now: dt.datetime,
) -> Outcome:
    """One issue's verdict. The order of the branches is the priority."""
    verdict = checks.verdict

    if verdict == PENDING:
        # Includes the unreadable case, which is pending for `Snapshot`'s
        # reason: a check set we could not fetch must never read as a check set
        # that passed, nor as one that failed.
        return Outcome(entry.number, PENDING, checks.summary())

    if verdict == EMPTY:
        return _decide_empty(entry, pull, policy, now)

    if verdict == PASSED:
        return _decide_passed(entry, pull, checks, policy)

    outside = foreign_failure(entry, checks.output)
    feedback = tail(checks.output, FEEDBACK_CHARS)
    if outside:
        # The failure this ticket exists to stop burning attempts on. Naming the
        # paths matters more than the label: it is the difference between "the
        # swarm gave up on #23" and "#23 broke tests/test_other.py, which #23 is
        # not allowed to edit".
        names = ", ".join(outside)
        return Outcome(
            number=entry.number,
            verdict=FAILING,
            detail=f"CI failed in {names}, outside this issue's ## Files",
            transition=Transition(
                ref=entry.ref,
                from_state=label_state(entry.state_label),
                to_state=NEEDS_HUMAN,
                reason=(
                    f"CI failed in {names}, which is outside this issue's ## Files - "
                    f"no attempt of this issue could fix it"
                ),
                task_id=entry.task_id,
                comment=(
                    f"apiary: CI failed on this PR, but every failing test is outside the "
                    f"`## Files` this issue declares ({names}), and a worker may not edit "
                    f"a file it did not declare. Retrying would reproduce it. A human needs "
                    f"to decide whether this change or that test is wrong.\n\n"
                    f"{_quote(feedback)}"
                ),
            ),
            escalated=True,
        )

    return _retry_or_give_up(entry, checks, max_attempts, feedback)


def _decide_empty(
    entry: LedgerEntry, pull: PullState, policy: MergePolicy, now: dt.datetime
) -> Outcome:
    """No checks at all: pending until the grace runs out, then a human's.

    Never merged and never retried. See the module docstring - the retry would
    touch the same paths and produce the same nothing, and the merge would be of
    code that nothing verified.
    """
    waited = pull.age_s(now)
    if waited < policy.zero_check_grace_s:
        return Outcome(
            entry.number,
            PENDING,
            f"no check runs yet, {waited:.0f}s into a {policy.zero_check_grace_s:.0f}s grace",
        )
    reason = (
        f"no check run was ever created for {pull.ref} in {waited:.0f}s - nothing gated this "
        f"pull request, so nothing verified it"
    )
    return Outcome(
        number=entry.number,
        verdict=EMPTY,
        detail=reason,
        transition=Transition(
            ref=entry.ref,
            from_state=label_state(entry.state_label),
            to_state=NEEDS_HUMAN,
            reason=reason,
            task_id=entry.task_id,
            comment=(
                f"apiary: {reason}. Merging it would put unverified code on the default "
                f"branch and retrying would produce the same empty check set, so this is a "
                f"human's call: merge it by hand, or give the repository a workflow that "
                f"runs on these paths (`swarm doctor` checks for one)."
            ),
        ),
        escalated=True,
    )


def _decide_passed(
    entry: LedgerEntry, pull: PullState, checks: CheckSet, policy: MergePolicy
) -> Outcome:
    """Green. Merge it, unless a human was asked to be the one who does.

    `swarm:review -> swarm:done` is written here rather than left to the next
    cycle's `Closes #<n>`: this module is §4's writer for that row, and a label
    that lags the merge by a cycle is a cap slot nobody can use in the meantime.
    """
    if pull.draft:
        return Outcome(entry.number, PENDING, "checks passed, but the pull request is a draft")
    if not policy.admin_override:
        return Outcome(
            entry.number,
            PASSED,
            f"checks passed - waiting for a human to merge PR {pull.number} "
            f"({ADMIN_OVERRIDE_ENV} is off)",
        )
    return Outcome(
        number=entry.number,
        verdict=PASSED,
        detail=f"{checks.summary()}; merging PR {pull.number}",
        transition=Transition(
            ref=entry.ref,
            from_state=label_state(entry.state_label),
            to_state=LANDED,
            reason=f"PR {pull.number} merged: {checks.summary()}",
            task_id=entry.task_id,
        ),
        merge=Merge(
            number=entry.number,
            pull=pull.number,
            branch=pull.branch,
            sha=pull.sha,
            merge_method=policy.merge_method,
            delete_branch=policy.delete_branch,
            admin_override=policy.admin_override,
            # The one line a `git log` on the default branch shows. `Closes
            # #<n>` lives in the PR body (`worker/pr.py`) and GitHub keeps the
            # body as the squash message, so the merge still closes the issue.
            commit_title=f"swarm[{entry.task_id}]: {entry.title}"[:72],
        ),
    )


def _retry_or_give_up(
    entry: LedgerEntry, checks: CheckSet, max_attempts: int, feedback: str
) -> Outcome:
    """Consume an attempt, and decide whether any remain.

    The same shape as `reconcile`'s counter rule and for the same reason: the
    increment rides on the transition so it is persisted *before* the label goes
    back to `swarm:ready` (`docs/issue-contract.md` §5), and a crash between the
    two costs an attempt rather than granting a free one.

    **Both branches comment, and the retry's is the one #248 was about.** The
    give-up branch has always said why, because a human is about to read it. The
    retry branch said nothing to anybody: the CI output went into a delimited
    block in the issue body that nothing ever read, and the worker looks for its
    feedback in the *comments* - so a task re-dispatched because CI went red was
    charged an attempt and told nothing. `reconcile.retry_comment` is the
    formatter, not a second one, because its first line is the string
    `worker.entrypoint.fetch_feedback` greps for and a second speller of that
    line is how the contract drifts.
    """
    attempt = entry.attempt + 1
    cap = max(int(max_attempts), 1)
    named = ", ".join(checks.failed) or "CI"
    if attempt >= cap:
        return Outcome(
            number=entry.number,
            verdict=FAILING,
            detail=f"{named} failed; {attempt} attempt(s) against a cap of {cap}",
            transition=Transition(
                ref=entry.ref,
                from_state=label_state(entry.state_label),
                to_state=NEEDS_HUMAN,
                reason=f"{named} failed; {attempt} attempt(s) made against a cap of {cap}",
                task_id=entry.task_id,
                attempt=attempt,
                comment=(
                    f"apiary: giving up after {attempt} attempt(s). {named} failed on the "
                    f"last one:\n\n{_quote(feedback)}"
                ),
            ),
            feedback=feedback,
        )
    reason = f"{named} failed on the pull request"
    return Outcome(
        number=entry.number,
        verdict=FAILING,
        detail=f"{named} failed; retrying as attempt {attempt} of {cap}",
        transition=Transition(
            ref=entry.ref,
            from_state=label_state(entry.state_label),
            to_state=ELIGIBLE,
            reason=reason,
            task_id=entry.task_id,
            attempt=attempt,
            # The CI output travels as `verify_output`, which is fenced and
            # clipped: it is a foreign log, and a line of it reading `## Verify`
            # at column 0 would corrupt the contract while reporting a failure.
            comment=retry_comment(attempt, reason, feedback),
        ),
        feedback=feedback,
    )


# --------------------------------------------------------------------------
# The retry's context
# --------------------------------------------------------------------------


def _quote(text: str) -> str:
    """CI output, indented so no line of it can be markdown or a section heading.

    See `QUOTE_INDENT`. Applied wherever this module writes the output itself,
    which since #249 is the two escalation comments and no longer the body block.
    The hazard it now guards is markdown rather than the contract: a comment is
    not parsed by `ledger.parse_contract`, so a log line reading `## Verify` at
    column 0 can no longer make an issue malformed - it can only render as a
    heading in the middle of a failure report, and three backticks at column 0
    can close a fence the log is sitting inside.

    **The retry comment neutralises the same hazard a different way**, and the
    two are worth telling apart rather than unifying. `reconcile.retry_comment`
    fences the output instead of indenting it, and its docstring argues the
    split: a fence is right for the text a *model* is about to be handed, an
    indent is right where the surrounding prose is a sentence to a human. Both
    are safe; neither is a fallback for the other.
    """
    body = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not body:
        return f"{QUOTE_INDENT}(no output)"
    return "\n".join(f"{QUOTE_INDENT}{line}".rstrip() for line in body.split("\n"))



# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Failure:
    """One thing this pass could not do, and to which issue.

    Collected rather than raised, exactly as `reconcile.apply_plan` collects:
    one PR GitHub will not merge must not stop the other nineteen from being
    decided.
    """

    number: int
    reason: str

    def __str__(self) -> str:
        return f"#{self.number}: {self.reason}"


@dataclass(frozen=True)
class ChecksReport:
    """What one pass actually achieved."""

    plan: ChecksPlan
    merged: tuple[int, ...] = ()
    applied: tuple[Transition, ...] = ()
    deleted: tuple[str, ...] = ()
    failures: tuple[Failure, ...] = ()
    #: Branches a merge should have deleted and this client cannot - see the
    #: module docstring. Not a failure; a gap worth reporting, because otherwise
    #: a long run quietly leaves one dead branch per task.
    undeleted: tuple[str, ...] = ()
    #: Comments this client had no method to post. `reconcile.post_comment`
    #: printed them instead.
    uncommented: tuple[int, ...] = ()
    #: The commit each merge produced, as (issue number, sha) - GitHub's answer
    #: to the `PUT .../merge`, which this module previously threw away.
    #: Announcement only: nothing here reads it, and `pr.merged` (#141) is the
    #: only consumer, because "which commit is this task now" is the one fact
    #: about a landed task that the run directory could not otherwise recover.
    #:
    #: A tuple, like every other collection on this frozen record and for the
    #: same reason: a `dict` field makes the generated `__hash__` raise, and a
    #: frozen dataclass that cannot be hashed is frozen in name only. Absent for
    #: a merge whose response carried no `sha` - `merge_pull_request` is typed
    #: `Any` and a body-less 200 is a real answer.
    merge_commits: tuple[tuple[int, str], ...] = ()

    @property
    def commit_by_issue(self) -> dict[int, str]:
        """`merge_commits` as a lookup, for the one caller that wants one."""
        return dict(self.merge_commits)

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        parts = [
            f"merged {len(self.merged)}/{len(self.plan.merges)}",
            f"applied {len(self.applied)}/{len(self.plan.transitions)} transition(s)",
        ]
        if self.failures:
            parts.append(
                f"{len(self.failures)} failed: " + "; ".join(str(f) for f in self.failures)
            )
        if self.undeleted:
            parts.append(
                f"could not delete {', '.join(self.undeleted)} - the client has no "
                + " or ".join(BRANCH_METHODS)
            )
        if self.uncommented:
            names = ", ".join(f"#{n}" for n in self.uncommented)
            parts.append(f"could not comment on {names} - the client has no {COMMENT_METHOD}")
        return ", ".join(parts)


def delete_branch(client: Any, branch: str) -> bool:
    """Delete one branch, or say the client cannot.

    Probed, because `GitHubClient`'s surface is "the endpoints v2 actually
    calls" and nothing has needed `DELETE /git/refs` until now; `client.py` is
    outside this ticket's file set. Both spellings are accepted so whichever
    name that method eventually takes works without a change here.
    """
    for name in BRANCH_METHODS:
        deleter = getattr(client, name, None)
        if deleter is None:
            continue
        try:
            deleter(branch)
        except GitHubError as exc:
            # The branch being gone already is the desired end state, and the
            # merge has landed either way - this must not fail the outcome.
            print(f"! deleting {branch}: {exc}", file=sys.stderr)
            return False
        return True
    return False


def apply_checks(
    client: Any,
    plan: ChecksPlan,
    *,
    store: TaskStore | None = None,
    dry_run: bool = False,
) -> ChecksReport:
    """Merge, then relabel. Never raises for one issue; see `Failure`.

    **Merge before label, and only label what merged.** The reverse order would
    mark an issue `swarm:done` and then discover GitHub refused the merge - a
    lie in the ledger that nothing else in the system can detect, because
    `swarm:done` is terminal and no later cycle looks at it again. A merge that
    landed and a label that did not is the survivable direction: the PR closed
    the issue through `Closes #<n>`, and #22's first rule reads a closed issue as
    `swarm:done` on the next cycle.

    Within a transition the order is `docs/issue-contract.md` §5 and
    `reconcile.apply_plan`'s: the judgment is recorded, then the counter and the
    retry feedback go out as one body `PATCH`, then the new label, then the old
    one. One patch rather than two because both are edits to the same body, and
    a second read between them is a window for a human's edit to be lost.

    `store` is where the judgment that goes with the counter is recorded (#159).
    `None` records none - for a caller exercising the label half alone - and
    `Reconciler` always passes one, because an attempt consumed without its
    signature recorded is an attempt that renews somebody's budget for free.

    **A merge that matches no outcome raises, before anything is merged (#181).**
    The refusal path is what makes "only label what merged" true: a merge GitHub
    turns down is collected into `refused` and its outcome's transition is
    skipped. That join is filed under the merge's task and read under the
    outcome's, and a merge whose ref no outcome answers to would be *filed under
    a key nobody looks up* - so the refusal is lost, the transition is applied,
    and `swarm:done` is written for a merge that did not happen. That is the one
    direction this function's own docstring calls a lie the system cannot
    detect, so it is not defaulted (#174's rule) and it is not reported as a
    `Failure` either, because a `Failure` is a thing that went wrong with *one
    issue* and this is the plan and the gate disagreeing about what the issues
    are.

    Checked here rather than at the join because here nothing has happened yet:
    `plan.merges` is derived from `plan.outcomes`, so both sides exist before
    the first API call, and `Reconciler.cycle`'s reason for catching
    `UnresolvedJoin` - that no merge was issued and no label of this gate's was
    written - stays true. The dry run is checked too: a plan this gate would
    refuse to apply is not a plan a dry run should call fine.
    """
    unresolved = sorted(
        str(ref)
        for ref in {merge.ref for merge in plan.merges}
        - {outcome.ref for outcome in plan.outcomes}
    )
    if unresolved:
        raise UnresolvedJoin(
            f"merge(s) for {unresolved} match no outcome in the plan that carries them. "
            f"Every merge is built onto one of these outcomes, so this is the two halves "
            f"having drifted apart - and the miss cannot be defaulted: a refusal filed "
            f"under a ref nothing reads is a swarm:done written for a merge GitHub turned "
            f"down. Outcomes: "
            f"{render_keys(str(outcome.ref) for outcome in plan.outcomes)}"
        )

    merged: list[int] = []
    applied: list[Transition] = []
    deleted: list[str] = []
    undeleted: list[str] = []
    failures: list[Failure] = []
    uncommented: list[int] = []
    merge_commits: dict[int, str] = {}

    if dry_run:
        return ChecksReport(plan=plan)

    # The merges GitHub turned down, by task. Ref-keyed rather than int-keyed
    # (#181) so the join below is a task against a set of tasks: the guard above
    # is what makes a miss impossible rather than merely unlikely, and
    # `strict_equality` is what stops either half drifting back to a number
    # without mypy saying so - `outcome.number in refused` is a
    # `comparison-overlap` error the moment somebody writes it.
    refused: set[TaskRef] = set()
    for merge in plan.merges:
        try:
            answer = client.merge_pull_request(
                # Un-minted here and nowhere earlier: this is the endpoint that
                # takes `{n}` in its path, and #185's point is that the
                # hand-back is one written-out call at the API boundary.
                pull_number(merge.pull),
                merge_method=merge.merge_method,
                sha=merge.sha or None,
                commit_title=merge.commit_title or f"swarm: issue #{merge.number}",
            )
        except GitHubError as exc:
            # A 405 (not mergeable), a 409 (the head moved since the checks were
            # read) or a ruleset this override does not in fact bypass. The
            # issue keeps `swarm:review` and the next cycle re-reads the checks.
            failures.append(Failure(merge.number, f"merging PR {merge.pull}: {exc}"))
            refused.add(merge.ref)
            continue
        merged.append(merge.number)
        # Whatever the merge answered, if it answered anything with a `sha`.
        # Guarded rather than indexed: `merge_pull_request` is typed `Any`, a
        # body-less response is a real answer, and a merge that *landed* must
        # not be reported as a failure because the commit it produced could not
        # be read back.
        if isinstance(answer, Mapping):
            commit = str(answer.get("sha") or "")
            if commit:
                merge_commits[merge.number] = commit
        if not merge.delete_branch:
            continue
        (deleted if delete_branch(client, merge.branch) else undeleted).append(merge.branch)

    for outcome in plan.outcomes:
        transition = outcome.transition
        if transition is None or outcome.ref in refused:
            continue
        try:
            if transition.attempt is not None:
                record_judgement(
                    store,
                    transition.ref,
                    transition.attempt,
                    blocker=transition.blocker,
                    streak=transition.streak,
                    renewals=transition.renewals,
                )
            if transition.attempt is not None and transition.task_id:
                # The counter, and nothing else. The failure text used to ride
                # along in a delimited block; #152 removed it - nothing read it
                # (`read_feedback` was exported for a worker call site never
                # written, and the worker reads comments). It travels as the
                # transition's `comment` since #248, posted below, which is where
                # `fetch_feedback` looks.
                bump_attempt(
                    client, issue_number(transition.ref), transition.task_id, transition.attempt
                )
            # One writer for the whole transition path (#152): the label names
            # are `reconcile.write_labels`'s business and not this module's, and
            # three copies of add-before-remove were three places to find when
            # the labels go.
            write_labels(client, transition)
        except (GitHubError, StoreError) as exc:
            failures.append(Failure(outcome.number, f"{transition.to_state}: {exc}"))
            continue
        applied.append(transition)
        if transition.comment and not post_comment(client, outcome.number, transition.comment):
            uncommented.append(outcome.number)

    return ChecksReport(
        plan=plan,
        merged=tuple(merged),
        applied=tuple(applied),
        deleted=tuple(deleted),
        failures=tuple(failures),
        undeleted=tuple(dict.fromkeys(undeleted)),
        uncommented=tuple(dict.fromkeys(uncommented)),
        merge_commits=tuple(merge_commits.items()),
    )


# --------------------------------------------------------------------------
# One pass
# --------------------------------------------------------------------------


def run_checks(
    client: Any,
    ledger: Ledger,
    *,
    policy: MergePolicy | None = None,
    max_attempts: int = SETTINGS.max_attempts_per_task,
    now: dt.datetime | None = None,
    store: TaskStore | None = None,
    dry_run: bool = False,
) -> ChecksReport:
    """Read, decide, write. The whole module in one call.

    Called with the ledger a cycle already read, so it costs one pull-request
    listing plus one check-run read per `swarm:review` issue and nothing else.
    Wiring it into `Reconciler.cycle` - after `apply_plan`, before readiness, so
    a merge frees its cap slot in the same cycle - is a change to
    `orchestrator/reconcile.py`, which is outside this ticket's file set.
    """
    pulls = read_pulls(client)
    # Keyed on the ref and built from exactly the entries `plan_checks` selects:
    # the two loops are the same loop, and `plan_checks` raises rather than
    # defaulting if they ever stop being.
    checks: dict[TaskRef, CheckSet] = {}
    if pulls is not None:
        for entry in ledger.entries.values():
            pull = pulls.get(entry.branch) if in_review(entry) else None
            if pull is not None:
                checks[entry.ref] = read_checks(client, pull.ref)
    plan = plan_checks(
        ledger,
        pulls=pulls,
        checks=checks,
        policy=policy,
        max_attempts=max_attempts,
        now=now,
    )
    return apply_checks(client, plan, store=store, dry_run=dry_run)


if __name__ == "__main__":  # pragma: no cover - manual dry run, see module docstring
    repo = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GITHUB_REPOSITORY", "")
    # Read-only on every path: no merge, no label, no comment, no adoption.
    gh = GitHubClient.from_env(repo)
    rules = MergePolicy.from_env()
    print(rules.summary())
    dry = run_checks(gh, load_ledger(gh, adopt=False), policy=rules, dry_run=True)
    for row in dry.plan.outcomes:
        print(f"  {row}")
    for planned in dry.plan.merges:
        print(f"would {planned}")
    for moved in dry.plan.transitions:
        print(f"would {moved}")
    print(dry.plan.summary())
