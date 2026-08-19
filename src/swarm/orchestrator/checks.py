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
The failure output is therefore persisted onto the issue *before* the label goes
back to `swarm:ready`, in a delimited block at the end of the body
(`write_feedback`), because the body is the only channel that survives the
container, the orchestrator process and the machine. `read_feedback` is the
other half and is exported for a caller this ticket cannot write: the worker
builds its prompt from `## Goal` plus the declared files
(`worker/entrypoint.run_worker`), so a retry sees the block only once that call
site passes it in, and `worker/entrypoint.py` is outside this ticket's file set.
Until it does, the block is still what a human reads to find out why an issue is
on its third attempt.

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
`swarm/issue-<n>` branch survives its merge and is reported). `client.py` is
outside this ticket's file set; both gaps degrade and neither is silent.

Manual dry run against a real repo - reads only, merges nothing, writes nothing:

    GITHUB_TOKEN=... python -m swarm.orchestrator.checks shahrestani-me/apiary
"""

from __future__ import annotations

import datetime as dt
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..config import SETTINGS
from ..github.client import GitHubClient, GitHubError
from ..github.ledger import Ledger, LedgerEntry, load_ledger
from ..github.readiness import READY
from ..github.refs import issue_number
from ..worker.result import tail
from .dispatcher import REVIEW, normalise
from .reconcile import (
    COMMENT_METHOD,
    DONE,
    FAILED,
    PULLS_METHOD,
    Transition,
    post_comment,
    rewrite_marker,
)

#: The client methods this module probes for. Named because the probe and the
#: sentence explaining what is missing must not drift, and because grepping for
#: either name should land on this line. `PULLS_METHOD` is imported from #22 for
#: the same reason it spells `READY` rather than restating it.
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

#: The delimiters of the retry-feedback block in an issue body. HTML comments,
#: for `docs/issue-contract.md` §2's reason for putting identity in one: they
#: are invisible in rendered markdown, survive edits above and below them, and
#: are preserved verbatim by GitHub's editor.
FEEDBACK_OPEN = "<!-- apiary:ci-failure -->"
FEEDBACK_CLOSE = "<!-- /apiary:ci-failure -->"

#: Every line of quoted CI output is indented by this much. Not decoration: a
#: check log containing `## Verify` at column 0 would otherwise add a second
#: `## Verify` section to the body and make the issue malformed
#: (`docs/issue-contract.md` §1.1), and the parser's heading anchor allows no
#: leading whitespace. Indenting is also what makes the block render as code
#: without a fence the log could itself close.
QUOTE_INDENT = "    "

#: How much of a failure to carry. The same bound the worker's own record uses
#: (`worker/result.tail`), because this text lands in an issue body that a human
#: reads and a model is meant to be given, and a megabyte of CI log is neither.
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
        value = str(output.get(key) or "").strip()
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

    number: int
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
            number=int(payload.get("number") or 0),
            branch=str(head.get("ref") or ""),
            sha=str(head.get("sha") or ""),
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

    number: int
    pull: int
    branch: str
    sha: str = ""
    merge_method: str = "squash"
    delete_branch: bool = True
    admin_override: bool = True
    commit_title: str = ""

    def __str__(self) -> str:
        how = "admin override" if self.admin_override else "no override"
        after = f", deleting {self.branch}" if self.delete_branch else ""
        return f"#{self.number}: merge PR #{self.pull} ({self.merge_method}, {how}{after})"


@dataclass(frozen=True)
class Outcome:
    """What one `swarm:review` issue's checks mean for it. One row per issue.

    A row rather than four parallel lists, because every interesting assertion -
    and every interesting log line - is about one issue: what its checks said,
    what label that moves, whether a human is now needed and what the next
    attempt must be told.
    """

    #: The issue this row is about. Still a number, where `Transition.ref` -
    #: the field right below - is a `TaskRef`: #142 retyped the *task-identity
    #: model* (the dependency graph, readiness and the reconciler's transition)
    #: and stopped there deliberately, so these policy rows are one vocabulary
    #: behind. They are internally consistent - built from the same `entry`,
    #: compared only against each other - and nothing keys a `TaskRef` map on
    #: one. Moving them is a follow-up, not an oversight.
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
    checks: Mapping[int, CheckSet],
    policy: MergePolicy | None = None,
    max_attempts: int = SETTINGS.max_attempts_per_task,
    now: dt.datetime | None = None,
) -> ChecksPlan:
    """Decide every `swarm:review` issue. Pure - no API call, no daemon, no model.

    `pulls` is the open pull requests by head ref (`None` when this cycle could
    not look), `checks` the folded check set per issue number, and `now` the
    moment the grace period is measured against. All three are facts somebody
    else read; keeping the I/O out of here is what makes the rules assertable.

    Issues with no open PR are skipped rather than decided: that is #22's row -
    it reads a `swarm:review` issue whose branch has no open PR as a PR closed
    unmerged - and two modules writing one transition is how a label ends up
    moving twice in one cycle.
    """
    rules = policy or MergePolicy()
    moment = now or dt.datetime.now(dt.timezone.utc)
    outcomes: list[Outcome] = []

    for entry in sorted(ledger.entries.values(), key=lambda entry: entry.ref):
        if entry.state_label != REVIEW or pulls is None:
            continue
        pull = pulls.get(entry.branch)
        if pull is None:
            continue
        outcomes.append(
            _decide(entry, pull, checks.get(entry.number, CheckSet()), rules, max_attempts, moment)
        )

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
                from_label=REVIEW,
                to_label=FAILED,
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
            from_label=REVIEW,
            to_label=FAILED,
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
            f"checks passed - waiting for a human to merge PR #{pull.number} "
            f"({ADMIN_OVERRIDE_ENV} is off)",
        )
    return Outcome(
        number=entry.number,
        verdict=PASSED,
        detail=f"{checks.summary()}; merging PR #{pull.number}",
        transition=Transition(
            ref=entry.ref,
            from_label=REVIEW,
            to_label=DONE,
            reason=f"PR #{pull.number} merged: {checks.summary()}",
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
                from_label=REVIEW,
                to_label=FAILED,
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
    return Outcome(
        number=entry.number,
        verdict=FAILING,
        detail=f"{named} failed; retrying as attempt {attempt} of {cap}",
        transition=Transition(
            ref=entry.ref,
            from_label=REVIEW,
            to_label=READY,
            reason=f"{named} failed on the pull request",
            task_id=entry.task_id,
            attempt=attempt,
        ),
        feedback=feedback,
    )


# --------------------------------------------------------------------------
# The retry's context
# --------------------------------------------------------------------------


def _quote(text: str) -> str:
    """CI output, indented so no line of it can be markdown or a section heading.

    See `QUOTE_INDENT`. Applied everywhere the output is written - the body
    block and both comments - because the hazard is the same in all three: a log
    line reading `## Verify` at column 0 makes the issue malformed, and the
    module that reports a CI failure must not be the one that corrupts the
    contract while doing it.
    """
    body = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not body:
        return f"{QUOTE_INDENT}(no output)"
    return "\n".join(f"{QUOTE_INDENT}{line}".rstrip() for line in body.split("\n"))


def write_feedback(body: str, text: str, *, attempt: int) -> str:
    """Put the failure in the issue body, replacing any block already there.

    Replacing rather than appending: three attempts would otherwise leave three
    logs in one body, and the only one that helps the next attempt is the last.
    The previous attempt's output is not lost - `worker/result.py` keeps one
    record per attempt in the run's artifacts, which is where the history
    belongs.

    Every byte outside the block is returned untouched, which is
    `reconcile.rewrite_marker`'s rule and load-bearing for the same reason: a
    human editing prose while the orchestrator records a CI failure must not
    lose their edit.
    """
    block = "\n".join(
        [
            FEEDBACK_OPEN,
            f"**apiary: CI failed on attempt {int(attempt)}.** The next attempt should fix this:",
            "",
            _quote(tail(text, FEEDBACK_CHARS)),
            FEEDBACK_CLOSE,
        ]
    )
    stripped = _strip_feedback(body)
    return f"{stripped}\n\n{block}\n" if stripped else f"{block}\n"


def read_feedback(body: str) -> str:
    """The quoted CI output of the last attempt, undented, or `""`.

    The half of this channel this ticket cannot call: the worker builds its
    prompt from `## Goal` and the declared files
    (`worker/entrypoint.run_worker`), so until that call site passes this in, a
    retry does not see it. Exported, named and tested so that wiring it is one
    line in a file this ticket's `## Files` does not include.
    """
    text = body or ""
    start = text.find(FEEDBACK_OPEN)
    end = text.find(FEEDBACK_CLOSE, start + 1)
    if start < 0 or end < 0:
        return ""
    inner = text[start + len(FEEDBACK_OPEN) : end]
    # Only the indented lines are the log. The sentence above them is this
    # module's framing, and a consumer wants what CI said, not what apiary said
    # about it.
    lines = [
        line[len(QUOTE_INDENT) :] for line in inner.split("\n") if line.startswith(QUOTE_INDENT)
    ]
    return "\n".join(lines).strip("\n")


def _strip_feedback(body: str) -> str:
    """The body without its block, so `write_feedback` can put a fresh one back."""
    text = body or ""
    start = text.find(FEEDBACK_OPEN)
    if start < 0:
        return text.rstrip("\n")
    end = text.find(FEEDBACK_CLOSE, start)
    if end < 0:
        # An opener with no closer is a body somebody edited by hand mid-block.
        # Truncating from the opener is the reading that cannot leave a stray
        # closer behind to swallow the next block's content.
        return text[:start].rstrip("\n")
    return (text[:start] + text[end + len(FEEDBACK_CLOSE) :]).rstrip("\n")


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
    #: The commit each merge produced, by issue number - GitHub's answer to the
    #: `PUT .../merge`, which this module previously threw away. Announcement
    #: only: nothing here reads it, and `pr.merged` (#141) is the only consumer,
    #: because "which commit is this task now" is the one fact about a landed
    #: task that the run directory could not otherwise recover. Absent for a
    #: client whose merge returns no `sha`, which the tests' doubles do.
    merge_commits: Mapping[int, str] = field(default_factory=dict)

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
    `reconcile.apply_plan`'s: the counter and the retry feedback are one body
    `PATCH` written first, then the new label, then the old one. One patch
    rather than two because both are edits to the same body, and a second read
    between them is a window for a human's edit to be lost.
    """
    merged: list[int] = []
    applied: list[Transition] = []
    deleted: list[str] = []
    undeleted: list[str] = []
    failures: list[Failure] = []
    uncommented: list[int] = []
    merge_commits: dict[int, str] = {}

    if dry_run:
        return ChecksReport(plan=plan)

    refused: set[int] = set()
    for merge in plan.merges:
        try:
            answer = client.merge_pull_request(
                merge.pull,
                merge_method=merge.merge_method,
                sha=merge.sha or None,
                commit_title=merge.commit_title or f"swarm: issue #{merge.number}",
            )
        except GitHubError as exc:
            # A 405 (not mergeable), a 409 (the head moved since the checks were
            # read) or a ruleset this override does not in fact bypass. The
            # issue keeps `swarm:review` and the next cycle re-reads the checks.
            failures.append(Failure(merge.number, f"merging PR #{merge.pull}: {exc}"))
            refused.add(merge.number)
            continue
        merged.append(merge.number)
        # Whatever the merge answered, if it answered anything with a `sha`.
        # Guarded rather than indexed: the doubles in the tests return `None`,
        # and a merge that landed must not be reported as a failure because the
        # commit it produced could not be read back.
        if isinstance(answer, Mapping):
            commit = str(answer.get("sha") or "")
            if commit:
                merge_commits[merge.number] = commit
        if not merge.delete_branch:
            continue
        (deleted if delete_branch(client, merge.branch) else undeleted).append(merge.branch)

    for outcome in plan.outcomes:
        transition = outcome.transition
        if transition is None or outcome.number in refused:
            continue
        try:
            if transition.attempt is not None or outcome.feedback:
                _patch_body(client, transition, outcome.feedback)
            # `Transition` is keyed on the task, and every call here addresses
            # the GitHub API, which takes issue numbers.
            number = issue_number(transition.ref)
            client.add_labels(number, [transition.to_label])
            if transition.from_label and transition.from_label != transition.to_label:
                client.remove_label(number, transition.from_label)
        except GitHubError as exc:
            failures.append(Failure(outcome.number, f"{transition.to_label}: {exc}"))
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
        merge_commits=merge_commits,
    )


def _patch_body(client: Any, transition: Transition, feedback: str) -> None:
    """One `PATCH`, carrying the counter and the failure the next attempt needs.

    The body is re-read immediately before the write, which §5 requires and
    which is cheap because it happens only for an issue that just finished an
    attempt: the copy in the ledger was fetched at the top of the cycle, and a
    human editing in between would have their edit overwritten by a patch built
    from the stale one.

    `rewrite_marker` is #22's, imported rather than reimplemented: two modules
    with their own idea of where the counter lives is how a counter stops
    bounding anything.
    """
    number = issue_number(transition.ref)
    issue = client.get_issue(number)
    body = issue.get("body") or ""
    if transition.attempt is not None and transition.task_id:
        body = rewrite_marker(body, transition.task_id, transition.attempt)
    if feedback:
        body = write_feedback(body, feedback, attempt=transition.attempt or 0)
    client.update_issue(number, body=body)


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
    checks: dict[int, CheckSet] = {}
    if pulls is not None:
        for entry in ledger.entries.values():
            pull = pulls.get(entry.branch) if entry.state_label == REVIEW else None
            if pull is not None:
                checks[entry.number] = read_checks(client, pull.ref)
    plan = plan_checks(
        ledger,
        pulls=pulls,
        checks=checks,
        policy=policy,
        max_attempts=max_attempts,
        now=now,
    )
    return apply_checks(client, plan, dry_run=dry_run)


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
