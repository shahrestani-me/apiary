"""Whether a green pull request is green about the base it is going to land on.

`checks.py` (#23) answers "did CI pass on this head". This module answers the
question that one cannot: **passed against what**. They are siblings by design -
the check gate decides a pull request on its own merits, this one decides it
against a base branch that other workers are moving underneath it - and the two
answers are combined here rather than in either module's own rules.

**The target repository's ruleset sets `strict_required_status_checks_policy`,
so every merge invalidates every other open pull request.** Not a hypothetical:
with N swarm PRs green at once, merging one leaves the other N-1 with checks that
passed on a commit that is no longer the base, and GitHub refuses them until
their branches are updated and re-checked. Nothing in #23 notices, because a
stale PR's checks are still `success` and its issue still carries
`swarm:review` - the queue looks healthy while it is stuck.

**The reason nobody has seen this yet is `--admin`, and that is not evidence it
cannot happen.** `checks.MergePolicy.admin_override` bypasses the up-to-date
requirement along with the review requirement, so a scheduler with admin rights
merges straight through staleness and never observes it. Turn the override off,
or run as an account without admin rights on the repo, and the second concurrent
PR hits the wall on its first cycle. A gate that only works for admins is not a
gate.

**#21's non-overlapping `## Files` rule does not help.** Staleness is
repo-wide, not file-scoped: two workers editing disjoint files still produce two
pull requests, only one of which can merge before the other needs updating. What
#21 does buy is that a re-dispatch caused by a conflict cannot collide with a
sibling that is still running, because a `swarm:review` issue still reserves its
files.

Four rules follow, and each of them is a way a run stalls rather than a way it
computes wrong:

- **Behind the base is not mergeable, however green.** The branch is updated
  from the base and the pull request goes back to waiting for checks - a fresh
  head means a fresh check run, which `checks.py` reads next cycle. Admitting the
  merge instead would be merging a result nothing verified against this base.
- **Conflicted is not a CI failure and retrying the same diff cannot fix it.**
  The attempt counter is bumped and the issue goes back to `swarm:ready`, which
  re-dispatches it from a fresh base commit: the dispatcher (#21) spawns a worker
  against the current base head, so "start from a fresh base" falls out of
  returning the issue to the queue rather than needing a mechanism of its own.
  What the next attempt needs and would not otherwise have is *why* - so the base
  it conflicted with, and the files involved, are written onto the issue body
  before the label moves (`write_conflict`), exactly as `checks.write_feedback`
  persists a CI failure.
- **Merges are serialised.** Merging N green pull requests in one cycle under a
  strict policy means N-1 immediately go stale; merge one, let the rest update,
  repeat. It costs a cycle per merge and buys a queue that drains.
- **The update-and-recheck loop is capped per pull request.** An unlucky branch
  can otherwise be invalidated indefinitely by faster siblings - it updates, a
  sibling merges, it is behind again - and spend a run's whole budget making no
  progress. That is starvation, it looks exactly like health from the outside,
  and after `UpdatePolicy.max_update_rounds` rounds the issue goes to a human
  with `swarm:failed` saying which. **Failing loudly is the point**: the
  alternative is an issue that sits in `swarm:review` for the rest of the run.

**Two things this module needs and cannot have**, probed for rather than added,
exactly as `checks.py` probes for `list_pull_requests` and a ref deletion:
`GitHubClient` has no way to update a pull request branch (`PUT
/pulls/{n}/update-branch`) and no way to list the files a pull request touches.
`client.py` is outside this ticket's file set. Both gaps degrade and neither is
silent - a branch this client cannot update still spends its round, so a
repository that never wires the method up gets a loud `swarm:failed` naming the
missing method rather than a review queue that never drains.

Wiring this in front of `checks.apply_checks` - one call, taking the plan #23
computed and returning the subset that may actually land - is a change to
`orchestrator/reconcile.py`, which is outside this ticket's file set too.

Manual dry run against a real repo - reads only, updates nothing, merges
nothing, writes nothing:

    GITHUB_TOKEN=... python -m swarm.orchestrator.mergeability shahrestani-me/apiary
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from ..config import SETTINGS
from ..github.client import GitHubClient, GitHubError
from ..github.ledger import Ledger, LedgerEntry, load_ledger
from ..github.readiness import READY
from ..github.refs import issue_number
from ..worker.result import tail
from .checks import (
    QUOTE_INDENT,
    ChecksPlan,
    Merge,
    Outcome,
    PullState,
    read_pulls,
)
from .dispatcher import REVIEW
from .reconcile import (
    COMMENT_METHOD,
    FAILED,
    Transition,
    post_comment,
    rewrite_marker,
)

#: The client methods this module probes for. Named because the probe and the
#: sentence explaining what is missing must not drift, and because grepping for
#: either name should land on this line. Both spellings of the update are
#: accepted so whichever one `client.py` eventually grows works without a change
#: here, which is the same courtesy `checks.BRANCH_METHODS` extends.
UPDATE_METHODS: tuple[str, ...] = ("update_branch", "update_pull_request_branch")
FILES_METHODS: tuple[str, ...] = ("list_pull_request_files",)

#: The four answers a pull request's mergeability has. Strings rather than an
#: enum for `checks.py`'s reason: they are printed, logged and asserted on far
#: more often than they are matched.
FRESH = "fresh"
BEHIND = "behind"
CONFLICTED = "conflicted"
COMPUTING = "computing"

#: GitHub's `mergeable_state`, folded. Only these three values are this module's
#: business. `blocked` (a required check or review is missing) and `unstable` (a
#: non-required check failed) are `checks.py`'s question and are `FRESH` here -
#: two modules with an opinion about CI is how an issue gets relabelled twice in
#: one cycle. Anything GitHub adds later lands in `FRESH` and is decided by the
#: check gate, which is the direction that cannot invent a stall.
DIRTY_STATES = frozenset({"dirty"})
BEHIND_STATES = frozenset({"behind"})
UNSETTLED_STATES = frozenset({"", "unknown"})

#: How many times one pull request may be updated and re-checked before the
#: starvation is called. Three is "the base moved under you three times": a busy
#: swarm produces one or two legitimately, and a branch that cannot win in three
#: is not going to win in thirty. Not in `Settings` - `config.py` is outside this
#: ticket's file set, and the number belongs to this decision.
DEFAULT_MAX_UPDATE_ROUNDS = 3

#: How many pull requests may be merged per cycle. One, because of the strict
#: policy in the module docstring. Configurable because a repository *without*
#: that policy pays a cycle per merge for nothing.
DEFAULT_MERGES_PER_CYCLE = 1

ROUNDS_ENV = "APIARY_MAX_UPDATE_ROUNDS"
MERGES_PER_CYCLE_ENV = "APIARY_MERGES_PER_CYCLE"

#: The delimiters of the conflict block in an issue body, and the sibling of
#: `checks.FEEDBACK_OPEN`. A separate block rather than the same one because the
#: two say different things to the next attempt - "your code failed a test" and
#: "your branch no longer applies" need different responses - and a reader that
#: found one sentence where the other belonged would act on the wrong one.
CONFLICT_OPEN = "<!-- apiary:conflict -->"
CONFLICT_CLOSE = "<!-- /apiary:conflict -->"

#: How much conflict detail to carry, matching `checks.FEEDBACK_CHARS`: this text
#: lands in an issue body a human reads and a model is meant to be given, and a
#: pull request touching four hundred files is neither.
CONTEXT_CHARS = 4000


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    """Loud on garbage, like `dispatcher._env_int` and `checks._env_flag`.

    A mistyped cap that silently fell back to the default would be a starvation
    bound nobody set, which is the one number here whose whole purpose is to be
    an upper bound somebody chose.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not an integer") from None
    if value < minimum:
        raise ValueError(f"{name}={raw!r} is below the minimum of {minimum}")
    return value


@dataclass(frozen=True)
class UpdatePolicy:
    """How hard this cycle tries to keep a pull request current, and for how long.

    Both fields exist because a target repository could reasonably want the other
    value, and the defaults are what a repository with a strict status policy
    needs: one merge at a time, and a bounded number of update rounds behind it.
    """

    max_update_rounds: int = DEFAULT_MAX_UPDATE_ROUNDS
    merges_per_cycle: int = DEFAULT_MERGES_PER_CYCLE

    @classmethod
    def from_env(cls) -> UpdatePolicy:
        """The policy this process was started with. Read once, at the call site."""
        return cls(
            max_update_rounds=_env_int(ROUNDS_ENV, DEFAULT_MAX_UPDATE_ROUNDS),
            merges_per_cycle=_env_int(MERGES_PER_CYCLE_ENV, DEFAULT_MERGES_PER_CYCLE),
        )

    def summary(self) -> str:
        """The line worth logging at startup, because it names what is bounded."""
        return (
            f"mergeability policy: {self.merges_per_cycle} merge(s) per cycle, at most "
            f"{self.max_update_rounds} update round(s) per pull request before a human "
            f"is asked ({ROUNDS_ENV}, {MERGES_PER_CYCLE_ENV})"
        )


@dataclass
class UpdateBudget:
    """How many times each pull request has already been dragged forward.

    Mutable and held by the caller across cycles, the way the reconciler holds
    its container handles: the count only means anything as a run-scoped
    sequence, and there is nowhere cheap to persist it - the identity marker
    carries one counter (`docs/issue-contract.md` §5) and it is the attempt
    counter, which updating deliberately does not consume.

    So a restarted orchestrator grants a fresh budget. That is the safe
    direction and the same one §5 chose for over-counting: the cost of forgetting
    is a few more update rounds for a branch a human just chose to keep working
    on, and the cost of the opposite would be a fresh run inheriting a cap it
    cannot see and failing a healthy pull request on its first cycle.
    """

    cap: int = DEFAULT_MAX_UPDATE_ROUNDS
    rounds: dict[int, int] = field(default_factory=dict)

    def spent(self, number: int) -> int:
        return self.rounds.get(number, 0)

    def spend(self, number: int) -> int:
        """Charge one round to an issue. Returns the new total."""
        self.rounds[number] = self.spent(number) + 1
        return self.rounds[number]

    def exhausted(self, number: int) -> bool:
        return self.spent(number) >= max(int(self.cap), 1)

    def clear(self, number: int) -> None:
        """Forget an issue: it merged, or it is going round again as new work."""
        self.rounds.pop(number, None)

    def retain(self, numbers: Iterable[int]) -> None:
        """Drop every issue not in this cycle's review queue.

        Without it a long run accumulates a count per task it ever opened, and -
        worse - an issue re-dispatched and re-reviewed would inherit the rounds
        its previous pull request spent.
        """
        keep = set(numbers)
        for number in [n for n in self.rounds if n not in keep]:
            del self.rounds[number]

    def summary(self) -> str:
        if not self.rounds:
            return "no pull request has needed updating"
        spent = ", ".join(f"#{n}x{c}" for n, c in sorted(self.rounds.items()))
        return f"update rounds (cap {self.cap}): {spent}"


# --------------------------------------------------------------------------
# What GitHub says about the merge
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Mergeability:
    """One pull request's standing against its base, as GitHub computes it.

    `mergeable` and `mergeable_state` are **not** in the pull request *listing* -
    GitHub only computes them for a single-PR read, and returns `null` on the
    first one while a background job runs. So this costs one `get_pull_request`
    per `swarm:review` issue per cycle, and `COMPUTING` is a real answer rather
    than an error path.
    """

    number: int
    branch: str = ""
    mergeable: bool | None = None
    #: GitHub's `mergeable_state`, lowercased and otherwise verbatim. Kept raw
    #: alongside the folded verdict because a state this module has not seen is
    #: something a human should be able to read off the log line.
    state: str = ""
    base: str = ""
    base_sha: str = ""
    head_sha: str = ""
    #: True when the pull request could not be read at all this cycle. Distinct
    #: from `COMPUTING` for `checks.CheckSet.unreadable`'s reason: "we did not
    #: look" and "GitHub has not finished looking" print differently, even though
    #: both wait a cycle.
    unreadable: bool = False

    @property
    def verdict(self) -> str:
        """`computing` first, then conflict, then staleness; everything else is fresh.

        The order is the safety argument. A `mergeable` of `null` means the
        answer is not ready, and reading it as "not mergeable" would re-dispatch
        a healthy pull request every time GitHub was slow - so nothing is decided
        until GitHub has decided.
        """
        if self.unreadable:
            return COMPUTING
        if self.mergeable is None or self.state in UNSETTLED_STATES:
            return COMPUTING
        if self.state in DIRTY_STATES or self.mergeable is False:
            return CONFLICTED
        if self.state in BEHIND_STATES:
            return BEHIND
        return FRESH

    @property
    def base_name(self) -> str:
        """What to call the base in a sentence a human reads."""
        if self.base and self.base_sha:
            return f"{self.base}@{self.base_sha[:7]}"
        return self.base or self.base_sha[:7] or "the base branch"

    def summary(self) -> str:
        if self.unreadable:
            return "the pull request could not be read"
        state = self.state or "no state"
        return f"mergeable={self.mergeable}, mergeable_state={state}"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Mergeability:
        head = payload.get("head") if isinstance(payload.get("head"), Mapping) else {}
        base = payload.get("base") if isinstance(payload.get("base"), Mapping) else {}
        raw = payload.get("mergeable")
        return cls(
            number=int(payload.get("number") or 0),
            branch=str(head.get("ref") or ""),  # type: ignore[union-attr]
            mergeable=raw if isinstance(raw, bool) else None,
            state=str(payload.get("mergeable_state") or "").lower(),
            base=str(base.get("ref") or ""),  # type: ignore[union-attr]
            base_sha=str(base.get("sha") or ""),  # type: ignore[union-attr]
            head_sha=str(head.get("sha") or ""),  # type: ignore[union-attr]
        )


def read_mergeability(client: Any, pull: PullState) -> Mergeability:
    """One pull request's standing. A read that failed is `unreadable`, not fresh.

    `get_pull_request` is the one endpoint here that `GitHubClient` already has,
    which is why reading is never the degraded path in this module and updating
    always is.
    """
    try:
        return Mergeability.from_payload(client.get_pull_request(pull.number))
    except GitHubError as exc:
        print(f"! PR #{pull.number} could not be read: {exc}", file=sys.stderr)
        return Mergeability(number=pull.number, branch=pull.branch, unreadable=True)


def read_touched_files(client: Any, pull: PullState) -> tuple[str, ...]:
    """The paths a pull request touches, or `()` when this client cannot say.

    Probed rather than added (module docstring). Empty is a supported answer: the
    conflict message degrades to naming the issue's declared `## Files`, which is
    the set the worker may edit anyway, and a conflict report without a file list
    is still a conflict report.
    """
    for name in FILES_METHODS:
        lister = getattr(client, name, None)
        if lister is None:
            continue
        try:
            payloads = lister(pull.number) or ()
        except GitHubError as exc:
            print(f"! files for PR #{pull.number}: {exc}", file=sys.stderr)
            return ()
        return tuple(
            str(item.get("filename") or "")
            for item in payloads
            if isinstance(item, Mapping) and item.get("filename")
        )
    return ()


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchUpdate:
    """One pull request this cycle would drag forward onto the current base.

    `round` and `cap` ride on the record rather than being read back from the
    budget at the call site, so the log line and the decision cannot disagree
    about how close this pull request is to being given up on.
    """

    number: int
    pull: int
    branch: str
    base: str = ""
    round: int = 1
    cap: int = DEFAULT_MAX_UPDATE_ROUNDS

    def __str__(self) -> str:
        return (
            f"#{self.number}: update PR #{self.pull} ({self.branch}) from "
            f"{self.base or 'the base branch'}, round {self.round} of {self.cap}"
        )


@dataclass(frozen=True)
class Decision:
    """What one `swarm:review` pull request's standing means for it.

    One row per issue, for `checks.Outcome`'s reason: every interesting assertion
    and every interesting log line is about one issue - what GitHub said about
    its merge, whether that costs it a round, and whether its merge is still
    going ahead this cycle.
    """

    #: The issue this row is about. Still a number, where `Transition.ref` -
    #: the field right below - is a `TaskRef`: #142 retyped the *task-identity
    #: model* (the dependency graph, readiness and the reconciler's transition)
    #: and stopped there deliberately, so these policy rows are one vocabulary
    #: behind. They are internally consistent - built from the same `entry`,
    #: compared only against each other - and nothing keys a `TaskRef` map on
    #: one. Moving them is a follow-up, not an oversight.
    number: int
    pull: int
    verdict: str
    detail: str = ""
    update: BranchUpdate | None = None
    transition: Transition | None = None
    #: Why the merge #23 planned is not happening this cycle, or `""` when there
    #: was no merge to hold. A held merge is not a failure and must not read as
    #: one: the pull request is fine, it is just not this cycle's.
    held: str = ""
    #: The conflict detail the next attempt needs, persisted onto the issue body
    #: before the label goes back to `swarm:ready`. Empty for every decision that
    #: is not a re-dispatch.
    context: str = ""

    def __str__(self) -> str:
        return f"#{self.number}: {self.verdict} - {self.detail}"


@dataclass(frozen=True)
class MergeabilityPlan:
    """Everything one pass would change, computed without writing anything.

    Pure, so every case that matters - a green pull request behind its base, two
    green ones racing, a conflicted one at its attempt cap, a starved one at its
    round cap - is testable as data rather than against a repository whose base
    somebody has to move by hand.
    """

    decisions: tuple[Decision, ...] = ()
    #: #23's plan with every merge this cycle will not allow stripped out, ready
    #: to hand to `checks.apply_checks`. The transition is stripped with the
    #: merge: a `swarm:done` written for a merge that did not happen is a lie
    #: nothing later in the system can detect, because `done` is terminal.
    admitted: ChecksPlan = ChecksPlan()
    #: True when mergeability could not be read at all this cycle. Nothing was
    #: decided and **nothing is admitted** - `checks.ChecksPlan.blind`'s rule,
    #: applied to the one question that decides whether a merge is real.
    blind: bool = False
    policy: UpdatePolicy = UpdatePolicy()

    @property
    def updates(self) -> tuple[BranchUpdate, ...]:
        return tuple(d.update for d in self.decisions if d.update is not None)

    @property
    def transitions(self) -> tuple[Transition, ...]:
        return tuple(d.transition for d in self.decisions if d.transition is not None)

    @property
    def held(self) -> tuple[int, ...]:
        """Issues whose merge #23 planned and this cycle did not allow."""
        return tuple(d.number for d in self.decisions if d.held)

    @property
    def merges(self) -> tuple[Merge, ...]:
        """The merges that survived the gate. At most `merges_per_cycle` of them."""
        return self.admitted.merges

    @property
    def changed(self) -> bool:
        return bool(self.updates or self.transitions)

    def summary(self) -> str:
        parts = [
            f"{len(self.merges)} merge(s) admitted",
            f"{len(self.updates)} branch update(s)",
            f"{len(self.transitions)} transition(s)",
        ]
        if self.held:
            parts.append("held: " + ", ".join(f"#{n}" for n in self.held))
        if self.blind:
            parts.append("mergeability unreadable - nothing merged")
        return ", ".join(parts)


def plan_mergeability(
    ledger: Ledger,
    checks: ChecksPlan,
    *,
    states: Mapping[int, Mergeability] | None,
    budget: UpdateBudget | None = None,
    policy: UpdatePolicy | None = None,
    files: Mapping[int, Sequence[str]] | None = None,
    max_attempts: int = SETTINGS.max_attempts_per_task,
) -> MergeabilityPlan:
    """Gate #23's plan on the base branch. Pure - no API call, no daemon, no clock.

    `checks` is the plan the check gate computed this cycle, `states` each review
    issue's mergeability (`None` when this cycle could not look), `budget` the
    rounds already spent, and `files` the paths each pull request touches when
    the client could say. All of them are facts somebody else read; keeping the
    I/O out of here is what makes the rules assertable.

    The round cap is `budget.cap` - a budget carries its own, and one built here
    takes it from the policy - so a caller raising `max_update_rounds` on a run
    already in flight raises it on the budget it kept, not on this argument.

    Issues #23 already decided against - a red pull request being retried, one
    escalated to a human - are skipped rather than decided twice. A branch that
    is behind its base matters only while somebody still intends to merge it, and
    a re-dispatch already starts from a fresh base commit.
    """
    rules = policy or UpdatePolicy()
    spent = budget if budget is not None else UpdateBudget(cap=rules.max_update_rounds)
    entries = {entry.number: entry for entry in ledger.entries.values()}
    seen = states or {}
    blind = checks.blind or states is None

    decisions: list[Decision] = []
    candidates: list[tuple[int, int]] = []  # (rounds spent, issue number), for the merge slot

    for outcome in checks.outcomes:
        entry = entries.get(outcome.number)
        if entry is None:
            continue
        if outcome.transition is not None and outcome.merge is None:
            # #23 has already moved this issue this cycle - a retry, an
            # escalation, a give-up. Two modules writing one issue's label is how
            # a label moves twice in a cycle.
            continue

        decision = _decide(
            entry=entry,
            outcome=outcome,
            state=seen.get(outcome.number),
            budget=spent,
            files=tuple((files or {}).get(outcome.number) or ()),
            max_attempts=max_attempts,
            blind=blind,
        )
        decisions.append(decision)
        if outcome.merge is not None and not decision.held:
            candidates.append((spent.spent(outcome.number), outcome.number))

    decisions = _serialise(decisions, candidates, rules)
    # A cycle that merges makes every other open pull request stale the moment it
    # lands, so an update issued alongside it is work that is already undone -
    # and, worse, a round spent for nothing. The next cycle updates them against
    # the base this merge created.
    merging = {number for _, number in candidates} & {d.number for d in decisions if not d.held}
    if merging:
        decisions = [_defer(d) if d.update is not None else d for d in decisions]

    held = {d.number: d.held for d in decisions if d.held}
    return MergeabilityPlan(
        decisions=tuple(decisions),
        admitted=_admit(checks, held),
        blind=blind,
        policy=rules,
    )


def _decide(
    *,
    entry: LedgerEntry,
    outcome: Outcome,
    state: Mergeability | None,
    budget: UpdateBudget,
    files: tuple[str, ...],
    max_attempts: int,
    blind: bool,
) -> Decision:
    """One pull request's standing. The order of the branches is the priority."""
    facts = state or Mergeability(number=0, branch=entry.branch, unreadable=blind)
    verdict = COMPUTING if blind else facts.verdict
    pull = outcome.merge.pull if outcome.merge is not None else facts.number

    if verdict == CONFLICTED:
        decision = _decide_conflicted(entry, pull, facts, files, max_attempts)
    elif verdict == BEHIND:
        decision = _decide_behind(entry, pull, facts, budget)
    elif verdict == COMPUTING:
        detail = facts.summary() if state is not None else "mergeability was not read this cycle"
        # Never merged on a maybe. GitHub computes mergeability lazily and a
        # merge issued against `null` is a merge against an unknown base.
        decision = Decision(
            number=entry.number,
            pull=pull,
            verdict=COMPUTING,
            detail=detail,
            held=f"mergeability is not settled yet ({detail})",
        )
    else:
        decision = Decision(number=entry.number, pull=pull, verdict=FRESH, detail=facts.summary())

    # `held` answers "why is the merge #23 planned not happening", so it means
    # nothing when #23 planned no merge - an issue still waiting for its checks
    # is held back by nothing.
    return decision if outcome.merge is not None else replace(decision, held="")


def _decide_conflicted(
    entry: LedgerEntry,
    pull: int,
    facts: Mergeability,
    files: tuple[str, ...],
    max_attempts: int,
) -> Decision:
    """A branch git cannot merge. Re-dispatch it, or hand it over.

    The attempt counter *is* consumed here, unlike the starvation case below: a
    conflict means this attempt's diff no longer applies, so the next attempt is
    a real second try at the same task and §5's bound is exactly what should hold
    it. The counter rides on the transition so it is persisted before the label
    goes back to `swarm:ready`.
    """
    context = conflict_context(entry, facts, files)
    attempt = entry.attempt + 1
    cap = max(int(max_attempts), 1)
    held = f"the branch conflicts with {facts.base_name}"

    if attempt >= cap:
        return Decision(
            number=entry.number,
            pull=pull,
            verdict=CONFLICTED,
            detail=f"conflicts with {facts.base_name}; {attempt} attempt(s) against a cap of {cap}",
            transition=Transition(
                ref=entry.ref,
                from_label=REVIEW,
                to_label=FAILED,
                reason=(
                    f"the branch conflicts with {facts.base_name} and {attempt} attempt(s) "
                    f"have been made against a cap of {cap}"
                ),
                task_id=entry.task_id,
                attempt=attempt,
                comment=(
                    f"apiary: giving up after {attempt} attempt(s). The last one conflicts "
                    f"with the base and no further attempt is left to rebuild it.\n\n"
                    f"{quote(context)}"
                ),
            ),
            held=held,
            context=context,
        )

    return Decision(
        number=entry.number,
        pull=pull,
        verdict=CONFLICTED,
        detail=f"conflicts with {facts.base_name}; re-dispatching as attempt {attempt} of {cap}",
        transition=Transition(
            ref=entry.ref,
            from_label=REVIEW,
            to_label=READY,
            reason=(
                f"the branch conflicts with {facts.base_name}; re-dispatching from a fresh "
                f"base commit rather than retrying the same diff"
            ),
            task_id=entry.task_id,
            attempt=attempt,
        ),
        held=held,
        context=context,
    )


def _decide_behind(
    entry: LedgerEntry,
    pull: int,
    facts: Mergeability,
    budget: UpdateBudget,
) -> Decision:
    """A branch whose checks passed against a base that has since moved.

    Never merged - that is this module's headline - and never retried either: the
    diff is fine, only its base is old, and a re-dispatch would throw away a
    finished piece of work to reproduce it against a base that will have moved
    again by the time it finishes.
    """
    cap = max(int(budget.cap), 1)
    held = f"the branch is behind {facts.base_name}"

    if budget.exhausted(entry.number):
        rounds = budget.spent(entry.number)
        reason = (
            f"the base moved under this pull request {rounds} time(s) and it has been updated "
            f"as often; it is being starved by faster siblings rather than making progress"
        )
        return Decision(
            number=entry.number,
            pull=pull,
            verdict=BEHIND,
            detail=reason,
            transition=Transition(
                ref=entry.ref,
                from_label=REVIEW,
                to_label=FAILED,
                reason=reason,
                task_id=entry.task_id,
                # The attempt budget is untouched: nothing about the work was
                # wrong, and charging it would take the retry away from whatever
                # goes wrong next.
                comment=(
                    f"apiary: {reason}. The branch itself is fine - PR #{pull} passed its "
                    f"checks every time - so the useful next step is a human merging it by "
                    f"hand, or running fewer workers at once so the base stops moving."
                ),
            ),
            held=held,
        )

    return Decision(
        number=entry.number,
        pull=pull,
        verdict=BEHIND,
        detail=(
            f"behind {facts.base_name}; updating the branch rather than merging checks that "
            f"passed against an older base"
        ),
        update=BranchUpdate(
            number=entry.number,
            pull=pull,
            branch=facts.branch or entry.branch,
            base=facts.base,
            round=budget.spent(entry.number) + 1,
            cap=cap,
        ),
        held=held,
    )


def _serialise(
    decisions: list[Decision], candidates: list[tuple[int, int]], policy: UpdatePolicy
) -> list[Decision]:
    """Hold every merge but this cycle's, and pick which one that is.

    The pull request with the most update rounds behind it wins, ties broken by
    issue number. That direction is the whole anti-starvation argument: the
    branch that has been invalidated most often is the one closest to being
    given up on, so it goes first rather than last, and the round cap becomes a
    bound nothing normally reaches instead of a queue's natural end state.
    """
    allowed = max(int(policy.merges_per_cycle), 1)
    if len(candidates) <= allowed:
        return decisions

    ranked = sorted(candidates, key=lambda item: (-item[0], item[1]))
    winners = {number for _, number in ranked[:allowed]}
    queued = {number for _, number in candidates} - winners
    names = ", ".join(f"#{n}" for n in sorted(winners))
    return [
        replace(
            decision,
            held=(
                f"merging {names} this cycle; a strict required-status policy invalidates "
                f"every other open pull request, so this one goes next cycle"
            ),
            detail=f"{decision.detail}; queued behind {names}",
        )
        if decision.number in queued
        else decision
        for decision in decisions
    ]


def _defer(decision: Decision) -> Decision:
    """Drop a branch update because a merge is landing in the same cycle."""
    return replace(
        decision,
        update=None,
        detail=(
            f"{decision.detail}; deferred - a merge lands this cycle and would make this "
            f"update stale again"
        ),
    )


def _admit(checks: ChecksPlan, held: Mapping[int, str]) -> ChecksPlan:
    """#23's plan with the held merges - and their `swarm:done` - taken out.

    `dataclasses.replace` rather than a new type, so what reaches
    `checks.apply_checks` is the same shape it computed and the gate is
    subtractive: this module can stop a merge and can never invent one.
    """
    if not held:
        return checks
    outcomes = tuple(
        replace(
            outcome,
            merge=None,
            transition=None,
            detail=f"{outcome.detail}; not merged: {held[outcome.number]}",
        )
        if outcome.number in held and outcome.merge is not None
        else outcome
        for outcome in checks.outcomes
    )
    return replace(checks, outcomes=outcomes)


# --------------------------------------------------------------------------
# The re-dispatch's context
# --------------------------------------------------------------------------


def quote(text: str) -> str:
    """Detail, indented so no line of it can be markdown or a section heading.

    `checks.QUOTE_INDENT` is imported rather than respelled: the hazard is
    identical - a line reading `## Verify` at column 0 adds a second `## Verify`
    section and makes the issue malformed (`docs/issue-contract.md` §1.1) - and
    two modules with their own idea of how to neutralise it is one module too
    many. File paths come from GitHub, so they are data, not trusted text.
    """
    body = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not body:
        return f"{QUOTE_INDENT}(no detail)"
    return "\n".join(f"{QUOTE_INDENT}{line}".rstrip() for line in body.split("\n"))


def conflict_context(
    entry: LedgerEntry, facts: Mergeability, files: Sequence[str] = ()
) -> str:
    """What the next attempt has to be told, and nothing it cannot act on.

    GitHub's API reports *that* a pull request conflicts, never which hunks - the
    conflict is computed by a merge it never persists - so this names the base
    the branch no longer applies to and the files where the collision can be, and
    says the one thing that changes the next attempt's behaviour: start from the
    base, do not try to fix the branch.

    `files` is the pull request's own paths when the client could list them, and
    the issue's declared `## Files` otherwise. The declared set is the honest
    fallback rather than a guess: it is the only set the worker is allowed to
    edit anyway (`worker/edit.apply_edits` refuses the rest).
    """
    touched = tuple(path for path in files if path) or entry.files
    lines = [
        f"The branch no longer applies to {facts.base_name}: git reports a conflict "
        f"({facts.summary()}).",
        "",
        "The base moved while this attempt was running - another task merged first. The "
        "next attempt starts from a fresh clone of the base, so re-apply the change on top "
        "of what is there now rather than trying to reconstruct the old branch.",
        "",
        "Files in play:",
    ]
    lines.extend(f"- {path}" for path in touched)
    return tail("\n".join(lines), CONTEXT_CHARS)


def write_conflict(body: str, text: str, *, attempt: int) -> str:
    """Put the conflict in the issue body, replacing any block already there.

    The sibling of `checks.write_feedback`, and the same three rules: replace
    rather than append, so the next attempt reads one conflict and not three;
    keep every byte outside the block, so a human editing prose while the
    orchestrator records a conflict does not lose their edit; indent the content,
    so nothing in it can become a contract section.
    """
    block = "\n".join(
        [
            CONFLICT_OPEN,
            f"**apiary: attempt {int(attempt)} starts from a fresh base.** The previous "
            f"branch could not be merged:",
            "",
            quote(tail(text, CONTEXT_CHARS)),
            CONFLICT_CLOSE,
        ]
    )
    stripped = strip_conflict(body)
    return f"{stripped}\n\n{block}\n" if stripped else f"{block}\n"


def read_conflict(body: str) -> str:
    """The quoted conflict detail of the last attempt, undented, or `""`.

    Exported for the call site this ticket cannot write, exactly as
    `checks.read_feedback` is: the worker builds its prompt from `## Goal` plus
    the declared files (`worker/entrypoint.run_worker`), so a re-dispatched
    attempt sees this only once that call site passes it in, and
    `worker/entrypoint.py` is outside this ticket's file set. Until it does, the
    block is what a human reads to find out why an issue went round again.
    """
    text = body or ""
    start = text.find(CONFLICT_OPEN)
    end = text.find(CONFLICT_CLOSE, start + 1)
    if start < 0 or end < 0:
        return ""
    inner = text[start + len(CONFLICT_OPEN) : end]
    # Only the indented lines are the detail; the sentence above them is this
    # module's framing, and a consumer wants the facts rather than the framing.
    lines = [
        line[len(QUOTE_INDENT) :] for line in inner.split("\n") if line.startswith(QUOTE_INDENT)
    ]
    return "\n".join(lines).strip("\n")


def strip_conflict(body: str) -> str:
    """The body without its block, so `write_conflict` can put a fresh one back."""
    text = body or ""
    start = text.find(CONFLICT_OPEN)
    if start < 0:
        return text.rstrip("\n")
    end = text.find(CONFLICT_CLOSE, start)
    if end < 0:
        # An opener with no closer is a body somebody edited by hand mid-block.
        # Truncating from the opener is the reading that cannot leave a stray
        # closer behind to swallow the next block's content.
        return text[:start].rstrip("\n")
    return (text[:start] + text[end + len(CONFLICT_CLOSE) :]).rstrip("\n")


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Failure:
    """One thing this pass could not do, and to which issue.

    Collected rather than raised, as `checks.apply_checks` collects: one branch
    GitHub will not update must not stop the other nineteen from being decided.
    """

    number: int
    reason: str

    def __str__(self) -> str:
        return f"#{self.number}: {self.reason}"


@dataclass(frozen=True)
class MergeabilityReport:
    """What one pass actually achieved."""

    plan: MergeabilityPlan
    updated: tuple[int, ...] = ()
    applied: tuple[Transition, ...] = ()
    failures: tuple[Failure, ...] = ()
    #: Branches that needed updating and this client cannot update - see the
    #: module docstring. Reported every cycle, and the round is still charged, so
    #: a missing method ends in a named `swarm:failed` rather than in a review
    #: queue that quietly never drains.
    unupdatable: tuple[str, ...] = ()
    #: Comments this client had no method to post. `reconcile.post_comment`
    #: printed them instead.
    uncommented: tuple[int, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        parts = [
            f"updated {len(self.updated)}/{len(self.plan.updates)} branch(es)",
            f"applied {len(self.applied)}/{len(self.plan.transitions)} transition(s)",
            f"admitted {len(self.plan.merges)} merge(s)",
        ]
        if self.failures:
            parts.append(
                f"{len(self.failures)} failed: " + "; ".join(str(f) for f in self.failures)
            )
        if self.unupdatable:
            parts.append(
                f"could not update {', '.join(self.unupdatable)} - the client has no "
                + " or ".join(UPDATE_METHODS)
            )
        if self.uncommented:
            names = ", ".join(f"#{n}" for n in self.uncommented)
            parts.append(f"could not comment on {names} - the client has no {COMMENT_METHOD}")
        return ", ".join(parts)


def update_branch(client: Any, update: BranchUpdate) -> bool:
    """Drag one branch onto the current base, or say the client cannot.

    Probed, because `GitHubClient`'s surface is "the endpoints v2 actually
    calls" and nothing has needed `PUT /pulls/{n}/update-branch` until now;
    `client.py` is outside this ticket's file set. Only the pull request number
    is passed, for `worker/pr.py`'s reason: a future method certainly takes it
    and may or may not take `expected_head_sha`, and guessing wrong would be a
    `TypeError` in the loop body.
    """
    for name in UPDATE_METHODS:
        updater = getattr(client, name, None)
        if updater is None:
            continue
        updater(update.pull)
        return True
    return False


def apply_mergeability(
    client: Any,
    plan: MergeabilityPlan,
    *,
    budget: UpdateBudget | None = None,
    dry_run: bool = False,
) -> MergeabilityReport:
    """Update, then relabel. Never raises for one issue; see `Failure`.

    **A round is charged for every planned update, whether or not it happened.**
    The cap counts cycles this pull request spent behind its base, not API calls
    that succeeded - a client with no update method would otherwise be an
    infinite wait charged to nobody, which is exactly the parked-in-review
    outcome this ticket exists to remove.

    Within a transition the order is `docs/issue-contract.md` §5 and
    `checks.apply_checks`': the counter and the conflict detail are one body
    `PATCH` written first, then the new label, then the old one. One patch rather
    than two because both are edits to the same body, and a second read between
    them is a window for a human's edit to be lost.
    """
    updated: list[int] = []
    applied: list[Transition] = []
    failures: list[Failure] = []
    unupdatable: list[str] = []
    uncommented: list[int] = []

    if dry_run:
        return MergeabilityReport(plan=plan)

    for update in plan.updates:
        if budget is not None:
            budget.spend(update.number)
        try:
            if not update_branch(client, update):
                unupdatable.append(update.branch)
                print(
                    f"! no {' or '.join(UPDATE_METHODS)}; {update.branch} is behind "
                    f"{update.base or 'its base'} and cannot be updated "
                    f"(round {update.round} of {update.cap})",
                    file=sys.stderr,
                )
                continue
        except GitHubError as exc:
            # A 422 (the branch is not behind after all, or is already updating)
            # or a 403. The issue keeps `swarm:review` and the next cycle re-reads
            # the pull request; the round is charged either way.
            failures.append(Failure(update.number, f"updating PR #{update.pull}: {exc}"))
            continue
        updated.append(update.number)

    for decision in plan.decisions:
        transition = decision.transition
        if transition is None:
            continue
        try:
            if transition.attempt is not None or decision.context:
                _patch_body(client, transition, decision.context)
            # `Transition` is keyed on the task; these calls address the
            # GitHub API, which takes issue numbers.
            number = issue_number(transition.ref)
            client.add_labels(number, [transition.to_label])
            if transition.from_label and transition.from_label != transition.to_label:
                client.remove_label(number, transition.from_label)
        except GitHubError as exc:
            failures.append(Failure(decision.number, f"{transition.to_label}: {exc}"))
            continue
        applied.append(transition)
        # The pull request this issue was being counted for is finished with -
        # merged, abandoned or about to be replaced by a re-dispatch's - so the
        # rounds it spent must not be inherited by whatever opens next.
        if budget is not None:
            budget.clear(decision.number)
        if transition.comment and not post_comment(client, decision.number, transition.comment):
            uncommented.append(decision.number)

    return MergeabilityReport(
        plan=plan,
        updated=tuple(updated),
        applied=tuple(applied),
        failures=tuple(failures),
        unupdatable=tuple(dict.fromkeys(unupdatable)),
        uncommented=tuple(dict.fromkeys(uncommented)),
    )


def _patch_body(client: Any, transition: Transition, context: str) -> None:
    """One `PATCH`, carrying the counter and the conflict the next attempt needs.

    The body is re-read immediately before the write, which §5 requires and which
    is cheap because it happens only for an issue that just lost its pull
    request. `rewrite_marker` is #22's, imported rather than reimplemented: two
    modules with their own idea of where the counter lives is how a counter stops
    bounding anything.
    """
    number = issue_number(transition.ref)
    issue = client.get_issue(number)
    body = issue.get("body") or ""
    if transition.attempt is not None and transition.task_id:
        body = rewrite_marker(body, transition.task_id, transition.attempt)
    if context:
        body = write_conflict(body, context, attempt=transition.attempt or 0)
    client.update_issue(number, body=body)


# --------------------------------------------------------------------------
# One pass
# --------------------------------------------------------------------------


def run_mergeability(
    client: Any,
    ledger: Ledger,
    checks: ChecksPlan,
    *,
    pulls: Mapping[str, PullState] | None = None,
    budget: UpdateBudget | None = None,
    policy: UpdatePolicy | None = None,
    max_attempts: int = SETTINGS.max_attempts_per_task,
    dry_run: bool = False,
) -> MergeabilityReport:
    """Read, decide, write, and hand `report.plan.admitted` to `checks.apply_checks`.

    Called with the ledger and the check plan a cycle already computed, so it
    costs one `get_pull_request` per `swarm:review` issue with an open pull
    request - the listing does not carry `mergeable` - plus one file listing per
    conflict, and nothing else. That is proportional to the review queue rather
    than to the ledger.

    `pulls` defaults to reading the open pull requests itself; a caller that
    already listed them for #23 should pass its mapping rather than paying for a
    second listing.
    """
    open_pulls = read_pulls(client) if pulls is None else pulls
    rules = policy or UpdatePolicy()
    spent = budget if budget is not None else UpdateBudget(cap=rules.max_update_rounds)

    states: dict[int, Mergeability] | None = None
    files: dict[int, tuple[str, ...]] = {}
    if open_pulls is not None:
        states = {}
        for entry in sorted(ledger.entries.values(), key=lambda entry: entry.ref):
            if entry.state_label != REVIEW:
                continue
            pull = open_pulls.get(entry.branch)
            if pull is None:
                continue
            state = read_mergeability(client, pull)
            states[entry.number] = state
            if state.verdict == CONFLICTED:
                files[entry.number] = read_touched_files(client, pull)
        spent.retain(states)

    plan = plan_mergeability(
        ledger,
        checks,
        states=states,
        budget=spent,
        policy=rules,
        files=files,
        max_attempts=max_attempts,
    )
    return apply_mergeability(client, plan, budget=spent, dry_run=dry_run)


if __name__ == "__main__":  # pragma: no cover - manual dry run, see module docstring
    # Read-only on every path: no update, no merge, no label, no comment, no
    # adoption. This module force-updates pull request branches when it is not in
    # a dry run, so nothing below is allowed to grow a write.
    from .checks import plan_checks, read_checks

    repo = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GITHUB_REPOSITORY", "")
    gh = GitHubClient.from_env(repo)
    rules = UpdatePolicy.from_env()
    print(rules.summary())
    live = load_ledger(gh, adopt=False)
    open_pulls = read_pulls(gh)
    seen = {
        entry.number: read_checks(gh, (open_pulls or {})[entry.branch].ref)
        for entry in live.entries.values()
        if entry.state_label == REVIEW and entry.branch in (open_pulls or {})
    }
    dry = run_mergeability(
        gh,
        live,
        plan_checks(live, pulls=open_pulls, checks=seen),
        pulls=open_pulls,
        policy=rules,
        dry_run=True,
    )
    for row in dry.plan.decisions:
        print(f"  {row}")
    for planned in dry.plan.updates:
        print(f"would {planned}")
    for moved in dry.plan.transitions:
        print(f"would {moved}")
    print(dry.plan.summary())
