"""Tests for the derived-state resolver, and the replay harness that judges it.

**Read the honesty note first, because it bounds what green here means.** The
corpus under `fixtures/runs/` is synthesised. `swarm run --new` refuses classic
and OAuth tokens by design (`security.assert_provision_token`), no fine-grained
PAT exists in this environment, and `docs/demo-run.md` records the same wall
from 2026-08-14 - so there are no recorded runs to replay and this session could
not make any. A green run of this file proves the **reducer is
self-consistent**: that it computes what its rules say it computes, over streams
that exercise each transition. It proves nothing about whether those rules match
a live run, which is the question #145 was created to answer and which #146's
shadow window has to answer instead.

Two groups of tests, and they fail for different reasons.

**The reducer, as data.** `resolve` is pure - no Docker, no network, no clock -
so every rule is a constructed `Observation` and an assertion about one verdict.
These are the tests that break when somebody changes a rule, and they are
written one rule per test so the failure names the rule.

**The replay.** Every directory under `fixtures/runs/` is loaded, resolved cycle
by cycle and diffed against the `swarm:*` labels that run's control plane held.
The corpus declares the divergences it should produce and the harness asserts
the set matches **exactly** - an undeclared divergence fails, and a declared one
that stops happening fails too. The second direction is the one that will earn
its keep: three of the declared divergences are places where ADR 0001's claim
about derivability is false, and the day one of them becomes derivable this
suite says so rather than silently agreeing.

There is no assertion anywhere in this file of the form "no divergences". The
ticket says a divergence must fail loudly and name both states, and a harness
tuned until everything agreed would have had to leave out the infrastructure
ceiling, the renewed retry budget and the goal-gate revival - which are the three
most interesting things the exercise found.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from fixtures.corpus import (
    RECORDED,
    SYNTHESISED,
    CorpusError,
    CorpusRun,
    corpus_runs,
    load_corpus,
)
from swarm.github.branches import task_branch
from swarm.github.refs import pull_ref, task_ref as ref
from swarm.orchestrator.derived import (
    BLOCKED,
    CLAIMED,
    ELIGIBLE,
    LANDED,
    NEEDS_HUMAN,
    REVIEW,
    STATES,
    AttemptFact,
    Budget,
    ContainerFact,
    Observation,
    PullFact,
    TaskFact,
    diverge,
    observe,
    report,
    resolve,
)
from swarm.orchestrator.lifecycle import INTERNAL_STATE, internal_state

RUNS = corpus_runs()


# --------------------------------------------------------------------------
# Builders. Every reducer test is one of these with one thing changed.
# --------------------------------------------------------------------------


def world(**facts: object) -> Observation:
    """An observation with one task, `#1 solo`, and nothing else going on."""
    facts.setdefault("tasks", (TaskFact(ref=ref(1), task_id="solo"),))
    return Observation(cycle=1, **facts)  # type: ignore[arg-type]


def running(number: int = 1, *, run_id: str = "run-1", alive: bool = True) -> ContainerFact:
    return ContainerFact(id=f"c{number:011d}", run_id=run_id, ref=ref(number), running=alive)


def open_pull(pull: int, number: int = 1, attempt: int = 0, **kwargs: object) -> PullFact:
    """A pull request fact. `pull` is minted here, as a real listing mints it.

    The builder takes an `int` and `PullFact.number` is a `PullRef` since #208,
    so the mint happens at this edge rather than in every caller - which is the
    same shape `console_board._pull_facts` and the corpus loader have, and it
    keeps the test bodies reading `open_pull(71)` instead of carrying the
    adapter's spelling into thirty call sites.
    """
    return PullFact(  # type: ignore[arg-type]
        number=pull_ref(pull), ref=ref(number), attempt=attempt, **kwargs
    )


def state(observation: Observation, task: str = "solo") -> str:
    return resolve(observation).state(task)


# --------------------------------------------------------------------------
# The reducer: one test per rule
# --------------------------------------------------------------------------


def test_a_task_with_no_dependencies_and_nothing_running_is_eligible() -> None:
    assert state(world()) == ELIGIBLE


def test_a_task_waiting_on_an_unlanded_dependency_is_blocked() -> None:
    observation = world(
        tasks=(
            TaskFact(ref=ref(1), task_id="solo", depends_on=(ref(2),)),
            TaskFact(ref=ref(2), task_id="dep"),
        )
    )
    assert state(observation) == BLOCKED
    # And the dependency itself is not blocked by its dependent, which is the
    # bug a one-pass resolver written in ledger order would have.
    assert state(observation, "dep") == ELIGIBLE


def test_a_dependency_becomes_discharged_when_its_pull_request_merges() -> None:
    observation = world(
        tasks=(
            TaskFact(ref=ref(1), task_id="solo", depends_on=(ref(2),)),
            TaskFact(ref=ref(2), task_id="dep"),
        ),
        pulls=(open_pull(50, 2, merged=True, closed=True),),
    )
    assert state(observation, "dep") == LANDED
    assert state(observation) == ELIGIBLE


def test_a_dependency_closed_as_completed_discharges_without_any_pull_request() -> None:
    """Half of what a real plan waits on is hand-written work no worker touches.

    `github/readiness.IssueState.satisfied` already runs on this rule, so a
    resolver that recognised only merges would hold every such task in
    `blocked` while the control plane had long since moved on.
    """
    observation = world(
        tasks=(
            TaskFact(ref=ref(1), task_id="solo", depends_on=(ref(2),)),
            TaskFact(ref=ref(2), task_id="dep", closed=True, state_reason="completed"),
        )
    )
    assert state(observation, "dep") == LANDED
    assert state(observation) == ELIGIBLE


def test_a_dependency_closed_as_not_planned_discharges_nothing() -> None:
    observation = world(
        tasks=(
            TaskFact(ref=ref(1), task_id="solo", depends_on=(ref(2),)),
            TaskFact(ref=ref(2), task_id="dep", closed=True, state_reason="not_planned"),
        )
    )
    # The abandoned item itself was `eligible` here until #147, which is the
    # gap #146 classified as `closed-not-planned` and ADR 0001 called "a gap
    # rather than a limit": `reconcile._closed_verdict` had escalated on this
    # fact since #22 and the resolver had no rule for it. It has one now.
    assert state(observation, "dep") == NEEDS_HUMAN
    assert state(observation) == BLOCKED


def test_a_work_item_closed_as_not_planned_is_needs_human_and_not_landed() -> None:
    """The rule #147 added, on its own rather than as a dependency.

    Both halves matter and they are the two ways of getting it wrong. Reading
    only `closed` would make an abandoned item `landed`, which unblocks every
    dependant on work somebody explicitly decided not to build. Reading nothing
    would leave it `eligible`, which is what shipped between #145 and #147 and
    made the swarm and its own control plane disagree about every issue a human
    cancelled.
    """
    abandoned = world(
        tasks=(TaskFact(ref=ref(1), task_id="solo", closed=True, state_reason="not_planned"),)
    )
    completed = world(
        tasks=(TaskFact(ref=ref(1), task_id="solo", closed=True, state_reason="completed"),)
    )

    assert state(abandoned) == NEEDS_HUMAN
    assert state(completed) == LANDED


def test_a_running_container_for_this_task_is_a_claim() -> None:
    assert state(world(containers=(running(),))) == CLAIMED


def test_an_exited_container_is_not_a_claim() -> None:
    """`docker ps --all` lists it, and the worker that filled it has finished.

    This is the whole reason `ContainerFact` carries `running` and
    `containers.manager.Handle` does not: the listing the manager returns today
    would hold a task in `claimed` from the moment its worker exited until the
    reaper got to it, which is precisely the cycle in which `claimed` and
    `review` disagree.
    """
    assert state(world(containers=(running(alive=False),))) == ELIGIBLE


def test_a_container_from_another_run_holds_no_claim() -> None:
    """`orchestrator/recovery.py`'s liveness rule, not a second one."""
    observation = world(
        containers=(running(run_id="some-other-run"),), live_run_ids=frozenset({"run-1"})
    )
    assert state(observation) == ELIGIBLE
    kept = world(containers=(running(run_id="sibling"),), live_run_ids=frozenset({"sibling"}))
    assert state(kept) == CLAIMED


def test_an_open_pull_request_is_review() -> None:
    assert state(world(pulls=(open_pull(7),))) == REVIEW


def test_a_merged_pull_request_is_landed() -> None:
    assert state(world(pulls=(open_pull(7, merged=True, closed=True),))) == LANDED


def test_a_pull_request_closed_unmerged_is_not_review() -> None:
    assert state(world(pulls=(open_pull(7, closed=True),))) == ELIGIBLE


def test_a_pull_request_apiary_did_not_mint_is_never_joined_to_a_task() -> None:
    """The join is on the ref inside the head branch (#144), not on `Closes #n`.

    There is no way to express "a pull request for issue 1 whose head is
    `fix/typo`" in a `PullFact` at all - the type requires a `TaskRef` that only
    `parse_task_branch` produces - so this asserts the loader's half instead,
    which is where a real listing would arrive.
    """
    from fixtures.corpus import _cycle  # noqa: PLC0415 - the private half is the subject

    cycle = _cycle(
        {
            "cycle": 1,
            "tasks": [{"ref": "#1", "task_id": "solo"}],
            "pulls": [{"number": 9, "head": "fix/typo"}],
            "control": {"solo": "swarm:ready"},
        },
        results=(),
        default_run_id="run-1",
        where="inline",
    )
    assert cycle.observation.pulls == ()
    assert resolve(cycle.observation).state("solo") == ELIGIBLE


def test_the_newest_attempts_pull_request_is_the_one_reported() -> None:
    """Two open pull requests for one task is `recovery.py`'s documented cost of #144.

    A retry dispatched over a worker that had in fact published opens a second
    pull request rather than updating the first. Until a human closes the
    orphan, both are open, and the number a reader is sent to must be the
    attempt the run is actually waiting on.
    """
    observation = world(pulls=(open_pull(70, attempt=0), open_pull(71, attempt=1)))
    verdict = resolve(observation).by_task["solo"]
    assert verdict.state == REVIEW
    assert verdict.pull == pull_ref(71)


def test_two_open_pull_requests_on_one_attempt_are_broken_by_number() -> None:
    """`_open_pull` compares `(attempt, number)`, so this is the branch that
    reaches the number - and before `PullRef` sorted (#208) reaching it raised.
    The higher number is the newer publication of the same dispatch."""
    observation = world(pulls=(open_pull(70, attempt=1), open_pull(71, attempt=1)))

    assert resolve(observation).by_task["solo"].pull == pull_ref(71)


@pytest.mark.parametrize("order", [(101, 104), (104, 101)])
def test_the_named_merge_does_not_depend_on_the_listing_order(order) -> None:
    """Both `sorted()` sites in this module, held to one answer.

    A task can carry two merged pull requests - `recovery.py`'s orphan, merged by
    a human rather than closed - so `_landed`'s sentence and `_merged_pull`'s
    number are each a choice among several. Both sort, so the choice is the same
    one twice and the same one every cycle: a reader told "pull request #101
    merged" and linked to #104 is being sent to a diff that explains nothing.
    Parametrised over both listing orders because the order GitHub returns is
    exactly what an unordered key would leave the answer to.
    """
    pulls = tuple(
        open_pull(n, merged=True, closed=True, attempt=i) for i, n in enumerate(order)
    )

    verdict = resolve(world(pulls=pulls)).by_task["solo"]

    assert verdict.state == LANDED
    assert verdict.pull == pull_ref(101)
    assert verdict.because == "pull request #101 merged"


def test_a_verdict_names_a_pull_request_with_exactly_one_hash() -> None:
    """The retype changed no message, and this is what says so.

    `PullRef` renders `#101` on its own - the ref carries the `#` - so the two
    format strings that used to write one themselves had to drop it. A `##101`
    in an operator's shadow report is the kind of regression a type change is
    allowed to introduce silently and nothing else would catch.
    """
    review = resolve(world(pulls=(open_pull(101),))).by_task["solo"]
    landed = resolve(world(pulls=(open_pull(101, merged=True, closed=True),))).by_task["solo"]

    assert review.because == "pull request #101 is open for this task"
    assert landed.because == "pull request #101 merged"


def test_a_spent_attempt_budget_is_needs_human() -> None:
    observation = world(
        results=tuple(
            AttemptFact(ref=ref(1), attempt=n, exit_code=1) for n in range(3)
        ),
        budget=Budget(max_attempts=3),
    )
    verdict = resolve(observation).by_task["solo"]
    assert verdict.state == NEEDS_HUMAN
    assert verdict.attempts_spent == 3


def test_exit_2_spends_no_budget() -> None:
    """`docs/issue-contract.md` §4, and the rule a broken host would otherwise ride."""
    observation = world(
        results=tuple(AttemptFact(ref=ref(1), attempt=0, exit_code=2) for _ in range(3)),
        budget=Budget(max_attempts=3),
    )
    verdict = resolve(observation).by_task["solo"]
    assert verdict.attempts_spent == 0
    assert verdict.state == ELIGIBLE


def test_a_successful_attempt_spends_no_budget_either() -> None:
    """An exit 0 moves no label and writes no counter (`reconcile._observe`).

    Reading `ResultRecord.consumes_attempt` here instead - which is true for an
    exit 0, because §4 is answering a different question - reported a task that
    succeeded on its third attempt as `needs-human` while its pull request sat
    open in review.
    """
    observation = world(
        results=(
            AttemptFact(ref=ref(1), attempt=0, exit_code=1),
            AttemptFact(ref=ref(1), attempt=1, exit_code=1),
            AttemptFact(ref=ref(1), attempt=2, exit_code=0),
        ),
        pulls=(open_pull(70, attempt=2),),
        budget=Budget(max_attempts=3),
    )
    verdict = resolve(observation).by_task["solo"]
    assert verdict.attempts_spent == 2
    assert verdict.state == REVIEW


def test_a_branch_alone_accounts_for_the_attempts_before_it() -> None:
    """#144's payoff: an orchestrator with no local memory still counts.

    No results, no containers, no pull requests - only a name on the remote.
    That is the state a crashed orchestrator restarts into, and the name is the
    only thing that survived it.
    """
    observation = world(
        branches=(_branch(1, 2),), budget=Budget(max_attempts=3)
    )
    verdict = resolve(observation).by_task["solo"]
    assert verdict.attempts_spent == 2
    assert verdict.state == ELIGIBLE


def test_attempts_are_the_maximum_of_the_sources_and_never_their_sum() -> None:
    """One attempt that both pushed a branch and wrote a record is one attempt."""
    observation = world(
        branches=(_branch(1, 1),),
        results=(AttemptFact(ref=ref(1), attempt=0, exit_code=1),),
        pulls=(open_pull(70, attempt=1),),
    )
    assert resolve(observation).by_task["solo"].attempts_spent == 1


def test_landed_outranks_needs_human() -> None:
    """The precedence that saves the goal-gate revival run from ending wrong.

    A revived task that finally merges is `landed`, whatever its counter says -
    and a resolver ordering these the other way would report a merged pull
    request as work waiting for a human.
    """
    observation = world(
        results=tuple(AttemptFact(ref=ref(1), attempt=n, exit_code=1) for n in range(3)),
        pulls=(open_pull(70, attempt=3, merged=True, closed=True),),
        budget=Budget(max_attempts=3),
    )
    assert state(observation) == LANDED


def test_needs_human_outranks_a_container_still_running() -> None:
    observation = world(
        results=tuple(AttemptFact(ref=ref(1), attempt=n, exit_code=1) for n in range(3)),
        containers=(running(),),
        budget=Budget(max_attempts=3),
    )
    assert state(observation) == NEEDS_HUMAN


def test_claimed_outranks_review() -> None:
    """`worker/pr.py` reuses one pull request across retries, so a re-claimed
    task still has an open one. A container is a claim about now."""
    assert state(world(containers=(running(),), pulls=(open_pull(70),))) == CLAIMED


def test_every_state_the_resolver_can_return_is_in_the_precedence_table() -> None:
    """`STATES` is the design decision written down; this keeps it honest."""
    assert set(STATES) == {ELIGIBLE, BLOCKED, CLAIMED, REVIEW, LANDED, NEEDS_HUMAN}
    assert set(STATES) == set(INTERNAL_STATE.values())


def test_the_label_and_state_vocabularies_round_trip() -> None:
    """The property #152's `Transition` retype rests on, in both directions.

    A `Transition` carries states and the label is looked up only at the moment
    one is written, which is safe **only** because the mapping is a bijection: a
    transition built from the label an issue is carrying has to translate back to
    that same label, or `write_labels` removes the wrong one and the issue ends
    up wearing two.

    Written as a round trip rather than as "the tables are inverses" because that
    is the claim the code makes, and because a seventh state - one with no label
    to store it - would break this line rather than surfacing as a `KeyError` in
    the middle of a cycle.
    """
    from swarm.orchestrator.lifecycle import STATE_LABEL, state_label

    assert set(STATE_LABEL) == set(STATES)
    for label, state in INTERNAL_STATE.items():
        assert state_label(state) == label
    for state in STATES:
        assert internal_state(state_label(state)) == state


def test_a_state_no_label_stores_is_refused_rather_than_invented() -> None:
    """`internal_state` guesses; `state_label` must not, and the asymmetry is the point.

    One is reading labels somebody else's repository may carry, where a guess is
    better than a crash. The other is about to *write* one, and a made-up name is
    a label GitHub creates with a random colour and no description - the exact
    failure `github/labels.py` provisions the six to prevent.
    """
    from swarm.orchestrator.lifecycle import state_label

    with pytest.raises(KeyError, match="no swarm:. label stores"):
        state_label("unresolved")


def test_the_observation_type_cannot_carry_a_label_or_a_state() -> None:
    """The sourcing invariant, made structural rather than conventional.

    Not a style check. A resolver that could read the control plane would agree
    with it perfectly, prove nothing about ADR 0001, and the agreement would be
    an artefact of the wiring. So there is nowhere in the input to put a label,
    and adding one is a field somebody has to defend in review.
    """
    fields: set[str] = set()
    for kind in (Observation, TaskFact, ContainerFact, PullFact, AttemptFact, Budget):
        fields |= set(getattr(kind, "__dataclass_fields__"))
    assert not {name for name in fields if "label" in name or "status" in name}
    assert "state" not in fields
    assert "state_label" not in fields
    # `state_reason` is GitHub's word for *why an issue was closed* and is the
    # fact `readiness.IssueState.satisfied` already runs on, so it is named
    # explicitly here rather than caught by a substring rule that would have to
    # be relaxed for it.
    assert "state_reason" in fields


def test_a_task_the_observation_never_carried_has_no_derived_state() -> None:
    assert resolve(world()).state("nobody") == ""


def test_resolve_is_pure_over_its_input() -> None:
    """Called twice on one observation, it answers twice the same.

    Cheap, and it is the property that lets #146 resolve a cycle for a shadow
    report and #147 resolve it again for a decision without the two disagreeing.
    """
    observation = world(containers=(running(),), pulls=(open_pull(70),))
    assert resolve(observation) == resolve(observation)


# --------------------------------------------------------------------------
# Divergence reporting
# --------------------------------------------------------------------------


def test_a_divergence_names_the_task_the_cycle_and_both_states() -> None:
    """#145's criterion, verbatim: not a boolean, not a count.

    A count would pass a run whose every disagreement was on `needs-human` -
    the one state ADR 0001 reports outbound and the one a customer's tracker
    cannot infer.
    """
    resolution = resolve(world(containers=(running(),)))
    found = diverge(resolution, {"solo": ELIGIBLE})
    assert len(found) == 1
    one = found[0]
    assert (one.cycle, one.task_id, one.derived, one.control) == (1, "solo", CLAIMED, ELIGIBLE)
    assert str(ref(1)) in str(one)
    assert "claimed" in str(one) and "eligible" in str(one)


def test_agreement_reports_nothing() -> None:
    assert diverge(resolve(world()), {"solo": ELIGIBLE}) == ()


def test_a_task_the_control_plane_does_not_hold_is_not_a_divergence() -> None:
    """A malformed issue never enters the ledger (§1.4) and has no state to diff."""
    assert diverge(resolve(world()), {}) == ()


def test_the_report_renders_the_verdicts_and_the_divergences_together() -> None:
    resolution = resolve(world(containers=(running(),)))
    text = report(resolution, diverge(resolution, {"solo": ELIGIBLE}))
    assert "claimed" in text and "eligible" in text
    assert "1 divergence(s)" in text
    assert "no divergence" in report(resolve(world()))


# --------------------------------------------------------------------------
# The edge
# --------------------------------------------------------------------------


class _FakeEntry:
    """The four attributes `observe` reads off a ledger entry. Nothing else."""

    def __init__(self, number: int, task_id: str, blocked_by: tuple = (), closed: bool = False):
        self.ref = ref(number)
        self.task_id = task_id
        self.blocked_by = blocked_by
        self.closed = closed
        # Present, and the point: `observe` must not reach for it.
        self.state_label = "swarm:claimed"


def test_observe_builds_an_observation_without_reading_the_state_label() -> None:
    observation = observe(
        cycle=4,
        entries=[_FakeEntry(1, "solo"), _FakeEntry(2, "dep")],
        branch_names=["main", task_branch(ref(1), 2), "swarm/issue-1", "fix/typo"],
    )
    assert observation.cycle == 4
    assert {fact.task_id for fact in observation.tasks} == {"solo", "dep"}
    # Only the one apiary minted under the current scheme survives; `main`, the
    # human's branch and the pre-#144 name are counted by the caller, not acted
    # on (`github/branches.py`'s "parsing never raises").
    assert [(str(one.ref), one.attempt) for one in observation.branches] == [("#1", 2)]
    assert resolve(observation).state("solo") == ELIGIBLE


# --------------------------------------------------------------------------
# The replay
# --------------------------------------------------------------------------


@pytest.mark.parametrize("run", RUNS, ids=lambda run: run.name)
def test_the_corpus_replays_with_exactly_the_declared_divergences(run: CorpusRun) -> None:
    """Replay one run. **This is the assertion #145 is about.**

    Every cycle is resolved from the world alone and diffed against the
    `swarm:*` labels that cycle's control plane held. The manifest declares the
    divergences the run should produce; anything else fails, and a declared one
    that no longer happens fails too.

    Green here means the reducer is self-consistent over a synthesised stream.
    It does not mean the reducer is correct about a live run - there are none,
    and #146's shadow window is where that question gets answered.
    """
    found = []
    for cycle in run.cycles:
        found.extend(diverge(resolve(cycle.observation), cycle.control))

    undeclared = [one for one in found if one.key not in run.expected_keys]
    assert not undeclared, "undeclared divergence(s) in " + run.name + ":\n" + "\n".join(
        f"  {one}" for one in undeclared
    )

    missing = run.expected_keys - {one.key for one in found}
    assert not missing, (
        f"{run.name} declares divergence(s) that no longer happen: {sorted(missing)}. "
        "If derived state has become able to reproduce the control plane here, "
        "delete the declaration and say so in the ADR - that is the epic moving."
    )


@pytest.mark.parametrize("run", RUNS, ids=lambda run: run.name)
def test_every_corpus_cycle_resolves_a_state_for_every_task_it_carries(run: CorpusRun) -> None:
    """No task falls through the resolver silently.

    The failure this catches is a `TaskFact` whose ref nothing else in the
    observation mentions - which is what a broken join looks like, and #174 is
    what a broken join costs when nothing asserts on it.
    """
    for cycle in run.cycles:
        resolution = resolve(cycle.observation)
        assert len(resolution.verdicts) == len(cycle.observation.tasks)
        for verdict in resolution.verdicts:
            assert verdict.state in STATES
            assert verdict.because


@pytest.mark.parametrize("run", RUNS, ids=lambda run: run.name)
def test_every_corpus_run_reaches_a_terminal_state_or_declares_why_not(run: CorpusRun) -> None:
    """A run that never finishes anything exercises nothing.

    Terminal is judged on the **control plane**, because that is the run's own
    account of how it ended - every corpus run stops at `swarm:done` or
    `swarm:failed`. The derived side is then required either to agree or to
    have declared the disagreement, which is how `05-infrastructure-exit-2`
    passes: its last cycle derives `eligible` against a `swarm:failed` label,
    and that gap is the single most important thing this corpus records. A test
    demanding a terminal derived state would have made that run inexpressible,
    which is the tuning the ticket warns against.
    """
    final = run.cycles[-1]
    terminal = {LANDED, NEEDS_HUMAN}
    assert set(final.control.values()) <= terminal, f"{run.name} does not end"
    resolution = resolve(final.observation)
    assert resolution.verdicts
    for verdict in resolution.verdicts:
        if verdict.state in terminal:
            continue
        key = (final.index, verdict.task_id, verdict.state, final.control[verdict.task_id])
        assert key in run.expected_keys, (
            f"{run.name} ends with {verdict}, which is neither terminal nor declared"
        )


@pytest.mark.parametrize("run", RUNS, ids=lambda run: run.name)
def test_every_declared_divergence_carries_a_reason(run: CorpusRun) -> None:
    """A declaration without a `why` is a disagreement somebody silenced.

    The whole value of declaring rather than forbidding is that each entry is an
    argument about why derived state cannot reproduce the control plane here.
    An empty one is the harness being tuned to pass, which is the failure mode
    the ticket names.
    """
    for expected in run.expected:
        assert len(expected.why) > 80, f"{run.name}: {expected.key} has no argument"


@pytest.mark.parametrize("run", RUNS, ids=lambda run: run.name)
def test_the_event_log_and_the_observations_describe_the_same_run(run: CorpusRun) -> None:
    """The #141 log is not evidence, and it must not be fiction either.

    The replay deliberately does not read `events.jsonl` - half of it announces
    writes that landed, and reading that half is the sourcing violation
    `derived.py` exists to avoid. But a corpus whose log had drifted from its
    observations would mislead the human debugging a divergence, so the two are
    cross-checked here on the facts they share: which cycles happened, and which
    tasks the run was about.
    """
    logged = {int(event["cycle"]) for event in run.events if "cycle" in event}
    assert {cycle.index for cycle in run.cycles} <= logged
    tasks = {fact.task_id for cycle in run.cycles for fact in cycle.observation.tasks}
    named = {str(event["task"]) for event in run.events if event.get("task")}
    assert named <= tasks


def test_the_corpus_covers_the_awkward_cases_and_not_only_the_happy_path() -> None:
    """A corpus of three green runs would pass every assertion above and prove
    nothing. These five are the cases #145 asks for by name."""
    covered = " ".join(one for run in RUNS for one in run.exercises).lower()
    for case in (
        "verify failure",
        "interrupted orchestrator",
        "container that died",
        "exit 2",
        "revival",
    ):
        assert case in covered, f"no corpus run exercises {case!r}"
    assert any(run.expected for run in RUNS), "no run records a divergence at all"


# --------------------------------------------------------------------------
# The property the corpus format exists for
# --------------------------------------------------------------------------


def test_a_recorded_run_replays_identically_to_a_synthesised_one(tmp_path: Path) -> None:
    """**The deliverable, asserted.** `origin` is metadata and nothing branches on it.

    #145's constraint is that a real recorded run drops in beside a synthesised
    one with no code change - same loader, same assertions. This copies a
    corpus run, flips `origin` from `synthesised` to `recorded`, and asserts the
    replay is identical. If somebody ever makes the harness lenient about
    synthesised data - a skip, a softer assertion, a special case - this is the
    test that fails, and it fails before the leniency has a chance to hide a
    real run's divergence.
    """
    source = RUNS[0]
    assert source.origin == SYNTHESISED
    copied = tmp_path / source.name
    shutil.copytree(source.path, copied)
    manifest = json.loads((copied / "corpus.json").read_text())
    manifest["origin"] = RECORDED
    (copied / "corpus.json").write_text(json.dumps(manifest, indent=2) + "\n")

    recorded = load_corpus(copied)
    assert recorded.origin == RECORDED
    assert [cycle.observation for cycle in recorded.cycles] == [
        cycle.observation for cycle in source.cycles
    ]
    assert [cycle.control for cycle in recorded.cycles] == [
        cycle.control for cycle in source.cycles
    ]
    assert [
        [str(one) for one in diverge(resolve(cycle.observation), cycle.control)]
        for cycle in recorded.cycles
    ] == [
        [str(one) for one in diverge(resolve(cycle.observation), cycle.control)]
        for cycle in source.cycles
    ]


def test_every_corpus_run_is_readable_as_an_ordinary_run_directory() -> None:
    """`swarm show` has to be able to read one, or the format has drifted.

    `load_corpus` calls `artifacts.read_run` for this reason and discards the
    result; this test is the statement of why that call is there. A corpus that
    only this loader understood would replay green while having stopped being
    evidence about the thing a real run produces.
    """
    from swarm.artifacts import read_run  # noqa: PLC0415 - the import is the assertion

    for run in RUNS:
        view = read_run(run.path)
        assert view.run_id


def test_a_corpus_with_no_runs_is_an_error_rather_than_a_pass(tmp_path: Path) -> None:
    """The one failure mode a harness must not have."""
    with pytest.raises(CorpusError):
        corpus_runs(tmp_path)


def test_a_cycle_naming_a_result_file_that_is_not_there_fails_loudly(tmp_path: Path) -> None:
    """Otherwise it surfaces as an off-by-one in an attempt count much later."""
    source = RUNS[1]
    copied = tmp_path / source.name
    shutil.copytree(source.path, copied)
    lines = (copied / "observed.jsonl").read_text().splitlines()
    broken = json.loads(lines[-1])
    broken["results"].append("issue-999-attempt-0.json")
    lines[-1] = json.dumps(broken)
    (copied / "observed.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(CorpusError, match="no such result file"):
        load_corpus(copied)


def test_a_corpus_line_that_does_not_parse_is_refused_not_skipped(tmp_path: Path) -> None:
    """`artifacts.read_events` skips a bad line, because a killed run's last line
    is expected to be half-written. A committed corpus is not a live append, and
    a silently dropped cycle is a divergence that silently stops being asserted."""
    source = RUNS[0]
    copied = tmp_path / source.name
    shutil.copytree(source.path, copied)
    with (copied / "observed.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    with pytest.raises(CorpusError):
        load_corpus(copied)


def test_a_corpus_recording_a_label_the_control_plane_never_had_is_refused(
    tmp_path: Path,
) -> None:
    source = RUNS[0]
    copied = tmp_path / source.name
    shutil.copytree(source.path, copied)
    lines = (copied / "observed.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    first["control"]["core"] = "swarm:in-progress"
    lines[0] = json.dumps(first)
    (copied / "observed.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(CorpusError, match="not a state label"):
        load_corpus(copied)


def _branch(number: int, attempt: int):
    from swarm.github.branches import parse_task_branch  # noqa: PLC0415

    parsed = parse_task_branch(task_branch(ref(number), attempt))
    assert parsed is not None
    return parsed
