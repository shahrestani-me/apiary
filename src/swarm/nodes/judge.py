"""Progress ledger - stall detection, now reading the tracker instead of memory.

v1 asked four questions after every round: is the request satisfied, is
progress being made, are we in a loop, and how many stalls has that cost. This
is what stops a swarm burning an afternoon making no progress, and it is the
single cheapest reliability win in the whole design. `docs/architecture-v2.md`
step 5 keeps it verbatim - "same progress ledger as v1" - and changes exactly
one thing: where the answers come from. In v1 they came from a dict that died
with the process; here they come from `Ledger` (#9), the results a worker left
on disk (#29) and the pull requests the reconciler could see (#22).

**The short-circuits are the design, not an optimisation.** v1 already refused
to spend a model call on a question arithmetic could answer, and v2 needs it
more: `config.py` measures a ~6.7 s model swap per orchestrator call on this
host, and the reconcile loop runs on a 15 s interval where v1 ran once per
round. #21's dispatcher is explicit that "a cycle that only dispatches costs no
model load"; a judge that ran on every cycle would hand that back. So the model
is consulted only when every deterministic reading of the ledger is genuinely
ambiguous, and `Verdict.consulted` records whether that happened.

**Satisfaction is never the model's call.** `docs/issue-contract.md` §3 spends
a paragraph on why `swarm:review` maps to `running` and not `verified`:
completion is the merge, and a judge that declared a run finished while its
output sat in open PRs is precisely the failure that mapping exists to prevent.
"Every issue is `swarm:done`" is arithmetic, so it is computed here and the
model's answer to that question is overwritten rather than trusted.

**Rebasing is progress.** #34 keeps swarm PRs mergeable against a base that
every merge invalidates, so a task can spend several cycles being updated and
re-checked without its label moving. Counting that as "no progress" would raise
a stall, and a stall replans - which throws away work that was one green check
from landing. So the base-update count is part of what a cycle observes, and a
task whose churn is climbing has moved even when nothing else about it has.

**A task can fail for a reason it is not allowed to fix.** The `## Files`
section is a hard boundary: a worker may not touch a path the issue does not
list. When the fix lives outside that set the task fails identically every
attempt, and no replan helps - a new decomposition of the same objective hands
the same wall to a differently-named task. That is not a looping model, it is a
scoping error, and it needs a human. `Blocker` is the attempt to distinguish
it, and `Verdict.needs_human` is what stops a replan from papering over it.

The v1 entry point, `judge_node`, is unchanged in signature and still what
`graph.py` calls; it observes v1's in-memory `TaskRecord`s through the same
ladder rather than through a second copy of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Mapping, Protocol, Sequence

from ..config import SETTINGS
from ..github.ledger import STATUS_BY_LABEL, Ledger
from ..llm import orchestrator_llm, structured
from ..state import ProgressJudgement, SwarmState, TaskRecord, TaskStatus
from ..worker.entrypoint import EXIT_OK
from ..worker.result import ResultRecord

SYSTEM = """You judge whether a multi-agent coding run is progressing.
Be strict. If the same tasks keep failing with the same error, that is a loop.
Return JSON only."""

#: The two `TaskStatus` values that mean a task is finished with, one way or
#: the other. Taken as statuses rather than as labels because this module reads
#: v1 records and v2 entries through one ladder, and `swarm:done -> verified`
#: and `swarm:failed -> abandoned` are the two rows of §3's table that lose
#: nothing in the projection.
TERMINAL: frozenset[TaskStatus] = frozenset({"verified", "abandoned"})

#: A container holds it, or a PR is open against it. §3's projection collapses
#: `swarm:claimed` and `swarm:review` into this one status, and that collapse
#: is exactly the distinction this module does not need: both mean somebody is
#: working on it, which is the only thing the ladder below asks.
IN_FLIGHT: frozenset[TaskStatus] = frozenset({"running"})

#: Attempts a task must have spent before a repeated failure is read as a
#: scoping error rather than as bad luck. Two, because one failure is a data
#: point and the second one having the identical shape is the evidence.
BLOCKER_MIN_ATTEMPTS = 2

#: How much of a failure's text carries its identity. The tail, because a
#: traceback's last lines say what broke and its first lines say where the run
#: started, and the whole of it would make two failures differ over a temporary
#: directory name.
SIGNATURE_CHARS = 400

# A repo-relative path with an extension: `src/swarm/nodes/judge.py`, but not
# `swarm` and not a bare `judge.py`. Requiring a slash is what keeps an English
# sentence from parsing as a file, and requiring an extension keeps a URL path
# or an issue reference out.
#
# The filename may itself contain dots. `[\w+-]+` did not allow that, so
# `src/calc.test.js` came back as `src/calc.test` - a path that does not exist,
# which therefore matched nothing in any `## Files` set and made every
# double-extension file invisible to the "could the worker have fixed it"
# question. That naming convention (`*.test.js`, `*.spec.tsx`) is the dominant
# one in exactly the stacks #87 is adding. Found by #93's agreement test, which
# exists to keep this expression honest.
_PATH_RE = re.compile(r"(?:[\w.@+-]+/)+[\w.@+-]*\.[A-Za-z0-9_]+")

# Digits are dropped from a failure signature: pytest reports `1 failed in
# 0.42s`, a traceback names line numbers, and a temporary directory carries a
# PID. Two runs of the same broken test differ in all three and are the same
# failure, which is the judgement this whole normalisation exists to make.
_DIGITS_RE = re.compile(r"\d+")
_SPACE_RE = re.compile(r"\s+")

# Paths that are not the repository's: an absolute path, and the two shapes an
# interpreter's own tree takes inside a container. A traceback through
# site-packages says nothing about whether a task can reach its own fix.
_FOREIGN = ("site-packages", "dist-packages", ".venv/", "/usr/", "/opt/")


class Oracle(Protocol):
    """The one model call this module can make. `structured(...)` satisfies it.

    A `Protocol` for the reason every other seam in this codebase is one: a
    judge test that needs Ollama running to reach the deterministic ladder is a
    test that does not run, and the ladder is the part worth testing.
    """

    def invoke(self, messages: Sequence[tuple[str, str]]) -> ProgressJudgement: ...


# --------------------------------------------------------------------------
# What a cycle observes
# --------------------------------------------------------------------------


def failure_signature(text: str) -> str:
    """One failure's identity, stable across re-runs of the same failure.

    Lowercased, digit-stripped and whitespace-collapsed, then the tail. The
    normalisation is what makes "the same error twice" a computable fact: the
    raw strings of two identical pytest failures differ in their timing line
    alone, and comparing them verbatim would never find a loop.
    """
    stripped = _DIGITS_RE.sub("#", (text or "").casefold())
    return _SPACE_RE.sub(" ", stripped).strip()[-SIGNATURE_CHARS:]


def mentioned_paths(text: str) -> tuple[str, ...]:
    """Repo-relative-looking paths named in a failure, in order, deduplicated.

    Deliberately conservative. Everything absolute and everything inside an
    interpreter's own tree is dropped, because the question this feeds is "could
    the worker have fixed it", and a path it could never have been given an
    answer about is not evidence either way.
    """
    found: list[str] = []
    for match in _PATH_RE.finditer(text or ""):
        path = match.group(0)
        if path.startswith("/") or any(part in path for part in _FOREIGN):
            continue
        # `./internal/calc/calc.go` and `internal/calc/calc.go` are one file,
        # and a `## Files` set never spells the first. `checks.failing_paths`
        # has always stripped it; this did not, so a Go build error named a
        # path that compared equal to nothing.
        path = path.lstrip("./")
        if path not in found:
            found.append(path)
    return tuple(found)


@dataclass(frozen=True)
class Blocker:
    """A task failing repeatedly on something its `## Files` cannot reach.

    Reported rather than acted on. The label move belongs to the reconciler
    (§4's `any -> failed` row) and the re-scoping belongs to a human; what this
    module owns is the distinction between "the model keeps writing the wrong
    code" and "the code that needs writing is in a file this task was never
    given", because only the first of those is fixed by replanning.
    """

    task_id: str
    attempts: int
    paths: tuple[str, ...]
    number: int | None = None

    def __str__(self) -> str:
        where = f"#{self.number} " if self.number is not None else ""
        outside = ", ".join(self.paths)
        return (
            f"{where}{self.task_id}: failed {self.attempts} time(s) on {outside}, "
            "which its ## Files does not list"
        )


@dataclass(frozen=True)
class Signal:
    """One task's progress-bearing state at one cycle.

    Everything a cycle can learn about a task that could distinguish "moved"
    from "did not", and nothing else - two signals comparing equal is the whole
    definition of a task that stood still. Note what is absent: the goal, the
    title and the verify command, all of which a human may edit mid-run without
    that meaning any work happened.
    """

    task_id: str
    status: TaskStatus
    attempts: int = 0
    #: The normalised text of the latest failure; "" when the task has not
    #: failed. Compared, never shown - `evidence` is the readable copy.
    failure: str = ""
    #: #34's base-update count for this task's pull request. A climbing number
    #: is a PR being rebased onto a base its siblings keep moving, which is
    #: work, not a stall.
    churn: int = 0
    number: int | None = None
    files: tuple[str, ...] = ()
    evidence: str = ""

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL

    @property
    def in_flight(self) -> bool:
        return self.status in IN_FLIGHT

    def moved(self, previous: Signal | None) -> bool:
        """Did anything about this task change since the previous cycle?

        `None` - a task the previous cycle had never heard of - counts as
        movement: it was just planned, or just adopted, and either way the
        ledger is not standing still.
        """
        if previous is None:
            return True
        return (
            self.status != previous.status
            or self.attempts != previous.attempts
            or self.churn != previous.churn
            or self.failure != previous.failure
        )

    def repeated(self, previous: Signal | None) -> bool:
        """Same failure, again. The one shape of "movement" that is not progress.

        An attempt counter that went up while the failure text stayed identical
        is the loop the ledger exists to catch, and it is also the case
        `moved` would otherwise report as progress forever - once per attempt,
        until the cap gives up and calls a human.

        Churn is the exception carved out for #34: a task whose pull request is
        being rebased has a reason for its attempt to move that is not another
        identical failure.
        """
        if previous is None or not self.failure:
            return False
        if self.churn != previous.churn:
            return False
        return self.failure == previous.failure and self.attempts > previous.attempts

    def blocker(self) -> Blocker | None:
        """Is this failure outside the task's own file set? See `Blocker`.

        Every path the failure names has to be outside `## Files` for this to
        fire. One path inside it is enough to make the failure something the
        worker could plausibly have fixed, and a false "a human is needed" is
        worse than a missed one - it parks a run that replanning would have
        rescued.
        """
        if self.attempts < BLOCKER_MIN_ATTEMPTS or not self.failure or not self.files:
            return None
        paths = mentioned_paths(self.evidence)
        if not paths:
            return None
        owned = {path.casefold() for path in self.files}
        if any(path.casefold() in owned for path in paths):
            return None
        return Blocker(
            task_id=self.task_id, attempts=self.attempts, paths=paths, number=self.number
        )


def _evidence(record: ResultRecord | None) -> str:
    """What a finished attempt said, in the order a human would read it.

    The reason first because it is the sentence somebody wrote for this, the
    verify output after because it is where the paths are. An attempt that
    exited 0 or 2 contributes nothing: exit 0 is a pull request and exit 2 is
    the host, and neither is the task failing.
    """
    if record is None or record.exit_code == EXIT_OK or not record.consumes_attempt:
        return ""
    return "\n".join(part for part in (record.reason, record.verify_output) if part)


@dataclass(frozen=True)
class Observation:
    """One cycle's reading of the ledger, reduced to what progress is measured on.

    Built from GitHub and the artifacts directory, never held across a cycle by
    anything but the caller: this is the one piece of state the progress ledger
    needs that the tracker cannot answer on its own, because "did it move" is a
    question about two readings and GitHub only ever shows the current one.
    """

    signals: Mapping[str, Signal]

    # --- construction ----------------------------------------------------

    @classmethod
    def of(
        cls,
        ledger: Ledger,
        *,
        results: Mapping[int, ResultRecord] | None = None,
        churn: Mapping[int, int] | None = None,
    ) -> Observation:
        """Observe a v2 ledger, plus what the workers and #34 left behind.

        `results` is `RunSummary.latest` - the newest attempt per issue, which
        is what the reconciler already reads for the same cycle - and `churn`
        is the per-issue count of base updates #34 has performed. Both are
        keyed by issue number because that is how the rest of v2 addresses an
        issue; the signals are keyed by task id because that is what identity
        is (§2), and what survives a replan.

        The newest record is used whatever attempt it belongs to, deliberately.
        The counter is bumped in the same cycle that observes the record that
        caused the bump (`reconcile._retry_or_give_up`), so a record whose
        attempt trails the entry's by one is the *reason* for that entry's
        current state, not stale evidence. Discarding it would erase the
        failure text at exactly the moment it starts mattering.
        """
        results = results or {}
        churn = churn or {}
        signals: dict[str, Signal] = {}
        for task_id, entry in ledger.entries.items():
            evidence = _evidence(results.get(entry.number))
            signals[task_id] = Signal(
                task_id=task_id,
                status=STATUS_BY_LABEL[entry.state_label],
                attempts=entry.attempt,
                failure=failure_signature(evidence),
                churn=int(churn.get(entry.number, 0)),
                number=entry.number,
                files=entry.files,
                evidence=evidence,
            )
        return cls(signals=signals)

    @classmethod
    def of_tasks(cls, tasks: Mapping[str, TaskRecord]) -> Observation:
        """Observe v1's in-memory ledger. Same ladder, older backing store.

        `graph.py` still runs, and it holds `TaskRecord`s rather than issues.
        The fields it carries are a subset of the ones above - no issue number,
        no pull request, so no churn - which is exactly why the v1 path reaches
        the model on cycles the v2 path settles by arithmetic.
        """
        signals: dict[str, Signal] = {}
        for task_id, task in tasks.items():
            evidence = task.get("last_error") or ""
            signals[task_id] = Signal(
                task_id=task.get("id", task_id),
                status=task.get("status", "pending"),
                attempts=int(task.get("attempts", 0) or 0),
                failure=failure_signature(evidence),
                files=tuple(task.get("files") or ()),
                evidence=evidence,
            )
        return cls(signals=signals)

    # --- what it says ----------------------------------------------------

    @property
    def empty(self) -> bool:
        return not self.signals

    @property
    def finished(self) -> bool:
        """Every task is done or abandoned. Not the same as satisfied."""
        return bool(self.signals) and all(s.terminal for s in self.signals.values())

    @property
    def satisfied(self) -> bool:
        """Every task merged. The only definition of done this module accepts."""
        return bool(self.signals) and all(s.status == "verified" for s in self.signals.values())

    @property
    def in_flight(self) -> tuple[str, ...]:
        return tuple(sorted(t for t, s in self.signals.items() if s.in_flight))

    @property
    def blockers(self) -> tuple[Blocker, ...]:
        """Tasks failing on files they were never given. See `Blocker`."""
        found = (signal.blocker() for _, signal in sorted(self.signals.items()))
        return tuple(blocker for blocker in found if blocker is not None)

    def moved(self, previous: Observation | None) -> tuple[str, ...]:
        """Ids that changed since `previous`, including ones that just appeared.

        A task that *left* the ledger is movement too - a human closed it, or a
        previous replan retired it - so the comparison runs over the union.
        """
        if previous is None:
            return ()
        ids = set(self.signals) | set(previous.signals)
        return tuple(
            sorted(
                task_id
                for task_id in ids
                if task_id not in self.signals
                or self.signals[task_id].moved(previous.signals.get(task_id))
            )
        )

    def repeats(self, previous: Observation | None) -> tuple[str, ...]:
        """Ids that failed again, identically. See `Signal.repeated`."""
        if previous is None:
            return ()
        return tuple(
            sorted(
                task_id
                for task_id, signal in self.signals.items()
                if signal.repeated(previous.signals.get(task_id))
            )
        )

    def prompt(self) -> str:
        """The task list as the model sees it, in v1's shape and order."""
        return "\n".join(
            f"- {signal.task_id}: status={signal.status} attempts={signal.attempts} "
            f"error={(signal.evidence or '')[:200]}"
            for _, signal in sorted(self.signals.items())
        )


# --------------------------------------------------------------------------
# The judgement
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """One cycle's answer to v1's four questions, plus how it was reached.

    `judgement` is the v1 record verbatim, because `graph.py` routes on it and
    `SwarmState` stores it. Everything around it is what v2 needs and v1 had
    nowhere to put: whether a model was consulted, whether it could be, and
    which tasks are stuck on something a replan cannot move.
    """

    judgement: ProgressJudgement
    stalls: int = 0
    observation: Observation | None = None
    blockers: tuple[Blocker, ...] = ()
    reason: str = ""
    #: A model was asked. False on every deterministic path, and the number
    #: worth watching over a run: it is one ~6.7 s model swap each.
    consulted: bool = False
    #: A model was asked and could not answer. Not a stall - see `stalled`.
    unresolved: bool = False

    @property
    def satisfied(self) -> bool:
        return self.judgement.request_satisfied

    @property
    def stalled(self) -> bool:
        """No progress, or the same failure again.

        An unresolved judgement is deliberately not a stall. A model that could
        not be reached is an infrastructure fact - `docs/issue-contract.md` §4
        makes the same call about exit 2 - and counting it would let a broken
        Ollama replan a run whose tasks were fine.
        """
        if self.unresolved:
            return False
        return not self.judgement.progress_being_made or self.judgement.in_loop

    @property
    def needs_human(self) -> bool:
        return bool(self.blockers)

    def should_replan(self, *, max_stalls: int = SETTINGS.max_stalls) -> bool:
        """Has this run earned a replan? `replan.py` is the caller.

        Four ways to answer no, and each of them is a way a replan makes things
        worse: the run is finished, the run is moving, the model that would
        write the new plan is unreachable, or the thing standing in the way is
        a file the plan cannot hand anybody. Only after `max_stalls` cycles of
        genuine stall does rewriting the tracker beat waiting one more cycle.
        """
        if self.satisfied or self.unresolved or self.needs_human or not self.stalled:
            return False
        return self.stalls >= max(int(max_stalls), 1)

    def summary(self) -> str:
        parts = [
            f"satisfied={self.judgement.request_satisfied}",
            f"progress={self.judgement.progress_being_made}",
            f"loop={self.judgement.in_loop}",
            f"stalls={self.stalls}",
            "model" if self.consulted else "arithmetic",
        ]
        if self.unresolved:
            parts.append("judgement unavailable")
        if self.blockers:
            parts.append(f"{len(self.blockers)} task(s) need a human")
        return f"{', '.join(parts)}: {self.reason or self.judgement.reason}"


def _verdict(
    observation: Observation,
    *,
    satisfied: bool,
    progress: bool,
    in_loop: bool,
    reason: str,
    stalls: int,
    consulted: bool = False,
    unresolved: bool = False,
) -> Verdict:
    """Assemble a verdict and settle the stall count in one place.

    Satisfaction is overwritten with the ledger's own answer on every path,
    including the model's - see the module docstring. The stall increment is
    here rather than at each call site because a rule that forgot it would
    produce a run that stalls forever without ever reaching the replan.
    """
    judgement = ProgressJudgement(
        request_satisfied=satisfied and observation.satisfied,
        progress_being_made=progress,
        in_loop=in_loop,
        reason=reason,
    )
    verdict = Verdict(
        judgement=judgement,
        stalls=stalls,
        observation=observation,
        blockers=observation.blockers,
        reason=reason,
        consulted=consulted,
        unresolved=unresolved,
    )
    return verdict if not verdict.stalled else replace(verdict, stalls=stalls + 1)


def judge(
    observation: Observation,
    previous: Observation | None = None,
    *,
    objective: str = "",
    stalls: int = 0,
    round_index: int = 0,
    oracle: Oracle | None = None,
) -> Verdict:
    """Answer the four questions, spending a model call only if arithmetic cannot.

    The ladder, in order, and the order is the argument:

    1. **An empty ledger.** Nothing was planned, or everything was retired.
       That is a stall by definition and there is nothing to ask about.
    2. **Everything terminal.** Satisfied when every task merged, finished
       either way. v1's fast path, unchanged.
    3. **The same failure again, and nothing else.** An attempt counter
       climbing on an identical error is movement that is not progress - the
       exact shape v1's prompt describes as a loop - so it is settled before
       movement is read as progress.
    4. **Anything moved.** A label transition, an attempt, a rebase (#34). The
       run is doing something; a model would only be asked to agree.
    5. **Anything in flight.** A container is running or a PR is open. Workers
       take minutes and cycles take seconds, so a cycle that sees no change
       while a worker is mid-task has learned nothing worth a swap.
    6. **A task walled off from its own fix.** `Blocker`, and a question no
       model can answer better than the file list already did.
    7. **Otherwise, ask.** Tasks sitting in `ready` while nothing dispatches,
       failures with different shapes each time, a first cycle with no history:
       genuinely ambiguous, and the one case worth 6.7 seconds.
    """
    if observation.empty:
        return _verdict(
            observation,
            satisfied=False,
            progress=False,
            in_loop=False,
            reason="the ledger is empty; there is nothing to make progress on",
            stalls=stalls,
        )

    if observation.finished:
        return _verdict(
            observation,
            satisfied=True,
            progress=True,
            in_loop=False,
            reason="all tasks reached a terminal state",
            stalls=stalls,
        )

    moved = observation.moved(previous)
    repeats = observation.repeats(previous)
    # A repeat is always movement - the attempt counter went up - so the loop
    # rule is not "something repeated" but "everything that moved was a
    # repeat". One task merging while another fails identically is a run making
    # progress, and replanning it would discard the merge's siblings.
    if repeats and set(moved) <= set(repeats):
        return _verdict(
            observation,
            satisfied=False,
            progress=False,
            in_loop=True,
            reason=f"failed again identically: {', '.join(repeats)}",
            stalls=stalls,
        )

    if moved:
        return _verdict(
            observation,
            satisfied=False,
            progress=True,
            in_loop=False,
            reason=f"the ledger moved since the last cycle: {', '.join(moved)}",
            stalls=stalls,
        )

    running = observation.in_flight
    if running:
        return _verdict(
            observation,
            satisfied=False,
            progress=True,
            in_loop=False,
            reason=f"{len(running)} task(s) in flight: {', '.join(running)}",
            stalls=stalls,
        )

    if observation.blockers:
        # A task failing on a file it was never given is not a question about
        # this run's progress, and the model has no more idea what to do about
        # it than the arithmetic does. Reported, not asked about.
        return _verdict(
            observation,
            satisfied=False,
            progress=False,
            in_loop=False,
            reason="; ".join(str(blocker) for blocker in observation.blockers),
            stalls=stalls,
        )

    try:
        llm = oracle if oracle is not None else structured(orchestrator_llm(), ProgressJudgement)
        answer = llm.invoke(
            [
                ("system", SYSTEM),
                (
                    "human",
                    f"Objective: {objective}\n\nRound {round_index}.\n"
                    f"Tasks:\n{observation.prompt()}",
                ),
            ]
        )
    except Exception as exc:  # noqa: BLE001 - any transport failure reads the same
        # No stall, and no replan: the planner would call the same model that
        # just refused to answer, and a run that rewrites its tracker because
        # Ollama was restarting is the expensive version of this failure.
        return _verdict(
            observation,
            satisfied=False,
            progress=False,
            in_loop=False,
            reason=f"judgement failed: {exc}",
            stalls=stalls,
            consulted=True,
            unresolved=True,
        )

    return _verdict(
        observation,
        satisfied=answer.request_satisfied,
        progress=answer.progress_being_made,
        in_loop=answer.in_loop,
        reason=answer.reason or "the model judged the run",
        stalls=stalls,
        consulted=True,
    )


# --------------------------------------------------------------------------
# The node
# --------------------------------------------------------------------------


def judge_node(state: SwarmState) -> dict:
    """v1's node, unchanged in signature: `graph.py` routes on what it returns.

    It observes the in-memory task ledger rather than the tracker, and holds no
    previous observation because `SwarmState` has nowhere to put one - so the
    ladder's history-dependent rungs (3 and 4) never fire here and the v1 path
    reaches the model exactly where it always did.
    """
    tasks = state.get("tasks", {})
    rnd = state.get("round", 0) + 1
    verdict = judge(
        Observation.of_tasks(tasks),
        objective=state.get("objective", ""),
        stalls=state.get("stalls", 0),
        round_index=rnd,
    )
    return {
        "round": rnd,
        "stalls": verdict.stalls,
        "last_judgement": verdict.judgement,
        "events": [
            f"round {rnd}: satisfied={verdict.judgement.request_satisfied} "
            f"progress={verdict.judgement.progress_being_made} "
            f"loop={verdict.judgement.in_loop} "
            f"stalls={verdict.stalls}/{SETTINGS.max_stalls}"
        ],
    }
