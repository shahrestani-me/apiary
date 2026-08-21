"""The last thing a run says when it cannot finish on its own (#293).

**The gap this fills is not logging.** Every escalation already writes a reason
into `events.jsonl` and a comment onto the issue, and `swarm show` renders both
afterwards. What no part of the system did was *address the operator*: a run that
needed a decision ended with the same one-line outcome as a run that merely hit
its cycle cap, so "apiary is stuck" and "apiary is waiting for you" were
indistinguishable from the terminal. The requirement this is built against says
the rare failure should "explain the problem" and then be replanned; the explain
half did not exist.

**It reports, it does not decide, and it does not block.** Three reasons, and the
third is the one that shaped the module:

- The decision is a human's, and the action they take already exists:
  `swarm reset <ref>` hands a task its retry budget back, and re-invoking `swarm
  run` attaches to the same ledger rather than replanning. So this needs to name
  the tasks and print the command, not grow a second way to do either.
- Blocking a process on stdin would make an unattended run hang forever, which
  is a worse failure than the one being fixed - and apiary runs unattended by
  design.
- The classification is already computed. Each escalation's `reason` was written
  by the rule that made it (`checks._retry_or_give_up`, the infrastructure
  ceiling, `_decide_empty`), and re-deriving it here from result records would be
  a second speller of a judgement that already exists - the drift `lifecycle.py`
  warns about, one layer up.

**Read from what landed, across every cycle.** `CycleReport.escalated` is one
cycle's answer and an escalation is permanent, so the report is folded over the
whole run. Deduplicated by ref with the last write winning, because a task can
be escalated, revived by the goal gate, and escalated again - and the sentence a
human needs is the one about why it died the last time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..taskref import TaskRef
from .reachable import reasons, stranded

__all__ = ["CAUSES", "UNCLASSIFIED", "Decision", "DecisionReport", "classify", "decisions"]

#: How an escalation's own reason maps to the sentence an operator can act on,
#: as (fragment, headline) pairs tried in order.
#:
#: Matched on the reason string rather than on a code, because the reason *is*
#: the classification: the rule that escalated the task wrote it, and the three
#: causes are already distinguishable in it. A stored enum would be a fourth
#: place the same judgement lives.
#:
#: The order is the priority, and the first two are the ones that matter: an
#: unsatisfiable task and a broken machine look identical in a status field and
#: call for opposite responses - fix the plan, or fix the plumbing - while
#: "the code was wrong" is the residue that actually wants a human reading a
#: diff.
CAUSES: tuple[tuple[str, str], ...] = (
    (
        # Ahead of the general infrastructure line because it is a *diagnosis*
        # rather than a category, and this one has now cost two debugging
        # sessions: a worker image built without its provider extra resolves the
        # model name against Ollama and gets a flat 400. The images bake in `pip
        # install .`, so a stale image is stale apiary code as well - the react
        # image that produced this was four days old and still writing
        # `swarm/issue-N` branches.
        "invalid model name",
        "the worker image cannot resolve the model - it is built without the "
        "provider extra, or is simply stale; rebuild it",
    ),
    (
        "infrastructure",
        "the machine, not the code - no attempt of this task ever ran",
    ),
    (
        "outside this issue's ## Files",
        "a file this task was never allowed to edit; re-scoping it is the fix",
    ),
    (
        "no check run was ever created",
        "nothing gated the pull request, so nothing verified it",
    ),
    (
        "conflicts with",
        "the branch no longer applies to the base and the retries are spent",
    ),
    (
        "starved",
        "the work is fine; the base moved under it faster than it could land",
    ),
    (
        "closed without merging",
        "its pull request was closed by hand",
    ),
    (
        "the verify command failed",
        "the gate went red the same way every time; the plan is the suspect",
    ),
)

#: What is said about a reason none of the above matched. Deliberately not a
#: guess: an unrecognised escalation is a rule this table has not been taught,
#: and inventing a cause for it would be the one thing worse than saying so.
UNCLASSIFIED = "escalated for a reason this report has no summary for; read it above"


def classify(reason: str) -> str:
    """The actionable sentence for one escalation reason. See `CAUSES`."""
    lowered = (reason or "").casefold()
    for fragment, headline in CAUSES:
        if fragment.casefold() in lowered:
            return headline
    return UNCLASSIFIED


@dataclass(frozen=True)
class Decision:
    """One task a human has to decide about, and everything needed to decide."""

    ref: TaskRef
    task_id: str
    reason: str
    cause: str
    #: Tasks that cannot run again until this one is resolved, with the sentence
    #: naming why. Empty is common and means the failure stranded nothing - the
    #: plan simply lost a leaf.
    stranded: tuple[tuple[TaskRef, str], ...] = ()

    def lines(self, *, repo: str = "") -> list[str]:
        out = [f"  {self.ref} {self.task_id}", f"     why    {self.cause}"]
        # The rule's own words, kept verbatim and second: the summary above is
        # this module's paraphrase and the operator has to be able to see what
        # was actually decided, not only how it was labelled.
        out.append(f"     said   {self.reason}")
        for ref, why in self.stranded:
            out.append(f"     blocks {ref} - {why}")
        scope = f" --repo {repo}" if repo else ""
        out.append(f"     retry  swarm reset '{self.ref}'{scope}")
        return out


@dataclass(frozen=True)
class DecisionReport:
    """Every task this run is waiting on a human for."""

    decisions: tuple[Decision, ...] = ()
    repo: str = ""

    def __bool__(self) -> bool:
        return bool(self.decisions)

    @property
    def refs(self) -> tuple[TaskRef, ...]:
        return tuple(item.ref for item in self.decisions)

    def text(self) -> str:
        """The block `swarm run` prints last. Empty string when there is nothing
        to decide, so the caller can print it unconditionally."""
        if not self.decisions:
            return ""
        head = (
            f"» this run needs a decision on {len(self.decisions)} task(s). "
            "Nothing else in it can proceed without one."
        )
        body: list[str] = []
        for item in self.decisions:
            body += ["", *item.lines(repo=self.repo)]
        tail = [
            "",
            "  Reset the ones worth another attempt, then re-run the same command:",
            "  the run attaches to this ledger rather than replanning it.",
        ]
        return "\n".join([head, *body, *tail])


def decisions(reports: Sequence[Any], *, repo: str = "") -> DecisionReport:
    """Fold a finished run's cycles into the decisions it is waiting on.

    Pure over the reports, so the terminal block and anything that renders this
    later cannot disagree about what the run decided.
    """
    if not reports:
        return DecisionReport(repo=repo)

    # Last write wins: a task escalated, revived by the goal gate and escalated
    # again should be explained by the failure that actually ended it.
    landed: dict[TaskRef, Any] = {}
    for report in reports:
        for transition in getattr(report, "escalated", ()):
            landed[transition.ref] = transition
    if not landed:
        return DecisionReport(repo=repo)

    # The *last* cycle's ledger and belief, because stranding is a fact about
    # where the run stopped: a task stranded in cycle 3 and revived in cycle 9 is
    # not stranded, and the final pair is the only one that knows that.
    final = reports[-1]
    ledger = getattr(final, "ledger", None)
    belief = getattr(final, "belief", None)
    behind = _behind(ledger, belief)

    ordered = sorted(landed.items(), key=lambda pair: str(pair[0]))
    return DecisionReport(
        decisions=tuple(
            Decision(
                ref=ref,
                task_id=transition.task_id,
                reason=transition.reason,
                cause=classify(transition.reason),
                stranded=behind.get(transition.task_id, ()),
            )
            for ref, transition in ordered
        ),
        repo=repo,
    )


def _behind(ledger: Any, belief: Any) -> dict[str, tuple[tuple[TaskRef, str], ...]]:
    """Stranded tasks, grouped under the failure nearest to each of them.

    Grouped by the *dependency* rather than listed flat, because the operator's
    question is "what does resetting this buy me" and the answer is this list.
    A task stranded two hops down is attributed to the failure it names, which
    is `reachable.reasons`' rule and the same reason: the transitive version is
    a paragraph nobody finishes.
    """
    if ledger is None or belief is None or not getattr(ledger, "entries", None):
        return {}
    ids = stranded(ledger, belief)
    if not ids:
        return {}
    said = reasons(ledger, belief, ids)
    grouped: dict[str, list[tuple[TaskRef, str]]] = {}
    entries = ledger.entries
    for task_id in sorted(ids):
        entry = entries[task_id]
        for dependency in entry.depends_on:
            grouped.setdefault(dependency, []).append((entry.ref, said[task_id]))
    return {key: tuple(value) for key, value in grouped.items()}
