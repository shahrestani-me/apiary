"""Progress-ledger node - stall detection.

After each round the orchestrator answers four questions. This is what stops
a swarm burning an afternoon making no progress, and it is the single
cheapest reliability win in the whole design.

Note the deterministic short-circuits: we only ask the model when the answer
is genuinely ambiguous. Never spend an LLM call on a question arithmetic can
answer.
"""

from __future__ import annotations

from ..config import SETTINGS
from ..llm import orchestrator_llm, structured
from ..state import ProgressJudgement, SwarmState

SYSTEM = """You judge whether a multi-agent coding run is progressing.
Be strict. If the same tasks keep failing with the same error, that is a loop.
Return JSON only."""


def judge_node(state: SwarmState) -> dict:
    tasks = state.get("tasks", {})
    rnd = state.get("round", 0) + 1
    stalls = state.get("stalls", 0)

    statuses = [t.get("status") for t in tasks.values()]
    done = all(s in {"verified", "abandoned"} for s in statuses) if statuses else False

    # Deterministic fast paths - no model call needed.
    if done:
        judgement = ProgressJudgement(
            request_satisfied=all(s == "verified" for s in statuses),
            progress_being_made=True,
            in_loop=False,
            reason="all tasks reached a terminal state",
        )
    else:
        summary = "\n".join(
            f"- {t['id']}: status={t.get('status')} attempts={t.get('attempts', 0)} "
            f"error={(t.get('last_error') or '')[:200]}"
            for t in tasks.values()
        )
        try:
            llm = structured(orchestrator_llm(), ProgressJudgement)
            judgement = llm.invoke(
                [
                    ("system", SYSTEM),
                    ("human", f"Objective: {state['objective']}\n\nRound {rnd}.\nTasks:\n{summary}"),
                ]
            )
        except Exception as exc:  # noqa: BLE001
            judgement = ProgressJudgement(
                request_satisfied=False,
                progress_being_made=False,
                in_loop=False,
                reason=f"judgement failed: {exc}",
            )

    if not judgement.progress_being_made or judgement.in_loop:
        stalls += 1

    return {
        "round": rnd,
        "stalls": stalls,
        "last_judgement": judgement,
        "events": [
            f"round {rnd}: satisfied={judgement.request_satisfied} "
            f"progress={judgement.progress_being_made} loop={judgement.in_loop} "
            f"stalls={stalls}/{SETTINGS.max_stalls}"
        ],
    }
