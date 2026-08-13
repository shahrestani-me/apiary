"""Planner node - builds (and rebuilds) the task ledger."""

from __future__ import annotations

from ..llm import orchestrator_llm, structured
from ..state import Plan, SwarmState, TaskRecord

SYSTEM = """You decompose a software objective into independent coding tasks.

HARD RULES:
- Two tasks must NEVER list the same file. If they would, merge them into one task.
- Prefer 2-4 tasks. Fewer, larger tasks beat many tiny ones.
- Each task must be completable by editing only the files it lists.
- Use depends_on only when a task literally cannot start before another finishes.

Return JSON only."""

REPLAN_SUFFIX = """

This is a REPLAN. The previous attempt stalled. Here is what happened:
{failures}

Produce a different decomposition. Do not repeat the approach that failed."""


def plan_node(state: SwarmState) -> dict:
    objective = state["objective"]
    existing = state.get("tasks") or {}

    prompt = SYSTEM
    if existing:
        failures = "\n".join(
            f"- {t['id']} ({t.get('status')}): {t.get('last_error', 'no error recorded')[:300]}"
            for t in existing.values()
            if t.get("status") in {"failed", "abandoned"}
        ) or "- no specific errors recorded; work simply did not converge"
        prompt = SYSTEM + REPLAN_SUFFIX.format(failures=failures)

    llm = structured(orchestrator_llm(), Plan)
    plan: Plan = llm.invoke(
        [("system", prompt), ("human", f"Objective:\n{objective}")]
    )

    tasks: dict[str, TaskRecord] = {}
    for t in plan.tasks:
        tasks[t.id] = TaskRecord(
            id=t.id,
            goal=t.goal,
            files=t.files,
            depends_on=t.depends_on,
            status="pending",
            attempts=0,
        )

    return {
        "tasks": tasks,
        "plan_reasoning": plan.reasoning,
        "round": state.get("round", 0),
        "events": [f"planned {len(tasks)} task(s): {', '.join(tasks)}"],
    }
