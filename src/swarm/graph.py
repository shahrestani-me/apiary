"""The orchestration graph.

    setup -> plan -> dispatch -=fan out=-> worker  ->  verify -> judge -> ?
                       ^                                                 |
                       |________________ retry ______________            |
                                                            |            |
                       replan <-- stalled ------------------             |
                                                                         v
                                                              integrate -> END

Deliberately boring. The orchestrator does arithmetic and routing; it does
not "reason" about coordination. All the model intelligence lives in the
workers. Emergent orchestration is where multi-agent systems burn hours.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .config import SETTINGS
from .nodes import integrate_node, judge_node, plan_node, verify_node, worker_node
from .state import SwarmState
from .worktree import base_branch, ensure_repo


def setup_node(state: SwarmState) -> dict:
    repo = ensure_repo(SETTINGS.repo_path)
    base = base_branch(repo)
    return {
        "base_branch": base,
        "round": 0,
        "stalls": 0,
        "verified_count": 0,
        "events": [f"repo={repo} base_branch={base} "
                   f"orchestrator={SETTINGS.orchestrator_model} worker={SETTINGS.worker_model}"],
    }


def dispatch_node(state: SwarmState) -> dict:
    """Pure pass-through; the fan-out happens on the conditional edge."""
    return {}


def fan_out(state: SwarmState):
    """Send one Send() per runnable task, capped at max_workers_parallel.

    LangGraph does not throttle Send fan-out, so we throttle by only
    dispatching a slice per round. Remaining tasks go out next round.
    """
    tasks = state.get("tasks", {})
    verified = {tid for tid, t in tasks.items() if t.get("status") == "verified"}

    runnable = [
        t
        for t in tasks.values()
        if t.get("status") in {"pending", "failed"}
        and t.get("attempts", 0) < SETTINGS.max_attempts_per_task
        and set(t.get("depends_on", [])).issubset(verified)
    ]

    if not runnable:
        return "verify"

    batch = runnable[: SETTINGS.max_workers_parallel]
    return [
        Send("worker", {"task": t, "base_branch": state.get("base_branch", "HEAD")})
        for t in batch
    ]


def route_after_judge(state: SwarmState) -> str:
    judgement = state.get("last_judgement")
    rnd = state.get("round", 0)
    stalls = state.get("stalls", 0)

    if judgement is not None and judgement.request_satisfied:
        return "integrate"
    if rnd >= SETTINGS.max_rounds:
        return "integrate"

    tasks = state.get("tasks", {})
    if tasks and all(t.get("status") in {"verified", "abandoned"} for t in tasks.values()):
        return "integrate"

    if stalls >= SETTINGS.max_stalls:
        return "plan"  # replan with the failure history in context

    return "dispatch"


def build_graph(checkpointer=None):
    g = StateGraph(SwarmState)

    g.add_node("setup", setup_node)
    g.add_node("plan", plan_node)
    g.add_node("dispatch", dispatch_node)
    g.add_node("worker", worker_node)
    g.add_node("verify", verify_node)
    g.add_node("judge", judge_node)
    g.add_node("integrate", integrate_node)

    g.add_edge(START, "setup")
    g.add_edge("setup", "plan")
    g.add_edge("plan", "dispatch")
    g.add_conditional_edges("dispatch", fan_out, ["worker", "verify"])
    g.add_edge("worker", "verify")
    g.add_edge("verify", "judge")
    g.add_conditional_edges("judge", route_after_judge, ["dispatch", "plan", "integrate"])
    g.add_edge("integrate", END)

    return g.compile(checkpointer=checkpointer)
