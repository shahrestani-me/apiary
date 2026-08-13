"""Graph state: the task ledger and the progress ledger.

This is the Magentic pattern, lifted from Microsoft's Magentic-One and
reimplemented on LangGraph:

  Task ledger     - the plan. Revisable.
  Progress ledger - after every round: are we done? are we moving? are we
                    looping? Drives stall detection and replanning.

Keeping these as explicit typed state (rather than letting them live inside
an LLM conversation) is the whole reason to use LangGraph here. It means the
orchestrator's decisions are inspectable, checkpointable, and testable.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

TaskStatus = Literal["pending", "running", "verified", "failed", "abandoned"]


# --------------------------------------------------------------------------
# Structured-output schemas (what we force the local model to emit)
# --------------------------------------------------------------------------

class PlannedTask(BaseModel):
    """One unit of work handed to exactly one worker."""

    id: str = Field(description="short kebab-case id, e.g. 'add-retry-logic'")
    goal: str = Field(description="one sentence: what must be true when done")
    files: list[str] = Field(
        default_factory=list,
        description="repo-relative paths this task may modify. Must not overlap other tasks.",
    )
    depends_on: list[str] = Field(
        default_factory=list, description="ids of tasks that must finish first"
    )


class Plan(BaseModel):
    tasks: list[PlannedTask]
    reasoning: str = Field(default="", description="one short paragraph on the decomposition")


class FileEdit(BaseModel):
    path: str = Field(description="repo-relative path")
    content: str = Field(description="the COMPLETE new contents of the file")


class WorkerOutput(BaseModel):
    """Small models cannot reliably drive a free-form tool loop.

    So we do not ask for tool calls. We ask for whole-file rewrites in a
    fixed schema and apply them ourselves. Far higher success rate on
    sub-10B models, at the cost of some token waste on large files.
    """

    edits: list[FileEdit] = Field(default_factory=list)
    notes: str = Field(default="", description="what changed and why, 2 sentences max")


class ProgressJudgement(BaseModel):
    request_satisfied: bool
    progress_being_made: bool
    in_loop: bool
    reason: str = Field(default="", description="one sentence")


# --------------------------------------------------------------------------
# Runtime records
# --------------------------------------------------------------------------

class TaskRecord(TypedDict, total=False):
    id: str
    goal: str
    files: list[str]
    depends_on: list[str]
    status: TaskStatus
    attempts: int
    branch: str
    worktree: str
    last_error: str
    notes: str


def _merge_tasks(
    left: dict[str, TaskRecord] | None, right: dict[str, TaskRecord] | None
) -> dict[str, TaskRecord]:
    """Reducer so parallel workers can each write back their own task record
    without clobbering their siblings."""
    merged = dict(left or {})
    for task_id, record in (right or {}).items():
        merged[task_id] = {**merged.get(task_id, {}), **record}
    return merged


class SwarmState(TypedDict, total=False):
    # input
    objective: str

    # task ledger
    tasks: Annotated[dict[str, TaskRecord], _merge_tasks]
    plan_reasoning: str

    # progress ledger
    round: int
    stalls: int
    verified_count: int
    last_judgement: ProgressJudgement | None

    # audit trail
    events: Annotated[list[str], operator.add]

    # output
    outcome: str
    merged_branches: list[str]

    # per-Send payload (set only on the worker branch of the graph)
    task: TaskRecord
    base_branch: str
