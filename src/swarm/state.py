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
    # The whole brief, because it is the whole brief: the worker is handed this
    # string and the file list and nothing else - not the objective, not the
    # other tasks. "One sentence" is what this used to ask for, and one sentence
    # is a worker guessing at the other ninety percent.
    goal: str = Field(
        description=(
            "the complete specification for this task, several sentences or a "
            "short list: what must be true when done, the names and signatures "
            "that matter, behaviour at the edges, and what the tests must "
            "assert. The worker sees only this and the file list."
        )
    )
    # Required, with no default. `normalise` rejects a task that lists no files
    # - a worker may only write what its task declares, so a task with none can
    # do nothing - and a default made that rejection silent: the model, asked
    # for detailed goals, spent its answer on prose and simply stopped emitting
    # this field. Nine of ten tasks in one observed plan had no files, and the
    # plan reached the reconciler as a single task with no error anywhere.
    # Without a default the schema-constrained decoder cannot leave it out.
    files: list[str] = Field(
        description=(
            "repo-relative paths this task creates or modifies, at least one. "
            "Must not overlap any other task's files."
        ),
    )
    depends_on: list[str] = Field(
        default_factory=list, description="ids of tasks that must finish first"
    )
    # The description is the prompt: this field is filled by a schema-forced
    # model call, so what the model is told about it lives here and nowhere
    # else. D5's defaults are stated rather than left to be inferred, and
    # "react" means React **web** - React Native is out of scope for #87.
    stack: str | None = Field(
        default=None,
        description=(
            "which toolchain this task's code is written in: 'python', 'node' "
            "or 'react'. Frontend or UI work is 'react' (React web, not React "
            "Native); backend, CLI and library work is 'python'. Omit when the "
            "task does not add code in a stack of its own."
        ),
    )


class Plan(BaseModel):
    tasks: list[PlannedTask]
    reasoning: str = Field(default="", description="one short paragraph on the decomposition")


class FileEdit(BaseModel):
    """One whole-file write - or, when `content` is empty, a deletion.

    Empty content deletes the file rather than emptying it (see
    `worker.edit.apply_edits` for the full weighing): a `deleted: bool` field
    would change this structured-decoding schema for every model call, and an
    intentionally empty source file is not an output this tool should produce
    anyway.
    """

    path: str = Field(description="repo-relative path")
    content: str = Field(
        description=(
            "the COMPLETE new contents of the file; empty content DELETES the file"
        )
    )


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


class ObjectiveAssessment(BaseModel):
    """Was the *objective* met, as opposed to the *plan* being exhausted.

    Deliberately a different question from `ProgressJudgement`, which asks
    whether a run is moving. A plan can run to completion, every task merged,
    and still leave the objective half-done - the decomposition only ever
    covered what the planner thought of on its first read. So this is asked once
    the ledger has nothing live left in it, and `missing` is what the follow-up
    plan is written from (`orchestrator/goal.py`).
    """

    objective_met: bool
    missing: list[str] = Field(
        default_factory=list,
        description="one line per thing the objective asks for that is not there yet",
    )
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
