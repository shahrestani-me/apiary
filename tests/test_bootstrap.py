"""Tests for the bootstrap phase - the first issue of every greenfield plan.

Two properties carry this file.

**The bootstrap is a task, not a special case.** Everything downstream - the
contract parser, the readiness pass, the dispatcher, the worker - must be able
to treat it as an ordinary issue, because the whole point of making it a
*phase* rather than a `Stack.build` implementation is that no other module has
to learn about it. So the assertions below are mostly "it round-trips through
`parse_contract`" and "it is ordered and blocked like anything else".

**No model, anywhere.** `choose_stack` takes the same `llm=None` injection seam
as `edit.propose_edits`, and every test here passes a double. A bootstrap test
that needed Ollama running to check that a dashboard resolves to React is a
test that does not run - and it is not hypothetical: leaving the real call live
turned `test_cli_run.py` into a 118-second suite before the seam was used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import pytest

from swarm.github.ledger import DEFAULT_STACK, KNOWN_STACKS, parse_contract
from swarm.greenfield.bootstrap import (
    BOOTSTRAP_FILES,
    BOOTSTRAP_TASK_ID,
    MAX_BOOTSTRAP_FILES,
    Bootstrap,
    StackChoice,
    choose_stack,
)
from swarm.nodes.planner import Draft, normalise, order_drafts, with_bootstrap
from swarm.state import PlannedTask

VERIFY = "python -m pytest -q"


@dataclass
class Answers:
    """An oracle with a scripted answer, recording what it was asked."""

    stack: str = "python"
    reason: str = "because"
    asked: list[Any] = field(default_factory=list)

    def invoke(self, messages: Sequence[tuple[str, str]]) -> StackChoice:
        self.asked.append(messages)
        return StackChoice(stack=self.stack, reason=self.reason)


class Unreachable:
    """Ollama restarting, a socket refusing, a model that is not pulled."""

    def invoke(self, messages: Sequence[tuple[str, str]]) -> StackChoice:
        raise RuntimeError("connection refused")


class Nonsense:
    """A model answering outside the vocabulary. `format` should make this
    impossible; the fallback exists because "should" is not "cannot"."""

    def invoke(self, messages: Sequence[tuple[str, str]]) -> StackChoice:
        return StackChoice(stack="haskell")


# --------------------------------------------------------------------------
# Which stack the prompt implies (D5)
# --------------------------------------------------------------------------


def test_a_ui_prompt_resolves_to_react():
    oracle = Answers(stack="react")

    assert choose_stack("a dashboard for warehouse pickers", llm=oracle) == "react"
    # The prompt reaches the model, rather than a summary of it: "a dashboard
    # for warehouse pickers" is a judgement, and a word list is what #87 is
    # removing.
    assert "warehouse pickers" in oracle.asked[0][-1][1]


def test_a_service_prompt_resolves_to_python():
    assert choose_stack("an API that stores receipts", llm=Answers(stack="python")) == "python"


def test_react_means_react_web():
    """D5 was narrowed deliberately: React Native is out of #87's scope, and a
    model that answered "react" meaning Native would get a Node image."""
    assert "react" in KNOWN_STACKS
    assert BOOTSTRAP_FILES["react"] != BOOTSTRAP_FILES["node"]
    assert any(path.endswith(".jsx") for path in BOOTSTRAP_FILES["react"])


def test_an_unreachable_model_falls_back_rather_than_failing_the_run():
    """The right failure. Python is the stack whose image this host has always
    had, and a greenfield run that refused to start because Ollama hiccuped
    during a one-word classification would be worse than one the operator
    re-runs with `--stack`."""
    assert choose_stack("anything", llm=Unreachable()) == DEFAULT_STACK


def test_an_answer_outside_the_vocabulary_falls_back():
    assert choose_stack("anything", llm=Nonsense()) == DEFAULT_STACK


def test_an_explicit_stack_is_not_a_question_for_the_model():
    """`swarm run --stack node` is the operator saying what the repository is.
    Asking anyway would spend a model call to maybe disagree with them."""
    oracle = Answers(stack="react")

    bootstrap = Bootstrap.for_prompt("a dashboard", stack="node", llm=oracle)

    assert bootstrap.stack == "node"
    assert oracle.asked == []


# --------------------------------------------------------------------------
# What the bootstrap is allowed to write
# --------------------------------------------------------------------------


def test_the_generated_set_is_bounded_by_the_file_list():
    """#88's constraint, and the reason it is a list rather than a limit: the
    generated set has to be bounded by something a human wrote down, not by a
    context window."""
    for stack, files in BOOTSTRAP_FILES.items():
        assert files, stack
        assert len(files) <= MAX_BOOTSTRAP_FILES, stack


def test_every_declarable_stack_can_be_bootstrapped():
    assert set(BOOTSTRAP_FILES) == KNOWN_STACKS


def test_the_bootstrap_does_not_redeclare_what_provisioning_committed():
    """`provision.files()` writes README.md, LICENSE and the workflow in the
    initial commit. A bootstrap redeclaring them would have its edits refused
    by `apply_edits` as outside its own file set, and a worker may never write
    `.github/workflows/*` at all - one that can edit CI can edit its grader."""
    for stack, files in BOOTSTRAP_FILES.items():
        assert "README.md" not in files, stack
        assert "LICENSE" not in files, stack
        assert not any(path.startswith(".github/") for path in files), stack


def test_the_goal_quotes_the_prompt_rather_than_paraphrasing_it():
    """It is the only statement of what the project *is*, and a planner that
    summarised it would be deciding the product where nobody reviews it."""
    bootstrap = Bootstrap.for_prompt(
        "a CLI that converts markdown tables to CSV", stack="python"
    )

    assert "markdown tables to CSV" in bootstrap.goal


def test_the_goal_asks_for_a_suite_that_asserts_something():
    """#88 measured `node --test` exiting 0 on a repository with no tests in
    it, so "the gate passed" is not evidence a suite exists. Asking is weaker
    than #102's falsification and costs nothing."""
    assert "at least one real assertion" in Bootstrap.for_prompt("x", stack="node").goal


# --------------------------------------------------------------------------
# It is a task like any other
# --------------------------------------------------------------------------


def test_the_bootstrap_round_trips_through_the_contract_parser():
    """The property that keeps every other module ignorant of it."""
    bootstrap = Bootstrap.for_prompt("a markdown table tool", stack="node")

    drafts, rejected = normalise([bootstrap.task], verify=VERIFY)

    assert rejected == ()
    contract = parse_contract(7, drafts[0].body())
    assert contract.task_id == BOOTSTRAP_TASK_ID
    assert contract.files == bootstrap.files
    assert contract.stack == "node"


def test_every_other_task_is_blocked_on_the_bootstrap():
    """Without this the dispatcher runs three workers against an empty
    repository in the first cycle, each generating its own idea of the
    project."""
    bootstrap = Bootstrap.for_prompt("a tool", stack="python").task
    planned = [
        PlannedTask(id="add-csv", goal="g", files=["src/csv.py"]),
        PlannedTask(id="add-cli", goal="g", files=["src/cli.py"], depends_on=["add-csv"]),
    ]

    tasks = with_bootstrap(planned, bootstrap)

    assert tasks[0].id == BOOTSTRAP_TASK_ID
    assert all(BOOTSTRAP_TASK_ID in task.depends_on for task in tasks[1:])
    # The dependency the model asked for survives alongside the one we added.
    assert "add-csv" in tasks[2].depends_on


def test_the_bootstrap_is_ordered_first():
    bootstrap = Bootstrap.for_prompt("a tool", stack="python").task
    tasks = with_bootstrap([PlannedTask(id="later", goal="g", files=["a.py"])], bootstrap)

    drafts, _ = normalise(tasks, verify=VERIFY)

    assert [draft.task_id for draft in order_drafts(drafts)][0] == BOOTSTRAP_TASK_ID


def test_a_replan_that_re_emits_the_bootstrap_does_not_block_it_on_itself():
    """`order_drafts` refuses a self-edge, so this would fail the whole replan
    rather than the one task. The id is fixed precisely so a replan recognises
    the bootstrap it already created."""
    bootstrap = Bootstrap.for_prompt("a tool", stack="python").task

    tasks = with_bootstrap([bootstrap], bootstrap)

    assert len(tasks) == 1
    assert tasks[0].depends_on == []
    order_drafts(normalise(tasks, verify=VERIFY)[0])  # does not raise


def test_the_bootstrap_carries_the_stack_it_resolved():
    """#99 spawns the image from this, and #96 sets up the CI toolchain from
    it. A bootstrap that resolved react and declared python would generate a
    React app in a Python container."""
    assert Bootstrap.for_prompt("a dashboard", stack="react").task.stack == "react"


def test_a_task_the_model_planned_keeps_its_own_stack_out_of_it():
    """#87's non-goals: one repo, one stack. The plan's stack is the
    bootstrap's, and `write_plan` applies it to every task."""
    bootstrap = Bootstrap.for_prompt("a dashboard", stack="react")
    planned = [PlannedTask(id="later", goal="g", files=["src/x.jsx"], stack="python")]

    drafts, _ = normalise(
        with_bootstrap(planned, bootstrap.task), verify=VERIFY, stack=bootstrap.stack
    )

    assert {draft.stack for draft in drafts} == {"react"}


def test_no_test_here_reaches_a_model():
    """Stated as a test because it is the file's premise. Every oracle above is
    a double, and `choose_stack`'s default argument is the only path to a real
    one."""
    import inspect

    source = inspect.getsource(Bootstrap.for_prompt)

    assert "llm=llm" in source
