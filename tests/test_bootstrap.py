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
from pathlib import Path
from typing import Any, Sequence

import pytest

from swarm.github.ledger import DEFAULT_STACK, KNOWN_STACKS, parse_contract
from swarm.config import SETTINGS
from swarm.greenfield.provision import PLACEHOLDER_VERIFY
from swarm.greenfield.bootstrap import (
    BOOTSTRAP_FILES,
    FALSIFY_TIMEOUT_S,
    ProposedGate,
    Verdict,
    choose_gate,
    empty_declared,
    falsify,
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


# --------------------------------------------------------------------------
# Falsifying the gate (#102)
# --------------------------------------------------------------------------
#
# `planner.py:560-566` refuses to let the model choose a verify command,
# because "a command guessed wrong is a gate that was red before any worker
# touched the task". A model-generated bootstrap reopens that: the model would
# write both the code and the gate that judges it.
#
# `run` is scripted as an ordered list of exit codes, exactly as
# `fixtures/github.py`'s transport scripts responses. The whole decision table
# needs no container and no subprocess.


@dataclass
class Runs:
    """A scripted `run`. Each call takes the next exit code and records where."""

    codes: list[int]
    seen: list[tuple[str, str]] = field(default_factory=list)

    def __call__(self, command: str, tree) -> int:
        self.seen.append((command, Path(tree).name))
        return self.codes[len(self.seen) - 1]


def a_tree(root: Path, files: Sequence[str] = ("src/main.py", "tests/test_main.py")) -> Path:
    tree = root / "clean"
    for relative in files:
        target = tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("real content\n")
    # Not declared, and therefore never mutated - the reason `test -f README.md`
    # stays refused.
    (tree / "README.md").write_text("# project\n")
    return tree


def test_a_gate_that_passes_clean_and_fails_broken_is_accepted(tmp_path):
    runs = Runs([0, 1])

    verdict = falsify("pytest -q", a_tree(tmp_path), ["src/main.py"], run=runs, workspace=tmp_path)

    assert verdict.accepted
    # Clean first, then the mutant - and the mutant is a different tree, not the
    # same one edited underneath the run that just passed.
    assert [where for _, where in runs.seen] == ["clean", "mutated"]


def test_a_gate_that_is_red_on_its_own_code_is_refused(tmp_path):
    """The failure `planner.py` already refuses to risk: a gate that would
    block every pull request before a worker touched the task."""
    runs = Runs([1, 1])

    verdict = falsify("pytest -q", a_tree(tmp_path), ["src/main.py"], run=runs, workspace=tmp_path)

    assert not verdict.accepted
    assert "red on the code it was written for" in verdict.reason
    # And it never bothered running the mutation.
    assert len(runs.seen) == 1


def test_a_gate_that_survives_the_mutation_is_refused(tmp_path):
    runs = Runs([0, 0])

    verdict = falsify("pytest -q", a_tree(tmp_path), ["src/main.py"], run=runs, workspace=tmp_path)

    assert not verdict.accepted
    assert "not reading the code" in verdict.reason


@pytest.mark.parametrize("command", ["true", "exit 0", ":", "  true  ", "TRUE"])
def test_a_tautology_is_refused_without_creating_a_container(tmp_path, command):
    runs = Runs([])

    verdict = falsify(command, a_tree(tmp_path), ["src/main.py"], run=runs, workspace=tmp_path)

    assert not verdict.accepted
    assert "tautological" in verdict.reason
    assert runs.seen == []


def test_the_placeholder_verify_stays_refused(tmp_path):
    """The permanent regression case. `test -f README.md` is refused because
    the mutation **empties** the declared files rather than deleting the tree:
    README is not declared, so it survives, so the command passes on the mutant
    and is caught. A falsifier that deleted files would admit it."""
    tree = a_tree(tmp_path)

    def really_run(command: str, where) -> int:
        # The real semantics of `test -f README.md`, without a shell.
        return 0 if (Path(where) / "README.md").is_file() else 1

    verdict = falsify(
        PLACEHOLDER_VERIFY, tree, ["src/main.py"], run=really_run, workspace=tmp_path
    )

    assert not verdict.accepted
    assert "not reading the code" in verdict.reason


def test_the_mutation_empties_the_declared_files_and_nothing_else(tmp_path):
    tree = a_tree(tmp_path)

    mutated = empty_declared(tree, ["src/main.py"], tmp_path / "mutated")

    assert (mutated / "src/main.py").read_text() == ""
    # Still present, which is the point - not deleted.
    assert (mutated / "src/main.py").is_file()
    # Undeclared files are untouched, and so is the original tree.
    assert (mutated / "README.md").read_text() == "# project\n"
    assert (mutated / "tests/test_main.py").read_text() == "real content\n"
    assert (tree / "src/main.py").read_text() == "real content\n"


def test_a_declared_file_that_does_not_exist_is_not_an_error(tmp_path):
    """A bootstrap whose generation was partial - #88 measured 9% of them -
    must still be falsifiable rather than crashing the provisioning run."""
    mutated = empty_declared(a_tree(tmp_path), ["src/never-written.py"], tmp_path / "m")

    assert mutated.is_dir()


def test_the_mutation_is_ours_and_deterministic(tmp_path):
    """Never model-chosen: a mutation the model picked would put the same model
    on both sides of the test, which is the failure falsification exists to
    prevent. Deterministic, so two runs of one plan agree."""
    first = empty_declared(a_tree(tmp_path), ["src/main.py"], tmp_path / "a")
    second = empty_declared(a_tree(tmp_path), ["src/main.py"], tmp_path / "b")

    assert sorted(p.name for p in first.rglob("*")) == sorted(p.name for p in second.rglob("*"))


def test_a_refusal_is_a_verdict_and_never_an_exception(tmp_path):
    """A planning refusal with a printed reason. Raising would abort a
    provisioning run that has already created a repository."""
    verdict = falsify("x", a_tree(tmp_path), ["src/main.py"], run=Runs([1]), workspace=tmp_path)

    assert isinstance(verdict, Verdict)
    assert "REFUSED" in str(verdict)
    assert verdict.command in str(verdict)


def test_the_falsification_clock_is_not_the_verify_clock():
    """They bound different things. Sharing the number would give a repository
    with a legitimately slow suite five minutes per probe, twice, at provision
    time."""
    assert FALSIFY_TIMEOUT_S != SETTINGS.verify_timeout_s


def test_nothing_under_greenfield_runs_a_shell():
    """The highest-severity item in the epic, as an assertion.

    Provisioning runs in the orchestrator process, which holds the boot key
    (`administration` and `workflows`) and a `DOCKER_HOST` with
    `ALLOW_START=1`. `assert_unprivileged` only inspects argv that
    `ContainerManager.spawn` built, so a shell one-liner calling `docker`
    directly bypasses it. One line of model-proposed shell there is host root
    plus the ability to rewrite the CI it is being falsified against.

    Asserted over the **parse tree**, not the source text: the text contains
    the word `subprocess` in the docstring explaining why there is none, which
    is exactly how a grep-based version of this test fails.
    """
    import ast

    import swarm.greenfield as greenfield

    for path in Path(greenfield.__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(a.name.split(".")[0] in {"subprocess", "os"} and a.name == "subprocess" for a in node.names), path.name
            if isinstance(node, ast.ImportFrom):
                assert node.module != "subprocess", path.name
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "shell":
                        assert not (
                            isinstance(keyword.value, ast.Constant) and keyword.value.value
                        ), path.name
                func = node.func
                name = getattr(func, "attr", None) or getattr(func, "id", None)
                assert name not in {"system", "popen", "execv", "fork"}, f"{path.name}: {name}"


def test_the_proposed_gate_asks_for_one_that_can_fail():
    """The schema description is the prompt. #88 measured `node --test` exiting
    0 on a repository with no tests in it, so "write a test command" is not
    enough of an instruction."""
    described = ProposedGate.model_json_schema()["properties"]["command"]["description"]

    assert "MUST fail" in described
    assert "empty" in described


@pytest.mark.docker
def test_falsification_runs_the_real_command_in_a_real_container(tmp_path):
    """The one live test. Everything above scripts `run`; this proves the
    default actually executes, in a container, with no network, and returns an
    exit code the decision table can read.

    Skipped rather than failed when the image is not built - that is a manual
    host step by design (`BUILD=0`), so its absence is a fact about the machine.
    """
    from swarm.containers.manager import ContainerError, StackImages, build_hint, missing_image

    tree = tmp_path / "clean"
    (tree / "tests").mkdir(parents=True)
    (tree / "main.py").write_text("def add(a, b):\n    return a + b\n")
    (tree / "tests" / "__init__.py").write_text("")
    (tree / "tests" / "test_main.py").write_text(
        "import unittest\n"
        "from main import add\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(1, 2), 3)\n"
    )

    try:
        verdict = falsify(
            "python3 -m unittest discover -q",
            tree,
            ["main.py", "tests/test_main.py"],
            workspace=tmp_path,
        )
    except ContainerError as exc:
        image = StackImages().for_stack("python")
        if not missing_image(exc):
            raise
        pytest.skip(f"{image} is not built on this host: {build_hint(image)}")

    # Green on the code it was written for, red once that code is emptied.
    assert verdict.accepted, verdict.reason


def test_an_operators_verify_command_is_never_falsified(tmp_path):
    """The escape hatch, and an escape hatch that can be refused is not one: a
    false rejection has no recovery path, because the operator has already told
    the system the answer."""
    runs = Runs([])

    verdict = choose_gate(
        "pytest -q", a_tree(tmp_path), ["src/main.py"], operator="make check", run=runs
    )

    assert verdict.accepted
    assert verdict.command == "make check"
    assert runs.seen == []


def test_a_refused_gate_keeps_the_placeholder_rather_than_failing_the_run(tmp_path):
    """Aborting would destroy a generated project over a gate, and the project
    is the expensive part."""
    verdict = choose_gate(
        "true",
        a_tree(tmp_path),
        ["src/main.py"],
        fallback=PLACEHOLDER_VERIFY,
        run=Runs([]),
        workspace=tmp_path,
    )

    assert not verdict.accepted
    assert verdict.command == PLACEHOLDER_VERIFY
    assert "tautological" in verdict.reason


def test_an_accepted_gate_replaces_the_placeholder(tmp_path):
    verdict = choose_gate(
        "pytest -q",
        a_tree(tmp_path),
        ["src/main.py"],
        fallback=PLACEHOLDER_VERIFY,
        run=Runs([0, 1]),
        workspace=tmp_path,
    )

    assert verdict.accepted
    assert verdict.command == "pytest -q"
