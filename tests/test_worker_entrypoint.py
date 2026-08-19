"""The worker loop: clone, edit, verify, commit - and everything it refuses.

Three claims from #16, and each has a test that would fail loudly if the claim
stopped being true:

- a scratch-repo issue produces a commit that passes verification;
- a deliberately impossible issue exits non-zero **without** committing;
- an edit aimed outside the declared file set is refused.

Everything here is hermetic. The GitHub side is `fixtures/github.py`'s scripted
transport, the git side is `fixtures/repo.py`'s scratch repository with a bare
`origin` on disk, and the model is a double - the real one runs in exactly one
test, which carries the `ollama` marker and is deselected by default.

The scratch repo's verify command is real, though: it runs a real pytest in a
real checkout, because the one thing this suite must not fake is the gate.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pytest
from fixtures.github import not_found, response
from fixtures.repo import VERIFY_COMMAND, ScratchRepo

from swarm.github.branches import task_branch
from swarm.github.ledger import ContractError
from swarm.github.refs import task_ref
from swarm.state import FileEdit, WorkerOutput
from swarm.worker import edit as edit_module
from swarm.worker import entrypoint
from swarm.worker.entrypoint import (
    EXIT_INFRASTRUCTURE,
    EXIT_OK,
    EXIT_TASK_FAILED,
    INFRASTRUCTURE,
    PASSED,
    TASK_FAILED,
    UNRUNNABLE_EXIT_CODES,
    VERIFY_ENV_REQUIRED,
    OUTPUT_TAIL_CHARS,
    InfrastructureError,
    WorkerResult,
    commit_edits,
    stageable,
    classify_verify,
    main,
    run_verify,
    run_worker,
    split_repo,
    verify_env,
)

ISSUE = 7

#: A verify command that always fails, whatever the repository contains. The
#: "deliberately impossible issue" of the ticket, in one line.
ALWAYS_FAILS = f'{sys.executable} -c "raise SystemExit(3)"'

#: `calc.py` with a second function, and the seeded test still passing.
GOOD_CALC = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"

#: `calc.py` that breaks the seeded test - a plausible bad generation.
BAD_CALC = "def add(a, b):\n    return a * b\n"


# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


@dataclass
class FakeEditor:
    """The worker model, minus the model.

    `propose_edits` only ever calls `.invoke(messages)`, so this is the whole
    seam. It keeps the prompts because what the model is *shown* is half of
    what #16 decides - the readable context set is only real if it reaches the
    prompt.
    """

    output: Any
    prompts: list[str] = field(default_factory=list)

    def invoke(self, messages: Sequence[tuple[str, str]]) -> WorkerOutput:
        self.prompts.append(messages[-1][1])
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def edits(paths: dict[str, str]) -> WorkerOutput:
    """A `WorkerOutput` from `{path: contents}` - what the model would return."""
    return WorkerOutput(edits=[FileEdit(path=path, content=text) for path, text in paths.items()])


def issue_body(
    *,
    goal: str = "`calc.sub` subtracts its second argument from its first.",
    files: Sequence[str] = ("calc.py",),
    verify: str = VERIFY_COMMAND,
    task_id: str | None = "add-sub",
    attempt: int = 0,
    marker: str | None = None,
) -> str:
    if marker is None:
        marker = f"<!-- apiary:task id={task_id} attempt={attempt} -->\n\n" if task_id else ""
    listed = "\n".join(f"- {path}" for path in files)
    return (
        f"{marker}## Goal\n{goal}\n\n"
        f"## Files\n{listed}\n\n"
        f"## Verify\n{verify}\n\n"
        "## Blocked by\n_none._\n"
    )


def issue(number: int = ISSUE, **kwargs: Any):
    """The first API response a worker run needs: `GET /issues/<n>`."""
    return response(
        200,
        {"number": number, "title": "Add a sub function", "body": issue_body(**kwargs)},
    )


def publishes(number: int = 42):
    """The three further responses a *verified* run needs, now that #17 landed.

    A run that passes its gate no longer stops at the commit: `_publish` opens
    the PR and applies `swarm:review`. Scripting them is what keeps these tests
    about the entrypoint rather than about how far the seam had got.
    """
    return (
        # The lookup for an existing PR comes first: `GitHubClient` grew
        # `list_pull_requests`, so `find_open_pull_request` now asks rather
        # than degrading to None.
        response(200, []),
        response(201, {"number": number, "html_url": f"https://example.invalid/pull/{number}"}),
        response(200, [{"name": "swarm:review"}]),
    )


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def worker_env(scratch_repo: ScratchRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the entrypoint's own git subprocesses the fixture's isolation.

    `ScratchRepo` keeps its identity and config in the environment rather than
    in `~/.gitconfig`; the worker shells out to git itself, so without this the
    commit would depend on the developer's `commit.gpgsign` and the test would
    pass on one laptop and hang on another.
    """
    for key, value in scratch_repo.env.items():
        monkeypatch.setenv(key, value)


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "workspace"


def argv(scratch_repo: ScratchRepo, workspace: Path, *extra: str) -> list[str]:
    return [
        "--repo",
        str(scratch_repo.remote),
        "--issue",
        str(ISSUE),
        "--base-commit",
        scratch_repo.head(),
        "--workspace",
        str(workspace),
        *extra,
    ]


def checkout(workspace: Path, scratch_repo: ScratchRepo) -> ScratchRepo:
    """The clone the worker made, wrapped so the git helpers are available."""
    return ScratchRepo(
        workspace / f"issue-{ISSUE}", scratch_repo.remote, env=dict(scratch_repo.env)
    )


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


@pytest.mark.usefixtures("worker_env")
def test_a_retry_branches_under_its_own_attempt_number(
    fake_github, scratch_repo, workspace
):
    """#144: the branch carries the ref *and* the attempt, read from the marker.

    Which means a second attempt does not recreate the first attempt's branch
    from the base commit - it gets one of its own. That is what lets an
    orchestrator with no memory left list the remote and see how much budget a
    task has already spent, and it is why `recovery.py` can derive `review`
    from a name instead of a label."""
    gh, _, _ = fake_github(
        issue(attempt=2, verify=ALWAYS_FAILS), response(200, feedback_comments())
    )
    editor = FakeEditor(edits({"calc.py": GOOD_CALC}))

    result = run_worker(
        repo=gh.repo,
        issue=ISSUE,
        base_commit=scratch_repo.head(),
        workspace=workspace,
        clone_url=str(scratch_repo.remote),
        client=gh,
        editor=editor,
    )

    assert result.branch == task_branch(task_ref(ISSUE), 2)
    assert checkout(workspace, scratch_repo).current_branch() == result.branch


@pytest.mark.usefixtures("worker_env")
def test_verified_task_produces_a_commit(fake_github, scratch_repo, workspace):
    gh, transport, _ = fake_github(issue(), *publishes())
    base = scratch_repo.head()
    editor = FakeEditor(edits({"calc.py": GOOD_CALC}))

    assert main(argv(scratch_repo, workspace), client=gh, editor=editor) == EXIT_OK

    work = checkout(workspace, scratch_repo)
    assert work.current_branch() == task_branch(task_ref(ISSUE), 0)
    assert work.head() != base
    assert work.subjects()[0].startswith("swarm[add-sub]:")
    assert work.read("calc.py") == GOOD_CALC
    # Read the contract, open the PR, apply the review label - in that order,
    # and nothing else. The label goes on last because a `swarm:review` issue
    # with no PR behind it is a state the reconciler cannot act on.
    assert transport.calls == [
        ("GET", f"/repos/{gh.repo}/issues/{ISSUE}"),
        ("GET", f"/repos/{gh.repo}/pulls"),
        ("POST", f"/repos/{gh.repo}/pulls"),
        ("POST", f"/repos/{gh.repo}/issues/{ISSUE}/labels"),
    ]


@pytest.mark.usefixtures("worker_env")
def test_commit_stages_only_the_declared_files(fake_github, scratch_repo, workspace):
    """The verify command is arbitrary shell and litters; the commit must not.

    `git add -A` after a test run sweeps caches, build output and anything else
    the command dropped into the tree straight into the PR.
    """
    litter = f'{VERIFY_COMMAND} && {sys.executable} -c "open(\'stray.txt\', \'w\')"'
    gh, _, _ = fake_github(issue(verify=litter), *publishes())
    editor = FakeEditor(edits({"calc.py": GOOD_CALC}))

    assert main(argv(scratch_repo, workspace), client=gh, editor=editor) == EXIT_OK

    work = checkout(workspace, scratch_repo)
    assert (work.path / "stray.txt").exists()
    assert work.out("show", "--name-only", "--format=", "HEAD").split() == ["calc.py"]
    assert "stray.txt" in work.out("status", "--porcelain")


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


@pytest.mark.usefixtures("worker_env")
def test_impossible_issue_fails_without_committing(fake_github, scratch_repo, workspace):
    gh, _, _ = fake_github(issue(verify=ALWAYS_FAILS))
    base = scratch_repo.head()
    editor = FakeEditor(edits({"calc.py": GOOD_CALC}))

    code = main(argv(scratch_repo, workspace, "--keep"), client=gh, editor=editor)

    assert code == EXIT_TASK_FAILED
    work = checkout(workspace, scratch_repo)
    assert work.head() == base
    # The edit is still on disk, unstaged: the container is thrown away, and
    # what matters is that nothing unverified became a commit.
    assert work.read("calc.py") == GOOD_CALC
    assert "calc.py" in work.out("status", "--porcelain")


@pytest.mark.usefixtures("worker_env")
def test_a_failed_verify_is_the_models_fault_not_the_commands(
    fake_github, scratch_repo, workspace
):
    """A bad generation fails the repository's own suite. No model is consulted."""
    gh, _, _ = fake_github(issue())
    base = scratch_repo.head()
    editor = FakeEditor(edits({"calc.py": BAD_CALC}))

    code = main(argv(scratch_repo, workspace, "--keep"), client=gh, editor=editor)

    assert code == EXIT_TASK_FAILED
    assert checkout(workspace, scratch_repo).head() == base


@pytest.mark.usefixtures("worker_env")
def test_verify_command_comes_from_the_issue(fake_github, scratch_repo, workspace):
    """Not `Settings.verify_command`, which would pass this task by accident.

    The repo-wide default (`python -m pytest -q`) is green in this checkout, so
    a worker that substituted it would commit. The contract's command leaves a
    marker and then fails, which distinguishes the two.
    """
    gate = f'{sys.executable} -c "open(\'gate.txt\', \'w\'); raise SystemExit(3)"'
    gh, _, _ = fake_github(issue(verify=gate))
    editor = FakeEditor(edits({"calc.py": GOOD_CALC}))

    result = run_worker(
        repo=gh.repo,
        issue=ISSUE,
        base_commit=scratch_repo.head(),
        workspace=workspace,
        clone_url=str(scratch_repo.remote),
        client=gh,
        editor=editor,
    )

    assert result.verify_command == gate
    assert (result.root / "gate.txt").exists()
    assert not result.passed and result.commit is None


@pytest.mark.usefixtures("worker_env")
def test_a_failed_run_leaves_no_checkout_behind(fake_github, scratch_repo, workspace):
    gh, _, _ = fake_github(issue(verify=ALWAYS_FAILS))
    editor = FakeEditor(edits({"calc.py": GOOD_CALC}))

    assert main(argv(scratch_repo, workspace), client=gh, editor=editor) == EXIT_TASK_FAILED
    assert not (workspace / f"issue-{ISSUE}").exists()


@pytest.mark.usefixtures("worker_env")
def test_no_usable_edit_skips_verification_entirely(fake_github, scratch_repo, workspace):
    """Nothing was written, so the command would report the seed tree's state."""
    marker = f'{sys.executable} -c "open(\'ran.txt\', \'w\')"'
    gh, _, _ = fake_github(issue(verify=marker))
    editor = FakeEditor(WorkerOutput(edits=[]))

    result = run_worker(
        repo=gh.repo,
        issue=ISSUE,
        base_commit=scratch_repo.head(),
        workspace=workspace,
        clone_url=str(scratch_repo.remote),
        client=gh,
        editor=editor,
    )

    assert not result.passed and result.commit is None
    assert not (result.root / "ran.txt").exists()


# --------------------------------------------------------------------------
# The guard rail
# --------------------------------------------------------------------------


@pytest.mark.usefixtures("worker_env")
def test_edit_outside_the_declared_set_is_refused(fake_github, scratch_repo, workspace):
    """README.md is in the *readable* set, which grants no right to write it."""
    gh, _, _ = fake_github(issue())
    editor = FakeEditor(edits({"calc.py": GOOD_CALC, "README.md": "# owned\n"}))

    result = run_worker(
        repo=gh.repo,
        issue=ISSUE,
        base_commit=scratch_repo.head(),
        workspace=workspace,
        clone_url=str(scratch_repo.remote),
        client=gh,
        editor=editor,
    )

    assert result.written == ("calc.py",)
    assert result.refused == (("README.md", "not in the declared file set"),)
    assert (result.root / "README.md").read_text().startswith("# scratch")
    assert result.passed and result.commit


def test_path_traversal_is_refused(tmp_path):
    """Even when the declared set itself is hostile.

    `parse_contract` rejects `..` in `## Files`, so this can only happen if a
    body bypassed the parser - but the guard is the last line before a write,
    and it has to hold on input nobody sanitised.
    """
    root = tmp_path / "checkout"
    (root / "src").mkdir(parents=True)

    applied = edit_module.apply_edits(
        root,
        [FileEdit(path="../escape.py", content="owned")],
        ["../escape.py"],
    )

    assert applied.written == ()
    assert applied.refused == (("../escape.py", "resolves outside the checkout"),)
    assert not (tmp_path / "escape.py").exists()


def test_a_symlinked_path_cannot_leave_the_checkout(tmp_path):
    """The reason the check resolves instead of comparing strings."""
    root = tmp_path / "checkout"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("original\n")
    (root / "link.py").symlink_to(outside)

    applied = edit_module.apply_edits(
        root, [FileEdit(path="link.py", content="owned")], ["link.py"]
    )

    assert applied.refused == (("link.py", "resolves outside the checkout"),)
    assert outside.read_text() == "original\n"


def test_declared_paths_are_matched_exactly(tmp_path):
    """`./calc.py` is `calc.py`; `Calc.py` is a different file the task never got."""
    root = tmp_path / "checkout"
    root.mkdir()

    applied = edit_module.apply_edits(
        root,
        [FileEdit(path="./calc.py", content="ok\n"), FileEdit(path="Calc.py", content="no\n")],
        ["calc.py"],
    )

    assert applied.written == ("calc.py",)
    assert applied.refused == (("Calc.py", "not in the declared file set"),)


# --------------------------------------------------------------------------
# Deletion: an empty-content edit removes the file
# --------------------------------------------------------------------------
#
# The vocabulary used to be whole-file writes only, and it made any goal that
# includes removing a file structurally unwinnable: observed live, a worker
# told to delete an obsolete persistence stack *emptied* the test files
# instead, and the collection audit (correctly) failed the attempt three times
# out of three - an existing `test_*.py` that collects zero tests proves
# nothing. Empty content now means delete, held to the same guard rails as a
# write.


def test_an_empty_content_edit_deletes_a_declared_file(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "obsolete.py").write_text("legacy = True\n")

    applied = edit_module.apply_edits(
        root, [FileEdit(path="obsolete.py", content="")], ["obsolete.py"]
    )

    assert applied.deleted == ("obsolete.py",)
    assert applied.written == ()
    assert applied.refused == ()
    assert not (root / "obsolete.py").exists()


def test_whitespace_only_content_counts_as_a_deletion(tmp_path):
    """A file of two spaces and a newline was never what anyone intended, and
    treating it as a write would let a model's stray whitespace produce exactly
    the emptied-not-removed file the delete vocabulary exists to end."""
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "obsolete.py").write_text("legacy = True\n")

    applied = edit_module.apply_edits(
        root, [FileEdit(path="obsolete.py", content="  \n\t\n")], ["obsolete.py"]
    )

    assert applied.deleted == ("obsolete.py",)
    assert not (root / "obsolete.py").exists()


def test_deleting_a_file_that_does_not_exist_is_a_refusal(tmp_path):
    """Deleting nothing is model confusion about the tree it was shown, and it
    is surfaced like every other refusal rather than swallowed as a no-op."""
    root = tmp_path / "checkout"
    root.mkdir()

    applied = edit_module.apply_edits(
        root, [FileEdit(path="ghost.py", content="")], ["ghost.py"]
    )

    assert applied.deleted == ()
    assert applied.written == ()
    assert applied.refused == (("ghost.py", "deletes a file that does not exist"),)


def test_a_deletion_outside_the_declared_set_is_still_refused(tmp_path):
    """The guard rail guards deletions exactly as it guards writes: readable
    context grants no right to remove a file either."""
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "README.md").write_text("# keep me\n")

    applied = edit_module.apply_edits(
        root, [FileEdit(path="README.md", content="")], ["calc.py"]
    )

    assert applied.refused == (("README.md", "not in the declared file set"),)
    assert (root / "README.md").read_text() == "# keep me\n"


def test_deletion_prunes_emptied_directories_but_never_the_root(tmp_path):
    """git cannot commit an empty directory, so one left behind exists only in
    the working tree - and the checkout root must survive even a deletion of
    the repository's last file."""
    root = tmp_path / "checkout"
    (root / "pkg" / "sub").mkdir(parents=True)
    (root / "pkg" / "sub" / "only.py").write_text("x = 1\n")
    (root / "only.py").write_text("y = 2\n")

    edit_module.apply_edits(
        root,
        [FileEdit(path="pkg/sub/only.py", content=""), FileEdit(path="only.py", content="")],
        ["pkg/sub/only.py", "only.py"],
    )

    assert not (root / "pkg").exists()
    assert root.is_dir()


def test_deletion_stops_pruning_at_the_first_occupied_directory(tmp_path):
    root = tmp_path / "checkout"
    (root / "pkg" / "sub").mkdir(parents=True)
    (root / "pkg" / "sub" / "only.py").write_text("x = 1\n")
    (root / "pkg" / "keep.py").write_text("y = 2\n")

    edit_module.apply_edits(
        root, [FileEdit(path="pkg/sub/only.py", content="")], ["pkg/sub/only.py"]
    )

    assert not (root / "pkg" / "sub").exists()
    assert (root / "pkg" / "keep.py").exists()


def test_the_ledger_reports_net_effect_per_path(tmp_path):
    """Written-then-deleted must not reach `commit_edits` in both lists: a path
    the batch itself created and then removed leaves nothing for `git add` to
    stage, and reporting it as either a write or a deletion would be a lie."""
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "old.py").write_text("x = 1\n")

    applied = edit_module.apply_edits(
        root,
        [
            # Pre-existing, written and then deleted: a real deletion.
            FileEdit(path="old.py", content="x = 2\n"),
            FileEdit(path="old.py", content=""),
            # Created by this batch and then deleted: net nothing.
            FileEdit(path="new.py", content="y = 1\n"),
            FileEdit(path="new.py", content=""),
        ],
        ["old.py", "new.py"],
    )

    assert applied.written == ()
    assert applied.deleted == ("old.py",)
    assert not (root / "old.py").exists() and not (root / "new.py").exists()


def test_the_worker_prompt_teaches_deletion():
    """The vocabulary only exists if the model is told it does: one sentence in
    `SYSTEM` says empty content deletes, and that removing an obsolete file is
    often the right cleanup edit."""
    assert "empty content" in edit_module.SYSTEM
    assert "DELETE" in edit_module.SYSTEM


@pytest.mark.usefixtures("worker_env")
def test_a_deletion_reaches_the_commit(fake_github, scratch_repo, workspace):
    """The headline: the commit records the file's removal, not an emptying."""
    scratch_repo.write("obsolete.py", "legacy = True\n")
    scratch_repo.commit("seed the obsolete module")
    scratch_repo.push()
    gh, _, _ = fake_github(issue(files=("calc.py", "obsolete.py")), *publishes())
    editor = FakeEditor(edits({"calc.py": GOOD_CALC, "obsolete.py": ""}))

    assert main(argv(scratch_repo, workspace), client=gh, editor=editor) == EXIT_OK

    work = checkout(workspace, scratch_repo)
    assert not (work.path / "obsolete.py").exists()
    assert sorted(work.out("show", "--name-only", "--format=", "HEAD").split()) == [
        "calc.py",
        "obsolete.py",
    ]
    assert "obsolete.py" not in work.out("ls-tree", "-r", "--name-only", "HEAD").split()


@pytest.mark.usefixtures("worker_env")
def test_a_deleted_test_file_is_exempt_from_the_parse_gate_and_the_audit(
    fake_github, scratch_repo, workspace
):
    """The live failure, made winnable end to end.

    An obsolete test file that does not even parse, in a repository whose
    `testpaths` would never collect it: the only correct edit is to remove it.
    Before deletion existed the worker emptied it instead, the collection audit
    (correctly) refused an existing `test_*.py` that collects zero tests, and
    the task burned all three attempts. A deleted file has nothing left to
    parse and nothing left to collect, so both gates must pass it by."""
    scratch_repo.write(
        "pyproject.toml", '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
    )
    scratch_repo.write("tests/test_main.py", "def test_ok():\n    assert True\n")
    scratch_repo.write("test_obsolete.py", "def broken(:\n")
    scratch_repo.commit("seed the debris")
    scratch_repo.push()
    gh, _, _ = fake_github(issue(files=("test_obsolete.py",)))
    editor = FakeEditor(edits({"test_obsolete.py": ""}))

    result = run_worker(
        repo=gh.repo,
        issue=ISSUE,
        base_commit=scratch_repo.head(),
        workspace=workspace,
        clone_url=str(scratch_repo.remote),
        client=gh,
        editor=editor,
    )

    assert result.passed and result.commit
    assert result.written == ()
    assert result.deleted == ("test_obsolete.py",)
    assert "test_obsolete.py (deleted)" in result.summary()
    assert not (result.root / "test_obsolete.py").exists()


# --------------------------------------------------------------------------
# Dependencies: the manifest a worker may always write, and the install
# --------------------------------------------------------------------------


@pytest.mark.usefixtures("worker_env")
def test_the_dependency_manifest_is_writable_without_being_declared(
    fake_github, scratch_repo, workspace, monkeypatch
):
    """`requirements.txt` joins the allowed set whether or not `## Files` lists
    it - declaring what its code needs is part of any task - while every other
    undeclared path keeps being refused."""
    installed: list[Path] = []
    monkeypatch.setattr(
        entrypoint, "install_dependencies", lambda root: installed.append(root) or None
    )
    gh, _, _ = fake_github(issue())
    editor = FakeEditor(
        edits(
            {
                "calc.py": GOOD_CALC,
                "requirements.txt": "# nothing beyond the standard library\n",
                "README.md": "# owned\n",
            }
        )
    )

    result = run_worker(
        repo=gh.repo,
        issue=ISSUE,
        base_commit=scratch_repo.head(),
        workspace=workspace,
        clone_url=str(scratch_repo.remote),
        client=gh,
        editor=editor,
    )

    assert "requirements.txt" in result.written
    assert result.refused == (("README.md", "not in the declared file set"),)
    assert result.passed and result.commit
    # And the install ran, once, against this checkout, before the gate.
    assert installed == [result.root]


@pytest.mark.usefixtures("worker_env")
def test_a_failed_install_is_a_failed_verify_not_infrastructure(
    fake_github, scratch_repo, workspace, monkeypatch
):
    """The pip error is the task's real blocker and the next retry's feedback,
    so it lands as a FAILED gate - exit 1, attempt consumed - and the verify
    command itself never runs against an incomplete environment."""
    failure = "dependency install failed (exit 1): pip install -r requirements.txt\nboom"
    monkeypatch.setattr(entrypoint, "install_dependencies", lambda root: failure)
    gate = f'{sys.executable} -c "open(\'gate.txt\', \'w\')"'
    gh, _, _ = fake_github(issue(verify=gate))
    editor = FakeEditor(edits({"calc.py": GOOD_CALC}))

    result = run_worker(
        repo=gh.repo,
        issue=ISSUE,
        base_commit=scratch_repo.head(),
        workspace=workspace,
        clone_url=str(scratch_repo.remote),
        client=gh,
        editor=editor,
    )

    assert not result.passed and result.commit is None
    assert result.exit_code == EXIT_TASK_FAILED
    assert result.verify_output == failure
    assert not (result.root / "gate.txt").exists()


def _pip(returncode: int, stdout: str = "", stderr: str = "", seen: dict | None = None):
    """A fake `subprocess.run` for the install step. No socket, no pip."""

    def run(command, **kwargs):
        if seen is not None:
            seen.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    return run


def test_no_manifest_installs_nothing(tmp_path, monkeypatch):
    def explode(*args, **kwargs):  # pragma: no cover - the assertion is that it never runs
        raise AssertionError("install ran with no manifest")

    monkeypatch.setattr(entrypoint.subprocess, "run", explode)

    assert entrypoint.install_dependencies(tmp_path) is None


def test_a_clean_install_reports_nothing_and_runs_in_the_checkout(tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("flask\n")
    seen: dict = {}
    monkeypatch.setattr(
        entrypoint.subprocess, "run", _pip(0, stdout="Successfully installed flask", seen=seen)
    )

    assert entrypoint.install_dependencies(tmp_path) is None
    assert seen["command"] == "pip install -r requirements.txt"
    assert seen["cwd"] == str(tmp_path)
    assert seen["timeout"] == entrypoint.INSTALL_TIMEOUT_S
    # The filtered environment, exactly as the verify command gets it: the
    # credentials are gone, and nothing else was rebuilt.
    assert "GITHUB_TOKEN" not in seen["env"]
    assert "APIARY_PUSH_TOKEN" not in seen["env"]


def test_a_failed_install_reports_pips_own_words(tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("no-such-package\n")
    monkeypatch.setattr(
        entrypoint.subprocess,
        "run",
        _pip(1, stderr="ERROR: No matching distribution found for no-such-package"),
    )

    failure = entrypoint.install_dependencies(tmp_path)

    assert failure is not None
    assert failure.startswith("dependency install failed (exit 1)")
    assert "No matching distribution" in failure
    # An ordinary pip failure earns no egress advice - that hint is reserved
    # for the failure it actually fixes.
    assert "APIARY_EGRESS_ALLOW" not in failure


def test_a_denied_index_names_the_operator_fix(tmp_path, monkeypatch):
    """Blocked egress names itself, and the message names the knob: the
    allowlist stays an operator decision (`security.EGRESS_EXTRA_ENV`), so the
    failure text is where the fix has to live."""
    (tmp_path / "requirements.txt").write_text("flask\n")
    monkeypatch.setattr(
        entrypoint.subprocess,
        "run",
        _pip(1, stderr="ERROR: HTTP error 403 while getting flask: 403 Filtered"),
    )

    failure = entrypoint.install_dependencies(tmp_path)

    assert failure is not None
    assert "APIARY_EGRESS_ALLOW=pypi.org,files.pythonhosted.org" in failure


def test_a_hung_install_is_a_failure_not_a_wait(tmp_path, monkeypatch):
    (tmp_path / "requirements.txt").write_text("flask\n")

    def hang(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(entrypoint.subprocess, "run", hang)

    failure = entrypoint.install_dependencies(tmp_path)

    assert failure is not None
    assert "timed out" in failure


# --------------------------------------------------------------------------
# The parse gate: a written .py file must at least parse
# --------------------------------------------------------------------------

#: The literal failure this defense exists for, observed merged in a real
#: generated repository (wallet-tracker-service): a model thought-leak inside
#: a test file, a SyntaxError no gate ever saw because pytest's `testpaths`
#: never collected the file.
THOUGHT_LEAK = (
    "def test_deposit():\n"
    "    amount=3.5 far, typo in my thought process again  # I'll use 3.50\n"
)


def test_a_written_syntax_error_names_file_line_and_text(tmp_path):
    (tmp_path / "calc.py").write_text(THOUGHT_LEAK)

    failure = edit_module.syntax_failure(tmp_path, ("calc.py",))

    assert failure is not None
    # The first line is pinned: `reconcile.diagnose` matches it by shape.
    assert failure.startswith("python syntax error in calc.py, line 2:")
    assert "amount=3.5 far" in failure
    assert "The verify command was not run" in failure


def test_a_fullwidth_unicode_stop_does_not_parse(tmp_path):
    # The second merged SyntaxError from the same repository: a full-width
    # `．` where a `.` belongs, invisible in review and fatal to the parser.
    (tmp_path / "test_money.py").write_text(
        "def test_price():\n    assert 3．50\n", encoding="utf-8"
    )

    failure = edit_module.syntax_failure(tmp_path, ("test_money.py",))

    assert failure is not None
    assert "python syntax error in test_money.py, line 2" in failure


def test_clean_python_and_non_python_files_pass_the_parse_gate(tmp_path):
    (tmp_path / "calc.py").write_text(GOOD_CALC)
    # Broken-looking non-Python content is none of the parser's business.
    (tmp_path / "requirements.txt").write_text("this is not python (\n")
    (tmp_path / "notes.md").write_text("# unbalanced ( everywhere\n")

    checked = ("calc.py", "requirements.txt", "notes.md")
    assert edit_module.syntax_failure(tmp_path, checked) is None


def test_every_broken_file_is_reported_not_just_the_first(tmp_path):
    # The text is the retry's feedback; one error per attempt would spend the
    # budget one line at a time.
    (tmp_path / "a.py").write_text("def broken(:\n")
    (tmp_path / "b.py").write_text("x = (\n")

    failure = edit_module.syntax_failure(tmp_path, ("a.py", "b.py"))

    assert failure is not None
    assert "python syntax error in a.py" in failure
    assert "python syntax error in b.py" in failure


@pytest.mark.usefixtures("worker_env")
def test_a_syntax_error_fails_the_attempt_before_anything_runs(
    fake_github, scratch_repo, workspace, monkeypatch
):
    """A file that does not parse fails the attempt with the SyntaxError as the
    verify output - before the install, before the gate, and with no commit -
    so the retry comment quotes the exact line to fix."""

    def never_installs(root):  # pragma: no cover - the assertion is that it never runs
        raise AssertionError("the install ran on a file that does not parse")

    monkeypatch.setattr(entrypoint, "install_dependencies", never_installs)
    gate = f'{sys.executable} -c "open(\'gate.txt\', \'w\')"'
    gh, _, _ = fake_github(issue(verify=gate))
    editor = FakeEditor(edits({"calc.py": THOUGHT_LEAK}))

    result = run_worker(
        repo=gh.repo,
        issue=ISSUE,
        base_commit=scratch_repo.head(),
        workspace=workspace,
        clone_url=str(scratch_repo.remote),
        client=gh,
        editor=editor,
    )

    assert not result.passed and result.commit is None
    assert result.exit_code == EXIT_TASK_FAILED
    assert result.verify_output.startswith("python syntax error in calc.py")
    assert not (result.root / "gate.txt").exists()


# --------------------------------------------------------------------------
# The collection audit: a passed pytest gate must have seen this attempt's tests
# --------------------------------------------------------------------------

#: What `pytest --collect-only -q` prints for a healthy one-test suite.
COLLECTED_MAIN = "tests/test_main.py::test_health\n\n1 test collected in 0.01s\n"


def _never_collects(*args, **kwargs):  # pragma: no cover - asserts non-execution
    raise AssertionError("the collection probe ran when the audit should not apply")


def test_the_audit_skips_a_gate_that_is_not_pytest(tmp_path, monkeypatch):
    monkeypatch.setattr(entrypoint.subprocess, "run", _never_collects)

    assert entrypoint.audit_collection(tmp_path, "make test", ("test_x.py",)) is None


def test_the_audit_skips_an_attempt_that_wrote_no_test_files(tmp_path, monkeypatch):
    monkeypatch.setattr(entrypoint.subprocess, "run", _never_collects)

    written = ("calc.py", "conftest.py", "requirements.txt")
    assert entrypoint.audit_collection(tmp_path, VERIFY_COMMAND, written) is None


def test_a_collected_file_passes_and_the_probe_names_no_paths(tmp_path, monkeypatch):
    """The probe is argument-less on purpose: `testpaths` only applies when the
    command line names nothing, so probing `--collect-only <file>` would
    collect exactly the file the real gate excludes."""
    seen: dict = {}
    monkeypatch.setattr(entrypoint.subprocess, "run", _pip(0, stdout=COLLECTED_MAIN, seen=seen))

    verdict = entrypoint.audit_collection(tmp_path, VERIFY_COMMAND, ("tests/test_main.py",))

    assert verdict is None
    assert seen["command"] == f"{sys.executable} -m pytest --collect-only -q"
    assert seen["cwd"] == str(tmp_path)
    assert seen["timeout"] == entrypoint.AUDIT_TIMEOUT_S
    # The verify command's own filtered environment, credentials gone.
    assert "GITHUB_TOKEN" not in seen["env"]


def test_an_uncollected_file_fails_with_the_pinned_sentence(tmp_path, monkeypatch):
    monkeypatch.setattr(entrypoint.subprocess, "run", _pip(0, stdout=COLLECTED_MAIN))

    failure = entrypoint.audit_collection(
        tmp_path, VERIFY_COMMAND, ("calc.py", "tests/test_db.py")
    )

    assert failure is not None
    assert "tests/test_db.py was not collected by the verify command" in failure
    assert "testpaths" in failure
    assert "a test that never runs proves nothing" in failure
    # And the collected suite travels along as evidence.
    assert "tests/test_main.py::test_health" in failure


def test_a_probe_that_errors_proves_nothing_and_fails(tmp_path, monkeypatch):
    # Exit 4 is pytest's usage error; whatever the cause, the file cannot be
    # proven to run, which is the same verdict as "did not run".
    monkeypatch.setattr(
        entrypoint.subprocess, "run", _pip(4, stderr="ERROR: unrecognized arguments")
    )

    failure = entrypoint.audit_collection(tmp_path, VERIFY_COMMAND, ("tests/test_db.py",))

    assert failure is not None
    assert "tests/test_db.py was not collected" in failure
    assert "unrecognized arguments" in failure


def test_a_hung_probe_fails_the_audit(tmp_path, monkeypatch):
    def hang(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(entrypoint.subprocess, "run", hang)

    failure = entrypoint.audit_collection(tmp_path, VERIFY_COMMAND, ("tests/test_db.py",))

    assert failure is not None
    assert "timed out" in failure


def test_the_audit_catches_a_testpaths_exclusion_with_real_pytest(tmp_path):
    """The observed failure, end to end and unfaked: `testpaths = ["tests"]`
    means an argument-less pytest never sees a test file outside `tests/`, so
    the gate is green while the file never runs. This is also the proof that
    the probe must be argument-less - `pytest --collect-only test_orphan.py`
    would collect the orphan happily, because explicit paths override
    `testpaths`."""
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("def test_ok():\n    assert True\n")
    (tmp_path / "test_orphan.py").write_text("def test_never_runs():\n    assert True\n")

    collected = entrypoint.audit_collection(tmp_path, VERIFY_COMMAND, ("tests/test_main.py",))
    orphaned = entrypoint.audit_collection(tmp_path, VERIFY_COMMAND, ("test_orphan.py",))

    assert collected is None
    assert orphaned is not None
    assert "test_orphan.py was not collected" in orphaned


@pytest.mark.usefixtures("worker_env")
def test_a_green_gate_that_never_collected_the_tests_is_a_failed_attempt(
    fake_github, scratch_repo, workspace, monkeypatch
):
    """The audit's verdict lands as a failed verify - no commit, exit 1, and
    the failure text as the output the retry comment quotes. It is handed only
    what this attempt wrote, never the repository's historical test files."""
    audited: list[tuple] = []
    failure = "the verify command passed, but tests/test_db.py was not collected"

    def fake_audit(root, command, written):
        audited.append((root, command, written))
        return failure

    monkeypatch.setattr(entrypoint, "audit_collection", fake_audit)
    gh, _, _ = fake_github(issue())
    editor = FakeEditor(edits({"calc.py": GOOD_CALC}))

    result = run_worker(
        repo=gh.repo,
        issue=ISSUE,
        base_commit=scratch_repo.head(),
        workspace=workspace,
        clone_url=str(scratch_repo.remote),
        client=gh,
        editor=editor,
    )

    assert not result.passed and result.commit is None
    assert result.exit_code == EXIT_TASK_FAILED
    assert result.verify_output == failure
    assert audited == [(result.root, VERIFY_COMMAND, ("calc.py",))]


@pytest.mark.usefixtures("worker_env")
def test_a_failed_gate_is_never_audited(fake_github, scratch_repo, workspace, monkeypatch):
    # A failed verify already carries its own, better feedback; auditing it
    # would only replace a real failure with a bureaucratic one.
    monkeypatch.setattr(entrypoint, "audit_collection", _never_collects)
    gh, _, _ = fake_github(issue(verify=ALWAYS_FAILS))
    editor = FakeEditor(edits({"calc.py": GOOD_CALC}))

    result = run_worker(
        repo=gh.repo,
        issue=ISSUE,
        base_commit=scratch_repo.head(),
        workspace=workspace,
        clone_url=str(scratch_repo.remote),
        client=gh,
        editor=editor,
    )

    assert not result.passed and result.commit is None


def test_written_test_files_match_pytests_default_shapes():
    written = (
        "calc.py",
        "tests/test_db.py",
        "wallet_test.py",
        "conftest.py",
        "test_data.json",
        "contest_entry.py",
    )

    assert entrypoint.written_test_files(written) == ("tests/test_db.py", "wallet_test.py")


# --------------------------------------------------------------------------
# Retry feedback: what a second attempt is told
# --------------------------------------------------------------------------


def feedback_comments() -> list[dict[str, Any]]:
    return [
        {"id": 1, "body": "a human said something encouraging"},
        {"id": 2, "body": "apiary: attempt 1 failed. stale earlier feedback"},
        {
            "id": 3,
            "body": (
                "apiary: attempt 2 failed. worker exit 1: the verify command failed\n\n"
                "Diagnosis: missing dependency 'sqlalchemy': declare it in "
                "requirements.txt (installed before the verify runs), or use the "
                "standard library instead"
            ),
        },
    ]


@pytest.mark.usefixtures("worker_env")
def test_a_retry_folds_the_latest_failure_report_into_its_goal(
    fake_github, scratch_repo, workspace
):
    """The other half of the reconciler's retry comment: a retry that is not
    told why the last attempt failed is the same attempt again. Latest comment
    wins - it is the verdict on the attempt this run is retrying."""
    gh, transport, _ = fake_github(
        issue(attempt=1, verify=ALWAYS_FAILS), response(200, feedback_comments())
    )
    editor = FakeEditor(edits({"calc.py": GOOD_CALC}))

    run_worker(
        repo=gh.repo,
        issue=ISSUE,
        base_commit=scratch_repo.head(),
        workspace=workspace,
        clone_url=str(scratch_repo.remote),
        client=gh,
        editor=editor,
    )

    prompt = editor.prompts[0]
    assert "A PREVIOUS ATTEMPT AT THIS EXACT TASK FAILED" in prompt
    assert "missing dependency 'sqlalchemy'" in prompt
    assert "stale earlier feedback" not in prompt
    # The goal itself survives underneath the failure report.
    assert "subtracts its second argument" in prompt
    assert ("GET", f"/repos/{gh.repo}/issues/{ISSUE}/comments") in transport.calls


@pytest.mark.usefixtures("worker_env")
def test_a_marker_carrying_the_reconcilers_signature_fields_is_a_normal_retry(
    fake_github, scratch_repo, workspace
):
    """The reconciler now writes `blocker=` and `streak=` into the identity
    marker. Only the reconciler consumes them; the worker's contract read must
    tolerate and ignore them, because a field that failed a container would
    turn every renewed retry into an infrastructure error."""
    gh, _, _ = fake_github(
        issue(
            marker=(
                "<!-- apiary:task id=add-sub attempt=1 "
                "blocker=ab12cd34ef streak=1 -->\n\n"
            ),
            verify=ALWAYS_FAILS,
        ),
        not_found(),
    )
    editor = FakeEditor(edits({"calc.py": GOOD_CALC}))

    result = run_worker(
        repo=gh.repo,
        issue=ISSUE,
        base_commit=scratch_repo.head(),
        workspace=workspace,
        clone_url=str(scratch_repo.remote),
        client=gh,
        editor=editor,
    )

    # The gate failed as scripted - a *task* verdict, which proves the marker
    # parsed, the attempt counted as a retry, and nothing choked on the fields.
    assert result.exit_code == EXIT_TASK_FAILED


@pytest.mark.usefixtures("worker_env")
def test_a_comment_fetch_failure_is_a_retry_without_feedback_not_an_error(
    fake_github, scratch_repo, workspace
):
    """A retry without feedback is yesterday's behaviour, not a failure: the
    comment is an aid, never a prerequisite, so a 404 on the thread must not
    turn the attempt into infrastructure."""
    gh, _, _ = fake_github(issue(attempt=1, verify=ALWAYS_FAILS), not_found())
    editor = FakeEditor(edits({"calc.py": GOOD_CALC}))

    result = run_worker(
        repo=gh.repo,
        issue=ISSUE,
        base_commit=scratch_repo.head(),
        workspace=workspace,
        clone_url=str(scratch_repo.remote),
        client=gh,
        editor=editor,
    )

    assert "A PREVIOUS ATTEMPT" not in editor.prompts[0]
    assert result.exit_code == EXIT_TASK_FAILED


def test_the_worker_prompt_teaches_the_dependency_rule():
    """One rule, stated where the code is written: packages exist only if
    declared in requirements.txt, and the standard library is preferred."""
    assert "requirements.txt" in edit_module.SYSTEM
    assert "standard library" in edit_module.SYSTEM


# --------------------------------------------------------------------------
# The readable context set
# --------------------------------------------------------------------------


def test_context_is_the_neighbours_and_the_front_matter(scratch_repo):
    context = edit_module.gather_context(scratch_repo.path, ["calc.py"])
    paths = [source.path for source in context]

    assert "README.md" in paths
    assert "test_calc.py" in paths
    # The write set is shown in full elsewhere in the same prompt.
    assert "calc.py" not in paths
    # Deterministic: two runs of the same task see the same repository.
    assert [s.path for s in edit_module.gather_context(scratch_repo.path, ["calc.py"])] == paths


def test_context_respects_its_budget(scratch_repo):
    assert edit_module.gather_context(scratch_repo.path, ["calc.py"], budget=1) == ()


def test_context_skips_binaries_and_vendor_directories(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "thing.py").write_text("x = 1\n")
    (root / "src" / "neighbour.py").write_text("y = 2\n")
    (root / "src" / "blob.png").write_bytes(b"\x89PNG\r\n")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "dep.js").write_text("module.exports = 1\n")

    paths = [source.path for source in edit_module.gather_context(root, ["src/thing.py"])]

    assert paths == ["src/neighbour.py"]


def _js_project(root: Path, *, lockfile: str = "package-lock.json") -> None:
    """A JS repo whose lockfile is the size real lockfiles actually are.

    600KB is not an exaggeration for effect: a measured Expo lockfile is 16,347
    lines. `_read` truncates it to `MAX_FILE_CHARS`, which is the whole problem
    - 20,000 of a 24,000 character budget, spent on a resolved dependency graph.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.js").write_text("require('./server')\n")
    (root / "package.json").write_text("j" * 2_000)
    (root / lockfile).write_text("L" * 600_000)
    # Both sort *after* `package-lock.json`, which is what makes them the
    # victims: the budget is already gone by the time the walk reaches them.
    (root / "server.js").write_text("s" * 3_000)
    (root / "utils.js").write_text("u" * 3_000)


def test_a_lockfile_is_not_read_into_the_context(tmp_path):
    """The ticket's headline criterion: the manifest survives, the lock does not."""
    root = tmp_path / "repo"
    _js_project(root)

    paths = [source.path for source in edit_module.gather_context(root, ["index.js"])]

    assert "package.json" in paths
    assert "package-lock.json" not in paths


def test_a_lockfile_starves_the_files_that_sort_after_it(tmp_path, monkeypatch):
    """Budget occupancy, before and against the same tree.

    This is the bug stated as a measurement rather than an assertion about a
    file list. With the skip disabled the lockfile takes 20,000 of the 24,000
    characters - 83% - and every neighbour after it alphabetically is dropped
    for want of room, on every task in that repository forever.
    """
    root = tmp_path / "repo"
    _js_project(root)

    def spend(context):
        return sum(len(source.text or "") for source in context)

    monkeypatch.setattr(edit_module, "CONTEXT_SKIP_FILES", frozenset())
    before = edit_module.gather_context(root, ["index.js"])
    monkeypatch.undo()
    after = edit_module.gather_context(root, ["index.js"])

    # Before: the lockfile is in, and the two real source files are not.
    assert [source.path for source in before] == ["package.json", "package-lock.json"]
    assert spend(before) > 20_000

    # After: the lockfile is out, and the budget it was holding buys the source.
    assert [source.path for source in after] == ["package.json", "server.js", "utils.js"]
    assert spend(after) < 10_000


@pytest.mark.parametrize("name", sorted(edit_module.CONTEXT_SKIP_FILES))
def test_every_named_lockfile_is_kept_out_of_the_context(tmp_path, name):
    """Parametrised over the constant itself, so an entry added without being
    reachable is still pinned - `CONTEXT_SUFFIXES` is an allow-list today and
    only two of these get past it, but adding `.lock` to it must not silently
    reopen this."""
    root = tmp_path / name.replace(".", "-")
    _js_project(root, lockfile=name)

    paths = [source.path for source in edit_module.gather_context(root, ["index.js"])]

    assert name not in paths


def test_a_lockfile_the_task_was_told_to_edit_is_still_read(tmp_path):
    """The skip is ambient. A task whose `## Files` names a lockfile is a task
    about that lockfile, and refusing to show it would make the task
    unimplementable rather than cheap."""
    root = tmp_path / "repo"
    _js_project(root)

    writable = edit_module.read_writable(root, ["package-lock.json"])

    assert [source.path for source in writable] == ["package-lock.json"]
    assert writable[0].text
    assert writable[0].truncated


@pytest.mark.usefixtures("worker_env")
def test_the_context_reaches_the_prompt_marked_read_only(fake_github, scratch_repo, workspace):
    gh, _, _ = fake_github(issue())
    editor = FakeEditor(edits({"calc.py": GOOD_CALC}))

    run_worker(
        repo=gh.repo,
        issue=ISSUE,
        base_commit=scratch_repo.head(),
        workspace=workspace,
        clone_url=str(scratch_repo.remote),
        client=gh,
        editor=editor,
    )

    prompt = editor.prompts[0]
    assert "--- calc.py (editable) ---" in prompt
    assert "--- README.md (read-only context) ---" in prompt
    assert "Files you may edit: calc.py" in prompt


# --------------------------------------------------------------------------
# Exit codes
# --------------------------------------------------------------------------


@pytest.mark.usefixtures("worker_env")
def test_a_dead_model_is_infrastructure(fake_github, scratch_repo, workspace):
    """Exit 2: the attempt must not be consumed by a broken host (#18)."""
    gh, _, _ = fake_github(issue())
    editor = FakeEditor(RuntimeError("connection refused"))

    assert main(argv(scratch_repo, workspace), client=gh, editor=editor) == EXIT_INFRASTRUCTURE


@pytest.mark.usefixtures("worker_env")
def test_a_malformed_contract_is_a_task_failure(fake_github, scratch_repo, workspace):
    """Exit 1: the body will parse exactly as badly next time."""
    body = "## Goal\nsomething\n\n## Files\n- calc.py\n\n## Blocked by\n_none._\n"
    gh, _, _ = fake_github(response(200, {"number": ISSUE, "title": "no gate", "body": body}))
    editor = FakeEditor(edits({"calc.py": GOOD_CALC}))

    assert main(argv(scratch_repo, workspace), client=gh, editor=editor) == EXIT_TASK_FAILED
    # And nothing was cloned: the contract is read before any work begins.
    assert not (workspace / f"issue-{ISSUE}").exists()


@pytest.mark.usefixtures("worker_env")
def test_a_contract_error_names_the_missing_section(fake_github, tmp_path):
    body = "## Goal\nsomething\n\n## Files\n- calc.py\n\n## Blocked by\n_none._\n"
    gh, _, _ = fake_github(response(200, {"number": ISSUE, "title": "no gate", "body": body}))

    with pytest.raises(ContractError) as caught:
        run_worker(
            repo=gh.repo,
            issue=ISSUE,
            base_commit="HEAD",
            workspace=tmp_path,
            clone_url="unused",
            client=gh,
            editor=FakeEditor(WorkerOutput(edits=[])),
        )

    assert caught.value.section == "Verify"


@pytest.mark.usefixtures("worker_env")
def test_a_clone_that_will_not_clone_is_infrastructure(fake_github, scratch_repo, workspace):
    gh, _, _ = fake_github(issue())
    editor = FakeEditor(edits({"calc.py": GOOD_CALC}))
    args = argv(scratch_repo, workspace, "--clone-url", str(workspace / "nowhere.git"))

    assert main(args, client=gh, editor=editor) == EXIT_INFRASTRUCTURE


def test_an_unusable_repo_argument_fails_before_anything_runs(tmp_path):
    """No client to fall back on, and a path is not `owner/name`."""
    code = main(
        ["--repo", str(tmp_path), "--issue", "7", "--base-commit", "abc1234",
         "--workspace", str(tmp_path / "ws")]
    )

    assert code == EXIT_INFRASTRUCTURE


# --------------------------------------------------------------------------
# Argument shapes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("shahrestani-me/apiary",
         ("shahrestani-me/apiary", "https://github.com/shahrestani-me/apiary.git")),
        ("https://github.com/shahrestani-me/apiary.git",
         ("shahrestani-me/apiary", "https://github.com/shahrestani-me/apiary.git")),
        ("git@github.com:shahrestani-me/apiary.git",
         ("shahrestani-me/apiary", "git@github.com:shahrestani-me/apiary.git")),
        ("/tmp/scratch.git", (None, "/tmp/scratch.git")),
    ],
)
def test_split_repo(value, expected):
    assert split_repo(value) == expected


def test_result_reports_the_gate_it_used(tmp_path):
    """#17 builds the PR body out of this, so the fields are part of the contract."""
    result = WorkerResult(
        issue=ISSUE,
        repo="shahrestani-me/apiary",
        task_id="add-sub",
        branch=task_branch(task_ref(ISSUE), 0),
        root=tmp_path,
        verify_command="pytest -q",
        verify_output="1 passed",
        passed=True,
        commit="abc1234",
        written=("calc.py",),
    )

    assert result.exit_code == EXIT_OK
    assert "PASS" in result.summary() and "pytest -q" in result.summary()
    # Verified but nothing to push is not a success: there is no PR to open.
    assert WorkerResult(**{**vars(result), "commit": None}).exit_code == EXIT_TASK_FAILED


# --------------------------------------------------------------------------
# The real model
# --------------------------------------------------------------------------


@pytest.mark.ollama
@pytest.mark.usefixtures("worker_env")
def test_the_real_worker_model_closes_the_loop(fake_github, scratch_repo, workspace):
    """The only test that calls Ollama: prompt in, commit out, gate believed.

    Deselected by default (`tests/conftest.py`), because a suite that needs a
    36 GB host is a suite CI cannot run. Enable with `--with-ollama`.
    """
    gh, _, _ = fake_github(issue(), *publishes())

    assert main(argv(scratch_repo, workspace), client=gh) == EXIT_OK
    assert checkout(workspace, scratch_repo).head() != scratch_repo.head()


# --------------------------------------------------------------------------
# A gate that never opened
# --------------------------------------------------------------------------


@pytest.mark.usefixtures("worker_env")
def test_a_verify_command_that_does_not_exist_is_infrastructure(
    fake_github, scratch_repo, workspace
):
    """Exit 2, not 1: nothing about this says the model was wrong.

    Found on the first real local run. The command was `python -m pytest`, on a
    machine that has only `python3`. The model's code was correct both times;
    the shell said 127. Classified as a task failure it would fail identically
    on all three attempts, exhaust the budget, reach `swarm:failed`, and feed
    the model three rounds of "command not found" as if it were review.
    """
    gh, _, _ = fake_github(issue(verify="definitely-not-a-real-command --quiet"))
    editor = FakeEditor(edits({"calc.py": GOOD_CALC}))

    code = main(argv(scratch_repo, workspace), client=gh, editor=editor)

    assert code == EXIT_INFRASTRUCTURE


@pytest.mark.usefixtures("worker_env")
def test_a_command_that_runs_and_fails_is_still_the_task(
    fake_github, scratch_repo, workspace
):
    """The narrowness is the point.

    Only 126 and 127 mean "never ran". A suite that exits 3 has genuinely
    failed, and calling that infrastructure would hand back an unconsumed
    attempt to work that can fail again for free.
    """
    gh, _, _ = fake_github(issue(verify=ALWAYS_FAILS))
    editor = FakeEditor(edits({"calc.py": GOOD_CALC}))

    code = main(argv(scratch_repo, workspace), client=gh, editor=editor)

    assert code == EXIT_TASK_FAILED


@pytest.mark.usefixtures("worker_env")
def test_a_private_clone_carries_credentials_and_never_the_url(monkeypatch, tmp_path, scratch_repo):
    """A private repository needs a token to clone, not only to push.

    The first real worker against a private repo failed with "could not read
    Username for 'https://github.com'", which reads like a missing terminal
    rather than a missing credential. The token goes through the same helper
    #17 uses for the push, so it never reaches .git/config or a git error
    string - asserted here by scanning the argv.
    """
    from swarm.worker.entrypoint import _credentials
    from swarm.worker.pr import TOKEN_ENV, TOKEN_ENV_SOURCE

    monkeypatch.setenv(TOKEN_ENV_SOURCE, "github_pat_" + "c" * 40)
    config, env = _credentials()

    assert config[:2] == ["-c", "credential.helper="]
    assert "credential.helper=" in config[3]
    assert env[TOKEN_ENV].startswith("github_pat_")
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    # The secret is named on the command line, never spelled on it.
    assert not any("github_pat_" in part for part in config)


@pytest.mark.usefixtures("worker_env")
def test_a_public_clone_still_needs_no_token(monkeypatch):
    """No credential, no helper - and prompts still disabled, so a private
    repo fails fast rather than hanging on a prompt no container can answer."""
    from swarm.worker.entrypoint import _credentials
    from swarm.worker.pr import TOKEN_ENV_SOURCE

    monkeypatch.delenv(TOKEN_ENV_SOURCE, raising=False)
    config, env = _credentials()

    assert config == []
    assert env == {"GIT_TERMINAL_PROMPT": "0"}


# --------------------------------------------------------------------------
# Telling a gate that never opened from a task that failed (#90)
# --------------------------------------------------------------------------
#
# Every test below is **unit, bare CI and unmarked**, deliberately.
# `.github/workflows/ci.yml` deselects the `docker` marker, so a regression
# test carrying it is a regression test that never runs - and the bug this
# section is about shipped once already with a green suite.
#
# `UNRUNNABLE_EXIT_CODES` is a *shell*-level signal: it fires only when the
# binary is absent. Every modern toolchain catches its own errors and
# normalises to exit 1, so commit c015e4f's fix held for 127 and reopened for
# every stack that installs anything.


def denied_npm() -> tuple[int, str, str]:
    """A denied `npm install`, as measured through the real egress tinyproxy.

    Exit **1**, in under a second. That is the whole problem in one line.
    """
    return 1, "", "npm error code E403\nnpm error 403 Filtered by the proxy"


def test_a_command_that_passed_is_passed():
    assert edit_module and classify_verify(0, "1 passed", "", 3.2).outcome == PASSED


@pytest.mark.parametrize("code", sorted(UNRUNNABLE_EXIT_CODES))
def test_a_shell_that_could_not_run_it_is_infrastructure(code):
    """Today's behaviour, preserved. 127 is not there, 126 may not be executed."""
    verdict = classify_verify(code, "", "pytest: command not found", 0.01)

    assert verdict.outcome == INFRASTRUCTURE
    assert str(code) in verdict.reason


def test_a_denied_install_is_infrastructure_even_though_it_exited_one():
    """The reopened half of c015e4f, and the reason this ticket exists.

    Three attempts burned in ~3 seconds and then `swarm:failed`, for a task
    whose code was never the problem.
    """
    verdict = classify_verify(*denied_npm(), 0.8)

    assert verdict.outcome == INFRASTRUCTURE
    assert "network" in verdict.reason


@pytest.mark.parametrize(
    "stderr",
    [
        "npm error code E403",
        "ERROR: 403 Filtered",
        "fatal: unable to access 'https://github.com/': Could not resolve host: github.com",
        "curl: (5) Could not resolve proxy: apiary-egress",
        "Temporary failure in name resolution",
    ],
)
def test_every_denial_signature_is_recognised(stderr):
    assert classify_verify(1, "", stderr, 0.4).outcome == INFRASTRUCTURE


def test_a_denial_on_stdout_counts_too():
    """npm splits its own error output across both streams depending on
    version and TTY, so looking at one of them is looking at half of them."""
    assert classify_verify(1, "npm error code E403", "", 0.4).outcome == INFRASTRUCTURE


def test_an_out_of_memory_kill_is_infrastructure_and_says_so():
    """The VM is 7.65 GiB and two workers at `--memory 4g` overcommit it.
    Python used ~100MB so nobody ever saw this; a toolchain will."""
    verdict = classify_verify(137, "", "", 41.0)

    assert verdict.outcome == INFRASTRUCTURE
    assert "memory" in verdict.reason


def test_an_ordinary_test_failure_is_the_tasks_fault():
    verdict = classify_verify(1, "1 failed, 3 passed\nassert 3 == 4", "", 12.0)

    assert verdict.outcome == TASK_FAILED
    assert not verdict.infrastructure


@pytest.mark.parametrize("code", [2, 3, 5, 139, 255])
def test_an_unrecognised_failure_fails_closed(code):
    """Fail closed. The two errors are not symmetric: a task failure misread as
    infrastructure never consumes an attempt, so it retries forever.

    139 is a native segfault and is deliberately *not* infrastructure - the
    generated code is as likely a cause as the host is.
    """
    assert classify_verify(code, "", "boom", 1.0).outcome == TASK_FAILED


def test_the_classifier_needs_no_subprocess_no_docker_and_no_model():
    """Stated as a test because it is the point of extracting the function.

    The decision used to live inline around a `subprocess.run`, which is why
    the only rule it ever grew was the one a shell hands you for free. Purity
    is what puts the truth table above in bare CI.

    Asserted over the bytecode's name table rather than over the source text,
    because the source text includes the docstring and the docstring is
    *about* subprocesses.
    """
    referenced = set(classify_verify.__code__.co_names)

    assert not referenced & {"subprocess", "run", "os", "environ", "SETTINGS", "Path"}


# --- the environment the gate can see ---------------------------------------


def test_the_verify_environment_carries_no_credential():
    environ = {
        "PATH": "/usr/bin",
        "HOME": "/workspace",
        "GITHUB_TOKEN": "github_pat_" + "a" * 30,
        "APIARY_PUSH_TOKEN": "github_pat_" + "b" * 30,
    }

    env = verify_env(environ)

    assert "GITHUB_TOKEN" not in env
    assert "APIARY_PUSH_TOKEN" not in env


@pytest.mark.parametrize("name", VERIFY_ENV_REQUIRED)
def test_every_load_bearing_variable_survives(name):
    """Dropping `HTTPS_PROXY` does not make a verify command fail, it makes it
    **hang**: a worker has no default route, so a request with nowhere to go
    waits until the outer container clock kills it several hundred seconds
    later, with a reason naming the container."""
    environ = {var: f"value-of-{var}" for var in VERIFY_ENV_REQUIRED}

    env = verify_env(environ)

    assert env[name] == f"value-of-{name}"


def test_the_environment_is_filtered_rather_than_rebuilt():
    """A fresh dict is how this goes wrong. Anything the target repository's
    own suite needs - `PYTHONPATH`, `NODE_ENV`, a database URL - is not
    something this module can enumerate, and an allow-list would break it in a
    way that looks like a bug in its own tests."""
    environ = {"PATH": "/usr/bin", "PYTHONPATH": "/src", "NODE_ENV": "test", "LANG": "C"}

    assert verify_env(environ) == environ


def test_a_credential_nobody_named_is_still_dropped():
    """The shape test is `containers.manager.SECRET_NAME_RE`, reused rather
    than restated - the same regex that already enrols a container's variables
    for redaction."""
    environ = {"PATH": "/usr/bin", "STRIPE_SECRET": "x", "MY_API_KEY": "y", "DB_PASSWORD": "z"}

    assert set(verify_env(environ)) == {"PATH"}


def test_a_verify_command_that_prints_its_environment_sees_no_token(
    tmp_path, monkeypatch
):
    """End to end through the real `run_verify`, with a scripted double for the
    command - the repo's own idiom. No Docker, no marker, so CI runs it."""
    monkeypatch.setenv("GITHUB_TOKEN", "github_pat_" + "a" * 30)
    monkeypatch.setenv("APIARY_PUSH_TOKEN", "github_pat_" + "b" * 30)
    # Every required name but `PATH`, which has to stay real: the verify
    # command is run through a shell, and a shell with a made-up `PATH` cannot
    # find `env` to print anything at all. That failure mode is itself the
    # point of `VERIFY_ENV_REQUIRED`.
    pinned = [name for name in VERIFY_ENV_REQUIRED if name != "PATH"]
    for name in pinned:
        monkeypatch.setenv(name, f"value-of-{name}")

    # Filtered rather than dumped whole: `run_verify` keeps only the last
    # `OUTPUT_TAIL_CHARS`, and a full `env` on a developer's machine overflows
    # that and cuts off the very lines under assertion.
    passed, output = run_verify(
        tmp_path, "env | grep -E 'TOKEN|PROXY|proxy|^PATH=|^HOME='"
    )

    assert passed
    assert "GITHUB_TOKEN" not in output
    assert "APIARY_PUSH_TOKEN" not in output
    assert "PATH=" in output
    for name in pinned:
        assert f"{name}=value-of-{name}" in output


def test_a_denied_command_never_consumes_the_attempt(tmp_path):
    """Through `run_verify`, so the wiring is asserted and not just the
    classifier. Reproduced with a scripted double, as `test_worker_entrypoint`
    already does at :207, :266 and :297."""
    with pytest.raises(InfrastructureError) as raised:
        run_verify(tmp_path, 'sh -c \'echo "npm error code E403" >&2; exit 1\'')

    assert "network" in str(raised.value)


def test_an_ordinary_failure_still_returns_rather_than_raising(tmp_path):
    passed, output = run_verify(tmp_path, 'sh -c \'echo "1 failed"; exit 1\'')

    assert not passed
    assert "1 failed" in output


# --------------------------------------------------------------------------
# What the record says about an attempt that died (#97)
# --------------------------------------------------------------------------


def test_a_truncated_verify_output_says_it_was_truncated(tmp_path):
    """A complete 3KB log and the last 4KB of a 400KB npm log used to be
    indistinguishable, which is the difference between "this is the failure"
    and "the failure is somewhere above this"."""
    # `seq` rather than a quoted Python string: the output has to be genuinely
    # long, and shell-quoting 13KB of it into the command is its own bug.
    _, output = run_verify(tmp_path, "sh -c 'seq 1 3000; exit 1'")

    assert "earlier characters elided" in output
    assert len(output) < 5_000


def test_a_short_verify_output_is_not_marked(tmp_path):
    _, output = run_verify(tmp_path, "sh -c 'echo \"1 failed\"; exit 1'")

    assert "elided" not in output
    assert "1 failed" in output


def test_the_marker_is_the_records_own(tmp_path):
    """`result.tail`, not a local slice. Two artifacts disagreeing about what a
    truncated log looks like is worse than neither marking one."""
    from swarm.worker.result import tail

    _, output = run_verify(
        tmp_path, f'{sys.executable} -c "print(\'z\' * 9000); raise SystemExit(1)"'
    )

    assert output.startswith(tail("z" * 9_000, OUTPUT_TAIL_CHARS)[:20])


def test_an_attempt_that_died_after_its_gate_reports_the_gate():
    """`_unrun` blanked `verify_command` and `written` on exactly the path
    where they matter most. "The clone failed" and "the gate passed and the
    push failed" are both exit 2; only one of them has a command and a file
    list, and reporting an empty command for it actively misleads."""
    error = InfrastructureError("pushing issue #7: remote hung up")
    error.learned(verify_command="npm test", written=("src/a.js",), task_id="add-thing")

    result = entrypoint._unrun(
        7,
        "owner/name",
        str(error),
        verify_command=error.verify_command,
        written=error.written,
        task_id=error.task_id,
    )

    assert result.verify_command == "npm test"
    assert result.written == ("src/a.js",)
    assert result.task_id == "add-thing"


def test_an_attempt_that_died_before_its_gate_still_reports_nothing():
    """The other half. A clone that failed genuinely has neither, and inventing
    a command for it would be the same lie in the other direction."""
    result = entrypoint._unrun(7, "owner/name", "cloning failed")

    assert result.verify_command == ""
    assert result.written == ()


def test_context_is_attached_by_the_frame_that_has_it():
    """`run_verify` cannot know which files were written and `commit_edits`
    cannot know the gate command, so `learned` fills in from the frame that has
    both rather than threading context through every raise site."""
    error = InfrastructureError("boom")
    assert (error.verify_command, error.written) == ("", ())

    error.learned(verify_command="pytest -q", written=("a.py",), task_id="t")
    # Idempotent, and never overwrites what the raiser did know.
    error.learned(verify_command="npm test", written=("b.js",), task_id="u")

    assert error.verify_command == "pytest -q"
    assert error.written == ("a.py",)


# --------------------------------------------------------------------------
# Files the gate generates but the model cannot write (#105)
# --------------------------------------------------------------------------
#
# `commit_edits` stages exactly the declared `## Files`, and that rule is
# right - `git add -A` after a verify run would sweep `node_modules` and every
# cache the command wrote into the PR. But a lockfile is neither declarable nor
# writable: a measured Expo lockfile is 16,347 lines against a 16,384-token
# window, and it carries SHA-512 hashes that cannot be produced by generation.
#
# Without this, the PR carries a `package.json` change and no lock, CI re-runs
# the command on neutral ground, and `npm ci` fails. "Add a dependency" is
# unimplementable.


def files_in(repo, commit: str) -> list[str]:
    """The paths one commit touched. A local helper rather than a `ScratchRepo`
    method: `fixtures/repo.py` is shared and outside this ticket's file set."""
    out = repo.out("show", "--name-only", "--format=", commit)
    return sorted(line.strip() for line in out.splitlines() if line.strip())


def test_a_generated_file_the_gate_produced_is_committed(scratch_repo):
    """The headline: a lockfile the verify command created reaches the commit
    even though no task declared it and no model wrote it."""
    (scratch_repo.path / "calc.py").write_text(GOOD_CALC)
    (scratch_repo.path / "package-lock.json").write_text('{"lockfileVersion": 3}')

    commit = commit_edits(
        scratch_repo.path, "swarm[t]: add sub", ["calc.py"], generated=["package-lock.json"]
    )

    assert commit
    assert "package-lock.json" in files_in(scratch_repo, commit)


def test_a_generated_file_the_gate_did_not_produce_is_simply_absent(scratch_repo):
    """Absent is normal, not an error. `GENERATED_FILES` names what a stack's
    gate *may* write, and most tasks add no dependency - failing over a file
    the task never needed would make the set a requirement rather than a
    permission."""
    (scratch_repo.path / "calc.py").write_text(GOOD_CALC)

    commit = commit_edits(
        scratch_repo.path, "swarm[t]: add sub", ["calc.py"], generated=["package-lock.json"]
    )

    assert commit
    assert files_in(scratch_repo, commit) == ["calc.py"]


def test_the_staging_rule_still_refuses_everything_else(scratch_repo):
    """The rule this ticket had to widen without loosening. `node_modules` is
    what `git add -A` would have swept in, and it is what a lockfile sits next
    to by construction."""
    (scratch_repo.path / "calc.py").write_text(GOOD_CALC)
    (scratch_repo.path / "package-lock.json").write_text("{}")
    modules = scratch_repo.path / "node_modules" / "left-pad"
    modules.mkdir(parents=True)
    (modules / "index.js").write_text("module.exports = 1\n")
    (scratch_repo.path / ".npm-cache").write_text("junk")

    commit = commit_edits(
        scratch_repo.path, "swarm[t]: add sub", ["calc.py"], generated=["package-lock.json"]
    )

    assert sorted(files_in(scratch_repo, commit)) == ["calc.py", "package-lock.json"]


def test_a_task_with_no_generated_set_commits_exactly_what_it_declared(scratch_repo):
    """Python generates nothing, so this is every task that exists today and
    its behaviour must be byte-identical."""
    (scratch_repo.path / "calc.py").write_text(GOOD_CALC)
    (scratch_repo.path / "package-lock.json").write_text("{}")

    commit = commit_edits(scratch_repo.path, "swarm[t]: add sub", ["calc.py"])

    assert files_in(scratch_repo, commit) == ["calc.py"]


def test_a_generated_path_is_re_resolved_against_the_root(tmp_path):
    """`stageable` turns a name into a `git add`, and a symlink is a name that
    points somewhere else. The constants are the repository's own, which is a
    reason to check cheaply rather than a reason not to check."""
    root = tmp_path / "repo"
    (root / "sub").mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (root / "package-lock.json").symlink_to(outside)

    assert stageable(root, ["package-lock.json"]) == ()


def test_a_generated_directory_is_not_a_generated_file(tmp_path):
    root = tmp_path / "repo"
    (root / "package-lock.json").mkdir(parents=True)

    assert stageable(root, ["package-lock.json"]) == ()
