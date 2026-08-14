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

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pytest
from fixtures.github import response
from fixtures.repo import VERIFY_COMMAND, ScratchRepo

from swarm.github.ledger import ContractError
from swarm.state import FileEdit, WorkerOutput
from swarm.worker import edit as edit_module
from swarm.worker.entrypoint import (
    EXIT_INFRASTRUCTURE,
    EXIT_OK,
    EXIT_TASK_FAILED,
    WorkerResult,
    main,
    run_worker,
    split_repo,
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
) -> str:
    marker = f"<!-- apiary:task id={task_id} attempt=0 -->\n\n" if task_id else ""
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
def test_verified_task_produces_a_commit(fake_github, scratch_repo, workspace):
    gh, transport, _ = fake_github(issue(), *publishes())
    base = scratch_repo.head()
    editor = FakeEditor(edits({"calc.py": GOOD_CALC}))

    assert main(argv(scratch_repo, workspace), client=gh, editor=editor) == EXIT_OK

    work = checkout(workspace, scratch_repo)
    assert work.current_branch() == f"swarm/issue-{ISSUE}"
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
        branch=f"swarm/issue-{ISSUE}",
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
