"""Push, one PR, and the token that must not be anywhere.

Four claims from #17, each with a test that fails loudly if it stops holding:

- a finished run pushes its branch and leaves exactly one open PR carrying
  `Closes #<n>`, the verify command and its output tail;
- a second run on the same issue updates that PR instead of opening another,
  on both the client that can find it and the client that cannot;
- the token reaches git through the environment and a credential helper, and
  appears in no argv, no `.git/config` and no error message;
- the worker writes `swarm:review` and no other label.

Hermetic throughout. The GitHub side is `fixtures/github.py`'s scripted
transport, the git side is `fixtures/repo.py`'s bare repository standing in for
`origin`, and one test drives `entrypoint.main` end to end so that the seam
between #16 and #17 - `swarm.worker.pr.publish(result, *, client)`, found by
`find_spec` - is exercised by something other than a comment.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pytest
from fixtures.github import REPO, response
from fixtures.repo import VERIFY_COMMAND, ScratchRepo

from swarm.github.branches import task_branch
from swarm.github.refs import task_ref
from swarm.state import FileEdit, WorkerOutput
from swarm.worker import pr as pr_module
from swarm.worker.entrypoint import EXIT_INFRASTRUCTURE, EXIT_OK, WorkerResult, main
from swarm.worker.pr import (
    CREDENTIAL_HELPER,
    REVIEW_LABEL,
    TOKEN_ENV,
    PublishError,
    publish,
    pull_request_body,
    push_branch,
    push_command,
)

ISSUE = 7
#: Attempt 0's branch: the fixtures below are a first attempt, and #144 puts
#: the attempt in the name rather than leaving one name to serve every one.
BRANCH = task_branch(task_ref(ISSUE), 0)
OWNER = REPO.split("/")[0]
TOKEN = "ghp-not-a-real-token-0123456789"

#: `calc.py` with a second function, and the seeded test still passing - the
#: same good generation `test_worker_entrypoint.py` uses.
GOOD_CALC = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"


# --------------------------------------------------------------------------
# Doubles and builders
# --------------------------------------------------------------------------


@dataclass
class FakeEditor:
    """The worker model, minus the model. `.invoke(messages)` is the whole seam."""

    output: Any
    prompts: list[str] = field(default_factory=list)

    def invoke(self, messages: Sequence[tuple[str, str]]) -> WorkerOutput:
        self.prompts.append(messages[-1][1])
        return self.output


def pull(number: int = 42, *, branch: str = BRANCH) -> dict[str, Any]:
    """A PR payload, in the shape GitHub returns and this module reads."""
    return {
        "number": number,
        "html_url": f"https://github.com/{REPO}/pull/{number}",
        "head": {"ref": branch},
        "state": "open",
    }


def already_exists() -> Any:
    """GitHub's 422 for a second PR on a head branch that already has one.

    The message is the load-bearing part: it is how the fallback path tells
    "the PR is already there" from "no commits between the branches", which is
    also a 422 and is a real failure.
    """
    return response(
        422,
        {
            "message": "Validation Failed",
            "errors": [
                {
                    "resource": "PullRequest",
                    "code": "custom",
                    "message": f"A pull request already exists for {OWNER}:{BRANCH}.",
                }
            ],
        },
    )


def no_open_pulls():
    """`GET /pulls?state=open` finding nothing.

    Every publish now begins with this: `find_open_pull_request` asks the
    client for the open PRs on this head, and since `GitHubClient` grew
    `list_pull_requests` the question actually reaches the transport instead of
    being skipped. A script that omits it is asserting the old behaviour.
    """
    return response(200, [])


def can_list_pulls(client: Any, *pulls: dict[str, Any]) -> Any:
    """Give a client the public listing method `GitHubClient` does not have yet.

    #17 may not edit `client.py`, so `pr.find_open_pull_request` probes for
    `list_pull_requests` and degrades without it. Both paths are real - this is
    the one where the reconciler (#23) has since added the method - so both are
    tested, and this is the seam that lets them be.
    """
    client.list_pull_requests = lambda **kwargs: list(pulls)
    return client


def issue_body(verify: str = VERIFY_COMMAND) -> str:
    return (
        "<!-- apiary:task id=add-sub attempt=0 -->\n\n"
        "## Goal\n`calc.sub` subtracts its second argument from its first.\n\n"
        "## Files\n- calc.py\n\n"
        f"## Verify\n{verify}\n\n"
        "## Blocked by\n_none._\n"
    )


def issue_response(**kwargs: Any) -> Any:
    return response(
        200, {"number": ISSUE, "title": "Add a sub function", "body": issue_body(**kwargs)}
    )


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def worker_env(scratch_repo: ScratchRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixture's git isolation, applied to the subprocesses this module runs.

    Without it the push would pick up the developer's `~/.gitconfig` - hooks,
    `commit.gpgsign`, a credential helper of their own - and the suite would
    pass on one laptop and hang on another.
    """
    for key, value in scratch_repo.env.items():
        monkeypatch.setenv(key, value)
    # Never inherit a real token from the developer's shell: `publish` falls
    # back to GITHUB_TOKEN, and a push to a bare repo on disk would not notice
    # while the assertions on argv would.
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)


@pytest.fixture()
def finished(scratch_repo: ScratchRepo, tmp_path: Path) -> WorkerResult:
    """A clone with a verified commit on the attempt-0 branch - a worker's end state."""
    return attempt(scratch_repo, tmp_path / "attempt-1", GOOD_CALC, "swarm[add-sub]: add sub")


def attempt(scratch: ScratchRepo, dest: Path, content: str, message: str) -> WorkerResult:
    """One worker attempt, up to the commit `entrypoint.run_worker` would make.

    The branch is recreated from the base commit every time, exactly as
    `prepare_checkout` does it, so a second attempt's history diverges from the
    first's - which is what makes the push a force.
    """
    work = scratch.clone(dest)
    work.git("switch", "-q", "--force-create", BRANCH, scratch.default_branch)
    work.write("calc.py", content)
    commit = work.commit(message)
    return WorkerResult(
        issue=ISSUE,
        repo=REPO,
        task_id="add-sub",
        branch=BRANCH,
        root=work.path,
        verify_command=VERIFY_COMMAND,
        verify_output="1 passed in 0.01s",
        passed=True,
        commit=commit,
        written=("calc.py",),
    )


def result_at(root: Path, **overrides: Any) -> WorkerResult:
    """A `WorkerResult` that needs no repository - for the pure body builders."""
    fields: dict[str, Any] = {
        "issue": ISSUE,
        "repo": REPO,
        "task_id": "add-sub",
        "branch": BRANCH,
        "root": root,
        "verify_command": "pytest -q",
        "verify_output": "1 passed",
        "passed": True,
        "commit": "abc1234",
        "written": ("calc.py",),
    }
    return WorkerResult(**{**fields, **overrides})


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


@pytest.mark.usefixtures("worker_env")
def test_publish_pushes_the_branch_and_opens_one_pr(fake_github, scratch_repo, finished):
    gh, transport, _ = fake_github(no_open_pulls(), response(201, pull()), response(200, [{"name": REVIEW_LABEL}]))

    published = publish(finished, client=gh)

    assert BRANCH in scratch_repo.remote_branches()
    assert scratch_repo.remote_head(BRANCH) == finished.commit
    assert published.created and published.number == 42
    assert published.url.endswith("/pull/42")
    assert transport.calls == [
        ("GET", f"/repos/{REPO}/pulls"),
        ("POST", f"/repos/{REPO}/pulls"),
        ("POST", f"/repos/{REPO}/issues/{ISSUE}/labels"),
    ]


@pytest.mark.usefixtures("worker_env")
def test_the_pr_states_the_gate_and_closes_the_issue(fake_github, finished):
    gh, transport, _ = fake_github(no_open_pulls(), response(201, pull()), response(200, []))

    publish(finished, client=gh)

    # [0] is the lookup for an existing PR; [1] is the create.
    sent = transport.sent[1].json()
    assert sent["head"] == BRANCH
    # The base is the branch the clone came from: read from the checkout, not
    # guessed and not a second API call.
    assert sent["base"] == "main"
    assert sent["title"] == "swarm[add-sub]: add sub"
    assert "`calc.py`" in sent["body"]
    assert VERIFY_COMMAND in sent["body"]
    assert "1 passed in 0.01s" in sent["body"]
    # Alone on its line, and the only closing keyword in the body: this is what
    # makes the merge close the issue.
    assert sent["body"].splitlines()[-1] == f"Closes #{ISSUE}"


def test_the_body_survives_backticks_in_the_verify_output(tmp_path):
    """Verify output is arbitrary text, and a three-backtick fence is not enough."""
    body = pull_request_body(result_at(tmp_path, verify_output="E   assert ```x``` == ok"))

    assert "````" in body
    # The output is intact, and the section after it still renders as markdown.
    assert "```x```" in body
    assert body.splitlines()[-1] == f"Closes #{ISSUE}"


def test_deleted_files_are_listed_as_deletions_not_writes(tmp_path):
    """A removed file presented as a plain change sends the reviewer looking
    for new contents that do not exist."""
    body = pull_request_body(result_at(tmp_path, deleted=("obsolete.py",)))

    assert "- `calc.py`" in body
    assert "- `obsolete.py` (deleted)" in body


def test_refused_edits_are_reported_to_the_reviewer(tmp_path):
    """The clearest signal that `## Files` and the work asked for have drifted."""
    body = pull_request_body(
        result_at(tmp_path, refused=(("README.md", "not in the declared file set"),))
    )

    assert "## Refused" in body
    assert "`README.md` - not in the declared file set" in body


# --------------------------------------------------------------------------
# The retry path
# --------------------------------------------------------------------------


@pytest.mark.usefixtures("worker_env")
def test_a_second_run_updates_the_pr_it_can_find(fake_github, scratch_repo, tmp_path, finished):
    """The whole point of #17: one issue, one PR, however many attempts."""
    first, _, _ = fake_github(no_open_pulls(), response(201, pull()), response(200, []))
    publish(finished, client=first)

    second = attempt(scratch_repo, tmp_path / "attempt-2", GOOD_CALC, "swarm[add-sub]: retry")
    gh, transport, _ = fake_github(response(200, pull()), response(200, []))

    published = publish(second, client=can_list_pulls(gh, pull()))

    assert not published.created and published.number == 42
    assert transport.calls == [
        ("PATCH", f"/repos/{REPO}/pulls/42"),
        ("POST", f"/repos/{REPO}/issues/{ISSUE}/labels"),
    ]
    # The body now describes this attempt, and the remote branch carries this
    # attempt's commit even though the two histories diverged.
    assert transport.sent[0].json()["title"] == "swarm[add-sub]: retry"
    assert scratch_repo.remote_head(BRANCH) == second.commit


@pytest.mark.usefixtures("worker_env")
def test_a_second_run_opens_no_second_pr_without_the_listing_method(
    fake_github, scratch_repo, tmp_path, finished, capsys
):
    """The fallback: GitHub's own 422 keeps the invariant when the client cannot.

    `GitHubClient` has no way to find a PR by head branch and #17 may not add
    one, so the retry path leans on the API refusing a second PR for a head
    branch that already has one. The push has already moved the open PR's head.
    """
    first, _, _ = fake_github(no_open_pulls(), response(201, pull()), response(200, []))
    publish(finished, client=first)

    second = attempt(scratch_repo, tmp_path / "attempt-2", GOOD_CALC, "swarm[add-sub]: retry")
    gh, transport, _ = fake_github(already_exists(), response(200, []))
    # `GitHubClient` now has the listing method, so the "cannot look" path has
    # to be asked for. It is still a real path - a client stubbed by a caller,
    # or a listing that 403s - and the 422 is what keeps the invariant then.
    gh.list_pull_requests = None

    published = publish(second, client=gh)

    assert not published.created and published.number is None
    assert transport.calls == [
        ("POST", f"/repos/{REPO}/pulls"),
        ("POST", f"/repos/{REPO}/issues/{ISSUE}/labels"),
    ]
    assert scratch_repo.remote_head(BRANCH) == second.commit
    # Loud, not silent: the PR body is one attempt behind until the listing
    # method lands.
    assert "already open" in capsys.readouterr().err


@pytest.mark.usefixtures("worker_env")
def test_any_other_422_is_a_failure(fake_github, finished):
    """"No commits between main and the branch" must not read as success."""
    gh, _, _ = fake_github(
        no_open_pulls(),
        response(
            422,
            {
                "message": "Validation Failed",
                "errors": [
                    {
                        "resource": "PullRequest",
                        "code": "custom",
                        "message": f"No commits between main and {BRANCH}",
                    }
                ],
            },
        )
    )

    with pytest.raises(PublishError):
        publish(finished, client=gh)


@pytest.mark.usefixtures("worker_env")
def test_the_push_refuses_to_clobber_a_branch_that_moved(
    fake_github, scratch_repo, tmp_path, finished
):
    """`--force-with-lease`, not `--force`: a human's commit on the PR survives.

    The retry recreates the branch from the base commit, so the push has to be
    a force; the lease is what keeps it from being a weapon. The order matters
    here - this worker's clone is taken *before* the other push, which is the
    only situation in which a lease has anything to say.
    """
    first, _, _ = fake_github(no_open_pulls(), response(201, pull()), response(200, []))
    publish(finished, client=first)

    second = attempt(scratch_repo, tmp_path / "attempt-2", GOOD_CALC, "swarm[add-sub]: retry")

    human = scratch_repo.clone(tmp_path / "human", branch=BRANCH)
    human.write("calc.py", GOOD_CALC + "\n# reviewed\n")
    human_head = human.commit("review fix")
    human.push(BRANCH)

    gh, transport, _ = fake_github()
    with pytest.raises(PublishError):
        publish(second, client=gh)

    assert scratch_repo.remote_head(BRANCH) == human_head
    assert transport.calls == []


# --------------------------------------------------------------------------
# The token
# --------------------------------------------------------------------------


def test_the_token_is_in_the_environment_and_not_in_the_argv():
    argv, env = push_command(BRANCH, token=TOKEN)

    assert TOKEN not in " ".join(argv)
    assert env[TOKEN_ENV] == TOKEN
    # What argv carries is the variable's *name*, which is what makes `ps`
    # inside the container and `docker inspect` outside it uninteresting.
    assert TOKEN_ENV in " ".join(argv)
    assert "--force-with-lease" in argv


@pytest.mark.usefixtures("worker_env")
def test_the_token_reaches_no_file_in_the_checkout(fake_github, scratch_repo, finished):
    gh, _, _ = fake_github(no_open_pulls(), response(201, pull()), response(200, []))

    publish(finished, client=gh, token=TOKEN)

    assert BRANCH in scratch_repo.remote_branches()
    config = (finished.root / ".git" / "config").read_text()
    assert TOKEN not in config
    assert "x-access-token@" not in config
    # `.git/config` is the famous one, but FETCH_HEAD, the reflogs and the
    # packed refs are all files a run archive (#29) could end up keeping.
    for path in (finished.root / ".git").rglob("*"):
        if path.is_file():
            assert TOKEN.encode() not in path.read_bytes()


def test_a_git_error_carrying_the_token_is_redacted(monkeypatch, tmp_path):
    """Git errors are the stream #15 captures and #29 writes to disk."""

    def leaky(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, 128, "", f"fatal: could not read Password for 'https://x-access-token:{TOKEN}@x'"
        )

    monkeypatch.setattr(pr_module.subprocess, "run", leaky)

    with pytest.raises(PublishError) as caught:
        push_branch(tmp_path, BRANCH, token=TOKEN)

    assert TOKEN not in str(caught.value)
    assert "***" in str(caught.value)


@pytest.mark.usefixtures("worker_env")
def test_a_failed_push_opens_no_pull_request(fake_github, scratch_repo, tmp_path, finished):
    """A commit that never reached the remote must not become a PR."""
    ScratchRepo(finished.root, scratch_repo.remote, env=dict(scratch_repo.env)).git(
        "remote", "set-url", "origin", str(tmp_path / "gone.git")
    )
    gh, transport, _ = fake_github()

    with pytest.raises(PublishError):
        publish(finished, client=gh, token=TOKEN)

    assert transport.calls == []
    assert BRANCH not in scratch_repo.remote_branches()


def test_the_credential_helper_actually_answers(tmp_path, monkeypatch):
    """The helper is a shell snippet, so it is worth proving git can run it.

    `git credential fill` is the code path a push takes to obtain a credential,
    minus the network - so this asserts the token is delivered without ever
    being written into a URL.
    """
    proc = subprocess.run(
        ["git", "-c", "credential.helper=", "-c", f"credential.helper={CREDENTIAL_HELPER}",
         "credential", "fill"],
        input="protocol=https\nhost=github.com\npath=owner/repo.git\n\n",
        cwd=str(tmp_path),
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", TOKEN_ENV: TOKEN},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stderr
    assert "username=x-access-token" in proc.stdout
    assert f"password={TOKEN}" in proc.stdout


# --------------------------------------------------------------------------
# The one label
# --------------------------------------------------------------------------


@pytest.mark.usefixtures("worker_env")
def test_the_worker_writes_only_the_review_label(fake_github, finished):
    """`docs/issue-contract.md` §4: one label, and `swarm:claimed` is not touched."""
    gh, transport, _ = fake_github(no_open_pulls(), response(201, pull()), response(200, []))

    published = publish(finished, client=gh)

    assert published.labelled
    label_call = transport.sent[-1]
    assert label_call.method == "POST"
    assert label_call.json() == {"labels": [REVIEW_LABEL]}
    assert "DELETE" not in {request.method for request in transport.sent}


@pytest.mark.usefixtures("worker_env")
def test_a_label_that_does_not_stick_does_not_undo_the_pr(fake_github, finished, capsys):
    """Exit 2 here would dispatch a second container over an open PR.

    Recovery (#35) can put `swarm:review` back by looking at the PR; nothing
    can put back a PR that was never reported as opened.
    """
    gh, _, _ = fake_github(no_open_pulls(), response(201, pull()), response(403, {"message": "Forbidden"}))

    published = publish(finished, client=gh)

    assert published.created and published.number == 42
    assert not published.labelled
    assert REVIEW_LABEL in capsys.readouterr().err


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------


@pytest.mark.usefixtures("worker_env")
def test_an_unverified_result_is_never_published(fake_github, finished):
    """Otherwise: a PR body claiming a passed gate, on work that did not pass one."""
    gh, transport, _ = fake_github()

    with pytest.raises(ValueError):
        publish(WorkerResult(**{**vars(finished), "passed": False}), client=gh)
    with pytest.raises(ValueError):
        publish(WorkerResult(**{**vars(finished), "commit": None}), client=gh)

    assert transport.calls == []


# --------------------------------------------------------------------------
# The seam with #16
# --------------------------------------------------------------------------


@pytest.mark.usefixtures("worker_env")
def test_the_entrypoint_finds_and_calls_publish(fake_github, scratch_repo, tmp_path):
    """One worker run, end to end: `main` -> `_publish` -> `publish`.

    `entrypoint._publish` probes for `swarm.worker.pr` with `find_spec` and
    calls `publish(result, client=client)`. Neither suite alone would notice if
    this module were named or shaped differently - each passes on its own - so
    the seam is asserted here, from the outside.
    """
    workspace = tmp_path / "workspace"
    gh, transport, _ = fake_github(
        issue_response(),
        no_open_pulls(),
        response(201, pull()),
        response(200, [{"name": REVIEW_LABEL}]),
    )
    editor = FakeEditor(WorkerOutput(edits=[FileEdit(path="calc.py", content=GOOD_CALC)]))

    code = main(
        ["--repo", str(scratch_repo.remote), "--issue", str(ISSUE),
         "--base-commit", scratch_repo.head(), "--workspace", str(workspace)],
        client=gh,
        editor=editor,
    )

    assert code == EXIT_OK
    assert transport.calls == [
        ("GET", f"/repos/{REPO}/issues/{ISSUE}"),
        ("GET", f"/repos/{REPO}/pulls"),
        ("POST", f"/repos/{REPO}/pulls"),
        ("POST", f"/repos/{REPO}/issues/{ISSUE}/labels"),
    ]
    assert BRANCH in scratch_repo.remote_branches()
    assert f"Closes #{ISSUE}" in transport.sent[2].json()["body"]


@pytest.mark.usefixtures("worker_env")
def test_a_publish_failure_is_infrastructure_not_a_consumed_attempt(
    fake_github, scratch_repo, tmp_path, monkeypatch
):
    """`PublishError` is a `GitHubError`, which is the clause `main` catches.

    The work passed its gate, so a remote that will not take it is the host's
    problem: exit 2 keeps the attempt budget for a real failure
    (`docs/issue-contract.md` §5).
    """
    def refuse(*args, **kwargs):
        raise PublishError("remote hung up")

    monkeypatch.setattr(pr_module, "push_branch", refuse)
    gh, _, _ = fake_github(issue_response())
    editor = FakeEditor(WorkerOutput(edits=[FileEdit(path="calc.py", content=GOOD_CALC)]))

    code = main(
        ["--repo", str(scratch_repo.remote), "--issue", str(ISSUE),
         "--base-commit", scratch_repo.head(), "--workspace", str(tmp_path / "workspace"),
         "--keep"],
        client=gh,
        editor=editor,
    )

    assert code == EXIT_INFRASTRUCTURE
    assert BRANCH not in scratch_repo.remote_branches()
