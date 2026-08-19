"""The last step of a worker's life: push the branch, open one PR, hand it over.

`entrypoint.py` stops at a verified commit inside a container that is about to
be destroyed. This module gets that commit onto the remote and turns it into
the one artefact the rest of the system reads: **exactly one open pull request
per issue**, carrying `Closes #<n>` so the merge closes the issue. That pull
request is the whole handover - the orchestrator derives `review` from it
(`orchestrator/derived.py`), so there is nothing left for this module to
announce.

`publish(result, *, client=...)` is the whole surface, and its signature is
fixed by `entrypoint._publish`, which probes for this module with `find_spec`
and calls it with exactly that shape. Changing it here would leave the worker
silently never publishing - it would keep exiting 0 with the commit stranded in
a container - so the shape stays even where a different one might read better.

## The token never goes in the remote URL

`https://x-access-token:<token>@github.com/...` is the obvious way to push with
a credential and it is the one thing this module must not do. That URL is
written into `.git/config` in the container, echoed by `git remote -v`, and
pasted verbatim into every git error message - which is the stream #15 captures
and #29 writes to disk, where a scoped token in a run artefact you keep forever
is still a leak (#28).

So the credential reaches git two hops removed from anything that gets
recorded: `git -c credential.helper=<shell snippet>` on the command line, where
the snippet contains the *name* of an environment variable, and the value lives
only in that variable in the child process's environment. Nothing goes in the
URL, nothing goes in `.git/config`, and nothing goes in `argv` - `ps` inside
the container and `docker inspect` outside it both see the variable's name.
The leading empty `credential.helper=` resets any helper inherited from a
developer's own git config, so a laptop's keychain never answers instead.

Git errors are redacted on the way out regardless. Redaction at the boundary is
cheap and the alternative is discovering a gap in it by reading a log.

## Pushing is a force, with a lease

The branch is `apiary/<ref>-attempt-<n>` (`github/branches.py`), minted by
`entrypoint.run_worker` from the marker's own counter, so #144 removed the
common reason this push had to force at all: a retry is a *different* attempt
and gets a *different* branch, rather than recreating one name from the base
commit and diverging from what the last attempt left there.

The lease stays, and not out of caution about a case that can no longer happen.
`--force-with-lease` overwrites what *this* worker's clone last saw on the
remote and refuses when the branch moved since, which is the check that catches
the two things attempt-scoped names do not rule out: a human pushing a fix onto
the PR branch, and a second container dispatched for an attempt that already
has one. A bare `--force` would silently discard both, and a plain push would
fail without saying which of them happened.

## Opening once per branch, updating thereafter

**Once per branch, and since #144 that is once per attempt.** It used to be
once per issue: one name served every attempt, so a retry force-pushed over the
first attempt's head and updated the pull request already open on it. Attempt-
scoped names end that, and deliberately - a retry's history diverges from the
attempt before it, and quietly replacing the head of a pull request somebody may
be reading is a worse outcome than a second pull request they can see and close.
The convergence that is lost is narrow and named in `orchestrator/recovery.py`:
a *re-dispatch of the same attempt*, which only a blind recovery sweep produces.

What has not changed is that the same branch must not collect two pull requests,
which is what the rest of this section is about. Finding the one already open
means asking GitHub for the open PR whose head is this branch, and `GitHubClient`
has no method for that: its
pull-request surface is `get`, `create`, `update` and `merge` by number, and
#17's file contract does not include `client.py`. So this module **probes** for
`client.list_pull_requests` - the public method the client grows when the
reconciler (#23) needs to read PRs - and uses it when it is there. That is the
same shape, and the same reason, as `entrypoint._publish`'s probe for this
module and `Dockerfile.worker`'s shim for that one: two tickets whose file sets
cannot reach each other must not deadlock.

Until that method exists the fallback is still correct, because GitHub itself
enforces the invariant: `POST /pulls` for a head branch that already has an
open PR is a 422, never a second PR. The push has already moved that PR's head
to the new commit, so a re-publish onto the same branch updates the PR that
exists rather than opening another. What is lost is only the PR *body* - it
keeps the previous publish's verify output until the listing method lands.
Nothing about that failure mode is silent: it is printed, and
`Published.created` is False.

## No tracker call, of any kind

This module used to write one label, `swarm:review`, immediately after opening
the pull request - a tracker write issued from inside a container that is
running model-generated code, which is the one thing `docs/security.md` is most
careful about. #148 deleted it, and the argument is that it had stopped buying
anything: `orchestrator/derived.py` reads `review` off *an open pull request
whose head branch carries this task's ref*, so the pull request this function
just opened **is** the state the label was announcing. Writing it as well only
narrowed a window that no longer exists.

What that removes is a permission, not a call. The credential a worker holds no
longer needs `issues:write` (`security.WORKER_PERMISSIONS`), so a bad
generation that got hold of it cannot relabel, comment on or close a work item
in the tracker at all. The label itself has not gone anywhere - the orchestrator
writes `claimed -> review` when it observes the exit 0 this publish produces
(`reconcile._verdict`), from outside the container, on the credential that was
always going to keep `issues:write` until #152 removes the label plane.

So the only endpoints reachable from here are the code host's: `git push`, and
`/pulls`. `tests/test_worker_pr.py` asserts that absence rather than the
presence of the state, because a test that only checked that `review` still
appears would pass with the write still in place.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..github.client import GitHubClient, GitHubError, GitHubHTTPError
from .entrypoint import OUTPUT_TAIL_CHARS, WorkerResult

#: The environment variable the credential helper reads the token out of. Its
#: *name* is what lands in `argv`; its value never leaves the child process.
TOKEN_ENV = "APIARY_PUSH_TOKEN"

#: A git credential helper, as a one-line shell function. `$1` is the operation:
#: only `get` is answered, so `store` and `erase` are no-ops and nothing is ever
#: written to a credential store inside the container. The username is GitHub's
#: convention for token authentication over HTTPS and carries no information.
CREDENTIAL_HELPER = (
    '!f() { test "$1" = get && '
    f'printf "username=x-access-token\\npassword=%s\\n" "${TOKEN_ENV}"; }}; f'
)

#: Where the token comes from when the caller does not pass one - the same
#: variable `GitHubClient.from_env` reads, because it is the same credential.
TOKEN_ENV_SOURCE = "GITHUB_TOKEN"

#: Used only when a checkout cannot say what its remote's default branch is.
DEFAULT_BASE_BRANCH = "main"

_GIT_TIMEOUT_S = 300


class PublishError(GitHubError):
    """Publishing failed - the push, or the API calls around it.

    A subclass of `GitHubError` on purpose. `entrypoint.main` wraps its call to
    `_publish` in `except GitHubError`, and that is the clause that turns a
    failure here into exit 2: the work is verified and committed, so a remote
    that would not take it is infrastructure, and the attempt must not be
    consumed for it (`docs/issue-contract.md` §4). Any other exception type
    escapes `main` as a traceback and exits 1, which would burn the attempt.
    """


@dataclass(frozen=True)
class Published:
    """What one `publish` call left behind, in the shape a caller can log.

    `created` is the interesting field: the second run against the same issue
    must report False, and a run that keeps reporting True is opening a PR per
    attempt - the exact failure this module exists to prevent.
    """

    issue: int
    branch: str
    commit: str
    number: int | None = None
    url: str | None = None
    created: bool = False

    def summary(self) -> str:
        what = "opened" if self.created else "updated"
        where = f"#{self.number}" if self.number is not None else f"the PR for {self.branch}"
        return f"issue #{self.issue}: {what} {where} from {self.commit[:12]}"


# --------------------------------------------------------------------------
# git
# --------------------------------------------------------------------------


def _git(
    root: Path,
    *args: str,
    label: str | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> str:
    """Run git in `root`. `check=False` turns a failure into an empty string."""
    proc = subprocess.run(
        ["git", *args],
        cwd=str(root),
        env=None if env is None else {**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_S,
    )
    if proc.returncode != 0:
        if not check:
            return ""
        raise PublishError(f"git {label or args[0]} failed:\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def push_command(
    branch: str,
    *,
    remote: str = "origin",
    token: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    """The push, as `(argv, environment overlay)` - the one place it is built.

    Public and separate from running it because "the token is in the
    environment and not in `argv`" is the security claim of this module, and a
    claim is worth asserting from the outside rather than by reading this file.
    """
    config: list[str] = []
    env: dict[str, str] = {"GIT_TERMINAL_PROMPT": "0"}
    if token:
        # The empty value first clears helpers inherited from the host's git
        # config; the second appends ours to the now-empty list.
        config = ["-c", "credential.helper=", "-c", f"credential.helper={CREDENTIAL_HELPER}"]
        env[TOKEN_ENV] = token
    argv = [
        *config,
        "push",
        "--quiet",
        "--force-with-lease",
        remote,
        f"{branch}:refs/heads/{branch}",
    ]
    return argv, env


def push_branch(
    root: Path,
    branch: str,
    *,
    remote: str = "origin",
    token: str | None = None,
) -> None:
    """Push `branch` to `remote`, forcing only over what this clone last saw.

    `token` is optional because it is not always needed: the test suite pushes
    to a bare repository on disk, where git authenticates nothing. When it is
    given it reaches git through `CREDENTIAL_HELPER` and the environment, never
    through the remote URL - see the module docstring.
    """
    argv, env = push_command(branch, remote=remote, token=token)
    try:
        _git(root, *argv, label="push", env=env)
    except PublishError as exc:
        raise PublishError(redact(str(exc), token)) from None


def base_branch(root: Path, *, remote: str = "origin") -> str:
    """The branch a PR from this checkout should target.

    Read from the clone rather than from the API: `git clone` records the
    remote's HEAD in `refs/remotes/<remote>/HEAD`, so this is the branch the
    worker actually branched from, with no extra request and no assumption that
    the API's `default_branch` and the clone agree.
    """
    ref = _git(root, "symbolic-ref", "--short", f"refs/remotes/{remote}/HEAD", check=False)
    prefix = f"{remote}/"
    return ref[len(prefix):] if ref.startswith(prefix) else DEFAULT_BASE_BRANCH


def redact(text: str, token: str | None) -> str:
    """Replace the token with a marker wherever it appears."""
    return text.replace(token, "***") if token else text


# --------------------------------------------------------------------------
# The pull request
# --------------------------------------------------------------------------


def pull_request_title(result: WorkerResult) -> str:
    """The commit's own subject, which `entrypoint` already built from the issue.

    Reading it back beats rebuilding it: the PR and the commit then say the same
    thing, and `WorkerResult` does not carry the issue title.
    """
    subject = _git(result.root, "log", "-1", "--format=%s", check=False)
    return subject or f"swarm[{result.task_id}]: issue #{result.issue}"


def pull_request_body(result: WorkerResult) -> str:
    """What changed, what the gate was, and what the gate said.

    A reviewer's first three questions, and the only process that ever knew all
    three answers is the container that is about to be destroyed. `Closes
    #<n>` last and alone on its line: it is what makes the merge close the
    issue, which is how `swarm:done` is ever reached.
    """
    # Deletions are named as deletions: a removed file listed as a plain
    # change would send the reviewer looking for new contents that do not
    # exist, which is the opposite of what "What changed" is for.
    written = [
        *(f"- `{path}`" for path in result.written),
        *(f"- `{path}` (deleted)" for path in result.deleted),
    ] or ["- _nothing_"]
    lines = [
        f"Opened by an apiary worker for issue #{result.issue} (`{result.task_id}`).",
        "",
        "## What changed",
        "",
        *written,
        "",
        "## Verify",
        "",
        "The gate is the issue's own `## Verify` command, and only its exit code was",
        "believed. It passed on the commit below, before that commit existed as a commit.",
        "",
        *_fenced(result.verify_command),
        "",
        "<details><summary>Output tail</summary>",
        "",
        *_fenced(result.verify_output[-OUTPUT_TAIL_CHARS:]),
        "",
        "</details>",
    ]
    if result.refused:
        # A reviewer wants to know that the model reached outside its file set,
        # even though nothing was written: it is the clearest signal that the
        # issue's `## Files` and the work it asked for have drifted apart.
        lines += [
            "",
            "## Refused",
            "",
            "Edits the model proposed and the guard rail did not apply:",
            "",
            *(f"- `{path}` - {reason}" for path, reason in result.refused),
        ]
    lines += ["", f"Closes #{result.issue}"]
    return "\n".join(lines)


def _fenced(text: str) -> list[str]:
    """A code fence long enough to survive backticks inside `text`.

    Verify output is arbitrary: a failing doctest or a markdown linter prints
    backtick runs, and a three-backtick fence around them closes early and
    turns the rest of the body into prose.
    """
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [fence, text, fence]


def find_open_pull_request(client: GitHubClient, branch: str) -> dict[str, Any] | None:
    """The open PR whose head is `branch`, or None if there is none to find.

    Returns None both when no such PR exists and when the client cannot look -
    see the module docstring for why those two are one answer here, and what
    the caller does with it.
    """
    lister = getattr(client, "list_pull_requests", None)
    if lister is None:
        return None
    # Only `state` is passed, and the head is matched here: a future
    # `list_pull_requests` is certain to take `state` (every list method on the
    # client does) and may or may not take `head`, and guessing wrong would be
    # a TypeError in the retry path of a task that is otherwise finished.
    for payload in lister(state="open") or ():
        if (payload.get("head") or {}).get("ref") == branch:
            return payload
    return None


def _already_open(exc: GitHubHTTPError) -> bool:
    """Tell "a PR for this branch is already open" from every other 422.

    The other 422s are real failures - "No commits between main and the branch",
    a base branch that does not exist - and swallowing those would report a PR
    that was never opened.
    """
    return exc.status == 422 and "already exist" in str(exc).lower()


# --------------------------------------------------------------------------
# The handover
# --------------------------------------------------------------------------


def publish(
    result: WorkerResult,
    *,
    client: GitHubClient,
    base: str | None = None,
    token: str | None = None,
    remote: str = "origin",
) -> Published:
    """Push the verified commit and leave exactly one open PR behind.

    Called by `entrypoint._publish` as `publish(result, client=client)`; `base`,
    `token` and `remote` exist for the tests and for a human running a worker by
    hand. Raises `PublishError` - a `GitHubError` - if the push or the PR call
    fails, which is what makes the worker exit 2 rather than consume an attempt
    for a remote that would not take work that already passed its gate.
    """
    if not (result.passed and result.commit):
        # Not a failure mode, a caller bug: an unverified result has nothing to
        # push, and publishing one would put unverified work in front of a
        # reviewer with a PR body claiming it passed.
        raise ValueError(f"issue #{result.issue}: refusing to publish an unverified result")

    token = token if token is not None else os.environ.get(TOKEN_ENV_SOURCE)
    push_branch(result.root, result.branch, remote=remote, token=token)
    print(f"  · pushed {result.branch} to {remote}")

    target = base or base_branch(result.root, remote=remote)
    title = pull_request_title(result)
    body = pull_request_body(result)

    existing = find_open_pull_request(client, result.branch)
    if existing is not None:
        payload = client.update_pull_request(int(existing["number"]), title=title, body=body)
        created = False
    else:
        payload, created = _open(client, title=title, head=result.branch, base=target, body=body)

    published = Published(
        issue=result.issue,
        branch=result.branch,
        commit=result.commit,
        number=_int_or_none((payload or {}).get("number")),
        url=(payload or {}).get("html_url"),
        created=created,
    )
    print(f"» {published.summary()}")
    return published


def _open(
    client: GitHubClient,
    *,
    title: str,
    head: str,
    base: str,
    body: str,
) -> tuple[dict[str, Any] | None, bool]:
    """Create the PR, tolerating the one 422 that means "it is already there".

    `(payload, created)`. A `None` payload is the fallback path of the module
    docstring: the PR exists, this client could not find its number, and the
    push has already put the new commit on it.
    """
    try:
        return client.create_pull_request(title=title, head=head, base=base, body=body), True
    except GitHubHTTPError as exc:
        if not _already_open(exc):
            raise PublishError(str(exc)) from exc
        print(f"  · a PR for {head} is already open; its head now carries this commit,"
              " but its body still describes the previous attempt", file=sys.stderr)
        return None, False
    except GitHubError as exc:
        raise PublishError(str(exc)) from exc


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
