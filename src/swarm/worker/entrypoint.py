"""One issue, one container, one commit: the worker's whole life.

    apiary-worker --repo shahrestani-me/apiary --issue 16 --base-commit 64afc6c

Clone at the base commit, branch `swarm/issue-<n>`, read the issue's contract,
edit the files it declares, run **its** `## Verify` command, and commit only if
that command exits zero. `Dockerfile.worker`'s shim delegates here as soon as
this module exists, argv untouched, so `main(argv)` is the argument contract
the orchestrator dispatches against.

**The invariant, inherited whole from v1's
[`verifier.py`](../nodes/verifier.py): only the verify command's exit code
decides pass or fail.** No model judges the work - not this one, not a second
one, not "the diff looks reasonable". Generation is cheap and verification is
the bottleneck, and a swarm that lets a model mark its own homework produces
volume rather than software.

Two consequences of that are easy to miss:

- **The command comes from the issue, never from `Settings.verify_command`.**
  A repo-wide default would quietly pass a task whose real gate was a new test
  file that was never run - the task would be verified against the tests that
  already existed. `docs/issue-contract.md` §1.3 makes the command part of the
  contract for exactly this reason, and a missing `## Verify` is a parse
  failure rather than an invitation to substitute one.
- **Verification runs before the commit, and the commit stages only the
  declared paths.** The verify command is arbitrary shell chosen by the target
  repository; it installs dependencies, writes caches, and may drop files
  anywhere in the tree. `git add -A` would sweep all of that into the PR, so
  the commit names the files the guard rail let through and nothing else.

## Exit codes are the protocol

`docs/issue-contract.md` §4, and the orchestrator reads them:

- `0` - the work is done and pushed. Today that means verified and committed;
  the push and the PR are #17, which this module calls into once it lands.
- `1` - the task failed. The attempt is consumed, and #22 either re-readies the
  issue or gives up on it.
- `2` - infrastructure failed, so the task never really ran and the attempt is
  **not** consumed: a dead Ollama, a clone that would not clone, GitHub
  refusing the read. #18's point is that a broken host would otherwise burn
  every issue's retry budget before a human noticed.

A malformed contract is a `1`, not a `2`, even though nothing about the task
was attempted: the body will parse exactly as badly on the next attempt, and
handing it back as retryable infrastructure would loop forever.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..config import SETTINGS
from ..github.client import GitHubClient, GitHubError
from ..github.ledger import ContractError, TaskContract, parse_contract
# GitError says "a git command failed" and means it in both v1's worktrees and
# v2's clones; a second error class for the same fact would only make callers
# catch both.
from ..run import RUN_ID_ENV
from ..worktree import GitError
from .edit import Applied, EditError, apply_edits, gather_context, propose_edits, read_writable

EXIT_OK = 0
EXIT_TASK_FAILED = 1
EXIT_INFRASTRUCTURE = 2

#: Where the container clones to. `Dockerfile.worker` makes /workspace the
#: workdir and the only directory the unprivileged user owns.
DEFAULT_WORKSPACE = "/workspace"

#: The shell's own codes for "I could not run that": 127 is command not found,
#: 126 is found but not executable. Both mean the gate never opened, so neither
#: is evidence about the work - see `run_verify`. Deliberately narrow: a suite
#: that runs and exits 2, or 5, has genuinely failed and is the task's problem.
UNRUNNABLE_EXIT_CODES = frozenset({126, 127})

#: How much of the verify output travels onward. #17 puts it in the PR body and
#: #29 writes it to disk, and neither wants a megabyte of pytest chatter.
OUTPUT_TAIL_CHARS = 4_000

_SLUG_RE = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")
_REMOTE_RE = re.compile(r"[:/](?P<owner>[A-Za-z0-9_.\-]+)/(?P<name>[A-Za-z0-9_.\-]+?)(?:\.git)?/?$")


class InfrastructureError(RuntimeError):
    """Something outside the task broke. Exit 2, attempt not consumed."""


@dataclass(frozen=True)
class WorkerResult:
    """Everything one worker run learned, in one object.

    Public because it is the handover to #17: the PR body has to state what
    changed, which command was the gate and what that command said, and the
    only process that ever knew those three things is this one.
    """

    issue: int
    repo: str
    task_id: str
    branch: str
    root: Path
    verify_command: str
    verify_output: str
    passed: bool
    commit: str | None = None
    written: tuple[str, ...] = ()
    refused: tuple[tuple[str, str], ...] = ()
    #: The attempt this run is, taken from the contract's identity marker. The
    #: worker is the only process that reads that marker before doing the work,
    #: so carrying it here is cheaper than threading a counter through
    #: `spawn` - and it keeps `result.py`'s one-file-per-attempt naming honest
    #: when a retry lands.
    attempt: int = 0

    @property
    def exit_code(self) -> int:
        return EXIT_OK if self.passed and self.commit else EXIT_TASK_FAILED

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        files = ", ".join(self.written) or "no files"
        return f"issue #{self.issue}: {verdict} [{self.verify_command}] on {files}"


# --------------------------------------------------------------------------
# git
# --------------------------------------------------------------------------


def _git(cwd: Path, *args: str, timeout: int = 300) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed:\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def prepare_checkout(clone_url: str, dest: Path, base_commit: str, branch: str) -> Path:
    """Clone, then put `branch` on `base_commit`. Returns the checkout root.

    The base commit is the orchestrator's, not `HEAD`: between planning and
    dispatch another task's PR may have merged, and a worker that branched from
    whatever main happened to be would verify against a tree nobody planned
    for.

    `--force-create` because this is also the retry path. A previous attempt
    pushed `swarm/issue-<n>`, the clone fetched it, and a fresh attempt starts
    from the base commit again rather than compounding the last failure - #17
    updates the existing PR instead of opening a second one.
    """
    dest = dest.resolve()
    if dest.exists() and any(dest.iterdir()):
        raise InfrastructureError(f"{dest} already exists and is not empty")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Cloning with a token in the URL would write it into .git/config and into
    # every git error message, which is the stream #15 captures and #29 saves.
    # Authenticated clones land with #17/#28's credential handling; the URL
    # stays a plain URL here.
    _git(dest.parent, "clone", "--quiet", clone_url, str(dest))
    _git(dest, "switch", "--quiet", "--force-create", branch, base_commit)
    return dest


def commit_edits(root: Path, message: str, paths: Sequence[str]) -> str | None:
    """Stage exactly `paths` and commit. `None` when they changed nothing.

    `--force` because the contract, not `.gitignore`, decides what this task
    is: a declared path that happens to be ignored is a file the planner asked
    for, and failing the whole attempt over it would be an obscure way to say
    so. Nothing outside the declared paths can reach this call.

    Identity comes from the environment - `Dockerfile.worker` sets
    `GIT_AUTHOR_*` and `GIT_COMMITTER_*` so the orchestrator can override them
    per run. Hard-coding `-c user.name=` here would overrule that from the one
    place a run cannot reach.
    """
    _git(root, "add", "--force", "--", *paths)
    if not _git(root, "status", "--porcelain", "--", *paths):
        return None
    _git(root, "commit", "--quiet", "-m", message)
    return _git(root, "rev-parse", "HEAD")


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def run_verify(root: Path, command: str) -> tuple[bool, str]:
    """Run the contract's command from the repository root. Believe the exit code.

    `shell=True` because `## Verify` is a shell command by definition
    (`docs/issue-contract.md` §1.3) - `pip install -e . && pytest -q` is a
    normal one. That is also why this only ever runs inside the container: it
    is the target repository's code, executed.

    A timeout is a failure, not an error. A suite that hangs is as unmergeable
    as one that fails, and calling it infrastructure would hand back an
    unconsumed attempt to a task that can hang again for free.

    **A command that could not be executed is a different thing entirely**, and
    raises `InfrastructureError` rather than returning a failure. The shell
    reports 127 for "command not found" and 126 for "found but not
    executable", and neither says anything about the code the model wrote. A
    target repository whose `## Verify` names a tool the image lacks - or
    simply says `python` on a host that only has `python3` - would otherwise
    fail identically on every attempt, exhaust its budget, land on
    `swarm:failed`, and feed the model three rounds of "command not found" as
    though it were a code review. That is exactly the case
    `docs/issue-contract.md` §4 separates exit 1 from exit 2 to prevent, and it
    is invisible inside the container, where the command usually does exist.
    """
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=SETTINGS.verify_timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"verify timed out after {SETTINGS.verify_timeout_s}s"
    except OSError as exc:
        # The shell itself could not be started. Nothing ran.
        raise InfrastructureError(f"could not run the verify command: {exc}") from exc

    output = f"{proc.stdout}\n{proc.stderr}".strip()
    if proc.returncode in UNRUNNABLE_EXIT_CODES:
        raise InfrastructureError(
            f"the verify command could not be run (exit {proc.returncode}): "
            f"{command!r}\n{output[-OUTPUT_TAIL_CHARS:]}"
        )
    return proc.returncode == 0, output[-OUTPUT_TAIL_CHARS:]


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def _fetch_contract(client: GitHubClient, issue: int) -> tuple[TaskContract, str]:
    try:
        payload = client.get_issue(issue)
    except GitHubError as exc:
        raise InfrastructureError(f"reading issue #{issue}: {exc}") from exc
    return parse_contract(issue, payload.get("body")), payload.get("title") or ""


def run_worker(
    *,
    repo: str,
    issue: int,
    base_commit: str,
    workspace: Path,
    clone_url: str,
    client: GitHubClient,
    editor=None,
) -> WorkerResult:
    """The whole task, start to finish.

    A failed task is a returned `WorkerResult`, never an exception: it is an
    ordinary outcome the caller reports. Exceptions are reserved for the two
    things that are not the task's doing - `ContractError` for a body that does
    not parse, and `InfrastructureError`/`EditError`/`GitError` for a host that
    could not run it.

    `editor` is the test seam that keeps this suite hermetic: anything with
    `.invoke(messages)` returning a `WorkerOutput`. Passing nothing reaches the
    host's Ollama over `host.docker.internal`, which is why the one test that
    does so carries the `ollama` marker.
    """
    contract, title = _fetch_contract(client, issue)
    branch = f"swarm/issue-{issue}"
    task_id = contract.task_id or f"issue-{issue}"

    root = prepare_checkout(clone_url, workspace / f"issue-{issue}", base_commit, branch)

    writable = read_writable(root, contract.files)
    readable = gather_context(root, contract.files)
    print(f"  · {len(writable)} file(s) to edit, {len(readable)} for context")

    output = propose_edits(contract.goal, writable, readable, llm=editor)
    applied: Applied = apply_edits(root, output.edits, contract.files)
    for path, reason in applied.refused:
        print(f"  ! refused {path}: {reason}", file=sys.stderr)

    if not applied.written:
        # No commit and no verification: there is nothing to verify, and
        # running the command anyway would report the repository's existing
        # state as this task's result.
        return WorkerResult(
            issue=issue,
            repo=repo,
            task_id=task_id,
            branch=branch,
            root=root,
            verify_command=contract.verify,
            verify_output="the model produced no edit inside the declared file set",
            passed=False,
            refused=applied.refused,
            attempt=contract.attempt,
        )

    print(f"  · wrote {', '.join(applied.written)}")
    print(f"  · verifying: {contract.verify}")
    passed, verify_output = run_verify(root, contract.verify)

    commit = None
    if passed:
        subject = f"swarm[{task_id}]: {(title or contract.goal)[:60]}"
        try:
            commit = commit_edits(root, subject, applied.written)
        except GitError as exc:
            raise InfrastructureError(f"committing issue #{issue}: {exc}") from exc

    return WorkerResult(
        issue=issue,
        repo=repo,
        task_id=task_id,
        branch=branch,
        root=root,
        verify_command=contract.verify,
        verify_output=verify_output,
        passed=passed,
        commit=commit,
        written=applied.written,
        refused=applied.refused,
        attempt=contract.attempt,
    )


def _publish(result: WorkerResult, client: GitHubClient) -> None:
    """Hand a verified run to the push-and-PR step, if that step exists yet.

    The same shape, and the same reason, as the shim in `Dockerfile.worker`:
    #17's contract lists `src/swarm/worker/pr.py` and nothing else, so it
    cannot edit this file to wire itself in, and a call to a module that does
    not exist yet would deadlock the two tickets against each other. #17 owns
    the signature `publish(result: WorkerResult, *, client: GitHubClient)`.

    `find_spec` rather than a try/except around the import, again for the
    Dockerfile's reason: once the module is there, an ImportError raised inside
    it must crash rather than silently downgrade the worker to "verified but
    never pushed", which reads as success in the exit code.
    """
    try:
        found = importlib.util.find_spec("swarm.worker.pr") is not None
    except ModuleNotFoundError:
        found = False
    if not found:
        print("! push and PR are not wired yet (#17); the commit stays in the container",
              file=sys.stderr)
        return
    from .pr import publish

    publish(result, client=client)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apiary-worker",
        description="Run one issue to a verified commit, in an isolated container.",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="target repository, as owner/name or as a clone URL",
    )
    parser.add_argument(
        "--issue", type=int, required=True, help="issue number holding the task contract"
    )
    parser.add_argument("--base-commit", required=True, help="commit the work branch starts from")
    parser.add_argument(
        "--clone-url",
        default=None,
        help="where to clone from, if that is not derivable from --repo",
    )
    parser.add_argument(
        "--workspace",
        default=DEFAULT_WORKSPACE,
        help=f"directory to clone into (default: {DEFAULT_WORKSPACE})",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="leave the checkout behind on failure, for debugging outside a container",
    )
    return parser


def split_repo(value: str) -> tuple[str | None, str]:
    """`--repo` in, `(slug, clone_url)` out.

    Both spellings are accepted because both callers are real: the orchestrator
    knows the repository as `owner/name` (it is what `GitHubClient` takes), and
    a human debugging a worker points it at a path or a URL. A path yields no
    slug, and the API calls then need `--repo`-shaped input or an injected
    client - so this returns `None` rather than guessing an owner.
    """
    value = value.strip()
    if _SLUG_RE.match(value) and "://" not in value:
        return value, f"https://github.com/{value}.git"
    match = _REMOTE_RE.search(value)
    if match and ("://" in value or value.startswith("git@")):
        return f"{match['owner']}/{match['name']}", value
    return None, value


def _record(result: WorkerResult, exit_code: int) -> None:
    """Write this attempt's result file. Never raises.

    The orchestrator reads exit codes from these records rather than blocking
    on `docker wait`, which is what lets it restart mid-run - so a worker that
    finishes without writing one is indistinguishable from a worker that
    crashed. It is best-effort by design: an unwritable artifacts directory is
    a reason to lose the record, never a reason to change the exit code the
    task actually earned.
    """
    # Imported here, not at module scope: `result.py` depends on this module
    # for `WorkerResult` and the exit codes, so the top-level import would be a
    # cycle. The dependency points that way on purpose - the record is built
    # from the worker's own vocabulary.
    from .result import report as write_worker_result

    try:
        path = write_worker_result(
            result,
            run_id=os.environ.get(RUN_ID_ENV, "unattached"),
            attempt=result.attempt,
            exit_code=exit_code,
        )
    except OSError as exc:
        print(f"! could not write the result record: {exc}", file=sys.stderr)
    else:
        print(f"  · recorded {path.name}")


def _unrun(issue: int, repo: str, reason: str) -> WorkerResult:
    """A result for an attempt that never got as far as a verify command."""
    return WorkerResult(
        issue=issue,
        repo=repo,
        task_id="",
        branch=f"swarm/issue-{issue}",
        root=Path("."),
        verify_command="",
        verify_output=reason,
        passed=False,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    client: GitHubClient | None = None,
    editor=None,
) -> int:
    """`client` and `editor` are the test seams, as `cli.main`'s `client` is."""
    args = build_parser().parse_args(argv)
    slug, clone_url = split_repo(args.repo)
    slug = slug or (client.repo if client is not None else None)
    if slug is None:
        print(f"! cannot tell the repository from {args.repo!r}; pass --repo owner/name",
              file=sys.stderr)
        return EXIT_INFRASTRUCTURE

    print(f"» issue #{args.issue} of {slug} at {args.base_commit[:12]}")
    try:
        result = run_worker(
            repo=slug,
            issue=args.issue,
            base_commit=args.base_commit,
            workspace=Path(args.workspace),
            clone_url=args.clone_url or clone_url,
            client=client if client is not None else GitHubClient.from_env(slug),
            editor=editor,
        )
    except ContractError as exc:
        # Task failure, not infrastructure: this body will not parse any better
        # on a second attempt (see the module docstring).
        print(f"! {exc}", file=sys.stderr)
        _record(_unrun(args.issue, slug, str(exc)), EXIT_TASK_FAILED)
        return EXIT_TASK_FAILED
    except (InfrastructureError, EditError, GitError, GitHubError, OSError) as exc:
        print(f"! {exc}", file=sys.stderr)
        # Exit 2 is the code the reconciler must see to leave the attempt
        # budget alone, and it can only see it in a record.
        _record(_unrun(args.issue, slug, str(exc)), EXIT_INFRASTRUCTURE)
        return EXIT_INFRASTRUCTURE

    print(f"» {result.summary()}")
    if not result.passed:
        print(result.verify_output, file=sys.stderr)
    elif result.commit is None:
        # Verified, but the edits were byte-identical to what was already
        # there. Nothing to push and nothing to review, so it is a failed
        # attempt rather than a silent success on somebody else's work.
        print("! the edits changed nothing; there is no commit to open a PR from",
              file=sys.stderr)
    else:
        print(f"» committed {result.commit[:12]} on {result.branch}")
        try:
            _publish(result, client if client is not None else GitHubClient.from_env(slug))
        except GitHubError as exc:
            print(f"! {exc}", file=sys.stderr)
            _record(result, EXIT_INFRASTRUCTURE)
            return EXIT_INFRASTRUCTURE

    _record(result, result.exit_code)

    if result.exit_code != EXIT_OK and not args.keep:
        # The container is cattle and its filesystem dies with it, so this only
        # matters when a human is running the worker on a laptop - where the
        # leftover checkout would collide with the next attempt.
        shutil.rmtree(result.root, ignore_errors=True)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
