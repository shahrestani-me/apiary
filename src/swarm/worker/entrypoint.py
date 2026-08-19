"""One issue, one container, one commit: the worker's whole life.

    apiary-worker --repo shahrestani-me/apiary --issue 16 --base-commit 64afc6c

Clone at the base commit, branch `apiary/<ref>-attempt-<n>`, read the contract,
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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..config import SETTINGS
from ..containers.manager import SECRET_NAME_RE
from ..github.branches import task_branch
from ..github.client import GitHubClient, GitHubError
from ..github.ledger import (
    ContractError,
    TaskContract,
    generated_for,
    parse_contract,
)
from ..github.refs import task_ref
# GitError says "a git command failed" and means it in both v1's worktrees and
# v2's clones; a second error class for the same fact would only make callers
# catch both.
from ..run import RUN_ID_ENV
from ..security import EGRESS_EXTRA_ENV
from ..worktree import GitError
from .edit import (
    Applied,
    EditError,
    apply_edits,
    fit_context,
    gather_context,
    propose_edits,
    read_writable,
    syntax_failure,
)

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

#: 128 + SIGKILL. On this host that is almost always the out-of-memory killer:
#: the Docker Desktop VM has 7.65 GiB and two workers at `--memory 4g` overcommit
#: it. Python used ~100MB so nobody ever saw it; a toolchain will.
#:
#: It is also what a `docker stop` produces, which is the same conclusion - the
#: gate did not reach a verdict, so there is no verdict to charge the task for.
OOM_EXIT_CODE = 137

#: Signatures that mean the command could not reach something it needed, on a
#: run where reaching it was never going to work. A worker sits on an
#: `internal: true` network whose only route out is the egress proxy, and the
#: enforced allowlist is a static block in `compose.yaml` - so a `## Verify`
#: that installs anything is denied, every attempt, identically.
#:
#: Measured through the real `apiary-egress` tinyproxy: a denied `npm install`
#: exits **1** in under a second with `npm error code E403 ... 403 Filtered`.
#: `UNRUNNABLE_EXIT_CODES` cannot see that - it is a *shell*-level signal, and
#: every modern toolchain catches its own errors and normalises to exit 1. So
#: commit c015e4f's fix ("a verify command that cannot run is infrastructure,
#: not a failed task") held for 127 and reopened for every non-Python stack:
#: three attempts burned in ~3 seconds, then `swarm:failed`.
#:
#: **Narrow on purpose, and the default is the task's fault.** Everything not
#: matched here is `TASK_FAILED`, because the cost of being wrong is not
#: symmetric: a task failure misread as infrastructure never consumes an
#: attempt and so retries forever, which is why #91 exists as the backstop.
DENIED_EGRESS_SIGNATURES: tuple[str, ...] = (
    # tinyproxy's own refusal body, and the code npm/yarn/pip surface it as.
    "403 filtered",
    "code e403",
    # No default route and no resolver, which is what the containment looks
    # like from inside when a command tries to leave.
    "temporary failure in name resolution",
    "could not resolve host",
    "could not resolve proxy",
    "proxy connect aborted",
    "tunnel connection failed",
)

#: The verdicts `classify_verify` returns. Strings for the reason the check
#: statuses in `orchestrator/checks.py` are strings: they are printed, logged
#: and asserted on far more often than they are matched.
PASSED = "passed"
TASK_FAILED = "task_failed"
INFRASTRUCTURE = "infrastructure"

#: Environment names the verify subprocess must never see. The *shape* test is
#: `containers.manager.SECRET_NAME_RE`, reused rather than restated - it is the
#: same question ("does this name hold a credential?") that `Redactor` already
#: answers, and a second opinion here would drift from the one that does the
#: redacting. The two literals are the ones this system actually sets, listed
#: so that grepping for either lands here even though the regex already covers
#: both. `pr.TOKEN_ENV` is that second name and is spelled out rather than
#: imported: `pr.py` imports *this* module, so the dependency only runs one way.
VERIFY_ENV_DENY = ("GITHUB_TOKEN", "APIARY_PUSH_TOKEN")

#: Names that survive whatever else happens. Not an allow-list - see
#: `verify_env` - but the set whose absence breaks the run rather than
#: degrading it, and the set the regression test names one by one.
VERIFY_ENV_REQUIRED = (
    "PATH",
    "HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "NO_PROXY",
    "no_proxy",
)

#: How much of the verify output travels onward. #17 puts it in the PR body and
#: #29 writes it to disk, and neither wants a megabyte of pytest chatter.
OUTPUT_TAIL_CHARS = 4_000

#: The dependency manifests a worker may write whether or not its contract
#: lists them. Declaring what its code needs is part of any implementation
#: task, and a planner cannot foresee every import - the alternative, observed
#: live, is a worker that writes `import sqlalchemy` into a container that has
#: no sqlalchemy and no permitted way to say so. Python-only for now; a node
#: manifest is one entry here plus an installer in `install_dependencies`, not
#: a second rule.
DEPENDENCY_MANIFESTS: tuple[str, ...] = ("requirements.txt",)
PYTHON_MANIFEST = "requirements.txt"

#: What marks a `## Verify` command as pytest-based, and therefore auditable
#: by `audit_collection`. A bare substring test, deliberately dumb: the
#: commands the swarm generates are `python -m pytest -q` and close variants,
#: and a cleverer parse would only add ways to miss one. A command this
#: matches by accident (`./run-pytest-like-thing`) costs one fast
#: `--collect-only` probe; a pytest command it missed would cost the audit
#: entirely, so the test errs broad.
PYTEST_MARKER = "pytest"

#: The filename shapes pytest collects by default (`test_*.py` / `*_test.py`).
#: Used to pick which of an attempt's written files the collection audit must
#: prove were seen; a helper or fixture file is legitimately never collected.
TEST_FILE_RE = re.compile(r"(?:^|/)(?:test_[^/]*\.py|[^/]*_test\.py)$")

#: The collection audit's own clock. A `--collect-only` imports files without
#: running tests, so it is fast when healthy - and a hang here (an import that
#: blocks, say) must not spend the container budget the way the verify's own
#: timeout already guards against.
AUDIT_TIMEOUT_S = 120

#: The install gets its own clock rather than sharing the verify command's:
#: a resolver walking a heavy dependency tree is slow in a way a test suite is
#: not, and a hung index - the proxy denying silently - must not be allowed to
#: spend the whole container budget before the failure text exists.
INSTALL_TIMEOUT_S = 300

#: The first line of a retry-feedback comment, exactly as
#: `orchestrator.reconcile.retry_comment` writes it. Matched by prefix rather
#: than parsed, because the comment is prose for humans too and the only
#: machine-read fact is "this is the reconciler describing a failed attempt".
FEEDBACK_PREFIX = "apiary: attempt"

#: How much of that comment is folded into the goal. The comment is already
#: bounded at the writing end; this is the belt to that brace, because the goal
#: shares a 16K-token window with every file the task may edit.
FEEDBACK_MAX_CHARS = 3_000

#: The image this container was created from, if the spawner said so. A worker
#: cannot ask the daemon - it has no socket and no `docker` binary, which is the
#: containment working - so the only way it can name its own image is to be
#: told. The orchestrator-side record (`result.synthesise`) reads it off the
#: `Handle` instead and needs no variable; this is the seam #99 fills so the
#: worker's own testimony can answer the same question.
IMAGE_ENV = "APIARY_WORKER_IMAGE"

#: Which attempt this container is, as the orchestrator counted it.
#:
#: The counter used to reach here one way only: out of the issue body's identity
#: marker (`docs/issue-contract.md` §2). That made a customer's issue body the
#: transport for one of apiary's own numbers, which is what ADR 0001 says a
#: tracker is not for, and #152 removes the write that maintained it. The
#: reconciler holds the counter at the moment it dispatches - it is in apiary's
#: store since #159 - so it now tells the container, the same way it tells it
#: which image it is (`IMAGE_ENV`).
#:
#: `containers/manager.ATTEMPT_ENV` is the other half, and
#: `test_the_attempt_variable_matches_the_workers_own` pins the two spellings
#: together. Nothing is imported across that boundary: this module runs *inside*
#: the container and imports nothing from the orchestrator's packages.
ATTEMPT_ENV = "APIARY_ATTEMPT"

_SLUG_RE = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")
_REMOTE_RE = re.compile(r"[:/](?P<owner>[A-Za-z0-9_.\-]+)/(?P<name>[A-Za-z0-9_.\-]+?)(?:\.git)?/?$")


class InfrastructureError(RuntimeError):
    """Something outside the task broke. Exit 2, attempt not consumed.

    Carries what the attempt had already established when it died, because the
    record built from it is written on exactly the path where that matters
    most. "The clone failed" and "the gate passed and the push failed" are both
    exit 2, and only one of them has a command and a file list to report.
    `deleted` rides along with `written` for the same reason: a cleanup task
    that removed its files and then hit a broken gate did real work, and a
    record listing nothing would read as an attempt that never started.

    `attempt` is which attempt this failure killed, or `None` when the failure
    happened before the contract - the only place the number is written down -
    could be read. `run_worker` stamps it the moment the contract is in hand;
    it is state on the exception rather than an argument threaded through every
    raise site, because most raisers (a git subprocess, the model transport)
    have no idea the concept exists.
    """

    def __init__(
        self,
        message: str,
        *,
        verify_command: str = "",
        written: Sequence[str] = (),
        deleted: Sequence[str] = (),
        task_id: str = "",
    ) -> None:
        super().__init__(message)
        self.verify_command = verify_command
        self.written = tuple(written)
        self.deleted = tuple(deleted)
        self.task_id = task_id
        self.attempt: int | None = None

    def learned(
        self,
        *,
        verify_command: str,
        written: Sequence[str],
        task_id: str,
        deleted: Sequence[str] = (),
    ) -> None:
        """Fill in context the raiser did not have. Mutates and does not re-raise.

        `run_verify` cannot know which files were written and `commit_edits`
        cannot know the gate command, so the context is attached by the frame
        that has both rather than threaded through every raise site.
        """
        self.verify_command = self.verify_command or verify_command
        self.written = self.written or tuple(written)
        self.deleted = self.deleted or tuple(deleted)
        self.task_id = self.task_id or task_id


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
    #: Files this attempt removed (an empty-content edit - see
    #: `edit.apply_edits`). Kept apart from `written` so the record and the PR
    #: body can say "deleted" where a "write" would tell the reviewer the
    #: opposite of what happened.
    deleted: tuple[str, ...] = ()
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
        touched = [*self.written, *(f"{path} (deleted)" for path in self.deleted)]
        files = ", ".join(touched) or "no files"
        return f"issue #{self.issue}: {verdict} [{self.verify_command}] on {files}"


# --------------------------------------------------------------------------
# git
# --------------------------------------------------------------------------


def _git(cwd: Path, *args: str, timeout: int = 300, env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, **env} if env else None,
    )
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed:\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def _credentials() -> tuple[list[str], dict[str, str]]:
    """Git config and environment that authenticate a clone, or empty pair.

    Imported from `pr.py` rather than restated: one credential helper, one
    variable name, one place to get the quoting wrong. `GIT_TERMINAL_PROMPT=0`
    regardless, so an unauthenticated private clone fails immediately instead
    of blocking on a prompt no container can answer.
    """
    from .pr import CREDENTIAL_HELPER, TOKEN_ENV, TOKEN_ENV_SOURCE

    env = {"GIT_TERMINAL_PROMPT": "0"}
    token = os.environ.get(TOKEN_ENV_SOURCE)
    if not token:
        return [], env
    env[TOKEN_ENV] = token
    return ["-c", "credential.helper=", "-c", f"credential.helper={CREDENTIAL_HELPER}"], env


def prepare_checkout(clone_url: str, dest: Path, base_commit: str, branch: str) -> Path:
    """Clone, then put `branch` on `base_commit`. Returns the checkout root.

    The base commit is the orchestrator's, not `HEAD`: between planning and
    dispatch another task's PR may have merged, and a worker that branched from
    whatever main happened to be would verify against a tree nobody planned
    for.

    `--force-create` because this is also the retry path. Since #144 a retry
    is a *different* branch - the attempt is in the name - so the force is no
    longer what stops a fresh attempt compounding the last one; branching from
    the base commit is. It stays because the clone fetched every previous
    attempt's branch and this is still the call that decides where HEAD lands,
    and because a worker re-run by hand at the same attempt must start over
    rather than resume.
    """
    dest = dest.resolve()
    if dest.exists() and any(dest.iterdir()):
        raise InfrastructureError(f"{dest} already exists and is not empty")
    dest.parent.mkdir(parents=True, exist_ok=True)

    # A private repository needs credentials to *clone*, not only to push, and
    # a worker with none fails as `could not read Username for
    # 'https://github.com'` - which reads like a missing terminal rather than a
    # missing token. Same credential helper #17 uses for the push, for the same
    # reason: the token reaches git through an environment variable named on
    # the command line, never through the URL, so it stays out of .git/config
    # and out of every git error string.
    config, git_env = _credentials()
    # Cloning with a token in the URL would write it into .git/config and into
    # every git error message, which is the stream #15 captures and #29 saves.
    # Authenticated clones land with #17/#28's credential handling; the URL
    # stays a plain URL here.
    _git(dest.parent, *config, "clone", "--quiet", clone_url, str(dest), env=git_env)
    _git(dest, "switch", "--quiet", "--force-create", branch, base_commit)
    return dest


def stageable(root: Path, generated: Sequence[str]) -> tuple[str, ...]:
    """Which of a stack's generated paths this run actually produced.

    Absent is normal, not an error: `GENERATED_FILES` names what a stack's gate
    *may* write, and most tasks add no dependency and generate no lockfile.
    Failing a task because a file it never needed did not appear would make the
    generated set a requirement rather than a permission.

    Resolved and re-checked against the root even though the constants are the
    repository's own, because this is the function that turns a name into a
    `git add` and a symlink is a name that points somewhere else.
    """
    root = root.resolve()
    found: list[str] = []
    for relative in generated:
        target = (root / relative).resolve()
        if not target.is_file():
            continue
        if not target.is_relative_to(root):
            continue
        found.append(relative)
    return tuple(found)


def commit_edits(
    root: Path,
    message: str,
    paths: Sequence[str],
    generated: Sequence[str] = (),
) -> str | None:
    """Stage `paths` plus whatever `generated` the gate produced, and commit.

    `--force` because the contract, not `.gitignore`, decides what this task
    is: a declared path that happens to be ignored is a file the planner asked
    for, and failing the whole attempt over it would be an obscure way to say
    so. **`.gitignore` is also exactly why `--force` matters for the generated
    set**: `package-lock.json` is not ignored by any sane project, but a
    generated file that some template ignored would otherwise be silently
    dropped and take CI with it.

    `generated` is the third category between "the task's files" and
    "everything else" - see `ledger.GENERATED_FILES`. It stays a separate
    argument rather than being folded into `paths` so this function's staging
    rule is still readable as "exactly what it was given", and so the two sets
    can be reported apart in the record.

    Nothing outside the declared paths and the stack's generated set can reach
    this call; `git add -A` after a verify run would sweep `node_modules` and
    every cache the command wrote into the pull request.

    A path in `paths` may name a file `apply_edits` *deleted* as well as one it
    wrote: since git 2.0, `git add --force -- <path>` stages the removal of a
    tracked file that is absent from the working tree, so deletions ride the
    same staging rule with no second command - which is why the caller passes
    written and deleted paths together rather than this function growing a
    `git rm` branch that would need its own missing-file semantics.

    Identity comes from the environment - `Dockerfile.worker` sets
    `GIT_AUTHOR_*` and `GIT_COMMITTER_*` so the orchestrator can override them
    per run. Hard-coding `-c user.name=` here would overrule that from the one
    place a run cannot reach.
    """
    staged = [*paths, *stageable(root, generated)]
    _git(root, "add", "--force", "--", *staged)
    if not _git(root, "status", "--porcelain", "--", *staged):
        return None
    _git(root, "commit", "--quiet", "-m", message)
    return _git(root, "rev-parse", "HEAD")


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """What one verify run means: `PASSED`, `TASK_FAILED` or `INFRASTRUCTURE`.

    A verdict carries its reason because the reason is the whole product here.
    A container killed at its outer cap and a suite that genuinely failed both
    arrive as "the attempt did not work"; which of the two it was decides
    whether an attempt is consumed, and an operator reading the run afterwards
    has nothing else to go on.
    """

    outcome: str
    reason: str = ""

    @property
    def passed(self) -> bool:
        return self.outcome == PASSED

    @property
    def infrastructure(self) -> bool:
        return self.outcome == INFRASTRUCTURE


def classify_verify(
    returncode: int, stdout: str, stderr: str, duration: float = 0.0
) -> Verdict:
    """Decide what a finished verify command means. Pure, and that is the point.

    Purity is what puts this truth table in bare CI with no subprocess, no
    Docker and no model. The previous version of this decision lived inline in
    `run_verify` around a `subprocess.run`, which is why the only rule it ever
    grew was the one a shell hands you for free.

    **Fail closed.** Only the cases below are infrastructure; everything else
    is the task's fault, including an exit code nobody here has seen. The two
    errors are not symmetric - a task failure misread as infrastructure never
    consumes an attempt, so it retries forever, and #91 exists precisely
    because even the correct classification can loop.

    `duration` is recorded, not decided on. A denied `npm install` returns in
    under a second and a real suite does not, which is tempting - and it would
    mean a fast machine and a slow one classifying the same failure
    differently. It goes in the reason, where a human can weigh it.
    """
    if returncode == 0:
        return Verdict(PASSED, "the verify command passed")

    if returncode in UNRUNNABLE_EXIT_CODES:
        return Verdict(
            INFRASTRUCTURE,
            f"the verify command could not be run (exit {returncode}); the shell "
            "reports 127 for a command that is not there and 126 for one it may "
            "not execute",
        )

    if returncode == OOM_EXIT_CODE:
        return Verdict(
            INFRASTRUCTURE,
            f"the verify command was killed (exit {returncode}), which on this host "
            "is the out-of-memory killer: the Docker VM has less memory than the "
            "configured workers add up to",
        )

    haystack = f"{stdout}\n{stderr}".casefold()
    for signature in DENIED_EGRESS_SIGNATURES:
        if signature in haystack:
            return Verdict(
                INFRASTRUCTURE,
                f"the verify command was denied the network (exit {returncode} after "
                f"{duration:.1f}s, matching {signature!r}); a worker reaches nothing "
                "but the egress proxy's allowlist, so this fails identically every "
                "attempt",
            )

    return Verdict(TASK_FAILED, f"the verify command failed (exit {returncode})")


def verify_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment the verify command gets: this one, minus the credentials.

    **Filtered, never rebuilt.** A fresh dict is how this goes wrong: a worker
    sits on an `internal: true` network with no default route, so dropping
    `HTTPS_PROXY` does not make the verify command fail, it makes it *hang* -
    until the outer container clock kills it, several hundred seconds later,
    with a reason naming the container. `VERIFY_ENV_REQUIRED` names the eight
    that are load-bearing and a test asserts each one survives.

    **A deny-list, not an allow-list**, which is a deliberate reversal of what
    the plan sketched. An allow-list is the stronger posture against an
    adversary and this is not an adversary problem: `## Verify` is arbitrary
    shell from the target repository, and a repo whose suite needs
    `DATABASE_URL` or `PYTHONPATH` or `NODE_ENV` would be broken by an
    allow-list in a way that looks like a bug in its own tests. The threat
    being addressed is a *leak* - generated code echoing its environment into a
    log that lands in a PR body - and for that, name-shaped filtering is the
    right size.

    The shape test is `containers.manager.SECRET_NAME_RE`, reused rather than
    restated. Handing a container a variable called `..._TOKEN` already enrols
    its value for redaction; this makes the same names invisible one layer
    further in, and the two cannot drift apart because there is one regex.

    **This stops accidents, not attackers.** Verified: `/proc/1/environ` still
    returns `GITHUB_TOKEN` to a scrubbed child, because PID 1 in the container
    is the worker and it is the same user. Anything that reads it deliberately
    still can. The three-container split (fetch / build / publish) is what
    would actually make this safe, and it is a follow-up on #87, not this.
    """
    source = os.environ if environ is None else environ
    return {
        name: value
        for name, value in source.items()
        if name not in VERIFY_ENV_DENY and not SECRET_NAME_RE.search(name)
    }


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
    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=SETTINGS.verify_timeout_s,
            # Filtered from this process's environment, never rebuilt - see
            # `verify_env`. Rebuilding drops the proxy variables, and a worker
            # with no route out does not fail, it hangs.
            env=verify_env(),
        )
    except subprocess.TimeoutExpired:
        return False, f"verify timed out after {SETTINGS.verify_timeout_s}s"
    except OSError as exc:
        # The shell itself could not be started. Nothing ran.
        raise InfrastructureError(f"could not run the verify command: {exc}") from exc

    # Imported here, not at module scope, for `_record`'s reason: `result.py`
    # depends on this module for `WorkerResult` and the exit codes. It has to be
    # *that* function rather than a local slice, or the record and the worker's
    # own output would disagree about what a truncated log looks like.
    from .result import tail

    output = tail(f"{proc.stdout}\n{proc.stderr}".strip(), OUTPUT_TAIL_CHARS)
    verdict = classify_verify(
        proc.returncode, proc.stdout, proc.stderr, time.monotonic() - started
    )
    if verdict.infrastructure:
        raise InfrastructureError(f"{verdict.reason}: {command!r}\n{output}")
    # `result.tail` rather than a bare slice: a complete 3KB log and the last
    # 4KB of a 400KB npm log used to be indistinguishable, and the difference
    # is the difference between "this is the failure" and "the failure is
    # somewhere above this".
    return verdict.passed, output


def install_dependencies(root: Path) -> str | None:
    """Install the checkout's declared dependencies, if it declares any.

    Runs before the verify command, because that is the only ordering under
    which `requirements.txt` means anything: the manifest is how a task says
    what its code imports, and a gate that runs against a bare interpreter
    reads every declaration as a `ModuleNotFoundError`.

    Returns `None` when there is nothing to install or the install succeeded,
    and the failure text otherwise - which the caller folds into the run as a
    FAILED verify. **Deliberately not `classify_verify`, and deliberately not
    infrastructure.** A pip denied the network matches
    `DENIED_EGRESS_SIGNATURES`, and classifying that as exit 2 would retry a
    task forever whose real blocker never changes; the deny-by-default egress
    is the operator's standing decision (`security.EGRESS_EXTRA_ENV`'s
    rationale), so the failure is the task's to report and the retry comment's
    to explain. The one thing this does add is the fix by name, because "403
    Filtered" on its own tells an operator nothing about which knob exists.

    Same execution shape as `run_verify` - shell, repo root, the filtered
    environment - because the install must land in exactly the interpreter and
    proxy configuration the verify command will run under.
    """
    manifest = root / PYTHON_MANIFEST
    if not manifest.is_file():
        return None
    command = f"pip install -r {PYTHON_MANIFEST}"
    print(f"  · installing dependencies: {command}")
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=INSTALL_TIMEOUT_S,
            env=verify_env(),
        )
    except subprocess.TimeoutExpired:
        return f"dependency install timed out after {INSTALL_TIMEOUT_S}s: {command}"
    except OSError as exc:
        # The shell itself could not be started - the same verdict, for the
        # same reason, as `run_verify`'s: nothing about the task ran.
        raise InfrastructureError(f"could not run the dependency install: {exc}") from exc
    if proc.returncode == 0:
        return None

    from .result import tail

    output = tail(f"{proc.stdout}\n{proc.stderr}".strip(), OUTPUT_TAIL_CHARS)
    failure = f"dependency install failed (exit {proc.returncode}): {command}\n{output}"
    haystack = output.casefold()
    if any(signature in haystack for signature in DENIED_EGRESS_SIGNATURES):
        failure += (
            "\n\nThe package index is not on the egress allowlist, so this install "
            "fails identically every attempt. To allow it, restart the run with "
            f"{EGRESS_EXTRA_ENV}=pypi.org,files.pythonhosted.org exported."
        )
    return failure


def written_test_files(written: Sequence[str]) -> tuple[str, ...]:
    """The written paths pytest would treat as test files. See `TEST_FILE_RE`."""
    return tuple(path for path in written if TEST_FILE_RE.search(path))


def audit_collection(root: Path, command: str, written: Sequence[str]) -> str | None:
    """A passed pytest verify only counts if it *collected* this attempt's tests.

    The failure this closes was observed whole in a generated repository: its
    `pyproject.toml` pinned `testpaths = ["tests"]`, so `python -m pytest -q`
    collected 7 tests in `tests/` while ~10 other test files - including two
    with SyntaxErrors and every DB test - were never executed by any gate.
    Workers "passed verify" on tests that verified something else, which is
    the one failure mode the exit-code invariant cannot see from the exit
    code: 0 is 0 however little ran.

    So, when the verify command looks pytest-based (`PYTEST_MARKER`; kept dumb
    on purpose) and it PASSED, ask pytest what an argument-less run collects
    and require every test file this attempt wrote to be in the answer.
    **Argument-less, not per-file**: `testpaths` only applies when the command
    line names nothing, so probing `--collect-only <file>` would collect
    exactly the file the real gate excludes and prove the wrong thing. The
    probe mirrors the gate's shape instead and checks membership in its
    output.

    `sys.executable` rather than a bare `python`: this process's interpreter
    is the container's one Python - the same one the generated verify commands
    run - and it exists by construction on a developer laptop where `python`
    may not. Same cwd and the same filtered environment as the verify itself,
    so the probe answers for the gate that actually ran.

    Scoped to what this attempt wrote (`written_test_files`), never the
    repository's historical test files: failing a worker for someone else's
    debt teaches its retry nothing it can fix. A probe that itself errors or
    times out proves nothing either way, and "cannot be proven to run" fails
    the audit for the same reason "did not run" does.

    Returns `None` when the audit passes or does not apply, and the failure
    text otherwise - the same outcome shape as `install_dependencies`, and for
    the same reason: the text is the retry comment's raw material. `OSError`
    alone is infrastructure, exactly as it is for the verify command: the
    shell itself failing to start says nothing about the task.
    """
    tests = written_test_files(written)
    if not tests or PYTEST_MARKER not in command:
        return None
    probe = f"{sys.executable} -m pytest --collect-only -q"
    try:
        proc = subprocess.run(
            probe,
            shell=True,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=AUDIT_TIMEOUT_S,
            env=verify_env(),
        )
    except subprocess.TimeoutExpired:
        missing = tests
        evidence = f"the collection probe timed out after {AUDIT_TIMEOUT_S}s"
    except OSError as exc:
        raise InfrastructureError(f"could not run the collection audit: {exc}") from exc
    else:
        from .result import tail

        # A collected file appears in `-q` output as `<path>::<test id>` lines.
        # Membership is judged on stdout alone; the return code is not trusted
        # here because a collection *error* in an unrelated file exits non-zero
        # while still listing the files it did collect.
        lines = proc.stdout.splitlines()
        missing = tuple(
            path
            for path in tests
            if not any(line.startswith(f"{path}::") for line in lines)
        )
        if not missing:
            return None
        evidence = tail(f"{proc.stdout}\n{proc.stderr}".strip(), OUTPUT_TAIL_CHARS)
    names = ", ".join(missing)
    return (
        f"the verify command passed, but {names} was not collected by the verify "
        "command - check pytest testpaths/config; a test that never runs proves "
        f"nothing.\n\ncollection probe: {probe}\n{evidence}"
    )


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def _fetch_contract(client: GitHubClient, issue: int) -> tuple[TaskContract, str]:
    try:
        payload = client.get_issue(issue)
    except GitHubError as exc:
        raise InfrastructureError(f"reading issue #{issue}: {exc}") from exc
    return parse_contract(issue, payload.get("body")), payload.get("title") or ""


def fetch_feedback(client: GitHubClient, issue: int) -> str:
    """The latest retry-feedback comment on this issue, or nothing. Never raises.

    The reconciler posts one `apiary: attempt N failed` comment per consumed
    attempt (`orchestrator.reconcile.retry_comment`), so the newest is the
    verdict on the attempt this run is retrying - the older ones are history it
    already superseded. GitHub returns comments oldest first, hence the walk
    from the end.

    A fetch that fails is a retry without feedback, which is exactly what every
    retry was before this method existed - degraded, said out loud, and never a
    reason to burn the attempt: the comment is an aid, not a prerequisite.
    Probed with `getattr` for the same reason the reconciler probes its comment
    methods - an injected client that predates `list_issue_comments` is today's
    behaviour, not an error.
    """
    lister = getattr(client, "list_issue_comments", None)
    if lister is None:
        return ""
    try:
        comments = lister(issue)
    except GitHubError as exc:
        print(f"! could not read retry feedback for #{issue}: {exc}", file=sys.stderr)
        return ""
    for payload in reversed(comments or []):
        text = str(payload.get("body") or "")
        if text.startswith(FEEDBACK_PREFIX):
            return text[:FEEDBACK_MAX_CHARS]
    return ""


def dispatched_attempt(marked: int, env: Mapping[str, str] | None = None) -> int:
    """Which attempt this container is: what it was told, or what the marker says.

    **The variable wins when it is set**, because the orchestrator is the process
    that owns the counter - it is in apiary's store, it was read at dispatch, and
    the marker is a copy of it that #152 stops maintaining. A container told `2`
    by the process that decided to run it must not be talked out of that by a
    body somebody edited.

    **The marker is the fallback, and the fallback is what makes this landable on
    its own.** Until #152 the marker is still written and still correct, so a
    worker that was not told anything behaves exactly as every worker did before:
    an older image under a newer orchestrator, a container started by hand, an
    integration probe. When #152 removes the write, this fallback becomes a lie -
    the marker will read `attempt=0` forever - so it is removed *there*, and the
    variable becomes required. That is one line, and this docstring is where the
    next person looking for it will be.

    **A value that is not a counter is refused rather than rounded.** Guessing
    would name a branch that either collides with a previous attempt's or claims
    budget nobody granted, and both of those are silent: the push succeeds and
    the reconciler goes looking for a branch that means something else. An
    `InfrastructureError` is exit 2, which costs no attempt - the right price for
    a wiring fault the task had nothing to do with.
    """
    raw = (env if env is not None else os.environ).get(ATTEMPT_ENV)
    if raw is None or not raw.strip():
        return marked
    try:
        value = int(raw.strip())
    except ValueError:
        raise InfrastructureError(
            f"{ATTEMPT_ENV}={raw!r} is not a whole number, and it is how this "
            f"container was told which attempt it is. The branch it would push "
            f"cannot be named without it."
        ) from None
    if value < 0:
        raise InfrastructureError(
            f"{ATTEMPT_ENV}={raw!r} is negative; an attempt counter starts at 0."
        )
    return value


def _with_feedback(goal: str, feedback: str) -> str:
    """One brief: what went wrong last time, then the task, clearly delimited.

    The failure leads because the model reads top-down and the single most
    important fact about a retry is that it *is* one - the same goal, handed to
    the same model over the same base commit, reproduces the same code unless
    something in the prompt has changed.
    """
    return (
        "A PREVIOUS ATTEMPT AT THIS EXACT TASK FAILED. Read the failure report "
        "below and do something different this time - repeating the same code "
        "will fail the same way.\n\n"
        f"{feedback}\n\n"
        "--- end of failure report ---\n\n"
        f"{goal}"
    )


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
    # The counter, from the process that owns it. `dispatched_attempt` prefers
    # what the orchestrator told this container and falls back to the marker;
    # everything below reads this local rather than `contract.attempt`, so there
    # is one answer per run and no path that mixes the two.
    attempt = dispatched_attempt(contract.attempt)
    # The name carries the ref and the attempt, so an orchestrator that lost
    # its memory can tell from the remote alone what was in flight and how much
    # budget it had spent (`github/branches.py`). It is the same counter the
    # ledger builds `LedgerEntry.branch` from, so the branch this pushes is the
    # one the reconciler goes looking for.
    branch = task_branch(task_ref(issue), attempt)
    task_id = contract.task_id or f"issue-{issue}"

    try:
        # A retry gets told why the last attempt failed, or it is not a retry -
        # it is the same attempt again. `attempt > 0` is the marker's own
        # testimony that at least one was consumed, and it is checked before
        # the fetch so a first attempt costs no comment listing at all.
        goal = contract.goal
        if attempt > 0:
            feedback = fetch_feedback(client, issue)
            if feedback:
                print(f"  · attempt {attempt + 1}; folding in the failure report")
                goal = _with_feedback(goal, feedback)

        root = prepare_checkout(clone_url, workspace / f"issue-{issue}", base_commit, branch)

        writable = read_writable(root, contract.files)
        readable = gather_context(root, contract.files)
        # The prompt must fit the context window BEFORE the model sees it: Ollama
        # silently truncates the front of an over-long prompt - the system
        # instructions - and the disoriented model then emits junk the parser
        # rejects, which reads as infrastructure and burns free retries on a task
        # whose real problem is that it is too big. Trimming drops context files;
        # a writable set that alone overflows is a failed *task*, reported with
        # the numbers, so the retry comment and the replan can split it.
        readable, oversized = fit_context(goal, writable, readable)
        print(f"  · {len(writable)} file(s) to edit, {len(readable)} for context")
        if oversized is not None:
            return WorkerResult(
                issue=issue,
                repo=repo,
                task_id=task_id,
                branch=branch,
                root=root,
                verify_command=contract.verify,
                verify_output=oversized,
                passed=False,
                attempt=attempt,
            )

        output = propose_edits(goal, writable, readable, llm=editor)
        # The declared set plus the dependency manifests: a worker may always
        # say what its code needs, and refusing `requirements.txt` while
        # installing it before the gate would make the install a rule the model
        # cannot satisfy. Everything else keeps `apply_edits`' refusal
        # semantics untouched.
        applied: Applied = apply_edits(
            root, output.edits, (*contract.files, *DEPENDENCY_MANIFESTS)
        )
        for path, reason in applied.refused:
            print(f"  ! refused {path}: {reason}", file=sys.stderr)

        if not applied.written and not applied.deleted:
            # No commit and no verification: there is nothing to verify, and
            # running the command anyway would report the repository's existing
            # state as this task's result. A deletion counts as an edit here -
            # a cleanup task may legitimately do nothing but remove files, and
            # the gate still has to prove the tree works without them.
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
                attempt=attempt,
            )

        if applied.written:
            print(f"  · wrote {', '.join(applied.written)}")
        if applied.deleted:
            print(f"  · deleted {', '.join(applied.deleted)}")
        print(f"  · verifying: {contract.verify}")
        # From here on the attempt knows what its gate is and what it wrote, so
        # an infrastructure failure carries both rather than reporting an empty
        # command for a run that had already got as far as a green suite.
        try:
            # A written Python file that does not parse fails here, before
            # anything is installed or run: the gate cannot be trusted to catch
            # it (the observed case is a suite whose `testpaths` never
            # collected the broken file), and the SyntaxError text is better
            # retry feedback than whatever a suite that tripped over it
            # second-hand would say. Deleted files are exempt by construction -
            # `applied.deleted` is a separate set, and a file that is gone has
            # nothing left to parse.
            unparsed = syntax_failure(root, applied.written)
            if unparsed is not None:
                passed, verify_output = False, unparsed
            else:
                # Declared dependencies go in before the gate opens; a failed
                # install IS the verify verdict, because it is the task's real
                # blocker and the next retry's feedback. See
                # `install_dependencies` for why it is a failed task and never
                # infrastructure.
                blocked = install_dependencies(root)
                if blocked is not None:
                    passed, verify_output = False, blocked
                else:
                    passed, verify_output = run_verify(root, contract.verify)
                    if passed:
                        # A green pytest gate only counts if it actually
                        # collected the tests this attempt wrote - see
                        # `audit_collection`. It runs on passes only: a failed
                        # gate already carries its own, better feedback. Handed
                        # `applied.written` and never `applied.deleted`: a
                        # deleted test file no longer needs collecting, which
                        # is exactly what makes a cleanup task that removes
                        # obsolete tests winnable at all.
                        uncollected = audit_collection(
                            root, contract.verify, applied.written
                        )
                        if uncollected is not None:
                            passed, verify_output = False, uncollected

            commit = None
            if passed:
                subject = f"swarm[{task_id}]: {(title or contract.goal)[:60]}"
                try:
                    # The gate has run by now, which is the only moment a
                    # lockfile exists to commit: it is produced *by* the verify
                    # command, so staging before verification would always find
                    # nothing. Deletions travel in the same path list: `git add
                    # --force -- <path>` stages the removal of a tracked file
                    # that is gone from the tree, and staging only
                    # `applied.written` would silently commit a cleanup that
                    # cleaned nothing up.
                    commit = commit_edits(
                        root,
                        subject,
                        (*applied.written, *applied.deleted),
                        generated=generated_for(contract.stack),
                    )
                except GitError as exc:
                    raise InfrastructureError(f"committing issue #{issue}: {exc}") from exc
        except InfrastructureError as exc:
            # Deletions travel with the writes, and for the record's reason:
            # "what had this attempt already done" is exactly the question an
            # exit-2 record exists to answer, and a cleanup task that removed
            # its files before the gate broke did real work.
            exc.learned(
                verify_command=contract.verify,
                written=applied.written,
                deleted=applied.deleted,
                task_id=task_id,
            )
            raise

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
            deleted=applied.deleted,
            attempt=attempt,
        )
    except (InfrastructureError, EditError, GitError, GitHubError, OSError) as exc:
        # The contract is in hand from here on, so any failure that escapes can
        # testify which attempt it killed. The record built from it files under
        # that number, and the reconciler discards a record whose attempt is
        # behind the ledger's counter as stale (`record.attempt >=
        # entry.attempt`) - observed live as a run showing "1 in flight"
        # forever against an exited container, because the exit-2 record was
        # hardcoded to attempt 0 against a ledger already on attempt 2. Stamped
        # as an attribute, fill-if-absent like `learned`, because most raisers
        # (a git subprocess, the model transport) rightly know nothing about
        # attempt counters.
        if getattr(exc, "attempt", None) is None:
            exc.attempt = attempt  # type: ignore[union-attr,attr-defined]
        raise


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


#: How much of a failure's text the record's `reason` field keeps. The field is
#: printed inline in run summaries, one line per issue, so it wants the head of
#: the story rather than the whole of it - `verify_output` is the field that
#: carries evidence at `OUTPUT_TAIL_CHARS`.
REASON_MAX_CHARS = 500


def _failure_reason(exc: BaseException) -> str:
    """The failure's own words, flattened and bounded for the record's `reason`.

    The head, not the tail, because an exception introduces itself first -
    `model call failed: OutputParserException: Invalid json output:` - and may
    drag kilobytes of model output behind it, which `verify_output` already
    preserves. This exists because the observed failure record said "no edit
    landed inside the declared file set" - the *previous* attempt's reason,
    inferred from an empty file list - while the OutputParserException that
    actually happened survived only in the container log. The reason field is
    the one line an operator and the reconciler's escalation comment read, so
    it must be the failure itself.
    """
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text[:REASON_MAX_CHARS]


def _record(
    result: WorkerResult,
    exit_code: int,
    *,
    attempt: int | None,
    reason: str | None = None,
) -> None:
    """Write this attempt's result file. Never raises.

    The orchestrator reads exit codes from these records rather than blocking
    on `docker wait`, which is what lets it restart mid-run - so a worker that
    finishes without writing one is indistinguishable from a worker that
    crashed. It is best-effort by design: an unwritable artifacts directory is
    a reason to lose the record, never a reason to change the exit code the
    task actually earned.

    `attempt` is passed explicitly, and is required, because the callers that
    know it best are the exception paths - where there is no honest
    `WorkerResult.attempt` to read, only whatever the escaping exception was
    stamped with. `None` means the failure happened before the contract - the
    only place the number lives - was readable; `result.report` then files the
    record under the next free index on disk rather than under a guessed 0. A
    wrong-but-fresh number keeps the observation alive and clobbers nothing; a
    hardcoded 0 was observed live to overwrite the real first attempt's record
    *and* be discarded as stale against a ledger already on attempt 2, wedging
    the run behind an exited container.

    `reason` is the failure's own text where the caller has one; left `None`,
    the record derives it from the result's shape, which is only honest for a
    result the worker actually returned.
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
            attempt=attempt,
            exit_code=exit_code,
            reason=reason,
            image=os.environ.get(IMAGE_ENV, ""),
        )
    except OSError as exc:
        print(f"! could not write the result record: {exc}", file=sys.stderr)
    else:
        print(f"  · recorded {path.name}")


def _unrun(
    issue: int,
    repo: str,
    reason: str,
    *,
    verify_command: str = "",
    written: tuple[str, ...] = (),
    deleted: tuple[str, ...] = (),
    task_id: str = "",
    attempt: int = 0,
) -> WorkerResult:
    """A result for an attempt that never got as far as a verify command.

    It used to blank `verify_command` and `written` unconditionally, which is
    exactly backwards: this is the *infrastructure* path, and the two questions
    a human asks about an attempt that died here are "what was it about to run"
    and "had it written anything first". A clone that failed genuinely has
    neither; a push that failed after a green gate has both, and reporting an
    empty command for it is a record that actively misleads. `deleted` is the
    same fact for the files the attempt removed - a cleanup task's whole
    output - and used to be dropped on exactly this path.
    """
    return WorkerResult(
        issue=issue,
        repo=repo,
        task_id=task_id,
        # `attempt` defaults to 0 because both callers get here without having
        # read the marker - one because the body would not parse, the other
        # because the host broke first. The branch is a field of the record and
        # nothing dispatches from it; naming the attempt-0 branch is the honest
        # answer when no attempt number was ever learned.
        branch=task_branch(task_ref(issue), attempt),
        root=Path("."),
        verify_command=verify_command,
        verify_output=reason,
        passed=False,
        written=written,
        deleted=deleted,
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
        # on a second attempt (see the module docstring). A body that does not
        # parse has no readable attempt marker either, so the attempt is
        # unknown here and the record files under the next free index.
        print(f"! {exc}", file=sys.stderr)
        _record(
            _unrun(args.issue, slug, str(exc)),
            EXIT_TASK_FAILED,
            attempt=None,
            reason=_failure_reason(exc),
        )
        return EXIT_TASK_FAILED
    except (InfrastructureError, EditError, GitError, GitHubError, OSError) as exc:
        print(f"! {exc}", file=sys.stderr)
        # Exit 2 is the code the reconciler must see to leave the attempt
        # budget alone, and it can only see it in a record - filed under the
        # attempt `run_worker` stamped on the exception, so the reconciler's
        # staleness guard does not discard it on a retry. `reason` is the
        # exception's own words: the record used to infer one from its empty
        # file list and claim "no edit landed" for a model call that blew up.
        _record(
            _unrun(
                args.issue,
                slug,
                str(exc),
                verify_command=getattr(exc, "verify_command", ""),
                written=tuple(getattr(exc, "written", ())),
                deleted=tuple(getattr(exc, "deleted", ())),
                task_id=getattr(exc, "task_id", ""),
            ),
            EXIT_INFRASTRUCTURE,
            attempt=getattr(exc, "attempt", None),
            reason=_failure_reason(exc),
        )
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
            # The run itself finished, so its result carries the real attempt;
            # only the reason is the publish failure's, because "verified and
            # committed" on an exit-2 record would be a sentence at war with
            # its own exit code.
            _record(
                result,
                EXIT_INFRASTRUCTURE,
                attempt=result.attempt,
                reason=_failure_reason(exc),
            )
            return EXIT_INFRASTRUCTURE

    _record(result, result.exit_code, attempt=result.attempt)

    if result.exit_code != EXIT_OK and not args.keep:
        # The container is cattle and its filesystem dies with it, so this only
        # matters when a human is running the worker on a laptop - where the
        # leftover checkout would collide with the next attempt.
        shutil.rmtree(result.root, ignore_errors=True)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
