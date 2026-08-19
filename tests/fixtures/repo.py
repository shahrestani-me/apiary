"""A scratch git repository, with a bare repo standing in for `origin`.

A v2 worker clones a repo, branches, edits, runs a verify command, commits and
pushes ([`docs/architecture-v2.md`](../../docs/architecture-v2.md)). Every step
of that is testable on a laptop with no network and no token, provided the
remote is a bare repository on disk - `git push` neither knows nor cares that
`origin` is a path.

So this builds one: a working tree with seeded commits and a `pytest` target
that actually passes, plus a bare sibling wired up as `origin`. `clone()` gives
a second working copy of the same remote, which is the shape the worker
container has.

Hermetic on purpose. Identity comes from the environment rather than from
`~/.gitconfig`, `HOME` points inside the temp directory and the system config
is switched off, so the fixture behaves the same on a developer machine with
opinionated git defaults (`init.defaultBranch`, `commit.gpgsign`, hooks) as it
does on a bare CI runner.

    def test_worker_pushes(scratch_repo):
        scratch_repo.branch("apiary/%237-attempt-0")
        scratch_repo.write("calc.py", "def add(a, b):\\n    return a + b\\n")
        scratch_repo.commit("fix add")
        scratch_repo.push()
        assert "apiary/%237-attempt-0" in scratch_repo.remote_branches()
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: The seed tree. `test_calc.py` passes, so `verify()` is green before anything
#: touches the repo - a fixture whose verify command starts red cannot tell a
#: broken edit from a broken fixture.
SEED_FILES: dict[str, str] = {
    ".gitignore": "__pycache__/\n.pytest_cache/\n",
    "README.md": "# scratch\n\nA throwaway repository built by the test suite.\n",
    "calc.py": "def add(a, b):\n    return a + b\n",
    "test_calc.py": (
        "from calc import add\n"
        "\n"
        "\n"
        "def test_add():\n"
        "    assert add(1, 2) == 3\n"
    ),
}

#: `sys.executable`, not `python`: the interpreter running the suite is the one
#: with pytest installed, and on a mac `python` may not exist at all.
VERIFY_COMMAND = f"{sys.executable} -m pytest -q"


@dataclass
class ScratchRepo:
    """A working tree, its bare `origin`, and the git commands worth wrapping."""

    path: Path
    remote: Path
    default_branch: str = "main"
    env: dict[str, str] = field(default_factory=dict)

    # --- construction --------------------------------------------------

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        name: str = "scratch",
        branch: str = "main",
        files: dict[str, str] | None = None,
        message: str = "seed",
    ) -> ScratchRepo:
        """A repo at `root/name`, committed, pushed to `root/name.git`."""
        work = root / name
        remote = root / f"{name}.git"
        work.mkdir(parents=True)

        repo = cls(work, remote, default_branch=branch, env=_git_env(root))
        repo.git("init", "-q", "-b", branch)
        repo.git("init", "-q", "--bare", "-b", branch, str(remote), cwd=root)
        repo.git("remote", "add", "origin", str(remote))

        for relative, content in (SEED_FILES if files is None else files).items():
            repo.write(relative, content)
        repo.commit(message)
        repo.git("push", "-q", "-u", "origin", branch)
        return repo

    def clone(self, dest: Path, *, branch: str | None = None) -> ScratchRepo:
        """A second working copy of the same remote - the worker's shape."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        args = ["clone", "-q", *(("--branch", branch) if branch else ()), str(self.remote), str(dest)]
        self.git(*args, cwd=dest.parent)
        return ScratchRepo(dest, self.remote, self.default_branch, dict(self.env))

    # --- plumbing ------------------------------------------------------

    def git(self, *args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=cwd or self.path,
            env={**os.environ, **self.env},
            check=check,
            capture_output=True,
            text=True,
        )

    def out(self, *args: str) -> str:
        """`git ...`, stripped stdout."""
        return self.git(*args).stdout.strip()

    # --- the working tree ----------------------------------------------

    def write(self, relative: str, content: str) -> Path:
        target = self.path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return target

    def read(self, relative: str) -> str:
        return (self.path / relative).read_text()

    def commit(self, message: str = "change") -> str:
        """Stage everything and commit. Returns the new sha."""
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.head()

    # --- refs -----------------------------------------------------------

    def branch(self, name: str) -> str:
        """Create and check out `name`. Returns it, for readability at call sites."""
        self.git("switch", "-q", "-c", name)
        return name

    def checkout(self, name: str) -> str:
        self.git("switch", "-q", name)
        return name

    def current_branch(self) -> str:
        return self.out("rev-parse", "--abbrev-ref", "HEAD")

    def head(self, ref: str = "HEAD") -> str:
        return self.out("rev-parse", ref)

    def subjects(self, ref: str = "HEAD") -> list[str]:
        """Commit subjects, newest first."""
        return self.out("log", "--format=%s", ref).splitlines()

    def branches(self) -> list[str]:
        return sorted(self.out("for-each-ref", "--format=%(refname:short)", "refs/heads").split())

    # --- the remote ------------------------------------------------------

    def push(self, branch: str | None = None) -> subprocess.CompletedProcess:
        return self.git("push", "-q", "origin", branch or self.current_branch())

    def remote_branches(self) -> list[str]:
        listing = self.git("for-each-ref", "--format=%(refname:short)", "refs/heads", cwd=self.remote)
        return sorted(listing.stdout.split())

    def remote_head(self, branch: str | None = None) -> str:
        return self.git(
            "rev-parse", branch or self.default_branch, cwd=self.remote
        ).stdout.strip()

    # --- the verify command ----------------------------------------------

    def verify(self, command: str = VERIFY_COMMAND) -> subprocess.CompletedProcess:
        """Run a contract's `## Verify` command from the repo root.

        `shell=True` and only the exit code is believed, exactly as
        `swarm/nodes/verifier.py` does it - a fixture that judged the output
        would be teaching the wrong lesson.
        """
        return subprocess.run(
            command,
            shell=True,
            cwd=self.path,
            env={**os.environ, **self.env},
            capture_output=True,
            text=True,
        )


def _git_env(root: Path) -> dict[str, str]:
    """Identity and isolation, so no test depends on the developer's git config."""
    home = root / "git-home"
    home.mkdir(exist_ok=True)
    config = home / ".gitconfig"
    config.touch()
    return {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home),
        "GIT_CONFIG_GLOBAL": str(config),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_AUTHOR_NAME": "apiary tests",
        "GIT_AUTHOR_EMAIL": "tests@apiary.invalid",
        "GIT_COMMITTER_NAME": "apiary tests",
        "GIT_COMMITTER_EMAIL": "tests@apiary.invalid",
    }
