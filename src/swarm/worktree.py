"""Git worktree isolation.

The single most important piece of infrastructure in a parallel coding
swarm. Two agents editing one checkout corrupts both. One worktree per
worker means each gets a private filesystem view on its own branch, and
integration becomes an ordinary merge you can inspect and revert.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class GitError(RuntimeError):
    pass


def _git(repo: Path, *args: str, timeout: int = 120) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed:\n{proc.stderr.strip()}")
    return proc.stdout.strip()


@dataclass
class Worktree:
    path: Path
    branch: str
    repo: Path

    def remove(self, force: bool = True) -> None:
        args = ["worktree", "remove", str(self.path)]
        if force:
            args.append("--force")
        try:
            _git(self.repo, *args)
        except GitError:
            shutil.rmtree(self.path, ignore_errors=True)
            _git(self.repo, "worktree", "prune")


def ensure_repo(repo_path: str | Path) -> Path:
    repo = Path(repo_path).resolve()
    if not (repo / ".git").exists():
        raise GitError(f"{repo} is not a git repository. Run `git init` first.")
    return repo


def base_branch(repo: Path) -> str:
    return _git(repo, "rev-parse", "--abbrev-ref", "HEAD")


def create_worktree(repo: Path, worktree_root: str, task_id: str, base: str) -> Worktree:
    """Create `<worktree_root>/<task_id>` on a fresh branch `swarm/<task_id>`."""
    branch = f"swarm/{task_id}"
    path = (repo / worktree_root / task_id).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Idempotent: clear any leftovers from a previous crashed run.
    if path.exists():
        Worktree(path, branch, repo).remove()
    existing = _git(repo, "branch", "--list", branch)
    if existing:
        _git(repo, "branch", "-D", branch)

    _git(repo, "worktree", "add", "-b", branch, str(path), base)
    return Worktree(path=path, branch=branch, repo=repo)


def commit_all(worktree: Worktree, message: str) -> bool:
    """Commit everything in the worktree. Returns False if nothing changed."""
    _git(worktree.path, "add", "-A")
    status = _git(worktree.path, "status", "--porcelain")
    if not status:
        return False
    _git(worktree.path, "-c", "user.name=swarm", "-c", "user.email=swarm@local",
         "commit", "-m", message)
    return True


def merge_branch(repo: Path, branch: str) -> tuple[bool, str]:
    """Merge a worker branch into the current branch.

    Returns (ok, message). On conflict we abort cleanly rather than leaving
    the repo in a half-merged state - the orchestrator decides what to do.
    """
    try:
        out = _git(repo, "merge", "--no-ff", "-m", f"swarm: merge {branch}", branch)
        return True, out
    except GitError as exc:
        try:
            _git(repo, "merge", "--abort")
        except GitError:
            pass
        return False, str(exc)


def cleanup_all(repo: Path, worktree_root: str) -> None:
    root = repo / worktree_root
    if root.exists():
        for child in root.iterdir():
            if child.is_dir():
                Worktree(child, f"swarm/{child.name}", repo).remove()
    _git(repo, "worktree", "prune")
