"""The whole pipeline from the console: fire a real run, watch it happen.

The console's three sites each fire *one model call* - that is their whole
point, and it is also why the operator who fired the planner from the page saw
a plan and then nothing: no repository, no issues, no workers, no pull
requests. Those only ever existed behind `swarm run` in a terminal. This
module puts the real thing behind a fourth tab.

**A subprocess, not an import.** The run loop in `cli._loop` owns signal
handlers (the reaper), installs a capture recorder for its artifacts
directory, and prints its progress line by line - all of which assume it *is*
the process. Importing it into the console's process would fight the console's
own recorder and make Ctrl-C ambiguous. So the tab execs the same command the
terminal would: `python -m swarm run ...`, unbuffered, stdout and stderr
merged, and the page shows exactly the lines a terminal would have shown.
What the operator debugs from the page is therefore what they would debug
from a shell, which is the console's founding rule (`prompt_for, never an
approximation`) applied to a whole run.

**The checkbox is the merge policy.** `APIARY_MERGE_ADMIN_OVERRIDE` is read
once by the run that starts (`MergePolicy.from_env`), so the page states it
explicitly per run rather than inheriting whatever the console's shell had:
checked writes `1` into the child's environment, unchecked writes `0`. An
inherited policy the operator cannot see on the page is one they cannot have
chosen - the same argument `cli._loop` makes for printing it.

**Single-flight, like the model tabs, for a different reason.** Two console
model calls are refused because Ollama loads one model at a time. Two *runs*
are refused because they would contend for the same worker-image slots and
the same Ollama, and because a page that fans out into N interleaved logs
answers no question. One run at a time, refused rather than queued.

**Tokens are checked before anything starts.** The child would discover a
missing `GITHUB_TOKEN` on its own - after a couple of seconds, as a one-line
error. But the greenfield path discovers a missing `APIARY_PROVISION_TOKEN`
only when `provision` reaches for it, and a refusal the console can issue
instantly, with the fix attached, beats an error the operator has to scroll
for. The check is presence, not validity: validity is GitHub's to judge.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .github.ledger import KNOWN_STACKS

__all__ = [
    "SWARM_SITE",
    "SwarmRunError",
    "SwarmRuns",
    "RunJob",
    "build_argv",
    "child_env",
]

#: The one credential every run needs, and the second one only `--new` needs.
#: Names imported nowhere: `security.PROVISION_TOKEN_ENV` exists, but importing
#: `security` here would drag its whole policy surface into the console's
#: import graph for one string constant.
WORK_TOKEN_ENV = "GITHUB_TOKEN"
PROVISION_TOKEN_ENV = "APIARY_PROVISION_TOKEN"

MERGE_OVERRIDE_ENV = "APIARY_MERGE_ADMIN_OVERRIDE"

#: `owner/name`, the only repo shape v2 accepts - same expression as `run.py`.
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

#: Memory cap for one run's log. A real run prints a few hundred lines; ten
#: thousand means something is looping, and the page saying so beats the
#: console process growing without bound.
MAX_LINES = 10_000
MAX_LINE_CHARS = 2_000

# What the run prints, read back for the page's summary strip. These parse
# `cli.py`'s own output, which is pinned by its tests; a format change there
# degrades this to "no summary" rather than to a wrong one.
_REPO_LINE = re.compile(r"\brepo ([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")
#: The run-identity line `_report_run` prints. Captured so the page can tell
#: "this console job" and "the same run seen through its artifacts" apart -
#: without it the external-run view would show every console run twice.
_RUN_ID_LINE = re.compile(r"»\s*run\s+([a-z0-9][a-z0-9-]*)\s")
_CYCLE_LINE = re.compile(r"\bcycle (\d+):")
_PR_REF = re.compile(r"\bPR #(\d+)")
_ISSUE_LINE = re.compile(r"^\s*#(\d+): ")


class SwarmRunError(ValueError):
    """A refusal an operator can fix, with the fix attached."""

    def __init__(self, message: str, *, fix: str = "") -> None:
        super().__init__(message)
        self.fix = fix


# --------------------------------------------------------------------------
# The form, and what it becomes
# --------------------------------------------------------------------------

#: The fourth tab's descriptor, shaped like `Site.to_dict()` plus a `kind` the
#: front end branches on. Served under its own key in `/sites` rather than
#: appended to the sites list, so nothing that iterates model-call sites picks
#: up a form with no prompt behind it.
SWARM_SITE: dict[str, Any] = {
    "key": "swarm",
    "kind": "swarm",
    "label": "run — the whole swarm",
    "blurb": (
        "The real thing, exactly as `swarm run` would do it from a terminal: attach to the "
        "repository — creating it first if it does not exist yet — plan the objective into "
        "GitHub issues, dispatch one worker container per ready issue, open a pull request "
        "per task, and merge what passes CI until the objective is met. The log below is "
        "the run's own output, live."
    ),
    "fields": [
        {"name": "objective", "label": "Objective / project brief", "kind": "area",
         "placeholder": "What the swarm should accomplish. For a new repository this is also the brief it is created from.", "value": ""},
        {"name": "repo", "label": "Repository — owner/name on GitHub, or a directory path when running locally; created for you if it does not exist yet",
         "kind": "text", "placeholder": "kamyar-finlex/expense-tracker", "value": ""},
        {"name": "local", "label": "Run locally — no GitHub: git worktrees instead of issues, host models instead of containers, merges instead of pull requests. The board stays empty; the log is the run.",
         "kind": "check", "value": ""},
        {"name": "public", "label": "Create a new repository public — a free GitHub plan cannot put branch protection on a private repository (existing repositories are untouched)",
         "kind": "check", "value": "1"},
        {"name": "verify", "label": "Verify command (optional; default: the scaffold's, else SWARM_VERIFY)",
         "kind": "text", "placeholder": "python -m pytest -q", "value": ""},
        {"name": "stack", "label": f"Stack (optional: {', '.join(sorted(KNOWN_STACKS))})",
         "kind": "text", "placeholder": "python", "value": ""},
        {"name": "max_cycles", "label": "Stop after this many cycles (optional; default: until the objective is met)",
         "kind": "text", "placeholder": "", "value": ""},
        {"name": "auto_merge", "label": "Merge green pull requests automatically (admin override) — unchecked, every PR waits for a human",
         "kind": "check", "value": "1"},
        {"name": "no_goal_check", "label": "Stop when the plan is exhausted — skip the goal gate that judges the objective and plans follow-up work (the run does exactly the tasks that were planned, no more)",
         "kind": "check", "value": ""},
    ],
}


def target(values: Mapping[str, str]) -> str:
    """The one repository this run is about, validated. The objective is
    checked here too, because nothing downstream is worth doing without either."""
    if not (values.get("objective") or "").strip():
        raise SwarmRunError("a run needs an objective",
                            fix="describe what the swarm should accomplish")
    repo = (values.get("repo") or "").strip()
    if not repo:
        raise SwarmRunError("a run needs a repository",
                            fix="owner/name — it is created for you if it does not exist yet")
    if not REPO_RE.match(repo):
        raise SwarmRunError(f"repo must be 'owner/name', got {repo!r}",
                            fix="e.g. kamyar-finlex/expense-tracker")
    return repo


def build_argv(values: Mapping[str, str], *, exists: bool) -> list[str]:
    """The exact command a terminal run would be, from what was typed.

    `exists` is the one fact the form does not carry: whether the repository
    is already on GitHub. The operator names *where the work should go* and
    the mode is derived - the earlier form that made them choose between an
    "existing repo" field and an "owner for a new one" field was filled in
    wrong twice on its first day, which is the form's failure, not theirs.

    Refusals mirror `cli._target`'s, issued here so they arrive as a 400 with
    a fix instead of as argparse's usage text in the log.
    """
    objective = (values.get("objective") or "").strip()
    repo = target(values)

    # `swarm.cli`, not `swarm`: the package ships no `__main__.py` on purpose
    # (the `swarm` entry point is the CLI), so `-m swarm` does not execute.
    argv = [sys.executable, "-u", "-m", "swarm.cli", "run"]
    if exists:
        argv += ["--repo", repo, "--objective", objective]
    else:
        # `--yes`: there is no terminal to answer provision's confirmation, and
        # the operator pressing the button *is* the confirmation.
        owner, name = repo.split("/", 1)
        argv += ["--new", objective, "--owner", owner, "--name", name, "--yes"]
        # Private is the CLI's default, and on a free plan it is also the one
        # that fails after the repository exists: rulesets on private repos
        # are 403 without Pro, which leaves a half-provisioned repo a human
        # has to delete. The page therefore states the choice explicitly.
        if values.get("public") == "1":
            argv += ["--public"]

    if verify := (values.get("verify") or "").strip():
        argv += ["--verify", verify]
    if stack := (values.get("stack") or "").strip():
        if stack not in KNOWN_STACKS:
            raise SwarmRunError(f"unknown stack {stack!r}",
                                fix=f"one of: {', '.join(sorted(KNOWN_STACKS))}, or leave it empty")
        argv += ["--stack", stack]
    argv += _cycles_flag(values, "--max-cycles")
    if values.get("no_goal_check") == "1":
        argv += ["--no-goal-check"]
    return argv


def build_local_argv(values: Mapping[str, str]) -> list[str]:
    """`swarm local`: the v1 graph on this machine, nothing on GitHub.

    The repo field is a directory path here, so the `owner/name` shape check
    does not apply - `swarm local` creates and initialises the directory if it
    is missing, keeping the field's promise either way."""
    objective = (values.get("objective") or "").strip()
    if not objective:
        raise SwarmRunError("a run needs an objective",
                            fix="describe what the swarm should accomplish")
    path = (values.get("repo") or "").strip()
    if not path:
        raise SwarmRunError("a local run needs a directory",
                            fix="a path like ~/poc/wallet-local — created for you if missing")
    argv = [sys.executable, "-u", "-m", "swarm.cli", "local",
            "--repo", path, "--objective", objective]
    if verify := (values.get("verify") or "").strip():
        argv += ["--verify", verify]
    argv += _cycles_flag(values, "--max-rounds")
    return argv


def _cycles_flag(values: Mapping[str, str], flag: str) -> list[str]:
    cycles = (values.get("max_cycles") or "").strip()
    if not cycles:
        return []
    if not cycles.isdigit():
        raise SwarmRunError(f"max cycles must be a number, got {cycles!r}",
                            fix="a whole number, or leave it empty")
    return [flag, cycles]


def child_env(
    values: Mapping[str, str],
    *,
    greenfield: bool,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The child's environment: the console's, with the page's choices written in.

    Also where the token preflight lives, because the environment is what it
    checks. Presence only - validity is GitHub's to judge, and the child's own
    error will name anything subtler.
    """
    env = dict(os.environ if base is None else base)
    if not env.get(WORK_TOKEN_ENV):
        raise SwarmRunError(
            f"{WORK_TOKEN_ENV} is not set in the console's environment",
            fix="start the console from a shell with the credentials loaded: "
                "`set -a; source ~/.config/apiary/env; set +a; swarm console`",
        )
    if greenfield and not env.get(PROVISION_TOKEN_ENV):
        raise SwarmRunError(
            f"creating a repository needs {PROVISION_TOKEN_ENV}, which is not set",
            fix=f"mint the boot key (administration, contents, workflows, issues, metadata), "
                f"export it as {PROVISION_TOKEN_ENV}, and restart the console",
        )
    # The checkbox is authoritative per run, whatever the shell said: a policy
    # the page shows and the child inherits differently would be the lie.
    env[MERGE_OVERRIDE_ENV] = "1" if values.get("auto_merge") == "1" else "0"
    return env


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------


@dataclass
class RunJob:
    """One in-flight or finished run, and everything the page shows about it."""

    id: str
    command: str                       # display form; carries no secrets
    started: float
    state: str = "running"             # running | done | failed | stopped
    returncode: int | None = None
    #: A local run's "repo" is a filesystem path, and a path reliably contains
    #: an `owner/name`-shaped substring (`Users/Kamyar`), so parsing it would
    #: hand the page a GitHub link to a repository that does not exist.
    local: bool = False
    lines: list[str] = field(default_factory=list)
    progress: dict[str, Any] = field(default_factory=lambda: {
        "repo": "", "repo_url": "", "run_id": "", "cycle": None,
        "issues": [], "prs": [], "note": "", "met": False,
    })
    stop_requested: bool = False

    def absorb(self, line: str) -> None:
        """File one output line and fold it into the summary strip."""
        if len(self.lines) >= MAX_LINES:
            if len(self.lines) == MAX_LINES:
                self.lines.append("… output truncated: the run printed "
                                  f"more than {MAX_LINES} lines …")
            return
        self.lines.append(line[:MAX_LINE_CHARS])

        p = self.progress
        if not p["run_id"] and (m := _RUN_ID_LINE.search(line)):
            p["run_id"] = m.group(1)
        if not self.local and not p["repo"] and (m := _REPO_LINE.search(line)):
            p["repo"] = m.group(1)
            # Built here from the validated slug, never lifted from the log:
            # the log quotes issue titles, which a model wrote.
            p["repo_url"] = f"https://github.com/{m.group(1)}"
        if m := _CYCLE_LINE.search(line):
            p["cycle"] = int(m.group(1))
        for m in _PR_REF.finditer(line):
            if (n := int(m.group(1))) not in p["prs"]:
                p["prs"].append(n)
        if m := _ISSUE_LINE.match(line):
            if (n := int(m.group(1))) not in p["issues"]:
                p["issues"].append(n)
        stripped = line.strip()
        if stripped.startswith(("»", "!")):
            p["note"] = stripped.lstrip("»! ").strip()
        if "objective met" in line:
            p["met"] = True

    def to_dict(self, *, since: int = 0, now: float | None = None) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "returncode": self.returncode,
            "elapsed_s": round((now or time.monotonic()) - self.started, 1),
            "command": self.command,
            "lines": self.lines[since:],
            "next": len(self.lines),
            "progress": self.progress,
        }


# --------------------------------------------------------------------------
# The manager
# --------------------------------------------------------------------------


def repo_exists(repo: str) -> bool:
    """Ask GitHub whether the repository is there. 404 is an answer, not an
    error - it is what turns a run greenfield. Anything else (an outage, a bad
    token) is a refusal: guessing "create it" against a 503 would provision a
    duplicate the moment GitHub recovers."""
    from .github.client import GitHubClient, GitHubError, GitHubHTTPError

    try:
        GitHubClient.from_env(repo).get_repo()
        return True
    except GitHubHTTPError as exc:
        if exc.status == 404:
            return False
        raise SwarmRunError(
            f"GitHub could not confirm whether {repo} exists: {exc}",
            fix="if GitHub is having an incident, wait and fire again",
        ) from exc
    except GitHubError as exc:
        raise SwarmRunError(str(exc),
                            fix="export GITHUB_TOKEN in the console's shell") from exc


@dataclass
class SwarmRuns:
    """Start, watch and stop runs. `spawn` and `exists` are the test seams, as
    `provision`'s `target` is: tests hand in a factory returning a fake
    process and an answer about the repository, and never touch a real
    subprocess or a real GitHub."""

    spawn: Callable[..., Any] = subprocess.Popen
    exists: Callable[[str], bool] = repo_exists
    jobs: dict[str, RunJob] = field(default_factory=dict)
    _procs: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _running: str = ""

    def start(self, values: Mapping[str, str]) -> RunJob:
        local = values.get("local") == "1"
        if local:
            # No GitHub: no tokens to check, no existence to probe. The child
            # creates the directory if it is missing.
            argv = build_local_argv(values)
            env = dict(os.environ)
        else:
            # The one read this module does before exec'ing the real thing: is
            # the repository there? The answer picks the mode, so the operator
            # only ever says *where*, never *which command*.
            found = self.exists(target(values))
            argv = build_argv(values, exists=found)
            env = child_env(values, greenfield=not found)

        with self._lock:
            if self._running:
                live = self.jobs.get(self._running)
                raise SwarmRunError(
                    "a run is already in flight, and two would contend for the same "
                    "workers and the same Ollama",
                    fix=f"watch or stop the running one ({live.id if live else '?'}) first",
                )
            job = RunJob(
                id=uuid.uuid4().hex[:16],
                # Without the interpreter path, which is long and says nothing.
                command="swarm " + " ".join(shlex.quote(a) for a in argv[4:]),
                started=time.monotonic(),
                local=local,
            )
            self.jobs[job.id] = job
            self._running = job.id

        try:
            proc = self.spawn(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except Exception as exc:
            with self._lock:
                job.state = "failed"
                job.absorb(f"! could not start the run: {type(exc).__name__}: {exc}")
                self._running = ""
            raise SwarmRunError(f"could not start the run: {exc}",
                                fix="is this the venv the swarm is installed in?") from exc

        self._procs[job.id] = proc
        threading.Thread(target=self._watch, args=(job, proc), daemon=True).start()
        return job

    def _watch(self, job: RunJob, proc: Any) -> None:
        """Read the child to EOF, then publish the ending. State is written
        last, under the lock, for the same reason `Console._work` orders it
        that way: a poll must never see a finished run with no verdict."""
        try:
            for line in iter(proc.stdout.readline, ""):
                with self._lock:
                    job.absorb(line.rstrip("\n"))
        finally:
            returncode = proc.wait()
            with self._lock:
                job.returncode = returncode
                if job.stop_requested:
                    job.state = "stopped"
                elif returncode == 0:
                    job.state = "done"
                else:
                    job.state = "failed"
                self._running = ""
            self._procs.pop(job.id, None)

    def stop(self, job_id: str) -> RunJob:
        job = self.jobs.get(job_id)
        if job is None:
            raise SwarmRunError("no such run")
        proc = self._procs.get(job_id)
        with self._lock:
            if job.state != "running" or proc is None:
                return job
            job.stop_requested = True
            job.absorb("! stop requested from the console; containers are being disposed")
        # SIGINT, precisely because it is what Ctrl-C sends: `cli._loop`
        # catches KeyboardInterrupt and disposes this run's containers on the
        # way out. SIGKILL would leave orphans for the next run's reaper.
        try:
            proc.send_signal(signal.SIGINT)
        except (ProcessLookupError, OSError):
            pass
        return job

    def status(self, job_id: str, *, since: int = 0) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            raise SwarmRunError("no such run")
        with self._lock:
            return job.to_dict(since=since)

    def latest(self) -> dict[str, Any] | None:
        """The most recently started run, whole log included, or None.

        This is what lets a reloaded page - or one that never started the run,
        because a terminal or another session did - pick it up mid-flight
        instead of showing an empty tab beside a working swarm."""
        with self._lock:
            if not self.jobs:
                return None
            job = next(reversed(self.jobs.values()))
            return job.to_dict()
