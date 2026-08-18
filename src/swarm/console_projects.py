"""The projects the console knows about, kept across restarts.

Everything the console shows today is either in one process's memory
(`SwarmRuns.jobs`) or re-derived per poll from GitHub or from run artifacts.
That is right for runs - a run is an event - but wrong for the thing an
operator actually returns to, which is a *project*: one repository, the
objective it is being driven toward, and the stack and gate it is driven with.
Today that continuity lives in the operator's head and in whatever the form
happened to have typed into it last; a console restart, or a second machine,
starts from a blank form. This module is the memory the page can grow a
project selector on: one row per repository, most recently active first.

**SQLite, stdlib, one file.** The store is a single table in
`.swarm/projects.sqlite`, beside the run directories whose history it
summarises. SQLite rather than a JSON file because the one operation that must
not lose data here is a read-modify-write from a threaded server, and SQLite's
own locking already arbitrates that; stdlib `sqlite3` because this project
takes no new dependencies for its console. The file sits at
`artifacts_root().parent / "projects.sqlite"` - the `.swarm/` directory by
default - so one directory still holds everything the swarm leaves on a
developer's disk, and an operator who moved the artifacts root has moved this
with it.

**A connection per operation, deliberately.** The console's handler is a
`ThreadingHTTPServer`, so these methods are called from whichever thread a
request happened to land on - and a `sqlite3` connection is, by default, bound
to the thread that created it. The alternative, one shared connection opened
with `check_same_thread=False` behind a lock, works only as long as every
cursor's whole life stays inside the lock, which is an invariant a later edit
can silently break. Opening per operation makes each call self-contained: no
lock scope to get wrong, SQLite's file locking (with the default five-second
busy timeout) serialises the writers, and the cost is one `open()` against a
table a human clicks at.

**Seeded from the run artifacts, idempotently.** The projects this store
should know about on day one already exist - as `run.json` files under
`.swarm/runs/`, one per recorded run. So the first use of a store scans them:
one project per distinct `repo`, its objective, stack and verify taken from
the **latest** run (latest by `started_at`), `created_at` from the earliest
run and `last_run_at` from the latest. Only repos of the `owner/name` shape
are migrated - a local run records a filesystem path in the same field, and
`Users/Kamyar` is a valid `owner/name`-shaped *substring* but not a project on
GitHub, which is exactly the trap `RunJob.local` already documents. A run
directory whose `run.json` is missing or malformed is skipped and said so on
stderr, never fatal: a half-written file from a killed run must not cost the
console its project list.

**Upserts never let an empty field erase a real one.** The console records a
project whenever a GitHub run starts, and the form's stack and verify fields
are optional - so most runs would otherwise blank out what a previous run, or
the migration, had written down. An empty objective, stack or verify in an
upsert therefore means "no news", not "erase"; and `last_run_at` only ever
moves forward, so re-running the migration after a console-started run cannot
rewind a project's recency.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import re
import sqlite3
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from .artifacts import RUN_FILE_NAME, artifacts_root

__all__ = ["ProjectError", "ProjectStore", "REPO_RE"]

#: `owner/name` - the same shape `console_runs.REPO_RE`, `console_board` and
#: `console_external` each check before a repo string becomes a command
#: argument or an href. Restated here rather than imported so this module's
#: import graph stays "stdlib plus artifacts", the way the other console
#: readers keep theirs.
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class ProjectError(ValueError):
    """A refusal an operator can fix, with the fix attached - the same shape
    `SwarmRunError` and `BoardError` have, rendered by the console as
    `{error, fix}`."""

    def __init__(self, message: str, *, fix: str = "") -> None:
        super().__init__(message)
        self.fix = fix


#: `repo` is the primary key because a project *is* a repository here: the
#: selector this store exists for offers repositories, and two rows for one
#: repo would be two answers to "what is this repo's objective".
_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    repo        TEXT PRIMARY KEY,
    objective   TEXT NOT NULL DEFAULT '',
    stack       TEXT NOT NULL DEFAULT '',
    verify      TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    last_run_at TEXT
)
"""

#: The empty-means-no-news rule, in the one place it is enforced. `created_at`
#: is deliberately absent from the UPDATE branch: when a project was first
#: written down is a fact about the row, not about the run that touched it.
#: `last_run_at` takes the maximum, so a replayed migration cannot rewind a
#: project behind a run the console recorded a minute ago.
_UPSERT = """
INSERT INTO projects (repo, objective, stack, verify, created_at, last_run_at)
VALUES (:repo, :objective, :stack, :verify, :created_at, :last_run_at)
ON CONFLICT(repo) DO UPDATE SET
    objective = CASE WHEN excluded.objective <> '' THEN excluded.objective
                     ELSE projects.objective END,
    stack     = CASE WHEN excluded.stack <> '' THEN excluded.stack
                     ELSE projects.stack END,
    verify    = CASE WHEN excluded.verify <> '' THEN excluded.verify
                     ELSE projects.verify END,
    last_run_at = CASE
        WHEN excluded.last_run_at IS NULL OR excluded.last_run_at = ''
            THEN projects.last_run_at
        WHEN projects.last_run_at IS NULL OR projects.last_run_at = ''
            THEN excluded.last_run_at
        WHEN excluded.last_run_at > projects.last_run_at
            THEN excluded.last_run_at
        ELSE projects.last_run_at
    END
"""

#: Most recently *active* first: a project that has run recently is the one the
#: operator is coming back to. Projects that never ran sort after every one
#: that has - a NULL `last_run_at` would otherwise win a DESC sort in SQLite -
#: and within each group newest first. Lexicographic order is chronological
#: here because every timestamp written is ISO 8601 with a UTC offset.
_ORDERED = """
SELECT repo, objective, stack, verify, created_at, last_run_at FROM projects
ORDER BY (last_run_at IS NULL OR last_run_at = '') ASC,
         last_run_at DESC,
         created_at DESC,
         repo ASC
"""


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _parse(value: Any) -> dt.datetime | None:
    """Forgiving, like `artifacts._parse`: an unreadable timestamp costs one
    run its vote in the migration, not the migration."""
    if value in (None, ""):
        return None
    try:
        moment = dt.datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=dt.timezone.utc)


def validate_repo(repo: str) -> str:
    if not REPO_RE.match(repo or ""):
        raise ProjectError(f"repo must be 'owner/name', got {repo!r}",
                           fix="e.g. kamyar-finlex/wallet-tracker-service")
    return repo


@dataclass
class ProjectStore:
    """One repository per row, upserted, ordered by recency. The console's
    project memory.

    `path` and `runs_root` are the test seams: a test hands in paths under
    `tmp_path` and never touches the real database or the real run
    directories. The defaults are the real thing - the database beside the
    artifacts root, seeded from the runs inside it.

    Schema creation and the migration run once per store instance, lazily,
    before the first operation - not at construction, so that building a
    `Console` (which carries a store by default factory) writes nothing to
    disk until a projects route or a run start actually needs the table.
    """

    path: Path = field(default_factory=lambda: artifacts_root().parent / "projects.sqlite")
    runs_root: Path | None = None
    _ready: bool = field(default=False, repr=False)
    #: Guards only the one-time ensure. The operations themselves need no lock
    #: - see the module docstring on connection-per-operation.
    _once: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- reading and writing ---------------------------------------------

    def list(self) -> list[dict[str, Any]]:
        """Every project, most recently active first."""
        with self._session() as conn:
            return [dict(row) for row in conn.execute(_ORDERED)]

    def get(self, repo: str) -> dict[str, Any] | None:
        with self._session() as conn:
            row = conn.execute(
                "SELECT repo, objective, stack, verify, created_at, last_run_at "
                "FROM projects WHERE repo = ?", (repo,)
            ).fetchone()
        return None if row is None else dict(row)

    def upsert(
        self,
        repo: str,
        objective: str = "",
        stack: str = "",
        verify: str = "",
        *,
        created_at: str | None = None,
        last_run_at: str | None = None,
    ) -> dict[str, Any]:
        """Insert or update one project, and return the stored row.

        Empty strings mean "no news" - an existing non-empty objective, stack
        or verify survives them - and `last_run_at` only moves forward. The
        repo shape is validated here rather than trusted from the caller,
        because a slug is about to become a primary key the page will render
        into links and forms.
        """
        validate_repo(repo)
        with self._session() as conn:
            conn.execute(_UPSERT, {
                "repo": repo,
                "objective": objective or "",
                "stack": stack or "",
                "verify": verify or "",
                "created_at": created_at or _now_iso(),
                "last_run_at": last_run_at,
            })
        stored = self.get(repo)
        assert stored is not None  # it was just written
        return stored

    def record_run(
        self, repo: str, *, objective: str = "", stack: str = "", verify: str = ""
    ) -> dict[str, Any]:
        """What a starting GitHub run files: the submitted values, and now as
        the moment of last activity. The seam `Console._swarm_start` calls."""
        return self.upsert(repo, objective, stack, verify, last_run_at=_now_iso())

    def submit(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """One `POST /projects` body, validated and stored.

        Here rather than in `Console` because "what makes a submitted project
        acceptable" is a decision, and the transport stays routing. The store's
        own upsert tolerates an empty objective (the migration needs it to);
        an *operator* saving a project without one has told the selector
        nothing worth selecting, so that is refused with the fix attached.
        """
        if not isinstance(values, Mapping):
            raise ProjectError("a project submission must be a JSON object",
                               fix='{"repo": "owner/name", "objective": "..."}')
        repo = str(values.get("repo") or "").strip()
        objective = str(values.get("objective") or "").strip()
        if not objective:
            raise ProjectError("a project needs an objective",
                               fix="describe what the swarm should accomplish there")
        return self.upsert(
            repo,
            objective,
            str(values.get("stack") or "").strip(),
            str(values.get("verify") or "").strip(),
        )

    # -- the once-per-store plumbing ---------------------------------------

    @contextlib.contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        """One operation's connection: opened here, committed and closed here.

        Rows come back as `sqlite3.Row` so the methods above can hand the page
        dictionaries without naming the columns twice.
        """
        self._ensure()
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure(self) -> None:
        """Create the schema and seed from the run artifacts, once per store.

        Under a lock because the handler is threaded and two first-requests
        must not both run the migration; idempotent anyway - the upsert
        semantics make a replayed seed a no-op - so a second console process
        against the same file is harmless too.
        """
        with self._once:
            if self._ready:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = self._connect()
            try:
                conn.execute(_SCHEMA)
                self._seed(conn)
                conn.commit()
            finally:
                conn.close()
            self._ready = True

    def history(self, repo: str) -> list[dict[str, Any]]:
        """One project's prompts, newest first - each recorded run is one.

        The history is *derived from the run artifacts*, never stored twice:
        every run writes the prompt it was fired with into its own
        `run.json`, so the list of prompts a project has received is the list
        of its runs, and firing another run is how a prompt is added. A
        second copy in this database would drift from the directory the
        moment a run was started from a terminal.

        `finished` is whether the run wrote its `summary.json` - the same
        evidence `RunView.complete` reads - so the page can tell a prompt
        whose run ended from one that is live or was killed.
        """
        validate_repo(repo)
        prompts = [
            {
                "run_id": str(payload.get("run_id") or directory.name),
                "objective": str(payload.get("objective") or ""),
                "stack": str(payload.get("stack") or ""),
                "verify": str(payload.get("verify") or ""),
                "started_at": started.isoformat(),
                "finished": (directory / "summary.json").is_file(),
            }
            for directory, payload, started in self._recorded()
            if str(payload.get("repo") or "") == repo
        ]
        return sorted(prompts, key=lambda p: p["started_at"], reverse=True)

    def _recorded(self) -> Iterator[tuple[Path, dict[str, Any], dt.datetime]]:
        """Every readable, `owner/name`-shaped, dated run under the runs root.

        Reads each run directory's `run.json` directly rather than through
        `artifacts.read_run`, which also loads results, logs and metrics that
        neither the migration nor the history has a question for. Everything
        unreadable is skipped out loud on stderr - the scan must never be the
        reason a console fails to answer.
        """
        root = self.runs_root if self.runs_root is not None else artifacts_root()
        if not root.is_dir():
            return
        for directory in sorted(root.iterdir()):
            record = directory / RUN_FILE_NAME
            if not directory.is_dir() or not record.is_file():
                continue
            try:
                payload = json.loads(record.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                print(f"! projects: skipping {directory.name}: {exc}", file=sys.stderr)
                continue
            if not isinstance(payload, dict):
                continue
            if not REPO_RE.match(str(payload.get("repo") or "")):
                # A local run records a directory path here; not a project.
                continue
            started = _parse(payload.get("started_at"))
            if started is None:
                continue
            yield directory, payload, started

    def _seed(self, conn: sqlite3.Connection) -> None:
        """Fold the recorded runs into projects: one per distinct repo."""
        # repo -> (earliest, latest, latest run's identity payload)
        found: dict[str, tuple[dt.datetime, dt.datetime, dict[str, Any]]] = {}
        for _directory, payload, started in self._recorded():
            repo = str(payload.get("repo") or "")
            known = found.get(repo)
            if known is None:
                found[repo] = (started, started, payload)
            else:
                earliest, latest, newest = known
                found[repo] = (
                    min(earliest, started),
                    max(latest, started),
                    payload if started >= latest else newest,
                )

        for repo, (earliest, latest, newest) in found.items():
            conn.execute(_UPSERT, {
                "repo": repo,
                "objective": str(newest.get("objective") or ""),
                "stack": str(newest.get("stack") or ""),
                "verify": str(newest.get("verify") or ""),
                "created_at": earliest.isoformat(),
                "last_run_at": latest.isoformat(),
            })
