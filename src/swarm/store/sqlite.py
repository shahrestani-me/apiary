"""The local backend: one SQLite file per project, and the only implementation.

ADR 0002 leaves the choice open - "SQLite or files under the artifacts root" -
and both were tried on paper. SQLite wins on the one requirement that is not
about convenience: **a corrupt store has to be recognisable as corrupt.** A
directory of JSON files degrades a byte at a time, and a reader looking at
thirty-nine good files and one truncated one has no way to tell "this task has
no judgment yet" from "this task's judgment is the thing that got truncated" -
the two are the same empty answer and one of them silently unbounds a retry
budget. A SQLite file answers that question itself: it either opens as a
database or it raises, the schema is either the one this build knows or it is
not, and a write either committed or did not. `worker/result.py` reaches the
opposite conclusion about its own directory for a reason that does not apply
here - a result record is evidence, and losing one costs a line in a summary.

**Where the file goes, and why not under the run's artifacts directory.** ADR
0002 says "files under the run artifacts root". Built against, that turns out
to be wrong, and wrong in the precise way the ticket's own acceptance criteria
forbid: a run directory is per *run*, so a store inside one is empty at the
start of every run, and an empty store reads as "every task is on attempt 0
with no blocker". The retry budget would reset on each invocation and a task
could never exhaust one. The store is per *project*, outlives every run against
that project, and therefore lives beside the run tree rather than inside it -
its own variable and its own default, deriving nothing from `artifacts_root()`,
for the reason `artifacts.CONSOLE_ROOT_ENV` states: an operator who moves runs
to `/var/apiary/runs` has said nothing about where anything else goes.

**One file per project, not one table with a project column.** Two projects
share nothing - not a task ref namespace, not a retry budget, not a lifetime -
and the moment they share a file, deleting one project's history is a `DELETE`
somebody has to get right instead of an `rm`. It also keeps the remote backend
honest: an organization database will be one store per project reached over a
network, and a local layout that already looks like that is one the seam can
swap without a caller noticing.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from ..taskref import TaskRef
from .base import SCHEMA_VERSION, StoreCorrupt, StoreError, StoreMissing, TaskJudgement

#: Where per-project stores live. A **sibling** of the artifacts root with its
#: own variable, deriving nothing from it - see the module docstring.
STORE_DIR_ENV = "APIARY_STORE_DIR"
DEFAULT_STORE_DIR = ".swarm/store"

#: The two tables. `meta` exists so that an existing file can be *identified*
#: rather than guessed at: without it, a SQLite database written by something
#: else opens cleanly, answers "no such table: judgement", and the tempting
#: reading of that is "an empty store".
_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS judgement (
        ref        TEXT PRIMARY KEY,
        attempt    INTEGER NOT NULL,
        blocker    TEXT    NOT NULL DEFAULT '',
        streak     INTEGER,
        renewals   INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT    NOT NULL DEFAULT ''
    )
    """,
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def store_root(default: str | Path | None = None) -> Path:
    """The directory per-project stores live in. Environment first.

    The shape `artifacts.artifacts_root` and `worker.result.result_dir` already
    use, deliberately: one convention for "a path the operator may move", and
    a third spelling of it would be a third thing to get wrong.
    """
    return Path(os.environ.get(STORE_DIR_ENV) or default or DEFAULT_STORE_DIR)


def project_slug(project: str) -> str:
    """A filesystem-safe name for a project, from whatever names it.

    `owner/repo` today, an organization key tomorrow. Lowercased and reduced to
    `[a-z0-9-]` because the value is used as a filename on three operating
    systems, and it is *not* required to be reversible: the file is found by
    building the same slug from the same project name, never by reading a
    project name back out of a path.
    """
    slug = _SLUG_RE.sub("-", project.strip().lower()).strip("-")
    if not slug:
        raise StoreError(f"project {project!r} has no filesystem-safe name")
    return slug


def store_path(project: str, root: str | Path | None = None) -> Path:
    """This project's store file."""
    base = store_root() if root is None else Path(root)
    return base / f"{project_slug(project)}.sqlite3"


class SqliteTaskStore:
    """One project's judgments, in one SQLite file. Satisfies `TaskStore`.

    Opened through `open` rather than constructed: opening is where every
    interesting decision is - does the file exist, is it this project's, is its
    schema one this build understands - and a constructor that took a live
    connection would let a caller skip all three.
    """

    def __init__(self, connection: sqlite3.Connection, path: Path, project: str) -> None:
        self._connection: sqlite3.Connection | None = connection
        self.path = path
        self.project = project

    # --- opening ---------------------------------------------------------

    @classmethod
    def open(
        cls,
        project: str,
        *,
        root: str | Path | None = None,
        create: bool = True,
    ) -> SqliteTaskStore:
        """Open this project's store, or say precisely why it cannot be trusted.

        `create=False` is the strict reading of ADR 0002's "a store missing at
        startup fails with a named fix": nothing may be assumed about a store
        that is not there. A run passes `create=True`, because a first run
        against a project legitimately has none and refusing to start would
        make the very first invocation an error - and because the seeding path
        in `github/ledger.py` fills a new store from whatever the issues'
        legacy markers still say, so creating one resets nobody's budget. The
        strict form exists for a caller that knows a store *should* be there:
        a health check, an audit, a migration.

        Every failure raises. None of them degrades to an empty store; see
        `StoreCorrupt`.
        """
        target = Path(store_path(project, root))
        existed = target.exists()
        if not existed and not create:
            raise StoreMissing(
                f"no task store for {project!r} at {target}. "
                f"If you moved it, set {STORE_DIR_ENV} to the directory holding it; "
                "if this project has never run, let the run create one."
            )
        if not existed:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise StoreError(
                    f"cannot create the task store directory {target.parent}: {exc}. "
                    f"Point {STORE_DIR_ENV} at a writable directory."
                ) from exc

        try:
            connection = sqlite3.connect(str(target), isolation_level=None)
            connection.row_factory = sqlite3.Row
            # WAL because the orchestrator writes a row per finished attempt
            # while nothing else is reading, and `synchronous=FULL` because the
            # crash this store is written against is the one between the write
            # and the label that re-readies the task: a judgment that reached
            # the page cache and not the disk is a retry granted for free.
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            for statement in _SCHEMA:
                connection.execute(statement)
        except sqlite3.DatabaseError as exc:
            raise StoreCorrupt(
                f"the task store at {target} is not a readable database ({exc}). "
                "Move it aside and let the run create a new one - the attempt "
                "counters themselves are still on the tracker, so a fresh store "
                "reseeds from them - or restore the file from a backup. It is "
                "deliberately not recreated automatically: an empty store reads "
                "as a fresh retry budget for every task at once."
            ) from exc
        except OSError as exc:
            raise StoreError(f"cannot open the task store at {target}: {exc}") from exc

        store = cls(connection, target, project)
        try:
            store._check(existed)
        except StoreError:
            store.close()
            raise
        return store

    def _check(self, existed: bool) -> None:
        """Identify the file, or refuse it. Runs once, at open.

        Three questions, and each has a way of being answered wrongly that this
        exists to prevent: an unstamped file is somebody else's database whose
        missing `judgement` rows would read as an empty store; a newer schema
        is rows this build may misread, and misreading a retry budget unbounds
        it; a different project's file is one whose refs collide with this
        project's and whose budgets are not this project's to spend.
        """
        stamped = self._meta()
        if not stamped:
            if existed and self._row_count() == 0 and self._foreign_tables():
                raise StoreCorrupt(
                    f"{self.path} is a SQLite database that is not an apiary task "
                    f"store (it holds {', '.join(self._foreign_tables())}). "
                    f"Point {STORE_DIR_ENV} at a different directory, or move that "
                    "file aside."
                )
            self._stamp()
            return

        schema = stamped.get("schema", "")
        try:
            version = int(schema)
        except ValueError as exc:
            raise StoreCorrupt(
                f"{self.path} carries an unreadable schema stamp {schema!r}. "
                "Move it aside and let the run create a new one."
            ) from exc
        if version > SCHEMA_VERSION:
            raise StoreCorrupt(
                f"{self.path} was written by a newer apiary (store schema {version}, "
                f"this build understands {SCHEMA_VERSION}). Upgrade apiary, or point "
                f"{STORE_DIR_ENV} at a different directory - this build must not read "
                "rows it may misread, because a misread retry budget is an unbounded one."
            )

        held = stamped.get("project", "")
        if held and held != self.project:
            raise StoreCorrupt(
                f"{self.path} holds judgments for {held!r}, not {self.project!r}. "
                f"Task refs are not comparable across projects, so this store's retry "
                f"budgets are not this run's to spend. Point {STORE_DIR_ENV} elsewhere."
            )

    def _stamp(self) -> None:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        for key, value in (
            ("schema", str(SCHEMA_VERSION)),
            ("project", self.project),
            ("created_at", now),
        ):
            self._execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO NOTHING",
                (key, value),
            )

    def _meta(self) -> dict[str, str]:
        rows = self._execute("SELECT key, value FROM meta").fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def _row_count(self) -> int:
        row = self._execute("SELECT COUNT(*) AS n FROM judgement").fetchone()
        return 0 if row is None else int(row["n"])

    def _foreign_tables(self) -> tuple[str, ...]:
        rows = self._execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT IN ('meta', 'judgement') AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return tuple(sorted(str(row["name"]) for row in rows))

    # --- the seam --------------------------------------------------------

    def read(self) -> Mapping[TaskRef, TaskJudgement]:
        """Every judgment in this store, in one query. See `TaskStore.read`."""
        rows = self._execute(
            "SELECT ref, attempt, blocker, streak, renewals, updated_at FROM judgement"
        ).fetchall()
        held: dict[TaskRef, TaskJudgement] = {}
        for row in rows:
            value = str(row["ref"])
            if not value.strip():
                # `TaskRef` refuses a blank value, and it is right to: a blank
                # key silently collides with every other blank one. A row that
                # holds one is a row nothing wrote through this class.
                raise StoreCorrupt(
                    f"{self.path} holds a judgment with no task ref. Move it aside "
                    "and let the run create a new one."
                )
            held[TaskRef(value)] = TaskJudgement(
                ref=TaskRef(value),
                attempt=int(row["attempt"]),
                blocker=str(row["blocker"] or ""),
                streak=None if row["streak"] is None else int(row["streak"]),
                renewals=int(row["renewals"] or 0),
                updated_at=_parse(row["updated_at"]),
            )
        return held

    def write(self, judgement: TaskJudgement) -> None:
        """Persist one judgment. See `TaskStore.write` for the durability rule."""
        self._execute(
            "INSERT INTO judgement (ref, attempt, blocker, streak, renewals, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(ref) DO UPDATE SET "
            "attempt=excluded.attempt, blocker=excluded.blocker, streak=excluded.streak, "
            "renewals=excluded.renewals, updated_at=excluded.updated_at",
            (
                judgement.ref.value,
                int(judgement.attempt),
                judgement.blocker,
                None if judgement.streak is None else int(judgement.streak),
                int(judgement.renewals),
                _iso(judgement.updated_at),
            ),
        )

    def close(self) -> None:
        """Release the connection. Safe to call twice, and called on every exit path."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    # --- context manager -------------------------------------------------

    def __enter__(self) -> SqliteTaskStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def summary(self) -> str:
        """The startup line, in the shape the run's other policies print."""
        return f"task store: {self.path} ({self._row_count()} task(s) judged)"

    # --- plumbing --------------------------------------------------------

    def _execute(self, statement: str, parameters: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        """One statement, with every SQLite failure turned into a named one.

        A `sqlite3.DatabaseError` reaching a caller would be an implementation
        detail escaping the seam, and - worse - one that a caller might be
        tempted to catch and continue past. `StoreCorrupt` says out loud that
        continuing is not on offer.
        """
        if self._connection is None:
            raise StoreError(f"the task store at {self.path} is closed")
        try:
            return self._connection.execute(statement, parameters)
        except sqlite3.DatabaseError as exc:
            raise StoreCorrupt(f"the task store at {self.path} failed on a query: {exc}") from exc


def _iso(moment: dt.datetime | None) -> str:
    """Aware UTC, always - `worker/result.py`'s rule, for its reason.

    A store is read on a host that need not share a zone with the one that
    wrote it, and a naive timestamp is read as local time exactly once.
    """
    if moment is None:
        return ""
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=dt.timezone.utc)
    return aware.astimezone(dt.timezone.utc).isoformat()


def _parse(value: Any) -> dt.datetime | None:
    """Forgiving, and the only forgiving read in this module.

    `updated_at` steers no decision - it is what makes a store somebody is
    debugging legible - so an unreadable one costs a display field rather than
    the run. Every other column is load-bearing and is read strictly.
    """
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(str(value))
    except ValueError:
        return None
