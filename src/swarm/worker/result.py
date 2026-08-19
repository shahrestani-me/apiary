"""What one attempt did, in a file the orchestrator can read without the container.

An exit code is one integer, and `docs/issue-contract.md` §4 spends it entirely
on the only question the reconciler must answer synchronously: does this
attempt count?

- `0` - the work is pushed and a PR is open. `claimed → review`.
- `1` - the task failed. The attempt is consumed; #22 re-readies the issue or
  gives up at the cap.
- `2` - the *infrastructure* failed, so the task never really ran. The attempt
  is **not** consumed, and repeated 2s halt the run instead of retrying it.

Conflating 1 and 2 is the failure this module exists to prevent, and it is not
theoretical: an Ollama that stops answering makes every worker die having done
no work, and a system that reads those deaths as task failures burns three
attempts on every issue in the ledger before a human notices anything is wrong.
The run then reports a repository's worth of impossible tasks, which is exactly
the wrong diagnosis.

**Why a file as well as the code.** The integer answers "retry?" and nothing
else. Everything a human or a run summary needs afterwards - which command was
the gate, what it said, which files were written, how long it took - lives in
one process, and that process is a container that is deleted seconds later
(`docs/architecture-v2.md`: "containers are cattle"). So the worker writes it
down before it exits, into the artifacts directory the run mounts, and the
record outlives the container that produced it. A run summary is then a
directory listing plus `json.loads`, which is what makes it reconstructible on
a different machine, after a crash, or a week later.

**Two writers, deliberately.** The worker writes its own record: it is the only
process that ever saw the verify output. But a worker killed by #19's wall
clock never reaches that line, and neither does one the kernel OOM-kills or one
whose image will not start. Those attempts happened and would otherwise vanish
from the artifacts entirely - the summary would show three attempts where five
were made, which reads as a healthier run than it was. So the container manager
synthesises a record from what it can see from outside (the exit code, the
container's captured log tail) and marks it `synthesised`, so nobody mistakes a
guess for the worker's own testimony.

`synthesise_timeout` is written against #15's actual behaviour rather than a
hoped-for one: `ContainerManager.wait` raises `ContainerTimeout` after stopping
the container, so the caller is already in an `except` block holding the handle
and can still call `dispose` for the logs. Nothing here needs a hook inside the
manager, and nothing here imports it - the `Container` protocol below names the
three attributes a `Handle` happens to have, which keeps the dependency
pointing from the orchestrator into the worker package and not back.

**A timeout kill is a `1`, not a `2`.** It is tempting to call it
infrastructure - the task was cut off rather than judged - but `run_verify` in
`entrypoint.py` already settled the same argument the other way: an attempt
that hangs and is not charged for hanging can hang again for free, forever, and
`docs/issue-contract.md` §5 is explicit that a counter which can fail to bound
retries is worse than one that over-counts. Unknown exit codes (a `137` from
the OOM killer, a `143` from a stop) are charged for the same reason: giving up
early puts a human in front of the problem, and looping forever does not.

**Reading is forgiving; writing is atomic.** The write goes to a temporary file
in the same directory and is renamed into place, because the process doing it
is the one being killed on a timer and half a JSON object is worse than none.
Reading ignores keys it does not know (so a newer worker's record still loads)
and skips files it cannot parse rather than failing the whole summary, because
one corrupt artifact must not cost a human the other forty.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .entrypoint import (
    EXIT_INFRASTRUCTURE,
    EXIT_OK,
    EXIT_TASK_FAILED,
    OUTPUT_TAIL_CHARS,
    WorkerResult,
)

#: Bumped when a field changes meaning, never when one is added: readers ignore
#: unknown keys, so additive change needs no version and a reader that sees a
#: higher number knows it is looking at a record it may misread.
SCHEMA_VERSION = 1

#: Where the worker writes, inside the container. The run's artifacts directory
#: (`Run.artifacts_dir`, #29) is mounted here; `APIARY_RESULT_DIR` overrides it
#: so the same code path works when a human runs a worker on a laptop.
RESULT_DIR_ENV = "APIARY_RESULT_DIR"
DEFAULT_RESULT_DIR = "/artifacts"

#: One file per attempt, not one per issue: a retry must not overwrite the
#: evidence of what the previous attempt did, which is the only place the reason
#: for the retry is written down.
RESULT_GLOB = "issue-*-attempt-*.json"

#: What each exit code is called in a summary and in the record itself. The
#: name is written to the file even though it is derivable, because the first
#: thing anyone does with an artifacts directory is grep it.
OUTCOMES: Mapping[int, str] = {
    EXIT_OK: "pr-open",
    EXIT_TASK_FAILED: "task-failed",
    EXIT_INFRASTRUCTURE: "infrastructure",
}

#: Any code the protocol does not define - the OOM killer's 137, a stop's 143,
#: a Python traceback's 1-shaped cousins. Charged to the attempt budget; see the
#: module docstring.
UNKNOWN_OUTCOME = "unknown"


class ResultError(RuntimeError):
    """A result file could not be written, read, or made sense of."""


class Container(Protocol):
    """The three things a synthesised record needs from a live container.

    Structural rather than an import of `containers.manager.Handle`: the worker
    package is what runs *inside* the container and must not depend on the
    module that spawns it. A `Handle` satisfies this as it stands.
    """

    id: str
    run_id: str
    issue: int | None
    #: Which image this container was created from. `Handle` already carries
    #: it, so nothing had to grow a field - it simply was not recorded, and
    #: once #99 picks the image per task it is the only place the answer lives.
    image: str = ""


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ResultRecord:
    """One attempt at one issue, as the orchestrator will remember it.

    Every field is either something only the worker knew or something the
    summary cannot be rebuilt without. `run_id` and `attempt` are here rather
    than inferred from the path because a record that has been copied into an
    issue comment, a bug report or a different directory must still say which
    run and which attempt it belongs to.
    """

    run_id: str
    issue: int
    attempt: int
    exit_code: int
    verify_command: str = ""
    verify_output: str = ""
    reason: str = ""
    task_id: str = ""
    repo: str = ""
    branch: str = ""
    commit: str | None = None
    written: tuple[str, ...] = ()
    #: Files the attempt deleted (empty-content edits - `worker.edit.apply_edits`).
    #: Additive with a default, like `image`: an old record loads with `()`,
    #: and folding these into `written` instead would make a summary claim the
    #: attempt wrote files that no longer exist.
    deleted: tuple[str, ...] = ()
    #: The container image this attempt ran in. Additive with a default, so no
    #: schema bump: a record written before this field existed reads back as
    #: `ResultRecord(image="")` rather than failing to load, and #99 makes the
    #: image vary per task - at which point "which image produced this result"
    #: stops being answerable from anything else in the directory.
    image: str = ""
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    synthesised: bool = False
    schema: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalise here rather than in the constructors, so nothing escapes it.

        Three invariants, and each has a caller that would otherwise break one:
        JSON has no tuples, a synthesised record is built from a container log
        that can be a quarter of a megabyte, and timestamps arrive naive from
        anyone who reached for `datetime.now()`.
        """
        object.__setattr__(self, "issue", int(self.issue))
        object.__setattr__(self, "attempt", int(self.attempt))
        object.__setattr__(self, "exit_code", int(self.exit_code))
        object.__setattr__(self, "written", tuple(str(path) for path in self.written))
        object.__setattr__(self, "deleted", tuple(str(path) for path in self.deleted))
        object.__setattr__(self, "verify_output", tail(self.verify_output))
        object.__setattr__(self, "started_at", _utc(self.started_at))
        object.__setattr__(self, "finished_at", _utc(self.finished_at))

    # --- what the reconciler reads --------------------------------------

    @property
    def outcome(self) -> str:
        return OUTCOMES.get(self.exit_code, UNKNOWN_OUTCOME)

    @property
    def consumes_attempt(self) -> bool:
        """The whole point of three codes rather than two.

        Only a clean infrastructure failure is free. Everything else - a failed
        gate, a timeout kill, an exit code nobody defined - costs the attempt,
        because the alternative is a task that can fail for free forever
        (`docs/issue-contract.md` §5).
        """
        return self.exit_code != EXIT_INFRASTRUCTURE

    @property
    def duration_s(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    def summary(self) -> str:
        elapsed = "" if self.duration_s is None else f" in {self.duration_s:.1f}s"
        mark = " (synthesised)" if self.synthesised else ""
        return (
            f"issue #{self.issue} attempt {self.attempt}: "
            f"{self.outcome} (exit {self.exit_code}){elapsed}{mark}"
        )

    # --- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """`outcome`, `consumes_attempt` and `duration_s` are written but never read.

        They are derived, so re-reading them would let a hand-edited file
        disagree with itself. They are written because the file's other audience
        is a human with `less`, to whom `"exit_code": 2` alone says nothing.
        """
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "issue": self.issue,
            "attempt": self.attempt,
            "task_id": self.task_id,
            "repo": self.repo,
            "branch": self.branch,
            "exit_code": self.exit_code,
            "outcome": self.outcome,
            "consumes_attempt": self.consumes_attempt,
            "reason": self.reason,
            "verify_command": self.verify_command,
            "verify_output": self.verify_output,
            "written": list(self.written),
            "deleted": list(self.deleted),
            "image": self.image,
            "commit": self.commit,
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "duration_s": self.duration_s,
            "synthesised": self.synthesised,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResultRecord:
        """Unknown keys are ignored, derived keys are recomputed. See the module docstring."""
        try:
            return cls(
                run_id=str(payload.get("run_id", "")),
                issue=int(payload["issue"]),
                attempt=int(payload.get("attempt", 0)),
                exit_code=int(payload["exit_code"]),
                verify_command=str(payload.get("verify_command", "")),
                verify_output=str(payload.get("verify_output", "")),
                reason=str(payload.get("reason", "")),
                task_id=str(payload.get("task_id", "")),
                repo=str(payload.get("repo", "")),
                branch=str(payload.get("branch", "")),
                commit=payload.get("commit") or None,
                written=tuple(payload.get("written") or ()),
                deleted=tuple(payload.get("deleted") or ()),
                # Additive, so absent is empty rather than an error: a record
                # written before this field existed must still load, which is
                # what "no schema bump" means in practice.
                image=str(payload.get("image", "")),
                started_at=_parse(payload.get("started_at")),
                finished_at=_parse(payload.get("finished_at")),
                synthesised=bool(payload.get("synthesised", False)),
                schema=int(payload.get("schema", SCHEMA_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ResultError(f"not a result record: {exc}") from exc


def tail(text: str, limit: int = OUTPUT_TAIL_CHARS) -> str:
    """The last `limit` characters, with a line saying what was dropped.

    The same bound `entrypoint.run_verify` already applies, restated here
    because a synthesised record's evidence is a container log rather than
    verify output and arrives at #15's `MAX_LOG_CHARS` - two orders of
    magnitude past what belongs in a per-attempt artifact.
    """
    if not text or len(text) <= limit:
        return text or ""
    return f"[apiary] {len(text) - limit} earlier characters elided\n{text[-limit:]}"


# --------------------------------------------------------------------------
# Building one
# --------------------------------------------------------------------------


def from_worker(
    result: WorkerResult,
    *,
    run_id: str,
    attempt: int,
    started_at: dt.datetime | None = None,
    finished_at: dt.datetime | None = None,
    exit_code: int | None = None,
    reason: str | None = None,
    image: str = "",
) -> ResultRecord:
    """The worker's own testimony, which is the authoritative kind.

    `exit_code` defaults to `result.exit_code` and is overridable for the one
    caller that has better information: `main` returns `2` for a failure raised
    *around* a run, and a record built after that should not claim the run's
    own verdict.
    """
    return ResultRecord(
        run_id=run_id,
        issue=result.issue,
        attempt=attempt,
        exit_code=result.exit_code if exit_code is None else exit_code,
        verify_command=result.verify_command,
        verify_output=result.verify_output,
        reason=reason if reason is not None else _worker_reason(result),
        task_id=result.task_id,
        repo=result.repo,
        branch=result.branch,
        commit=result.commit,
        written=result.written,
        deleted=result.deleted,
        image=image,
        started_at=started_at,
        finished_at=finished_at or dt.datetime.now(dt.timezone.utc),
    )


def _worker_reason(result: WorkerResult) -> str:
    if result.passed and result.commit:
        return "verified and committed"
    if result.passed:
        return "verified, but the edits changed nothing there was a commit to make"
    if not result.written and not result.deleted:
        return "no edit landed inside the declared file set"
    return "the verify command failed"


def synthesise(
    container: Container,
    *,
    exit_code: int,
    reason: str,
    attempt: int,
    started_at: dt.datetime | None = None,
    finished_at: dt.datetime | None = None,
    logs: str = "",
    verify_command: str = "",
    task_id: str = "",
    repo: str = "",
) -> ResultRecord:
    """A record for an attempt whose worker never wrote one.

    The container's captured log tail goes into `verify_output` rather than a
    field of its own: it is the same slot in a summary - the best evidence there
    is of what the attempt was doing - and `synthesised` already tells a reader
    it is not the gate's own words. A field only synthesised records ever fill
    would need every consumer to know about both.
    """
    if container.issue is None:
        # The label is written at `docker create` and read back off the daemon,
        # so a handle without one is a container this system did not spawn -
        # filing it under an issue would be a guess about somebody else's work.
        raise ResultError(f"container {container.id[:12]} carries no issue label")
    return ResultRecord(
        run_id=container.run_id,
        issue=container.issue,
        attempt=attempt,
        exit_code=exit_code,
        verify_command=verify_command,
        verify_output=logs,
        reason=reason,
        task_id=task_id,
        repo=repo,
        image=getattr(container, "image", "") or "",
        started_at=started_at,
        finished_at=finished_at or dt.datetime.now(dt.timezone.utc),
        synthesised=True,
    )


def synthesise_timeout(
    container: Container,
    *,
    attempt: int,
    timeout_s: float,
    started_at: dt.datetime | None = None,
    finished_at: dt.datetime | None = None,
    logs: str = "",
    task_id: str = "",
    repo: str = "",
) -> ResultRecord:
    """The #19 wall clock fired. Exit 1, and the attempt is charged.

    Call it from the `except ContainerTimeout` branch around
    `ContainerManager.wait`, after `dispose` has handed back the logs:

        try:
            code = manager.wait(handle)
        except ContainerTimeout:
            record = synthesise_timeout(
                handle, attempt=n, timeout_s=manager.timeout_s, logs=manager.dispose(handle)
            )
            write_result(record, artifacts, overwrite=False)

    `overwrite=False` because a worker that wrote its record and *then* hung on
    a push has already said something truer than this.
    """
    return synthesise(
        container,
        exit_code=EXIT_TASK_FAILED,
        reason=f"killed after {timeout_s:g}s by the run's wall clock; no result of its own",
        attempt=attempt,
        started_at=started_at,
        finished_at=finished_at,
        logs=logs,
        task_id=task_id,
        repo=repo,
    )


# --------------------------------------------------------------------------
# Disk
# --------------------------------------------------------------------------


def result_dir(default: str | Path | None = None) -> Path:
    """Where result files live for this process. Environment first."""
    return Path(os.environ.get(RESULT_DIR_ENV) or default or DEFAULT_RESULT_DIR)


def record_path(directory: str | Path, issue: int, attempt: int) -> Path:
    """`<dir>/issue-<n>-attempt-<k>.json`, the only name a reader has to know."""
    return Path(directory) / f"issue-{int(issue)}-attempt-{int(attempt)}.json"


def has_result(directory: str | Path, issue: int, attempt: int) -> bool:
    return record_path(directory, issue, attempt).exists()


def next_attempt(directory: str | Path, issue: int) -> int:
    """The first attempt index no record file for `issue` occupies yet.

    "Free" means one past the highest taken, never a gap below it: the
    reconciler discards any record whose attempt is behind the ledger's counter
    as stale (`record.attempt >= entry.attempt`), so a late record filed into a
    low gap would be a record nobody ever acts on - which is exactly the wedge
    this function exists to avoid. An empty or absent directory answers 0, the
    honest number for a first attempt.
    """
    pattern = re.compile(rf"^issue-{int(issue)}-attempt-(\d+)\.json$")
    taken = [
        int(match.group(1))
        for path in Path(directory).glob(f"issue-{int(issue)}-attempt-*.json")
        if (match := pattern.match(path.name))
    ]
    return max(taken, default=-1) + 1


def write_result(
    record: ResultRecord,
    directory: str | Path | None = None,
    *,
    overwrite: bool = True,
) -> Path:
    """Write one record, atomically, and never over another. Returns the path used.

    Temporary file plus `os.replace`, because the process most likely to be
    writing this is the one being torn down: a reader that finds a truncated
    object cannot tell it from a corrupt one, and the rename means a result file
    either exists whole or does not exist.

    **No call replaces an existing record file, whatever the arguments say.** A
    record is the only testimony its attempt leaves behind - the container is
    deleted seconds later - so overwriting one destroys history that cannot be
    rebuilt, and it was observed live doing exactly that: an exception-path
    record that guessed attempt 0 replaced the real first attempt's file. When
    the target name is taken, `overwrite=False` keeps the existing file and
    writes nothing, because the caller is saying what is already there is truer
    (a synthesised record must never displace the worker's own); the default
    writes anyway, under the next free `issue-N-attempt-*.json` index. Only the
    filename moves in that case - the record inside still says which attempt it
    testifies about, because a reader is told to trust the field, not the path.
    """
    where = directory if directory is not None else result_dir()
    target = record_path(where, record.issue, record.attempt)
    if target.exists():
        if not overwrite:
            return target
        target = record_path(where, record.issue, next_attempt(where, record.issue))
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(record.to_dict(), indent=2, sort_keys=False) + "\n"
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=str(target.parent), prefix=f"{target.stem}.", suffix=".tmp", delete=False
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        os.replace(temporary, target)
    except OSError as exc:
        raise ResultError(f"writing {target}: {exc}") from exc
    return target


def read_result(path: str | Path) -> ResultRecord:
    """One record from disk. Raises `ResultError` on anything unreadable."""
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ResultError(f"reading {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResultError(f"{path}: expected a JSON object, got {type(payload).__name__}")
    return ResultRecord.from_dict(payload)


def load_results(directory: str | Path) -> tuple[ResultRecord, ...]:
    """Every readable record in a directory, ordered by issue then attempt.

    Unreadable ones are reported on stderr and skipped. The alternative - one
    truncated file raising out of the summary - loses the other thirty-nine
    records to protect nothing, and the run this happened to is precisely the
    run somebody is trying to understand.

    The `.tmp` files `write_result` renames from are invisible here: they do not
    match the glob, so a partial write in flight cannot be read as a result.
    """
    records: list[ResultRecord] = []
    for path in sorted(Path(directory).glob(RESULT_GLOB)):
        try:
            records.append(read_result(path))
        except ResultError as exc:
            print(f"! skipping {path.name}: {exc}", file=sys.stderr)
    return tuple(sorted(records, key=lambda record: (record.issue, record.attempt)))


def report(
    result: WorkerResult,
    *,
    run_id: str,
    attempt: int | None,
    directory: str | Path | None = None,
    started_at: dt.datetime | None = None,
    finished_at: dt.datetime | None = None,
    exit_code: int | None = None,
    reason: str | None = None,
    image: str = "",
) -> Path:
    """Build the worker's record and write it. One call, because it is one act.

    This is the line the worker process runs last. It is separate from
    `entrypoint.main` for the reason `_publish` is: `docs/issue-contract.md`
    makes the file set of an issue a contract, and #18 does not own
    `entrypoint.py`.

    `attempt=None` means the worker died before the contract - the only place
    the attempt number is written down - was readable. The record still has to
    carry *some* number: the number names the file, and the reconciler discards
    any record whose attempt is behind the ledger's counter as stale. So the
    next free index on disk is used. That number may be wrong, and wrong is the
    lesser evil said out loud: a wrong-but-fresh attempt keeps the observation
    alive and clobbers no earlier record, where the previous behaviour - a
    hardcoded 0 - silently destroyed the real first attempt's file and left a
    live run showing "in flight" forever against a container that had exited.
    """
    where = Path(directory) if directory is not None else result_dir()
    if attempt is None:
        attempt = next_attempt(where, result.issue)
    record = from_worker(
        result,
        run_id=run_id,
        attempt=attempt,
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
        reason=reason,
        image=image,
    )
    return write_result(record, where)


# --------------------------------------------------------------------------
# The run summary
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSummary:
    """What a whole run did, derived from its artifacts and nothing else.

    No GitHub call and no live container: this is what can still be answered
    about a run whose orchestrator died, on a machine that never dispatched it.
    """

    run_id: str
    records: tuple[ResultRecord, ...]

    @property
    def attempts(self) -> int:
        return len(self.records)

    @property
    def counts(self) -> dict[str, int]:
        """Attempts per outcome, in the protocol's order then the rest."""
        tally: dict[str, int] = {}
        for record in self.records:
            tally[record.outcome] = tally.get(record.outcome, 0) + 1
        order = [*OUTCOMES.values(), UNKNOWN_OUTCOME]
        return {name: tally[name] for name in order if name in tally}

    @property
    def consumed(self) -> int:
        """Attempts that cost budget. The gap from `attempts` is the infrastructure tax."""
        return sum(1 for record in self.records if record.consumes_attempt)

    @property
    def infrastructure(self) -> tuple[ResultRecord, ...]:
        """Every attempt that failed for reasons the task cannot fix.

        More than a handful of these in one run is the signal to stop dispatching
        and look at the host rather than at the issues.
        """
        return tuple(r for r in self.records if r.exit_code == EXIT_INFRASTRUCTURE)

    @property
    def latest(self) -> dict[int, ResultRecord]:
        """Issue number -> its highest-numbered attempt. Where each task stands."""
        newest: dict[int, ResultRecord] = {}
        for record in self.records:
            current = newest.get(record.issue)
            if current is None or record.attempt >= current.attempt:
                newest[record.issue] = record
        return newest

    def text(self) -> str:
        header = f"run {self.run_id or '(unknown)'}: {self.attempts} attempt(s), {self.consumed} consumed"
        tally = ", ".join(f"{count} {name}" for name, count in self.counts.items()) or "nothing"
        lines = [header, f"  {tally}"]
        for issue in sorted(self.latest):
            record = self.latest[issue]
            lines.append(f"  {record.summary()}  {record.reason}".rstrip())
        return "\n".join(lines)


def summarise(records: Iterable[ResultRecord], *, run_id: str | None = None) -> RunSummary:
    """Fold records into a summary. Pure; `load_results` is the half that touches disk."""
    ordered: Sequence[ResultRecord] = tuple(
        sorted(records, key=lambda record: (record.issue, record.attempt))
    )
    resolved = run_id if run_id is not None else (ordered[0].run_id if ordered else "")
    return RunSummary(run_id=resolved, records=tuple(ordered))


def summarise_dir(directory: str | Path, *, run_id: str | None = None) -> RunSummary:
    """The whole promise of this module in one call: a directory in, a run summary out."""
    return summarise(load_results(directory), run_id=run_id)


# --------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------


def _utc(moment: dt.datetime | None) -> dt.datetime | None:
    """Aware UTC, always. A naive timestamp is read as local time, once.

    Records are compared across a container (UTC by convention) and a host that
    may be anything, and a summary that sorts two attempts by a mixture of the
    two is wrong in a way nobody spots.
    """
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=dt.timezone.utc)
    return moment.astimezone(dt.timezone.utc)


def _iso(moment: dt.datetime | None) -> str | None:
    return None if moment is None else moment.isoformat()


def _parse(value: Any) -> dt.datetime | None:
    if value in (None, ""):
        return None
    try:
        return _utc(dt.datetime.fromisoformat(str(value)))
    except ValueError as exc:
        raise ResultError(f"{value!r} is not an ISO-8601 timestamp") from exc
