"""One directory per run, holding everything needed to understand it afterwards.

`docs/architecture-v2.md` says the orchestrator "holds no irreplaceable state"
and that logs are "captured to the run artifact directory *before* removal".
This module is that directory: what is in it, who writes each part, and how it
is read back once the process that produced it is gone.

**The question the directory has to answer is not "did the run succeed".** It
is *where the run needed a human*. GitHub answers the first one on its own - a
merged PR is a merged PR - and answers the second one badly, because the
evidence is in containers that were deleted seconds after they exited and in an
orchestrator process that has since died. So a run directory holds four things:

    .swarm/runs/<run-id>/
      run.json                        who this run was, written at startup
      events.jsonl                    the orchestrator's structured log
      results/issue-<n>-attempt-<k>.json   the worker's own testimony (#18)
      logs/issue-<n>-<short-id>.log   what each container printed (#15, #20)
      summary.json                    the fold of all of it, written at the end

**JSON lines, not a human-readable log.** The line a human reads is generated
from the record rather than being the record, because the interesting questions
about a bad run are aggregations - how many API calls did cycle 3 make, how
much of the wall clock went into model swaps - and those are `json.loads` on a
structured stream and regular expressions on a prose one. Append-only and one
object per line means a killed orchestrator leaves a file that still parses up
to the last complete line, which is exactly the run somebody wants to read.

**Wall clock alone does not diagnose a slow run.** "The run took four hours" is
compatible with a bad model, a saturated inference queue, and a host that spent
the afternoon reading weights off disk - three different fixes. `config.py`
measures a model swap at 6.7 s and `orchestrator/dispatcher.py` reserves an
inference slot because of it, so the swap count, the swap time, the peak queue
depth and the per-cycle API call count are recorded per cycle. They are the
numbers that tell those three stories apart, and none of them can be recovered
after the fact.

**Redaction is a precondition, not a follow-up.** This directory is the single
most likely thing in the system to be pasted into an issue or a blog post, and
it is assembled from container output produced by LLM-generated code holding a
push token. Every string written here passes through
`containers.manager.Redactor` first - the same class, using the same
`SECRET_PATTERNS` and `SECRET_NAME_RE` that redact at the capture boundary,
because a second opinion here about what a secret looks like would drift from
the one that does the redacting. `audit()` then re-checks the finished
directory with `security.scan_artifacts`, so "no artifact contains a
credential" is a property that can be re-established after a bad run rather than
only asserted in CI.

**What this module deliberately does not do.**

- It does not mount anything. `ContainerManager.spawn` (#15) mounts no volume,
  so a result file written at `/artifacts` inside a container has nowhere on the
  host to land and dies with it. The mount belongs to whoever owns that call
  site - the dispatcher (#21) - so `mount_flags` and `worker_env` below are the
  two values that call site needs, and the wiring is its change to make, not
  this module's.
- It does not add CLI subcommands. `swarm runs` and `swarm show <run-id>` live
  in `cli.py`, which already reserves them in its parser comment; `list_runs`,
  `load_run`, `runs_text` and `show_text` here are exactly what those two
  subcommands print, so wiring them is an argparse change and no logic.
- It does not decide anything. Nothing in an artifacts directory feeds back
  into a dispatch or a label. It is evidence, and evidence that steers the run
  producing it is a second control plane.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .containers.manager import Handle, Redactor
from .run import Run, RunError, validate_run_id
from .security import Leak, scan_artifacts
from .worker.result import (
    DEFAULT_RESULT_DIR,
    RESULT_DIR_ENV,
    ResultRecord,
    RunSummary,
    summarise_dir,
    tail,
)

__all__ = [
    "ARTIFACTS_ROOT_ENV",
    "CONSOLE_ROOT_ENV",
    "CONTAINER_LOGGED",
    "CORPUS_MANIFEST_NAME",
    "CORPUS_SCHEMA",
    "CYCLE_FINISHED",
    "CYCLE_STARTED",
    "DEFAULT_ARTIFACTS_ROOT",
    "DEFAULT_CONSOLE_ROOT",
    "EVENT_LOG_NAME",
    "LOGS_DIR_NAME",
    "OBSERVED_LOG_NAME",
    "PR_CHECKS",
    "PR_MERGED",
    "PR_OPENED",
    "RESULTS_DIR_NAME",
    "RUN_FILE_NAME",
    "RUN_FINISHED",
    "RUN_STARTED",
    "SCHEMA_VERSION",
    "STATE_DIVERGENCE",
    "STATE_OVERRIDE",
    "STATE_SHADOW",
    "SUMMARY_FILE_NAME",
    "TASK_CLAIMED",
    "TASK_ELIGIBLE",
    "TASK_LANDED",
    "TASK_NEEDS_HUMAN",
    "TASK_RESULT",
    "ArtifactsError",
    "CycleMetrics",
    "DivergenceTally",
    "EventLog",
    "ObservationLog",
    "RunArtifacts",
    "RunMetrics",
    "RunView",
    "artifacts_root",
    "console_root",
    "HOST_ROOT_ENV",
    "host_path",
    "write_json",
    "list_runs",
    "load_run",
    "read_events",
    "read_run",
    "run_dir",
    "runs_text",
    "show_text",
]

#: Bumped when a field changes meaning, never when one is added - the same rule
#: `worker/result.py` states, for the same reason: readers here ignore keys they
#: do not know, so additive change costs nothing and a reader that meets a
#: higher number knows it may be misreading what it found.
SCHEMA_VERSION = 1

#: Where run directories live. `security.scan_artifacts` already names
#: `.swarm/runs/` as the thing kept forever by design; this is that path, and it
#: sits beside `.swarm/worktrees` from `config.py` so one directory holds
#: everything the swarm leaves on a developer's disk.
ARTIFACTS_ROOT_ENV = "APIARY_ARTIFACTS_DIR"
DEFAULT_ARTIFACTS_ROOT = ".swarm/runs"

#: Where console sessions live. A **sibling** of the artifacts root with its own
#: variable, deriving nothing from it: a console capture is not a run, and
#: `list_runs` skips directory names that are not well-formed run ids, so
#: nesting these under `.swarm/runs` would make them either invisible or
#: malformed. Deriving the path as `artifacts_root().parent / "console"` would
#: have been tidier to write and wrong to operate - an operator who moves runs
#: to `/var/apiary/runs` has said nothing about where console sessions go, and
#: silently relocating them is how a capture ends up somewhere nothing audits.
CONSOLE_ROOT_ENV = "APIARY_CONSOLE_DIR"
DEFAULT_CONSOLE_ROOT = ".swarm/console"

#: The four names inside a run directory. Constants rather than literals because
#: `swarm show` reads what the orchestrator wrote, and a typo in one of two
#: spellings produces an empty report rather than an error.
RUN_FILE_NAME = "run.json"
EVENT_LOG_NAME = "events.jsonl"
#: One line per shadowed cycle: the world that cycle read, and the labels the
#: control plane held for it (#146). Named here rather than in the test corpus
#: because it is now written by a live run: `tests/fixtures/corpus.py` imports
#: this constant, which is what makes "a recorded run drops in beside a
#: synthesised one with no code change" a fact rather than an intention.
OBSERVED_LOG_NAME = "observed.jsonl"
#: The manifest a corpus run needs beside the four files a run already writes.
#: Emitted with `origin: "recorded"` and **no declared divergences**, so the
#: replay harness refuses the run until a human has written the argument for
#: each one - `tests/fixtures/runs/README.md`'s rule, and the one part of
#: recording a machine must not do on somebody's behalf.
CORPUS_MANIFEST_NAME = "corpus.json"
#: The corpus format's schema number, owned here because the recorder stamps it
#: and `tests/fixtures/corpus.py` checks it. Two spellings would be worse than
#: none: the loader only refuses a number *greater* than its own, so a bump made
#: on one side alone would leave the recorder stamping the old number and the
#: loader silently reading new-meaning fields as old ones. Bumped when a field
#: in `observed.jsonl` changes meaning, never when one is added.
CORPUS_SCHEMA = 1
SUMMARY_FILE_NAME = "summary.json"
RESULTS_DIR_NAME = "results"
LOGS_DIR_NAME = "logs"

#: Container logs are `.log` and nothing else is, so a listing needs no
#: knowledge of the naming scheme below it.
LOG_GLOB = "*.log"

#: Above this share of inference time spent loading weights, the summary says so
#: in words. 0.2 is not a measurement; it is the point past which "the model was
#: slow" stops being the likeliest reading of a slow run, and a number a human
#: reads once beats a ratio they have to divide in their head.
SWAP_SHARE_NOTE = 0.2


class ArtifactsError(RuntimeError):
    """A run directory could not be written, found, or made sense of."""


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------

_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _utc(moment: dt.datetime | None) -> dt.datetime | None:
    """Aware UTC, always - the rule `worker/result.py` states and this obeys.

    A run directory is read on a host that need not share a zone with the one
    that wrote it, and two timestamps compared across a mixture of the two are
    wrong in a way nobody spots.
    """
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=dt.timezone.utc)
    return moment.astimezone(dt.timezone.utc)


def _iso(moment: dt.datetime | None) -> str | None:
    aware = _utc(moment)
    return None if aware is None else aware.isoformat()


def _parse(value: Any) -> dt.datetime | None:
    """Forgiving: an unreadable timestamp costs a field, not a whole run report."""
    if value in (None, ""):
        return None
    try:
        return _utc(dt.datetime.fromisoformat(str(value)))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------


#: The artifacts root **as the Docker daemon sees it**, when those are not the
#: same path. They are not the same path whenever the orchestrator is itself a
#: container: it reaches the daemon through a socket, so every `--volume` source
#: it names is resolved on the *host*, not inside itself. `/var/apiary/runs` is
#: real inside the orchestrator and absent on the host, so the daemon would
#: silently create an empty directory there and the worker's result file would
#: land somewhere nothing ever reads - a run that looks healthy and learns
#: nothing about what its workers did.
#:
#: Unset when the orchestrator runs on the host, where one filesystem serves
#: both and the translation is the identity.
HOST_ROOT_ENV = "APIARY_ARTIFACTS_HOST_ROOT"


def host_path(path: str | Path, *, root: str | Path | None = None) -> Path:
    """Rewrite a path under the artifacts root into the daemon's view of it.

    The identity when `APIARY_ARTIFACTS_HOST_ROOT` is unset, which is the host
    case. A path that is not under the artifacts root is returned unchanged:
    this translates one mount, and guessing about others would be worse than
    not trying.
    """
    inside = Path(path)
    host_root = os.environ.get(HOST_ROOT_ENV)
    if not host_root:
        return inside
    base = Path(root) if root is not None else artifacts_root()
    try:
        relative = inside.relative_to(base)
    except ValueError:
        return inside
    return Path(host_root) / relative


def artifacts_root(default: str | Path | None = None) -> Path:
    """The directory run directories live in. Environment first.

    Same shape as `worker.result.result_dir`, deliberately: one convention for
    "a path the operator may move", and the orchestrator and the worker halves
    of this system should not have two.
    """
    return Path(os.environ.get(ARTIFACTS_ROOT_ENV) or default or DEFAULT_ARTIFACTS_ROOT)


def console_root(default: str | Path | None = None) -> Path:
    """The directory console sessions live in. Environment first.

    Same shape as `artifacts_root`, and deliberately not defined in terms of it
    - see `CONSOLE_ROOT_ENV`.
    """
    return Path(os.environ.get(CONSOLE_ROOT_ENV) or default or DEFAULT_CONSOLE_ROOT)


def run_dir(run: Run | str, root: str | Path | None = None) -> Path:
    """This run's directory. Accepts a `Run` or an id that arrived from a human.

    `Run.artifacts_dir` (#33) does the join for the first case and
    `validate_run_id` guards the second, because the id in `swarm show <run-id>`
    is a string a human typed and `../../etc` is a valid string.
    """
    base = artifacts_root() if root is None else Path(root)
    if isinstance(run, Run):
        return run.artifacts_dir(base)
    return base / validate_run_id(run)


def _identity(text: str) -> str:
    """The no-op redactor, for a caller that has already redacted or has nothing."""
    return text


def _default_redactor(env: Mapping[str, str] | None = None) -> Redactor:
    """A redactor enrolled with this process's credentials, by variable name.

    `add_env` is what makes redaction automatic rather than remembered: the
    orchestrator's own `GITHUB_TOKEN` is registered without this module naming
    it, and a credential added to the environment later is covered without
    anyone editing this file.
    """
    redactor = Redactor()
    redactor.add_env(os.environ if env is None else env)
    return redactor


def _redacted(value: Any, redact: Callable[[str], str]) -> Any:
    """Every string reachable from `value`, redacted, structure preserved.

    Recursive rather than redacting the serialised JSON, so a secret split
    across an escape sequence by `json.dumps` cannot slip past a literal match.
    Anything that is not a JSON scalar or container is stringified *before*
    redaction - a `Path` or an exception carrying a token would otherwise reach
    the file through `json.dumps(default=str)` having been seen by nobody.
    """
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        return {redact(str(key)): _redacted(item, redact) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_redacted(item, redact) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact(str(value))


def write_json(
    target: Path,
    payload: Mapping[str, Any],
    *,
    redact: Callable[[str], str] = _identity,
    indent: int = 2,
) -> Path:
    """Write one JSON document atomically, redacting every string on the way in.

    Module-level because redaction in this file is **per writer, not per
    directory** - `EventLog.emit`, `ObservationLog.write`,
    `RunArtifacts._write_json` and `container_log` redact, and
    `worker/result.py` writes into the same tree without redacting at all (`tests/test_artifacts.py` asserts that gap). A
    fourth writer that hand-rolled `json.dumps` would inherit the gap rather
    than the guarantee, so there is one function to reach for and it takes the
    redactor as an argument rather than defaulting to something safe-looking.

    `ensure_ascii` is left at its default `True` on purpose: a model response
    containing U+2028 would otherwise reach the file as a raw line break, which
    `read_events` would split into two unparseable halves.
    """
    body = json.dumps(_redacted(dict(payload), redact), indent=indent, default=str) + "\n"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=str(target.parent), prefix=f"{target.stem}.", suffix=".tmp", delete=False
        ) as handle:
            handle.write(body)
            temporary = Path(handle.name)
        os.replace(temporary, target)
    except OSError as exc:
        raise ArtifactsError(f"writing {target}: {exc}") from exc
    return target


# --------------------------------------------------------------------------
# The event log
# --------------------------------------------------------------------------


@dataclass
class EventLog:
    """Append-only JSON lines, redacted on the way in.

    Opened per write rather than held open. It costs a syscall per event, which
    against a run that spends minutes in inference is nothing, and it buys two
    things worth more: `tail -f` on a live run shows every event that happened
    rather than every event that has been flushed, and a `SIGKILL`ed
    orchestrator leaves a file whose last line is either complete or absent.
    """

    path: Path
    redact: Callable[[str], str] = _identity
    clock: Callable[[], dt.datetime] = _now

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        """Write one event and return what was written, already redacted.

        The return value is what a caller prints. Printing the payload rather
        than the arguments means the terminal and the file cannot disagree, and
        it is the terminal that a human is looking at when a token leaks.
        """
        payload: dict[str, Any] = {"ts": _iso(self.clock()), "event": str(event)}
        payload.update(_redacted(dict(fields), self.redact))
        line = json.dumps(payload, default=str, sort_keys=False)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            raise ArtifactsError(f"writing {self.path}: {exc}") from exc
        return payload

    def events(self) -> tuple[dict[str, Any], ...]:
        return read_events(self.path)


@dataclass
class ObservationLog:
    """`observed.jsonl`: one recorded cycle per line, redacted on the way in.

    A sibling of `EventLog` rather than a use of it, because the two files have
    different readers and one of them is strict. `EventLog.emit` stamps every
    line with `ts` and `event`; `tests/fixtures/corpus.py` reads this file with
    a loader that **refuses** a line it cannot parse (a cycle silently dropped
    there is a divergence that silently stops being asserted), so the payload
    written here is exactly the object the loader expects and nothing else.

    Opened per write for `EventLog`'s reasons: a `SIGKILL`ed orchestrator leaves
    a file whose last line is complete or absent, which is the only shape a
    strict loader can survive.
    """

    path: Path
    redact: Callable[[str], str] = _identity

    def write(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Append one cycle. Returns what was written, already redacted."""
        recorded = _redacted(dict(payload), self.redact)
        line = json.dumps(recorded, default=str, sort_keys=False)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            raise ArtifactsError(f"writing {self.path}: {exc}") from exc
        return recorded


def read_events(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Every parseable event in a log, in the order it was written.

    A line that does not parse is reported on stderr and skipped, exactly as
    `worker.result.load_results` skips a corrupt record: the last line of a
    killed run's log is *expected* to be a half-written object, and refusing to
    read the file because of it loses the whole run to protect nothing.
    """
    events: list[dict[str, Any]] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise ArtifactsError(f"reading {path}: {exc}") from exc

    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except ValueError as exc:
            print(f"! skipping {Path(path).name}:{number}: {exc}", file=sys.stderr)
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return tuple(events)


# --------------------------------------------------------------------------
# What explains a slow run
# --------------------------------------------------------------------------


@dataclass
class CycleMetrics:
    """One reconcile cycle's cost, in the four currencies that differ.

    Wall clock is the fifth and is the one everybody already has. It does not
    distinguish a model that generated bad code slowly from a host that spent
    the cycle reading 17 GB of weights off disk, and those have different fixes
    (`config.py`: change the model, or stop swapping between two). So the
    inference time, the swap count, the swap time and the peak queue depth are
    recorded separately, and the API call count with them - a cycle that made
    four hundred GitHub calls was rate-limited, not slow.

    Mutable, and recorded into by the orchestrator as the cycle runs. The
    aggregate (`RunMetrics`) is frozen, because by then the numbers are history.
    """

    cycle: int
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    api_calls: int = 0
    inference_calls: int = 0
    inference_s: float = 0.0
    model_swaps: int = 0
    swap_s: float = 0.0
    peak_queue_depth: int = 0

    #: The model Ollama is believed to be holding. Seeded from the previous
    #: cycle rather than reset, because `OLLAMA_MAX_LOADED_MODELS=1` means the
    #: first orchestrator call of cycle 4 swaps out whatever the last worker
    #: call of cycle 3 loaded - a cost that belongs to a cycle boundary and
    #: would be invisible to a counter that started each cycle from nothing.
    loaded_model: str = ""

    # --- recording ------------------------------------------------------

    def api_call(self, count: int = 1) -> None:
        """One GitHub request. `count` for a paginated read that made several."""
        self.api_calls += max(int(count), 0)

    def inference(
        self,
        model: str,
        *,
        seconds: float = 0.0,
        load_s: float = 0.0,
        queued: int | None = None,
    ) -> bool:
        """One Ollama call. Returns whether it paid for a model swap.

        The count and the time are measured independently on purpose. The count
        comes from the sequence of models - a request for a model other than the
        resident one is a swap whether or not anybody timed it - and the time
        comes from `load_s`, which is Ollama's own `load_duration` for the call.
        Either can be available without the other: a caller that only knows
        which model it asked for still gets an honest count, and a caller that
        reports a load duration for a model already resident is describing an
        eviction, which cost the same seconds whatever the count says.
        """
        self.inference_calls += 1
        self.inference_s += max(float(seconds), 0.0)
        if queued is not None:
            self.queue(queued)

        swapped = bool(model) and model != self.loaded_model
        if swapped:
            self.model_swaps += 1
            self.loaded_model = model
        self.swap_s += max(float(load_s), 0.0)
        return swapped

    def queue(self, depth: int) -> None:
        """Note how many callers were waiting on an inference slot.

        Peak, not mean: two containers queueing against two slots for a moment
        is the design (`docs/architecture-v2.md`, second constraint), and it is
        the moment six were waiting that explains the afternoon.
        """
        self.peak_queue_depth = max(self.peak_queue_depth, int(depth))

    # --- reading --------------------------------------------------------

    @property
    def duration_s(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "duration_s": self.duration_s,
            "api_calls": self.api_calls,
            "inference_calls": self.inference_calls,
            "inference_s": round(self.inference_s, 3),
            "model_swaps": self.model_swaps,
            "swap_s": round(self.swap_s, 3),
            "peak_queue_depth": self.peak_queue_depth,
            "loaded_model": self.loaded_model,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CycleMetrics:
        """Tolerant: an event written by a newer orchestrator still folds."""
        try:
            return cls(
                cycle=int(payload.get("cycle", 0)),
                started_at=_parse(payload.get("started_at")),
                finished_at=_parse(payload.get("finished_at")),
                api_calls=int(payload.get("api_calls", 0)),
                inference_calls=int(payload.get("inference_calls", 0)),
                inference_s=float(payload.get("inference_s", 0.0)),
                model_swaps=int(payload.get("model_swaps", 0)),
                swap_s=float(payload.get("swap_s", 0.0)),
                peak_queue_depth=int(payload.get("peak_queue_depth", 0)),
                loaded_model=str(payload.get("loaded_model", "")),
            )
        except (TypeError, ValueError) as exc:
            raise ArtifactsError(f"not a cycle record: {exc}") from exc


@dataclass(frozen=True)
class RunMetrics:
    """Every cycle's numbers, folded. The paragraph `swarm show` prints."""

    cycles: tuple[CycleMetrics, ...] = ()

    @property
    def api_calls(self) -> int:
        return sum(cycle.api_calls for cycle in self.cycles)

    @property
    def inference_calls(self) -> int:
        return sum(cycle.inference_calls for cycle in self.cycles)

    @property
    def inference_s(self) -> float:
        return sum(cycle.inference_s for cycle in self.cycles)

    @property
    def model_swaps(self) -> int:
        return sum(cycle.model_swaps for cycle in self.cycles)

    @property
    def swap_s(self) -> float:
        return sum(cycle.swap_s for cycle in self.cycles)

    @property
    def peak_queue_depth(self) -> int:
        return max((cycle.peak_queue_depth for cycle in self.cycles), default=0)

    @property
    def busiest_cycle(self) -> CycleMetrics | None:
        """The cycle that made the most API calls, which is the one to look at."""
        return max(self.cycles, key=lambda cycle: cycle.api_calls, default=None)

    @property
    def swap_share(self) -> float | None:
        """Fraction of inference time spent loading weights, or `None`.

        A share rather than an addition: Ollama's `load_duration` is part of the
        request's own wall clock, so swap time is inside inference time and
        adding them would double-count the thing being explained.
        """
        if self.inference_s <= 0:
            return None
        return min(self.swap_s / self.inference_s, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_calls": self.api_calls,
            "inference_calls": self.inference_calls,
            "inference_s": round(self.inference_s, 3),
            "model_swaps": self.model_swaps,
            "swap_s": round(self.swap_s, 3),
            "peak_queue_depth": self.peak_queue_depth,
            "swap_share": None if self.swap_share is None else round(self.swap_share, 3),
            "cycles": [cycle.to_dict() for cycle in self.cycles],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunMetrics:
        """Only `cycles` is read back; every other key above is derived from it."""
        rows = payload.get("cycles") or ()
        return cls(tuple(CycleMetrics.from_dict(row) for row in rows if isinstance(row, Mapping)))

    @classmethod
    def from_events(cls, events: Iterable[Mapping[str, Any]]) -> RunMetrics:
        """Fold a run's event log, for a run that never wrote a summary.

        This is what "structured logging so a run can be replayed" buys: an
        orchestrator killed mid-run wrote no `summary.json`, and that run is
        precisely the one somebody is trying to understand.
        """
        return cls(
            tuple(
                CycleMetrics.from_dict(event)
                for event in events
                if event.get("event") == CYCLE_FINISHED
            )
        )

    def text(self) -> str:
        if not self.cycles:
            return "no cycles recorded"
        lines = [
            f"{len(self.cycles)} cycle(s), {self.api_calls} API call(s), "
            f"{self.inference_calls} inference call(s) in {self.inference_s:.1f}s",
            f"  {self.model_swaps} model swap(s) costing {self.swap_s:.1f}s, "
            f"peak inference queue depth {self.peak_queue_depth}",
        ]
        share = self.swap_share
        if share is not None and share >= SWAP_SHARE_NOTE:
            # The sentence this whole section exists for. Without it a reader
            # sees a big number for inference and concludes the model is slow.
            lines.append(
                f"  {share:.0%} of inference time went into loading weights, not "
                f"generating tokens - the host was swapping, not thinking"
            )
        busiest = self.busiest_cycle
        if busiest is not None and busiest.api_calls:
            lines.append(f"  busiest cycle {busiest.cycle}: {busiest.api_calls} API call(s)")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Event names
# --------------------------------------------------------------------------

#: Named because they are read back, not just written. `CYCLE_FINISHED` in
#: particular is the join between the writer and `RunMetrics.from_events`, and a
#: string literal in two files is a rename waiting to break a replay.
RUN_STARTED = "run.started"
RUN_FINISHED = "run.finished"
CYCLE_STARTED = "cycle.started"
CYCLE_FINISHED = "cycle.finished"
CONTAINER_LOGGED = "container.logged"

#: The per-task lifecycle (#141). A cycle is minutes long and everything worth
#: watching happens inside one, so the run-level names above cannot say when a
#: task became eligible, which container took it, or why it stopped. These can.
#:
#: **They speak apiary's own vocabulary, never the tracker's.** Every one of
#: them is keyed by the *task ref* - the slug `Transition.task_id` already
#: carries - and the states they name are apiary's own internal ones (ADR
#: 0001's, plus `blocked`), not the `swarm:*` labels that store them today. `events.jsonl` is
#: append-only and read back (`RunMetrics.from_events`, `console_external`), so
#: a run recorded now is still readable after epic #140 has removed the labels;
#: an issue number or a label name baked in here would have invalidated every
#: recorded run the day it was removed.
TASK_ELIGIBLE = "task.eligible"
TASK_CLAIMED = "task.claimed"
TASK_RESULT = "task.result"
PR_OPENED = "pr.opened"
PR_CHECKS = "pr.checks"
PR_MERGED = "pr.merged"
TASK_LANDED = "task.landed"
TASK_NEEDS_HUMAN = "task.needs_human"

#: The derived-state shadow window (#146). `state.shadow` is emitted once per
#: shadowed cycle whether or not anything disagreed, and that is not redundancy:
#: an event log with no `state.divergence` line in it is either a clean run or a
#: run with `APIARY_DERIVED_SHADOW=0`, and epic #140's go/no-go - ten runs with
#: no unexplained divergence - is not a gate at all if it cannot tell zero from
#: unmeasured. `state.divergence` names one task, one cycle and both states, in
#: apiary's own vocabulary on both sides, because #145's acceptance criteria
#: refuse a count: "the resolver agreed 94% of the time" is compatible with
#: every disagreement being on `needs-human`.
STATE_SHADOW = "state.shadow"
STATE_DIVERGENCE = "state.divergence"

#: The cutover (#147). `state.divergence` is sampled at the *end* of a cycle and
#: says the label and the resolver disagree; this one is sampled at the top,
#: before anything is decided, and says the orchestrator **acted on the
#: resolver's answer rather than the label's**. They are not the same claim and
#: they cannot be the same event: a divergence the cycle then repaired leaves no
#: `state.divergence` line at all, and "a human edited a label mid-run and the
#: swarm carried on" is precisely the thing #147 has to be able to show.
#: `kind` names the account when `orchestrator/authority.py` moved the answer off
#: the resolver's own verdict - one of ADR 0001's three non-derivable states -
#: and is empty when the label was simply stale.
STATE_OVERRIDE = "state.override"


# --------------------------------------------------------------------------
# Writing a run
# --------------------------------------------------------------------------


@dataclass
class RunArtifacts:
    """One run's directory, open for writing. Bound to a `Run` (#33).

    Everything written through here is redacted first, which is why the
    redactor is a field rather than a call each writer remembers to make: a
    contributor adding a fifth kind of artifact gets redaction by having used
    this object at all.
    """

    run: Run
    root: Path
    redact: Callable[[str], str] = _identity
    #: What `open` recorded into `run.json`, kept so `finish` can repeat it into
    #: `summary.json` without re-reading the file it just wrote.
    stack: str = ""
    verify: str = ""
    _cycles: list[CycleMetrics] = field(default_factory=list, repr=False)
    _loaded_model: str = field(default="", repr=False)

    @classmethod
    def open(
        cls,
        run: Run,
        *,
        root: str | Path | None = None,
        redact: Callable[[str], str] | None = None,
        env: Mapping[str, str] | None = None,
        stack: str = "",
        verify: str = "",
    ) -> RunArtifacts:
        """Create the directory, record who this run is, and say it started.

        `run.json` is written at startup rather than at the end because the run
        that most needs identifying is the one that never reached its end.

        `stack` and `verify` are recorded here rather than on `Run` because
        `run.py` is deliberately "small and inert" - an identity, not a
        description of the work - and because both are facts about *this
        directory's* run that a reader has no other way to recover. #87's
        success signal is a query over `.swarm/runs/*/summary.json` returning a
        non-Python run with merged PRs, and that query cannot be written
        against a file that records neither.
        """
        artifacts = cls(
            run=run,
            root=artifacts_root() if root is None else Path(root),
            redact=redact if redact is not None else _default_redactor(env),
            stack=stack,
            verify=verify,
        )
        for directory in (artifacts.path, artifacts.results_dir, artifacts.logs_dir):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ArtifactsError(f"creating {directory}: {exc}") from exc
        artifacts._write_json(
            artifacts.path / RUN_FILE_NAME,
            {
                "schema": SCHEMA_VERSION,
                "run_id": run.id,
                "repo": run.repo,
                "objective": run.objective,
                "started_at": _iso(run.started_at),
                "stack": stack,
                "verify": verify,
            },
        )
        artifacts.event(RUN_STARTED, repo=run.repo, objective=run.objective, stack=stack)
        return artifacts

    # --- layout ---------------------------------------------------------

    @property
    def path(self) -> Path:
        return run_dir(self.run, self.root)

    @property
    def results_dir(self) -> Path:
        """Where #18's result files land, and the *only* part a container writes.

        A subdirectory rather than the run directory itself, because the mount
        that makes it reachable from inside a container is read-write and the
        container runs LLM-generated code: mounting the run directory would put
        the orchestrator's own event log and every other issue's result under
        that code's pen.
        """
        return self.path / RESULTS_DIR_NAME

    @property
    def logs_dir(self) -> Path:
        return self.path / LOGS_DIR_NAME

    @property
    def events_path(self) -> Path:
        return self.path / EVENT_LOG_NAME

    @property
    def log(self) -> EventLog:
        return EventLog(self.events_path, redact=self.redact)

    @property
    def observed_path(self) -> Path:
        return self.path / OBSERVED_LOG_NAME

    @property
    def observations(self) -> ObservationLog:
        return ObservationLog(self.observed_path, redact=self.redact)

    def observed(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Record one shadowed cycle's world and control plane (#146).

        The seam `Reconciler.record` is wired to, mirroring `event` exactly so
        that a caller holding a `RunArtifacts` gets redaction by having used
        this object at all.

        **Why a live run writes this at all.** Four of the five files a replay
        corpus run needs - `run.json`, `events.jsonl`, `results/*.json` and the
        directory itself - are already produced verbatim by every run.
        `observed.jsonl` was the fifth and the only one that needed a recorder,
        and `tests/fixtures/runs/README.md` names writing it "the cheapest
        retirement of the largest risk in the epic": the corpus that proves the
        derived resolver is entirely synthesised, and a recorded run is a
        `cp -r` away from fixing that. The manifest goes with it - see
        `corpus_manifest`, which declares nothing on purpose.
        """
        first = not self.observed_path.exists()
        recorded = self.observations.write(payload)
        if first:
            self.corpus_manifest()
        return recorded

    def corpus_manifest(self) -> Path:
        """Write `corpus.json` so this directory is a replay corpus run.

        `origin: "recorded"` is metadata and nothing branches on it -
        `tests/test_derived.py` proves that by flipping the field and asserting
        the replay is identical. `expected_divergences` is **empty and stays
        empty**: the harness fails on an undeclared divergence, so a recorded
        run refuses to pass until somebody writes the argument for each one, and
        the README is explicit that the declaration *is* the argument. A
        recorder that pre-declared what it had just seen would be a harness
        tuning itself to agree, which is the one thing the corpus was built not
        to be.
        """
        target = self.path / CORPUS_MANIFEST_NAME
        if not target.exists():
            self._write_json(
                target,
                {
                    "schema": CORPUS_SCHEMA,
                    "origin": "recorded",
                    "describes": self.run.objective or f"recorded run {self.run.id}",
                    "exercises": [],
                    "expected_divergences": [],
                },
            )
        return target

    def mount_flags(self) -> list[str]:
        """The `docker create` flags that give a worker somewhere to write.

        `ContainerManager.spawn` mounts no volume, so today a worker's result
        file is written inside a container that is removed seconds later. The
        spawn call site owns the argv - the dispatcher (#21) - and this is the
        value it needs; `security.assert_unprivileged` passes it, because the
        source is a run directory rather than the socket or the host root.

        The source is translated through `host_path`, because the daemon
        resolves it. When the orchestrator is a container those are different
        filesystems, and naming its own path would mount an empty directory
        the host happened to create.
        """
        source = host_path(self.results_dir, root=self.root)
        return ["--volume", f"{source}:{DEFAULT_RESULT_DIR}"]

    def worker_env(self) -> dict[str, str]:
        """The other half of the same wiring: where the worker writes inside.

        Explicit even though it equals `DEFAULT_RESULT_DIR`, so that changing
        the mount point is one edit here rather than a mount and an environment
        variable that can disagree.
        """
        return {RESULT_DIR_ENV: DEFAULT_RESULT_DIR}

    # --- writing --------------------------------------------------------

    def event(self, name: str, **fields: Any) -> dict[str, Any]:
        """Record one structured event. Returns the redacted payload."""
        return self.log.emit(name, run=self.run.id, **fields)

    @contextlib.contextmanager
    def cycle(self, number: int) -> Iterator[CycleMetrics]:
        """Measure one reconcile cycle.

            with artifacts.cycle(n) as metrics:
                metrics.api_call()
                metrics.inference(SETTINGS.worker_model, seconds=41.2, load_s=6.7)

        The numbers are written in a `finally`, so a cycle that raised still
        leaves its cost on disk. The cycle that blew up is the one whose API
        call count is worth having, and a metrics record that only survives the
        happy path measures the runs nobody needed to measure.
        """
        metrics = CycleMetrics(cycle=number, started_at=_now(), loaded_model=self._loaded_model)
        self.event(CYCLE_STARTED, cycle=number)
        try:
            yield metrics
        finally:
            metrics.finished_at = _now()
            self._cycles.append(metrics)
            self._loaded_model = metrics.loaded_model
            self.event(CYCLE_FINISHED, **metrics.to_dict())

    def container_log(self, handle: Handle) -> Path:
        """Persist what a container printed, before it is only in memory.

        Written from `Handle.captured`, which `dispose_container` filled *before*
        removing the container - so this is called after disposal, never before
        it, and there is no ordering to remember because there is nothing else to
        write.

        An orphan's log is filed under the run that produced it, not under this
        one. The reaper (#20) sweeps containers belonging to dead orchestrators,
        and filing a dead run's evidence in a live run's directory is precisely
        how "which run went wrong" stops being answerable.
        """
        directory = self._logs_dir_for(handle.run_id)
        target = directory / _log_name(handle)
        # `captured is None` means nothing ever read this container, which is a
        # different fact from "it printed nothing" and the more alarming of the
        # two: it says the capture-before-removal ordering was not followed.
        body = (
            "[apiary] nothing was captured from this container before it was written down\n"
            if handle.captured is None
            else self.redact(handle.captured)
        )
        header = self.redact(
            f"[apiary] container {handle.id[:12]}  name {handle.name or '(none)'}  "
            f"image {handle.image or '(unknown)'}  run {handle.run_id}  issue {handle.issue}\n"
        )
        try:
            directory.mkdir(parents=True, exist_ok=True)
            target.write_text(header + body, encoding="utf-8")
        except OSError as exc:
            raise ArtifactsError(f"writing {target}: {exc}") from exc
        self.event(
            CONTAINER_LOGGED,
            container=handle.id[:12],
            issue=handle.issue,
            of_run=handle.run_id,
            path=str(target),
            chars=len(body),
        )
        return target

    @property
    def log_sink(self) -> Callable[[Handle], None]:
        """`Reaper.sink`, which is typed as exactly this.

            Reaper(run=run, sink=artifacts.log_sink).guard()

        A property rather than making `RunArtifacts` callable: the reaper's sink
        is one of several things this object does, and an object whose `__call__`
        means "write a container log" reads as a mystery at the call site.
        """
        return lambda handle: self.container_log(handle)

    # --- folding --------------------------------------------------------

    def results(self) -> RunSummary:
        """#18's records, summarised. Reads the directory; holds nothing."""
        return summarise_dir(self.results_dir, run_id=self.run.id)

    def metrics(self) -> RunMetrics:
        return RunMetrics(tuple(self._cycles))

    def finish(self, *, note: str = "", finished_at: dt.datetime | None = None) -> RunView:
        """Write `summary.json`, then read the directory back as a `RunView`.

        Reading it back rather than returning what was just folded is the point
        of the return type: what a caller gets is what `swarm show` will print
        for this run tomorrow, so a summary that did not land is a failure here
        rather than a surprise later.
        """
        moment = finished_at or _now()
        results = self.results()
        self._write_json(
            self.path / SUMMARY_FILE_NAME,
            {
                "schema": SCHEMA_VERSION,
                "run_id": self.run.id,
                "repo": self.run.repo,
                "objective": self.run.objective,
                "started_at": _iso(self.run.started_at),
                "finished_at": _iso(moment),
                # Repeated from `run.json` rather than cross-referenced: this
                # is the file people grep, and a query that has to open two
                # files to answer one question is a query nobody writes.
                "stack": self.stack,
                "verify": self.verify,
                "note": note,
                "metrics": self.metrics().to_dict(),
                # Derived from `results/` and never read back - the files are the
                # source of truth and re-reading this block would let a
                # hand-edited summary disagree with them. Written because the
                # first thing anyone does with a run directory is grep it, and
                # `summary.json` is where they grep.
                "results": {
                    "attempts": results.attempts,
                    "consumed": results.consumed,
                    "counts": results.counts,
                    "issues": {
                        str(issue): record.outcome for issue, record in results.latest.items()
                    },
                },
            },
        )
        # Totals only. The per-cycle rows are already in this log as their own
        # `cycle.finished` events, and repeating them here would make the last
        # line of the file the largest one in it.
        totals = {key: value for key, value in self.metrics().to_dict().items() if key != "cycles"}
        self.event(
            RUN_FINISHED,
            note=note,
            attempts=results.attempts,
            consumed=results.consumed,
            **totals,
        )
        return read_run(self.path)

    def audit(self, *, env: Mapping[str, str] | None = None) -> list[Leak]:
        """Re-check the finished directory for credentials. Believe an empty list.

        Every write above is redacted, so this should find nothing - which is
        exactly why it is worth running: it is the check that notices a path
        that stopped going through the redactor.
        """
        return scan_artifacts(self.path, env=os.environ if env is None else env)

    # --- context manager -------------------------------------------------

    def __enter__(self) -> RunArtifacts:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Finish on the way out, including on the way out through an exception.

        The note names the exception rather than swallowing it: a run directory
        whose summary says nothing about why the orchestrator stopped is the
        directory this module exists to stop existing.
        """
        self.finish(note="" if exc is None else f"{type(exc).__name__}: {exc}")

    # --- plumbing --------------------------------------------------------

    def _logs_dir_for(self, run_id: str) -> Path:
        if not run_id or run_id == self.run.id:
            return self.logs_dir
        try:
            validate_run_id(run_id)
        except RunError:
            # Wearing our label with a value we never minted. It is not one of
            # ours to file under a run id, but the log is still worth keeping.
            return self.logs_dir
        return self.root / run_id / LOGS_DIR_NAME

    def _write_json(self, target: Path, payload: Mapping[str, Any]) -> Path:
        """Atomic, and redacted, for the same reasons `worker/result.py` is.

        Temporary file then `os.replace`, because the process writing a run's
        summary is the one shutting down, and half a JSON object is worse than
        none - a reader cannot tell it from a file somebody corrupted.

        The mechanism moved to the module-level `write_json` when `capture.py`
        needed the same guarantee; this stays as the method that knows which
        redactor to hand it.
        """
        return write_json(target, payload, redact=self.redact)


def _log_name(handle: Handle) -> str:
    """`issue-<n>-<short id>.log`, or the container's id alone if it has neither.

    The short id is in the name rather than the attempt number because a handle
    does not carry one, and because two attempts at one issue within a run are
    two containers: naming by issue alone would have the retry overwrite the log
    that says why there was a retry.
    """
    short = handle.id[:12] or "unknown"
    if handle.issue is None:
        return f"container-{short}.log"
    return f"issue-{int(handle.issue)}-{short}.log"


# --------------------------------------------------------------------------
# Reading a run back
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DivergenceTally:
    """What the derived-state shadow saw across one run (#146).

    **`ran` is the field that matters, and it is not a count.** Zero
    divergences and a shadow that never executed produce the same empty list of
    `state.divergence` events, and epic #140's go/no-go is "ten consecutive runs
    with zero unexplained divergences" - a gate that reads an unmeasured run as
    a clean one passes on nothing at all. So `state.shadow` is emitted every
    shadowed cycle and its presence is what `ran` reports.

    `unexplained` rather than `total` is the number the gate reads.
    `orchestrator/shadow.py` classifies each divergence against ADR 0001's three
    non-derivable states and the cycle's own after-the-read actions, and an
    expected divergence is evidence the model is right rather than a defect.
    """

    ran: bool = False
    cycles: int = 0
    #: Cycles the window ran and could not see - `checks.read_pulls` answered
    #: `None`. Counted apart from `cycles` because a blind cycle compared
    #: nothing and must not be read as a cycle that agreed.
    blind: int = 0
    #: Task-cycles compared, and the subset whose label the cycle did **not**
    #: write. Without the first, "0 divergences over 40 cycles" is printed for a
    #: run that compared nothing - unmeasured reading as clean, which is the
    #: failure this whole tally exists to prevent, one level up. Without the
    #: second, a reader cannot tell how much of a clean window was the cycle
    #: agreeing with its own arithmetic: see `orchestrator/shadow.py`'s
    #: "What a clean window is, and is not, evidence of".
    compared: int = 0
    independent: int = 0
    total: int = 0
    unexplained: int = 0
    #: Kind -> count, expected kinds only, in descending order of count.
    by_kind: tuple[tuple[str, int], ...] = ()
    #: Task ids with at least one unexplained divergence, in first-seen order.
    #: Named rather than counted for `derived.Divergence`'s reason: a count is
    #: compatible with every disagreement being on the one state ADR 0001
    #: reports outbound.
    unexplained_tasks: tuple[str, ...] = ()
    #: The cutover's own number (#147): task-cycles where the orchestrator acted
    #: on the resolver rather than on the label. Kept apart from `total` because
    #: it is sampled at the other end of the cycle and answers the other
    #: question - not "do the two agree afterwards" but "which one decided".
    #: Zero over a run with coverage is the interesting reading: it says the
    #: labels never lied, not that nothing was believed.
    overrides: int = 0
    #: Kind -> count for the overrides that carried an account, and the task ids
    #: of the ones that did not. An unaccounted override is the label being
    #: stale, which is ordinary; an accounted one is apiary knowing something
    #: the code host cannot show, which is one of ADR 0001's three.
    override_kinds: tuple[tuple[str, int], ...] = ()
    override_tasks: tuple[str, ...] = ()

    @classmethod
    def from_events(cls, events: Iterable[Mapping[str, Any]]) -> DivergenceTally:
        cycles = 0
        blind = 0
        compared = 0
        independent = 0
        total = 0
        unexplained = 0
        kinds: dict[str, int] = {}
        tasks: list[str] = []
        overrides = 0
        override_kinds: dict[str, int] = {}
        override_tasks: list[str] = []
        ran = False
        for payload in events:
            name = payload.get("event")
            if name == STATE_OVERRIDE:
                # Deliberately **not** setting `ran`: an override is evidence
                # about the cutover, not about the shadow window, and a run with
                # `APIARY_DERIVED_SHADOW=0` still emits these. Reporting "the
                # shadow ran" off them would put unmeasured back where the whole
                # of this class exists to keep it out.
                overrides += 1
                kind = str(payload.get("kind") or "")
                task = str(payload.get("task") or "")
                if kind:
                    override_kinds[kind] = override_kinds.get(kind, 0) + 1
                elif task and task not in override_tasks:
                    override_tasks.append(task)
                continue
            if name == STATE_SHADOW:
                ran = True
                cycles += 1
                if payload.get("blind"):
                    blind += 1
                    continue
                compared += int(payload.get("tasks") or 0)
                independent += int(payload.get("independent") or 0)
                continue
            if name != STATE_DIVERGENCE:
                continue
            ran = True
            total += 1
            kind = str(payload.get("kind") or "")
            if kind:
                kinds[kind] = kinds.get(kind, 0) + 1
                continue
            unexplained += 1
            task = str(payload.get("task") or "")
            if task and task not in tasks:
                tasks.append(task)
        return cls(
            ran=ran,
            cycles=cycles,
            blind=blind,
            compared=compared,
            independent=independent,
            total=total,
            unexplained=unexplained,
            by_kind=tuple(sorted(kinds.items(), key=lambda one: (-one[1], one[0]))),
            unexplained_tasks=tuple(tasks),
            overrides=overrides,
            override_kinds=tuple(
                sorted(override_kinds.items(), key=lambda one: (-one[1], one[0]))
            ),
            override_tasks=tuple(override_tasks),
        )

    def text(self) -> str:
        """The lines `swarm show` prints. Coverage first, because a divergence
        count with no coverage beside it is a number that cannot be read."""
        if not self.ran:
            head = "derived shadow: not run (APIARY_DERIVED_SHADOW off, or a run from before #146)"
            return "\n".join([head, *self._override_lines()])
        if not self.compared:
            # Every other branch below would print "0 unexplained" for this,
            # which is true and reads as a clean run. It is not one: the window
            # ran and compared nothing.
            head = (
                f"derived shadow: ran for {self.cycles} cycle(s) and compared no tasks"
                + (f" ({self.blind} blind)" if self.blind else "")
                + " - this run is unmeasured, not clean"
            )
            return "\n".join([head, *self._override_lines()])
        lines = [
            f"derived shadow: {self.compared} task-cycle(s) compared over "
            f"{self.cycles} cycle(s)"
            + (f", {self.blind} blind" if self.blind else "")
            + f"; {self.independent} of them independent of this cycle's own writes"
        ]
        head = f"  {self.total} divergence(s), {self.unexplained} unexplained"
        if self.by_kind:
            head += " (" + ", ".join(f"{kind} {count}" for kind, count in self.by_kind) + ")"
        lines.append(head)
        if self.unexplained_tasks:
            lines.append("  unexplained on: " + ", ".join(self.unexplained_tasks))
        lines += self._override_lines()
        return "\n".join(lines)

    def _override_lines(self) -> list[str]:
        """What the cutover did, printed whether or not the shadow ran.

        Silent when nothing was overridden, because a line reading "0 overrides"
        on every run of a system whose labels are usually right is a line people
        stop seeing - and unlike the coverage number above, zero here is not
        ambiguous: the overrides are emitted by the decision path itself, so
        none means none rather than unmeasured.
        """
        if not self.overrides:
            return []
        head = (
            f"  {self.overrides} state override(s): the orchestrator's answer differed "
            "from the label, the resolver, or both"
        )
        if self.override_kinds:
            head += " (" + ", ".join(
                f"{kind} {count}" for kind, count in self.override_kinds
            ) + ")"
        lines = [head]
        if self.override_tasks:
            lines.append("  stale labels on: " + ", ".join(self.override_tasks))
        return lines


@dataclass(frozen=True)
class RunView:
    """One run directory, read from disk. What `swarm show <run-id>` prints.

    Built from files and nothing else - no GitHub call, no Docker daemon, no
    orchestrator - which is the whole claim of the ticket: a completed run can
    be understood from its artifacts without re-reading GitHub, on a machine
    that never dispatched it.
    """

    path: Path
    run_id: str
    repo: str = ""
    objective: str = ""
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    note: str = ""
    #: Which stack this run targeted and the repo-wide gate it ran. Empty for a
    #: run recorded before either was written down, which is why `show_text`
    #: prints "(unrecorded)" rather than an empty column.
    stack: str = ""
    verify: str = ""
    metrics: RunMetrics = RunMetrics()
    results: RunSummary = RunSummary(run_id="", records=())
    container_logs: tuple[Path, ...] = ()
    event_count: int = 0

    @property
    def complete(self) -> bool:
        """Did the orchestrator reach its own end? `summary.json` is the evidence."""
        return self.finished_at is not None

    @property
    def needs_human(self) -> tuple[int, ...]:
        """Issues whose latest attempt did not end in an open PR.

        The ticket's actual question. A run is not interesting for its success
        rate; it is interesting for where it stopped being able to proceed on
        its own, and that is this list.
        """
        return tuple(
            issue
            for issue, record in sorted(self.results.latest.items())
            if record.outcome != "pr-open"
        )

    def divergences(self) -> DivergenceTally:
        """The shadow window's verdict on this run (#146).

        A method rather than a field, and read lazily on purpose: `list_runs`
        builds a `RunView` per directory for `swarm runs`, and folding every
        run's event log to print one table would make listing cost the whole
        history. `swarm show` asks for one run and pays for one read.
        """
        return DivergenceTally.from_events(read_events(self.path / EVENT_LOG_NAME))

    def audit(self, *, env: Mapping[str, str] | None = None) -> list[Leak]:
        return scan_artifacts(self.path, env=os.environ if env is None else env)

    def line(self) -> str:
        """One run, one line - the row `swarm runs` prints."""
        when = "" if self.started_at is None else self.started_at.strftime("%Y-%m-%d %H:%M")
        state = "finished" if self.complete else "incomplete"
        return (
            f"{self.run_id}  {when}  {self.results.attempts} attempt(s), "
            f"{self.results.consumed} consumed  ({state})"
        )


def read_run(path: str | Path) -> RunView:
    """Read one run directory. Works on a run that never finished.

    Three sources, in order of authority: the result files (#18's records,
    always re-read rather than trusted from the summary), `summary.json` if the
    orchestrator got that far, and the event log folded into metrics if it did
    not. A killed run therefore still reports its cycles, which is the run
    somebody is most likely to be looking at.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise ArtifactsError(f"{directory} is not a run directory")

    identity = _read_json(directory / RUN_FILE_NAME)
    summary = _read_json(directory / SUMMARY_FILE_NAME)
    run_id = str(summary.get("run_id") or identity.get("run_id") or directory.name)

    metrics = (
        RunMetrics.from_dict(summary["metrics"])
        if isinstance(summary.get("metrics"), Mapping)
        else RunMetrics.from_events(read_events(directory / EVENT_LOG_NAME))
    )
    return RunView(
        path=directory,
        run_id=run_id,
        repo=str(summary.get("repo") or identity.get("repo") or ""),
        objective=str(summary.get("objective") or identity.get("objective") or ""),
        started_at=_parse(summary.get("started_at") or identity.get("started_at")),
        finished_at=_parse(summary.get("finished_at")),
        note=str(summary.get("note") or ""),
        stack=str(summary.get("stack") or identity.get("stack") or ""),
        verify=str(summary.get("verify") or identity.get("verify") or ""),
        metrics=metrics,
        results=summarise_dir(directory / RESULTS_DIR_NAME, run_id=run_id),
        container_logs=tuple(sorted((directory / LOGS_DIR_NAME).glob(LOG_GLOB))),
        event_count=_count_lines(directory / EVENT_LOG_NAME),
    )


def load_run(run_id: str, root: str | Path | None = None) -> RunView:
    """`swarm show <run-id>`. The id is validated before it becomes a path."""
    directory = run_dir(run_id, root)
    if not directory.is_dir():
        raise ArtifactsError(f"no artifacts for run {run_id!r} under {directory.parent}")
    return read_run(directory)


def list_runs(root: str | Path | None = None) -> tuple[RunView, ...]:
    """Every run under `root`, oldest first. `swarm runs`.

    A directory whose name is not a well-formed run id is skipped rather than
    read: run ids are validated at mint (`run.py`), so anything else under the
    root was put there by something that is not this system, and reading it
    would be guessing.
    """
    base = artifacts_root() if root is None else Path(root)
    if not base.is_dir():
        return ()

    views: list[RunView] = []
    for directory in sorted(base.iterdir()):
        if not directory.is_dir():
            continue
        try:
            validate_run_id(directory.name)
        except RunError:
            continue
        try:
            views.append(read_run(directory))
        except ArtifactsError as exc:
            print(f"! skipping {directory.name}: {exc}", file=sys.stderr)
    # Run ids are UTC-stamped and lexicographically chronological within one
    # repo (`run.py`), so the id is the tiebreaker for two runs whose `run.json`
    # is missing and which therefore have no timestamp at all.
    return tuple(sorted(views, key=lambda view: (view.started_at or _EPOCH, view.run_id)))


def runs_text(views: Sequence[RunView]) -> str:
    """`swarm runs`, grouped by repository.

    Grouped because that is what a human is asking: `run.py` records that a
    human's continuity across restarts is "the repository and its `swarm:*`
    issues", so the runs of one repository are one story and the runs of another
    are noise in the middle of it.
    """
    if not views:
        return "no runs recorded"
    by_repo: dict[str, list[RunView]] = {}
    for view in views:
        by_repo.setdefault(view.repo or "(unknown repo)", []).append(view)

    lines: list[str] = []
    for repo in sorted(by_repo):
        lines.append(repo)
        lines += [f"  {view.line()}" for view in by_repo[repo]]
    return "\n".join(lines)


#: How much of one attempt's verify output `swarm show` prints per issue.
#: Small on purpose: this is a summary, and the full tail is in the result file
#: two lines below it. Enough to tell a stack trace from a 403.
SHOWN_OUTPUT_CHARS = 800


def _needed_a_human(issue: int, record: ResultRecord) -> list[str]:
    """One issue's account of why it stopped, and what the gate actually said.

    Printing `record.reason` alone is what this used to do, and for a run with
    zero merges the entire user-visible answer was the string "the verify
    command failed", repeated - so "the model wrote bad React" and "npm was
    unreachable" were byte-identically labelled. `verify_output` was on disk
    the whole time and no code path printed it.

    The command is printed beside the output because a gate that never opened
    is diagnosed from the command far more often than from what it printed:
    `npm test` in a repository with no `package.json` says everything.

    **Safe to print only because #90 landed.** Result files are the one
    artifact written without passing through the redactor, and the verify
    subprocess used to inherit `GITHUB_TOKEN`; printing this before the scrub
    would have risked putting a credential on a terminal.
    """
    lines = [f"  #{issue}: {record.reason or record.outcome}"]
    if record.verify_command:
        lines.append(f"      gate: {record.verify_command}")
    if record.image:
        lines.append(f"      image: {record.image}")
    output = (record.verify_output or "").strip()
    if output:
        shown = tail(output, SHOWN_OUTPUT_CHARS)
        lines += [f"      | {line}" for line in shown.splitlines()]
    return lines


def show_text(view: RunView) -> str:
    """`swarm show <run-id>`: one run, in the order the questions get asked.

    What was it, how did it go, what did it cost, and where did it need a human.
    The last one is deliberately last and deliberately present even when it is
    empty, because "nothing needed a human" is an answer somebody came for.
    """
    lines = [
        f"run {view.run_id}",
        f"  repo       {view.repo or '(unknown)'}",
        f"  objective  {view.objective or '(unrecorded)'}",
        f"  stack      {view.stack or '(unrecorded)'}",
        f"  verify     {view.verify or '(unrecorded)'}",
        f"  started    {_iso(view.started_at) or '(unrecorded)'}",
        f"  finished   {_iso(view.finished_at) or '(never - this run did not reach its end)'}",
    ]
    if view.note:
        lines.append(f"  note       {view.note}")
    lines += ["", view.results.text(), "", view.metrics.text(), ""]

    if view.needs_human:
        lines.append("needed a human: " + ", ".join(f"#{issue}" for issue in view.needs_human))
        for issue in view.needs_human:
            lines += _needed_a_human(issue, view.results.latest[issue])
    else:
        lines.append("needed a human: nothing")

    lines += [
        "",
        # Before the file counts rather than after: an unexplained divergence is
        # the one thing in a run directory that says the system's own model of
        # its state is wrong, and #146 asks that a shadowed run report without
        # anybody reading the JSONL by hand.
        view.divergences().text(),
        "",
        f"{view.event_count} event(s) in {EVENT_LOG_NAME}, "
        f"{len(view.container_logs)} container log(s) in {LOGS_DIR_NAME}/",
        f"at {view.path}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Small shared pieces
# --------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    """A JSON object from disk, or `{}` for anything that is not one.

    Missing is ordinary - a run that was killed wrote no summary - and corrupt
    is survivable, because everything read through here has another source.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _count_lines(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except (OSError, UnicodeDecodeError):
        return 0
