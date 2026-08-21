"""One model call, without GitHub, without Docker, without a repository.

Seeing what a model actually answers costs a whole run today: a repo slug, a
token, a Docker daemon, built worker images, two exported variables and about
nine minutes. That is the reason nobody looks, and the reason a bad answer gets
diagnosed by guessing.

This serves one page on loopback where an operator types the *human* turn of a
prompt, fires it at the same local Ollama a run would use, and reads the
schema-constrained result, the raw response, the timings and - when it breaks -
the real exception beside the prompt that caused it.

**Two sites, not nine.** Of the nine places this system calls a model, three
interpolate every variable they have into the *system* prompt (so a human-turn
box would let you change nothing), one is dead on the v2 path, one sends a
constant, and one is usually answered by arithmetic before a model is consulted.
`propose_edits` and `choose_stack` are what is left, and they are the two worth
having: the first is where whole-file generation goes wrong, the second answers
in seconds and proves the wiring.

**Prompts come from `prompt_for`, never from here.** Both sites export the exact
`(system, human)` pair they send, and this module calls it. A console that built
its own approximation would eventually show a prompt production does not send,
which is worse than showing nothing.

**Threads, deliberately, and exactly two.** The plan for this module said
single-threaded with a blocking POST, to avoid introducing the first thread into
this codebase. Then the T1 spike measured a real `propose_edits` call at
106-163 seconds. A blocking POST means the browser shows nothing for three
minutes, cannot be refreshed, and loses the answer if anything times out. So the
call runs on a worker thread and the page polls - and the concurrency that would
otherwise make a process-wide recorder unsafe is bounded by single-flight: one
inference at a time, refused rather than queued.

**And one tab that is not a model call at all.** Firing the planner from this
page and watching nothing appear on GitHub taught the obvious lesson: an
operator who has just read a good plan wants to *run* it, and making them
reconstruct the `swarm run` invocation in a terminal is where the console
stopped helping. The `swarm` tab (`console_runs.py`) execs the real CLI as a
subprocess and streams its log to the page - repository, issues, workers,
pull requests, merges, the goal verdict - without this module gaining any
pipeline logic of its own.

**And one decision that is this module's, deliberately.** #129 gave the planner
tab a Start building button that provisions a repository and writes the plan
into it as issues; #130 made that button go on to *run* it. The provisioning
half lives in `console_build`, the supervision half in `console_runs`, and the
line joining them is `_start_run` here - because it is the only place that
holds both a finished `BuildReport` and the `SwarmRuns` that will watch the
child. What it decides is small and worth finding: which repository the run
attaches to, that it must not be asked to provision a second one, and that a
run which will not start leaves the build finished rather than failed.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping

from . import capture as capture_mod
from .artifacts import console_root
from .capture import CAPTURE_ENV, Capture, Recorder
from .config import ConfigError, SETTINGS
from .console_board import BoardError, BoardReader
from .console_build import BUILD_SITE, BUILD_SITE_KEY, BuildError, Builder
from .console_intake import QUESTIONS as INTAKE_QUESTIONS
from .console_projects import ProjectError, ProjectStore
from .console_runs import (
    REPO_RE,
    SWARM_SITE,
    SwarmRunError,
    SwarmRuns,
    check_run_values,
)
from .store import StoreError

__all__ = [
    "ASSETS",
    "ASSET_TYPES",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "LOOPBACK",
    "Console",
    "Response",
    "SITES",
    "Site",
    "asset",
    "page",
    "serve",
    "validate_capture_id",
]

#: Loopback only, and checked rather than assumed. This is not ceremony: the
#: egress allow-list that lets a worker container reach Ollama matches
#: `host.docker.internal` by *host*, with no port term, so tinyproxy permits a
#: container to reach any port on the host gateway. A wildcard bind would put
#: every captured prompt - whole file bodies from the target repository - one
#: HTTP request away from the one process in this system that runs
#: model-generated code.
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
DEFAULT_HOST = "127.0.0.1"

#: Fixed rather than ephemeral, so the `Host` allow-list below is checkable and
#: so the operator can bookmark it.
DEFAULT_PORT = 8117

#: A capture id arrives back from a browser on its way to becoming a path, and
#: `../../etc` is a valid string. Same defence, and the same alphabet, as
#: `validate_run_id`.
CAPTURE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class ConsoleError(ValueError):
    """A refusal an operator can fix. Rendered by `cli.main` as one `!` line."""


class ConsoleBusy(ConsoleError):
    """Single flight, refused. Carries the fix, and is always a 409.

    Its own type because two routes raise it now - a model call and a build -
    and the alternative was each of them re-deriving "is something running,
    and what do I say if it is" from `_running`. See `Console._claim`.

    Not every busy refusal on this page is one of these: the two #130 added
    (`_swarm_build` refusing while a run is live, `_swarm_start` refusing while
    a build is) are about `SwarmRuns`' latch rather than this one, and answer
    in the same `{error, fix}` shape without borrowing this type's meaning.
    """

    def __init__(self, message: str, *, fix: str = "") -> None:
        super().__init__(message)
        self.fix = fix


def validate_capture_id(value: str) -> str:
    if not CAPTURE_ID.match(value or ""):
        raise ConsoleError(f"not a capture id: {value!r}")
    return value


# --------------------------------------------------------------------------
# The sites
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Field:
    name: str
    label: str
    kind: str = "text"          # "text" | "area"
    placeholder: str = ""
    value: str = ""


@dataclass(frozen=True)
class Site:
    """One exposed call site: what it needs typed, and how to build and fire it.

    `prompt` and `run` are separate so the page can render the prompt *before*
    the call starts. On a cold 31b that is a two-minute wait, and having the
    exact prompt to read during it is most of what makes the wait bearable.
    """

    key: str
    label: str
    blurb: str
    fields: tuple[Field, ...]
    prompt: Callable[[Mapping[str, str]], tuple[str, str]]
    run: Callable[[Mapping[str, str], Any], Any]
    #: Whether this site accepts a per-call model override. Declared per site
    #: rather than offered everywhere, because a site whose model an operator
    #: could change without the change taking effect would be worse than one
    #: that does not offer the control at all - `plan` and `intake` build their
    #: own models several layers down, and wiring a spec through them is not
    #: this ticket.
    overridable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "blurb": self.blurb,
            "overridable": self.overridable,
            "fields": [
                {"name": f.name, "label": f.label, "kind": f.kind,
                 "placeholder": f.placeholder, "value": f.value}
                for f in self.fields
            ],
        }


def _sources(payload: Mapping[str, str]):
    """The writable and readable sets, read from a checkout exactly as a worker
    reads them - `read_writable` and `gather_context`, same budget, same order.
    """
    from .worker.edit import gather_context, read_writable

    root = Path(payload.get("root", "")).expanduser()
    if not root.is_dir():
        raise ConsoleError(f"not a directory: {root}")
    files = [part.strip() for part in (payload.get("files") or "").split(",") if part.strip()]
    if not files:
        raise ConsoleError("name at least one file to edit, comma separated")
    return read_writable(root, files), gather_context(root, files)


def _edits_prompt(payload: Mapping[str, str]) -> tuple[str, str]:
    from .worker.edit import prompt_for

    writable, readable = _sources(payload)
    return prompt_for(payload.get("goal", ""), writable, readable)


def _edits_run(payload: Mapping[str, str], spec: Any = None) -> Any:
    from .llm import structured, worker_llm
    from .state import WorkerOutput
    from .worker.edit import propose_edits

    writable, readable = _sources(payload)
    # The `llm=` seam `propose_edits` already has, used for the first time by
    # something that is not a test. Building the model here rather than adding
    # a `spec=` parameter to `propose_edits` keeps the override entirely inside
    # the console: the worker in its container reaches the factory with no
    # argument and resolves exactly as it did before.
    model = None if spec is None else structured(worker_llm(spec), WorkerOutput, spec)
    output = propose_edits(payload.get("goal", ""), writable, readable, llm=model)
    return {
        "edits": [
            {"path": edit.path, "chars": len(edit.content), "content": edit.content}
            for edit in output.edits
        ],
        "notes": getattr(output, "notes", "") or "",
    }


def _plan_prompt(payload: Mapping[str, str]) -> tuple[str, str]:
    from .nodes.planner import prompt_for

    # The configured gate, which is what a run with no `--verify` would use.
    # Without it the console would show a prompt naming a different command
    # from the one production names, which is the one thing this must not do.
    return prompt_for(payload.get("objective", ""), verify=SETTINGS.verify_command)


def _plan_run(payload: Mapping[str, str], spec: Any = None) -> Any:
    from .nodes.planner import draft_plan

    plan = draft_plan(payload.get("objective", ""), verify=SETTINGS.verify_command)
    return {
        "reasoning": plan.reasoning,
        "tasks": [
            {
                "id": task.id,
                "goal": task.goal,
                "files": list(task.files),
                "depends_on": list(task.depends_on),
                # Rendered nowhere, and carried anyway: #129 rebuilds the plan
                # it writes out of this payload, and a task whose stack was
                # dropped on the way to the screen would be written back with a
                # different one than the model chose.
                "stack": task.stack,
            }
            for task in plan.tasks
        ],
    }


def _stack_prompt(payload: Mapping[str, str]) -> tuple[str, str]:
    from .greenfield.bootstrap import prompt_for

    return prompt_for(payload.get("brief", ""))


def _stack_run(payload: Mapping[str, str], spec: Any = None) -> Any:
    from .greenfield.bootstrap import StackChoice, choose_stack
    from .llm import orchestrator_llm, structured

    model = None if spec is None else structured(orchestrator_llm(spec), StackChoice, spec)
    return {"stack": choose_stack(payload.get("brief", ""), llm=model)}


def _intake_prompt(payload: Mapping[str, str]) -> tuple[str, str]:
    from .console_intake import compose_brief, prompt_for

    return prompt_for(compose_brief(payload))


def _intake_run(payload: Mapping[str, str], spec: Any = None) -> Any:
    from .console_intake import propose

    return propose(payload)


SITES: dict[str, Site] = {
    "planner": Site(
        key="planner",
        label="plan_node — the planner",
        blurb=(
            "How the objective is decomposed into tasks, which gates everything "
            "downstream: each task becomes one issue and one worker. Fresh-plan mode "
            "only — a replan's system prompt carries the failure history, so there is "
            "no single planner prompt to show. Runs the 31b, and asks GitHub for nothing."
        ),
        fields=(
            Field("objective", "Objective", kind="area",
                  placeholder="What the swarm should accomplish."),
        ),
        prompt=_plan_prompt,
        run=_plan_run,
    ),
    "edits": Site(
        key="edits",
        label="propose_edits — the worker",
        blurb=(
            "The whole-file generation the worker does inside its container. This runs on "
            "the host against the same model, so anything environment-dependent may differ. "
            "A cold call takes one to three minutes."
        ),
        fields=(
            Field("root", "Repository checkout", placeholder="/path/to/a/checkout"),
            Field("files", "Files it may edit (comma separated)", placeholder="tests/test_planner.py"),
            Field("goal", "Goal", kind="area", placeholder="What the task asks for."),
        ),
        prompt=_edits_prompt,
        run=_edits_run,
        overridable=True,
    ),
    "stack": Site(
        key="stack",
        label="choose_stack — greenfield",
        blurb=(
            "One word out, seconds to answer, and the site that used to lie most: until "
            "recently a failure here returned 'python' with nothing printed at all."
        ),
        fields=(
            Field("brief", "Project brief", kind="area",
                  placeholder="a dashboard for warehouse pickers"),
        ),
        prompt=_stack_prompt,
        run=_stack_run,
        overridable=True,
    ),
    "intake": Site(
        key="intake",
        label="describe — for non-developers",
        blurb=(
            "Say what you need in plain language - no paths, no commands. A model "
            "proposes the technical setup: a repository name and a stack, with the "
            "rest derived from them. Nothing is created anywhere until the proposal "
            "is fired as a run."
        ),
        fields=tuple(Field(**question) for question in INTAKE_QUESTIONS),
        prompt=_intake_prompt,
        run=_intake_run,
    ),
}


# --------------------------------------------------------------------------
# Requests and responses
# --------------------------------------------------------------------------


@dataclass
class Response:
    status: int
    body: bytes
    content_type: str = "application/json; charset=utf-8"

    @classmethod
    def json(cls, payload: Any, status: int = 200) -> "Response":
        return cls(status, json.dumps(payload, default=str).encode("utf-8"))

    @classmethod
    def html(cls, markup: str, status: int = 200) -> "Response":
        return cls(status, markup.encode("utf-8"), "text/html; charset=utf-8")

    @classmethod
    def error(cls, message: str, status: int = 400, *, fix: str = "") -> "Response":
        return cls.json({"error": message, "fix": fix}, status)


@dataclass
class Job:
    """One in-flight or finished call."""

    id: str
    site: str
    started: float
    state: str = "running"
    result: Any = None
    #: Which model this call actually used, provider-qualified. Carried on the
    #: job rather than read off the capture, because a call that failed before
    #: reaching the model has no capture and is exactly the case where "which
    #: model was that?" is the question being asked.
    model: str = ""
    #: Whether it came from a per-call override rather than from the default.
    #: The page labels the two differently, because a result an operator is
    #: comparing must say which of the two it is.
    overridden: bool = False
    capture: dict[str, Any] | None = None
    #: `Any` rather than `str` because a build's failure carries doctor's
    #: failing checks alongside the message and the fix (#129), and a check is
    #: an object rather than a line. Everything in here is still JSON.
    error: dict[str, Any] | None = None

    def to_dict(self, *, now: float | None = None) -> dict[str, Any]:
        return {
            "id": self.id,
            "site": self.site,
            "state": self.state,
            "elapsed_s": round((now or time.monotonic()) - self.started, 1),
            "model": self.model,
            "overridden": self.overridden,
            "result": self.result,
            "capture": self.capture,
            "error": self.error,
        }


@dataclass
class Console:
    """The decision half. `render` is pure enough to test without a socket.

    Everything the HTTP layer knows how to do is here; the handler below is
    transport. That split is the same one `runs_text` / `show_text` already use
    in `artifacts.py`, and it is what lets the `Host` check, the routing and the
    single-flight refusal all be tested directly.
    """

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    sink: Recorder | None = None
    jobs: dict[str, Job] = field(default_factory=dict)
    runs: SwarmRuns = field(default_factory=SwarmRuns)
    builder: Builder = field(default_factory=Builder)
    board: BoardReader = field(default_factory=BoardReader)
    projects: ProjectStore = field(default_factory=ProjectStore)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _running: str = ""

    # -- guards ---------------------------------------------------------

    def allowed_host(self, header: str | None) -> bool:
        """Reject anything but the loopback names this server is reachable at.

        `BaseHTTPRequestHandler` validates no headers at all, so without this
        any page the operator happens to be browsing can POST here - and a DNS
        rebind can then *read* the answers, which are prompts, which are whole
        source files from a private repository.
        """
        if not header:
            return False
        name = header.rsplit(":", 1)[0].strip("[]") if ":" in header else header
        return name in LOOPBACK and header in {f"{h}:{self.port}" for h in LOOPBACK} | set(LOOPBACK)

    # -- routes ---------------------------------------------------------

    def render(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str] | None = None,
        body: bytes = b"",
    ) -> Response:
        headers = headers or {}
        if not self.allowed_host(headers.get("Host")):
            return Response.error(
                f"refused a request for host {headers.get('Host')!r}",
                403,
                fix=f"open http://{DEFAULT_HOST}:{self.port}/ directly",
            )

        # The route is the path *without* the query, and every exact match
        # below is against it. `do_GET` hands over `self.path` verbatim, so
        # matching the whole string meant `/?debug=1` - the documented way to
        # bring the tab strip back, and therefore the only way to reach the
        # planner tab and Start building - answered "no route for GET
        # /?debug=1". The routes that read a query parse it themselves out of
        # `path`, which is why they keep it.
        route = path.partition("?")[0]

        if method == "GET" and route in ("/", "/index.html"):
            return Response.html(page())
        if method == "GET" and route in ASSET_TYPES:
            return Response(200, asset(route.lstrip("/")).encode("utf-8"), ASSET_TYPES[route])
        if method == "GET" and route == "/sites":
            return Response.json({"sites": [s.to_dict() for s in SITES.values()],
                                  # Its own key, not a fourth entry in `sites`:
                                  # nothing that iterates model-call sites may
                                  # pick up a form with no prompt behind it.
                                  "swarm": SWARM_SITE,
                                  # Its own key for the same reason `swarm` is:
                                  # it is a form, not a call site, and nothing
                                  # that iterates `sites` may fire it at a model.
                                  "build": BUILD_SITE,
                                  # The three original keys stay, spelled the
                                  # same: an older page served by a newer
                                  # backend still renders its header. The
                                  # resolved half is additive beside them.
                                  "models": {"orchestrator": SETTINGS.orchestrator_model,
                                             "worker": SETTINGS.worker_model,
                                             "base_url": SETTINGS.ollama_base_url,
                                             "resolved": _models_payload()}})
        if method == "POST" and route == "/prompt":
            return self._prompt(body)
        if method == "POST" and route == "/run":
            return self._run(body)
        if method == "GET" and path.startswith("/status"):
            return self._status(path)
        if method == "POST" and route == "/swarm/build":
            return self._swarm_build(body)
        if method == "POST" and route == "/swarm/start":
            return self._swarm_start(body)
        if method == "POST" and route == "/swarm/stop":
            return self._swarm_stop(body)
        if method == "POST" and route == "/swarm/reset":
            return self._swarm_reset(body)
        if method == "GET" and route == "/swarm/latest":
            latest = self.runs.latest()
            return Response.json(latest) if latest else Response.error("no runs yet", 404)
        if method == "GET" and path.startswith("/swarm/status"):
            return self._swarm_status(path)
        if method == "GET" and path.startswith("/swarm/board"):
            return self._swarm_board(path)
        if method == "GET" and path.startswith("/swarm/worker"):
            return self._swarm_worker(path)
        if method == "GET" and path.startswith("/swarm/external"):
            return self._swarm_external(path)
        if method == "GET" and path.startswith("/swarm/outcome"):
            return self._swarm_outcome(path)
        if method == "GET" and route == "/models":
            return Response.json(_models_payload())
        if method == "POST" and route == "/models":
            return self._models_save(body)
        if method == "GET" and route == "/projects":
            return self._projects_list()
        if method == "POST" and route == "/projects":
            return self._projects_save(body)
        if method == "GET" and path.startswith("/projects/history"):
            return self._projects_history(path)
        return Response.error(f"no route for {method} {path}", 404)

    def _models_save(self, body: bytes) -> Response:
        """Set, or clear, the saved default for one role.

        Deliberately a different control from the per-call override, and the
        two are never the same request: this one persists and affects every
        subsequent run, that one affects nothing else. A page that conflated
        them would repoint an operator's future runs the first time they
        wanted to try something.
        """
        from .llm import ModelSpec
        from .models import ROLES as MODEL_ROLES, parse_model, parse_options, store

        try:
            data = json.loads(body or b"{}")
            role = str(data.get("role") or "")
            if role not in MODEL_ROLES:
                raise ConsoleError(f"unknown role {role!r}; the roles are {', '.join(MODEL_ROLES)}")
            name = str(data.get("model") or "").strip()
            if not name:
                store().save(role, None)
            else:
                provider, model = parse_model(name)
                store().save(role, ModelSpec(
                    provider=provider,
                    model=model,
                    options=parse_options(str(data.get("options") or "")),
                ))
        except (ConsoleError, ConfigError) as exc:
            return Response.error(str(exc), 400, fix=getattr(exc, "fix", "") or
                                  "write it as `provider:model`, or leave it empty to clear")
        except json.JSONDecodeError as exc:
            return Response.error(f"bad request body: {exc}", 400)
        return Response.json(_models_payload())

    def _payload(self, body: bytes) -> tuple[Site, dict[str, str], Any]:
        """The site, its typed values, and an optional one-call model override.

        The override is **not persisted** and is deliberately a different thing
        from the saved default below. Conflating them is the obvious mistake:
        an operator who wanted to *try* a model once and accidentally
        repointed every future run has been badly served.
        """
        data = json.loads(body or b"{}")
        site = SITES.get(str(data.get("site", "")))
        if site is None:
            raise ConsoleError(f"unknown site: {data.get('site')!r}")
        values = {k: str(v) for k, v in (data.get("values") or {}).items()}
        spec = self._spec(site, data.get("model"))
        return site, values, spec

    @staticmethod
    def _spec(site: Site, wanted: Any) -> Any:
        """A `ModelSpec` from what the page sent, or `None` for "the default"."""
        if not isinstance(wanted, Mapping):
            return None
        name = str(wanted.get("model") or "").strip()
        if not name:
            return None
        if not site.overridable:
            raise ConsoleError(
                f"{site.key} does not take a per-call model; it builds its models "
                f"several layers down and the override would not take effect"
            )
        from .llm import ModelSpec
        from .models import parse_model, parse_options

        try:
            provider, model = parse_model(name)
            return ModelSpec(
                provider=provider,
                model=model,
                options=parse_options(str(wanted.get("options") or "")),
            )
        except ConfigError as exc:
            raise ConsoleError(str(exc)) from exc

    def _prompt(self, body: bytes) -> Response:
        try:
            site, values, _ = self._payload(body)
            system, human = site.prompt(values)
        except ConsoleError as exc:
            return Response.error(str(exc), 400)
        except Exception as exc:  # noqa: BLE001 - a bad fixture is the operator's, not a crash
            return Response.error(f"{type(exc).__name__}: {exc}", 400)
        return Response.json({"system": system, "human": human,
                              "chars": len(system) + len(human)})

    def _claim(self, site: str) -> Job:
        """One thing at a time, on one latch, for every kind of work there is.

        #129 asked for a second Start building to be refused "the same way a
        second inference is refused". The same way means the same latch, not a
        second one that agrees with it most of the time: a build provisions a
        repository and writes a backlog, and two of those racing would create
        two repositories from one plan. So a build takes a `Job` and claims
        `_running` exactly as a model call does, and each kind explains itself
        in terms of what is *already* running rather than what was asked for -
        Ollama loading one model at a time is the reason for the first and has
        nothing to do with the second.
        """
        with self._lock:
            if self._running:
                running = self.jobs.get(self._running)
                kind = getattr(running, "site", "") or "it"
                if kind == BUILD_SITE_KEY:
                    raise ConsoleBusy(
                        "a build is already in flight, and a second one would create "
                        "a second repository from the same plan",
                        fix="wait for it to finish, or reload the page",
                    )
                raise ConsoleBusy(
                    "a call is already in flight, and Ollama loads one model at a time",
                    fix=f"wait for {kind} to finish, or reload the page",
                )
            job = Job(id=uuid.uuid4().hex[:16], site=site, started=time.monotonic())
            self.jobs[job.id] = job
            self._running = job.id
            return job

    def _run(self, body: bytes) -> Response:
        try:
            site, values, spec = self._payload(body)
            job = self._claim(site.key)
            job.model = _model_label(site, spec)
            job.overridden = spec is not None
        except ConsoleBusy as exc:
            return Response.error(str(exc), 409, fix=exc.fix)
        except ConsoleError as exc:
            return Response.error(str(exc), 400)

        threading.Thread(target=self._work, args=(site, values, job, spec), daemon=True).start()
        return Response.json(job.to_dict(), 202)

    def _work(self, site: Site, values: Mapping[str, str], job: Job, spec: Any = None) -> None:
        """`job.state` is written last, and that ordering is load-bearing.

        The page polls `/status` from another thread. Publishing "done" or
        "error" before the result, the capture and the fix are attached lets a
        poll land in between and render a finished call with nothing in it -
        which is indistinguishable, on screen, from a model that answered
        nothing.
        """
        state, result, error = "error", None, None
        try:
            result = site.run(values, spec)
            state = "done"
        except Exception as exc:  # noqa: BLE001 - every failure belongs on the page
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "fix": _fix_for(exc),
                "traceback": traceback.format_exc()[-2000:],
            }
        finally:
            sink = self.sink or capture_mod.recorder()
            if sink is not None and sink.last is not None:
                job.capture = sink.last.to_dict()
            job.result = result
            job.error = error
            job.state = state
            with self._lock:
                self._running = ""

    def _status(self, path: str) -> Response:
        _, _, query = path.partition("?")
        wanted = dict(part.split("=", 1) for part in query.split("&") if "=" in part)
        try:
            job = self.jobs.get(validate_capture_id(wanted.get("id", "")))
        except ConsoleError as exc:
            return Response.error(str(exc), 400)
        if job is None:
            return Response.error("no such call", 404)
        return Response.json(job.to_dict())

    # -- the swarm tab ----------------------------------------------------
    #
    # Thin on purpose: everything these decide lives in `console_runs`, which
    # is testable without a socket, exactly as `render` itself is. The only
    # logic here is translating `SwarmRunError` into the same
    # `{error, fix}` shape every other refusal on this page uses.

    def _swarm_start(self, body: bytes) -> Response:
        # The other direction of `_swarm_build`'s gate, and it is not symmetry
        # for its own sake. A build spends minutes provisioning before it asks
        # for the single run slot; a Fire pressed on the swarm tab in that
        # window takes the slot, and the build's own `_start_run` is then
        # refused into `run_error` - leaving a repository and a written backlog
        # with no swarm on it, which is exactly the "a human has to go and
        # delete it" cost the build-side gate was added to avoid.
        with self._lock:
            running = self.jobs.get(self._running) if self._running else None
            building = running is not None and running.site == BUILD_SITE_KEY
        if building:
            return Response.error(
                "a build is in flight, and the run it is about to start needs the "
                "slot this one would take", 409,
                fix="wait for it to finish - it starts its own run when the "
                    "issues are written",
            )
        try:
            values = {k: str(v) for k, v in (json.loads(body or b"{}").get("values") or {}).items()}
            job = self.runs.start(values)
        except SwarmRunError as exc:
            return Response.error(str(exc), 409 if "in flight" in str(exc) else 400,
                                  fix=exc.fix)
        except json.JSONDecodeError as exc:
            return Response.error(f"bad request body: {exc}", 400)
        # A GitHub run that started is a project the operator will come back
        # to, so it is written down here - the layer that owns the store -
        # rather than inside `SwarmRuns`, which manages processes and should
        # not learn bookkeeping. Local runs record a filesystem path in the
        # repo field and are not projects. Best-effort, out loud: the run is
        # already in flight, and a projects-file hiccup must not report a
        # started run as a failure to start.
        if values.get("local") != "1":
            try:
                self.projects.record_run(
                    (values.get("repo") or "").strip(),
                    objective=(values.get("objective") or "").strip(),
                    stack=(values.get("stack") or "").strip(),
                    verify=(values.get("verify") or "").strip(),
                )
            except Exception as exc:  # noqa: BLE001 - bookkeeping must not mask the run
                print(f"! projects: could not record {values.get('repo')!r}: {exc}",
                      file=sys.stderr)
        return Response.json(self.runs.status(job.id), 202)

    # -- Start building ---------------------------------------------------

    def _swarm_build(self, body: bytes) -> Response:
        """The plan on the screen, as a repository and a backlog.

        The plan travels as the **id of the call that produced it**, not as
        tasks posted back up from the browser. That is the whole point of the
        ticket made structural: the console writes the decomposition it
        returned, and no round trip through a page - or through anything
        pretending to be one - can substitute a different set of tasks between
        the operator reading them and GitHub receiving them. `job.result` is
        the exact payload `/status` served, so "the plan shown is the plan
        written" is a property of the code rather than a promise in a docstring.

        On a thread, and polled at `/status` like every other job, for the
        reason the module docstring gives about blocking POSTs: provisioning is
        a repository creation, a commit, a label sweep and a ruleset, and then
        one issue per task. That is minutes on a slow morning, and a browser
        showing nothing for minutes is a browser that gets reloaded.
        """
        try:
            data = json.loads(body or b"{}")
        except json.JSONDecodeError as exc:
            return Response.error(f"bad request body: {exc}", 400)

        values = {k: str(v) for k, v in (data.get("values") or {}).items()}
        # Free here, expensive afterwards - `console_build`'s own ordering rule,
        # applied to the fields the *run* needs. They are only read by
        # `_start_run`, minutes later and after a repository exists, so a blank
        # objective or a cap reading "ten" bought a repository and a backlog
        # and then failed to run. `Builder` cannot catch either: it derives its
        # prompt from the plan's reasoning when the objective is blank, so it
        # succeeds precisely where the run cannot.
        #
        # `check_run_values` rather than the same checks retyped: the refusals
        # and their fixes belong to `target` and `_cycles_flag`, and a second
        # copy here is the one that would still say "leave it empty" after the
        # original learned to say something else. The repository is not checked
        # - this build is about to create it.
        try:
            check_run_values(dict(values, repo="owner/name"))
        except SwarmRunError as exc:
            return Response.error(str(exc), 400, fix=exc.fix)
        try:
            source = self.jobs.get(validate_capture_id(str(data.get("plan", ""))))
        except ConsoleError as exc:
            return Response.error(str(exc), 400)
        if source is None:
            return Response.error(
                "no such call, so there is no plan to build", 404,
                fix="run the planner and press Start building on its answer",
            )
        if source.site != "planner" or source.state != "done":
            return Response.error(
                f"call {source.id} is a {source.site} call in state {source.state!r}, "
                "which is not a plan", 400,
                fix="press Start building on a finished planner call",
            )

        # The latch `_claim` holds covers what runs *in this process*, and a
        # build's own work ends the moment the issues are written - the loop it
        # then starts is a child, watched by `SwarmRuns`. So a second Start
        # building would sail past `_claim` while the first build's swarm was
        # still working, provision a second repository, and only then be
        # refused by `SwarmRuns.start` - with a repository and a backlog left
        # over for a human to delete. The gate has to be here, before anything
        # is created, and it has to name the live run: "something is already
        # running" that does not say what is not a refusal anyone can act on.
        if (live := self.runs.live()) is not None:
            where = live.progress.get("repo") or live.id
            return Response.error(
                f"the swarm is already building {where}, and a second build would "
                f"create a second repository and a second run against the same "
                f"workers and the same Ollama", 409,
                fix=f"stop the run on {where} first, or wait for it to finish",
            )

        try:
            job = self._claim(BUILD_SITE_KEY)
        except ConsoleBusy as exc:
            return Response.error(str(exc), 409, fix=exc.fix)

        threading.Thread(
            target=self._build, args=(source.result, values, job), daemon=True
        ).start()
        return Response.json(job.to_dict(), 202)

    def _build(self, result: Any, values: Mapping[str, str], job: Job) -> None:
        """`_work`, minus the capture - and the omission is deliberate.

        `_work` attaches `sink.last` to every job it finishes, which is right
        when the job *was* a model call. This one is not, and the recorder's
        last record is therefore the planner call that produced the plan
        several minutes ago. Attaching it would draw a "the call" card with
        timings and a raw response under a build that never spoke to a model,
        which is the single most misleading thing this page could render given
        what the ticket is about.
        """
        state, report, error = "error", None, None
        try:
            report = self.builder.run(result, values).to_dict()
            self._record(report, values)
            self._start_run(report, values)
            state = "done"
        except BuildError as exc:
            error = {"type": "BuildError", "message": str(exc), "fix": exc.fix,
                     "traceback": "", "checks": exc.checks}
        except Exception as exc:  # noqa: BLE001 - every failure belongs on the page
            error = {"type": type(exc).__name__, "message": str(exc),
                     "fix": _fix_for(exc), "traceback": traceback.format_exc()[-2000:]}
        finally:
            job.result = report
            job.error = error
            job.state = state
            with self._lock:
                self._running = ""

    def _start_run(self, report: dict[str, Any], values: Mapping[str, str]) -> None:
        """Hand the repository that now exists to the swarm, and follow it.

        The whole of #130's first criterion, and almost none of its code: the
        supervision it asks for - a child running the real `swarm run`, its log
        streamed to the page, `SIGINT` on Stop so `cli._loop` disposes the run's
        containers on the way out - is `SwarmRuns`, already built for the swarm
        tab and already the thing a terminal would do. Reimplementing the loop
        in this process would mean a second assembly of the fleet, the reaper
        and the three policies, which the ticket names as the thing not to grow.

        `exists=True` is not an optimisation. `SwarmRuns.start` otherwise asks
        GitHub whether the repository is there, and a repository created
        seconds ago is exactly the one GitHub is most likely to answer 404
        about - which would send this run down the greenfield branch and
        provision a **second** repository over the backlog just written. This
        console created it; there is nothing to ask.

        Best-effort, and out loud in the report rather than as an exception: by
        this line the repository and its issues are real. A build that called
        itself failed because Docker was down would be describing the wrong
        thing, and would send an operator looking for a repository that is
        sitting there waiting for `swarm run --repo ...`.
        """
        try:
            job = self.runs.start(
                {
                    # The repository the provisioner reported, never the form's
                    # owner/name guess: `ProvisionPlan` derives a name when the
                    # field is blank, and the run has to attach to what exists.
                    "repo": report["repo"],
                    "objective": (values.get("objective") or "").strip(),
                    # The gate in the commit that now exists, for the reason
                    # `Builder.run` gives for writing it into every `## Verify`.
                    "verify": report["verify_command"],
                    "stack": report["stack"],
                    "max_cycles": values.get("max_cycles", ""),
                    "auto_merge": values.get("auto_merge", ""),
                },
                known_to_exist=True,
            )
        except Exception as exc:  # noqa: BLE001 - a build that worked must not read as failed
            # Both halves, always. The exception's own fix names the cause -
            # a missing token, the wrong venv - and the second sentence names
            # the thing the operator would otherwise go looking for: a
            # repository that exists, with its backlog written, waiting.
            pickup = (
                f"the repository and its issues are there - "
                f"`swarm run --repo {report['repo']} --objective ...` picks the work up"
            )
            own = getattr(exc, "fix", "")
            report["run_error"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "fix": f"{own}; {pickup}" if own else pickup,
            }
            print(f"! build: {report['repo']} was created, but the run did not start: {exc}",
                  file=sys.stderr)
            return
        # The page follows this exactly as it follows a run fired from the
        # swarm tab - same `/swarm/status`, same Stop button, same view. Read
        # back through `status`, which takes `SwarmRuns`' lock: the watcher
        # thread is already appending log lines, and a snapshot taken beside it
        # can pair a `lines` slice with a `next` cursor past its end - which
        # costs the page the run's first lines, silently.
        report["run"] = self.runs.status(job.id)

    def _record(self, report: Mapping[str, Any], values: Mapping[str, str]) -> None:
        """File the new repository as a project, before the verdict is published.

        Here rather than in `Builder` for the reason `_swarm_start` files a
        started run here: this is the layer that owns the store, and
        provisioning should not learn bookkeeping.

        Before, not after, and that ordering is the same one `_work` writes
        down: the page renders the report the instant `state` becomes "done",
        and a selector still missing the repository at that moment is a
        repository the operator has to retype the name of. Best-effort and out
        loud - the repository and its issues are already real, so a store
        hiccup must not turn a finished build into a failed one.
        """
        try:
            self.projects.record_run(
                report["repo"],
                objective=(values.get("objective") or "").strip(),
                stack=report["stack"],
                verify=report["verify_command"],
            )
        except Exception as exc:  # noqa: BLE001 - bookkeeping must not mask the build
            print(f"! projects: could not record {report['repo']!r}: {exc}",
                  file=sys.stderr)

    def _swarm_stop(self, body: bytes) -> Response:
        try:
            job_id = validate_capture_id(str(json.loads(body or b"{}").get("id", "")))
            job = self.runs.stop(job_id)
        except (ConsoleError, SwarmRunError) as exc:
            return Response.error(str(exc), 404 if "no such" in str(exc) else 400)
        except json.JSONDecodeError as exc:
            return Response.error(f"bad request body: {exc}", 400)
        return Response.json(job.to_dict())

    def _swarm_reset(self, body: bytes) -> Response:
        """Give one task its retry budget back, from the board (#293).

        The gesture ADR 0002 describes and ADR 0005 moved into `swarm reset`,
        moved once more - to where the operator actually is when they learn they
        are needed. The run's decision report ends by printing a command, and a
        board that says "needs a human" beside a button is one fewer context
        switch than a board that says it beside an instruction.

        **The ref is taken verbatim from the card that offered it.** The store is
        keyed by ref, the board already carries it, and minting one here from an
        issue number would put the GitHub adapter in this handler - the thing
        `cli._reset` refuses to do and `test_framework_boundary` enforces below.
        `TaskRef` still validates it, because it arrives over HTTP.

        Refused while a run is live. A reset writes the counter the dispatcher
        is reading, and the cycle that read it a moment ago would dispatch on a
        budget the operator has since changed - the same single-flight argument
        `_swarm_build` makes, and the same answer: stop the run, reset, continue.
        """
        from .store.reset import reset_budget
        from .taskref import TaskRef

        try:
            asked = json.loads(body or b"{}")
        except json.JSONDecodeError as exc:
            return Response.error(f"bad request body: {exc}", 400)
        if (live := self.runs.live()) is not None:
            return Response.error(
                f"{live.id} is in flight; a reset now would change a counter it is reading",
                409,
                fix="stop the run, reset the task, then continue it",
            )
        repo = str(asked.get("repo", "")).strip()
        if not REPO_RE.match(repo):
            return Response.error("a reset needs the repository the task is in", 400,
                                  fix="owner/name, as the board shows it")
        try:
            ref = TaskRef(str(asked.get("ref", "")).strip())
        except ValueError as exc:
            return Response.error(str(exc), 400,
                                  fix="the ref as the board shows it, like #12")
        try:
            outcome = reset_budget(repo, ref)
        except (StoreError, ValueError) as exc:
            return Response.error(f"could not reset {ref}: {exc}", 400)
        return Response.json({"ok": outcome.changed, "message": outcome.sentence()})

    def _swarm_status(self, path: str) -> Response:
        _, _, query = path.partition("?")
        wanted = dict(part.split("=", 1) for part in query.split("&") if "=" in part)
        since = wanted.get("since", "0")
        try:
            job_id = validate_capture_id(wanted.get("id", ""))
            status = self.runs.status(job_id, since=int(since) if since.isdigit() else 0)
        except ConsoleError as exc:
            return Response.error(str(exc), 400)
        except SwarmRunError as exc:
            return Response.error(str(exc), 404)
        return Response.json(status)

    def _swarm_external(self, path: str) -> Response:
        """The latest artifacts-recorded run, whoever launched it.

        Read from `.swarm/runs/` exactly as `swarm runs` reads it, so a run
        started from a terminal - or surviving a console restart - still has
        an account on the page. 404 when nothing ever ran; the page treats
        that as "no external run", not as an error worth showing.
        """
        import urllib.parse

        from .console_external import latest_external

        _, _, query = path.partition("?")
        wanted = dict(part.split("=", 1) for part in query.split("&") if "=" in part)
        since = wanted.get("since", "0")
        try:
            latest = latest_external(
                since=int(since) if since.isdigit() else 0,
                run_id=urllib.parse.unquote(wanted.get("run", "")),
            )
        except Exception as exc:  # noqa: BLE001 - an unreadable artifacts root belongs on the page
            return Response.error(f"{type(exc).__name__}: {exc}", 502)
        if latest is None:
            return Response.error("no recorded runs", 404)
        return Response.json(latest)

    def _swarm_outcome(self, path: str) -> Response:
        """How a run ended, once it has (#134). 404 until then.

        Keyed by run id rather than "the latest", because both views that ask
        hold one: the external card names the run it drew, and the console's
        own job carries the id it parsed out of its child's first line. A panel
        answered with a *different* run's ending is the wrong answer this
        route must never give, so `run_outcome` refuses an id it cannot find
        rather than falling back to the newest.

        404 is the ordinary case, not an error: a run that is still going has
        no ending to render, and the page treats "not yet" as nothing to draw.
        """
        import urllib.parse

        from .console_external import run_outcome

        _, _, query = path.partition("?")
        wanted = dict(part.split("=", 1) for part in query.split("&") if "=" in part)
        try:
            ended = run_outcome(urllib.parse.unquote(wanted.get("run", "")))
        except Exception as exc:  # noqa: BLE001 - an unreadable artifacts root belongs on the page
            return Response.error(f"{type(exc).__name__}: {exc}", 502)
        if ended is None:
            return Response.error("this run has not ended yet", 404)
        return Response.json(ended)

    def _swarm_board(self, path: str) -> Response:
        """One repository's tickets in lifecycle columns, read from GitHub.

        Read-only and rate-limit-cheap enough to poll; the caching that makes
        it so lives in `BoardReader`, which owns the whole projection.
        """
        import urllib.parse

        _, _, query = path.partition("?")
        wanted = dict(part.split("=", 1) for part in query.split("&") if "=" in part)
        repo = urllib.parse.unquote(wanted.get("repo", ""))
        try:
            return Response.json(self.board.read(repo))
        except BoardError as exc:
            return Response.error(str(exc), 400, fix=exc.fix)
        except Exception as exc:  # noqa: BLE001 - GitHub being down belongs on the page
            # A repository that is not there yet is the normal state of a
            # greenfield run's first minute, not a credentials problem.
            if getattr(exc, "status", None) == 404:
                return Response.error(
                    f"{repo} does not exist yet", 404,
                    fix="fire the run - it creates the repository, and this board follows",
                )
            return Response.error(
                f"{type(exc).__name__}: {exc}", 502,
                fix="is GITHUB_TOKEN exported in the console's shell, and does it reach this repo?",
            )

    def _swarm_worker(self, path: str) -> Response:
        """What one worker container is printing, right now (#133).

        Thin, like the board route beside it: which containers may be read,
        what bounds the answer and where the redaction happens are all
        `console_board.worker_log`'s, because they are the decisions - and a
        second copy of the "only our containers" rule at the transport is the
        copy that gets missed.

        404 rather than an error for an id the daemon does not list: a worker
        finishing between the board poll that drew the card and the click that
        expanded it is the ordinary case, not a fault.
        """
        import urllib.parse

        from .console_board import worker_log

        _, _, query = path.partition("?")
        wanted = dict(part.split("=", 1) for part in query.split("&") if "=" in part)
        container = urllib.parse.unquote(wanted.get("container", ""))
        try:
            log = worker_log(container)
        except BoardError as exc:
            return Response.error(str(exc), 400, fix=exc.fix)
        except Exception as exc:  # noqa: BLE001 - an unreachable daemon belongs on the page
            return Response.error(
                f"{type(exc).__name__}: {exc}", 502,
                fix="is the Docker daemon this console can see the one running the workers?",
            )
        if log is None:
            return Response.error(
                "no apiary container by that id", 404,
                fix="the worker has been disposed; its full log is in the run directory",
            )
        return Response.json(log)

    # -- projects -----------------------------------------------------------
    #
    # Thin, like the swarm routes: what a project is, how the rows are
    # ordered, and what makes a submission acceptable all live in
    # `console_projects`, which is testable without a socket. Here is only the
    # translation of `ProjectError` into the `{error, fix}` shape every other
    # refusal on this page uses.

    def _projects_list(self) -> Response:
        try:
            return Response.json({"projects": self.projects.list()})
        except Exception as exc:  # noqa: BLE001 - an unreadable store belongs on the page
            return Response.error(
                f"{type(exc).__name__}: {exc}", 502,
                fix=f"is {self.projects.path} readable and not corrupt?",
            )

    def _projects_save(self, body: bytes) -> Response:
        try:
            stored = self.projects.submit(json.loads(body or b"{}"))
        except ProjectError as exc:
            return Response.error(str(exc), 400, fix=exc.fix)
        except json.JSONDecodeError as exc:
            return Response.error(f"bad request body: {exc}", 400)
        return Response.json(stored)

    def _projects_history(self, path: str) -> Response:
        """One project's prompts, newest first - each recorded run is one.

        What the page shows when a project is selected: the objectives this
        repository has already been asked for, read from the same run
        artifacts `swarm runs` reads, so a run fired from a terminal is in
        the history too and nothing is stored twice.
        """
        import urllib.parse

        _, _, query = path.partition("?")
        wanted = dict(part.split("=", 1) for part in query.split("&") if "=" in part)
        repo = urllib.parse.unquote(wanted.get("repo", ""))
        try:
            return Response.json({"repo": repo, "prompts": self.projects.history(repo)})
        except ProjectError as exc:
            return Response.error(str(exc), 400, fix=exc.fix)
        except Exception as exc:  # noqa: BLE001 - an unreadable runs root belongs on the page
            return Response.error(f"{type(exc).__name__}: {exc}", 502)


def _role_for(site: Site) -> str:
    """Which role's model a site's call will use.

    `edits` is the worker - it is the whole-file generation a worker does
    inside its container - and everything else here is orchestration.
    """
    return "worker" if site.key == "edits" else "orchestrator"


def _model_label(site: Site, spec: Any) -> str:
    """What to show as "which model answered", before the call is made."""
    if spec is not None:
        return str(spec.label)
    from .models import resolve

    try:
        return str(resolve(_role_for(site)).spec.label)
    except ConfigError as exc:  # pragma: no cover - reported by the call itself
        return f"unresolved ({exc})"


def _models_payload() -> dict[str, Any]:
    """What each role will use, where that was decided, and what else it could be.

    The first two are the point: switching models used to be an export and a
    restart, which in practice meant nobody tried the alternative and the
    choice never got revisited. Showing the *source* alongside the model is
    what makes the saved default discoverable at all - it is the one rung an
    operator cannot see from a shell.
    """
    from .llm import PROVIDERS
    from .models import ROLES as MODEL_ROLES, resolve, store

    roles: dict[str, Any] = {}
    for role in MODEL_ROLES:
        try:
            resolution = resolve(role)
        except ConfigError as exc:
            roles[role] = {"error": str(exc)}
            continue
        roles[role] = {
            "label": resolution.spec.label,
            "provider": resolution.spec.provider,
            "model": resolution.spec.model,
            "options": dict(resolution.spec.options),
            "credential": resolution.spec.credential,
            "source": resolution.source,
            "detail": resolution.detail,
        }
    saved = store().all()
    return {
        "roles": roles,
        "saved": {role: spec.label for role, spec in saved.items()},
        "providers": [
            {
                "name": name,
                "options": [
                    {"name": option.name, "default": option.default, "doc": option.doc}
                    for option in provider.options
                ],
            }
            for name, provider in PROVIDERS.items()
        ],
    }


def _fix_for(exc: BaseException) -> str:
    """What to do about it, in doctor's sense: a failure that names no fix is
    a failure an operator has to go and research."""
    # An exception that arrives carrying its own fix - `IntakeError`, like
    # `SwarmRunError` - already knows better than any pattern below.
    named = getattr(exc, "fix", "")
    if isinstance(named, str) and named:
        return named
    text = f"{type(exc).__name__}: {exc}".lower()
    base = SETTINGS.ollama_base_url
    # Credentials first, and before the connection patterns below: a remote
    # provider that has no key never opens a socket, and "start Ollama" is the
    # answer this function used to give for it - advice about a server the
    # operator is not using. `swarm doctor` says the same three things
    # (`model.reachable`), and a failure on this page should not need a second
    # tool to interpret.
    if "is not set" in text and "_key" in text:
        return ("export the API key this provider reads, or name a different variable in the "
                "model's `api_key_env` option; `swarm doctor` reports it as model.reachable")
    if "needs a region" in text or "could not authenticate" in text:
        return ("configure AWS for this shell - `aws sso login --profile <name>` - and give the "
                "model `profile=<name>,region=<region>`; `swarm doctor` reports it as "
                "model.reachable")
    if "does not install by default" in text:
        return "install the provider's extra: `pip install -e \".[openai]\"` or `\".[bedrock]\"`"
    if "connection" in text or "refused" in text or "connect" in text:
        return f"start Ollama - `ollama serve`, or launch the app - and confirm with `curl {base}/api/version`"
    if "not found" in text or "no such model" in text or "pull" in text:
        return f"pull the model: `ollama pull {SETTINGS.worker_model}` (or {SETTINGS.orchestrator_model})"
    if "timeout" in text or "timed out" in text:
        return "the model is loading or the context is too large; retry, or lower SWARM_WORKER_CTX"
    if isinstance(exc, ConsoleError):
        return "check the checkout path and the file list"
    return f"check that {base} is the Ollama you meant; `swarm doctor` checks the rest"


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------


def _handler(console: Console) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "apiary-console"
        protocol_version = "HTTP/1.1"
        #: One held-open connection must not be able to stall the console.
        timeout = 30

        def _respond(self, response: Response) -> None:
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            #: **Nothing this console serves may be cached** (#293), and the
            #: assets are the reason. `asset()` deliberately re-reads its file
            #: per request "so that editing `app.js` and reloading the browser
            #: shows the change - the console is a developer tool"; the browser
            #: then cached `/app.js` at a stable URL and defeated the whole
            #: point. Observed live: three new controls shipped, verified being
            #: served by `curl`, and invisible on the page through several
            #: restarts of the server.
            #:
            #: On every response rather than only the assets, because the same
            #: argument covers the rest of it: every other route is live state -
            #: a run's log, a board, a project list - and a cached answer about
            #: any of them is a page reporting a past that has moved on. There
            #: is no response here worth caching and no bandwidth to save on
            #: loopback.
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(response.body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's contract
            self._respond(console.render("GET", self.path, dict(self.headers)))

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            #: `Content-Length` is whatever the client claimed. A prompt is
            #: large, a request body is not: cap before reading, not after.
            if length > 4_000_000:
                self._respond(Response.error("request body too large", 413))
                return
            self._respond(console.render("POST", self.path, dict(self.headers), self.rfile.read(length)))

        def log_message(self, fmt: str, *args: Any) -> None:
            """Method and path only. The default logs the whole request line,
            and a query string is exactly where a prompt must never be."""
            return

    return Handler


def _ollama_note() -> list[str]:
    """What the startup banner should say about Ollama, checked not assumed.

    Two GETs against the server, reusing doctor's `HostInference` rather than
    reimplementing its probe. Worth the half-second: without it the first sign
    that Ollama is down or a model is unpulled arrives *after* the operator has
    filled in a form and waited, and the whole point of this tool is to shorten
    that loop. Never fatal - a console that refused to start because a probe
    failed would be harder to use than one that says so and serves anyway.
    """
    from .doctor import HostInference

    probe = HostInference()
    try:
        version = probe.version()
    except Exception as exc:  # noqa: BLE001 - the reason is the message
        return [
            f"ollama {probe.base_url} — UNREACHABLE ({type(exc).__name__})",
            "fix: start it with `ollama serve`, or launch the Ollama app",
        ]

    lines = [f"ollama {probe.base_url} — v{version}"]
    try:
        installed = set(probe.installed())
    except Exception:  # noqa: BLE001 - reachable but not listable; not worth a scene
        return lines
    missing = [m for m in (SETTINGS.orchestrator_model, SETTINGS.worker_model) if m not in installed]
    lines += [f"{model} is NOT pulled — fix: `ollama pull {model}`" for model in missing]
    return lines


def serve(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    directory: Path | None = None,
    forever: bool = True,
) -> Console:
    """Bind, install a console recorder, and serve until interrupted."""
    if host not in LOOPBACK:
        raise ConfigError(
            f"the console refuses to bind {host!r}: it serves captured prompts, which are "
            f"whole files from the repository under test, and a worker container can reach "
            f"any port on the host gateway. Bind {DEFAULT_HOST} instead."
        )

    console = Console(host=host, port=port, sink=Recorder.for_console(directory or console_root()))
    capture_mod.set_recorder(console.sink)
    server = ThreadingHTTPServer((host, port), _handler(console))
    console.port = server.server_address[1]

    print(f"» console on http://{host}:{console.port}")
    print(f"  · orchestrator {SETTINGS.orchestrator_model} · worker {SETTINGS.worker_model}")
    for line in _ollama_note():
        print(f"  · {line}")
    print(f"  · captures in {console.sink.directory}")
    print("  · ctrl-c to stop")
    if forever:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n» stopped")
        finally:
            server.server_close()
    return console


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------


#: The front end, as three ordinary files beside this module.
#:
#: It began as one Python string holding the whole page - 320 lines of HTML,
#: CSS and JavaScript inside `console.py`. That kept the server to a single
#: file and needed no build step, and it cost more than it saved: the front end
#: was invisible in a file listing, unreviewable in a diff, had no highlighting
#: or linting, and could only be pinned by tests asserting on substrings of a
#: Python literal. The first person to look for "the web part" could not find
#: it.
#:
#: Still no build step and still nothing fetched from a network - the files are
#: served from disk, on loopback, by the same handler. An allow-list, not a
#: directory: serving a path the caller names is how a static route becomes a
#: traversal, and there are three files.
ASSETS = Path(__file__).parent / "console_assets"

ASSET_TYPES = {
    "/app.css": "text/css; charset=utf-8",
    "/app.js": "text/javascript; charset=utf-8",
}


def asset(name: str) -> str:
    """One named asset's text, read at request time.

    Read per request rather than cached at import so that editing `app.js` and
    reloading the browser shows the change - the console is a developer tool,
    and the file is a few kilobytes.
    """
    return (ASSETS / name).read_text(encoding="utf-8")


def page() -> str:
    """The page shell.

    Every value that came from a model or from the operator reaches the DOM
    through `textContent`, in `app.js`. The model's output originates in
    whatever repository it was pointed at, so treating it as markup is how a
    README becomes script running on this origin.
    """
    return asset("index.html")
