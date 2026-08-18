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
from .console_intake import QUESTIONS as INTAKE_QUESTIONS
from .console_projects import ProjectError, ProjectStore
from .console_runs import SWARM_SITE, SwarmRunError, SwarmRuns

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
    run: Callable[[Mapping[str, str]], Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "blurb": self.blurb,
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


def _edits_run(payload: Mapping[str, str]) -> Any:
    from .worker.edit import propose_edits

    writable, readable = _sources(payload)
    output = propose_edits(payload.get("goal", ""), writable, readable)
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


def _plan_run(payload: Mapping[str, str]) -> Any:
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
            }
            for task in plan.tasks
        ],
    }


def _stack_prompt(payload: Mapping[str, str]) -> tuple[str, str]:
    from .greenfield.bootstrap import prompt_for

    return prompt_for(payload.get("brief", ""))


def _stack_run(payload: Mapping[str, str]) -> Any:
    from .greenfield.bootstrap import choose_stack

    return {"stack": choose_stack(payload.get("brief", ""))}


def _intake_prompt(payload: Mapping[str, str]) -> tuple[str, str]:
    from .console_intake import compose_brief, prompt_for

    return prompt_for(compose_brief(payload))


def _intake_run(payload: Mapping[str, str]) -> Any:
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
    capture: dict[str, Any] | None = None
    error: dict[str, str] | None = None

    def to_dict(self, *, now: float | None = None) -> dict[str, Any]:
        return {
            "id": self.id,
            "site": self.site,
            "state": self.state,
            "elapsed_s": round((now or time.monotonic()) - self.started, 1),
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

        if method == "GET" and path in ("/", "/index.html"):
            return Response.html(page())
        if method == "GET" and path in ASSET_TYPES:
            return Response(200, asset(path.lstrip("/")).encode("utf-8"), ASSET_TYPES[path])
        if method == "GET" and path == "/sites":
            return Response.json({"sites": [s.to_dict() for s in SITES.values()],
                                  # Its own key, not a fourth entry in `sites`:
                                  # nothing that iterates model-call sites may
                                  # pick up a form with no prompt behind it.
                                  "swarm": SWARM_SITE,
                                  "models": {"orchestrator": SETTINGS.orchestrator_model,
                                             "worker": SETTINGS.worker_model,
                                             "base_url": SETTINGS.ollama_base_url}})
        if method == "POST" and path == "/prompt":
            return self._prompt(body)
        if method == "POST" and path == "/run":
            return self._run(body)
        if method == "GET" and path.startswith("/status"):
            return self._status(path)
        if method == "POST" and path == "/swarm/start":
            return self._swarm_start(body)
        if method == "POST" and path == "/swarm/stop":
            return self._swarm_stop(body)
        if method == "GET" and path == "/swarm/latest":
            latest = self.runs.latest()
            return Response.json(latest) if latest else Response.error("no runs yet", 404)
        if method == "GET" and path.startswith("/swarm/status"):
            return self._swarm_status(path)
        if method == "GET" and path.startswith("/swarm/board"):
            return self._swarm_board(path)
        if method == "GET" and path.startswith("/swarm/external"):
            return self._swarm_external(path)
        if method == "GET" and path == "/projects":
            return self._projects_list()
        if method == "POST" and path == "/projects":
            return self._projects_save(body)
        if method == "GET" and path.startswith("/projects/history"):
            return self._projects_history(path)
        return Response.error(f"no route for {method} {path}", 404)

    def _payload(self, body: bytes) -> tuple[Site, dict[str, str]]:
        data = json.loads(body or b"{}")
        site = SITES.get(str(data.get("site", "")))
        if site is None:
            raise ConsoleError(f"unknown site: {data.get('site')!r}")
        return site, {k: str(v) for k, v in (data.get("values") or {}).items()}

    def _prompt(self, body: bytes) -> Response:
        try:
            site, values = self._payload(body)
            system, human = site.prompt(values)
        except ConsoleError as exc:
            return Response.error(str(exc), 400)
        except Exception as exc:  # noqa: BLE001 - a bad fixture is the operator's, not a crash
            return Response.error(f"{type(exc).__name__}: {exc}", 400)
        return Response.json({"system": system, "human": human,
                              "chars": len(system) + len(human)})

    def _run(self, body: bytes) -> Response:
        try:
            site, values = self._payload(body)
        except ConsoleError as exc:
            return Response.error(str(exc), 400)

        with self._lock:
            if self._running:
                running = self.jobs.get(self._running)
                return Response.error(
                    "a call is already in flight, and Ollama loads one model at a time",
                    409,
                    fix=f"wait for {running.site if running else 'it'} to finish, or reload the page",
                )
            job = Job(id=uuid.uuid4().hex[:16], site=site.key, started=time.monotonic())
            self.jobs[job.id] = job
            self._running = job.id

        threading.Thread(target=self._work, args=(site, values, job), daemon=True).start()
        return Response.json(job.to_dict(), 202)

    def _work(self, site: Site, values: Mapping[str, str], job: Job) -> None:
        """`job.state` is written last, and that ordering is load-bearing.

        The page polls `/status` from another thread. Publishing "done" or
        "error" before the result, the capture and the fix are attached lets a
        poll land in between and render a finished call with nothing in it -
        which is indistinguishable, on screen, from a model that answered
        nothing.
        """
        state, result, error = "error", None, None
        try:
            result = site.run(values)
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
        return Response.json(job.to_dict(), 202)

    def _swarm_stop(self, body: bytes) -> Response:
        try:
            job_id = validate_capture_id(str(json.loads(body or b"{}").get("id", "")))
            job = self.runs.stop(job_id)
        except (ConsoleError, SwarmRunError) as exc:
            return Response.error(str(exc), 404 if "no such" in str(exc) else 400)
        except json.JSONDecodeError as exc:
            return Response.error(f"bad request body: {exc}", 400)
        return Response.json(job.to_dict())

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
