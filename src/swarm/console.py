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
"""

from __future__ import annotations

import json
import re
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

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "LOOPBACK",
    "Console",
    "Response",
    "SITES",
    "Site",
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


def _stack_prompt(payload: Mapping[str, str]) -> tuple[str, str]:
    from .greenfield.bootstrap import prompt_for

    return prompt_for(payload.get("brief", ""))


def _stack_run(payload: Mapping[str, str]) -> Any:
    from .greenfield.bootstrap import choose_stack

    return {"stack": choose_stack(payload.get("brief", ""))}


SITES: dict[str, Site] = {
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
            return Response.html(page(self.port))
        if method == "GET" and path == "/sites":
            return Response.json({"sites": [s.to_dict() for s in SITES.values()],
                                  "models": {"orchestrator": SETTINGS.orchestrator_model,
                                             "worker": SETTINGS.worker_model,
                                             "base_url": SETTINGS.ollama_base_url}})
        if method == "POST" and path == "/prompt":
            return self._prompt(body)
        if method == "POST" and path == "/run":
            return self._run(body)
        if method == "GET" and path.startswith("/status"):
            return self._status(path)
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


def _fix_for(exc: BaseException) -> str:
    """What to do about it, in doctor's sense: a failure that names no fix is
    a failure an operator has to go and research."""
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


def page(port: int = DEFAULT_PORT) -> str:
    """One file, no build step, no external request.

    Every value that came from a model or from the operator reaches the DOM
    through `textContent`. The model's output originates in whatever repository
    it was pointed at, so treating it as markup is how a README becomes script
    running on this origin.
    """
    return _PAGE.replace("__PORT__", str(port))


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>apiary console</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #fbfbfa; --fg: #1b1b19; --muted: #6b6b66; --line: #e2e2dd;
    --card: #ffffff; --accent: #7a5cff; --bad: #b3261e; --ok: #1a7f37;
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#16161a; --fg:#e8e8e3; --muted:#9a9a93; --line:#2c2c31;
            --card:#1d1d22; --accent:#a48bff; --bad:#ff6b5e; --ok:#54d17a; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui, sans-serif; }
  header { padding:18px 22px; border-bottom:1px solid var(--line); display:flex;
           gap:14px; align-items:baseline; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:650; letter-spacing:.01em; }
  .meta { color:var(--muted); font-size:12.5px; font-family:var(--mono); }
  main { max-width:1180px; margin:0 auto; padding:22px; display:grid;
         grid-template-columns: minmax(300px, 400px) 1fr; gap:22px; align-items:start; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; }
  .tabs { display:flex; gap:6px; margin-bottom:14px; flex-wrap:wrap; }
  .tab { padding:6px 11px; border:1px solid var(--line); border-radius:999px;
         background:transparent; color:var(--muted); cursor:pointer; font-size:13px; }
  .tab[aria-selected="true"] { background:var(--accent); border-color:var(--accent); color:#fff; }
  label { display:block; font-size:12.5px; color:var(--muted); margin:12px 0 5px; }
  input, textarea { width:100%; padding:9px 10px; border:1px solid var(--line); border-radius:7px;
                    background:var(--bg); color:var(--fg); font:13px/1.5 var(--mono); }
  textarea { min-height:96px; resize:vertical; }
  .row { display:flex; gap:9px; margin-top:15px; }
  button.go { flex:1; padding:10px; border:0; border-radius:7px; background:var(--accent);
              color:#fff; font-weight:600; cursor:pointer; font-size:14px; }
  button.ghost { padding:10px 13px; border:1px solid var(--line); border-radius:7px;
                 background:transparent; color:var(--fg); cursor:pointer; font-size:14px; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .blurb { color:var(--muted); font-size:12.5px; margin:0 0 4px; }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.07em; color:var(--muted);
       margin:0 0 8px; font-weight:600; }
  pre { margin:0; padding:12px; background:var(--bg); border:1px solid var(--line);
        border-radius:7px; overflow:auto; max-height:440px; font:12.5px/1.5 var(--mono);
        white-space:pre-wrap; word-break:break-word; }
  .stack > * + * { margin-top:16px; }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11.5px;
          font-family:var(--mono); border:1px solid var(--line); color:var(--muted); }
  .pill.bad { color:var(--bad); border-color:var(--bad); }
  .pill.ok  { color:var(--ok);  border-color:var(--ok); }
  .err { border-color:var(--bad); }
  .err h2 { color:var(--bad); }
  .fix { margin-top:9px; font-size:13px; color:var(--fg); }
  .fix code { font-family:var(--mono); background:var(--bg); padding:1px 5px;
              border-radius:4px; border:1px solid var(--line); }
  .empty { color:var(--muted); font-size:13.5px; }
  .file + .file { margin-top:11px; }
  .file summary { cursor:pointer; font-family:var(--mono); font-size:12.5px; padding:3px 0; }
</style>
</head>
<body>
<header>
  <h1>apiary console</h1>
  <span class="meta" id="models"></span>
</header>
<main>
  <section class="card">
    <div class="tabs" id="tabs" role="tablist"></div>
    <p class="blurb" id="blurb"></p>
    <form id="form" autocomplete="off"></form>
    <div class="row">
      <button class="go" id="go">Fire</button>
      <button class="ghost" id="peek" title="Render the prompt without calling the model">Prompt</button>
    </div>
    <p class="blurb" id="hint" style="margin-top:12px"></p>
  </section>
  <section class="stack" id="out">
    <div class="card empty" id="idle">
      Pick a site, fill it in, and fire. <strong>Prompt</strong> renders exactly what would be
      sent without calling the model — worth reading while a cold model loads.
    </div>
  </section>
</main>
<script>
(function () {
  "use strict";
  var sites = [], current = null, timer = null;
  var $ = function (id) { return document.getElementById(id); };

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;   // never as markup
    return node;
  }
  function card(title, body, cls) {
    var box = el("div", "card" + (cls ? " " + cls : ""));
    if (title) box.appendChild(el("h2", null, title));
    box.appendChild(body);
    return box;
  }
  function pre(text) { return el("pre", null, text); }

  function api(path, payload) {
    var opts = { headers: { "Content-Type": "application/json" } };
    if (payload) { opts.method = "POST"; opts.body = JSON.stringify(payload); }
    return fetch(path, opts).then(function (r) {
      return r.json().then(function (b) { return { ok: r.ok, status: r.status, body: b }; });
    });
  }

  function values() {
    var out = {};
    current.fields.forEach(function (f) {
      var node = document.querySelector('[name="' + f.name + '"]');
      out[f.name] = node ? node.value : "";
    });
    return out;
  }

  function drawForm() {
    var form = $("form");
    form.textContent = "";
    $("blurb").textContent = current.blurb;
    current.fields.forEach(function (f) {
      form.appendChild(el("label", null, f.label));
      var node = el(f.kind === "area" ? "textarea" : "input");
      node.name = f.name;
      node.placeholder = f.placeholder || "";
      node.value = f.value || "";
      form.appendChild(node);
    });
  }

  function drawTabs() {
    var tabs = $("tabs");
    tabs.textContent = "";
    sites.forEach(function (site) {
      var b = el("button", "tab", site.label);
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", String(site.key === current.key));
      b.onclick = function () { current = site; drawTabs(); drawForm(); };
      tabs.appendChild(b);
    });
  }

  function show(nodes) {
    var out = $("out");
    out.textContent = "";
    nodes.filter(Boolean).forEach(function (n) { out.appendChild(n); });
  }

  function promptCard(p) {
    var body = el("div");
    body.appendChild(el("h2", null, "system"));
    body.appendChild(pre(p.system));
    body.appendChild(el("h2", null, "human"));
    body.appendChild(pre(p.human));
    return card("prompt · " + p.chars + " chars", body);
  }

  function errorCard(e) {
    var body = el("div");
    body.appendChild(pre(e.type + ": " + e.message));
    if (e.fix) {
      var fix = el("p", "fix");
      fix.appendChild(el("strong", null, "Try: "));
      fix.appendChild(el("code", null, e.fix));
      body.appendChild(fix);
    }
    if (e.traceback) {
      var d = el("details");
      d.appendChild(el("summary", null, "traceback"));
      d.appendChild(pre(e.traceback));
      body.appendChild(d);
    }
    return card("failed", body, "err");
  }

  function captureCard(c) {
    var body = el("div");
    var pills = el("div");
    [["model", c.model], ["schema", c.schema_name],
     ["total", c.total_s !== null ? c.total_s + "s" : "?"],
     ["load", c.load_s !== null ? c.load_s + "s" : "?"],
     ["in", c.prompt_tokens], ["out", c.output_tokens]].forEach(function (pair) {
      if (pair[1] === null || pair[1] === undefined || pair[1] === "") return;
      var p = el("span", "pill", pair[0] + " " + pair[1]);
      p.style.marginRight = "6px";
      pills.appendChild(p);
    });
    body.appendChild(pills);
    if (c.response && c.response.text) {
      body.appendChild(el("h2", null, "raw response"));
      body.appendChild(pre(c.response.text));
    }
    return card("the call", body);
  }

  function resultCard(site, r) {
    var body = el("div");
    if (site === "stack") {
      body.appendChild(el("pre", null, r.stack));
      return card("answer", body);
    }
    if (!r.edits || !r.edits.length) {
      body.appendChild(el("p", "empty", "the model returned no edits"));
      return card("answer", body);
    }
    r.edits.forEach(function (edit) {
      var d = el("details", "file");
      d.appendChild(el("summary", null, edit.path + "  ·  " + edit.chars + " chars"));
      d.appendChild(pre(edit.content));
      body.appendChild(d);
    });
    return card("answer · " + r.edits.length + " file(s)", body);
  }

  function waiting(job) {
    var body = el("div");
    body.appendChild(el("p", null, "calling the model — " + job.elapsed_s + "s elapsed"));
    body.appendChild(el("p", "empty",
      "A cold model loads before it generates; the first call of the day is the slow one."));
    return card("running", body);
  }

  function poll(id, promptNode) {
    api("/status?id=" + encodeURIComponent(id)).then(function (res) {
      var job = res.body;
      if (job.state === "running") {
        show([waiting(job), promptNode]);
        timer = setTimeout(function () { poll(id, promptNode); }, 1000);
        return;
      }
      $("go").disabled = false;
      show([
        job.state === "error" ? errorCard(job.error) : resultCard(job.site, job.result),
        job.capture ? captureCard(job.capture) : null,
        promptNode
      ]);
    });
  }

  function fire() {
    $("go").disabled = true;
    clearTimeout(timer);
    var payload = { site: current.key, values: values() };
    api("/prompt", payload).then(function (p) {
      var promptNode = p.ok ? promptCard(p.body) : null;
      if (!p.ok) { $("go").disabled = false; show([errorCard({ type: "bad input",
                    message: p.body.error, fix: p.body.fix })]); return; }
      show([card("running", el("p", null, "starting…")), promptNode]);
      api("/run", payload).then(function (r) {
        if (!r.ok) {
          $("go").disabled = false;
          show([errorCard({ type: "refused", message: r.body.error, fix: r.body.fix }), promptNode]);
          return;
        }
        poll(r.body.id, promptNode);
      });
    });
  }

  function peek() {
    api("/prompt", { site: current.key, values: values() }).then(function (p) {
      show([p.ok ? promptCard(p.body)
                 : errorCard({ type: "bad input", message: p.body.error, fix: p.body.fix })]);
    });
  }

  $("go").onclick = function (e) { e.preventDefault(); fire(); };
  $("peek").onclick = function (e) { e.preventDefault(); peek(); };

  api("/sites").then(function (res) {
    sites = res.body.sites;
    current = sites[0];
    var m = res.body.models;
    $("models").textContent = "worker " + m.worker + "  ·  orchestrator " + m.orchestrator
                            + "  ·  " + m.base_url;
    $("hint").textContent = "Captures are written for every call, successful or not.";
    drawTabs();
    drawForm();
  });
})();
</script>
</body>
</html>
"""
