"""The transport apiary reaches a customer's task system through.

ADR 0001 removes every tracker adapter from this repository: a customer's
Linear, Jira or GitHub Issues is reached through *their* MCP server, running
with credentials they authorized, and apiary holds a config field rather than
an integration. That decision needs exactly one piece of code to be real, and
this is it - connect, list the tools a server offers, call one of them by name
with an argument dict, and turn every way that can fail into a named exception.

**This module knows nothing about what the tools mean.** Which tool fulfils
intake, which one posts a comment, which one files a ticket - that is the
capability contract (#150), and it is deliberately absent here. The spike in
#143 already found that the shape ADR 0001 sketched does not survive contact
with real servers; a transport that had baked that shape in would have had to
be rewritten alongside it. So the surface is `call_tool(name, arguments)`, and
what a name means is somebody else's problem, permanently.

**Retry only what retrying can fix.** The same rule `github/client.py` reasons
through, for the same reason, with one deliberate divergence:

- A 5xx or a 429 is transient. Backoff, then try again.
- Any other 4xx is a verdict on the request. Retrying cannot change the answer,
  and the caller wants the error now rather than after the backoff schedule has
  run its course.
- **A 401 or 403 is never read as throttling.** `github/client.py` has to treat
  some 403s as rate limiting because GitHub overloads that status; MCP servers
  use 429. Here the failure that will actually happen in production is a
  credential that expired or an admin who revoked it (#143), so a 401 that got
  slept on instead of raised would present as a hung orchestrator and a tracker
  that silently stopped being updated. It surfaces immediately, and it says
  which variable to re-export.

**This client is called by code, not by a model.** The reconcile loop decides
when to call and with what; MCP is only what makes that tracker-agnostic. Two
consequences show up in the API. Tool results that carry `isError` are raised
rather than returned - the flag exists so a *model* can read its own mistake
and try again, and a control loop that treated a failed write as a successful
one would report a PR link nobody can see. And nothing here consults an
inference to pick a tool: roughly 40% of `propose_edits` calls on this host
emit broken output, and unlike bad code a bad tracker write is not caught by
CI.

**Auth is a static bearer credential, and that is not a simplification.** #143
probed the live authorization-server metadata for every tracker apiary targets
and none of them offers the `client_credentials` grant, so there is no
machine-to-machine OAuth flow to drive: the orchestrator holds a pre-minted
token, sends it as a header, and holds no refresh token. Linear additionally
advertises `jwt-bearer`, an assertion grant, and it is declined on purpose -
one mechanism for every tracker is worth more than a marginally better one for
a single tracker.

**Two transports, because the trackers need two.** Remote servers speak
streamable HTTP; the GitHub profile does not use one. #143 found that the
remote GitHub MCP server at `api.githubcopilot.com` advertises classic OAuth
scopes - the `ghp_`/`gho_`/`ghu_`/`ghr_` family that
`security.assert_scoped_token` refuses outright, because their scope is a verb
rather than a repository. The resolution is the *local stdio*
`github-mcp-server`, which takes apiary's existing fine-grained PAT from its
environment and talks to `api.github.com`. So the GitHub tracker needs no new
credential and no new egress hole, and `StdioTransport` is not an optional
extra: without it the priority tracker has no route at all.

**Egress is checked before the socket is opened.** A worker and the
orchestrator both sit on `internal: true` networks whose only route out is the
egress proxy, so an endpoint missing from `security.EGRESS_ALLOWLIST` fails as
`403 Filtered` from tinyproxy - a refusal that reads like the *server* denying
the request. `assert_endpoint_allowed` turns that into the sentence that names
the tuple to edit, before any request is made. Only `mcp.linear.app` is added:
a hole nobody needs is the quiet widening `tests/test_security.py` exists to
catch, and the GitHub path deliberately needs none.

Manual smoke test against a real server:

    APIARY_TRACKER_TOKEN=... python -m swarm.mcp.client https://mcp.linear.app/mcp
"""

from __future__ import annotations

import json
import os
import random
import select
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import IO, Any, Callable, Mapping, Protocol, Sequence

from ..security import DEFAULT_EGRESS, EgressPolicy, assert_no_provision_token

__all__ = [
    "McpError",
    "McpTransportError",
    "McpUnreachable",
    "McpEgressBlocked",
    "McpHTTPError",
    "McpAuthError",
    "McpRateLimitError",
    "McpProtocolError",
    "McpRpcError",
    "McpToolError",
    "Response",
    "Transport",
    "UrllibTransport",
    "StdioTransport",
    "RetryPolicy",
    "ServerInfo",
    "ToolSpec",
    "ToolResult",
    "McpClient",
    "assert_endpoint_allowed",
    "PROTOCOL_VERSION",
    "STDIO_SCHEME",
    "TRACKER_TOKEN_ENV",
    "TRACKER_ENDPOINT_ENV",
]

#: The MCP revision this client speaks, sent both in `initialize` and - as the
#: spec requires from the second request onward - in the `MCP-Protocol-Version`
#: header. A server that speaks something else answers `initialize` with its own
#: version; `ServerInfo.protocol_version` records what it said, so a mismatch is
#: visible in `doctor` output rather than showing up as a confusing 400 later.
PROTOCOL_VERSION = "2025-06-18"

CLIENT_NAME = "apiary-swarm"
CLIENT_VERSION = "0.1.0"

#: Where the pre-minted tracker credential lives by default. Deliberately not
#: `GITHUB_TOKEN`: that name already holds the code-host key, and ADR 0001's
#: whole point is that the tracker and the code host stop being the same thing.
#:
#: It is a *default* rather than a constant because the capability contract
#: (#150) names the variable per tracker - a customer on Linear and a customer
#: on Jira do not share a credential, and the spike's recommended config block
#: carries a `value_env` field for exactly that reason.
TRACKER_TOKEN_ENV = "APIARY_TRACKER_TOKEN"

#: Where the endpoint URL lives by default, same caveat.
TRACKER_ENDPOINT_ENV = "APIARY_TRACKER_MCP_URL"

#: A safety stop for the `tools/list` cursor loop. A well-behaved server stops
#: sending `nextCursor`; this only fires if one ever hands us a cycle, and
#: looping forever inside a reconcile cycle is a worse failure than raising.
MAX_PAGES = 100

#: How the `Mcp-Session-Id` header is spelled. Case-insensitive on the wire,
#: named once here so the request and the response agree.
SESSION_HEADER = "Mcp-Session-Id"

#: The endpoint prefix that means "this one is a subprocess, not a URL". A
#: scheme rather than a separate flag, because the endpoint string is what
#: appears in every error message and in `doctor` output, and
#: `stdio://github-mcp-server` reads correctly there where `None` does not.
STDIO_SCHEME = "stdio://"

#: What a stdio server is started with when the caller names no environment.
#: An explicit floor rather than `os.environ`: the orchestrator's environment
#: holds `GITHUB_TOKEN`, `APIARY_PROVISION_TOKEN` and whatever else compose
#: passed it, and handing all of that to a third-party binary is the same
#: mistake as handing it to a worker.
STDIO_BASE_ENV = ("PATH", "HOME", "LANG", "TZ")


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class McpError(RuntimeError):
    """Base for everything this module raises."""


class McpTransportError(McpError):
    """The request never produced an HTTP response - DNS, TCP, TLS, timeout.

    Retryable by definition: there is no response to interpret, so we cannot
    have offended the server.
    """


class McpUnreachable(McpTransportError):
    """Retries ran out and no response ever arrived.

    Separate from its parent because this is the one an operator reads. The
    parent is what a single failed attempt raises inside the retry loop, where
    a message about how to fix the configuration would be premature - the next
    attempt may well succeed.
    """

    def __init__(self, endpoint: str, reason: str, *, attempts: int) -> None:
        self.endpoint = endpoint
        self.reason = reason
        self.attempts = attempts
        if endpoint.lower().startswith(STDIO_SCHEME):
            fix = (
                "The server is a local subprocess, so this is the binary rather than the "
                "network - check that it is on PATH in the orchestrator image and that it "
                "starts by hand:\n"
                "    docker compose run --entrypoint sh orchestrator -c "
                "'command -v " + endpoint[len(STDIO_SCHEME):] + "'"
            )
        else:
            fix = (
                "The orchestrator's only route off its network is the egress proxy, so "
                "check that the endpoint is right and that the proxy is up:\n"
                "    docker compose ps egress-proxy\n"
                "    docker compose logs egress-proxy"
            )
        super().__init__(f"{endpoint} did not answer after {attempts} attempts ({reason}). {fix}")


class McpEgressBlocked(McpError):
    """The endpoint's host is not on the egress allowlist.

    Raised before any request is sent, because the alternative is tinyproxy's
    `403 Filtered` - which arrives looking exactly like the tracker refusing
    the credential, and sends whoever is debugging it to rotate a token that
    was never the problem.
    """

    def __init__(self, endpoint: str, host: str) -> None:
        self.endpoint = endpoint
        self.host = host
        super().__init__(
            f"{host} is not on the egress allowlist, so {endpoint} is unreachable from "
            f"the orchestrator: its network has no default route and the proxy denies "
            f"by default. Add the host to EGRESS_ALLOWLIST in src/swarm/security.py, "
            f"then paste the regenerated block into compose.yaml's `egress-allowlist` "
            f"config:\n"
            f"    python -c \"from swarm.security import EgressPolicy; "
            f"print(chr(10).join(EgressPolicy().filter_lines()))\"\n"
            f"APIARY_EGRESS_ALLOW does not help here: it is read by EgressPolicy and "
            f"by nothing that enforces anything."
        )


class McpHTTPError(McpError):
    """The server answered with a status we will not act on."""

    def __init__(
        self, status: int, method: str, url: str, body: bytes, *, hint: str = ""
    ) -> None:
        self.status = status
        self.method = method
        self.url = url
        self.body = body
        message = f"{method} {url} -> {status}: {_error_message(body)}"
        super().__init__(f"{message}. {hint}" if hint else message)


class McpAuthError(McpHTTPError):
    """401 or 403: the credential is absent, expired, or revoked.

    Never retried, and the reason is worth stating rather than assuming. #143
    established that this is the failure that will actually happen in
    production - static credentials expire and admins revoke them - so it is
    the one 4xx whose handling is load-bearing rather than theoretical. A
    backoff schedule applied to it would turn "your token died" into "the
    orchestrator seems slow".
    """

    def __init__(
        self,
        status: int,
        method: str,
        url: str,
        body: bytes,
        *,
        token_env: str,
        challenge: str | None = None,
    ) -> None:
        self.token_env = token_env
        self.challenge = challenge
        super().__init__(
            status,
            method,
            url,
            body,
            hint=(
                (f"The server said {challenge!r}. " if challenge else "")
                + f"The tracker credential in {token_env} was rejected. This is expiry "
                f"or revocation, not a transient failure, so it is not retried - mint a "
                f"new one and re-export it before the next run:\n"
                f"    export {token_env}=...   # then: docker compose run orchestrator ..."
            ),
        )


class McpRateLimitError(McpHTTPError):
    """Throttled, and waiting it out is not this client's decision.

    Raised when the server asked for longer than `RetryPolicy.max_wait_s`, or
    when the attempts ran out while still being throttled. `retry_after_s` is
    what the server asked for, so the caller can sleep, shed load, or stop.
    """

    def __init__(
        self, status: int, method: str, url: str, body: bytes, retry_after_s: float | None
    ) -> None:
        self.retry_after_s = retry_after_s
        super().__init__(status, method, url, body)


class McpProtocolError(McpError):
    """The answer was not a JSON-RPC message this client can read.

    A well-formed refusal is `McpRpcError`; this is for an answer that does not
    parse, carries the wrong id, or arrives with a content type the streamable
    HTTP transport does not define.
    """


class McpRpcError(McpError):
    """The server answered the JSON-RPC call with an `error` object.

    A real answer, not a failure of transport: the method does not exist, the
    params were wrong, the session is gone. Carries the server's own code and
    data so a caller can tell those apart without re-parsing a string.
    """

    def __init__(self, method: str, code: int | None, message: str, data: Any = None) -> None:
        self.method = method
        self.code = code
        self.data = data
        self.server_message = message
        super().__init__(f"{method} failed: {message}" + (f" (code {code})" if code is not None else ""))


class McpToolError(McpRpcError):
    """One tool call failed - either as a JSON-RPC error, or as `isError`.

    Both spellings mean the same thing to a caller that is code rather than a
    model, so they arrive as one exception. `result` is populated for the
    `isError` case, where the server did return content explaining itself and
    throwing it away would leave the caller with nothing to log.
    """

    def __init__(
        self,
        tool: str,
        message: str,
        *,
        code: int | None = None,
        data: Any = None,
        result: ToolResult | None = None,
    ) -> None:
        self.tool = tool
        self.result = result
        super().__init__(f"tools/call {tool}", code, message, data)


def _error_message(body: bytes) -> str:
    """Pull something human out of an error body, falling back to raw text.

    Servers disagree about the shape: a JSON-RPC `error.message`, a bare
    `{"error": "..."}`, an OAuth `error_description`, or HTML from a proxy that
    the request never got past.
    """
    text = body.decode("utf-8", "replace") if body else ""
    try:
        payload = json.loads(text)
    except ValueError:
        return text.strip()[:200]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or error.get("error_description") or "")
            if message:
                return message.strip()[:200]
        elif isinstance(error, str) and error:
            description = payload.get("error_description")
            joined = f"{error}: {description}" if description else error
            return joined.strip()[:200]
        message = payload.get("message")
        if isinstance(message, str) and message:
            return message.strip()[:200]
    return json.dumps(payload)[:200]


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Response:
    """One HTTP response, transport-agnostic.

    Deliberately a second copy of `github.client.Response` rather than an
    import of it. ADR 0001 exists to stop the tracker path and the code-host
    path being one thing; a shared private dataclass would be the first thread
    sewing them back together, and it is eleven lines.
    """

    status: int
    headers: Mapping[str, str]
    body: bytes

    def header(self, name: str, default: str | None = None) -> str | None:
        # Case-insensitive on purpose: a fake transport in a test hands us a
        # plain dict where `urllib` hands us a case-insensitive HTTPMessage.
        target = name.lower()
        for key, value in self.headers.items():
            if key.lower() == target:
                return value
        return default

    @property
    def content_type(self) -> str:
        return (self.header("Content-Type") or "").split(";")[0].strip().lower()


class Transport(Protocol):
    """The seam that keeps this module testable without a network or a token.

    An implementation returns 4xx and 5xx as ordinary `Response` objects - the
    retry policy needs the status and the headers to decide - and raises
    `McpTransportError` only when no response arrived at all.

    It is also where a second MCP transport lands. #143 concluded that the
    GitHub profile should run the *local stdio* `github-mcp-server`, because
    the remote one wants a token family `security.assert_scoped_token` refuses;
    that is a subprocess speaking the same JSON-RPC over pipes, so it fits here
    rather than requiring `McpClient` to be reshaped around it.
    """

    def send(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> Response: ...


class UrllibTransport:
    """Default transport, stdlib only.

    `urllib` raises on 4xx/5xx; we convert those back into responses so the
    single retry loop in `McpClient` sees every outcome the same way. It also
    honours `HTTP_PROXY`/`HTTPS_PROXY` from the environment, which is how a
    request reaches the egress proxy without this module knowing it exists.
    """

    def send(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> Response:
        request = urllib.request.Request(url, data=body, method=method)
        for key, value in headers.items():
            request.add_header(key, value)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return Response(response.status, dict(response.headers.items()), response.read())
        except urllib.error.HTTPError as exc:
            headers_out = dict(exc.headers.items()) if exc.headers else {}
            return Response(exc.code, headers_out, exc.read())
        except urllib.error.URLError as exc:
            raise McpTransportError(f"{method} {url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise McpTransportError(f"{method} {url}: timed out after {timeout}s") from exc


class StdioTransport:
    """A local MCP server as a subprocess, presented as if it were HTTP.

    The GitHub tracker profile runs here rather than over the network (#143),
    so this is a first-class transport and not a convenience. The mapping is
    the whole trick: newline-delimited JSON-RPC over a pipe carries no status
    code, so every successful exchange is reported as a `200` and every way the
    pipe can fail is reported as `McpTransportError`. `McpClient`'s retry loop
    then behaves the way it does for a remote server without knowing which kind
    it has - and the retry is not decorative here, because a crashed server is
    respawned on the next attempt.

    There is nothing to rate limit and nothing to authorize at this layer: the
    credential reaches the server through `env`, which is why `over_stdio`
    takes one and why `assert_no_provision_token` is run against it. A binary
    that speaks MCP is code apiary did not write, and the boot key must not
    reach it any more than it may reach a worker.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        spawn: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        if not command:
            raise ValueError("a stdio transport needs a command to run")
        self.command = list(command)
        self.env = dict(env or {})
        assert_no_provision_token(self.env)
        self.cwd = cwd
        self._spawn = spawn
        self._process: subprocess.Popen | None = None

    # --- lifecycle --------------------------------------------------------

    def start(self) -> subprocess.Popen:
        """Spawn the server if it is not already running.

        Lazy rather than done in `__init__` so that constructing a client is
        free of side effects, and so that a process that died between two
        reconcile cycles is replaced by the next call rather than poisoning
        every one after it.
        """
        if self._process is not None and self._process.poll() is None:
            return self._process
        try:
            self._process = self._spawn(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # Inherited, not captured. A captured stderr nobody drains
                # fills its pipe buffer and deadlocks the server mid-answer,
                # and the orchestrator's own log is where a server's complaints
                # are useful anyway.
                stderr=None,
                env=self.env,
                cwd=self.cwd,
                bufsize=0,
            )
        except OSError as exc:
            raise McpTransportError(
                f"could not start {self.command[0]!r}: {exc}. The MCP server binary has "
                f"to be on PATH inside the orchestrator image - see docs/security.md"
            ) from exc
        return self._process

    def close(self) -> None:
        """Terminate the server, best effort."""
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        for stream in (process.stdin, process.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - a server ignoring SIGTERM
            process.kill()

    # --- the Transport protocol -------------------------------------------

    def send(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> Response:
        """Write one JSON-RPC message and read the answer that matches its id.

        `method`, `url` and `headers` are ignored, which is the honest thing to
        do: none of them exists on a pipe. They stay in the signature because
        `Transport` is one protocol, and a second signature for the local case
        would push the difference up into `McpClient`, which is exactly the
        knowledge this class is here to absorb.
        """
        del method, url, headers
        if body is None:
            raise McpTransportError("a stdio request needs a body")

        request_id = _message_id(body)
        process = self.start()
        stdin, stdout = process.stdin, process.stdout
        if stdin is None or stdout is None:  # pragma: no cover - Popen always gives both
            raise McpTransportError("the stdio server has no pipes")

        try:
            stdin.write(body + b"\n")
            stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.close()
            raise McpTransportError(f"{self.command[0]}: the server closed its input ({exc})") from exc

        if request_id is None:
            # A notification. There is no answer to wait for, and waiting for
            # one would block until the next unrelated message arrived.
            return Response(202, {"Content-Type": "application/json"}, b"")

        line = self._read_matching(stdout, request_id, timeout)
        return Response(200, {"Content-Type": "application/json"}, line)

    def _read_matching(self, stdout: IO[bytes], request_id: Any, timeout: float) -> bytes:
        """Lines until one is the JSON-RPC answer to `request_id`.

        Skipping rather than failing on the others is required, not lenient: a
        server may emit log notifications and progress updates on the same pipe
        between the request and its answer, and a reader that took the first
        line would read one of those roughly whenever the server was chatty.
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise McpTransportError(
                    f"{self.command[0]}: no answer to request {request_id} within {timeout}s"
                )
            # `select` on the pipe rather than a bare blocking `readline`,
            # because a server that hangs must fail the cycle rather than the
            # whole run. POSIX only, which is what the orchestrator image is.
            ready, _, _ = select.select([stdout], [], [], remaining)
            if not ready:
                continue
            line = stdout.readline()
            if not line:
                self.close()
                raise McpTransportError(
                    f"{self.command[0]}: the server exited before answering request {request_id}"
                )
            candidate = line.strip()
            if not candidate:
                continue
            if _message_id(candidate) == request_id:
                return candidate


def _message_id(body: bytes) -> Any:
    """The `id` of a JSON-RPC message, or None if it has none or does not parse."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return payload.get("id") if isinstance(payload, Mapping) else None


# --------------------------------------------------------------------------
# Retry policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """How hard to try, and how long to be willing to wait.

    The same fields and the same defaults as `github.client.RetryPolicy`, on
    purpose: an operator who has learned what one of them means should not have
    to learn a second vocabulary for the other half of the same cycle.

    `max_wait_s` is the one that matters. A server that asks for a ten-minute
    wait is asking for something an orchestrator cannot distinguish from a
    hang, so past that threshold we raise and let the caller decide.
    """

    max_attempts: int = 4
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 30.0
    jitter: float = 0.1          # fraction of the delay, to desynchronise callers
    max_wait_s: float = 60.0     # refuse to sleep out a distant reset


# --------------------------------------------------------------------------
# What the server tells us
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ServerInfo:
    """What `initialize` came back with.

    `capabilities` and `raw` are kept whole rather than reduced to the fields
    this client happens to read today. A server that offers no `tools`
    capability is a configuration mistake worth reporting precisely, and #150's
    contract validation needs the same payload.
    """

    name: str
    version: str
    protocol_version: str
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    instructions: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def supports_tools(self) -> bool:
        return "tools" in self.capabilities


@dataclass(frozen=True)
class ToolSpec:
    """One entry from `tools/list`.

    `input_schema` is the JSON Schema the server published, and it is carried
    verbatim because it is the only machine-readable statement of what a tool's
    arguments must look like. The capability contract validates against it;
    this module does not, because a client that rejected an argument dict the
    server would have accepted is a client that makes a tracker unusable over
    a schema disagreement.
    """

    name: str
    title: str | None = None
    description: str | None = None
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ToolSpec:
        return cls(
            name=str(payload.get("name") or ""),
            title=payload.get("title"),
            description=payload.get("description"),
            input_schema=payload.get("inputSchema") or {},
            output_schema=payload.get("outputSchema"),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class ToolResult:
    """What a successful `tools/call` returned.

    Three renderings of one answer, because callers differ. `structured` is the
    typed object a server that publishes an `outputSchema` returns and is what
    the reconcile loop wants; `text` is the concatenation of the text blocks,
    which is what a log line wants; `raw` is everything, for the caller this
    module has not met yet.
    """

    tool: str
    content: Sequence[Mapping[str, Any]] = ()
    structured: Mapping[str, Any] | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Every `text` block, joined by newlines. Empty if there are none."""
        return "\n".join(
            str(block.get("text") or "")
            for block in self.content
            if isinstance(block, Mapping) and block.get("type") == "text"
        )


# --------------------------------------------------------------------------
# Egress
# --------------------------------------------------------------------------


def assert_endpoint_allowed(endpoint: str, policy: EgressPolicy | None = None) -> None:
    """Refuse an endpoint the egress proxy would refuse, and say how to fix it.

    Checked against `EgressPolicy()` - the *generated* allowlist, the one
    tinyproxy actually enforces - rather than `EgressPolicy.from_env()`. That
    is not an oversight: `APIARY_EGRESS_ALLOW` is documented in four places and
    read by nothing that filters anything, so honouring it here would let this
    check pass for a host the proxy then blocks, which is worse than not
    checking at all.
    """
    host = _hostname(endpoint)
    if not (policy or DEFAULT_EGRESS).allows(host):
        raise McpEgressBlocked(endpoint, host)


def _hostname(endpoint: str) -> str:
    """`https://mcp.linear.app:443/mcp` -> `mcp.linear.app`."""
    value = endpoint.strip().lower()
    if "://" in value:
        value = value.split("://", 1)[1]
    value = value.split("/", 1)[0].split("@")[-1]
    if value.startswith("["):  # bracketed IPv6 literal
        return value.partition("]")[0] + "]"
    return value.split(":", 1)[0]


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class McpClient:
    """A retrying, session-aware MCP client for one endpoint.

    Streamable HTTP: every JSON-RPC message is a POST, and the answer arrives
    either as `application/json` or as a one-message `text/event-stream`. Both
    are accepted because servers choose, not clients - Linear answers with SSE
    where a smaller server answers with JSON, and a client that handled one of
    them works against exactly half the ecosystem.

    Not thread-safe, and deliberately so: the request id counter and the
    session id are mutable state, and the reconcile loop is a single sequential
    cycle. One client per endpoint per loop.
    """

    def __init__(
        self,
        endpoint: str,
        token: str | None,
        *,
        transport: Transport | None = None,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        timeout_s: float = 60.0,
        egress: EgressPolicy | None = None,
        token_env: str = TRACKER_TOKEN_ENV,
        protocol_version: str = PROTOCOL_VERSION,
        client_name: str = CLIENT_NAME,
        client_version: str = CLIENT_VERSION,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        lowered = endpoint.lower()
        if lowered.startswith(STDIO_SCHEME):
            # No host, so no egress question: a subprocess reaches the network
            # on its own account and the allowlist applies to it exactly as it
            # applies to everything else in the container.
            if transport is None:
                raise ValueError(
                    f"{endpoint!r} names a local server, which needs a StdioTransport; "
                    f"use McpClient.over_stdio()"
                )
        elif lowered.startswith(("http://", "https://")):
            assert_endpoint_allowed(endpoint, egress)
        else:
            raise ValueError(
                f"endpoint must be an http(s) URL or a {STDIO_SCHEME} label, got {endpoint!r}"
            )

        self.endpoint = endpoint
        self.token_env = token_env
        self.retry = retry or RetryPolicy()
        self.protocol_version = protocol_version
        self._transport = transport or UrllibTransport()
        self._sleep = sleep
        self._timeout_s = timeout_s
        self._client_name = client_name
        self._client_version = client_version
        self._next_id = 1
        self._session_id: str | None = None
        self._server: ServerInfo | None = None

        self._headers: dict[str, str] = {
            # Both, because the server picks. Omitting `text/event-stream` is
            # answered with 406 by servers that only stream.
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": f"{client_name}/{client_version}",
        }
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        if headers:
            self._headers.update(headers)

    @classmethod
    def from_env(
        cls,
        endpoint: str | None = None,
        *,
        token_env: str = TRACKER_TOKEN_ENV,
        endpoint_env: str = TRACKER_ENDPOINT_ENV,
        env: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> McpClient:
        """Build a client from the environment, refusing early if it is not there.

        Both names are arguments rather than constants because the capability
        contract (#150) names them per tracker: a customer running two trackers
        holds two credentials, and one variable for both is how the wrong one
        gets sent.
        """
        source = os.environ if env is None else env
        endpoint = endpoint or source.get(endpoint_env)
        if not endpoint:
            raise McpError(
                f"no MCP endpoint given and {endpoint_env} is not set. ADR 0001 reaches "
                f"the task system through the customer's own MCP server, so this is "
                f"configuration rather than a default:\n"
                f"    export {endpoint_env}=https://mcp.example.com/mcp"
            )
        token = source.get(token_env)
        if not token:
            raise McpError(
                f"{token_env} is not set. The orchestrator authenticates to an MCP "
                f"server with a pre-minted static credential - it drives no OAuth flow "
                f"and holds no refresh token (#143) - so there is nothing to fall back "
                f"on:\n"
                f"    export {token_env}=..."
            )
        return cls(endpoint, token, token_env=token_env, **kwargs)

    @classmethod
    def over_stdio(
        cls,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        token_env: str = TRACKER_TOKEN_ENV,
        inherit: Sequence[str] = STDIO_BASE_ENV,
        source_env: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> McpClient:
        """A client speaking to a locally spawned MCP server.

        This is the GitHub path (#143): `github-mcp-server` reads a
        fine-grained PAT from its own environment, so apiary's existing token
        is passed through unchanged and `assert_scoped_token`'s guarantee
        survives - where the remote server would have demanded a classic token
        that `security.py` refuses on purpose.

        `env` is what the server gets *in addition to* `inherit`, and `inherit`
        is a short explicit list rather than the whole environment. Naming the
        four variables a subprocess genuinely needs costs one line and means a
        credential added to the orchestrator later does not silently become a
        credential a third-party binary holds.
        """
        source = os.environ if source_env is None else source_env
        child_env = {name: source[name] for name in inherit if name in source}
        child_env.update(env or {})
        transport = StdioTransport(command, env=child_env, cwd=cwd)
        return cls(
            f"{STDIO_SCHEME}{command[0]}",
            None,
            transport=transport,
            token_env=token_env,
            **kwargs,
        )

    # --- lifecycle --------------------------------------------------------

    def connect(self) -> ServerInfo:
        """Perform the `initialize` handshake. Idempotent.

        Two round trips, not one: MCP requires the client to follow the
        server's `initialize` response with an `initialized` notification
        before any other request, and a server that never receives it is
        entitled to refuse everything afterwards.
        """
        if self._server is not None:
            return self._server

        payload = self._rpc(
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                # Honest and minimal: this client neither samples nor takes
                # roots. Advertising a capability we do not implement invites a
                # server to make a request nothing here can answer.
                "capabilities": {},
                "clientInfo": {"name": self._client_name, "version": self._client_version},
            },
            allow_reconnect=False,
        )
        info = payload if isinstance(payload, Mapping) else {}
        announced = info.get("serverInfo") or {}
        server = ServerInfo(
            name=str(announced.get("name") or "unknown"),
            version=str(announced.get("version") or ""),
            protocol_version=str(info.get("protocolVersion") or self.protocol_version),
            capabilities=dict(info.get("capabilities") or {}),
            instructions=info.get("instructions"),
            raw=dict(info),
        )
        # Recorded only once the notification has landed. A half-finished
        # handshake that looked complete would make `connect` idempotent about
        # the wrong thing: every later call would return early, and the server
        # would go on refusing requests from a client it never saw initialize.
        self._notify("notifications/initialized")
        self._server = server
        return server

    @property
    def server(self) -> ServerInfo | None:
        """What `connect` learned, or None if it has not run."""
        return self._server

    @property
    def session_id(self) -> str | None:
        """The server's `Mcp-Session-Id`, if it issued one. Not every server does."""
        return self._session_id

    def close(self) -> None:
        """End the session if the server issued one, best effort.

        Best effort because the caller is finishing either way, and a failure
        to tear down a session must never be the exception that escapes a
        reconcile cycle. A server that does not implement `DELETE` answers 405,
        which the spec permits and which is not a problem.
        """
        session = self._session_id
        self._session_id = None
        self._server = None

        # A local server is a process, and the way to end one is to end it. It
        # has no session to DELETE and would read the attempt as a malformed
        # request.
        stop = getattr(self._transport, "close", None)
        if isinstance(self._transport, StdioTransport) and callable(stop):
            stop()
            return

        if not session:
            return
        try:
            self._transport.send(
                "DELETE",
                self.endpoint,
                {**self._headers, SESSION_HEADER: session},
                None,
                self._timeout_s,
            )
        except McpError:
            pass

    def __enter__(self) -> McpClient:
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # --- tools ------------------------------------------------------------

    def list_tools(self) -> list[ToolSpec]:
        """Every tool the server offers, following `nextCursor` to the end.

        The whole list, not a page. #150 has to decide which tool fulfils which
        capability and `doctor` has to prove the configured names exist, and
        both of those are questions about the complete set - a paginated answer
        would make "this tool is not there" indistinguishable from "it was on
        the page you did not ask for".
        """
        self.connect()
        tools: list[ToolSpec] = []
        cursor: str | None = None
        for _ in range(MAX_PAGES):
            params = {"cursor": cursor} if cursor else {}
            payload = self._rpc("tools/list", params)
            page = payload.get("tools") if isinstance(payload, Mapping) else None
            for entry in page or []:
                if isinstance(entry, Mapping) and entry.get("name"):
                    tools.append(ToolSpec.from_payload(entry))
            cursor = payload.get("nextCursor") if isinstance(payload, Mapping) else None
            if not cursor:
                return tools
        raise McpProtocolError(
            f"tools/list did not terminate after {MAX_PAGES} pages at {self.endpoint}"
        )

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> ToolResult:
        """Call one tool by name with an argument dict.

        The entire point of this module, and the reason it says nothing about
        what `name` is. The caller decides; #150 decides how a caller decides;
        no inference is consulted at any point.

        A server that reports failure - as a JSON-RPC error, or as a result
        carrying `isError` - raises `McpToolError` rather than returning
        something the caller has to remember to check. The flag exists so a
        model can read its own mistake and try again; a control loop that
        forgot to look at it would post a comment nobody received and record
        the task as reported.
        """
        if not name:
            raise ValueError("call_tool needs a tool name")
        self.connect()
        try:
            payload = self._rpc("tools/call", {"name": name, "arguments": dict(arguments or {})})
        except McpRpcError as exc:
            raise McpToolError(
                name, exc.server_message, code=exc.code, data=exc.data
            ) from exc

        body = payload if isinstance(payload, Mapping) else {}
        result = ToolResult(
            tool=name,
            content=list(body.get("content") or []),
            structured=body.get("structuredContent"),
            raw=dict(body),
        )
        if body.get("isError"):
            raise McpToolError(
                name,
                result.text or "the server reported the call failed and said nothing more",
                result=result,
            )
        return result

    # --- JSON-RPC ---------------------------------------------------------

    def _rpc(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        allow_reconnect: bool = True,
    ) -> Any:
        """One JSON-RPC request/response pair, retried per `RetryPolicy`."""
        request_id = self._next_id
        self._next_id += 1
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params:
            message["params"] = dict(params)

        try:
            response = self._request(method, message)
        except McpHTTPError as exc:
            # A 404 against a live session is the spec's way of saying the
            # session expired - servers are allowed to drop them, and a
            # long-running reconcile loop will outlive one. Re-initializing is
            # not a retry of a 4xx: the request that gets sent again carries a
            # new session, so it is a different request, and it happens once.
            if exc.status != 404 or not self._session_id or not allow_reconnect:
                raise
            self._session_id = None
            self._server = None
            self.connect()
            message["id"] = self._next_id
            self._next_id += 1
            response = self._request(method, message)

        return _rpc_result(method, response, message["id"])

    def _notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        """Send a notification - no id, and therefore no answer to wait for."""
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            message["params"] = dict(params)
        self._request(method, message)

    def _request(self, method: str, message: Mapping[str, Any]) -> Response:
        """One POST, retried according to `RetryPolicy`.

        The whole retry decision lives here so there is exactly one place to
        read when asking why the orchestrator hammered somebody's tracker.
        """
        body = json.dumps(message).encode("utf-8")
        url = self.endpoint
        last_reason = "no response"

        for attempt in range(1, self.retry.max_attempts + 1):
            last_attempt = attempt == self.retry.max_attempts
            headers = dict(self._headers)
            if self._session_id:
                headers[SESSION_HEADER] = self._session_id
            if method != "initialize":
                # The spec wants this from the second request onward, and only
                # then: sending it on `initialize` pins a version before the
                # server has had the chance to negotiate one.
                headers["MCP-Protocol-Version"] = self.protocol_version

            try:
                response = self._transport.send("POST", url, headers, body, self._timeout_s)
            except McpTransportError as exc:
                last_reason = str(exc)
                if last_attempt:
                    raise McpUnreachable(url, last_reason, attempts=attempt) from exc
                self._sleep(self._backoff(attempt))
                continue

            if response.status < 400:
                session = response.header(SESSION_HEADER)
                if session:
                    self._session_id = session
                return response

            if response.status in (401, 403):
                # Ahead of the throttling check on purpose. These two never
                # mean "wait": a 403 here is the credential or the proxy, and
                # reading either as a rate limit sleeps through the one failure
                # an operator has to be told about.
                raise McpAuthError(
                    response.status,
                    "POST",
                    url,
                    response.body,
                    token_env=self.token_env,
                    challenge=response.header("WWW-Authenticate"),
                )

            if response.status < 500 and response.status != 429:
                # A 4xx is a verdict on this request. Retrying cannot change
                # the answer, and the caller wants the error now.
                raise McpHTTPError(response.status, "POST", url, response.body)

            requested = _retry_after_s(response)
            if last_attempt:
                if response.status == 429:
                    raise McpRateLimitError(response.status, "POST", url, response.body, requested)
                raise McpHTTPError(response.status, "POST", url, response.body)
            if requested is not None and requested > self.retry.max_wait_s:
                raise McpRateLimitError(response.status, "POST", url, response.body, requested)
            self._sleep(requested if requested is not None else self._backoff(attempt))

        raise McpError(f"POST {url}: retry loop exited without a result")

    def _backoff(self, attempt: int) -> float:
        delay = min(self.retry.backoff_base_s * (2 ** (attempt - 1)), self.retry.backoff_cap_s)
        if self.retry.jitter:
            delay += random.uniform(0.0, self.retry.jitter * delay)
        return delay


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _rpc_result(method: str, response: Response, request_id: int) -> Any:
    """The `result` of the JSON-RPC message answering `request_id`.

    Streamable HTTP lets a server answer a single request with a whole SSE
    stream, so the message we want is not necessarily the first one on it -
    notifications and progress updates travel the same channel. Matching on the
    id rather than on position is the difference between reading the answer and
    reading whatever happened to be sent first.
    """
    message = _message_for(method, response, request_id)
    if "error" in message:
        error = message.get("error")
        if isinstance(error, Mapping):
            raise McpRpcError(
                method,
                error.get("code") if isinstance(error.get("code"), int) else None,
                str(error.get("message") or "no message"),
                error.get("data"),
            )
        raise McpRpcError(method, None, str(error))
    return message.get("result")


def _message_for(method: str, response: Response, request_id: int) -> Mapping[str, Any]:
    content_type = response.content_type
    if content_type == "text/event-stream":
        for candidate in _sse_messages(response.body):
            if candidate.get("id") == request_id:
                return candidate
        raise McpProtocolError(
            f"{method}: no event-stream message answered request id {request_id}"
        )

    if content_type and content_type != "application/json":
        raise McpProtocolError(
            f"{method}: expected application/json or text/event-stream, got {content_type!r}"
        )

    try:
        payload = json.loads(response.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise McpProtocolError(f"{method}: answer was not JSON: {exc}") from exc

    if isinstance(payload, list):
        # A batched answer. Rare from a server answering one request, but legal.
        for candidate in payload:
            if isinstance(candidate, Mapping) and candidate.get("id") == request_id:
                return candidate
        raise McpProtocolError(f"{method}: no message in the batch answered id {request_id}")

    if not isinstance(payload, Mapping):
        raise McpProtocolError(f"{method}: answer was {type(payload).__name__}, not an object")
    if payload.get("id") != request_id:
        raise McpProtocolError(
            f"{method}: answer carried id {payload.get('id')!r}, expected {request_id}"
        )
    return payload


def _sse_messages(body: bytes) -> list[Mapping[str, Any]]:
    """Every JSON object carried on `data:` lines of an SSE body.

    Deliberately forgiving: a `data:` line that is not JSON is skipped rather
    than raised on, because comment lines, keep-alives and `event:` names are
    ordinary traffic on this channel and none of them is an error.
    """
    messages: list[Mapping[str, Any]] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        joined = "\n".join(buffer)
        buffer.clear()
        try:
            payload = json.loads(joined)
        except ValueError:
            return
        if isinstance(payload, Mapping):
            messages.append(payload)

    for raw_line in body.decode("utf-8", "replace").splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            flush()
            continue
        if line.startswith(":"):
            continue
        field_name, _, value = line.partition(":")
        if field_name == "data":
            buffer.append(value[1:] if value.startswith(" ") else value)
    flush()
    return messages


def _retry_after_s(response: Response) -> float | None:
    """How long the server asked us to wait, if it said. None means "use backoff".

    Only the delta-seconds spelling of `Retry-After` is honoured. The HTTP-date
    spelling is legal and is not parsed here on purpose: it depends on the two
    clocks agreeing, and a skewed one produces either a wait of zero or a wait
    of hours, both of which are worse than the backoff schedule.
    """
    retry_after = response.header("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after.strip()))
        except ValueError:
            return None
    reset = response.header("X-RateLimit-Reset")
    if reset:
        try:
            return max(0.0, float(reset.strip()) - time.time())
        except ValueError:
            return None
    return None


if __name__ == "__main__":  # pragma: no cover - manual smoke test, see module docstring
    import sys

    with McpClient.from_env(sys.argv[1] if len(sys.argv) > 1 else None) as mcp:
        assert mcp.server is not None
        print(f"{mcp.server.name} {mcp.server.version} (MCP {mcp.server.protocol_version})")
        for tool in mcp.list_tools():
            print(f"  {tool.name:<32} {(tool.description or '').splitlines()[0][:80]}")
