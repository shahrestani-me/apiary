"""Unit tests for the MCP client, against a scripted transport.

No network, no token and no subprocess: every test drives `FakeTransport`,
which replays canned responses and records what was asked of it. That is what
the `Transport` seam is for, and the reason it matters more here than it does
for the GitHub client is that the retry classification is the deliverable. The
happy path is one assertion; the failure classification is most of this file.

**The expensive bug this file exists to catch is a 4xx that retries.** A 401
against a tracker is a credential that expired or an admin who revoked it
(#143), and it is the failure that will actually happen in production. Slept on
for a backoff schedule instead of raised, it presents as an orchestrator that
has gone quiet - so `test_a_401_is_never_retried` and its neighbours assert on
`transport.sent` and on `slept`, not only on the exception type. An exception
raised after four attempts and thirty seconds is the wrong behaviour even
though it is the right exception.

The stdio half is exercised through a fake `Popen` over real pipes, because the
thing that can go wrong there is framing - a reader that takes the first line
rather than the line answering its id, or one that hangs on a dead process.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pytest

from swarm.mcp.client import (
    PROTOCOL_VERSION,
    SESSION_HEADER,
    STDIO_SCHEME,
    TRACKER_ENDPOINT_ENV,
    TRACKER_TOKEN_ENV,
    McpAuthError,
    McpClient,
    McpEgressBlocked,
    McpError,
    McpHTTPError,
    McpProtocolError,
    McpRateLimitError,
    McpToolError,
    McpTransportError,
    McpUnreachable,
    Response,
    RetryPolicy,
    StdioTransport,
    UrllibTransport,
    assert_endpoint_allowed,
)
from swarm.security import EgressPolicy

#: On the allowlist, so the egress pre-flight is not what these tests measure.
ENDPOINT = "https://mcp.linear.app/mcp"


# --------------------------------------------------------------------------
# The mocked transport
# --------------------------------------------------------------------------


@dataclass
class SentRequest:
    method: str
    url: str
    headers: Mapping[str, str]
    body: bytes | None

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8")) if self.body else None

    @property
    def rpc_method(self) -> str:
        return (self.json() or {}).get("method", "")


@dataclass
class FakeTransport:
    """Replays scripted responses in order; records every request.

    A scripted entry may be an exception instance, which is raised instead of
    returned - that is how transport-level failures are simulated. A callable
    entry is called with the request, which is how a response that has to echo
    the request's id is built. Running out of script is an assertion failure
    rather than a default, because an unexpected extra request is exactly the
    bug these tests exist to catch.
    """

    script: list[Any]
    sent: list[SentRequest] = field(default_factory=list)

    def send(self, method, url, headers, body, timeout):
        request = SentRequest(method, url, dict(headers), body)
        self.sent.append(request)
        assert self.script, f"unscripted request: {method} {url} {body!r}"
        nxt = self.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if callable(nxt):
            nxt = nxt(request)
        return nxt


def rpc(payload: Any, *, status: int = 200, **headers: str) -> Any:
    """A JSON-RPC answer that echoes whatever id the request carried."""

    def build(request: SentRequest) -> Response:
        message = dict(payload)
        message.setdefault("jsonrpc", "2.0")
        message["id"] = (request.json() or {}).get("id")
        body = json.dumps(message).encode("utf-8")
        return Response(status, {"Content-Type": "application/json", **headers}, body)

    return build


def raw(status: int, body: Any = None, **headers: str) -> Response:
    """A response that is not a JSON-RPC answer - an error page, a 429, a 500."""
    encoded = b"" if body is None else (
        body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    )
    return Response(status, headers, encoded)


INITIALIZE = rpc(
    {
        "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": True}},
            "serverInfo": {"name": "linear", "version": "1.4.0"},
        }
    },
    **{SESSION_HEADER: "sess-1"},
)

#: The `initialized` notification the handshake sends after `initialize`.
NOTIFIED = Response(202, {}, b"")

HANDSHAKE = [INITIALIZE, NOTIFIED]


def client(*script: Any, **kwargs: Any) -> tuple[McpClient, FakeTransport, list[float]]:
    """A client wired to a scripted transport and a recording sleep."""
    transport = FakeTransport(list(script))
    slept: list[float] = []
    retry = kwargs.pop("retry", RetryPolicy(max_attempts=3, backoff_base_s=0.5, jitter=0.0))
    mcp = McpClient(
        kwargs.pop("endpoint", ENDPOINT),
        kwargs.pop("token", "test-credential"),
        transport=transport,
        retry=retry,
        sleep=slept.append,
        **kwargs,
    )
    return mcp, transport, slept


def connected(*script: Any, **kwargs: Any) -> tuple[McpClient, FakeTransport, list[float]]:
    """A client that has already completed the handshake."""
    mcp, transport, slept = client(*HANDSHAKE, *script, **kwargs)
    mcp.connect()
    transport.sent.clear()
    slept.clear()
    return mcp, transport, slept


# --------------------------------------------------------------------------
# The handshake
# --------------------------------------------------------------------------


def test_connect_initializes_then_notifies_and_keeps_the_session():
    mcp, transport, _ = client(*HANDSHAKE)
    info = mcp.connect()

    assert [request.rpc_method for request in transport.sent] == [
        "initialize",
        "notifications/initialized",
    ]
    assert info.name == "linear"
    assert info.version == "1.4.0"
    assert info.supports_tools
    assert mcp.session_id == "sess-1"


def test_connect_is_idempotent():
    """A second call is not a second handshake; every method calls it."""
    mcp, transport, _ = client(*HANDSHAKE)
    first = mcp.connect()
    assert mcp.connect() is first
    assert len(transport.sent) == 2


def test_a_handshake_that_did_not_finish_is_not_remembered_as_finished():
    """Otherwise `connect` becomes idempotent about the wrong thing.

    The server never saw the `initialized` notification, so it is entitled to
    refuse everything afterwards - and every later `connect` would return the
    cached answer instead of trying again.
    """
    mcp, transport, _ = client(INITIALIZE, McpTransportError("pipe"), McpTransportError("pipe"),
                               McpTransportError("pipe"))
    with pytest.raises(McpUnreachable):
        mcp.connect()
    assert mcp.server is None


def test_every_request_carries_the_credential_and_accepts_both_content_types():
    mcp, transport, _ = client(*HANDSHAKE)
    mcp.connect()

    sent = transport.sent[0]
    assert sent.method == "POST"
    assert sent.url == ENDPOINT
    assert sent.headers["Authorization"] == "Bearer test-credential"
    assert "application/json" in sent.headers["Accept"]
    assert "text/event-stream" in sent.headers["Accept"]


def test_the_protocol_version_header_is_absent_on_initialize_and_present_after():
    """Pinning a version before the server has negotiated one is the bug."""
    mcp, transport, _ = client(*HANDSHAKE)
    mcp.connect()

    initialize, notification = transport.sent
    assert "MCP-Protocol-Version" not in initialize.headers
    assert notification.headers["MCP-Protocol-Version"] == PROTOCOL_VERSION


def test_the_session_id_is_echoed_on_later_requests():
    mcp, transport, _ = connected(rpc({"result": {"tools": []}}))
    mcp.list_tools()
    assert transport.sent[0].headers[SESSION_HEADER] == "sess-1"


def test_a_server_that_issues_no_session_is_fine():
    """Not every server is stateful, and a client that required one works with few."""
    handshake = [
        rpc({"result": {"capabilities": {"tools": {}}, "serverInfo": {"name": "small"}}}),
        NOTIFIED,
    ]
    mcp, transport, _ = client(*handshake, rpc({"result": {"tools": []}}))
    mcp.connect()
    assert mcp.session_id is None
    assert mcp.list_tools() == []
    assert SESSION_HEADER not in transport.sent[-1].headers


# --------------------------------------------------------------------------
# Listing and calling
# --------------------------------------------------------------------------


TOOL_PAYLOAD = {
    "name": "create_comment",
    "title": "Create comment",
    "description": "Comment on an issue",
    "inputSchema": {"type": "object", "properties": {"issueId": {"type": "string"}}},
}


def test_list_tools_returns_specs_with_the_schema_intact():
    mcp, _, _ = connected(rpc({"result": {"tools": [TOOL_PAYLOAD]}}))
    tools = mcp.list_tools()

    assert [tool.name for tool in tools] == ["create_comment"]
    assert tools[0].input_schema["properties"]["issueId"]["type"] == "string"
    assert tools[0].raw == TOOL_PAYLOAD


def test_list_tools_follows_the_cursor_to_the_end():
    """A paginated answer would make "not offered" and "not on this page" the same."""
    page_one = rpc({"result": {"tools": [TOOL_PAYLOAD], "nextCursor": "c2"}})
    page_two = rpc({"result": {"tools": [{"name": "create_issue"}]}})
    mcp, transport, _ = connected(page_one, page_two)

    assert [tool.name for tool in mcp.list_tools()] == ["create_comment", "create_issue"]
    assert "params" not in transport.sent[0].json()
    assert transport.sent[1].json()["params"] == {"cursor": "c2"}


def test_list_tools_skips_entries_with_no_name():
    mcp, _, _ = connected(rpc({"result": {"tools": [{"description": "nameless"}, TOOL_PAYLOAD]}}))
    assert [tool.name for tool in mcp.list_tools()] == ["create_comment"]


def test_a_cursor_that_never_terminates_raises_rather_than_looping():
    forever = [rpc({"result": {"tools": [], "nextCursor": "same"}}) for _ in range(200)]
    mcp, _, _ = connected(*forever)
    with pytest.raises(McpProtocolError, match="did not terminate"):
        mcp.list_tools()


def test_call_tool_sends_the_name_and_the_arguments_verbatim():
    """The whole surface: a name the caller chose and a dict it built."""
    answer = rpc(
        {
            "result": {
                "content": [{"type": "text", "text": "posted"}],
                "structuredContent": {"id": "cmt_1"},
            }
        }
    )
    mcp, transport, _ = connected(answer)
    result = mcp.call_tool("create_comment", {"issueId": "ENG-1", "body": "hi"})

    params = transport.sent[0].json()["params"]
    assert params == {"name": "create_comment", "arguments": {"issueId": "ENG-1", "body": "hi"}}
    assert result.text == "posted"
    assert result.structured == {"id": "cmt_1"}
    assert result.tool == "create_comment"


def test_call_tool_without_arguments_still_sends_a_dict():
    mcp, transport, _ = connected(rpc({"result": {"content": []}}))
    mcp.call_tool("list_issues")
    assert transport.sent[0].json()["params"]["arguments"] == {}


def test_call_tool_needs_a_name():
    mcp, _, _ = connected()
    with pytest.raises(ValueError):
        mcp.call_tool("")


def test_no_inference_is_consulted_to_pick_a_tool():
    """ADR 0001's programmatic call site, asserted rather than assumed.

    A model in this path would duplicate tickets and invent keys, and unlike
    bad code a bad tracker write is not caught by CI. The check is structural:
    nothing in the module imports the LLM seam.
    """
    import swarm.mcp.client as module

    source = (module.__file__ or "").replace(".pyc", ".py")
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "from ..llm" not in text and "import llm" not in text
    assert "langchain" not in text and "ollama" not in text.lower()


# --------------------------------------------------------------------------
# Failures the server describes
# --------------------------------------------------------------------------


def test_a_tool_reporting_is_error_raises_with_its_own_words():
    """`isError` exists so a model can retry; a control loop must not ignore it."""
    answer = rpc(
        {"result": {"content": [{"type": "text", "text": "issue ENG-9 not found"}], "isError": True}}
    )
    mcp, _, _ = connected(answer)
    with pytest.raises(McpToolError) as caught:
        mcp.call_tool("create_comment", {"issueId": "ENG-9"})

    assert "issue ENG-9 not found" in str(caught.value)
    assert caught.value.tool == "create_comment"
    # The content survives, because it is the only explanation there is.
    assert caught.value.result is not None
    assert caught.value.result.text == "issue ENG-9 not found"


def test_a_tool_reporting_is_error_with_nothing_to_say_still_raises():
    mcp, _, _ = connected(rpc({"result": {"content": [], "isError": True}}))
    with pytest.raises(McpToolError, match="said nothing more"):
        mcp.call_tool("create_comment")


def test_a_jsonrpc_error_on_a_tool_call_becomes_a_tool_error():
    answer = rpc({"error": {"code": -32602, "message": "unknown tool: nope", "data": {"x": 1}}})
    mcp, _, _ = connected(answer)
    with pytest.raises(McpToolError) as caught:
        mcp.call_tool("nope")

    assert caught.value.code == -32602
    assert caught.value.data == {"x": 1}
    assert "unknown tool" in str(caught.value)


def test_a_jsonrpc_error_is_not_retried():
    """The server answered. Asking again gets the same answer, slower."""
    mcp, transport, slept = connected(rpc({"error": {"code": -32601, "message": "no such method"}}))
    with pytest.raises(McpToolError):
        mcp.call_tool("whatever")
    assert len(transport.sent) == 1
    assert slept == []


# --------------------------------------------------------------------------
# Retry classification - the part that is expensive to get wrong
# --------------------------------------------------------------------------


def test_a_401_is_never_retried_and_says_which_variable_to_re_export():
    """The failure #143 says will actually happen: expiry or revocation.

    Retried, it becomes an orchestrator that appears to hang while a tracker
    silently stops being updated. So this asserts on the attempt count as much
    as on the type.
    """
    mcp, transport, slept = connected(
        raw(401, {"error": "invalid_token"}, **{"WWW-Authenticate": 'Bearer error="invalid_token"'})
    )
    with pytest.raises(McpAuthError) as caught:
        mcp.call_tool("create_comment")

    assert len(transport.sent) == 1, "a 401 was retried"
    assert slept == []
    assert caught.value.status == 401
    assert TRACKER_TOKEN_ENV in str(caught.value)
    assert "export" in str(caught.value)
    assert caught.value.challenge == 'Bearer error="invalid_token"'


def test_a_403_is_an_auth_failure_and_is_never_read_as_throttling():
    """The deliberate divergence from `github/client.py`.

    GitHub overloads 403 for rate limiting; MCP servers use 429. Reading a 403
    here as throttling would sleep through a revoked credential.
    """
    mcp, transport, slept = connected(raw(403, {"error": "forbidden"}, **{"Retry-After": "30"}))
    with pytest.raises(McpAuthError):
        mcp.call_tool("create_comment")
    assert len(transport.sent) == 1
    assert slept == []


@pytest.mark.parametrize("status", [400, 405, 409, 410, 422])
def test_other_4xx_fail_immediately(status: int):
    """Retrying a verdict on the request burns time and delays the error.

    404 is absent on purpose and has its own tests below: against a live
    session it means the session expired, which is a different thing.
    """
    mcp, transport, slept = connected(raw(status, {"error": {"message": "no"}}))
    with pytest.raises(McpHTTPError) as caught:
        mcp.call_tool("create_comment")

    assert caught.value.status == status
    assert len(transport.sent) == 1
    assert slept == []


def test_the_servers_own_message_survives_into_the_exception():
    mcp, _, _ = connected(raw(422, {"error": {"message": "issueId is required"}}))
    with pytest.raises(McpHTTPError, match="issueId is required"):
        mcp.call_tool("create_comment")


def test_a_5xx_is_retried_with_backoff_and_then_succeeds():
    mcp, transport, slept = connected(
        raw(503, {"error": "upstream"}), rpc({"result": {"content": []}})
    )
    mcp.call_tool("create_comment")

    assert len(transport.sent) == 2
    assert slept == [0.5]


def test_a_5xx_that_never_clears_raises_after_the_last_attempt():
    mcp, transport, slept = connected(*[raw(500) for _ in range(3)])
    with pytest.raises(McpHTTPError) as caught:
        mcp.call_tool("create_comment")

    assert caught.value.status == 500
    assert len(transport.sent) == 3
    assert slept == [0.5, 1.0]


def test_a_429_is_retried_for_as_long_as_the_server_asked():
    mcp, transport, slept = connected(
        raw(429, {"error": "slow down"}, **{"Retry-After": "7"}),
        rpc({"result": {"content": []}}),
    )
    mcp.call_tool("create_comment")

    assert slept == [7.0]
    assert len(transport.sent) == 2


def test_a_429_asking_for_longer_than_we_will_wait_raises_immediately():
    """An orchestrator asleep for an hour is indistinguishable from a hung one."""
    mcp, transport, slept = connected(raw(429, None, **{"Retry-After": "3600"}))
    with pytest.raises(McpRateLimitError) as caught:
        mcp.call_tool("create_comment")

    assert caught.value.retry_after_s == 3600.0
    assert slept == []
    assert len(transport.sent) == 1


def test_a_429_that_never_clears_raises_a_rate_limit_error_not_a_plain_one():
    mcp, _, _ = connected(*[raw(429) for _ in range(3)])
    with pytest.raises(McpRateLimitError):
        mcp.call_tool("create_comment")


def test_an_unparseable_retry_after_falls_back_to_backoff():
    mcp, _, slept = connected(
        raw(429, None, **{"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
        rpc({"result": {"content": []}}),
    )
    mcp.call_tool("create_comment")
    assert slept == [0.5]


def test_a_reset_timestamp_is_honoured_when_there_is_no_retry_after(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    mcp, _, slept = connected(
        raw(429, None, **{"X-RateLimit-Reset": "1012"}), rpc({"result": {"content": []}})
    )
    mcp.call_tool("create_comment")
    assert slept == [12.0]


def test_a_transport_failure_is_retried_and_then_named():
    mcp, transport, slept = connected(*[McpTransportError("connection refused") for _ in range(3)])
    with pytest.raises(McpUnreachable) as caught:
        mcp.call_tool("create_comment")

    assert len(transport.sent) == 3
    assert slept == [0.5, 1.0]
    assert caught.value.attempts == 3
    assert "egress-proxy" in str(caught.value)


def test_a_transport_failure_that_clears_is_invisible_to_the_caller():
    mcp, _, slept = connected(
        McpTransportError("dns"), rpc({"result": {"content": [{"type": "text", "text": "ok"}]}})
    )
    assert mcp.call_tool("create_comment").text == "ok"
    assert slept == [0.5]


def test_backoff_is_capped_and_jitter_stays_inside_its_fraction():
    mcp, _, slept = connected(
        *[raw(500) for _ in range(6)],
        retry=RetryPolicy(max_attempts=6, backoff_base_s=10.0, backoff_cap_s=20.0, jitter=0.1),
    )
    with pytest.raises(McpHTTPError):
        mcp.call_tool("create_comment")

    assert len(slept) == 5
    assert all(delay <= 22.0 for delay in slept)
    assert slept[-1] >= 20.0


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------


def test_a_404_against_a_live_session_re_initializes_once():
    """Servers may drop sessions, and a reconcile loop outlives one.

    Not a retried 4xx: what goes out the second time carries a new session, so
    it is a different request. It happens once, and a second 404 raises.
    """
    mcp, transport, _ = connected(
        raw(404, {"error": "session not found"}),
        INITIALIZE,
        NOTIFIED,
        rpc({"result": {"content": [{"type": "text", "text": "ok"}]}}),
    )
    assert mcp.call_tool("create_comment").text == "ok"
    assert [request.rpc_method for request in transport.sent] == [
        "tools/call",
        "initialize",
        "notifications/initialized",
        "tools/call",
    ]


def test_a_second_404_after_re_initializing_gives_up():
    mcp, _, _ = connected(
        raw(404), INITIALIZE, NOTIFIED, raw(404, {"error": "still gone"})
    )
    with pytest.raises(McpHTTPError) as caught:
        mcp.call_tool("create_comment")
    assert caught.value.status == 404


def test_a_404_during_the_handshake_is_not_a_stale_session():
    """There is nothing to re-establish, so it must surface as itself."""
    mcp, transport, _ = client(raw(404, {"error": "no such endpoint"}))
    with pytest.raises(McpHTTPError):
        mcp.connect()
    assert len(transport.sent) == 1


def test_close_ends_the_session_and_forgets_it():
    mcp, transport, _ = connected(Response(200, {}, b""))
    mcp.close()

    assert mcp.session_id is None
    assert mcp.server is None
    assert transport.sent[0].method == "DELETE"
    assert transport.sent[0].headers[SESSION_HEADER] == "sess-1"


def test_close_survives_a_server_that_refuses_to_be_closed():
    """A teardown failure must never be the exception that escapes a cycle."""
    mcp, _, _ = connected(McpTransportError("gone"))
    mcp.close()
    assert mcp.session_id is None


def test_the_context_manager_connects_and_closes():
    transport = FakeTransport([*HANDSHAKE, Response(200, {}, b"")])
    with McpClient(ENDPOINT, "t", transport=transport, sleep=lambda _: None) as mcp:
        assert mcp.server is not None
    assert transport.sent[-1].method == "DELETE"


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def sse(*messages: Any, status: int = 200) -> Any:
    """An event stream whose `"id": ECHO` placeholders become the request's id.

    Echoed rather than hardcoded because the id counter is client state, and a
    test that pinned it would break the first time the handshake changed shape
    - which says nothing about the framing these tests are actually about.
    """

    def build(request: SentRequest) -> Response:
        request_id = (request.json() or {}).get("id")
        lines = []
        for message in messages:
            resolved = {k: (request_id if v == ECHO else v) for k, v in message.items()}
            lines.append(f"event: message\ndata: {json.dumps(resolved)}\n\n")
        return Response(status, {"Content-Type": "text/event-stream"}, "".join(lines).encode())

    return build


#: Placeholder for "whatever id the request carried".
ECHO = "<echo>"


def test_an_event_stream_answer_is_read_like_a_json_one():
    """Servers choose the content type; a client that handled one works against half."""
    handshake = [
        sse({"jsonrpc": "2.0", "id": ECHO, "result": {"serverInfo": {"name": "linear"}}}),
        NOTIFIED,
    ]
    mcp, _, _ = client(*handshake, sse({"jsonrpc": "2.0", "id": ECHO, "result": {"tools": []}}))
    mcp.connect()
    assert mcp.list_tools() == []


def test_the_answer_is_matched_by_id_not_by_position():
    """Progress notifications share the channel with the answer."""
    mcp, _, _ = connected(
        sse(
            {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}},
            {"jsonrpc": "2.0", "id": 999, "result": {"content": [{"type": "text", "text": "no"}]}},
            {"jsonrpc": "2.0", "id": ECHO, "result": {"content": [{"type": "text", "text": "yes"}]}},
        )
    )
    assert mcp.call_tool("create_comment").text == "yes"


def test_an_event_stream_that_never_answers_raises():
    mcp, _, _ = connected(sse({"jsonrpc": "2.0", "method": "notifications/message"}))
    with pytest.raises(McpProtocolError, match="no event-stream message"):
        mcp.call_tool("create_comment")


def stream(request: SentRequest, body: bytes) -> Response:
    """A hand-written stream body with `%ID%` replaced by the request's id."""
    request_id = str((request.json() or {}).get("id")).encode()
    return Response(200, {"Content-Type": "text/event-stream"}, body.replace(b"%ID%", request_id))


def test_keep_alives_and_comments_on_the_stream_are_not_errors():
    body = (
        b": keep-alive\n\n"
        b'event: message\ndata: {"jsonrpc": "2.0", "id": %ID%, "result": {"content": []}}\n\n'
    )
    mcp, _, _ = connected(lambda request: stream(request, body))
    assert mcp.call_tool("create_comment").content == []


def test_a_multiline_data_field_is_joined():
    body = b'data: {"jsonrpc": "2.0", "id": %ID%,\ndata:  "result": {"content": []}}\n\n'
    mcp, _, _ = connected(lambda request: stream(request, body))
    assert mcp.call_tool("create_comment").content == []


def test_an_answer_carrying_the_wrong_id_is_a_protocol_error():
    """Reading somebody else's answer as ours is the failure worth catching."""
    mcp, _, _ = connected(
        Response(
            200,
            {"Content-Type": "application/json"},
            json.dumps({"jsonrpc": "2.0", "id": 99, "result": {"content": []}}).encode(),
        )
    )
    with pytest.raises(McpProtocolError, match="expected"):
        mcp.call_tool("create_comment")


def test_an_answer_that_is_not_json_is_a_protocol_error():
    mcp, _, _ = connected(Response(200, {"Content-Type": "application/json"}, b"<html>hi</html>"))
    with pytest.raises(McpProtocolError, match="not JSON"):
        mcp.call_tool("create_comment")


def test_an_unexpected_content_type_is_a_protocol_error():
    mcp, _, _ = connected(Response(200, {"Content-Type": "text/html"}, b"<html>"))
    with pytest.raises(McpProtocolError, match="text/html"):
        mcp.call_tool("create_comment")


def test_a_batched_answer_is_searched_for_our_id():
    def batched(request: SentRequest) -> Response:
        payload = [
            {"jsonrpc": "2.0", "id": 42, "result": {"content": []}},
            {
                "jsonrpc": "2.0",
                "id": (request.json() or {}).get("id"),
                "result": {"content": [{"type": "text", "text": "ours"}]},
            },
        ]
        return Response(200, {"Content-Type": "application/json"}, json.dumps(payload).encode())

    mcp, _, _ = connected(batched)
    assert mcp.call_tool("create_comment").text == "ours"


def test_text_joins_only_the_text_blocks():
    mcp, _, _ = connected(
        rpc(
            {
                "result": {
                    "content": [
                        {"type": "text", "text": "one"},
                        {"type": "image", "data": "..."},
                        {"type": "text", "text": "two"},
                    ]
                }
            }
        )
    )
    assert mcp.call_tool("create_comment").text == "one\ntwo"


# --------------------------------------------------------------------------
# Construction, configuration, egress
# --------------------------------------------------------------------------


def test_an_endpoint_off_the_allowlist_is_refused_before_anything_is_sent():
    """tinyproxy's `403 Filtered` reads like the tracker refusing the credential.

    Which sends whoever is debugging it to rotate a token that was never the
    problem, so the refusal happens here instead and names the tuple to edit.
    """
    with pytest.raises(McpEgressBlocked) as caught:
        McpClient("https://mcp.notatracker.example/mcp", "t")

    message = str(caught.value)
    assert "EGRESS_ALLOWLIST" in message
    assert "src/swarm/security.py" in message
    assert "compose.yaml" in message
    assert caught.value.host == "mcp.notatracker.example"


def test_the_egress_check_ignores_the_inert_widening_variable(monkeypatch):
    """`APIARY_EGRESS_ALLOW` is read by nothing that filters anything.

    Honouring it here would let the check pass for a host tinyproxy then
    blocks, which is worse than not checking at all.
    """
    monkeypatch.setenv("APIARY_EGRESS_ALLOW", "mcp.notatracker.example")
    with pytest.raises(McpEgressBlocked):
        McpClient("https://mcp.notatracker.example/mcp", "t")


def test_the_allowlist_admits_the_linear_endpoint():
    """The positive control: the check can be passed, not only failed."""
    assert_endpoint_allowed(ENDPOINT)
    assert_endpoint_allowed("https://mcp.linear.app:443/mcp")


def test_an_operator_may_pass_a_widened_policy_explicitly():
    policy = EgressPolicy(hosts=("mcp.internal.example",))
    McpClient("https://mcp.internal.example/mcp", "t", egress=policy)


@pytest.mark.parametrize("endpoint", ["ftp://x/y", "mcp.linear.app", "", "ws://x"])
def test_an_endpoint_that_is_not_a_url_is_refused(endpoint: str):
    with pytest.raises(ValueError):
        McpClient(endpoint, "t")


def test_from_env_reads_the_endpoint_and_the_credential():
    mcp = McpClient.from_env(
        env={TRACKER_ENDPOINT_ENV: ENDPOINT, TRACKER_TOKEN_ENV: "k"},
        transport=FakeTransport([]),
    )
    assert mcp.endpoint == ENDPOINT


def test_from_env_names_the_variable_that_is_missing():
    with pytest.raises(McpError, match=TRACKER_ENDPOINT_ENV):
        McpClient.from_env(env={})
    with pytest.raises(McpError, match=TRACKER_TOKEN_ENV):
        McpClient.from_env(env={TRACKER_ENDPOINT_ENV: ENDPOINT})


def test_from_env_takes_the_variable_names_as_arguments():
    """#150 names them per tracker; two trackers do not share a credential."""
    mcp = McpClient.from_env(
        token_env="APIARY_LINEAR_TOKEN",
        endpoint_env="APIARY_LINEAR_URL",
        env={"APIARY_LINEAR_URL": ENDPOINT, "APIARY_LINEAR_TOKEN": "k"},
        transport=FakeTransport([]),
    )
    assert mcp.token_env == "APIARY_LINEAR_TOKEN"


def test_the_credential_is_named_in_the_auth_error_it_was_configured_under():
    transport = FakeTransport([*HANDSHAKE, raw(401)])
    mcp = McpClient(
        ENDPOINT, "k", transport=transport, token_env="APIARY_LINEAR_TOKEN", sleep=lambda _: None
    )
    with pytest.raises(McpAuthError, match="APIARY_LINEAR_TOKEN"):
        mcp.call_tool("create_comment")


def test_a_client_with_no_credential_sends_no_authorization_header():
    """A local stdio server takes its credential from its environment instead."""
    transport = FakeTransport([*HANDSHAKE])
    McpClient(ENDPOINT, None, transport=transport, sleep=lambda _: None).connect()
    assert "Authorization" not in transport.sent[0].headers


# --------------------------------------------------------------------------
# The local stdio transport - the GitHub path (#143)
# --------------------------------------------------------------------------


class FakePopen:
    """A subprocess double over real in-memory pipes.

    Answers each request written to stdin according to `handler`, so the thing
    under test is the framing - which line the reader takes and what it does
    when the process stops answering.
    """

    def __init__(self, handler, *, exit_after: int | None = None) -> None:
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO()
        self._handler = handler
        self._exit_after = exit_after
        self._served = 0
        self.terminated = False
        self.returncode: int | None = None

        # Overridden so a write is answered immediately, in the same buffer the
        # reader is about to read from.
        outer = self

        class Stdin(io.BytesIO):
            def write(self, data):  # type: ignore[override]
                outer._serve(data)
                return len(data)

            def flush(self):  # type: ignore[override]
                return None

        self.stdin = Stdin()

    def _serve(self, data: bytes) -> None:
        if self._exit_after is not None and self._served >= self._exit_after:
            self.returncode = 1
            return
        self._served += 1
        for line in self._handler(data):
            position = self.stdout.tell()
            self.stdout.seek(0, io.SEEK_END)
            self.stdout.write(line + b"\n")
            self.stdout.seek(position)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):  # pragma: no cover - only for a server ignoring SIGTERM
        self.returncode = -9

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


def echo_handler(data: bytes) -> list[bytes]:
    """Answer a request with a result naming its method; ignore notifications."""
    message = json.loads(data.decode("utf-8"))
    if "id" not in message:
        return []
    method = message["method"]
    if method == "initialize":
        result: Any = {"serverInfo": {"name": "github-mcp-server", "version": "0.9"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "issue_write"}]}
    else:
        result = {"content": [{"type": "text", "text": method}]}
    return [json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}).encode()]


@pytest.fixture()
def stdio_client(monkeypatch):
    """A client over a fake local server, plus the `select` that pipes need."""
    import swarm.mcp.client as module

    monkeypatch.setattr(module.select, "select", lambda r, w, x, t: (r, w, x))

    def build(handler=echo_handler, **kwargs):
        process = FakePopen(handler, **kwargs)
        transport = StdioTransport(["github-mcp-server", "stdio"], spawn=lambda *a, **k: process)
        return McpClient(
            f"{STDIO_SCHEME}github-mcp-server",
            None,
            transport=transport,
            sleep=lambda _: None,
            retry=RetryPolicy(max_attempts=2, backoff_base_s=0.0, jitter=0.0),
        ), process

    return build


def test_a_local_server_completes_the_same_handshake(stdio_client):
    mcp, _ = stdio_client()
    info = mcp.connect()
    assert info.name == "github-mcp-server"


def test_a_local_server_lists_and_calls_tools(stdio_client):
    mcp, _ = stdio_client()
    assert [tool.name for tool in mcp.list_tools()] == ["issue_write"]
    assert mcp.call_tool("issue_write", {"method": "create"}).text == "tools/call"


def test_a_stdio_endpoint_needs_a_transport():
    with pytest.raises(ValueError, match="over_stdio"):
        McpClient(f"{STDIO_SCHEME}github-mcp-server", None)


def test_a_stdio_client_asks_no_egress_question():
    """A subprocess has no host, and the allowlist applies to it as to anything."""
    transport = StdioTransport(["x"], spawn=lambda *a, **k: FakePopen(echo_handler))
    McpClient(f"{STDIO_SCHEME}x", None, transport=transport)


def test_the_reader_skips_messages_that_are_not_the_answer(stdio_client):
    """A chatty server would otherwise have its log read as the result."""

    def chatty(data: bytes) -> list[bytes]:
        message = json.loads(data.decode("utf-8"))
        if "id" not in message:
            return []
        noise = json.dumps({"jsonrpc": "2.0", "method": "notifications/message"}).encode()
        other = json.dumps({"jsonrpc": "2.0", "id": 9999, "result": {"content": []}}).encode()
        return [noise, other, *echo_handler(data)]

    mcp, _ = stdio_client(chatty)
    assert mcp.call_tool("issue_write").text == "tools/call"


def test_a_server_that_exits_mid_run_is_reported_as_unreachable(stdio_client):
    mcp, _ = stdio_client(exit_after=2)
    mcp.connect()  # initialize + the notification
    with pytest.raises(McpUnreachable) as caught:
        mcp.call_tool("issue_write")
    assert "github-mcp-server" in str(caught.value)
    assert "PATH" in str(caught.value)


def test_close_terminates_the_local_server(stdio_client):
    mcp, process = stdio_client()
    mcp.connect()
    mcp.close()
    assert process.terminated


def test_a_binary_that_is_not_there_says_so():
    def missing(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory")

    transport = StdioTransport(["github-mcp-server"], spawn=missing)
    with pytest.raises(McpTransportError, match="PATH"):
        transport.start()


def test_a_stdio_transport_needs_a_command():
    with pytest.raises(ValueError):
        StdioTransport([])


def test_over_stdio_passes_only_the_named_environment():
    """A credential added to the orchestrator later is not one a binary holds."""
    mcp = McpClient.over_stdio(
        ["github-mcp-server", "stdio"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": "github_pat_x"},
        source_env={"PATH": "/usr/bin", "GITHUB_TOKEN": "github_pat_x", "SECRET": "s"},
    )
    transport = mcp._transport
    assert isinstance(transport, StdioTransport)
    assert transport.env == {"PATH": "/usr/bin", "GITHUB_PERSONAL_ACCESS_TOKEN": "github_pat_x"}
    assert mcp.endpoint == f"{STDIO_SCHEME}github-mcp-server"


def test_over_stdio_refuses_to_hand_a_local_server_the_boot_key(monkeypatch):
    """A binary apiary did not write is not a safer home for `administration`.

    The same refusal `ContainerManager` runs before starting a worker, at the
    other place the orchestrator hands a credential to code it does not own.
    """
    from swarm.security import PROVISION_TOKEN_ENV, PolicyError

    monkeypatch.setenv(PROVISION_TOKEN_ENV, "github_pat_boot")
    with pytest.raises(PolicyError):
        McpClient.over_stdio(["x"], env={PROVISION_TOKEN_ENV: "github_pat_boot"}, source_env={})
    with pytest.raises(PolicyError):
        McpClient.over_stdio(["x"], env={"RENAMED": "github_pat_boot"}, source_env={})


# --------------------------------------------------------------------------
# The default transport
# --------------------------------------------------------------------------


def test_the_urllib_transport_returns_4xx_as_a_response_rather_than_raising():
    """The retry loop needs the status and the headers to decide anything."""
    import urllib.error

    class Headers(dict):
        def items(self):
            return super().items()

    error = urllib.error.HTTPError(
        ENDPOINT, 429, "Too Many Requests", Headers({"Retry-After": "5"}), io.BytesIO(b"slow down")
    )

    def raising(*args, **kwargs):
        raise error

    import urllib.request

    original = urllib.request.urlopen
    urllib.request.urlopen = raising
    try:
        answer = UrllibTransport().send("POST", ENDPOINT, {}, b"{}", 1.0)
    finally:
        urllib.request.urlopen = original

    assert answer.status == 429
    assert answer.header("retry-after") == "5"


def test_the_urllib_transport_turns_a_connection_failure_into_a_transport_error():
    import urllib.error
    import urllib.request

    def raising(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    original = urllib.request.urlopen
    urllib.request.urlopen = raising
    try:
        with pytest.raises(McpTransportError, match="connection refused"):
            UrllibTransport().send("POST", ENDPOINT, {}, b"{}", 1.0)
    finally:
        urllib.request.urlopen = original


def test_the_documented_variable_names_are_the_ones_the_code_reads():
    """compose.yaml and the module docstring both spell these out.

    A rename that missed one of the three would present as a credential the
    operator exported and the orchestrator never saw.
    """
    assert TRACKER_TOKEN_ENV == "APIARY_TRACKER_TOKEN"
    assert TRACKER_ENDPOINT_ENV == "APIARY_TRACKER_MCP_URL"

    compose = (Path(__file__).resolve().parents[1] / "compose.yaml").read_text()
    for name in (TRACKER_TOKEN_ENV, TRACKER_ENDPOINT_ENV):
        assert f"      {name}: ${{{name}:-}}\n" in compose, name
