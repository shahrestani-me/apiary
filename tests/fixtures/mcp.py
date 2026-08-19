"""An MCP server, faked at the transport rather than at the client.

`test_mcp_client.py` has `FakeTransport`, which replays a script - the right
double for testing the client, because what that file measures is retry
classification and a script is how you provoke a 429 followed by a 200. This is
the other double: a server that *behaves*, answering `initialize` and
`tools/list` from state rather than from a queue.

The difference matters for anything above the transport. `doctor` makes two
requests in an order it decides, `contract.py` decides which tool names it will
ask about, and neither is a fixed sequence a script could stand in for. A double
that answers by method also lets a test say the thing it actually cares about -
"this server has no `create_issue`" - instead of hand-rolling the JSON-RPC
envelope for a tools/list page.

It records every JSON-RPC method it was asked for, which is how
`tests/test_doctor.py` asserts the read-only property against a *tracker*: the
preflight may `initialize` and `tools/list`, and a `tools/call` in that list is
a comment somebody received or a ticket somebody now has to triage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from swarm.mcp.client import PROTOCOL_VERSION, McpClient, Response

#: On `security.EGRESS_ALLOWLIST`, so a test that wanted to measure something
#: else does not measure the egress pre-flight instead.
ENDPOINT = "https://mcp.linear.app/mcp"

#: What the built-in `linear` profile names, which is what makes the default
#: server here one the default contract can actually be checked against.
LINEAR_TOOLS: tuple[str, ...] = ("list_issues", "create_comment", "create_issue")


@dataclass
class FakeMcpServer:
    """A server that answers by method, and remembers what it was asked.

    Each field is one thing that can be wrong with a real one - the tool list,
    the advertised capabilities, the credential, whether it is there at all -
    so a test breaks exactly one and leaves the rest healthy.
    """

    tools: Sequence[str] = LINEAR_TOOLS
    name: str = "linear-mcp"
    version: str = "1.4.0"
    capabilities: Mapping[str, Any] = field(default_factory=lambda: {"tools": {}})
    #: What it puts in `Mcp-Session-Id`. Servers differ on whether they issue
    #: one at all, and it matters here rather than being decoration: a client
    #: with no session has nothing to DELETE, so a teardown is observable only
    #: against a server that issued one - which Linear's does.
    session: str | None = "01J-fake-session"
    #: Answer everything 401, the way an expired credential does.
    unauthorized: bool = False
    #: Raise instead of answering, the way an unreachable host does.
    unreachable: Exception | None = None

    #: Every JSON-RPC method asked of it, in order.
    methods: list[str] = field(default_factory=list)
    #: Every `Authorization`-style header value it was sent.
    authorizations: list[str | None] = field(default_factory=list)

    # --- the transport seam ----------------------------------------------

    def send(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> Response:
        self.authorizations.append(
            next((value for key, value in headers.items() if key.lower() == "authorization"), None)
        )
        if method == "DELETE":
            self.methods.append("session/delete")
            return Response(200, {}, b"")

        message = json.loads((body or b"{}").decode("utf-8"))
        self.methods.append(str(message.get("method")))

        if self.unreachable is not None:
            raise self.unreachable
        if self.unauthorized:
            return Response(
                401,
                {"WWW-Authenticate": 'Bearer error="invalid_token"'},
                b'{"error": "unauthorized"}',
            )
        return self._answer(message)

    def _answer(self, message: Mapping[str, Any]) -> Response:
        method = str(message.get("method"))
        if method == "initialize":
            return self._result(
                message,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": dict(self.capabilities),
                    "serverInfo": {"name": self.name, "version": self.version},
                },
                headers={"Mcp-Session-Id": self.session} if self.session else {},
            )
        if method.startswith("notifications/"):
            # A notification has no id and wants no answer.
            return Response(202, {"Content-Type": "application/json"}, b"")
        if method == "tools/list":
            return self._result(
                message,
                {
                    "tools": [
                        {"name": name, "inputSchema": {"type": "object"}} for name in self.tools
                    ]
                },
            )
        if method == "tools/call":
            # Reached only by a bug: nothing in a preflight may call a tool.
            return self._result(message, {"content": [{"type": "text", "text": "ok"}]})
        return Response(
            200,
            {"Content-Type": "application/json"},
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "error": {"code": -32601, "message": f"no such method: {method}"},
                }
            ).encode("utf-8"),
        )

    @staticmethod
    def _result(
        message: Mapping[str, Any],
        result: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        payload = {"jsonrpc": "2.0", "id": message.get("id"), "result": dict(result)}
        return Response(
            200,
            {"Content-Type": "application/json", **(headers or {})},
            json.dumps(payload).encode("utf-8"),
        )

    # --- what a test asks it ---------------------------------------------

    @property
    def called_tools(self) -> list[str]:
        """Every `tools/call`. Required to be empty of any read-only caller."""
        return [method for method in self.methods if method == "tools/call"]

    @property
    def closed(self) -> bool:
        return "session/delete" in self.methods


def client(server: FakeMcpServer, *, endpoint: str = ENDPOINT, **kwargs: Any) -> McpClient:
    """A real `McpClient` speaking to `server`.

    Real on purpose: a double standing in for the client as well as for the
    server would prove that two doubles agree.
    """
    return McpClient(endpoint, kwargs.pop("token", None), transport=server, **kwargs)
