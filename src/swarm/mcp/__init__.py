"""Reaching a customer's task system, as an integration rather than a subsystem.

ADR 0001: apiary ships no tracker adapter. Linear, Jira and GitHub Issues are
reached through the customer's own MCP server, and this package is the one
piece of apiary that talks to it - a transport, with no opinion about which
tool means what. That opinion is the capability contract (#150), and keeping it
out of here is what lets the contract change shape without the transport
changing at all.

Two modules, and the seam between them is the point:

- `client.py` connects, lists tools, and calls one by name. It has no idea what
  a name means and never will.
- `contract.py` says which tool means what, as per-organization configuration.
  #143 found that ADR 0001's sketched contract could not make a single real
  call; the shape moved, and the transport did not have to.
"""

from __future__ import annotations

from .client import (
    PROTOCOL_VERSION,
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
    McpRpcError,
    McpToolError,
    McpTransportError,
    McpUnreachable,
    Response,
    RetryPolicy,
    ServerInfo,
    StdioTransport,
    ToolResult,
    ToolSpec,
    Transport,
    UrllibTransport,
    assert_endpoint_allowed,
)
from .contract import (
    CANONICAL_FIELDS,
    CAPABILITIES,
    COMMENT,
    CREATE,
    INTAKE,
    PROFILES,
    Auth,
    Capability,
    ContractError,
    TrackerContract,
    client_for,
    load_tracker,
    parse_tracker,
)

__all__ = [
    "CANONICAL_FIELDS",
    "CAPABILITIES",
    "COMMENT",
    "CREATE",
    "INTAKE",
    "PROFILES",
    "PROTOCOL_VERSION",
    "STDIO_SCHEME",
    "TRACKER_ENDPOINT_ENV",
    "TRACKER_TOKEN_ENV",
    "McpAuthError",
    "McpClient",
    "McpEgressBlocked",
    "McpError",
    "McpHTTPError",
    "McpProtocolError",
    "McpRateLimitError",
    "McpRpcError",
    "McpToolError",
    "McpTransportError",
    "McpUnreachable",
    "Auth",
    "Capability",
    "ContractError",
    "Response",
    "RetryPolicy",
    "ServerInfo",
    "StdioTransport",
    "ToolResult",
    "ToolSpec",
    "TrackerContract",
    "Transport",
    "UrllibTransport",
    "assert_endpoint_allowed",
    "client_for",
    "load_tracker",
    "parse_tracker",
]
