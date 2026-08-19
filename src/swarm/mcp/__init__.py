"""Reaching a customer's task system, as an integration rather than a subsystem.

ADR 0001: apiary ships no tracker adapter. Linear, Jira and GitHub Issues are
reached through the customer's own MCP server, and this package is the one
piece of apiary that talks to it - a transport, with no opinion about which
tool means what. That opinion is the capability contract (#150), and keeping it
out of here is what lets the contract change shape without the transport
changing at all.
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

__all__ = [
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
    "Response",
    "RetryPolicy",
    "ServerInfo",
    "StdioTransport",
    "ToolResult",
    "ToolSpec",
    "Transport",
    "UrllibTransport",
    "assert_endpoint_allowed",
]
