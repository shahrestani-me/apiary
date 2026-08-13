"""GitHub access for the v2 control plane.

One module, one door: `client.GitHubClient`. Nothing else in the swarm talks
to GitHub, and nothing at all shells out to `gh`.
"""

from __future__ import annotations

from .client import (
    GitHubClient,
    GitHubError,
    GitHubHTTPError,
    RateLimitError,
    Response,
    RetryPolicy,
    Transport,
    TransportError,
    UrllibTransport,
)

__all__ = [
    "GitHubClient",
    "GitHubError",
    "GitHubHTTPError",
    "RateLimitError",
    "Response",
    "RetryPolicy",
    "Transport",
    "TransportError",
    "UrllibTransport",
]
