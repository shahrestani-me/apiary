"""A fake GitHub transport: canned responses, recorded requests, injected faults.

`swarm.github.client.Transport` exists so that no test needs a network or a
token, and so that the parts most likely to be wrong - retry, backoff, rate
limiting, conditional requests - are testable at all. You cannot ask GitHub for
a 503 on demand.

The double is deliberately dumb. It replays what it was scripted with, records
what it was asked, and asserts when it is asked something nobody scripted; an
unexpected extra request is exactly the class of bug these tests exist to
catch. Anything smarter - a store that serves back what it was told to create -
belongs in the test that needs it, layered on top through `handler`.

Failures are constructors rather than literals (`not_found()`,
`secondary_rate_limit(retry_after=2)`) because GitHub's error shapes are
load-bearing: the client tells a throttled 403 from a permission 403 by reading
the message and the headers, and a fake that got that shape subtly wrong would
prove the wrong thing.

    from fixtures.github import not_found, response

    def test_it(fake_github):
        gh, transport, slept = fake_github(response(200, {"number": 7}))
        assert gh.get_issue(7)["number"] == 7
        assert transport.sent[0].path == "/repos/shahrestani-me/apiary/issues/7"
"""

from __future__ import annotations

import json
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from swarm.github.client import GitHubClient, Response, RetryPolicy, TransportError

API = "https://api.github.com"
REPO = "shahrestani-me/apiary"
TOKEN = "test-token"

#: Deterministic by construction: no jitter, so `slept` is comparable to an
#: exact list, and a short base so a test that does sleep for real costs
#: nothing. Tests that care about the policy pass their own.
RETRY = RetryPolicy(max_attempts=3, backoff_base_s=0.5, jitter=0.0)


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------


@dataclass
class SentRequest:
    """One request the client made, as the transport saw it."""

    method: str
    url: str
    headers: Mapping[str, str]
    body: bytes | None

    def json(self) -> Any:
        """The request body, decoded. `None` when there was no body."""
        return json.loads(self.body.decode("utf-8")) if self.body else None

    @property
    def path(self) -> str:
        return urllib.parse.urlsplit(self.url).path

    @property
    def query(self) -> dict[str, str]:
        return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.url).query))

    def header(self, name: str) -> str | None:
        """Case-insensitive header lookup, matching `Response.header`."""
        target = name.lower()
        return next((v for k, v in self.headers.items() if k.lower() == target), None)


# --------------------------------------------------------------------------
# The transport
# --------------------------------------------------------------------------


@dataclass
class FakeTransport:
    """Replays scripted responses in order; records every request.

    A scripted entry may be:

    - a `Response` - returned as-is;
    - an `Exception` - raised, which is how transport-level failures (no HTTP
      response at all) are simulated;
    - a callable taking the `SentRequest` and returning a `Response`, for the
      rare case where the answer depends on what was asked.

    When the script runs out, `handler` - if one was given - answers instead.
    That is the seam for a stateful fake: keep the recording and the fault
    injection here, put the store in the handler. With no handler and no
    script left, an unscripted request is an assertion failure rather than a
    default, because silently answering one hides a bug.
    """

    script: list[Any] = field(default_factory=list)
    handler: Callable[[SentRequest], Response] | None = None
    sent: list[SentRequest] = field(default_factory=list)

    def send(self, method, url, headers, body, timeout) -> Response:
        request = SentRequest(method, url, dict(headers), body)
        self.sent.append(request)

        if self.script:
            nxt = self.script.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt(request) if callable(nxt) else nxt

        if self.handler is not None:
            return self.handler(request)

        raise AssertionError(f"unscripted request: {method} {url}")

    def queue(self, *responses: Any) -> FakeTransport:
        """Append to the script mid-test. Returns self, so it chains."""
        self.script.extend(responses)
        return self

    @property
    def paths(self) -> list[str]:
        return [request.path for request in self.sent]

    @property
    def calls(self) -> list[tuple[str, str]]:
        return [(request.method, request.path) for request in self.sent]


def client(
    *script: Any,
    repo: str = REPO,
    token: str = TOKEN,
    handler: Callable[[SentRequest], Response] | None = None,
    **kwargs: Any,
) -> tuple[GitHubClient, FakeTransport, list[float]]:
    """A client wired to a scripted transport and a recording sleep.

    Returns `(client, transport, slept)`. Nothing sleeps for real: `slept` is
    the list of durations the client asked for, which is the only way to assert
    on backoff without making the suite slow.
    """
    transport = FakeTransport(list(script), handler=handler)
    slept: list[float] = []
    kwargs.setdefault("retry", RETRY)
    kwargs.setdefault("api_url", API)
    gh = GitHubClient(repo, token, transport=transport, sleep=slept.append, **kwargs)
    return gh, transport, slept


# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------


def response(status: int, payload: Any = None, **headers: str) -> Response:
    """A JSON response. `payload=None` means an empty body, not `null`."""
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    return Response(status, headers, body)


def page(items: list[Any], next_url: str | None = None, **headers: str) -> Response:
    """One page of a list endpoint, with the `Link` header if there is a next."""
    if next_url:
        headers["Link"] = f'<{next_url}>; rel="next", <{next_url}>; rel="last"'
    return response(200, items, **headers)


def not_modified(**headers: str) -> Response:
    """A 304. Carries no body and no `Link`, exactly like the real thing."""
    return response(304, None, **headers)


# --------------------------------------------------------------------------
# Failures
#
# Every one of these is a response the real API produces and no test can
# provoke on demand. The message strings matter: the client classifies a 403
# by reading them.
# --------------------------------------------------------------------------


def not_found(message: str = "Not Found") -> Response:
    return response(404, {"message": message})


def server_error(status: int = 502, message: str = "Server Error") -> Response:
    """A retryable 5xx."""
    return response(status, {"message": message})


def validation_failed(*errors: Mapping[str, str], message: str = "Validation Failed") -> Response:
    """A 422. Not retryable: the payload will be just as invalid next time."""
    return response(422, {"message": message, "errors": [dict(e) for e in errors]})


def forbidden(message: str = "Resource not accessible by integration") -> Response:
    """A 403 with no throttling signals - a missing scope, not a rate limit.

    The distinction is the point: this one must fail immediately rather than
    after three backoffs.
    """
    return response(403, {"message": message})


def secondary_rate_limit(retry_after: float | None = None) -> Response:
    """The abuse-detection 403. `Retry-After` is GitHub's number, not ours."""
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
    return response(
        403,
        {"message": "You have exceeded a secondary rate limit"},
        **headers,
    )


def primary_rate_limit(reset_in_s: float = 60.0) -> Response:
    """The hourly-quota 403, with the reset `reset_in_s` seconds from now."""
    return response(
        403,
        {"message": "API rate limit exceeded"},
        **{
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(time.time() + reset_in_s)),
        },
    )


def too_many_requests(message: str = "Too many requests") -> Response:
    return response(429, {"message": message})


def connection_reset(message: str = "connection reset") -> TransportError:
    """No HTTP response at all - DNS, TCP, TLS or timeout. Script it to raise."""
    return TransportError(message)
