"""The single door to the GitHub API.

GitHub is the database (`docs/architecture-v2.md`), so every read and write of
the control plane funnels through this one module. Two rules follow from that,
and both are load-bearing:

**No `gh` subprocess calls anywhere in library code.** The orchestrator runs in
a container where `gh`'s auth flow is awkward, and shelling out hides API
errors behind an exit code - the reconcile loop cannot tell "issue does not
exist" from "token lacks the scope" from "GitHub is having an outage", yet it
must react differently to each.

**Retry only what retrying can fix.** 5xx and rate limiting are transient, so
they get backoff. A 4xx is a statement about the request itself: retrying a 404
or a 422 burns rate-limit budget that the workers share and delays the error by
however long the backoff schedule lasts. Those surface immediately.

The surface here is deliberately small - the endpoints v2 actually calls, and
no more. Wrapping GitHub is not the goal.

Manual smoke test against a real repo:

    GITHUB_TOKEN=... python -m swarm.github.client shahrestani-me/apiary
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Protocol, Sequence

DEFAULT_API_URL = "https://api.github.com"
API_VERSION = "2022-11-28"
USER_AGENT = "apiary-swarm"

# A safety stop for the pagination loop. GitHub's `Link` headers terminate on
# their own; this only fires if the API ever hands us a cycle, and looping
# forever inside a reconcile cycle is a worse failure than raising.
MAX_PAGES = 100


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class GitHubError(RuntimeError):
    """Base for everything this module raises."""


class TransportError(GitHubError):
    """The request never produced an HTTP response - DNS, TCP, TLS, timeout.

    Retryable by definition: there is no response to interpret, so we cannot
    have offended the server.
    """


class GitHubHTTPError(GitHubError):
    """GitHub answered with a status we will not act on."""

    def __init__(self, status: int, method: str, url: str, body: bytes) -> None:
        self.status = status
        self.method = method
        self.url = url
        self.body = body
        super().__init__(f"{method} {url} -> {status}: {_error_message(body)}")


class RateLimitError(GitHubHTTPError):
    """Rate limited, and waiting it out is not this client's decision.

    Raised when the reset is further away than `RetryPolicy.max_wait_s`, or
    when retries ran out while still being throttled. `retry_after_s` is what
    GitHub asked for, so the caller can decide to sleep, shed load, or stop.
    """

    def __init__(self, status: int, method: str, url: str, body: bytes,
                 retry_after_s: float | None) -> None:
        self.retry_after_s = retry_after_s
        super().__init__(status, method, url, body)


def _error_message(body: bytes) -> str:
    """Pull GitHub's `message` out of an error body, falling back to the raw text."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return body.decode("utf-8", "replace")[:200]
    if isinstance(payload, dict):
        message = payload.get("message") or ""
        errors = payload.get("errors")
        if errors:
            message = f"{message} {json.dumps(errors)}"
        return message.strip() or json.dumps(payload)[:200]
    return json.dumps(payload)[:200]


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Response:
    """One HTTP response, transport-agnostic."""

    status: int
    headers: Mapping[str, str]
    body: bytes

    def header(self, name: str, default: str | None = None) -> str | None:
        # Case-insensitive on purpose: a fake transport in a test hands us a
        # plain dict, where `urllib` hands us a case-insensitive HTTPMessage.
        # Normalising here means test doubles need no special knowledge.
        target = name.lower()
        for key, value in self.headers.items():
            if key.lower() == target:
                return value
        return default

    def json(self) -> Any:
        if not self.body:
            return None
        return json.loads(self.body.decode("utf-8"))


class Transport(Protocol):
    """The seam that keeps this module testable without a network or a token.

    An implementation returns 4xx and 5xx as ordinary `Response` objects - the
    retry policy needs the status and the headers to decide - and raises
    `TransportError` only when no response arrived at all.
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
    single retry loop in `GitHubClient` sees every outcome the same way.
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
            # A 304 arrives here too, since urllib treats anything >= 300 as an
            # error. It is a normal conditional-request outcome, not a failure.
            headers_out = dict(exc.headers.items()) if exc.headers else {}
            return Response(exc.code, headers_out, exc.read())
        except urllib.error.URLError as exc:
            raise TransportError(f"{method} {url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise TransportError(f"{method} {url}: timed out after {timeout}s") from exc


# --------------------------------------------------------------------------
# Retry policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """How hard to try, and how long to be willing to wait.

    `max_wait_s` is the one that matters: a primary rate-limit reset can be up
    to an hour away, and an orchestrator asleep for an hour is indistinguishable
    from a hung one. Past that threshold we raise and let the caller decide.
    """

    max_attempts: int = 4
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 30.0
    jitter: float = 0.1          # fraction of the delay, to desynchronise workers
    max_wait_s: float = 60.0     # refuse to sleep out a distant rate-limit reset


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class GitHubClient:
    """A thin, retrying, conditional-request-aware REST client for one repo.

    Every method returns GitHub's JSON as plain dicts and lists. Mapping those
    onto `TaskRecord` and friends is the loader's job (#9) - this module knows
    about HTTP, not about the issue contract.
    """

    def __init__(
        self,
        repo: str,
        token: str,
        *,
        api_url: str | None = None,
        transport: Transport | None = None,
        retry: RetryPolicy | None = None,
        etag_cache: MutableMapping[str, tuple[str, Any, str | None]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        timeout_s: float = 30.0,
    ) -> None:
        if repo.count("/") != 1 or not all(repo.split("/")):
            raise ValueError(f"repo must be 'owner/name', got {repo!r}")
        self.repo = repo
        self.api_url = (api_url or os.environ.get("GITHUB_API_URL") or DEFAULT_API_URL).rstrip("/")
        self.retry = retry or RetryPolicy()
        # Injectable so the reconcile loop (#22) can share one cache across
        # cycles - a cache that dies with the client saves nothing.
        self.etag_cache = {} if etag_cache is None else etag_cache
        self._transport = transport or UrllibTransport()
        self._sleep = sleep
        self._timeout_s = timeout_s
        self._headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": USER_AGENT,
        }

    @classmethod
    def from_env(cls, repo: str | None = None, **kwargs: Any) -> GitHubClient:
        """Build a client from `GITHUB_TOKEN` and `GITHUB_REPOSITORY`."""
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise GitHubError("GITHUB_TOKEN is not set")
        repo = repo or os.environ.get("GITHUB_REPOSITORY")
        if not repo:
            raise GitHubError("no repo given and GITHUB_REPOSITORY is not set")
        return cls(repo, token, **kwargs)

    # --- issues -----------------------------------------------------------

    def get_issue(self, number: int) -> dict[str, Any]:
        return self._get(self._url(f"/repos/{self.repo}/issues/{number}"))

    def list_issues(
        self,
        *,
        state: str = "open",
        labels: Sequence[str] | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        """List issues, pull requests excluded.

        `GET /issues` returns PRs as well - GitHub models a PR as an issue with
        extra fields. The ledger is issues only, and a dispatcher that treats a
        PR as a task tries to run a container against a body that has no
        contract in it. Filtering on the `pull_request` key is the documented
        way to tell them apart.
        """
        params: dict[str, str] = {"state": state}
        if labels:
            params["labels"] = ",".join(labels)
        if since:
            params["since"] = since
        items = self._paginate(f"/repos/{self.repo}/issues", params)
        return [item for item in items if "pull_request" not in item]

    def create_issue(
        self,
        title: str,
        *,
        body: str | None = None,
        labels: Sequence[str] | None = None,
        milestone: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"title": title}
        if body is not None:
            payload["body"] = body
        if labels:
            payload["labels"] = list(labels)
        if milestone is not None:
            payload["milestone"] = milestone
        return self._send_json("POST", self._url(f"/repos/{self.repo}/issues"), payload)

    def update_issue(
        self,
        number: int,
        *,
        title: str | None = None,
        body: str | None = None,
        state: str | None = None,
        state_reason: str | None = None,
        milestone: int | None = None,
    ) -> dict[str, Any]:
        """Edit an issue. Note the absence of a `labels` argument.

        `PATCH /issues/{n}` *replaces* the label set, so a caller that patches a
        body and passes labels it happens to know about silently drops the ones
        it does not - a human's `area/*` label, or a state label written by
        another component between the read and the write. Label changes go
        through `add_labels` / `remove_label`, which are additive and scoped.
        """
        payload: dict[str, Any] = {}
        for key, value in (
            ("title", title),
            ("body", body),
            ("state", state),
            ("state_reason", state_reason),
            ("milestone", milestone),
        ):
            if value is not None:
                payload[key] = value
        if not payload:
            raise ValueError("update_issue called with nothing to update")
        return self._send_json("PATCH", self._url(f"/repos/{self.repo}/issues/{number}"), payload)

    # --- labels -----------------------------------------------------------

    def add_labels(self, number: int, labels: Iterable[str]) -> list[dict[str, Any]]:
        """Add labels without disturbing the ones already there."""
        names = list(labels)
        if not names:
            raise ValueError("add_labels called with no labels")
        return self._send_json(
            "POST",
            self._url(f"/repos/{self.repo}/issues/{number}/labels"),
            {"labels": names},
        )

    def remove_label(self, number: int, label: str) -> bool:
        """Remove one label. Returns False if it was not there.

        The 404 is tolerated because removing a label the issue does not carry
        is the desired end state anyway, and the reconciler races with humans
        by design (`docs/issue-contract.md` §3) - a human who already removed
        the label should not fail the cycle.
        """
        path = f"/repos/{self.repo}/issues/{number}/labels/{urllib.parse.quote(label, safe='')}"
        try:
            self._send_json("DELETE", self._url(path), None)
        except GitHubHTTPError as exc:
            if exc.status == 404:
                return False
            raise
        return True

    def create_issue_comment(self, number: int, body: str) -> dict[str, Any]:
        """Comment on an issue.

        `docs/issue-contract.md` §1.4 requires a malformed contract to be
        reported as a comment on the offending issue, and the same channel
        carries the reason when an issue reaches `swarm:failed`. Without it
        `reconcile.post_comment` printed to stderr and recorded the issue on
        `ReconcileReport.uncommented`, which means the explanation lived only
        in the orchestrator's own logs - exactly where the human looking at
        the tracker will not find it.
        """
        path = f"/repos/{self.repo}/issues/{number}/comments"
        return self._send_json("POST", self._url(path), {"body": body})

    # --- pull requests ----------------------------------------------------

    def delete_branch(self, branch: str) -> bool:
        """Delete a branch, returning False if it was already gone.

        Called after a merge so a long run does not leave one dead
        `swarm/issue-<n>` branch per task. Already-deleted is the ordinary
        case when GitHub's own "delete branch on merge" setting is enabled, so
        a 404 (and the 422 the refs endpoint returns for an absent ref) is a
        no-op rather than a failure.
        """
        path = f"/repos/{self.repo}/git/refs/heads/{urllib.parse.quote(branch, safe='/')}"
        try:
            self._send_json("DELETE", self._url(path), None)
        except GitHubHTTPError as exc:
            if exc.status in (404, 422):
                return False
            raise
        return True

    def get_pull_request(self, number: int) -> dict[str, Any]:
        """Fetch a PR, including `mergeable` and `mergeable_state` (#34).

        GitHub computes mergeability asynchronously: the first read after a
        push routinely returns `mergeable: null` with `mergeable_state:
        "unknown"`. That is a valid answer meaning "ask again", not an error,
        so it is returned as-is and the caller polls.
        """
        return self._get(self._url(f"/repos/{self.repo}/pulls/{number}"))

    def list_pull_requests(
        self,
        *,
        state: str = "open",
        head: str | None = None,
        base: str | None = None,
    ) -> list[dict[str, Any]]:
        """List PRs, optionally narrowed to one head branch.

        Three modules probed for this method and degraded without it, each
        saying so out loud rather than guessing: `worker/pr.py` fell back to
        GitHub's 422 on a duplicate head, and `orchestrator/checks.py` and
        `orchestrator/mergeability.py` reported `blind` and decided nothing.
        Deciding nothing was the right refusal - an unreadable PR list must
        never be read as "the PR was closed" - but it also meant the merge
        gate was inert.

        `head` is qualified with the owner when it is not already, because
        `GET /pulls?head=branch` silently matches nothing: the API wants
        `owner:branch`.
        """
        params: dict[str, str] = {"state": state}
        if head:
            params["head"] = head if ":" in head else f"{self.repo.split('/')[0]}:{head}"
        if base:
            params["base"] = base
        return self._paginate(f"/repos/{self.repo}/pulls", params)

    def create_pull_request(
        self,
        *,
        title: str,
        head: str,
        base: str,
        body: str | None = None,
        draft: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"title": title, "head": head, "base": base, "draft": draft}
        if body is not None:
            payload["body"] = body
        return self._send_json("POST", self._url(f"/repos/{self.repo}/pulls"), payload)

    def update_pull_request(
        self,
        number: int,
        *,
        title: str | None = None,
        body: str | None = None,
        base: str | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in (("title", title), ("body", body), ("base", base), ("state", state)):
            if value is not None:
                payload[key] = value
        if not payload:
            raise ValueError("update_pull_request called with nothing to update")
        return self._send_json("PATCH", self._url(f"/repos/{self.repo}/pulls/{number}"), payload)

    def list_check_runs(self, ref: str) -> list[dict[str, Any]]:
        """Check runs for a commit sha or branch - the CI gate for `review` issues.

        Check runs, not commit statuses: CI here is GitHub Actions, which
        reports through the checks API. An empty list means no checks have been
        created yet, which is not the same as "all checks passed" and must not
        be read as one.
        """
        path = f"/repos/{self.repo}/commits/{urllib.parse.quote(ref, safe='')}/check-runs"
        return self._paginate(path, None, key="check_runs")

    def merge_pull_request(
        self,
        number: int,
        *,
        merge_method: str = "squash",
        sha: str | None = None,
        commit_title: str | None = None,
    ) -> dict[str, Any]:
        """Merge a PR.

        Pass `sha` to make the merge conditional on the head the caller
        inspected; GitHub answers 409 if the branch moved since. Both that and
        the 405 for "not mergeable" are 4xx, so they surface immediately rather
        than being retried into a merge of code nobody checked.
        """
        payload: dict[str, Any] = {"merge_method": merge_method}
        if sha is not None:
            payload["sha"] = sha
        if commit_title is not None:
            payload["commit_title"] = commit_title
        return self._send_json(
            "PUT", self._url(f"/repos/{self.repo}/pulls/{number}/merge"), payload
        )

    # --- milestones -------------------------------------------------------

    def list_milestones(self, *, state: str = "open") -> list[dict[str, Any]]:
        return self._paginate(f"/repos/{self.repo}/milestones", {"state": state})

    # --- plumbing ---------------------------------------------------------

    def _url(self, path: str, params: Mapping[str, str] | None = None) -> str:
        url = f"{self.api_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        return url

    def _get(self, url: str) -> Any:
        payload, _ = self._get_cached(url)
        return payload

    def _get_cached(self, url: str) -> tuple[Any, str | None]:
        """GET with `If-None-Match`, returning (payload, next page url).

        Conditional requests are the cheap half of the rate-limit problem: the
        reconcile loop (#22) re-reads the same issue list every cycle and a 304
        does not count against the primary limit, which is a budget shared with
        every worker. The `Link` header is cached alongside the payload because
        a 304 carries no body *and* no `Link`, so pagination would otherwise
        stop dead on the first cache hit.
        """
        cached = self.etag_cache.get(url)
        response = self._request("GET", url, if_none_match=cached[0] if cached else None)
        if response.status == 304:
            if cached is None:
                # An interposing cache answered a request we did not make
                # conditional. There is no body to fall back on, and treating
                # an empty one as data would report an empty ledger.
                raise GitHubError(f"GET {url} -> 304 with nothing cached to serve")
            return cached[1], cached[2]

        payload = response.json()
        next_url = _next_link(response.header("Link"))
        etag = response.header("ETag")
        if etag:
            self.etag_cache[url] = (etag, payload, next_url)
        return payload, next_url

    def _paginate(
        self,
        path: str,
        params: Mapping[str, str] | None,
        *,
        key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Walk `Link: rel="next"` to the end and concatenate.

        `key` names the array inside an envelope response - the checks API
        returns `{"total_count": n, "check_runs": [...]}` rather than a bare
        list.
        """
        query = dict(params or {})
        query.setdefault("per_page", "100")
        url = self._url(path, query)
        items: list[dict[str, Any]] = []
        for _ in range(MAX_PAGES):
            payload, next_url = self._get_cached(url)
            chunk = payload if key is None else (payload or {}).get(key)
            if isinstance(chunk, list):
                items.extend(chunk)
            if not next_url:
                return items
            url = next_url
        raise GitHubError(f"pagination did not terminate after {MAX_PAGES} pages: {path}")

    def _send_json(self, method: str, url: str, payload: Any) -> Any:
        response = self._request(method, url, payload=payload)
        return response.json()

    def _request(
        self,
        method: str,
        url: str,
        *,
        payload: Any = None,
        if_none_match: str | None = None,
    ) -> Response:
        """One request, retried according to `RetryPolicy`.

        The whole retry decision lives here so there is exactly one place to
        read when asking "why did the swarm hammer GitHub".
        """
        headers = dict(self._headers)
        body: bytes | None = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if if_none_match:
            headers["If-None-Match"] = if_none_match

        for attempt in range(1, self.retry.max_attempts + 1):
            last_attempt = attempt == self.retry.max_attempts
            try:
                response = self._transport.send(method, url, headers, body, self._timeout_s)
            except TransportError:
                if last_attempt:
                    raise
                self._sleep(self._backoff(attempt))
                continue

            if response.status < 400:
                return response

            throttled = _is_rate_limited(response)
            if not throttled and response.status < 500:
                # A 4xx is a verdict on this request. Retrying cannot change
                # the answer, and the caller wants the error now.
                raise GitHubHTTPError(response.status, method, url, response.body)

            requested = _retry_after_s(response) if throttled else None
            if last_attempt:
                if throttled:
                    raise RateLimitError(response.status, method, url, response.body, requested)
                raise GitHubHTTPError(response.status, method, url, response.body)

            if requested is not None and requested > self.retry.max_wait_s:
                raise RateLimitError(response.status, method, url, response.body, requested)
            self._sleep(requested if requested is not None else self._backoff(attempt))

        raise GitHubError(f"{method} {url}: retry loop exited without a result")

    def _backoff(self, attempt: int) -> float:
        delay = min(self.retry.backoff_base_s * (2 ** (attempt - 1)), self.retry.backoff_cap_s)
        if self.retry.jitter:
            delay += random.uniform(0.0, self.retry.jitter * delay)
        return delay


def _is_rate_limited(response: Response) -> bool:
    """Distinguish throttling from a genuine permission failure.

    Both arrive as 403, and the difference decides between "wait and retry" and
    "stop, the token is wrong". GitHub signals throttling three different ways
    depending on which limiter fired, so all three are checked.
    """
    if response.status == 429:
        return True
    if response.status != 403:
        return False
    if response.header("Retry-After"):
        return True
    if (response.header("X-RateLimit-Remaining") or "").strip() == "0":
        return True
    message = _error_message(response.body).lower()
    return "rate limit" in message or "abuse detection" in message


def _retry_after_s(response: Response) -> float | None:
    """How long GitHub asked us to wait, if it said. None means "use backoff"."""
    retry_after = response.header("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            return None
    reset = response.header("X-RateLimit-Reset")
    if reset and (response.header("X-RateLimit-Remaining") or "").strip() == "0":
        try:
            return max(0.0, float(reset) - time.time())
        except ValueError:
            return None
    return None


def _next_link(header: str | None) -> str | None:
    """Extract `rel="next"` from a `Link` header."""
    if not header:
        return None
    for part in header.split(","):
        segments = part.split(";")
        target = segments[0].strip()
        if len(segments) < 2 or not (target.startswith("<") and target.endswith(">")):
            continue
        for attribute in segments[1:]:
            normalised = attribute.strip().replace(" ", "").replace('"', "").lower()
            if normalised == "rel=next":
                return target[1:-1]
    return None


if __name__ == "__main__":  # pragma: no cover - manual smoke test, see module docstring
    import sys

    client = GitHubClient.from_env(sys.argv[1] if len(sys.argv) > 1 else None)
    for issue in client.list_issues():
        names = ",".join(label["name"] for label in issue.get("labels", []))
        print(f"#{issue['number']:<4} {issue['title']}  [{names}]")
