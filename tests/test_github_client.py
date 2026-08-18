"""Unit tests for the GitHub client, against a scripted transport.

No network and no token: every test drives `FakeTransport`, which replays a
list of canned responses and records what was asked of it. That is the whole
point of the `Transport` seam - the retry and rate-limit behaviour is the part
most likely to be wrong and the part least testable against the real API,
since you cannot ask GitHub for a 503 on demand.

`FakeTransport` and the `response()` helper below are written to be lifted
wholesale into the shared fixture set (#31); they know nothing about the
individual tests.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import pytest

from swarm.github.client import (
    GitHubClient,
    GitHubError,
    GitHubHTTPError,
    RateLimitError,
    Response,
    RetryPolicy,
    TransportError,
)

API = "https://api.github.com"
REPO = "shahrestani-me/apiary"


# --------------------------------------------------------------------------
# The mocked transport (#31 will lift this into shared fixtures)
# --------------------------------------------------------------------------


@dataclass
class SentRequest:
    method: str
    url: str
    headers: Mapping[str, str]
    body: bytes | None

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8")) if self.body else None


@dataclass
class FakeTransport:
    """Replays scripted responses in order; records every request.

    A scripted entry may be an exception instance, which is raised instead of
    returned - that is how transport-level failures are simulated. Running out
    of scripted responses is an assertion failure rather than a default,
    because an unexpected extra request is exactly the bug these tests exist
    to catch.
    """

    script: list[Any]
    sent: list[SentRequest] = field(default_factory=list)

    def send(self, method, url, headers, body, timeout):
        self.sent.append(SentRequest(method, url, dict(headers), body))
        assert self.script, f"unscripted request: {method} {url}"
        nxt = self.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def response(status: int, payload: Any = None, **headers: str) -> Response:
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    return Response(status, headers, body)


def client(*script: Any, **kwargs: Any) -> tuple[GitHubClient, FakeTransport, list[float]]:
    """A client wired to a scripted transport and a recording sleep."""
    transport = FakeTransport(list(script))
    slept: list[float] = []
    retry = kwargs.pop("retry", RetryPolicy(max_attempts=3, backoff_base_s=0.5, jitter=0.0))
    gh = GitHubClient(
        REPO,
        "test-token",
        api_url=API,
        transport=transport,
        retry=retry,
        sleep=slept.append,
        **kwargs,
    )
    return gh, transport, slept


# --------------------------------------------------------------------------
# Request shape
# --------------------------------------------------------------------------


def test_every_request_carries_auth_and_api_version():
    gh, transport, _ = client(response(200, {"number": 7}))
    gh.get_issue(7)

    sent = transport.sent[0]
    assert sent.method == "GET"
    assert sent.url == f"{API}/repos/{REPO}/issues/7"
    assert sent.headers["Authorization"] == "Bearer test-token"
    assert sent.headers["X-GitHub-Api-Version"] == "2022-11-28"
    assert sent.headers["Accept"] == "application/vnd.github+json"


def test_repo_must_be_owner_slash_name():
    with pytest.raises(ValueError):
        GitHubClient("apiary", "t")


def test_from_env_requires_a_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(GitHubError, match="GITHUB_TOKEN"):
        GitHubClient.from_env(REPO)


# --------------------------------------------------------------------------
# Issues
# --------------------------------------------------------------------------


def test_list_issues_excludes_pull_requests():
    # GET /issues returns PRs too; a PR in the ledger would be dispatched as a
    # task whose body has no contract in it.
    gh, _, _ = client(
        response(200, [
            {"number": 7, "title": "issue"},
            {"number": 8, "title": "pr", "pull_request": {"url": "..."}},
        ])
    )
    assert [i["number"] for i in gh.list_issues()] == [7]


def test_list_issues_passes_state_and_labels():
    gh, transport, _ = client(response(200, []))
    gh.list_issues(state="all", labels=["swarm:ready", "area/control-plane"])

    url = transport.sent[0].url
    assert "state=all" in url
    assert "labels=swarm%3Aready%2Carea%2Fcontrol-plane" in url
    assert "per_page=100" in url


def test_pagination_follows_the_next_link():
    page2 = f"{API}/repos/{REPO}/issues?page=2"
    gh, transport, _ = client(
        response(200, [{"number": 1}], Link=f'<{page2}>; rel="next", <{page2}>; rel="last"'),
        response(200, [{"number": 2}]),
    )
    assert [i["number"] for i in gh.list_issues()] == [1, 2]
    assert transport.sent[1].url == page2


def test_create_issue_posts_the_payload():
    gh, transport, _ = client(response(201, {"number": 42}))
    gh.create_issue("t", body="b", labels=["swarm:ready"], milestone=1)

    sent = transport.sent[0]
    assert sent.method == "POST"
    assert sent.url == f"{API}/repos/{REPO}/issues"
    assert sent.json() == {"title": "t", "body": "b", "labels": ["swarm:ready"], "milestone": 1}


def test_update_issue_cannot_clobber_labels():
    # PATCH /issues replaces the label set, so the client refuses to expose it;
    # labels move only through the additive endpoints.
    gh, transport, _ = client(response(200, {"number": 7}))
    gh.update_issue(7, body="new body", state="closed", state_reason="completed")

    sent = transport.sent[0]
    assert sent.method == "PATCH"
    assert sent.json() == {"body": "new body", "state": "closed", "state_reason": "completed"}
    with pytest.raises(TypeError):
        gh.update_issue(7, labels=["swarm:done"])


def test_update_issue_with_no_fields_is_a_programming_error():
    gh, _, _ = client()
    with pytest.raises(ValueError):
        gh.update_issue(7)


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def test_add_labels_is_additive():
    gh, transport, _ = client(response(200, [{"name": "swarm:claimed"}]))
    gh.add_labels(7, ["swarm:claimed"])

    sent = transport.sent[0]
    assert sent.method == "POST"
    assert sent.url == f"{API}/repos/{REPO}/issues/7/labels"
    assert sent.json() == {"labels": ["swarm:claimed"]}


def test_remove_label_url_encodes_the_name():
    gh, transport, _ = client(response(200, []))
    assert gh.remove_label(7, "swarm:ready") is True
    assert transport.sent[0].method == "DELETE"
    assert transport.sent[0].url == f"{API}/repos/{REPO}/issues/7/labels/swarm%3Aready"


def test_remove_label_tolerates_an_already_absent_label():
    # A human who removed it first must not fail the reconcile cycle.
    gh, _, _ = client(response(404, {"message": "Label does not exist"}))
    assert gh.remove_label(7, "swarm:ready") is False


def test_list_issue_comments_walks_the_whole_thread():
    """The read half of the retry-feedback channel, paginated like every listing.

    The newest comment is the one a retrying worker wants, and on a
    much-argued issue it lives on the last page - so stopping at the first
    would hand the worker the oldest feedback instead of the latest.
    """
    page2 = f"{API}/repos/{REPO}/issues/7/comments?page=2"
    gh, transport, _ = client(
        response(200, [{"id": 1, "body": "human chatter"}],
                 Link=f'<{page2}>; rel="next", <{page2}>; rel="last"'),
        response(200, [{"id": 2, "body": "apiary: attempt 1 failed. details"}]),
    )

    comments = gh.list_issue_comments(7)

    assert [c["id"] for c in comments] == [1, 2]
    sent = transport.sent[0]
    assert sent.method == "GET"
    assert sent.url.startswith(f"{API}/repos/{REPO}/issues/7/comments")
    assert "per_page=100" in sent.url


# --------------------------------------------------------------------------
# Pull requests
# --------------------------------------------------------------------------


def test_get_pull_request_returns_mergeable_state_verbatim():
    # `unknown` means "GitHub is still computing, ask again" - a valid answer
    # the client must not translate or raise on (#34).
    gh, _, _ = client(response(200, {"number": 3, "mergeable": None, "mergeable_state": "unknown"}))
    pr = gh.get_pull_request(3)
    assert pr["mergeable"] is None
    assert pr["mergeable_state"] == "unknown"


def test_create_pull_request_posts_head_base_and_draft():
    gh, transport, _ = client(response(201, {"number": 3}))
    gh.create_pull_request(title="t", head="swarm/issue-7", base="main", body="Closes #7")

    assert transport.sent[0].json() == {
        "title": "t",
        "head": "swarm/issue-7",
        "base": "main",
        "draft": False,
        "body": "Closes #7",
    }


def test_update_pull_request_patches():
    gh, transport, _ = client(response(200, {"number": 3}))
    gh.update_pull_request(3, state="closed")
    assert transport.sent[0].method == "PATCH"
    assert transport.sent[0].json() == {"state": "closed"}


def test_list_check_runs_unwraps_the_envelope():
    gh, transport, _ = client(
        response(200, {"total_count": 2, "check_runs": [{"conclusion": "success"}, {"conclusion": "failure"}]})
    )
    runs = gh.list_check_runs("deadbeef")
    assert [r["conclusion"] for r in runs] == ["success", "failure"]
    assert transport.sent[0].url.startswith(f"{API}/repos/{REPO}/commits/deadbeef/check-runs")


def test_merge_pull_request_can_pin_the_head_sha():
    gh, transport, _ = client(response(200, {"merged": True}))
    gh.merge_pull_request(3, sha="deadbeef")
    assert transport.sent[0].method == "PUT"
    assert transport.sent[0].json() == {"merge_method": "squash", "sha": "deadbeef"}


def test_merge_conflict_surfaces_immediately():
    # 405 means "not mergeable". Retrying it would either keep failing or
    # succeed later against code nobody re-checked.
    gh, transport, slept = client(response(405, {"message": "Pull Request is not mergeable"}))
    with pytest.raises(GitHubHTTPError) as exc:
        gh.merge_pull_request(3)
    assert exc.value.status == 405
    assert "not mergeable" in str(exc.value)
    assert len(transport.sent) == 1
    assert slept == []


def test_list_milestones():
    gh, transport, _ = client(response(200, [{"number": 1, "title": "M1 Control plane"}]))
    assert gh.list_milestones()[0]["title"] == "M1 Control plane"
    assert "state=open" in transport.sent[0].url


# --------------------------------------------------------------------------
# Retry behaviour
# --------------------------------------------------------------------------


def test_5xx_is_retried_then_succeeds():
    gh, transport, slept = client(response(502, {"message": "Bad gateway"}), response(200, {"number": 7}))
    assert gh.get_issue(7)["number"] == 7
    assert len(transport.sent) == 2
    assert slept == [0.5]


def test_backoff_grows_and_retries_are_bounded():
    gh, transport, slept = client(*[response(503, {"message": "unavailable"})] * 3)
    with pytest.raises(GitHubHTTPError) as exc:
        gh.get_issue(7)
    assert exc.value.status == 503
    assert len(transport.sent) == 3            # max_attempts, not more
    assert slept == [0.5, 1.0]                 # no sleep after the last attempt


def test_4xx_is_not_retried():
    gh, transport, slept = client(response(404, {"message": "Not Found"}))
    with pytest.raises(GitHubHTTPError) as exc:
        gh.get_issue(7)
    assert exc.value.status == 404
    assert len(transport.sent) == 1
    assert slept == []


def test_422_reports_the_validation_errors():
    gh, _, _ = client(
        response(422, {"message": "Validation Failed", "errors": [{"field": "head", "code": "invalid"}]})
    )
    with pytest.raises(GitHubHTTPError) as exc:
        gh.create_pull_request(title="t", head="nope", base="main")
    assert "Validation Failed" in str(exc.value)
    assert "invalid" in str(exc.value)


def test_403_without_throttling_signals_is_a_permission_error():
    # The token lacking a scope must fail now, not after three backoffs.
    gh, transport, slept = client(response(403, {"message": "Resource not accessible by integration"}))
    with pytest.raises(GitHubHTTPError) as exc:
        gh.add_labels(7, ["swarm:ready"])
    assert not isinstance(exc.value, RateLimitError)
    assert len(transport.sent) == 1
    assert slept == []


def test_transport_failures_are_retried():
    gh, transport, slept = client(TransportError("connection reset"), response(200, {"number": 7}))
    assert gh.get_issue(7)["number"] == 7
    assert len(transport.sent) == 2
    assert slept == [0.5]


def test_transport_failure_propagates_once_attempts_run_out():
    gh, _, _ = client(*[TransportError("connection reset")] * 3)
    with pytest.raises(TransportError):
        gh.get_issue(7)


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


def test_secondary_rate_limit_waits_exactly_as_asked():
    gh, transport, slept = client(
        response(403, {"message": "You have exceeded a secondary rate limit"}, **{"Retry-After": "2"}),
        response(200, {"number": 7}),
    )
    assert gh.get_issue(7)["number"] == 7
    assert slept == [2.0]                      # GitHub's number, not our backoff
    assert len(transport.sent) == 2


def test_secondary_rate_limit_without_retry_after_falls_back_to_backoff():
    gh, _, slept = client(
        response(403, {"message": "You have exceeded a secondary rate limit"}),
        response(200, {"number": 7}),
    )
    assert gh.get_issue(7)["number"] == 7
    assert slept == [0.5]


def test_429_is_treated_as_throttling():
    gh, _, slept = client(response(429, {"message": "Too many requests"}), response(200, {"number": 7}))
    assert gh.get_issue(7)["number"] == 7
    assert slept == [0.5]


def test_primary_rate_limit_waits_when_the_reset_is_near():
    gh, _, slept = client(
        response(
            403,
            {"message": "API rate limit exceeded"},
            **{"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(time.time()) + 5)},
        ),
        response(200, {"number": 7}),
    )
    assert gh.get_issue(7)["number"] == 7
    assert len(slept) == 1 and 0 < slept[0] <= 5


def test_primary_rate_limit_refuses_to_sleep_out_a_distant_reset():
    # An orchestrator asleep for an hour is indistinguishable from a hung one.
    reset = int(time.time()) + 3600
    gh, transport, slept = client(
        response(
            403,
            {"message": "API rate limit exceeded"},
            **{"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(reset)},
        )
    )
    with pytest.raises(RateLimitError) as exc:
        gh.get_issue(7)
    assert exc.value.retry_after_s > 3000
    assert slept == []
    assert len(transport.sent) == 1


def test_persistent_throttling_raises_rate_limit_error_not_a_plain_http_error():
    gh, _, slept = client(
        *[response(429, {"message": "Too many requests"})] * 3
    )
    with pytest.raises(RateLimitError):
        gh.get_issue(7)
    assert slept == [0.5, 1.0]


# --------------------------------------------------------------------------
# Conditional requests
# --------------------------------------------------------------------------


def test_second_read_sends_if_none_match_and_a_304_serves_the_cache():
    gh, transport, _ = client(
        response(200, [{"number": 7}], ETag='W/"abc"'),
        response(304, None, ETag='W/"abc"'),
    )
    first = gh.list_issues()
    second = gh.list_issues()

    assert first == second == [{"number": 7}]
    assert "If-None-Match" not in transport.sent[0].headers
    assert transport.sent[1].headers["If-None-Match"] == 'W/"abc"'


def test_the_etag_cache_is_injectable_and_shared_across_clients():
    # The reconcile loop re-reads the same issue list every cycle; a cache that
    # dies with the client saves no rate-limit budget.
    shared: dict = {}
    first_client, _, _ = client(response(200, [{"number": 7}], ETag='W/"abc"'), etag_cache=shared)
    first_client.list_issues()
    assert shared

    second_client, transport, _ = client(response(304, None), etag_cache=shared)
    assert second_client.list_issues() == [{"number": 7}]
    assert transport.sent[0].headers["If-None-Match"] == 'W/"abc"'


def test_a_304_still_knows_where_the_next_page_is():
    # A 304 carries no Link header, so the cached entry has to remember it or
    # pagination silently truncates to one page on every cache hit.
    page2 = f"{API}/repos/{REPO}/issues?page=2"
    gh, _, _ = client(
        response(200, [{"number": 1}], ETag='W/"p1"', Link=f'<{page2}>; rel="next"'),
        response(200, [{"number": 2}], ETag='W/"p2"'),
        response(304, None),
        response(304, None),
    )
    assert [i["number"] for i in gh.list_issues()] == [1, 2]
    assert [i["number"] for i in gh.list_issues()] == [1, 2]


def test_headers_are_matched_case_insensitively():
    # urllib gives a case-insensitive mapping, a fake transport gives a dict.
    gh, _, _ = client(
        response(200, [{"number": 7}], etag='W/"abc"'),
        response(304, None),
    )
    gh.list_issues()
    assert gh.list_issues() == [{"number": 7}]


def test_an_unexpected_304_raises_rather_than_reporting_an_empty_ledger():
    gh, _, _ = client(response(304, None))
    with pytest.raises(GitHubError, match="nothing cached"):
        gh.list_issues()


def test_pagination_that_never_terminates_raises():
    same = f"{API}/repos/{REPO}/milestones?state=open&per_page=100"
    gh, _, _ = client(*[response(200, [{"number": 1}], Link=f'<{same}>; rel="next"')] * 200)
    with pytest.raises(GitHubError, match="did not terminate"):
        gh.list_milestones()
