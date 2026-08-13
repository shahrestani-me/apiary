"""Unit tests for label provisioning, against a repo that actually stores labels.

The acceptance criterion here is a property of two runs, not of one - "running
it twice produces the labels once and reports no changes the second time" - so
most of these tests drive `FakeRepo`, a transport that keeps the labels it is
told to create and serves them back. A scripted response list can only assert
what the first run sends; a store can be asked whether the second run was a
no-op, which is the actual claim.

`FakeRepo` and the helpers below are written to be lifted into the shared
fixture set (#31), alongside `test_github_client.py`'s `FakeTransport`.
"""

from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Mapping

import pytest

from swarm.github.client import GitHubClient, GitHubError, GitHubHTTPError, Response
from swarm.github.labels import (
    SWARM_LABELS,
    LabelReport,
    LabelSpec,
    ensure_labels,
    list_label_names,
)

API = "https://api.github.com"
REPO = "shahrestani-me/apiary"


# --------------------------------------------------------------------------
# A repo with just enough label API to run ensure_labels for real (#31)
# --------------------------------------------------------------------------


def response(status: int, payload: Any = None, **headers: str) -> Response:
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    return Response(status, headers, body)


@dataclass
class FakeRepo:
    """Serves `GET/POST /repos/{repo}/labels` out of a dict.

    Duplicate creation answers 422 `already_exists` exactly as GitHub does,
    matching names case-insensitively - that is GitHub's uniqueness rule, and
    a fake that allowed `Swarm:Ready` beside `swarm:ready` would let a bug
    through.
    """

    labels: dict[str, dict[str, str]] = field(default_factory=dict)
    requests: list[tuple[str, str, Any]] = field(default_factory=list)
    pages: int = 1

    @classmethod
    def with_labels(cls, *labels: Mapping[str, str], **kwargs: Any) -> FakeRepo:
        repo = cls(**kwargs)
        for label in labels:
            repo.labels[label["name"].casefold()] = dict(label)
        return repo

    @property
    def posted(self) -> list[Any]:
        return [payload for method, _, payload in self.requests if method == "POST"]

    def send(self, method, url, headers, body, timeout):
        payload = json.loads(body.decode("utf-8")) if body else None
        split = urllib.parse.urlsplit(url)
        self.requests.append((method, split.path, payload))
        assert split.path == f"/repos/{REPO}/labels", f"unexpected path {split.path}"

        if method == "GET":
            return self._get(split)
        if method == "POST":
            return self._post(payload)
        raise AssertionError(f"unexpected method {method}")

    def _get(self, split: urllib.parse.SplitResult) -> Response:
        # Hand every label out one page at a time when asked to paginate, so
        # the "already present" set is only complete if the caller follows the
        # Link header to the end.
        page = int(dict(urllib.parse.parse_qsl(split.query)).get("page", "1"))
        chunks = _chunk(list(self.labels.values()), self.pages)
        headers = {}
        if page < len(chunks):
            headers["Link"] = f'<{API}{split.path}?page={page + 1}>; rel="next"'
        return response(200, chunks[page - 1] if page <= len(chunks) else [], **headers)

    def _post(self, payload: Any) -> Response:
        key = payload["name"].casefold()
        if key in self.labels:
            return response(
                422,
                {
                    "message": "Validation Failed",
                    "errors": [
                        {"resource": "Label", "code": "already_exists", "field": "name"}
                    ],
                },
            )
        self.labels[key] = dict(payload)
        return response(201, self.labels[key])


def _chunk(items: list[Any], pages: int) -> list[list[Any]]:
    if pages <= 1 or not items:
        return [items]
    size = -(-len(items) // pages)
    return [items[start:start + size] for start in range(0, len(items), size)]


def client_for(repo: FakeRepo) -> GitHubClient:
    return GitHubClient(REPO, "test-token", api_url=API, transport=repo)


# --------------------------------------------------------------------------
# The label set itself
# --------------------------------------------------------------------------


def test_the_set_is_exactly_the_six_state_labels():
    # docs/issue-contract.md §3: one of six, and the label set is authoritative.
    assert [spec.name for spec in SWARM_LABELS] == [
        "swarm:ready",
        "swarm:blocked",
        "swarm:claimed",
        "swarm:review",
        "swarm:done",
        "swarm:failed",
    ]


def test_no_attempt_labels_are_provisioned():
    # docs/issue-contract.md §5: the counter is a marker body field, and
    # `swarm:attempt/1..3` is removed from the protocol. Provisioning them
    # would put a second, competing counter in the ledger.
    assert not [spec for spec in SWARM_LABELS if spec.name.startswith("swarm:attempt")]


def test_specs_are_validated_at_construction():
    # A bad colour is a 422 halfway through a run otherwise.
    with pytest.raises(ValueError, match="hex"):
        LabelSpec("swarm:x", "#0e8a16", "leading hash is not what the API wants")
    with pytest.raises(ValueError, match="hex"):
        LabelSpec("swarm:x", "green", "not hex at all")
    with pytest.raises(ValueError, match="description"):
        LabelSpec("swarm:x", "0e8a16", "d" * 101)


# --------------------------------------------------------------------------
# The acceptance criterion: create once, then nothing
# --------------------------------------------------------------------------


def test_a_fresh_repo_gets_every_label_with_its_colour_and_description():
    repo = FakeRepo()
    report = ensure_labels(client_for(repo))

    assert report.created == tuple(spec.name for spec in SWARM_LABELS)
    assert report.existing == ()
    assert report.changed is True
    assert repo.posted == [
        {"name": spec.name, "color": spec.color, "description": spec.description}
        for spec in SWARM_LABELS
    ]


def test_the_second_run_creates_nothing_and_reports_no_changes():
    repo = FakeRepo()
    first = ensure_labels(client_for(repo))
    posts_after_first = len(repo.posted)

    second = ensure_labels(client_for(repo))

    assert first.changed is True
    assert second.changed is False
    assert second.created == ()
    assert second.existing == tuple(spec.name for spec in SWARM_LABELS)
    assert len(repo.posted) == posts_after_first     # not one further write
    assert len(repo.labels) == len(SWARM_LABELS)


def test_only_the_missing_labels_are_created():
    repo = FakeRepo.with_labels({"name": "swarm:done", "color": "c2e0c6", "description": ""})
    report = ensure_labels(client_for(repo))

    assert report.existing == ("swarm:done",)
    assert "swarm:done" not in report.created
    assert [payload["name"] for payload in repo.posted] == [
        spec.name for spec in SWARM_LABELS if spec.name != "swarm:done"
    ]


# --------------------------------------------------------------------------
# Do not fight the user's repo
# --------------------------------------------------------------------------


def test_an_existing_label_keeps_its_own_colour_and_description():
    # Recolouring is a deliberate human edit; a run that reverts it every cycle
    # is the system arguing with its user.
    theirs = {"name": "swarm:ready", "color": "ff00ff", "description": "mine, thanks"}
    repo = FakeRepo.with_labels(theirs)

    ensure_labels(client_for(repo))

    assert repo.labels["swarm:ready"] == theirs
    assert not [method for method, _, _ in repo.requests if method in {"PATCH", "PUT"}]


def test_names_are_matched_case_insensitively():
    # GitHub's uniqueness rule is case-insensitive, so a POST here would be a
    # guaranteed 422 rather than a second label.
    repo = FakeRepo.with_labels({"name": "Swarm:Ready", "color": "0e8a16", "description": ""})
    report = ensure_labels(client_for(repo))

    assert "swarm:ready" in report.existing
    assert "swarm:ready" not in [payload["name"] for payload in repo.posted]


def test_labels_outside_the_swarm_namespace_are_untouched():
    repo = FakeRepo.with_labels(
        {"name": "bug", "color": "d73a4a", "description": ""},
        {"name": "area/control-plane", "color": "0052cc", "description": ""},
    )
    ensure_labels(client_for(repo))

    assert repo.labels["bug"]["color"] == "d73a4a"
    assert len(repo.labels) == len(SWARM_LABELS) + 2


# --------------------------------------------------------------------------
# Reading the existing set
# --------------------------------------------------------------------------


def test_existing_labels_are_read_across_every_page():
    # A label sitting on page two looks missing to a single-page read, and the
    # POST that follows is a 422 on a repo that was already correct.
    repo = FakeRepo(pages=3)
    for spec in SWARM_LABELS:
        repo.labels[spec.name] = {"name": spec.name, "color": spec.color, "description": ""}

    report = ensure_labels(client_for(repo))

    assert report.changed is False
    assert repo.posted == []


def test_list_label_names_returns_what_the_repo_has():
    repo = FakeRepo.with_labels(
        {"name": "swarm:ready", "color": "0e8a16", "description": ""},
        {"name": "bug", "color": "d73a4a", "description": ""},
    )
    assert list_label_names(client_for(repo)) == ["swarm:ready", "bug"]


# --------------------------------------------------------------------------
# Races and errors
# --------------------------------------------------------------------------


def test_losing_the_create_race_counts_as_present_not_as_a_failure():
    # Another orchestrator provisioning the same repo between our list and our
    # POST produces the end state we wanted anyway.
    class RacingRepo(FakeRepo):
        def _get(self, split):
            # Report an empty repo, but have the label stored by the time the
            # POST arrives - exactly the window the race lives in.
            self.labels["swarm:ready"] = {"name": "swarm:ready", "color": "0e8a16"}
            return response(200, [])

    racing = RacingRepo()
    report = ensure_labels(client_for(racing), labels=(SWARM_LABELS[0],))

    assert report.created == ()
    assert report.existing == ("swarm:ready",)
    assert report.changed is False
    assert len(racing.posted) == 1              # it did try, and the 422 was absorbed


def test_a_422_that_is_not_already_exists_still_raises():
    # Swallowing every 422 would report success on a repo permanently missing
    # the label - an invalid colour comes back the same way.
    class RejectingRepo(FakeRepo):
        def _post(self, payload):
            return response(
                422,
                {
                    "message": "Validation Failed",
                    "errors": [{"resource": "Label", "code": "invalid", "field": "color"}],
                },
            )

    with pytest.raises(GitHubHTTPError) as exc:
        ensure_labels(client_for(RejectingRepo()))
    assert exc.value.status == 422


def test_a_permission_failure_surfaces_immediately():
    # A token without `issues: write` cannot provision anything, and a run that
    # continues past it fails later, further from the cause.
    class ForbiddenRepo(FakeRepo):
        def _post(self, payload):
            return response(403, {"message": "Resource not accessible by integration"})

    with pytest.raises(GitHubHTTPError) as exc:
        ensure_labels(client_for(ForbiddenRepo()))
    assert exc.value.status == 403


# --------------------------------------------------------------------------
# Call shape
# --------------------------------------------------------------------------


def test_a_repo_string_builds_a_client_from_the_environment(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(GitHubError, match="GITHUB_TOKEN"):
        ensure_labels(REPO)


def test_the_caller_may_supply_its_own_label_set():
    repo = FakeRepo()
    spec = LabelSpec("swarm:custom", "ededed", "not part of the protocol")
    report = ensure_labels(client_for(repo), labels=(spec,))

    assert report.created == ("swarm:custom",)
    assert list(repo.labels) == ["swarm:custom"]


def test_the_report_summarises_both_outcomes():
    created = LabelReport(REPO, created=("swarm:ready",), existing=("swarm:done",))
    assert "created swarm:ready" in created.summary()

    unchanged = LabelReport(REPO, existing=("swarm:ready",))
    assert "no changes" in unchanged.summary()
    assert unchanged.changed is False
