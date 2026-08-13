"""Provision the `swarm:*` labels the protocol needs on a target repository.

Labels are the protocol (`docs/architecture-v2.md`), so a repo missing one is a
repo where a transition silently cannot be written: `POST /issues/{n}/labels`
with an unknown name creates the label with a random colour and no description,
and the ledger ends up carrying labels nobody chose. This module puts the six
state labels there deliberately, once, before anything reads or writes them.

It runs at the start of every run, so the whole design is "idempotent or
nothing":

**Create what is missing, touch nothing that exists.** A label already present
is left exactly as the repository has it - colour, description and all. Someone
who recoloured `swarm:ready` to match their own scheme meant it, and a run that
reverts that edit every cycle is fighting the user's repo. The names are the
protocol; the appearance is not.

**No attempt labels.** `docs/issue-contract.md` §5 moved the retry counter into
the identity marker's `attempt=` field and removed `swarm:attempt/1..3` from the
protocol. That is also what dissolves the collision this issue anticipated -
"create on demand" versus "leave existing alone" only conflicts for labels whose
*value* changes, and no label here has one.

Only names are compared, and case-insensitively, because GitHub's own uniqueness
rule for label names is case-insensitive: a repo carrying `Swarm:Ready` will
answer a `POST` of `swarm:ready` with a 422, not a second label.

This module reaches GitHub through `GitHubClient`'s request plumbing rather than
opening its own - the one-door rule in `client.py` is the point of that module.
Repo-label endpoints are not on the client's public surface (it exposes the
*issue* label endpoints v2 dispatches through), and adding them there is outside
this issue's file contract, so the two calls below use the client's
package-internal helpers and inherit its retry, pagination and error handling.

Manual smoke test against a real repo:

    GITHUB_TOKEN=... python -m swarm.github.labels shahrestani-me/apiary
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from .client import GitHubClient, GitHubHTTPError

# GitHub's own limits on the fields we send.
_COLOR_RE = re.compile(r"^[0-9a-f]{6}$")
_MAX_DESCRIPTION = 100


@dataclass(frozen=True)
class LabelSpec:
    """One label as this project wants it created.

    Validated at construction so a typo in `SWARM_LABELS` fails at import time
    rather than as a 422 in the middle of the first cycle of a run.
    """

    name: str
    color: str          # six lowercase hex digits, no leading '#', as the API wants
    description: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("label name must not be empty")
        if not _COLOR_RE.match(self.color):
            raise ValueError(f"{self.name}: colour must be six hex digits, got {self.color!r}")
        if len(self.description) > _MAX_DESCRIPTION:
            raise ValueError(f"{self.name}: description exceeds {_MAX_DESCRIPTION} characters")


# The six state labels of `docs/issue-contract.md` §3, and nothing else. Routing
# labels (`area/*`, `size/*`) are planner-assigned and open-ended - there is no
# fixed set to provision - and the attempt counter is not label-shaped (§5).
SWARM_LABELS: tuple[LabelSpec, ...] = (
    LabelSpec("swarm:ready", "0e8a16", "Dependencies met; the dispatcher may pick this up"),
    LabelSpec("swarm:blocked", "fbca04", "Waiting on another issue listed under ## Blocked by"),
    LabelSpec("swarm:claimed", "1d76db", "A worker container holds this issue right now"),
    LabelSpec("swarm:review", "5319e7", "PR open, awaiting checks and review"),
    LabelSpec("swarm:done", "c2e0c6", "PR merged; the work is on the base branch"),
    LabelSpec("swarm:failed", "b60205", "Attempts exhausted or contract malformed; needs a human"),
)


@dataclass(frozen=True)
class LabelReport:
    """What one `ensure_labels` call did, in the shape a caller can log.

    `changed` is the interesting bit: the second run against the same repo must
    report False, and a run that keeps reporting True is provisioning labels
    something else keeps deleting.
    """

    repo: str
    created: tuple[str, ...] = ()
    existing: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.created)

    def summary(self) -> str:
        if not self.created:
            return f"{self.repo}: {len(self.existing)} swarm labels already present, no changes"
        return (
            f"{self.repo}: created {', '.join(self.created)}"
            f" ({len(self.existing)} already present)"
        )


def ensure_labels(
    target: GitHubClient | str,
    *,
    labels: Sequence[LabelSpec] = SWARM_LABELS,
) -> LabelReport:
    """Create any missing label from `labels`; leave every existing one alone.

    `target` is a `GitHubClient`, or an `owner/name` string to build one from
    the environment. Returns a `LabelReport`; raises whatever the client raises
    for anything that is not "this label is already there".
    """
    client = GitHubClient.from_env(target) if isinstance(target, str) else target

    present = {name.casefold() for name in list_label_names(client)}
    created: list[str] = []
    existing: list[str] = []

    for spec in labels:
        if spec.name.casefold() in present:
            existing.append(spec.name)
            continue
        # Two orchestrators may provision the same repo at the same moment, and
        # the window between the list above and this POST is wide enough to
        # matter. Losing that race is the desired end state, not a failure.
        (created if _create_label(client, spec) else existing).append(spec.name)
        present.add(spec.name.casefold())

    return LabelReport(client.repo, tuple(created), tuple(existing))


def list_label_names(client: GitHubClient) -> list[str]:
    """Every label name defined on the repository, in GitHub's order."""
    labels = client._paginate(f"/repos/{client.repo}/labels", None)
    return [label["name"] for label in labels if label.get("name")]


def _create_label(client: GitHubClient, spec: LabelSpec) -> bool:
    """Create one label. False means it already existed, which is not an error."""
    payload = {"name": spec.name, "color": spec.color, "description": spec.description}
    try:
        client._send_json("POST", client._url(f"/repos/{client.repo}/labels"), payload)
    except GitHubHTTPError as exc:
        if exc.status == 422 and _is_already_exists(exc):
            return False
        raise
    return True


def _is_already_exists(exc: GitHubHTTPError) -> bool:
    """Tell "this label exists" apart from every other way a 422 can happen.

    A 422 is also how an invalid colour or an over-long description comes back,
    and swallowing those would leave the repo permanently missing a label while
    reporting success.
    """
    try:
        payload = json.loads(exc.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return False
    errors: Any = payload.get("errors") if isinstance(payload, dict) else None
    if not isinstance(errors, list):
        return False
    return any(
        isinstance(error, dict) and error.get("code") == "already_exists" for error in errors
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke test, see module docstring
    import os
    import sys

    repo = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GITHUB_REPOSITORY", "")
    print(ensure_labels(repo).summary())
