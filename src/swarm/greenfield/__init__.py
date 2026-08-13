"""Greenfield mode: a prompt goes in, a repository the swarm can work in comes out.

`docs/architecture-v2.md` splits v2 into two modes. Existing-repo mode plans
against code that is already there; greenfield mode has to build the ground
first, because an empty repository has no test command, therefore no quality
gate, therefore no way to judge the first worker's output.

Two steps, in this order:

- `provision` creates the repository, the initial commit, the labels and the
  branch protection - the outward-facing, irreversible half, which is why it
  asks before it acts;
- `scaffold` (#26) fills it with a minimal skeleton whose tests actually run,
  so the first planned task has something to verify against.
"""

from __future__ import annotations

from .provision import (
    CHECK_NAME,
    RULESET_NAME,
    ProvisionAborted,
    ProvisionError,
    ProvisionPlan,
    ProvisionReport,
    provision,
)

__all__ = [
    "CHECK_NAME",
    "RULESET_NAME",
    "ProvisionAborted",
    "ProvisionError",
    "ProvisionPlan",
    "ProvisionReport",
    "provision",
]
