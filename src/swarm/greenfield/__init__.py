"""Greenfield mode: a prompt goes in, a repository the swarm can work in comes out.

`docs/architecture-v2.md` splits v2 into two modes. Existing-repo mode plans
against code that is already there; greenfield mode has to build the ground
first, because an empty repository has no test command, therefore no quality
gate, therefore no way to judge the first worker's output.

Two modules, and deliberately not two steps:

- `provision` creates the repository, the initial commit, the labels and the
  branch protection - the outward-facing, irreversible half, which is why it
  asks before it acts;
- `scaffold` (#26) generates the minimal skeleton whose tests actually run, and
  the bootstrap issue generates it in a worker, and its pull request replaces
  the placeholder gate the initial commit was provisioned with.

The composition is the point. A scaffold committed second would leave the first
commit - the one the required status check first reports on, and the one a
worker branches from - with no tests and a placeholder command. So `cli`
provisions a plain plan carrying the resolved stack, and the command it declares is the one
string the workflow runs and every planned issue carries as its `## Verify`.
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
