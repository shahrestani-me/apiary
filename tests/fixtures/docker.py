"""A Docker daemon that answers `ps`, `logs` and `rm` from a container table.

The double behind every hermetic reaper and container test. It parses
`--filter` for real rather than ignoring it, which is what makes the selection
logic - "remove exactly the orphans, and never list somebody's whole machine" -
testable without a daemon: a sweep that dropped its label filter gets an
assertion here instead of the developer's database containers.

Here rather than inside one test module because two suites need it now:
`test_reaper.py` for the policy, and `test_console_run.py` to prove that the
console's Stop really does dispose the containers a run spawned.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Sequence

from swarm.containers.manager import ISSUE_LABEL
from swarm.run import RUN_LABEL

__all__ = ["Container", "Daemon"]


@dataclass
class Container:
    """One container on the fake daemon."""

    id: str
    run_id: str | None = None
    issue: int | None = None
    name: str = "probe"
    image: str = "apiary-worker"
    logs: str = ""
    #: What `docker ps --format {{.State}}` would print for it. `exited` is the
    #: default because that is what an orphan usually is by the time a sweep
    #: finds it: nobody was left to `docker wait` on it, and it has been
    #: sitting on a clone ever since.
    state: str = "exited"

    @property
    def labels(self) -> dict[str, str]:
        labels: dict[str, str] = {}
        if self.run_id is not None:
            labels[RUN_LABEL] = self.run_id
        if self.issue is not None:
            labels[ISSUE_LABEL] = str(self.issue)
        return labels


@dataclass
class Daemon:
    """A `Runner` that answers `ps`, `logs` and `rm` against a container table.

    It parses `--filter` for real. A reaper that listed without a label filter
    would get an assertion rather than the human's database containers, which is
    the only version of that mistake anybody wants to make.
    """

    containers: list[Container] = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)
    #: ids whose `docker rm` fails, and the message it fails with
    rm_fails: dict[str, str] = field(default_factory=dict)

    def __call__(
        self, argv: Sequence[str], *, timeout_s: float | None, merge: bool
    ) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        subcommand = argv[1]
        handler = getattr(self, f"_{subcommand}", None)
        if handler is None:
            raise AssertionError(f"the reaper issued an unexpected {subcommand!r}: {list(argv)}")
        stdout, stderr, code = handler(list(argv))
        if merge:
            return subprocess.CompletedProcess(list(argv), code, stdout + stderr, "")
        return subprocess.CompletedProcess(list(argv), code, stdout, stderr)

    # --- subcommands ----------------------------------------------------

    def _ps(self, argv: list[str]) -> tuple[str, str, int]:
        filters = [argv[i + 1] for i, part in enumerate(argv) if part == "--filter"]
        if not any(f.startswith("label=") for f in filters):
            raise AssertionError(
                "a listing with no label filter is a `docker ps -a` over somebody's "
                f"whole machine: {argv}"
            )
        rows = [
            "\t".join(
                [
                    container.id,
                    container.name,
                    container.image,
                    container.labels.get(RUN_LABEL, ""),
                    container.labels.get(ISSUE_LABEL, ""),
                    container.state,
                ]
            )
            for container in self.containers
            if all(matches(container, f) for f in filters)
        ]
        return "\n".join(rows) + ("\n" if rows else ""), "", 0

    def _logs(self, argv: list[str]) -> tuple[str, str, int]:
        found = self.find(argv[-1])
        if found is None:
            return "", f"Error: No such container: {argv[-1]}\n", 1
        return found.logs, "", 0

    def _rm(self, argv: list[str]) -> tuple[str, str, int]:
        container_id = argv[-1]
        if container_id in self.rm_fails:
            return "", self.rm_fails[container_id], 1
        found = self.find(container_id)
        if found is None:
            return "", f"Error: No such container: {container_id}\n", 1
        self.containers.remove(found)
        return container_id + "\n", "", 0

    # --- what the assertions read --------------------------------------

    def find(self, container_id: str) -> Container | None:
        return next((c for c in self.containers if c.id == container_id), None)

    @property
    def ids(self) -> list[str]:
        return [container.id for container in self.containers]

    @property
    def commands(self) -> list[str]:
        return [call[1] for call in self.calls]

    def argvs_for(self, subcommand: str) -> list[list[str]]:
        return [call for call in self.calls if call[1] == subcommand]


def matches(container: Container, spec: str) -> bool:
    if spec.startswith("status="):
        # Honoured for real, so a sweep that narrowed itself to running
        # containers would visibly stop finding the orphans it exists for
        # rather than quietly keep passing against a double that ignored it.
        return container.state == spec.split("=", 1)[1]
    if not spec.startswith("label="):
        return True
    key = spec.split("=", 1)[1]
    if "=" in key:
        name, value = key.split("=", 1)
        return container.labels.get(name) == value
    return key in container.labels
