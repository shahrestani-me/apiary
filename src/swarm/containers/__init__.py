"""The execution plane: worker containers, and their whole life.

One module, one door - `manager.ContainerManager`. Nothing else in the swarm
talks to Docker, for the same reason nothing but `swarm.github.client` talks to
GitHub: a second caller is a second place that forgets to label a container,
redact a token or capture logs before removing the thing that held them.

`reaper.py` (#20) joins this package and reaches the daemon through the same
`DockerCLI`.
"""

from __future__ import annotations

from .manager import (
    DEFAULT_LIMITS,
    ISSUE_LABEL,
    MAX_LOG_CHARS,
    PLACEHOLDER,
    RUNNING_STATE,
    WORKER_IMAGE,
    ContainerError,
    ContainerManager,
    ContainerTimeout,
    DockerCLI,
    DockerError,
    Handle,
    Limits,
    Redactor,
    Runner,
    capture_logs,
    dispose_container,
    find_containers,
    is_missing,
    subprocess_runner,
)

__all__ = [
    "DEFAULT_LIMITS",
    "ISSUE_LABEL",
    "MAX_LOG_CHARS",
    "PLACEHOLDER",
    "RUNNING_STATE",
    "WORKER_IMAGE",
    "ContainerError",
    "ContainerManager",
    "ContainerTimeout",
    "DockerCLI",
    "DockerError",
    "Handle",
    "Limits",
    "Redactor",
    "Runner",
    "capture_logs",
    "subprocess_runner",
    "dispose_container",
    "find_containers",
    "is_missing",
]
