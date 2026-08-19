"""A `subprocess.Popen` the test scripts, and the wait that goes with it.

`SwarmRuns` supervises the real `swarm run` as a child process, so every test
of the console's start/watch/stop behaviour needs a process it can hold open,
feed lines to, signal and end on command. This is that process.

Here rather than in one test module because two suites drive `SwarmRuns` now:
`test_console_run.py` for the swarm tab, and `test_console_build.py` since #130
made Start building start a run of its own.
"""

from __future__ import annotations

import queue
import threading
import time

__all__ = ["FakeProc", "Script", "settle", "spawner"]


class Script:
    """A stdout whose lines arrive when the test says so."""

    def __init__(self) -> None:
        self._lines: queue.Queue[str] = queue.Queue()

    def feed(self, *lines: str) -> None:
        for line in lines:
            self._lines.put(line + "\n")

    def close(self) -> None:
        self._lines.put("")

    def readline(self) -> str:
        return self._lines.get()


class FakeProc:
    def __init__(self, returncode: int = 0) -> None:
        self.stdout = Script()
        self.returncode = returncode
        self.signals: list[int] = []
        self._done = threading.Event()

    def feed(self, *lines: str) -> None:
        self.stdout.feed(*lines)

    def finish(self, returncode: int | None = None) -> None:
        if returncode is not None:
            self.returncode = returncode
        self.stdout.close()
        self._done.set()

    def wait(self) -> int:
        self._done.wait(timeout=5)
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        self.finish(130)


def spawner(proc: FakeProc):
    """A `spawn` that records the argv and env it was invoked with."""

    def spawn(argv, **kwargs):
        spawn.argv = argv
        spawn.env = kwargs.get("env")
        return proc

    spawn.argv = None
    spawn.env = None
    return spawn


def settle(job, *, state: str | None = None) -> None:
    """Wait for the watcher thread to publish, bounded rather than flaky."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if state is None and job.state != "running":
            return
        if state is not None and job.state == state:
            return
        time.sleep(0.01)
    raise AssertionError(f"run never settled: state={job.state!r}")
