"""Shared test fixtures.

The shared doubles live here, and every test that needs one should import it
rather than grow its own:

- `fixtures.github` - a fake `Transport` for `swarm.github.client`: canned
  responses, a record of every request, and constructors for the failures that
  are impossible to provoke against the real API (5xx, a secondary rate limit,
  a 404).
- `fixtures.repo` - a scratch git repository in a temp directory, with seeded
  commits, a working `pytest` target and a bare repo standing in for `origin`,
  so clone/branch/push paths are exercisable with no network.
- `fixtures.docker` - a Docker daemon answering `ps`, `logs` and `rm` from a
  container table, parsing `--filter` for real, so container selection and
  disposal are testable with no daemon.
- `fixtures.procs` - a `subprocess.Popen` whose output and exit the test
  scripts, for the console's supervision of a real `swarm run` child.

`tests/conftest.py` exposes the first two as pytest fixtures (`fake_github`,
`scratch_repo`) and registers the `docker`, `network` and `ollama` markers that
gate everything these doubles exist to avoid needing.

The modules import only from the standard library and from `swarm`, so they can
be imported directly (`from fixtures.github import response`) as well as
requested as fixtures.
"""

from __future__ import annotations
