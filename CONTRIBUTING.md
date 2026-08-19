# Contributing

Contributions are welcome. This is a research project, so an issue describing
what you observed is often worth as much as a patch.

## Ground rules for `main`

`main` is protected and cannot be pushed to directly. It also cannot be deleted
or force-pushed. Every change goes through a pull request that:

- passes CI (`pytest -q -s` and `mypy`), and
- is approved by the code owner ([@kamyarshahrestani](https://github.com/kamyarshahrestani)).

Approvals are dismissed when new commits are pushed, so a re-review is required
after changes.

## Before you open a PR

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q -s
mypy
```

The test suite stubs the model, so it needs no Ollama and runs anywhere. If it
passes locally it will pass in CI.

`mypy` takes no arguments: its target, its strictness and its excluded backlog
are all in `pyproject.toml` under `[tool.mypy]`. CI runs the same bare command.

The check exists because a green suite is not evidence that a typed refactor is
correct — #142 retyped the internal model across 24 files, passed 1480 tests and
had a dead judgement path, because a dict read with the wrong key type returns
rather than raises (#168).

**If it reports something you did not write**, say so in the pull request rather
than widening a `disable_error_code` list to get green: those lists are
per-module, so adding a code there switches it off for every line in the file,
including the one that would have caught the next silent mismatch. The backlog
that predates the check is excluded at its own line inside `orchestrator/`,
`github/` and `nodes/`, and module by module elsewhere; both are meant to
shrink.

## What makes a change easy to accept

- **Keep the verifier honest.** Nothing in this repo may let an LLM decide
  whether code is correct. The exit code of `$SWARM_VERIFY` is the only
  authority, and that is deliberate.
- **Keep the orchestrator boring.** Planning, routing and stall detection should
  stay arithmetic and explicit state. Model intelligence belongs in the workers.
- **Preserve worktree isolation.** Workers must never share a checkout.
- **Report measurements, not impressions.** Throughput, memory and swap numbers
  in the docs were measured on real hardware; if you change one, say what you
  measured it on.

## Reporting model quality problems

Run `pytest -q -s` first. If it passes, the plumbing is fine and what you are
seeing is model quality — which is useful to report, but please include the
model, the objective, and the contents of `$SWARM_VERIFY`.
