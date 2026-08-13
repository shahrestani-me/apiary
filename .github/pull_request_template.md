## What and why

<!-- What changes, and what problem it solves. Link an issue if there is one. -->

## Verification

<!--
How you know it works. `pytest -q -s` is the floor, not the ceiling.
If you changed anything with a performance or memory claim attached, say what
hardware and model you measured on — the numbers in the docs are measured, not
estimated, and should stay that way.
-->

- [ ] `pytest -q -s` passes locally

## Invariants

<!-- Tick these, or explain in the PR why the change needs to break one. -->

- [ ] No LLM decides whether code is correct — `$SWARM_VERIFY`'s exit code is still the only authority
- [ ] The orchestrator stays arithmetic and explicit state; model intelligence stays in the workers
- [ ] Workers still never share a checkout
