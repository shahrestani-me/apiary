# Demo run — greenfield, 2026-08-14

Status: **attempted, did not complete.** The run created a repository and then
stopped, before planning, on a credential problem that no amount of retrying
fixes. Nothing was merged, because no worker ever started.

This document is the record #27 asks for. It is not the record #27 wanted. The
ticket says a demo that only shows the happy path teaches nothing, and the
inverse held: the attempt taught more by dying where it did than a green run
would have, because **`swarm run --new` cannot complete with a correctly-scoped
work key.** That is a property of the design, not of this host or this night.

The demo repository created by this run was deleted afterwards, at the
operator's instruction. Its URL is recorded below for the record; it will 404.

---

## 1. What was attempted

| | |
|---|---|
| Objective | a shopping-cards app: add things you need; per item a link, name, description, and a reminder to buy it; monorepo with backend and frontend |
| Repository | `shahrestani-me/shopping-card`, private (since deleted) |
| Command | `swarm run --new @brief.txt --owner shahrestani-me --name shopping-card --yes` |
| Started | 2026-08-14, ~21:49 UTC |
| Stopped | 2026-08-14, ~21:50 UTC, exit 1 |

`--yes` was passed because the run was driven from a non-interactive shell, and
[`provision`][provision]'s confirmation prompt would have blocked forever. That
prompt is the only interlock on an action the module's own docstring says it
"cannot undo", so skipping it is worth naming: the operator answered it out of
band, in conversation, before the command was issued.

[provision]: ../src/swarm/greenfield/provision.py

## 2. Environment

Everything here was verified by `swarm doctor` immediately before the run:
**15 checks, 12 ok, 1 failed, 2 not attempted.** The one failure was
`github.repo` — "no target repository" — which is the expected state before a
`--new` run, since the repository is what the run is about to create.

| | |
|---|---|
| Host | macOS (darwin 25.5.0), Docker Desktop |
| Ollama | 0.32.9 at `http://localhost:11434`, on the host (see architecture-v2, constraint 1) |
| Orchestrator model | `gemma4:31b` |
| Worker model | `gemma4:26b` |
| Docker daemon | 29.2.1 |
| Worker images | `apiary-worker` (`sha256:65f07a48`), `apiary-worker-node` (`sha256:85215fa6`) |
| Timeouts | verify 300s, inside-worker 1200s |
| Credentials | two distinct fine-grained PATs, `GITHUB_TOKEN` and `APIARY_PROVISION_TOKEN` |

Both token checks passed: `github.token` reported "a fine-grained token is
set", `github.boot-token` reported "a fine-grained boot key is set, distinct
from the work key". **Both were true, and the run still could not proceed.**
Section 4 is about why those two checks are not sufficient.

## 3. What happened, in order

1. **A model chose the stack.** [`Bootstrap.for_prompt`][bootstrap] calls
   `choose_stack`, which asked `gemma4:31b` what the prompt implies. It answered
   **react**. The brief asked for a monorepo containing both a backend and a
   frontend; `stack` is a single-valued field, so the answer silently kept the
   frontend half. Nothing in the run reports this as a narrowing.
2. **The repository was created.** `created_at` 21:50:02Z. Private, as the
   default requires.
3. **The initial commit was pushed.** `main @ e162e4e`, `pushed_at` 21:50:08Z —
   README.md, LICENSE, `.github/workflows/ci.yml`, and nothing else.
4. **The six `swarm:*` labels were created**: ready, blocked, claimed, review,
   done, failed. Zero already present.
5. **`main` was protected** with `deletion`, `non_fast_forward` and
   `required_status_checks`.
6. **The run exited 1**, printing no error, having planned nothing. The
   repository ended with **0 issues**.

Provisioning itself took **about six seconds** end to end. That is the only
wall-clock figure this attempt produced, and it is not an interesting one.

[bootstrap]: ../src/swarm/greenfield/bootstrap.py

## 4. Where it stopped, and why

After `_target` returns, `_run` calls `start_run`, whose first act is to read
the ledger — with the **work** key. That read 404s.

Four requests, one repository, two tokens, taken by hand after the failure:

| | `GET /repos/…/shopping-card` | `GET …/issues` | `GET /user` |
|---|---|---|---|
| work key (`GITHUB_TOKEN`) | **404** | **404** | 200 |
| boot key (`APIARY_PROVISION_TOKEN`) | 200 | 200 | 200 |

The work key was valid, live, and correctly shaped. It simply could not see the
repository.

### Why this is structural

`docs/security.md` §1 requires the work key to be a fine-grained PAT with
**"Only select repositories" naming exactly one**. A fine-grained PAT's
repository selection is fixed **when the token is minted**. The repository this
run targets does not exist until step 2 above.

So the two requirements cannot both hold at once:

- the work key must name the target repository, and
- the target repository is created by the run, after the key exists.

`swarm run --new` therefore **cannot** proceed past provisioning with a work key
scoped the way the security model requires. Every greenfield run stops here on
its first execution. The only ways through are to widen the work key to all of
the owner's repositories — which is the account-wide reach
[`security.py`][security] exists to prevent, differing from a classic PAT in
paperwork more than in blast radius — or to have a human edit the token's
repository selection between provisioning and dispatch.

**This run took the second path.** The operator opened the token settings and
added `shahrestani-me/shopping-card` to the work key's selected repositories.
The work key then answered 200 on both endpoints. That is one human
intervention, in a browser, in the middle of what the epic describes as one
prompt and one command — and it is unavoidable as the design stands.

[security]: ../src/swarm/security.py

## 5. What this run did not measure

The ticket asks for wall clock, tokens, model-swap time, first-time
verification rate, how many PRs needed a rebase, and where it needed a human.
Only the last is answered above. **The rest were not measured, because no
worker container ever started.** They are listed here so nobody reads their
absence as a zero:

- tasks passing verification first time — **not measured**, nothing was planned
- PRs needing a rebase before merge — **not measured**, no PR was opened
- tokens consumed — **not measured**; the only inference was the stack choice
- model-swap time — **not measured**
- merges via the admin override — **none occurred**

## 6. Three findings that outlive this run

**The failure is invisible to the tooling built to describe it.** Run artifacts
are created by `start_run`, and this run died on `start_run`'s first read. So
`.swarm/runs` holds no entry, and `swarm runs` and `swarm show` — the output
#97 added precisely so a write-up like this one would have raw material — can
say nothing about it. Every failure earlier than that line is unreportable by
the reporting tool.

**It exited 1 while printing nothing.** [`cli.py`][cli]'s shared handler prints
`! {exc}` and returns 1. The log's last two lines are blank, so the exception
carried an empty message. A run that stops with no stated reason is
indistinguishable, to its operator, from one that finished.

**The gate the whole design rests on was a placeholder, and it stayed one.**
Generated CI ran:

```yaml
- name: verify
  run: |
    test -f README.md
```

`README.md` is in the initial commit. That check passes on every commit that
could ever be pushed, including one that deletes the entire application.
[`_target`][cli] explains why it has to start this way — the required status
check must report on the first commit, before any code exists — but nothing in
the run ever replaces it. Combined with the admin override (#23), a merged PR
in a greenfield repository would have carried a green check certifying that a
file nobody touched still existed. **The green checkmark, in a greenfield run,
is currently worth nothing at all**, and any future demo that reports a merge
count without saying so is reporting a number it has not earned.

[cli]: ../src/swarm/cli.py

## 7. Reproducing this

Two fine-grained PATs are required, and they must be distinct — `swarm doctor`
fails the run if they are equal.

```bash
export GITHUB_TOKEN=github_pat_...            # contents:write, pull_requests:write, issues:write, metadata:read
export APIARY_PROVISION_TOKEN=github_pat_...  # administration:write, contents:write, workflows:write, issues:write, metadata:read

docker build -f Dockerfile.worker      -t apiary-worker      .
docker build -f Dockerfile.worker.node -t apiary-worker-node .

swarm doctor                    # expect 12 ok, github.repo failing, 2 not attempted
swarm run --new "<brief>" --owner <account> --name <name> --yes
```

Expect it to stop exactly as described in section 4. To continue past it:

1. Open the **work** token at <https://github.com/settings/personal-access-tokens>
2. Add the newly created repository to its selected repositories, and save
3. Resume against the now-existing repository — `--new` has already done its job:

```bash
swarm run --repo <owner>/<name> --objective "<brief>"
```

Pass `--verify` deliberately when resuming. The default for a `--repo` run is
`python -m pytest -q`, which on a repository the planner typed as `react` fails
in every worker for a reason that has nothing to do with the model's output.

## 8. Honest summary

One prompt and one command produced a real, correctly-configured, protected,
labelled repository in about six seconds, and then stopped dead. The stop was
not bad luck; it is where every greenfield run stops, and getting past it needs
a human in a browser editing a credential.

Nothing here says the swarm cannot build software — this attempt never got far
enough to ask that question. What it establishes is narrower and more useful:
**the greenfield path has never been run end to end, and could not have been**,
because the credential the design mandates cannot reference a repository that
does not yet exist. Until that is resolved, #27's own acceptance criterion —
one prompt, one command, a real repository with issues, PRs and merges — is not
reachable.
