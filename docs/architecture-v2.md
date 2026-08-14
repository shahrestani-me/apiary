# apiary v2 — a GitHub-coordinated, containerized swarm

Status: **design, not yet built.** Tracked by the "apiary v2" epic.

v1 keeps the task ledger in process memory and isolates workers with git
worktrees. It works, but the state dies with the process, only one machine can
see it, and a human cannot intervene mid-run.

v2 moves the control plane onto GitHub and the execution plane into containers.

## The one-line summary

**GitHub is the database. Containers are the sandbox. The orchestrator is a
scheduler that reconciles the two.**

## Shape

```
   prompt
     │
     ▼
┌──────────────────────────────────┐
│  orchestrator (container)        │
│    plan ──▶ provision ──▶ dispatch ──▶ reconcile ──▶ dispose
└───┬──────────────────────────┬───┘
    │ GitHub REST/GraphQL      │ Docker API
    ▼                          ▼
┌────────────────────┐   ┌───────────────────────────┐
│ target repository  │   │ worker container (per task)│
│  issues  (ledger)  │◀──│  clone → branch → edit →   │
│  labels  (protocol)│   │  verify → push → open PR   │
│  PRs     (output)  │   └─────────────┬─────────────┘
└────────────────────┘                 │
                                       │ HTTP
                         ┌─────────────▼──────────────┐
                         │ Ollama on the HOST (Metal) │
                         │  host.docker.internal:11434│
                         └────────────────────────────┘
```

## Three constraints that shape everything

### 1. Ollama cannot run in the container

Docker Desktop on macOS runs a Linux VM with **no Metal passthrough**. An Ollama
inside a container falls back to CPU and loses roughly an order of magnitude of
throughput, which destroys the entire point of the local setup.

So Ollama stays on the host, and every container reaches it over
`host.docker.internal:11434`. Containers isolate **code execution**, not model
inference. Anyone who "fixes" this by adding an Ollama service to the compose
file has made the system dramatically slower.

### 2. Containers do not buy parallelism

`OLLAMA_NUM_PARALLEL=2` is a property of the host server, not of the callers.
Ten worker containers still queue against two inference slots. Containers buy
**isolation, reproducibility and disposability** — a worker can install
dependencies, run arbitrary test suites, and be destroyed without residue.

Worker concurrency should therefore be capped near the inference slot count.
More containers than slots adds queueing, not speed.

### 3. Two things here run untrusted code with credentials

A worker container executes LLM-generated code *and* holds a token that can
push. That is the classic exfiltration shape. Mitigations are not optional
polish:

- Fine-grained token scoped to the **single** target repo, contents+PR write
  only. Never a classic PAT, never org-wide.
- Worker egress restricted to the GitHub API and the host Ollama.
- The orchestrator's Docker socket access is effectively host root; put a
  socket proxy in front of it with a restricted API surface.

## Control plane: GitHub

### Issues are the task ledger

An issue is a task. Its body carries the contract; its labels carry the state.
The orchestrator's in-process state holds only run-scoped ephemera — which
container is running what, right now. **On any disagreement, GitHub wins.**

This is what makes runs resumable across machines and sessions, and what lets a
human retitle, re-scope, close or reprioritize a task mid-run and have the
swarm respect it.

Issue body schema (parsed, so it is a contract):

```markdown
## Goal
One sentence: what must be true when this is done.

## Files
- src/thing.py
- tests/test_thing.py

## Verify
python -m pytest -q tests/test_thing.py

## Blocked by
- #12
```

The **Files** section is load-bearing. Non-overlapping file sets across
concurrently-runnable issues is what keeps merges sane; it is the v1 rule that
survives unchanged.

### Labels are the protocol

| Label | Meaning | Set by |
|---|---|---|
| `swarm:ready` | dependencies met, may be dispatched | orchestrator |
| `swarm:blocked` | waiting on another issue | orchestrator |
| `swarm:claimed` | a worker container holds it now | orchestrator |
| `swarm:review` | PR open, awaiting checks/review | worker |
| `swarm:done` | PR merged | orchestrator |
| `swarm:failed` | attempts exhausted, needs a human | orchestrator |
| `swarm:attempt/1..3` | retry counter | orchestrator |

Plus routing labels the planner assigns: `area/*`, `size/*`.

### The orchestrator is the sole dispatcher

Workers never choose their own issue. This is a deliberate simplification:
GitHub offers no atomic compare-and-swap on labels, so self-selecting workers
would race for the same issue and need a distributed lock. Central dispatch
removes the problem instead of solving it.

### PRs are the integration mechanism

v1 merged branches directly. v2 opens a PR per issue, linked with `Closes #N`,
and lets CI plus branch protection be the gate. The verifier's rule is
unchanged and now doubled: the worker runs the verify command locally, and CI
runs it again on neutral ground.

## Execution plane: containers

### Worker lifecycle

```
create (labeled apiary.run=<id>, apiary.issue=<n>)
  → clone repo at base commit, checkout swarm/issue-<n>
  → read issue contract
  → edit loop against host Ollama
  → run Verify command
  → commit, push, open PR, label swarm:review
  → exit(0 = PR open, 1 = failed, 2 = infrastructure error)
orchestrator observes exit code → disposes container → updates labels
```

Disposal is unconditional: containers are cattle. Logs are captured to the run
artifact directory *before* removal, because a destroyed container's logs are
the main thing you want when diagnosing a bad run.

An **orphan reaper** sweeps containers matching `apiary.run=<id>` at startup and
shutdown. Crashed orchestrators otherwise leak containers that hold clones and
disk.

### Resource limits

Every worker gets explicit `--cpus`, `--memory`, and `--pids-limit`. An LLM that
writes a fork bomb or an infinite loop is not a hypothetical, and the failure
mode without limits is "the Mac becomes unusable" rather than "one task fails".

## Orchestration loop

Each cycle reconciles desired state (issues) with actual state (containers, PRs):

1. **Read** all `swarm:*` issues and open PRs from GitHub.
2. **Compute readiness** — an issue is ready when every `Blocked by` reference
   is closed. Relabel `blocked`/`ready` accordingly.
3. **Dispatch** ready issues up to the concurrency cap, one container each.
4. **Observe** finished containers and PR check status. Merged → `done`. Failed
   checks → bump `swarm:attempt/N`, reopen for retry, or `failed` at the cap.
   Green PRs are merged by the orchestrator, not by a human: the `## Verify`
   command and CI are the gate, and `APIARY_MERGE_ADMIN_OVERRIDE=0` is how a
   repository asks for a person to press the button instead.
5. **Judge** progress. Same progress ledger as v1: satisfied? progressing?
   looping? Stall triggers a replan, which rewrites issues rather than
   in-memory tasks.
6. **Close the loop.** When no issue is left in a non-terminal state the plan is
   finished — which is not the same as the objective being met. The orchestrator
   asks whether the objective the run was given is actually delivered by the
   work that landed, and if it is not, plans *additional* issues and keeps
   going. Bounded at two follow-up rounds, refused outright when a task was
   abandoned (`swarm:failed` means a human is needed) or when the model that
   would answer is unreachable.

The distinction in step 6 is the difference between a swarm that stops when its
first decomposition runs out and one that stops when the objective is met or it
has run out of ways to reach it. A plan is one model's first reading of an
objective, written before any of the code existed; running it to completion
proves the plan is finished and nothing more. Follow-up rounds only ever *add*
issues — nothing that already merged is rewritten or retired.

Because every input to that loop comes from GitHub, the orchestrator is
restartable at any point and holds no irreplaceable state.

## Two modes

**Existing repo.** Point it at a repo, give an objective, it plans issues
against the code that is there.

**Greenfield.** Given only a prompt, the orchestrator creates the repository,
applies license/CI/ruleset/labels, and plans issues against it — the first of
which **is the project**. This is the "prompt in, project out" path.

The initial commit carries a README, a LICENSE and a workflow whose gate is
`test -f README.md`, which is the only command that passes on an empty
repository. The bootstrap issue (#101) then generates the project in a worker
container like any other task, and its pull request replaces both the files and
the gate — the latter only after the proposed command has been observed to fail
on deliberately broken code (#102).

### Which stacks, and who decides

**The host decides, not the prompt.** #99 gives each stack a worker image, and
`swarm run --new` accepts any stack an image exists for and refuses the rest by
naming the missing image and the `docker build` line.

This replaced a 26-word deny list whose three justifications were all measured
and found wrong:

| claim | measured |
|---|---|
| "a dependency install costs four minutes on every attempt forever" | cold Vite React+TS `npm install` **19s**, Expo **59s**, warm cache **6s**, under the real `--cpus 2 --memory 4g` limits |
| "the `## Verify` command installs what the repo needs at run time" | it cannot — a worker's only route out is the egress proxy's static allowlist, so an install is *denied* in under a second |
| "the registry is the seam for a second entry" | decorative: the resolver returned Python unconditionally |

The no-toolchain rule's stated purpose was stack-agnosticism. Baking in *none*
narrowed the swarm to Python instead, because the one toolchain every image
carried was the Python the package itself needs. Agnosticism is bought by
several images selected per task.

**A green gate is not a rendered UI.** A React project whose suite passes can
render a blank page. The exit-code invariant is the only authority this system
has, and it cannot see pixels — so "the swarm can build a React app" means its
tests pass, not that anyone has looked at it. Anything stronger needs a human
or a different kind of check, and neither is in v2.

## What v1 code survives

| v1 module | Fate in v2 |
|---|---|
| `state.py` schemas | Kept; issue bodies serialize to the same shapes |
| `graph.py` | Kept; nodes swap their backing store |
| `nodes/planner.py` | Kept, now writes issues |
| `nodes/judge.py` | Kept; observes the ledger, and the reconciler calls it |
| `nodes/verifier.py` | Moves into the worker container |
| `nodes/integrator.py` | Replaced by PR + CI + branch protection |
| `worktree.py` | Demoted; containers clone rather than share a checkout |
| `llm.py`, `config.py` | Kept, plus container-aware defaults |

## Deliberately out of scope for v2

- Hosted/remote workers. Everything stays on one machine.
- Multi-repo tasks. One task, one repo.
- Workers reviewing each other's PRs. The verifier and CI are the gate; adding
  LLM review reintroduces exactly the "model judges correctness" failure this
  project exists to avoid.
- Judging whether a generated UI *looks* right. See "a green gate is not a
  rendered UI" above.
- Polyglot repositories. One repo, one stack.
