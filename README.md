# apiary

A local multi-agent coding swarm for Apple Silicon. Orchestration on
[LangGraph](https://github.com/langchain-ai/langgraph), models on
[Ollama](https://ollama.com), isolation via git worktrees. **Nothing leaves the
machine.**

```
setup ─→ plan ─→ dispatch ═fan out═▶ worker(worktree-1) ─┐
           ▲         ▲               worker(worktree-2) ─┼─→ verify ─→ judge ─┐
           │         └──── retry ────────────────────────┘                    │
           └──────── replan (on stall) ───────────────────────────────────────┤
                                                                              │
                                                     integrate (merge) ◀──────┘
```

You give it an objective. It decomposes the objective into non-overlapping
tasks, runs each task in its own git worktree on its own branch, verifies each
one by **running your real test command and believing only the exit code**, and
merges back what passed.

## Why this shape

Most multi-agent coding systems fail in the same two places, so both are
designed out rather than tuned around:

- **Agents editing one checkout corrupt each other.** Every worker gets
  `git worktree add -b swarm/<task-id>` — a private filesystem view on its own
  branch. Integration is then an ordinary merge you can inspect and revert.
- **Agents that judge their own work are unreliable.** No LLM decides whether
  code is correct here. The verifier runs `$SWARM_VERIFY` in the worktree and
  reads the exit code. Nothing else counts as done.

The orchestrator does arithmetic and routing; it does not "reason" about
coordination. All model intelligence lives in the workers. The task ledger and
progress ledger are explicit typed state (the
[Magentic-One](https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/)
pattern), which is what makes the orchestrator's decisions inspectable,
checkpointable and testable.

State is checkpointed to sqlite after every node, so a crashed or interrupted
run resumes with `swarm --resume <thread-id>` instead of restarting from zero.

## Quick start

```bash
git clone https://github.com/shahrestani-me/apiary.git ~/sources/apiary
cd ~/sources/apiary
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q -s
```

That test builds a throwaway git repo, plans a task, creates a worktree,
applies an edit, runs **real pytest** inside the worktree, judges progress and
merges the branch back — all with a stubbed model, so it needs no Ollama.

If it passes, every remaining problem you hit is **model quality, not
plumbing.** That distinction is worth days.

Then point it at a real repo:

```bash
export SWARM_REPO=~/sources/your-repo
export SWARM_VERIFY="python -m pytest -q"   # must match YOUR repo's test command
swarm run --repo owner/name --objective @brief.md
```

`@path` reads the objective from a file, and for anything real that is what you
want: the planner decomposes this text into tasks, and a phrase gives it
nothing to decompose. Both `--objective` and `--new` accept it.

Full install, model choice and memory budgeting: **[SETUP.md](SETUP.md)**.
Credentials and what currently runs end to end: **[Running v2 locally](#running-v2-locally)**.

## Requirements

- Apple Silicon Mac (this is tuned for unified memory; it will run elsewhere,
  the memory arithmetic just won't apply)
- Ollama with at least one instruct model that supports schema-forced JSON
- Python 3.11+
- A target repo that is a **clean git repo with a working test command**. No
  tests means no verification, and no verification means the swarm produces
  plausible garbage confidently.

## Configuration

Everything is environment variables — see [`src/swarm/config.py`](src/swarm/config.py).

| Variable | Default | Notes |
|---|---|---|
| `SWARM_REPO` | `$PWD` | Target git repo. |
| `SWARM_VERIFY` | `python -m pytest -q` | Your quality gate, and the `## Verify` every planned issue carries. `swarm run --verify` overrides it, and `--new` takes the command its scaffold committed instead. |
| `SWARM_ORCHESTRATOR_MODEL` | `gemma4:31b` | Planning, routing, stall judgement. Dense — buy quality. |
| `SWARM_WORKER_MODEL` | `gemma4:26b` | Writes the code. MoE — buy throughput. |
| `SWARM_MAX_PARALLEL` | `2` | Concurrent workers. Memory scales with this. |
| `SWARM_MAX_ROUNDS` | `8` | Hard stop. |
| `SWARM_MAX_STALLS` | `2` | No-progress rounds before **replanning** instead of retrying. |
| `SWARM_MAX_ATTEMPTS` | `3` | Per-task retries before abandoning. |
| `APIARY_MERGE_ADMIN_OVERRIDE` | `1` | Whether the swarm merges its own green PRs. `0` leaves them for a human, and the issues sit in `swarm:review` until one presses the button. |
| `APIARY_MERGE_METHOD` | `squash` | How it merges. Must be a method the target repo allows. |
| `APIARY_MAX_UPDATE_ROUNDS` | `3` | How many times one PR may be dragged forward onto a moving base before the starvation is called and a human is asked. |
| `APIARY_MERGES_PER_CYCLE` | `1` | Merges are serialised under a strict status-check policy, because each one invalidates its siblings. |
| `APIARY_MAX_INFRASTRUCTURE` | `3` | Consecutive **infrastructure** failures before a task reaches a human. Exit 2 does not consume an attempt, so without a ceiling a missing image or a denied registry retries for free forever. Separate from `SWARM_MAX_ATTEMPTS`: these are the failures that never cost an attempt. |
| `SWARM_WORKER_CTX` | `16384` | Never set this to a model's advertised 256K — the KV cache would cost more than the weights. Passed through to worker containers, so an export on the host tunes the workers too. |
| `SWARM_WORKER_TIMEOUT` | `1200` | Wall clock for a whole worker container: clone, inference, verify, commit, push, PR. |
| `SWARM_VERIFY_TIMEOUT` | `300` | Wall clock for `SWARM_VERIFY` alone, **inside** the above. Raising this without raising the outer one buys nothing — the container dies at the outer cap first, and the attempt is recorded against the container rather than the gate. `swarm doctor` refuses an inverted pair and `swarm run` will not start on one. |
| `APIARY_STATE_SOURCE` | `derived` | Which control plane the orchestrator **believes** about a task's state (#147). `derived` reads it from the code host, the containers, the run artifacts and apiary's own store ([ADR 0001](docs/adr/0001-task-systems-are-integrations.md), [ADR 0002](docs/adr/0002-apiary-owns-a-thin-task-store.md)); the `swarm:*` labels are still written and still compared, so anything watching the repository sees no change. `labels` restores the pre-#147 behaviour completely and is the way back from a bad cutover — apiary develops itself on this control plane, so a cutover with no escape hatch would block its own repair. A task whose label a human edits mid-run does not change what the swarm does and the disagreement is emitted as `state.override`, which `swarm show` counts. Loud on anything else: a typo here must stop the run rather than silently leave the new behaviour on. |
| `APIARY_DERIVED_SHADOW` | `1` | Run the derived-state resolver beside the `swarm:*` labels every cycle and record where the two disagree (#146). Since #147 the labels are the side being checked, and a divergence here says the label has drifted from what the orchestrator acted on rather than the reverse. It adds no API call, and it also records `observed.jsonl` into the run directory, which is what makes a real run replayable as a test corpus. `swarm show` reports the divergence count, the coverage behind it, and how much of that coverage was independent of the cycle's own writes. Off on a dry run regardless. |
| `APIARY_CAPTURE` | unset (off) | Record every model call — the prompt, the raw response, Ollama's own load/total durations, and the real exception when one ends it. Off by default because a prompt carries whole file bodies from the repo under test. `swarm run` prints one line when it is on. `swarm console` turns it on for itself. |
| `APIARY_CAPTURE_MAX_CHARS` | `8192` | Per-field truncation for captures written during a run. The SHA-256 always covers the full text, so a truncated record still answers "did the prompt change between attempt 1 and attempt 3?". Console captures are never truncated. |
| `APIARY_CONSOLE_DIR` | `.swarm/console` | Where `swarm console` writes its captures. A sibling of the artifacts root with its own variable — moving runs does not silently move these. |
| `APIARY_STORE_DIR` | `.swarm/store` | Where apiary keeps its own judgments about its own execution — one SQLite file per project holding each task's failure signature, its streak and how often its retry budget was renewed ([ADR 0002](docs/adr/0002-apiary-owns-a-thin-task-store.md)). Per *project*, not per run: a store that started empty each run would hand every task a fresh retry budget every time. A store that is corrupt or belongs to another project stops the run with a named fix rather than being recreated, because an empty store reads as "nothing has ever failed". |

## Seeing what the model was asked

```bash
swarm console
```

One page on `http://127.0.0.1:8117`. Type the human turn of a prompt, fire it at
the same local Ollama a run uses, and read the schema-constrained answer, the raw
response, the timings, and — when it breaks — the real exception beside the prompt
that caused it. The model-call tabs need no GitHub token, no Docker, no repository
and no nine-minute run.

The page is more than that now. The swarm tab runs the real thing — and since
#130 the planner tab's **Start building** provisions a repository from a plan you
have read, writes it up as issues, and starts the swarm on it, with a live log and
a Stop that disposes the run's containers. Those parts do need a token and a
Docker daemon, and they create things on GitHub.

Two call sites are exposed: `propose_edits`, the worker's whole-file generation,
and `choose_stack`, which answers in seconds and is the cheapest way to prove the
wiring. Prompts come from the same `prompt_for` production calls, so what you read
is what a run would send.

Loopback only, and it refuses to bind anything else: a capture holds whole files
from the repository under test, and a worker container can reach any port on the
host gateway.

## Running v2 locally

v2 puts the ledger on GitHub, so it needs a token. This section is the whole
credential story, and an honest account of which parts of the loop are wired.

### The token: fine-grained, one repository

Mint a **fine-grained** PAT at
<https://github.com/settings/personal-access-tokens/new>. Classic (`ghp_`) and
OAuth (`gho_`) tokens are refused by prefix, including the one `gh auth token`
prints — their scope is a *verb* like `repo`, so they reach every repository the
account can reach. A fine-grained token is scoped to repositories, which is the
whole point.

- **Resource owner:** your account
- **Repository access:** *Only select repositories* → the one target repo
- **Permissions:** Contents `write`, Pull requests `write`, Issues `write`,
  Metadata `read`. Nothing else.

Leave **Workflows, Actions, Administration, Secrets, Environments, Packages and
Members off.** `security.py` names them rather than merely omitting them, and
`workflows` is the sharp one: with it, generated code can rewrite
`.github/workflows/*`, and CI is the neutral ground that independently re-runs
the verify command. A worker that can edit CI can edit its own grader.

### Creating projects needs a second token

`swarm run --new "a trip planner"` creates the repository itself — private by
default — seeds it with a CI workflow and a passing test, and then plans issues
into it. Creating a repo needs `administration` and pushing a workflow needs
`workflows`, and those are exactly the two permissions the work key must never
have: a worker holding `workflows` can rewrite the CI that independently
re-runs its own verify command.

So they are two credentials, not one widened credential:

| | Work key (`GITHUB_TOKEN`) | Boot key (`APIARY_PROVISION_TOKEN`) |
|---|---|---|
| Job | The whole run | Create the repository, once |
| Permissions | contents, pull_requests, issues, actions (read), metadata | administration, contents, workflows, issues, metadata |
| Lives | In every worker container | In the orchestrator, for seconds |
| Sees model output | Yes | Never — it runs before any container exists |

`ContainerManager` refuses to start a container whose environment carries the
boot key, by name *or* by value, so the separation is enforced rather than
documented. `swarm doctor` reports whether a boot key is present and fails if
it is the same token as the work key.

If you only ever run against repositories that already exist, skip the boot key
entirely — nothing else needs it.

**One caveat:** branch protection on *private* repositories needs a paid GitHub
plan. A generated private repo may end up with CI but no enforced protection,
and protection plus CI is what makes the merge gate mean anything. `provision`
detects this and says so.

### Where to put it

**Outside the repo.** Nothing in the Python loads `.env` — there is no
`python-dotenv` dependency and no `load_dotenv` call — so a `.env` file is
silently ignored by `swarm` and `swarm doctor`, and you get "token is not set"
while looking at a file containing the token. `.env` is read by **`docker
compose` only**.

```bash
mkdir -p ~/.config/apiary && chmod 700 ~/.config/apiary
cat > ~/.config/apiary/env <<'EOF'
GITHUB_TOKEN=github_pat_...
APIARY_MERGE_ADMIN_OVERRIDE=0
SWARM_MAX_PARALLEL=1
EOF
chmod 600 ~/.config/apiary/env

set -a; source ~/.config/apiary/env; set +a
```

That serves both paths: the CLI reads the exported variables, and compose
inherits them from the shell without needing a `.env` at all.

The two extra settings are deliberate for a first run. `APIARY_MERGE_ADMIN_OVERRIDE=0`
means **nothing merges itself** — you review every PR the swarm opens.
`SWARM_MAX_PARALLEL=1` matches what the dispatcher computes on a 36 GB machine
anyway (see below).

### Check the machine before involving any model

```bash
swarm doctor owner/name
```

(`python -m swarm.doctor owner/name` is the same command and still works.)

Thirteen checks, each naming the command that fixes it: Ollama's client target
and reachability, both models present, **schema-forced JSON actually honoured**,
token shape and scope, boot key distinct from the work key, repo readable,
`swarm:*` labels present, CI configured, the two timeouts consistent with each
other, Docker CLI and daemon, worker image built. Every one of these fails in a way
that looks like something else — an absent model looks like a planning bug, a
token missing a scope looks like a permissions bug three modules away.

`doctor` writes nothing, ever. Missing labels are reported, not created; the fix
it prints is `python -m swarm.github.labels owner/name`.

### Run the orchestrator on the host, not in the container

The image builds and `docker compose run --rm orchestrator --help` works, but
the orchestrator image ships `git` and **no `docker` binary**, so `DOCKER_HOST`
is honoured by nothing and it cannot spawn workers from inside. Running `swarm`
from the venv uses the host's own Docker and Ollama and sidesteps this
entirely. The container is for reproducibility, not a requirement.

### What actually runs today

Run identity, ledger attach and readiness work:

```bash
swarm run --repo owner/name --objective "..." --dry-run
```

`--dry-run` reads the ledger and writes nothing to GitHub. Without it, unmarked
issues carrying a `swarm:*` state label are **adopted** — their bodies rewritten
to add an identity marker — so try a scratch repo before a real one.

The loop is wired end to end: `swarm run` plans issues, dispatches containers,
reads CI, merges what passes, dispatches whatever that unblocks, and stops when
the objective is met.

**It merges its own pull requests.** A `swarm:review` issue whose PR is green
and mergeable is merged by the orchestrator, and the issue it closes unblocks
its dependants on the next cycle. That is deliberate and it is also the whole
quality argument: `## Verify` and CI are the only gate, so the gate had better
be real. `APIARY_MERGE_ADMIN_OVERRIDE=0` turns it off and leaves the review
queue to a person; `swarm run --no-merge` does the same for one run.

**It does not stop when the plan runs out.** When nothing is left in a
non-terminal state, the run asks whether the objective was actually met — the
plan being finished is a fact about the planner's first guess, not about the
objective — and plans follow-up issues if it was not, for at most two rounds.
`--no-goal-check` stops at plan exhaustion instead. The run exits non-zero when
it stops short, and prints what is still missing.

One piece is still not reachable: the CI failure that `checks.write_feedback`
persists onto an issue is not yet read back into the next attempt's prompt
(`worker/entrypoint.py`), so a retry sees the same context the first attempt
did. It is on the issue for a human either way.

### Reading a run afterwards

Every run writes a directory under `.swarm/runs/<run-id>/` — what was planned,
what each container printed, what each worker's gate said, and where the run
needed a person. Two commands read it back:

```bash
swarm runs                  # every recorded run, newest first, grouped by repo
swarm show <run-id>         # one run: outcome, cost, and what needed a human
```

Both are read-only and neither needs a token — the directory is the whole
input. `--root` points them at an artifacts directory other than the default
(`$APIARY_ARTIFACTS`, else `.swarm/runs`).

### Concurrency is bounded by inference, not memory

The dispatcher derives its cap from `OLLAMA_NUM_PARALLEL` **minus one slot for
the orchestrator's own planner and judge calls** — with
`OLLAMA_MAX_LOADED_MODELS=1` those are model swaps, not queued requests. On a
36 GB M4 Max that arithmetic yields **one worker**. The memory bound is looser
than it looks for a second reason: Docker Desktop's Linux VM gets ~7.6 GB, not
the host's 36 GB, so container limits divide that, not the machine.

Containers buy isolation, reproducibility and disposability. They do not buy
parallelism.

### Choosing the two models

Split them by **architecture, not size**. On Apple Silicon, generation speed is
roughly `bandwidth ÷ bytes-read-per-token`: a dense model reads its whole file
per token, an MoE reads only its active experts. Measured on an M4 Max / 36 GB:

| Model | Type | Size | Active | Throughput |
|---|---|---|---|---|
| `gemma4:31b` | dense | 19 GB | 30.7B | 17.4 tok/s |
| `gemma4:26b` | MoE | 17 GB | 3.8B | **81.7 tok/s** (4.7×) |

The orchestrator emits a few hundred tokens of schema-constrained JSON per
round, so it can afford the dense model. The worker emits whole files, which is
where the wall-clock actually goes, so it gets the MoE. Hence the defaults.

Both together are 36 GB of weights against a ~27 GB GPU budget, so they cannot
stay resident. Keep `OLLAMA_MAX_LOADED_MODELS=1` and let Ollama swap — a swap
measured 6.7 s, about twice a round, against minutes saved on every worker
call. If you'd rather have zero swapping, set both roles to `gemma4:26b`.

## Verified against

`langgraph 1.2.11`, `langgraph-checkpoint-sqlite 3.1.1`, `langchain-ollama
1.1.0`, `langchain-core 1.5.4`, `pydantic 2.13.4`, Python 3.11.15, Ollama
0.32.9, on an M4 Max / 36 GB.

## Where to spend effort

In order of payoff:

1. **Task decomposition and file scoping.** The worker only sees the files its
   task lists. If two tasks touch the same file, they are one task —
   overlapping file sets are the top cause of merge chaos.
2. **A stricter verify command.** Each of lint and typecheck catches a whole
   class of failure the model can then fix itself on retry.
3. **A hand-written `AGENTS.md`** in your target repo. Write it yourself:
   developer-written context measurably improves agent success, LLM-generated
   context slightly reduces it.
4. Framework tuning. Last. It matters least.

## Upgrade path

The graph does not change. Only [`src/swarm/nodes/worker.py`](src/swarm/nodes/worker.py)
does. Swap the schema-forced single-shot call for a real agentic worker (Claude
Agent SDK, an Ollama tool loop, a hosted model) and keep everything else — the
ledgers, worktrees, verification, merge logic, checkpointing. That is the whole
point of putting the intelligence in the workers rather than the orchestrator.

## Honest expectations

Round one will disappoint you. A local sub-30B model on a real repo writes code
that doesn't compile a meaningful fraction of the time. That is normal, and it
is exactly why the verifier and retry loop exist. **Judge the system by what
survives verification, never by what the model emits.**

## Contributing

Pull requests are welcome. `main` is protected: every change goes through a PR
that CI must pass and the code owner must approve. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Author

Kamyar Shahrestani ([@kamyarshahrestani](https://github.com/kamyarshahrestani)) —
<info@shahrestani.me>.

If you use this work, see [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE). Copyright © 2026 Kamyar Shahrestani.

## Architecture

v1 (this code) is a single process with git-worktree isolation. v2 moves the
control plane onto GitHub issues and the execution plane into containers — see
[docs/architecture-v2.md](docs/architecture-v2.md).
