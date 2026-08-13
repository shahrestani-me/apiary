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
swarm "add exponential backoff retry to the HTTP client"
```

Full install, model choice and memory budgeting: **[SETUP.md](SETUP.md)**.

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
| `SWARM_VERIFY` | `python -m pytest -q` | Your quality gate. Add lint/typecheck for a stricter one. |
| `SWARM_ORCHESTRATOR_MODEL` | `gemma4:31b` | Planning, routing, stall judgement. |
| `SWARM_WORKER_MODEL` | `gemma4:31b` | Writes the code. Quality here dominates. |
| `SWARM_MAX_PARALLEL` | `2` | Concurrent workers. Memory scales with this. |
| `SWARM_MAX_ROUNDS` | `8` | Hard stop. |
| `SWARM_MAX_STALLS` | `2` | No-progress rounds before **replanning** instead of retrying. |
| `SWARM_MAX_ATTEMPTS` | `3` | Per-task retries before abandoning. |
| `SWARM_WORKER_CTX` | `16384` | Never set this to a model's advertised 256K — the KV cache would cost more than the weights. |

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

## License

MIT — see [LICENSE](LICENSE).
