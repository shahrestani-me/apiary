# Local multi-agent coding swarm — setup

**Target:** Apple Silicon Mac, everything local, nothing leaves the machine.
**Stack:** Ollama (models) + LangGraph 1.2 (orchestration) + git worktrees (isolation).

Verified against: `langgraph 1.2.11`, `langchain-ollama 1.1.0`, `langchain-core 1.5.4`, Python 3.11.

---

## 0. Your machine: Mac Studio, M4 Max, 36 GB

```
Chip:    Apple M4 Max (10 performance + 4 efficiency cores, 32-core GPU)
Memory:  36 GB unified
Bandwidth: ~410 GB/s
macOS:   26.5.1
```

Strong machine — but **36 GB is the number that decides everything below**, because
macOS caps the GPU working set at roughly **75% of unified memory ≈ 27 GB**. That is
your real budget, not 36 GB.

Memory bandwidth is the other constraint. On Apple Silicon, generation speed is
roughly `bandwidth ÷ bytes-read-per-token`. For a **dense** model that means the
whole file; for an **MoE** only the active experts. This is why the model
architecture matters more than its parameter count here:

| Model | Type | Size | Active params | Est. speed on your M4 Max |
|---|---|---|---|---|
| `gemma4:31b` | dense | 20 GB | 30.7B | **~15 tok/s** |
| `gemma4:26b` | MoE | 18 GB | **3.8B** | **~50–70 tok/s** |

Same family, similar quality (MMLU Pro 85.2 vs 82.6 — about 3 points), **3–4× the
speed**. In a swarm that makes dozens of sequential model calls per run, that is
the difference between a 4-minute run and a 15-minute one.

**You have `gemma4:31b`, and that is the configured default.** It fits comfortably
and it is the higher-quality model. Start there.

### Your memory budget, worked out

```
GPU budget (75% of 36 GB)          ~27.0 GB
  gemma4:31b weights (Q4_K_M)      -19.0 GB
  KV cache (16K ctx x 2 parallel,
            q8_0 quantized)         -3.0 GB
                                   =========
  headroom for macOS + your apps     ~5.0 GB   ✓ fits
```

Why **single-tier** (one model for both orchestrator and worker): adding
`gemma4:e2b` alongside would be 19 + 7.2 = 26.2 GB of weights before any KV cache,
which breaks the budget. And a model swap between graph nodes costs 10–30 s —
more than the small model could ever save you. One model, always warm, wins.

### The one number to watch: output speed

At ~15 tok/s, a worker rewriting a 300-line file (~4,000 tokens) takes **~4.5
minutes** — per task, per attempt. With retries that adds up fast.

Two ways out if that bites, in order of effort:

1. **Pull the MoE sibling:** `ollama pull gemma4:26b` (18 GB) and set
   `SWARM_WORKER_MODEL=gemma4:26b`. ~4× faster for ~3 points of quality. For an
   iterate-and-verify loop that is almost always the right trade.
2. **Reduce output tokens.** The worker currently emits *whole files* because
   sub-10B models can't produce reliable diffs — but a 30B-class model can. Switching
   to search/replace block edits cuts output by 5–10× on large files. Ask and I'll
   add that mode.

---

## 0b. Model roles

One model fills both roles. The *prompts* differ, not the model:

| Role | What it does | Why one model is fine |
|---|---|---|
| **Orchestrator** | Plan, route, judge "are we stuck?" | Short schema-constrained JSON calls. Cheap even on a big model. |
| **Worker** | Write the code | The expensive calls. This is where quality shows. |

If you later move to a machine with 64 GB+, split them: a small fast model as
orchestrator, the biggest coder you can fit as worker. On 36 GB, don't.

**Alternative worth trying** if `gemma4:26b` underwhelms on real code:
`qwen3-coder:30b` (19 GB, MoE, 3.3B active) is a *dedicated* coding model rather
than a generalist. Gemma's headline coding number (80% LiveCodeBench) measures
competitive-programming puzzles, which is not the same skill as "modify these four
files without breaking the tests."

---

## 1. Ollama setup

```bash
brew install ollama          # or: brew upgrade ollama
brew services start ollama   # keeps the server at localhost:11434
ollama --version
```

You already have the model:

```bash
ollama list
# gemma4:31b    6316f0629137    19 GB
```

Sanity-check that tool/JSON mode works:

```bash
curl -s http://localhost:11434/api/chat -d '{
  "model": "gemma4:31b",
  "messages": [{"role":"user","content":"Return JSON: the numbers 1 to 3."}],
  "format": {"type":"object","properties":{"nums":{"type":"array","items":{"type":"integer"}}},"required":["nums"]},
  "stream": false
}' | python3 -c "import sys,json; print(json.load(sys.stdin)['message']['content'])"
```

You should get `{"nums":[1,2,3]}`. **If this fails, stop and fix it** — schema-forced
JSON is the foundation the whole orchestrator sits on.

### Ollama tuning for 36 GB

```bash
export OLLAMA_KEEP_ALIVE=30m        # keep the model warm between graph nodes
export OLLAMA_MAX_LOADED_MODELS=1   # 36 GB fits exactly one big model
export OLLAMA_NUM_PARALLEL=2        # 2 concurrent workers
export OLLAMA_FLASH_ATTENTION=1     # required for KV quantization below
export OLLAMA_KV_CACHE_TYPE=q8_0    # halves KV cache memory, negligible quality cost
```

**Where you put these depends on how the server is started, and getting it
wrong fails silently.**

- **Started from a shell** (`ollama serve`, or `brew services` with the vars in
  its plist): `~/.zshrc` works. Restart with `brew services restart ollama`.
- **Started by the Ollama desktop app** (`/Applications/Ollama.app`): `~/.zshrc`
  is **never read**. launchd starts the app, not your shell. The exports appear
  to be set — `env | grep OLLAMA` in your terminal shows them — while the server
  process has none of them.

For the desktop app, use `launchctl setenv` (which GUI apps inherit at launch)
and restart it. There is a script for this:

```bash
./scripts/ollama-tuning.sh
```

Either way, **verify against the server process itself**, not your shell:

```bash
./scripts/ollama-tuning.sh --check
# or, by hand:
ps eww "$(pgrep -f 'Ollama.app/Contents/Resources/ollama' | head -1)" \
  | tr ' ' '\n' | grep '^OLLAMA_'
```

Anything absent from that output is not in effect. Note that `launchctl setenv`
does not survive a reboot.

Three of these are load-bearing on your machine:

- **`OLLAMA_KEEP_ALIVE`** — without it an 18 GB model unloads between calls and
  you eat a 10–30 s reload stall on *every node in the graph*. It will feel broken.
- **`OLLAMA_KV_CACHE_TYPE=q8_0`** (needs `OLLAMA_FLASH_ATTENTION=1`) — halves KV
  cache memory. Measured perplexity cost is 0.002–0.05, i.e. undetectable.
- **`OLLAMA_NUM_PARALLEL`** — careful: **RAM scales with `NUM_PARALLEL ×
  context`**. At 2 parallel and 16K context Ollama allocates 32K worth of KV
  cache. Don't raise both at once.

---

## 2. Python environment

```bash
brew install uv        # fast, and it manages the Python version for you

cd ~/sources/apiary  # wherever you cloned this
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## 3. Prove the plumbing works — before involving any model

```bash
pytest -q -s
```

This builds a throwaway git repo, plans a task, creates a worktree, applies an
edit, runs **real pytest** inside the worktree, judges progress, and merges the
branch back — all with a stubbed model.

Expected output:

```
repo=/tmp/.../demo base_branch=main orchestrator=gemma4:31b worker=gemma4:31b
planned 1 task(s): add-sub
[add-sub] wrote 1 file(s): calc.py
[add-sub] PASS
round 1: satisfied=True progress=True loop=False stalls=0/2
merged swarm/add-sub
cleaned up worktrees
1 passed
```

If this passes, every remaining problem you hit is **model quality**, not
plumbing. That distinction will save you days of debugging.

---

## 4. Point it at a real repo

```bash
export SWARM_REPO=~/sources/your-repo
export SWARM_ORCHESTRATOR_MODEL=gemma4:31b
export SWARM_WORKER_MODEL=gemma4:31b
export SWARM_VERIFY="python -m pytest -q"     # must match YOUR repo's test command
export SWARM_MAX_PARALLEL=2

swarm "add exponential backoff retry to the HTTP client"
```

**Your repo must be a clean git repo with a working test command.** No tests
means no verification, and no verification means the swarm produces plausible
garbage confidently.

Crashed run? Resume it — state is checkpointed to sqlite after every node:

```bash
swarm --resume a3f9c1d2
```

---

## 5. Tuning knobs

All via environment variables — see `src/swarm/config.py`.

| Variable | Default | Notes |
|---|---|---|
| `SWARM_MAX_PARALLEL` | 3 | Concurrent workers. On a laptop, 2 is realistic — each worker holds the model. |
| `SWARM_MAX_ROUNDS` | 8 | Hard stop. Prevents afternoon-long runs. |
| `SWARM_MAX_STALLS` | 2 | Consecutive no-progress rounds before **replanning** instead of retrying. |
| `SWARM_MAX_ATTEMPTS` | 3 | Per-task retries before abandoning it. |
| `SWARM_VERIFY` | `python -m pytest -q` | Your quality gate. Add lint/typecheck: `ruff check . && mypy . && pytest -q` |
| `SWARM_WORKER_CTX` | 16384 | Worker context window. **Never set this to gemma4's advertised 256K** — the KV cache at that size costs more memory than the 20 GB of weights. Lower to 8192 if you're memory-constrained. |

---

## 6. What to expect, honestly

**Round one will disappoint you.** A local sub-30B model on a real repo will
write code that doesn't compile maybe a third of the time. That is normal and
it is exactly why the verifier + retry loop exists. Judge the *system* by
what survives verification, never by what the model emits.

With `gemma4:31b` as the worker you're starting from a decent baseline — this
is not a toy model. Expect it to handle single-file and small multi-file tasks
reasonably, and to struggle when a task requires understanding code it wasn't
shown. Curating which files each task lists is therefore doing real work.

Where to spend your effort, in order:

1. **Task decomposition and file scoping.** With a fixed local model this is
   now your biggest lever — the worker only sees the files the task lists.
2. **A stricter verify command.** Add lint and typecheck; each one catches a
   whole class of failure the LLM can then fix itself on retry.
3. **A hand-written `AGENTS.md`** in your repo with conventions, architecture,
   and gotchas. Measured effect: developer-written context improves agent
   success rates; LLM-generated context slightly *reduces* them. Write it yourself.
4. **Better task decomposition.** If two tasks touch the same file, they are
   one task. Overlapping file sets are the #1 cause of merge chaos.
5. Framework tuning. Last. It matters least.

### The upgrade path when local models cap out

The graph does not change. Only `src/swarm/nodes/worker.py` does. Swap the
schema-forced single-shot call for a real agentic coding worker (Claude Agent
SDK, Codex SDK, or an Ollama tool loop) and keep everything else — the
ledgers, worktrees, verification, merge logic, checkpointing. That is the
whole point of putting the intelligence in the workers and not the orchestrator.

You can also go hybrid: a small local orchestrator (free, private, always on)
+ a hosted model for workers only when a task fails twice locally.

---

## Architecture

```
setup ─→ plan ─→ dispatch ═fan out═▶ worker(worktree-1) ─┐
           ▲         ▲               worker(worktree-2) ─┼─→ verify ─→ judge ─┐
           │         └──── retry ────────────────────────┘                    │
           └──────── replan (on stall) ────────────────────────────────────────┤
                                                                               │
                                                      integrate (merge) ◀──────┘
```

- **Task ledger** (`state.tasks`) — the plan, revisable. From Magentic-One.
- **Progress ledger** (`judge` node) — every round answers: satisfied? progressing?
  looping? Drives stall detection → replan rather than blind retry.
- **Worktree per worker** — `git worktree add -b swarm/<task-id>`. Parallel agents
  never see each other's files.
- **Verifier** believes only the exit code. No LLM judges code correctness.
- **Checkpointed after every node** — crash-resume is free.

The orchestrator does arithmetic and routing; it does not "reason" about
coordination. All model intelligence lives in the workers.
