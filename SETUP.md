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

Measured on this machine, not estimated:

| Model | Type | Size | Active params | Throughput |
|---|---|---|---|---|
| `gemma4:31b` | dense | 19 GB | 30.7B | **17.4 tok/s** |
| `gemma4:26b` | MoE | 17 GB | **3.8B** | **81.7 tok/s** |

Same family, similar quality (MMLU Pro 85.2 vs 82.6 — about 3 points), **4.7× the
speed**. In a swarm that makes dozens of sequential model calls per run, that is
the difference between a 4-minute run and a 15-minute one.

**You have both, and the defaults use both** — `gemma4:31b` to plan, `gemma4:26b`
to write code. The reasoning is in §0b.

### Your memory budget, worked out

```
GPU budget (75% of 36 GB)          ~27.0 GB
  gemma4:26b weights (Q4_K_M)      -17.0 GB   (worker: resident while coding)
  KV cache (16K ctx x 2 parallel,
            q8_0 quantized)         -3.0 GB
                                   =========
  headroom for macOS + your apps     ~7.0 GB   ✓ fits
```

That is the budget for **one** model at a time. Both together are 19 + 17 = 36 GB
of weights, which does not fit — so keep `OLLAMA_MAX_LOADED_MODELS=1` and let
Ollama swap between them.

Swapping is the cost of the two-model split, and it is smaller than it sounds: a
swap measured **6.7 s** here, and the graph crosses the boundary roughly twice a
round (plan → worker → judge). Call it ~13 s a round, against minutes saved on
every worker call. Worth it. The `judge` node also short-circuits without a model
call whenever arithmetic can answer, which cuts some of those crossings.

If you would rather have zero swapping, set **both** roles to `gemma4:26b`. You
lose ~3 MMLU points on the planning calls, which is the place it matters least.

### The one number to watch: output speed

At 17.4 tok/s, a worker rewriting a 300-line file (~4,000 tokens) takes **~4
minutes** — per task, per attempt, and retries multiply it. At 81.7 tok/s the
same rewrite is **~50 seconds**. That gap is the entire reason for the split.

If output speed still bites, the next lever is **fewer output tokens**. The worker
currently emits *whole files* because sub-10B models can't produce reliable diffs
— but a 26B-class model can. Switching to search/replace block edits cuts output
by 5–10× on large files.

---

## 0b. Model roles

Split the roles by **architecture, not by size** — which is the opposite of the
usual "small model orchestrates, big model works" advice, and it is right here
because of how Apple Silicon generates tokens:

| Role | Default | What it does | Why this model |
|---|---|---|---|
| **Orchestrator** | `gemma4:31b` (dense) | Plan, route, judge "are we stuck?" | A few hundred tokens of schema-constrained JSON per round. Slow generation barely registers, so spend the budget on judgement quality. |
| **Worker** | `gemma4:26b` (MoE) | Write the code | Emits whole files. This is where the wall-clock goes, so buy throughput. |

The usual advice assumes tokens cost money. Locally they cost *time*, and time
is spent almost entirely in the worker — so the expensive-per-token model
belongs on the short calls, not the long ones.

Override either:

```bash
export SWARM_ORCHESTRATOR_MODEL=gemma4:31b
export SWARM_WORKER_MODEL=gemma4:26b
```

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

You already have both models:

```bash
ollama list
# gemma4:31b    6316f0629137    19 GB
# gemma4:26b    5571076f3d70    17 GB
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
repo=/tmp/.../demo base_branch=main orchestrator=gemma4:31b worker=gemma4:26b
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

## 4. Build the worker images

**Do this before the first run.** The orchestrator spawns one container per
issue, and it can neither build nor pull the image it needs: the Docker socket
it reaches is behind a proxy with `BUILD=0` and `IMAGES=0`
([`docs/security.md`](docs/security.md) §4), which is the narrowing working as
designed rather than a gap. So the images are a human's to build, once.

```bash
docker build -f Dockerfile.worker       -t apiary-worker       .
docker build -f Dockerfile.worker.node  -t apiary-worker-node  .
docker build -f Dockerfile.worker.react -t apiary-worker-react .
```

One image per stack, chosen per task from the issue's `## Stack` section:

| Stack | Image | Carries |
|---|---|---|
| `python` | `apiary-worker` | git, Python 3.12 |
| `node` | `apiary-worker-node` | git, Python 3.12, Node 22, npm |
| `react` | `apiary-worker-react` | the same, plus the React toolchain at `/node_modules` |

The React image is the slow one — roughly two minutes and about 1.1 GB, most
of it the `npm install` in the build. That install is deliberately here rather
than in the worker: a worker has no route to a registry, so the toolchain has
to arrive with the image, and this is the step where the network is allowed.
Nothing about it widens what a *running* container can reach.

Build only the ones you will use; a task whose stack has no image is refused
before it is claimed, with the build line in the message. `APIARY_WORKER_IMAGES`
overrides the mapping as `stack=image` pairs (`node=my-node:dev`), merged over
the defaults so overriding one stack does not un-configure the others.

`swarm doctor` reports which of these are actually present.

---

## 5. Point it at a real repo

```bash
export SWARM_REPO=~/sources/your-repo
export SWARM_ORCHESTRATOR_MODEL=gemma4:31b
export SWARM_WORKER_MODEL=gemma4:26b
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

## 5b. Point it at your task system (optional today, required soon)

apiary reaches a task system through **your own MCP server**, named in one
config file. It ships no Linear, Jira or GitHub Issues adapter, and GitHub
Issues goes through the same path as everything else — that is
`docs/adr/0001-task-systems-are-integrations.md`, and it is what makes
supporting another tracker configuration rather than code.

Without this file apiary reads and writes issues directly, which is still a
normal installation. `swarm run` prints which of the two it is doing on every
run, and `swarm doctor` reports an absent tracker as a skip.

**One file, and for GitHub it needs no new credential.**

```bash
mkdir -p .swarm/tracker
cat > .swarm/tracker/tracker.yaml <<'YAML'
tracker:
  mcp: github
  args:
    owner: your-org
    repo: your-repo
  intake:
    args:
      # The server's own filter parameters, forwarded verbatim. apiary never
      # parses these, so their names and spellings are the server's — check
      # them against your MCP server's tool schema, not against this file.
      state: all
      perPage: 100
YAML
export APIARY_TRACKER_CONFIG=.swarm/tracker/tracker.yaml
```

`mcp: github` selects a built-in profile, so the block above is only the
constants nobody but you can know. The profile pins the tool names, the
`method: create` discriminator GitHub's fused create/update tool needs, the
`issue_number` field map and the `number` ref rule. It also selects the **local
stdio** server (`github-mcp-server`), which takes a fine-grained PAT from its
own environment — the same `GITHUB_TOKEN` apiary already holds, in a second use.
No new variable, and no new hole in the egress allowlist.

Check the block, and then check the server, without starting a run:

```bash
python -m swarm.mcp.contract .swarm/tracker/tracker.yaml   # opens no socket
python -m swarm.mcp.tracker                                # one intake call, reads only
swarm doctor                                               # tracker.config/reachable/auth/tools
```

**Two arguments in `intake.args` deserve a sentence each**, because apiary
cannot supply them for you and both fail quietly:

- **List closed items too.** A finished task is a *closed* issue, and a ledger
  that cannot see closed work reports the run unfinished forever and loses the
  dependency edges pointing at it. The parameter that does this is the server's
  (`state` on GitHub), so it lives here rather than in the profile.
- **Raise the page size.** Intake is one call: the capability contract carries
  no paging rule yet, so a project with more items than one page returns a
  partial ledger — and a task nothing lists is a task nothing looks at again.

**A second tracker is a second block, and this is the whole of it.** Everything
server-shaped is already in the `linear` profile — the endpoint, the tool names,
the `issueId` comment argument, the `description` body field, and `identifier`
as the ref rule, because `ENG-123` is branch-safe and Linear's uuid is not.
What is left is one scope constant and one filter:

```bash
cat > .swarm/tracker/tracker.yaml <<'YAML'
tracker:
  mcp: linear
  intake:
    args:
      teamId: 00000000-0000-0000-0000-000000000000
      limit: 100
  create:
    args:
      teamId: 00000000-0000-0000-0000-000000000000
YAML
export APIARY_LINEAR_TOKEN=lin_api_...   # https://linear.app/settings/api
export APIARY_TRACKER_CONFIG=.swarm/tracker/tracker.yaml
```

**Note where `teamId` is, and where it is not.** A top-level `args:` merges into
all three capabilities, which is what you want for GitHub's `owner`/`repo` — all
three calls need them — and what you do not want here, because Linear's
`create_comment` takes no `teamId` and is entitled to reject one. The scope
constant is per-capability on this tracker and common on the other. That
asymmetry is not visible in either profile, so it is written here.

Your `teamId` is a uuid, and the same call proves the credential works:

```bash
curl -s https://api.linear.app/graphql -H "Authorization: $APIARY_LINEAR_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"query":"{ teams { nodes { id key name } } }"}'
```

**How far this gets you today: the read-only ladder, and not a run.** The block
above validates offline against `python -m swarm.mcp.contract`, which is the
only one of the three checks anyone has run on this tracker — the other two need
a workspace and a credential, and running them is the point of configuring it.
What happens *after* intake is the part already known not to work. The
payloads are handed to the GitHub adapter unchanged (`mcp/tracker.py` says so at
the line it happens), and that adapter mints a task ref out of a `number` field
with a `number` field's type, so a Linear tracker ends up with the contract's
`identifier` and the ledger's `#42` naming the same item in one cycle. See #260,
which also carries the one unverified prerequisite underneath all of it:
`mcp.linear.app` answers `WWW-Authenticate: Bearer realm="OAuth"`, and that a
Linear API key is accepted directly is so far a claim in Linear's documentation
rather than a call anyone here has made.

So configure Linear to exercise the seam — transport, contract and intake are
the part that tests whether a second tracker really costs configuration rather
than code — and run the swarm against GitHub until #260 closes.

**What still goes direct**, deliberately: pull requests, check runs, merges,
branches and the repository tree are the *code host*, which is GitHub by design
in this architecture. The six `swarm:*` labels and the attempt counter in the
issue body are also still written directly — they are apiary's own vocabulary,
which the ADR forbids putting in your tracker at all, and they are being removed
rather than routed.

**In the containerised orchestrator** (`compose.yaml`), `./.swarm/tracker` is
mounted read-only at `/etc/apiary/tracker`, so set
`APIARY_TRACKER_CONFIG=/etc/apiary/tracker/tracker.yaml` there. Note that the
`github` profile spawns a local server binary, and the orchestrator image does
not carry one — a containerised orchestrator on the GitHub profile needs
`github-mcp-server` added to `Dockerfile`, or a tracker reached over HTTP.

---

## 6. Tuning knobs

All via environment variables — see `src/swarm/config.py`.

| Variable | Default | Notes |
|---|---|---|
| `SWARM_MAX_PARALLEL` | 3 | Concurrent workers. On a laptop, 2 is realistic — each worker holds the model. |
| `SWARM_MAX_ROUNDS` | 8 | Hard stop. Prevents afternoon-long runs. |
| `SWARM_MAX_STALLS` | 2 | Consecutive no-progress rounds before **replanning** instead of retrying. |
| `SWARM_MAX_ATTEMPTS` | 3 | Per-task retries before abandoning it. |
| `SWARM_VERIFY` | `python -m pytest -q` | Your quality gate. Add lint/typecheck: `ruff check . && mypy . && pytest -q` |
| `SWARM_WORKER_CTX` | 16384 | Worker context window. **Never set this to gemma4's advertised 256K** — the KV cache at that size costs more memory than the 20 GB of weights. Lower to 8192 if you're memory-constrained. Exported on the host, it reaches the worker containers too — the orchestrator passes it through. |
| `SWARM_WORKER_TIMEOUT` | 1200 | Wall clock for a whole worker container: clone, one inference call, the verify run, the commit, the push, the PR. |
| `SWARM_VERIFY_TIMEOUT` | 300 | Wall clock for `SWARM_VERIFY` alone, **inside** the above. |
| `APIARY_CAPTURE` | unset (off) | Record every model call: prompt, raw response, Ollama's load/total durations, and the real exception. Off by default — a worker prompt carries whole file bodies from the repo under test. |
| `APIARY_CAPTURE_MAX_CHARS` | 8192 | Per-field truncation for run captures; the digest still covers the full text. Console captures are not truncated. |
| `APIARY_CONSOLE_DIR` | `.swarm/console` | Where `swarm console` writes captures. |

**These two are one setting with two numbers.** The verify command runs inside
the container, so `SWARM_VERIFY_TIMEOUT` is only reachable if
`SWARM_WORKER_TIMEOUT` is comfortably larger — the outer clock has to cover the
clone and a whole-file inference call at ~83 tok/s before the gate even starts.

Raising the verify budget on its own buys **nothing**: the container is killed
at the outer cap, and the attempt is recorded against *the container*, with a
reason naming a timeout that has nothing to do with your tests. That is why the
default that moved is the outer one. `swarm doctor` refuses an inverted pair,
and `swarm run` will not start on one.

---

## 7. What to expect, honestly

**Round one will disappoint you.** A local sub-30B model on a real repo will
write code that doesn't compile maybe a third of the time. That is normal and
it is exactly why the verifier + retry loop exists. Judge the *system* by
what survives verification, never by what the model emits.

With `gemma4:26b` as the worker you're starting from a decent baseline — this
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
