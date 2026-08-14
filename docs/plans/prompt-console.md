# See what the models are actually asked, and what they answer

> Planning document. Not published as GitHub issues — the ticket sections below are
> issue-ready bodies if that changes.

**Repo:** `shahrestani-me/apiary` · **Status:** built on `feat/prompt-console`

## T1 verdict: reproduced — build the console

The spike ran the recorded fixture for `shahrestani-me/trip-planner#2` (goal
"Create a comprehensive test suite…", writable `tests/test_planner.py`, 5,364-char
human turn) through a **host-side** `propose_edits` against `gemma4:26b`, the same
model and settings the worker uses.

**4 of the first 10 attempts produced corrupt output**, covering every failure
class in the recorded run:

| Attempt | Failure |
|---|---|
| 2 | chatter written into the file — `# Wait, I made a mistake in my thought process above. Let me fix it.` |
| 5 | `SyntaxError: unterminated triple-quoted string literal` at line 62 |
| 8 | `SyntaxError: unterminated triple-quoted string literal` at line 41 |
| 10 | `SyntaxError: invalid syntax` — `trip1 = Trip("Paris", date(2:` |

Attempt 10 is the recorded run's own signature: the original
`results/issue-2-attempt-0.json` failed on `start_date = date(2:`, the same
mid-token truncation. Attempt 2 is the chatter from `issue-2-attempt-2.json`.

So the corruption is **not** an artefact of the container, the worktree, or the
verify harness — it reproduces on the host, at roughly a 40% rate, from a prompt
a human can now edit and re-fire in the console. The kill criterion did not
trigger; T4 was built.

Two things this rules in for follow-up work: the failure is in generation rather
than in `apply_edits`, and it is frequent enough that `SWARM_MAX_ATTEMPTS=3` will
often burn every attempt on one task.

---

---

## Problem

When a model call goes wrong, apiary keeps nothing. The prompt is built inline at the call site and discarded; the raw response is consumed by the structured-output parser before anyone sees it; the exception is flattened to a string at nearly every site and at `greenfield/bootstrap.py:209` is not even bound to a name before a default is silently returned.

The evidence is on disk. In `.swarm/runs/trip-planner-20260814-151653-zvqpxv/results/` the worker wrote the model's own chatter *into a source file* — "Wait, I'm still making typos in my thought process" — alongside token-level corruption (`start_date = date(2:` in attempt 0, `date(20int(2023, 5, 15))` in attempt 1). That is the signature of schema-constrained decoding going wrong, and it was noticed only because Python refused to parse the file. The same text in Markdown, JSON or a comment passes the verify gate silently.

Diagnosing it today costs a full distributed run — GitHub token, Docker, worker images, ~9 minutes — and still ends in a guess.

## The honest limit of this epic — read this first

The corruption above comes from `propose_edits`, whose only production caller is `worker/entrypoint.py:616` — **inside the worker container**. In-container capture is a non-goal here: the container's only host-writable channel is the `results/` mount, whose writer (`worker/result.py:451`) provably bypasses redaction, a gap `tests/test_artifacts.py:439` already asserts. Routing 60 KB prompts through it would enlarge a known leak surface by orders of magnitude.

So **run capture covers the six host-side orchestrator calls and never the worker's**. The motivating bug is reachable only by re-firing `propose_edits` on the host from a fixture — and whether a host-side re-fire reproduces the corruption at all is unknown. That is why T1 is a spike with a kill criterion, and why it comes first.

## Outcome

A bad model response can be read verbatim, next to the exact prompt that produced it and the real exception that ended it, without starting a run.

**Success signal:** the operator edits a prompt and re-fires the same site a second time within one sitting.

## Approach

Capture is a LangChain `BaseCallbackHandler` attached to the `ChatOllama` constructor inside `orchestrator_llm()` / `worker_llm()`, **auto-attached when `APIARY_CAPTURE` is set** and absent otherwise. `llm.py`'s three public signatures do not change and no call site is touched.

Spiked against the repo's own `.venv` (langchain-ollama 1.1.0, langchain-core 1.5.4), success *and* error paths: `on_chat_model_start` yields the message list, `on_llm_end` yields `response_metadata` (`model`, `total_duration`, `load_duration`, `prompt_eval_count`, `eval_count`), `on_llm_error` yields the exception intact and still propagates it to the caller — all paired by one `run_id`. `format` reaches the callback at invoke time, so `schema_name` is derivable from `format['title']`.

The console is a stdlib `http.server` on `127.0.0.1`, single-threaded, synchronous POST, single-flight. A threaded server would introduce the first thread in a codebase with none.

## Non-goals

- The **objective console** (launching/supervising real `swarm run` from the browser).
- **In-container capture** — see the honest limit above.
- Editing SYSTEM prompts, or overriding model / temperature / `num_ctx`.
- **Replay** of a captured call.
- Making `summary.json`'s `inference_calls` truthful. `CycleMetrics.inference()` has no production caller — but neither does `RunArtifacts.cycle()`, so every summary reports `cycles: []` and `api_calls: 0` too. That is #29's unmet Done-when and is filed separately rather than smuggled in here.
- Streaming tokens (meaningless under schema-forced decoding).
- Pruning `.swarm/runs/` — kept forever by design (#29).

## Console scope: two sites, not nine

Of the nine call sites, three are **human-turn inert** (`goal.propose`, `replan.propose`, planner replan-mode — every variable input is interpolated into the SYSTEM side), one is **dead** on the v2 path (`nodes/worker.py`; `cli.py:12-21` compiles no graph), one sends a **constant** (`doctor.schema_probe`), and one sits behind four deterministic rungs that usually answer without the model at all (`judge.py:605-636`).

That leaves two worth opening:

| Site | Why |
|---|---|
| `edit.propose_edits` | Where the corruption lives. Needs a fixture built from a repo root. |
| `bootstrap.choose_stack` | Proves the wire in seconds rather than minutes, and is the site that lies most today (`bootstrap.py:209` swallows everything unbound). |

`planner.plan_node` is real but low-yield next to the actual bug; it is a follow-up.

## Rejected alternatives

- **Wrapping what `structured()` returns** — it is a `RunnableSequence` whose parser has already consumed the `AIMessage`; a wrapper sees the prompt and the parsed object and nothing else.
- **`include_raw=True`** — changes the return type at all nine sites to `{"raw","parsed","parsing_error"}` *and* swallows parse errors into a field, so capture-on and capture-off would crash differently. `doctor.py:366`'s `isinstance(answer, Ping)` would start failing its own probe.
- **A recorder parameter through the call sites** — `tests/test_graph_stub.py:93` patches `structured` as `lambda _llm, schema:`; a third argument raises `TypeError` in three test files. `doctor.py:318` also stores the factories by reference in a zero-arg table.
- **`ollama run gemma4:31b`** — an instant REPL that cannot apply constrained decoding, which is the prime suspect. Useless for this bug.

## Sequencing

**T1 (spike) first — it can kill the console half.** If a host-side `propose_edits` re-fire cannot reproduce the corruption, T4 should not be written.

Then T2 (capture) → T3, T4, T5 in parallel. Critical path: T1 → T2 → T4.

## Point of no return

The first capture file written without a `"schema": 1` field. `.swarm/` is gitignored, unpruned and kept forever; without a version stamp there is no migration story, only "delete your history".


---

## Tickets

| # | Title | Size | Depends on |
|---|---|---|---|
| T1 | spike: can a host-side propose_edits re-fire reproduce the corruption? | S | — |
| T2 | feat(capture): record every model call — prompt, response, timing, exception | M | — |
| T3 | fix(errors): keep the type of the exception that killed a model call | S | T2 |
| T4 | feat(console): swarm console — fire one model call and read the answer | M | T1, T2 |
| T5 | feat(capture): cap a capture record, and say when capture is on | S | T2 |

---

## T1 — spike: can a host-side propose_edits re-fire reproduce the corruption?

`spike` · size **S** · depends on: none · labels: `area/worker`, `size/S`

## Why

This is the go/no-go for the console half of the epic, and it is cheap to answer.

The bug that motivates all of this — model chatter and token corruption written into source files, visible in `.swarm/runs/trip-planner-20260814-151653-zvqpxv/results/issue-2-attempt-0.json` (`start_date = date(2:`) and `-attempt-1.json` (`date(20int(2023, 5, 15))`) — comes from `propose_edits`. Its only production caller is `worker/entrypoint.py:616`, **inside the worker container**, which this epic explicitly does not instrument.

So the console's value rests on an unverified assumption: that firing `propose_edits` on the *host*, from a fixture built the same way, reproduces the same failure. If it does not, the console cannot chase this bug and T4 should not be written.

## What

Timeboxed to half a day. No production code. The deliverable is a decision written into this issue.

1. Rebuild the fixture from the recorded run: the task goal, the writable set, and the readable set, using `read_writable` (`edit.py:210`) and `gather_context` (`:220`) against a checkout at the same commit.
2. Call `propose_edits(goal, writable, readable)` on the host against the same `SWARM_WORKER_MODEL`, several times.
3. Record: does corruption appear? At what rate? Is it in the raw text or introduced by the parser?
4. While there, note whether `with_structured_output`'s parse failures are even reachable — `format=<json schema>` constrains decoding, so "capture the real error" may be mostly about *transport* errors, which changes what T3 is worth.

## Kill criterion

**If corruption does not reproduce on the host in ~20 attempts, T4 (the console) is not written**, and the epic reduces to capture + error legibility (T2, T3, T5) — which stand on their own for the six host-side call sites.

## Acceptance criteria

- [ ] A written answer in this issue: reproduced (with rate), or not reproduced.
- [ ] A statement on whether host-side and in-container behaviour can be told apart at all with what is on disk today.
- [ ] A recommendation on T4: write it, or close it.

## Out of scope

Fixing the corruption. Instrumenting the container. Any production code.

---

## T2 — feat(capture): record every model call — prompt, response, timing, exception

`feature` · size **M** · depends on: none · labels: `area/ops`, `size/M`

## Why

Nothing in apiary records what was asked of a model or what came back. When a call fails, the exception is flattened to a string at nearly every site and at `greenfield/bootstrap.py:209` is not even bound to a name. There is no way to see why a model call failed short of re-running the whole swarm.

This is the walking skeleton; T3, T4 and T5 depend on it.

## What

A `BaseCallbackHandler` attached to the `ChatOllama` constructor inside `orchestrator_llm()` and `worker_llm()` (`src/swarm/llm.py:18,28`), plus `swarm/capture.py` holding the record type and the writer.

**The handler is attached by the factory itself when `APIARY_CAPTURE` is set, and not attached otherwise.** There is no `install_recorder()`: a module-level recorder constructed on first use is enough, since nothing in `src/swarm/` imports `threading`, `asyncio` or `contextvars` and the console (T4) is deliberately single-threaded. This is what makes "no call site is touched" literally true — and it is what makes the `swarm doctor` acceptance criterion below reachable, since `doctor.py:364` goes through `orchestrator_llm()` like everything else.

Spiked and confirmed on the pinned `langchain-ollama` 1.1.0 / `langchain-core` 1.5.4, on both the success and error paths:
- `on_chat_model_start` → the full message list; `format` (the bound JSON schema) arrives at invoke time, so `schema_name` comes from `format['title']` rather than being stamped
- `on_llm_end` → `response_metadata` = `{model, total_duration, load_duration, prompt_eval_count, eval_count, done_reason}`
- `on_llm_error` → the exception, **which still propagates to the caller**
- all paired by one `run_id`

`invocation_params` carries no model name, so the factory stamps `role` and `model` — it holds `SETTINGS.orchestrator_model` / `worker_model` two lines above.

### Record schema

```
{ "schema": 1, "ts": "…Z", "event": "llm.call",
  "id": "<run_id>", "role": "orchestrator"|"worker", "model": "gemma4:31b",
  "schema_name": "Plan",
  "messages": [{"role":"system","content":"…"},{"role":"human","content":"…"}],
  "prompt_chars": 41823, "prompt_sha256": "…", "prompt_truncated": true,
  "response": "…", "response_chars": 812, "response_sha256": "…",
  "parsed_ok": true,
  "error": null | {"type": "ConnectError", "message": "…"},
  "total_s": 13.401, "load_s": 10.417,
  "prompt_tokens": 231, "output_tokens": 11 }
```

Hashes are of the **full pre-truncation** text. `"schema": 1` is mandatory from the first write — `.swarm/` is gitignored, unpruned and kept forever, so the version stamp is the entire migration story. `total_s` / `load_s` are Ollama's own `total_duration` / `load_duration` (ns → s), not a wall clock.

### Storage, and who owns it

This ticket owns **`console_root()` and `APIARY_CONSOLE_DIR`** (default `.swarm/console`), because its own `swarm doctor` acceptance criterion writes there. It must be a **sibling** of `artifacts_root()` (`artifacts.py:239`) with its own env var, deriving nothing from it — otherwise setting `APIARY_ARTIFACTS_DIR` silently relocates captures.

Captures go to `run_dir()/llm.jsonl` when a run is active, `.swarm/console/<id>.json` otherwise — one file per console capture, so a leak-audit hit names a record rather than naming a day.

### Redaction — the part that is easy to get silently wrong

Redaction here is **per-writer, not per-directory**: only `EventLog.emit` (`artifacts.py:328`), `_write_json` (`:937`) and `container_log` (`:798`) redact, and `worker/result.py:451` already bypasses it. The capture writer must route through `_redacted` / `_default_redactor`. Do not hand-roll `json.dumps`.

Two traps: (a) `EventLog.emit` **raises** `ArtifactsError` on `OSError` (`artifacts.py:334`), and (b) LangChain swallows callback-handler exceptions by default — so naive reuse produces *silence* on a full disk, not an error. Decide explicitly: a capture write failure is caught inside the handler and reported as one stderr line, and it must never change a run's control flow or exit code.

### Audit and docs (folded in — it is one line and one paragraph)

`scan_artifacts` (`security.py:656`) is the only genuinely directory-wide control and is pointed at `.swarm/runs` (`docs/security.md:305`). Widen it to `.swarm/` so the new tree is covered — `.swarm/` contains only `runs/` today, so there is no false-positive risk. Add a short `docs/security.md` section stating what a capture contains and, plainly, **what the redactor does not catch**: it knows only env-enrolled literals and three GitHub shapes (`gh[pousr]_`, `github_pat_`, `://user:pass@`). An AWS `AKIA…`, an `sk-…`, a PEM block or a JWT in a target repo's `application.yml` is matched by neither the redactor nor `find_secrets`, which share the patterns. Capture is justified by locality and retention, not by redaction — no ticket or PR body in this epic may claim otherwise.

Also state the `INHERITED_ENV` decision: `APIARY_CAPTURE` does **not** join it (`containers/manager.py`), so a host-side flag cannot silently switch on capture inside a container.

## Acceptance criteria

- [ ] With `APIARY_CAPTURE` unset, a subprocess `swarm run` ends with `"swarm.capture" not in sys.modules` and no directory created.
- [ ] With `APIARY_CAPTURE=1`, `swarm doctor` writes exactly one capture record containing a real prompt, a real response and a non-zero `total_s` — via doctor's own schema probe, the cheapest real model call in the system. **This is the end-to-end proof and it requires no other ticket.**
- [ ] A source pin, in the shape of `tests/test_reconcile.py:1289`, asserts `include_raw` appears nowhere and `ChatOllama(` appears only in `llm.py`.
- [ ] A failing call produces a record with `error.type` set to the real exception class name and the prompt that caused it, and the exception still reaches the caller unchanged.
- [ ] A registered credential placed in a prompt does not reach disk.
- [ ] The existing suite passes with no test double, seam or call site modified.

## Test notes

- **Hermetic, and this is the point.** The handler is unit-tested by driving it directly with synthetic `on_chat_model_start` / `on_llm_end` / `on_llm_error` payloads. Seam-injected doubles bypass the factory and therefore bypass capture — correct (a fake model made no real call), but it means the call sites cannot be what exercises capture.
- **No-op guard**, modelled on `test_the_audit_is_not_a_scanner_that_never_matches` (`tests/test_artifacts.py:429`): fails if the recorder silently regresses to writing nothing. This is the test that catches LangChain swallowing a handler exception.
- **Redaction test**, modelled on `test_no_artifact_carries_a_credential` (`:406`) — not on the acknowledged-gap test at `:441`.
- **One `ollama`-marked test** end-to-end (marker count 5 → 6); CI deselects it, which is the existing contract.
- Inject the clock as `EventLog.clock` (`artifacts.py:318`) already does, or the timing field makes these the first flaky tests in this suite.
- Promote the `.invoke` fake to `tests/fixtures/model.py` alongside `github.py` / `repo.py` / `failures.py`. There are ~8 private copies; `conftest.py:5-8` bans copies, not fixtures.

## Files

`src/swarm/llm.py`, new `src/swarm/capture.py`, `src/swarm/config.py`, `src/swarm/artifacts.py` (`console_root`, writer), `src/swarm/security.py`, `docs/security.md`, new `tests/test_capture.py`, new `tests/fixtures/model.py`, `README.md` + `SETUP.md` env tables.

## Out of scope

In-container capture. Making `summary.json` truthful — `RunArtifacts.cycle()` has no production caller either, so `cycles: []` and `api_calls: 0` are equally false; that is #29's unmet Done-when and is filed separately.

---

## T3 — fix(errors): keep the type of the exception that killed a model call

`bug` · size **S** · depends on: T2 · labels: `area/orchestrator`, `size/S`

## Why

The exception object is destroyed at nearly every model call site. Only `doctor.py:366` preserves the exception's **type name**; only it and `edit.py:326` use `raise ... from exc`. The worst case is `greenfield/bootstrap.py:208-210`:

```python
except Exception:  # noqa: BLE001 - local model failures are varied
    return DEFAULT_STACK
```

`exc` is not bound at all. Type a React brief, get `python`, and you cannot tell whether the model said python or Ollama was down. `tests/test_bootstrap.py:105` asserts the fallback and asserts nothing about diagnosis — it blesses the blindness.

## What

Preserve `type(exc).__name__` alongside the message at each swallowing site. **No control flow changes** — every fallback still fires exactly as today.

| Site | Today | After |
|---|---|---|
| `greenfield/bootstrap.py:209` | `except Exception:` — `exc` unbound | bind it; distinguish "fell back after `{type}: {msg}`" from "the model answered python" |
| `worker/edit.py:326` | `EditError(f"model call failed: {exc}")` | include the type name |
| `nodes/judge.py:668` | reason `judgement failed: {exc}` | include the type name |
| `orchestrator/goal.py:261`, `replan.py:314` | `the planner could not be reached: {exc}` | include the type name |
| `nodes/planner.py:924` | no `try` at all | leave as is |

## Acceptance criteria

- [ ] `choose_stack` with an unreachable model still returns `DEFAULT_STACK`, and the reason it fell back names the exception type.
- [ ] Every site in the table carries `type(exc).__name__`.
- [ ] No control flow, return value or exit code changes anywhere.
- [ ] `tests/test_bootstrap.py:105` is extended, not weakened.

## Test notes

Note the trap: a test that injects a double via the `llm=` / `oracle=` seam **bypasses the factory and therefore bypasses capture**, so it cannot assert on a capture file. Assert on the returned reason string here, and leave capture assertions to `tests/test_capture.py`, which drives the handler directly.

## Files

`src/swarm/greenfield/bootstrap.py`, `src/swarm/worker/edit.py`, `src/swarm/nodes/judge.py`, `src/swarm/orchestrator/goal.py`, `src/swarm/orchestrator/replan.py`, and their tests.

## Out of scope

A fix-string taxonomy (that is the console's error panel, T4). A new exception hierarchy.

---

## T4 — feat(console): swarm console — fire one model call and read the answer

`feature` · size **M** · depends on: T1, T2 · labels: `area/console`, `size/M`

## Why

Seeing one model response today requires `--repo owner/name`, a GitHub token, Docker running, worker images built, `SWARM_REPO`/`SWARM_VERIFY` exported, and ~9 minutes; `swarm run` has 14 flags. The console's promise is **one model response without GitHub, without Docker, without a repo**.

**Gated on T1.** If the spike shows a host-side `propose_edits` re-fire cannot reproduce the corruption, close this ticket instead of building it.

## What

A `swarm console` subcommand serving one page from stdlib `http.server`, exposing two sites:

| Site | Human turn | Fixture |
|---|---|---|
| `edit.propose_edits` | `build_prompt(goal, writable, readable)` | repo root + comma-separated paths + goal, reusing `read_writable` (`edit.py:210`) and `gather_context` (`:220`) |
| `bootstrap.choose_stack` | the raw prompt | none |

`propose_edits` is the reason this exists; `choose_stack` proves the wire in seconds and is the site that lies most today. Do **not** ask the operator to paste 20 KB of file bodies — the fixture builder reads them from a repo root exactly as the worker does.

### `prompt_for` — the real deliverable

Extract one pure `prompt_for(...) -> (system, human)` per exposed site, **called by both production and the console**. Without it the console re-creates the call site's two lines and diverges silently the day either changes — and a console that shows a prompt production does not send is worse than no console.

The concrete failure it prevents: an operator pastes a goal, gets a clean answer, concludes the model is fine and the bug is downstream — when the real cause was `gather_context` (`edit.py:221`) spending 83% of `CONTEXT_BUDGET_CHARS` on a truncated lockfile, so the file carrying the convention never reached the model. The console built no readable set at all.

### Shape

- `GET /` the page; `GET /prompt` renders the prompt **before** firing, so there is something to read during the 30–120 s wait; `POST /run` fires, blocks, returns the capture.
- **Single-threaded, synchronous POST, single-flight** — a second concurrent call is refused, not queued. No `ThreadingHTTPServer`: it would add the first thread in a codebase with none.
- Split as `render(method, path, headers, body) -> Response`, a pure function tested directly, plus ~15 lines of `BaseHTTPRequestHandler` glue. **Headers are a parameter** — the `Host` check below lives in them, and a `render(method, path, body)` signature would leave it untestable through the seam.

### Security — scoped to a loopback dev tool, not a public service

- **Bind literal `127.0.0.1`**, never `""` / `0.0.0.0`, and refuse a wildcard bind at startup with a `ConfigError`. Not paranoia: `EGRESS_ALLOWLIST` includes `host.docker.internal` (`security.py:309`) and `EgressPolicy.filter_lines` (`:408`) generates `rf"^(.*\.)?{re.escape(entry)}$"` — **host-only, no port term** — so tinyproxy permits a worker container to reach *any* port on the host gateway. A wildcard bind is reachable from the one process the whole threat model is built around.
- **`Host` header allowlist** (exact `127.0.0.1:<port>` / `localhost:<port>`). Three lines, and it is what stops DNS rebinding turning a visited web page into a capture reader.
- **Render capture data via `textContent` only**, never string-concatenated HTML — model output originates in an arbitrary target repo.
- **`validate_capture_id`** modelled on `validate_run_id` (`run.py:168`), then `resolve()` + `is_relative_to(console_dir)`. Precedent test: `test_an_id_from_a_human_cannot_escape_the_root` (`tests/test_artifacts.py:524`).
- Human turn goes in the POST body, never a query string; override `log_message` to log method and path only.

Deliberately **not** included: CSP, `nosniff`, `Referrer-Policy`, `Origin`/`Sec-Fetch-Site`, a pre-read body cap. A CSP without `unsafe-inline` forces served `.js` and `.css` and therefore a static-file route with its own traversal surface — a poor trade for a loopback tool with a lifetime measured in minutes. Revisit if the console ever binds anything but loopback.

### Error panel

A legible error names: site, model, the **resolved** `base_url` (`config.py`'s container/host resolution means it is not what the operator assumes), the exception **type**, and a fix. Follow doctor's contract (`doctor.py:198` raises if a failing check names no fix) and reuse doctor's existing probe for the not-running / not-pulled states rather than reimplementing them. Take the console's model seam from `doctor.Inference` (`doctor.py:295`), whose docstring already calls it "the seam that keeps the suite hermetic" — `HostInference.schema_probe` (`:353`) is already 90% of the last hop.

## Acceptance criteria

- [ ] A test asserts the console and production render **byte-identical** `(system, human)` tuples for both sites, asserting on `prompt_for`'s return value — not on `FakeEditor`'s recorded prompt, which captures only the human turn.
- [ ] `swarm console` serves on `127.0.0.1`; asserted against `server.server_address`. A wildcard bind is refused with one `!` line and exit 1.
- [ ] Pointing the `propose_edits` tab at a repo root and firing returns the model's edits, the raw response and the duration.
- [ ] Stopping Ollama and firing shows a legible error naming the exception type, the resolved `base_url` and a fix — not a stack trace, not a blank box.
- [ ] `<script>alert(1)</script>` as the human turn renders as text.
- [ ] A request with a foreign `Host` is refused; `GET /capture?id=../../…` is refused.
- [ ] Every fired call leaves a capture record in `.swarm/console/`.
- [ ] A new test pins the subcommand set and the console's options. Nothing enumerates subcommands today — `tests/test_cli_run.py:1060` only asserts `doctor.build_parser()`'s dests are a subset — so adding `console` breaks no existing test and option drift would go unnoticed.

## Test notes

**Never bind a socket in a test** — this suite has no port-in-use flakiness and should not acquire any. Assert on `render()` directly; drive the handler once over `io.BytesIO` to prove the glue is wired. Hermetic tests inject an `Inference` fake; one `ollama`-marked test uses the real one.

## Files

New `src/swarm/console.py`, `src/swarm/cli.py`, `src/swarm/worker/edit.py` and `src/swarm/greenfield/bootstrap.py` (`prompt_for` extraction), new `tests/test_console.py`, `README.md`, `SETUP.md`.

## Out of scope

`planner.plan_node` and `goal.assess` tabs (follow-ups). The three human-turn-inert sites. `nodes/worker.py` (dead on the v2 path). Replay.

---

## T5 — feat(capture): cap a capture record, and say when capture is on

`chore` · size **S** · depends on: T2 · labels: `area/ops`, `size/S`

## Why

Worker prompts embed whole file bodies — `MAX_FILE_CHARS` 20,000 per file plus `CONTEXT_BUDGET_CHARS` 24,000 for the read-only set (`edit.py:77,82`). Uncapped, that is 2–5 MB per run against a `.swarm/runs` that is 180 KB across ten runs today and is never pruned, by design.

And an optional behaviour that nothing announces is one nobody remembers enabling. A developer turns capture on to debug one prompt; eighteen months later the laptop holds a copy of every private repo the swarm has touched — with, per T2, no redaction guarantee for third-party secrets.

## What

- **Per-record truncation**: `APIARY_CAPTURE_MAX_CHARS`, default 8192, for run capture. Console captures are unbounded — the operator typed them. The SHA-256 always covers the full pre-truncation text and `*_truncated` is visible in the record: a silently truncated tail is worse than no capture when the bug is in the tail.
- **A `capture: ON → <path>` line in the `swarm run` banner**, alongside the existing merge / update / infrastructure policy lines at `cli.py:458-465`.
- Document `APIARY_CAPTURE` and `APIARY_CAPTURE_MAX_CHARS` in the `README.md` and `SETUP.md` env tables.

## Acceptance criteria

- [ ] A prompt over the cap is truncated, marked `truncated`, and its hash still covers the full text.
- [ ] `swarm run` with capture on prints one line naming the capture path; with capture off it prints nothing new.
- [ ] Both env vars appear in both env tables.

## Files

`src/swarm/capture.py`, `src/swarm/config.py`, `src/swarm/cli.py`, `README.md`, `SETUP.md`, `tests/test_capture.py`.

## Out of scope

A per-file byte cap, a session ring buffer and `--purge` — premature for a default-off flag writing well under 1 MB per run; revisit if capture is ever left on. Pruning `.swarm/runs/`. A `swarm doctor` check: doctor has no "fine, but you should know" state (`doctor.py:186-190`), and adding a check costs two hard-coded count assertions (`tests/test_doctor.py:789,813`) plus a hand-maintained width table (`doctor.py:165`).

---

## Follow-ups (not in this epic)

- Wire RunArtifacts.cycle() into the reconcile loop so CycleMetrics.inference() and api_call() finally have producers — every summary.json today reports cycles: [], inference_calls: 0, api_calls: 0. This is #29's unmet Done-when, not new work.
- Replay: re-fire a captured call from the console with a visible diff against the original prompt. The first thing that will be asked for after v1.
- In-container capture for worker/edit.py — the highest-value prompt in the system. Blocked on closing the worker/result.py:451 redaction gap first.
- Close the worker/result.py:451 redaction gap that tests/test_artifacts.py:441 currently asserts as a known hole.
- A second pattern set for find_secrets (AWS, OpenAI, Stripe, PEM, JWT) so the audit is meaningful for target-repo secrets, not just GitHub ones.
- Console tabs for planner.plan_node (fresh-plan mode only) and goal.assess — real but lower-yield than propose_edits, and goal.assess needs a label saying production usually answers arithmetically before the model is consulted.
- A carve-out letting interpolated data blocks (failures, existing, shipped, missing) be edited while SYSTEM text stays frozen — would make goal.propose, replan.propose and planner replan-mode reachable.
- A producer for CycleMetrics.queue(depth) — built, serialised, printed and caller-less.
- logs/ is empty in every recorded run — container stdout is not reaching the artifacts dir. Separate bug, adjacent pain.
- EGRESS_ALLOWLIST matches by host with no port term, so allowing Ollama on 11434 allows every other port on the host gateway. Standalone hardening ticket.
- The objective console: launching and supervising real swarm run processes from the browser. Deferred by decision; the sync-POST design does not block it.
