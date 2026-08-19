# Credentials, egress and the Docker API

Status: **implemented, with two named seams.**
[`src/swarm/security.py`](../src/swarm/security.py) is the machine-readable
version of this document, [`compose.yaml`](../compose.yaml) is the deployed
version, and `tests/test_security.py` asserts that all three agree. Where they
disagree, the module wins — it is the one the tests read.

[`docs/architecture-v2.md`](architecture-v2.md), "Three constraints", third one:

> A worker container executes LLM-generated code *and* holds a token that can
> push. That is the classic exfiltration shape.

Everything below follows from that sentence. It is a defence in four layers, and
the layers are listed in order of how much they actually buy:

| Layer | Stops | Does not stop |
|---|---|---|
| **Token scope** | reaching any repository but the target | anything inside the target repository |
| **Token handling** | the credential surviving in a log, a config file or a process listing | a process that holds it while it runs |
| **Egress filter** | reaching any host but GitHub and the host's Ollama | reaching a different repository on GitHub |
| **Docker API narrowing** | exec, build, image pull, volume and network creation from the orchestrator | a privileged `create`, which is why `assert_unprivileged` exists |

No layer is sufficient. The token is what makes "a worker cannot reach an
unrelated repo" true; the egress filter is what makes "and nothing else at all"
true; and neither says anything about the third row, which is why token
*handling* gets as much of this document as token *scoping*.

**All of it describes `swarm run`.** The other runner, `swarm local`, has none
of these four layers — it executes model-written code on the host, unconfined,
and no part of this document is true of it. §7 says what that means, and what
the command now does about saying so itself.

---

## 1. The token

### Minimum scopes

A **fine-grained personal access token**, resource owner set to the account or
org that owns the target repository, and **"Only select repositories" naming
exactly one**:

| Permission | Level | Why |
|---|---|---|
| Contents | Read and write | push `swarm/issue-<n>` |
| Pull requests | Read and write | open and update the one PR per issue |
| Issues | Read and write | read the contract, write `swarm:*` labels and comments |
| Metadata | Read-only | mandatory; GitHub adds it for you |

That is the whole list. `swarm.security.REQUIRED_PERMISSIONS` is the same list in
code.

### What must stay off

`swarm.security.FORBIDDEN_PERMISSIONS`, and one of them matters far more than the
rest: **Workflows.** With it, generated code can rewrite `.github/workflows/*` —
and CI re-running the verify command on neutral ground is the gate the whole
design rests on ([architecture-v2](architecture-v2.md), "PRs are the integration
mechanism"). A swarm that can edit its own gate has no gate. The others
(Actions, Administration, Secrets, Environments, Packages, Members) are listed
rather than merely omitted because the GitHub UI puts them one checkbox away
from the four above.

### Never a classic PAT

A classic PAT's scopes are **verbs, not repositories**: `repo` means every
repository the account can reach, private ones included. So does an OAuth token
— which is what `gh auth token` prints, making it simultaneously the most
convenient credential on a developer's machine and the worst one to hand a
container.

`swarm.security.assert_scoped_token` refuses the `ghp_`, `gho_`, `ghu_` and
`ghr_` prefixes for exactly this reason and accepts `github_pat_` (fine-grained)
and `ghs_` (GitHub App installation token, which is per-repository *and*
short-lived — the better answer where an App is available).

### Rotation

Fine-grained PATs expire; pick the shortest expiry you will tolerate, and treat
the run that fails with a 401 as the reminder rather than as an incident.

1. Mint the replacement first, with the table above, on the one repository.
2. Put it in the `.env` file next to `compose.yaml`, or in the environment of
   the shell that runs `docker compose`. It is never a build argument and never
   an `ENV` default — `docker history` on either image shows nothing worth
   stealing, and that property is worth keeping.
3. Restart the orchestrator. Containers read the token at create time, so
   in-flight workers keep the old one until they exit; there is no reload.
4. **Revoke the old one** at
   `github.com/settings/personal-access-tokens` — the step that is skipped, and
   the one that makes rotation mean anything.

Rotate immediately, not at expiry, if a token ever reached a terminal, a
screenshot, an issue comment or a run artifact. Section 5 is how you find out
whether it did.

---

## 2. Token handling

A scoped token that ends up in an artifact you keep forever is still a leak. The
credential passes four boundaries and each one is closed in code:

**It does not go in a URL.** `worker/pr.py` pushes with
`git -c credential.helper=<snippet>`, where the snippet names an *environment
variable* and the value lives only in the child process. The obvious
`https://x-access-token:<token>@github.com/...` would be written into
`.git/config`, echoed by `git remote -v`, and quoted verbatim in every git error
message. `worker/entrypoint.prepare_checkout` clones from a plain URL for the
same reason.

**It does not go in `argv`.** `ContainerManager._env_flags` emits bare
`--env GITHUB_TOKEN` when the value is already in the orchestrator's own
environment, so the docker CLI reads it there. The value appears in no process
listing, in no `docker inspect` output, and in the text of no failed `create`.

**It does not go in an image.** Neither `Dockerfile` nor `Dockerfile.worker`
takes it as a build argument or sets it as an `ENV` default.

**It does not go in a captured string.** `containers/manager.py` redacts at the
capture boundary — `DockerCLI` owns a `Redactor` and applies it to everything
the daemon returns, container logs *and* the text of every failure, before any
caller sees either. Redaction happens before truncation, so a secret cannot
survive as two halves either side of an elision. The redactor knows both the
literals it was handed and the shapes it was not (`ghp_…`, `github_pat_…`,
`https://user:pass@…`), because a worker can print a credential this process
never saw.

That last one is the boundary that matters most, because `#29` writes captured
logs to `.swarm/runs/` and keeps them.

---

## 3. Egress

### Topology

```
   apiary-control (internal)          apiary-egress (internal)      apiary-uplink
  ┌───────────────────────┐         ┌────────────────────────┐    ┌────────────┐
  │ orchestrator ─────────┼─────────┼─ orchestrator          │    │            │
  │        │              │         │  worker containers ────┼────┼─ egress ───┼──▶ world
  │        ▼              │         │                        │    │   proxy    │
  │ docker-socket-proxy   │         │                        │    │            │
  └───────────────────────┘         └────────────────────────┘    └────────────┘
            │                                                            │
      /var/run/docker.sock                                    host.docker.internal
```

Both `apiary-control` and `apiary-egress` are `internal: true`, so containers on
them have **no default route**. A worker that ignores `HTTP_PROXY` does not
bypass the filter; it fails to connect. That is the difference between a policy
and a suggestion.

The two networks are separate so that a worker — the container running generated
code — cannot route to the Docker API at all. Docker isolates user-defined
bridge networks from each other by default, and the socket proxy is on
`apiary-control` only.

`host.docker.internal` is resolved in exactly one place, the egress proxy, which
is the only service with `extra_hosts: host-gateway`. Everything else names the
host in a proxied request and lets the proxy resolve it — which is what an HTTP
proxy does with a hostname, and why the orchestrator no longer needs
`extra_hosts` of its own.

### The allowlist

`swarm.security.EGRESS_ALLOWLIST`: `github.com`, `api.github.com`,
`codeload.github.com`, `host.docker.internal`, `mcp.linear.app`, plus any
subdomain of those.
tinyproxy runs `FilterDefaultDeny Yes` against regexes generated by
`EgressPolicy.filter_lines()` and pasted into `compose.yaml`; the suite fails if
the two drift, so a widened allowlist cannot land in the YAML alone.

The regexes are anchored at both ends. An unanchored `github\.com` also matches
`github.com.attacker.net`, and a proxy allowlist is the wrong place to learn
that.

#### The one entry ADR 0001 adds, and the two it does not

`mcp.linear.app` is the tracker MCP endpoint the orchestrator reaches a
customer's task system through (#149). It is the whole of the widening:

- **GitHub adds nothing.** The remote server at `api.githubcopilot.com`
  advertises the classic OAuth scopes — the `ghp_`/`gho_`/`ghu_`/`ghr_` family
  §2 refuses, because their scope is a verb rather than a repository — so it is
  incompatible with the token policy above. The GitHub tracker runs as a
  **local stdio** `github-mcp-server`, which takes apiary's existing
  fine-grained PAT from its environment and talks to `api.github.com`, already
  on the list (#143).
- **Jira is deferred**, so `api.atlassian.com` is absent. A hole for a tracker
  nobody has configured is precisely the quiet widening this section exists to
  prevent.

Note what an entry here cannot express. tinyproxy filters on hostname, so
`mcp.linear.app` is reachable by every container on `apiary-egress`, workers
included. **The tracker credential is what confines it**, exactly as §2 argues
for `github.com`: one host serves every customer and only the token knows
which. `APIARY_TRACKER_TOKEN` is passed to the orchestrator service alone, and
`containers.manager.INHERITED_ENV` carries `GITHUB_TOKEN` and `OLLAMA_HOST` and
nothing else — so a worker that reached the endpoint is answered 401.
`tests/test_security.py::test_the_tracker_credential_never_reaches_a_worker` is
that claim as an assertion.

### The knob, and its cost

`APIARY_EGRESS_ALLOW=pypi.org,files.pythonhosted.org`

The honest default breaks something: a `## Verify` command that runs
`pip install -e .` needs a package index. It is an environment variable rather
than a constant because **a registry that accepts uploads is an exfiltration
channel** — a token fits in a package name — so it is a decision an operator
makes per target repository, out loud,
rather than a default nobody reads.

### Why the worker images carry toolchains

`Dockerfile.worker` used to argue for baking in no toolchain at all, on the
grounds that "baking a stack in would quietly narrow the swarm to repos that
happen to use that stack", with the `## Verify` command installing whatever the
target repo required at run time.

**That argument does not survive contact with this section.** A worker sits on
an `internal: true` network whose only route out is the egress proxy, and the
enforced allowlist is the static block above — so a run-time install is not
slow, it is *denied*, in under a second. And the intent was stack-agnosticism
while the effect was the opposite: baking in none narrowed the swarm to Python,
because the one toolchain every image did carry was the Python the package
itself needs.

So agnosticism is bought by **several images, selected per task** (#99), each
carrying one stack and nothing else. That is a security improvement rather than
a regression, and for the reason this whole section exists: an image with a
toolchain in it needs no registry at run time, so the honest default stops
breaking anything and `APIARY_EGRESS_ALLOW` stops being the thing standing
between a green gate and a widened allowlist.

The images are built on the **host**, never by the orchestrator — the socket
proxy sets `BUILD=0` and `IMAGES=0` (§4), so it can neither build nor pull one.
`SETUP.md` step 4 is the command; `containers.manager.build_hint` is the same
line, produced by the code that refuses.

### The stack that was supposed to force a wider allowlist, and did not

#87 planned React web as the ticket where this posture broke: a stack with real
dependencies needs a package registry, so the enforced allowlist would have to
grow from GitHub-only to GitHub-plus-npmjs, and the exfiltration surface with
it. It is worth writing down that this **did not happen**, because the argument
generalises.

`Dockerfile.worker.react` installs react, react-dom, vitest, a JSX transform
and a DOM at **image build time** — on the host, where the network is allowed
and where step 4 already puts a human — and moves them to `/node_modules`, at
the filesystem root, where Node's ESM resolver finds them from any working
directory. The gate is then `vitest run`, and it was measured green in a
container started with `--network none`. The allowlist in §3 is byte-for-byte
what it was before React existed as a stack.

`EgressPolicy.from_env()` and `APIARY_EGRESS_ALLOW` are therefore still what
they were: **an escape hatch with no production caller, which widens nothing
today** — "The knob, and its cost" above describes the design, not the
behaviour, and `compose.yaml`'s "widen it with `APIARY_EGRESS_ALLOW`" says the
same thing with the same caveat missing. Setting that variable has no effect on
the enforced allowlist; the enforced list is the static block in `compose.yaml`
and nothing reads the environment into it. That remains a real gap for a target
repository whose own `## Verify` needs an index — but it is not React's gap, and
"make the allowlist enforceable" is no longer a prerequisite for a
dependency-carrying stack.

**What moved instead: the build-time trust set.** The *run-time* surface is
unchanged, and that is the narrower claim than it sounds. Before React, an
image was trusted to the Debian archive, this repository's own Python package
and the official Node image. `Dockerfile.worker.react` adds roughly 170 npm
packages resolved at build time, and the honest description of that is a new
supply-chain dependency, not an absence of one. What bounds it:

- lifecycle scripts are **not** run (`--ignore-scripts`), so installing a
  package does not execute its code;
- the resolution happens in a **throwaway build stage** as that image's
  unprivileged `node` user, and the runtime stage takes only the resulting
  directory with `COPY`, which executes nothing;
- `/node_modules/.bin` is **last** on `PATH`, so a package shipping a `bin`
  called `git` cannot get between the worker and the real one;
- `/node_modules` is root-owned and the worker runs as uid 10001, so the gate
  reads the toolchain and cannot change it.

What does **not** bound it: there is no lockfile and no integrity pinning below
the major version, so the exact set differs between two builds of the same
Dockerfile and cannot be audited after the fact. An operator who needs that
should generate a lockfile on the host and switch the `deps` stage to `npm ci`.

**And the cost to the gate's meaning.** A worker's toolchain comes from its
image and a GitHub runner's cannot, so the generated workflow installs the same
packages itself (`greenfield.provision.CI_SETUP`). `npm ci` is not available to
close that either: it needs a lockfile, and producing one needs the registry
the worker is denied. Two things follow, and the second is the sharper one:

- both sides install from the same pinned **major** ranges
  (`greenfield.stacks.REACT_TOOLCHAIN`) and can still resolve different patch
  versions;
- CI's `npm install` also resolves whatever the generated `package.json`
  declares, and the worker's gate installs nothing, so it never validates that
  list. A model that writes a package name no source file imports is
  **worker-green and CI-red** — a set divergence, not a version one.

Closing this properly means publishing the worker image to a registry so the
workflow can use `container:`, which is a much larger change than this one.

### What this does not buy

`github.com` is one host and every repository lives behind it. The egress filter
cannot tell the target repository from an unrelated one; only the token can. Both
layers, or neither.

### What the verify subprocess can see

`## Verify` is arbitrary shell chosen by the target repository, run inside the
container, in a process whose parent is holding two credentials. Its output
goes into the PR body, into `.swarm/runs/<id>/results/`, and in front of a
human — so a suite that echoes its own environment publishes them.

`worker/entrypoint.verify_env` therefore hands the subprocess an environment
**filtered from the worker's own**, dropping `GITHUB_TOKEN`, `APIARY_PUSH_TOKEN`
and anything whose name matches `containers.manager.SECRET_NAME_RE` — the same
regex that already enrols a container's variables for redaction, reused so the
two cannot drift apart.

Two things about the shape of that filter:

- **It is a deny-list, and that is deliberate.** An allow-list is the stronger
  posture and the wrong tool: the verify command belongs to the target
  repository, and a suite that needs `PYTHONPATH`, `NODE_ENV` or a database URL
  would be broken by one, in a way that looks like a bug in its own tests.
- **It filters rather than rebuilds.** A freshly constructed dict drops
  `HTTPS_PROXY`, and a worker has no default route — so the verify command
  would not fail, it would **hang**, until the outer container clock killed it
  several hundred seconds later with a reason naming the container. The eight
  load-bearing names are `VERIFY_ENV_REQUIRED` and each is pinned by a test.

**This stops accidents, not attackers.** Verified: a scrubbed child can still
read `/proc/1/environ` and get `GITHUB_TOKEN` back, because PID 1 in the
container is the worker and it runs as the same user. Anything that goes
looking still finds them. What removes that is the three-container split
(fetch / build / publish), so the push token and a widened egress never coexist
in one process — §6.

---

## 4. The Docker API

A bind-mounted `/var/run/docker.sock` is host root for whoever holds it, and the
orchestrator holds it while running model output. So it does not get the socket:
it gets `DOCKER_HOST=tcp://docker-socket-proxy:2375`, and the docker CLI that
`containers/manager.py` shells out to honours that without the module knowing a
proxy exists.

The proxy's surface is `swarm.security.SOCKET_PROXY_ENV`: `/containers`, the
`/version` handshake the CLI performs first, and `/_ping`. Everything else
returns 403. Three of the zeroes are worth naming:

- **`EXEC=0`** — `docker exec` turns any running container into a shell, and it
  is the first thing an attacker with a Docker API reaches for.
- **`BUILD=0`** — a build context is arbitrary code with a filesystem.
- **`IMAGES=0`** — the orchestrator cannot pull, so it can only run images
  already on this host. `apiary-worker` must be built locally, which is a
  constraint worth having: a compromised orchestrator cannot fetch an image of
  its choosing to run as its worker.

`:ro` on the socket mount is worth having and is **not** what makes this safe. A
unix socket is bidirectional whatever the mount says; a read-only socket still
accepts `POST /containers/create`. The narrowing is the environment, not the
mount flag.

### The residual risk, and what covers it

The proxy routes on **method and path**. It cannot read a request body, so it
cannot distinguish `POST /containers/create` for an ordinary worker from the
same POST with `Privileged: true`, `--pid=host`, or a bind mount of the Docker
socket. That endpoint has to be open for the swarm to work at all.

`swarm.security.assert_unprivileged(argv)` is the layer that does look. It
rejects `--privileged`, `--cap-add`, `--device`, `--group-add`, any namespace
flag set to `host`, an `unconfined` `--security-opt`, a mount of the Docker
socket or of a host system path, and `--user root`. The suite feeds
`ContainerManager.spawn`'s real argv through it, so the one code path that
creates containers is checked on every run of `pytest -q` rather than by
inspection.

---

## 5. Proving nothing leaked

`.swarm/runs/` is kept forever by design (#29), which means a token that reaches
it is a token that leaked forever, however tightly it was scoped. So the check is
a function rather than only a test:

```python
from pathlib import Path
from swarm.security import scan_artifacts
import os

leaks = scan_artifacts(Path(".swarm"), env=os.environ)
```

Point it at `.swarm/` rather than `.swarm/runs/`: capture (below) writes into
`.swarm/console/` too, and a scanner aimed at the one directory that predates it
would keep returning an empty list while the new one filled up.

An empty list is the answer you want. Each `Leak` carries a path, a line number
and the *kind* of match — never the matched text, because a leak report that
quotes the leak is a second copy of it, in a file that gets pasted into an issue
precisely because something went wrong.

It detects what the redactor redacts, using that module's own `SECRET_PATTERNS`
and `SECRET_NAME_RE` rather than a second opinion about what a secret looks like.
`tests/test_security.py` runs a container that deliberately echoes its token
through the real capture path, writes what was captured to a run directory, and
asserts the scan finds nothing — with a control that writes the raw token to the
same directory and asserts the scan *does* find it, because a scanner that finds
nothing because it looks at nothing passes the first test perfectly.

---

## 5a. Capture, and the console that reads it

`APIARY_CAPTURE=1` records every model call: the rendered prompt, the raw
response, Ollama's own `load_duration` / `total_duration`, and the real
exception when one ends the call. `swarm console` serves a page for firing one
call by hand and reading the result.

**What a capture contains.** A worker prompt is built by `read_writable` and
`gather_context`, so it holds *whole file bodies* from the repository under
test — up to `MAX_FILE_CHARS` per writable file plus `CONTEXT_BUDGET_CHARS` of
read-only context. `read_writable`'s docstring is explicit that the skip-list is
not consulted, so a path named in an issue's `## Files` is read whatever it is,
including a `.env`. `gather_context` skips dotfiles but reads `.json`, `.yaml`,
`.ini`, `.cfg` and `.toml` from the checkout root, which is exactly where an
`application.yml` password or an `appsettings.json` connection string lives.

**What redaction does and does not catch.** Every capture write goes through the
same `_redacted` / `_default_redactor` path as `events.jsonl` and `summary.json`
— redaction here is per *writer*, not per directory, so this is a property of
routing through `artifacts.write_json` rather than of the directory the file
lands in. That redactor knows two things: literals enrolled from **this
process's** environment by `SECRET_NAME_RE`, and three shapes — `gh[pousr]_`,
`github_pat_`, and `://user:pass@`.

It therefore does **not** catch a third party's credential that arrived in a
prompt from the target repository: an AWS `AKIA…`, an OpenAI `sk-…`, a Stripe
key, a PEM block, a JWT. `find_secrets` shares those patterns, so the audit does
not catch them either — there is no second opinion. **Capture is justified by
locality and retention, not by redaction:** the files are local, `.swarm/` is
gitignored, run captures are truncated by default, and nothing publishes them.
Do not treat a capture file as safe to paste into an issue.

**The console's own surface.** It binds `127.0.0.1` and refuses anything else,
because `EGRESS_ALLOWLIST` matches `host.docker.internal` by host with no port
term — a worker container can reach *any* port on the host gateway through the
sanctioned proxy, so a wildcard bind would put every captured prompt one request
away from the process that runs model-written code. It checks the `Host` header
(`BaseHTTPRequestHandler` validates none), which is what stops a page the
operator happens to be browsing from firing calls here and a DNS rebind from
reading the answers. Model output reaches the DOM only through `textContent`:
the model's input is an arbitrary repository, so its output is hostile by
construction.

**Not captured: anything inside a worker container.** `propose_edits` runs in
the container, whose only host-writable channel is the `results/` mount, and
that writer bypasses redaction (see §6). `APIARY_CAPTURE` is deliberately not on
`INHERITED_ENV`, so a host-side flag cannot switch capture on inside a
container. The console's `propose_edits` runs on the *host* instead, which is
close enough to be useful and different enough to be labelled as such on the
page.

---

## 6. Seams that are not closed yet

Two, both waiting on tickets whose file sets this one could not reach.

**The dispatcher does not pass the confinement flags yet (#21).**
`swarm.security.worker_create_flags()` returns
`--network apiary-egress --cap-drop ALL --security-opt no-new-privileges:true`,
and `ContainerManager.spawn` does not take extra create flags today, so nothing
calls it. Until #21 wires it, a worker lands on the daemon's default bridge with
ordinary egress and the token is the only thing between it and an unrelated
repository. The orchestrator itself *is* confined — it is a compose service — so
this gap is the worker's, and it closes with one argument.

**Nothing calls `assert_scoped_token` in production yet.** The natural place is a
preflight in `swarm.cli`, alongside the existing "GITHUB_TOKEN is not set" check
in `github/client.py`; neither file is in this ticket's `## Files`. Until then it
is a function a human runs, and the suite is what keeps it correct.

**The orchestrator image has no docker CLI.** `containers/manager.py` shells out
to `docker`, `Dockerfile` installs only git, and `DOCKER_HOST` is useless without
a binary to honour it. That is #21's Dockerfile change, not this ticket's, but it
is the first thing to hit when the socket proxy is first exercised end to end.


---

## 7. The local runner is outside all of it

`swarm local` is a second runner over the same repository, and everything above
is absent from it — not weakened, absent.

| | `swarm run` | `swarm local` |
|---|---|---|
| Container sandbox | yes | **no** |
| Egress policy | yes | **no** |
| Scrubbed verify environment | yes | **no** |
| Pull request and CI gate | yes | **no** |
| Merge queue | yes | **no** |

It builds the v1 LangGraph pipeline (`graph.py`) rather than the v2 reconcile
loop, and that pipeline reaches none of `containers/`, `worker/pr.py`,
`orchestrator/checks.py`, `orchestrator/mergeability.py` or `security.py`. The
verify command ends up in `nodes/verifier.py`, which is:

```python
subprocess.run(SETTINGS.verify_command, shell=True, cwd=worktree, ...)
```

A shell, on this host, in a worktree a model has just written into, inheriting
the invoking shell's environment. §2's token handling, §3's allowlist and §3's
`verify_env` filter are all somewhere else. Whatever the model wrote runs with
everything the person who typed the command has, including a `GITHUB_TOKEN`
that happens to be exported — nothing on this path filters an environment,
because there is no container boundary for one to be filtered across.

That is not a defect in the local runner. It is what "worktrees and host Ollama,
no GitHub" means once it is said in capability terms instead of convenience
ones — and saying it the convenient way was the defect, because
`docs/architecture-v2.md`'s third constraint ("a worker container executes
LLM-generated code") is the entire reason the other four sections exist.

**So the runner declares it** — [ADR 0003](adr/0003-orchestration-framework-is-a-detail.md)'s
decision 4, a runner declares its capabilities and the choice is presented in
those terms. `swarm local` refuses to start without
`--unsandboxed`; its `--help` carries the table above; and the run repeats one
sentence of it on stderr as it starts. The flag turns nothing off — there is no
sandbox for it to disable — and it grants nothing. It exists so that nobody
arrives on this path without having read one line saying what the path is.

**What it is still for.** A throwaway repository, on a machine and in a shell
you are willing to spend: the case it was built for, which was GitHub being
down or a network not being wanted. What it is not for is a checkout whose
contents matter, a shell holding credentials, or an objective written by
someone other than the person running it.

Whether the local runner remains supported at all is a separate question, and
open — ADR 0003 leaves it with the maintainer under "Still open". This section
is what holds while it does exist.

## The boot key

Creating a repository and doing the work inside it are two jobs with
incompatible permissions, so apiary uses two credentials.

| | Work key | Boot key |
|---|---|---|
| Variable | `GITHUB_TOKEN` | `APIARY_PROVISION_TOKEN` |
| Permissions | contents:write, pull_requests:write, issues:write, actions:read, metadata:read | administration:write, contents:write, workflows:write, issues:write, metadata:read |
| Used by | Orchestrator and every worker | `greenfield/provision.py`, once |
| Lifetime | The whole run | The seconds it takes to create the repo |
| Reaches model output | Yes | No — it runs before the first container exists |

The work key's forbidden list and the boot key's required list overlap on
exactly `administration` and `workflows`. That overlap is the argument: no
single token can do both jobs without handing model-written code the ability
to rewrite `.github/workflows/*` — the file that independently re-runs the
verify command a task is judged by. A worker that can edit CI can make
anything pass, and every gate downstream becomes decoration.

The separation is a lifetime as much as a scope, and it is enforced in code
rather than in this document:

- `security.assert_no_provision_token` refuses an environment carrying the boot
  key **by name or by value** — renaming it on the way in does not narrow what
  it can do.
- `ContainerManager.__post_init__` runs every worker environment through that
  check before a container is created, so the failure is a refusal to start.
- The boot key's value is enrolled in the redactor even though it is never
  passed, so an unforeseen route cannot put it in a log.
- `doctor` fails if the two are the same token.

Rotation is per key. The boot key can be revoked the moment a project has been
created; the work key lives as long as runs against that repository do.

Both remain scoped to a single repository. The split is not about what apiary
can reach — it is about what model-written code may touch inside the one
repository it is working in.
