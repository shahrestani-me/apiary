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
`codeload.github.com`, `host.docker.internal`, plus any subdomain of those.
tinyproxy runs `FilterDefaultDeny Yes` against regexes generated by
`EgressPolicy.filter_lines()` and pasted into `compose.yaml`; the suite fails if
the two drift, so a widened allowlist cannot land in the YAML alone.

The regexes are anchored at both ends. An unanchored `github\.com` also matches
`github.com.attacker.net`, and a proxy allowlist is the wrong place to learn
that.

### The knob, and its cost

`APIARY_EGRESS_ALLOW=pypi.org,files.pythonhosted.org`

The honest default breaks something: `Dockerfile.worker` deliberately bakes in no
toolchain, so a `## Verify` command that runs `pip install -e .` needs a package
index. It is an environment variable rather than a constant because **a registry
that accepts uploads is an exfiltration channel** — a token fits in a package
name — so it is a decision an operator makes per target repository, out loud,
rather than a default nobody reads.

### What this does not buy

`github.com` is one host and every repository lives behind it. The egress filter
cannot tell the target repository from an unrelated one; only the token can. Both
layers, or neither.

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

leaks = scan_artifacts(Path(".swarm/runs"), env=os.environ)
```

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


## The boot key

Creating a repository and doing the work inside it are two jobs with
incompatible permissions, so apiary uses two credentials.

| | Work key | Boot key |
|---|---|---|
| Variable | `GITHUB_TOKEN` | `APIARY_PROVISION_TOKEN` |
| Permissions | contents:write, pull_requests:write, issues:write, metadata:read | administration:write, contents:write, workflows:write, issues:write, metadata:read |
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
