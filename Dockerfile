# The orchestrator image: the scheduler that reconciles GitHub with containers.
#
# It holds no secrets. GITHUB_TOKEN arrives at run time from the environment
# (see compose.yaml), never as a build argument and never as an ENV default, so
# `docker history` on this image shows nothing worth stealing.
#
# It also does not contain Ollama, and must not. Docker Desktop on macOS runs a
# Linux VM with no Metal passthrough, so an in-container Ollama silently falls
# back to CPU and loses roughly an order of magnitude. Inference stays on the
# host and is reached over host.docker.internal - see the "Three constraints"
# section of docs/architecture-v2.md.

FROM python:3.12-slim AS build

# The virtualenv is assembled here and copied whole into the runtime stage, so
# pip, hatchling and the build tree never reach the shipped image.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /src
# pyproject.toml names README.md and LICENSE as metadata, so the build fails
# without them; nothing else outside src/ is build input.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .


FROM python:3.12-slim AS runtime

# git: the graph still shells out to it (worktree.py, nodes/integrator.py), and
# a missing binary there surfaces deep inside a run rather than at startup.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# docker: containers/manager.py drives the daemon by shelling out to the CLI
# rather than through a Python SDK, so that DOCKER_HOST - and therefore #28's
# socket proxy - is honoured for free. Without the client on PATH that choice
# is silently unsatisfiable: the proxy answers, and nothing can speak to it.
#
# Only the client binary, never the daemon. Pinned to a major rather than the
# rolling `cli` tag, and the API version is negotiated, so a 29.x daemon is not
# a requirement.
COPY --from=docker:29-cli /usr/local/bin/docker /usr/local/bin/docker

COPY --from=build /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Reach the host's Ollama, not a container's. compose.yaml can override the
# host part; the point of the default is that it is never "localhost".
ENV OLLAMA_HOST=http://host.docker.internal:11434

# Run artifacts - container logs above all - are written under a mounted
# directory so they outlive a container that is disposable by design. #29 owns
# what actually goes in here. SWARM_ prefix to match every other setting the
# package reads (src/swarm/config.py).
ENV SWARM_RUN_ARTIFACTS=/var/apiary/runs

# Unprivileged. This process drives the Docker API through a proxy (#28); every
# privilege it does not hold is one an LLM-driven run cannot spend.
RUN useradd --create-home --uid 10001 apiary \
 && mkdir -p /var/apiary/runs /workspace \
 && chown -R apiary:apiary /var/apiary /workspace

USER apiary
WORKDIR /workspace

# Args from `docker compose run orchestrator ...` land on the CLI's argv.
ENTRYPOINT ["swarm"]
