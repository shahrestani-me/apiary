"""Central configuration.

Two model tiers, deliberately:

  ORCHESTRATOR_MODEL - small, fast, cheap. Does planning, routing, progress
                       judgement. These are short structured-output calls, so
                       a 2-4B model is genuinely fine here.

  WORKER_MODEL       - the one that writes code. Quality here dominates the
                       quality of the whole system. Use the biggest coding
                       model your RAM allows.

Override anything via environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # --- Ollama ---------------------------------------------------------
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))

    # --- Models ---------------------------------------------------------
    # Defaults tuned for Mac Studio M4 Max / 36 GB unified memory.
    #
    # SINGLE-TIER on purpose. With a ~27 GB GPU budget you cannot keep a 20 GB
    # worker and a second model resident without paging, and a model swap
    # between graph nodes costs more than the small model ever saves. One
    # model, always warm, wins.
    orchestrator_model: str = field(default_factory=lambda: _env("SWARM_ORCHESTRATOR_MODEL", "gemma4:31b"))
    worker_model: str = field(default_factory=lambda: _env("SWARM_WORKER_MODEL", "gemma4:31b"))

    # Low temperature: we want obedient structure, not creativity.
    orchestrator_temperature: float = 0.0
    worker_temperature: float = 0.1

    # Context window requested from Ollama.
    #
    # IMPORTANT: gemma4 advertises 256K context, but the KV cache for a 31B
    # model at 256K would need more memory than the weights themselves. Never
    # request the advertised maximum. 16-32K is plenty for one focused task,
    # and curated context beats large context on every model we tested.
    # Keep these EQUAL when both roles use the same model. Ollama allocates a
    # runner per (model, options) combination - different num_ctx values for
    # the same model can spawn a second runner and double your memory use.
    orchestrator_num_ctx: int = int(_env("SWARM_ORCH_CTX", "16384"))
    worker_num_ctx: int = int(_env("SWARM_WORKER_CTX", "16384"))

    # --- Repo / execution ----------------------------------------------
    repo_path: str = field(default_factory=lambda: _env("SWARM_REPO", os.getcwd()))
    worktree_root: str = field(default_factory=lambda: _env("SWARM_WORKTREES", ".swarm/worktrees"))
    verify_command: str = field(default_factory=lambda: _env("SWARM_VERIFY", "python -m pytest -q"))

    # --- Safety rails (the part that stops runaway cost / infinite loops) -
    # 2 on a 36 GB machine: Ollama allocates KV cache as
    # OLLAMA_NUM_PARALLEL x num_ctx, so raising this raises memory linearly.
    max_workers_parallel: int = int(_env("SWARM_MAX_PARALLEL", "2"))
    max_rounds: int = int(_env("SWARM_MAX_ROUNDS", "8"))
    max_stalls: int = int(_env("SWARM_MAX_STALLS", "2"))
    max_attempts_per_task: int = int(_env("SWARM_MAX_ATTEMPTS", "3"))
    worker_timeout_s: int = int(_env("SWARM_WORKER_TIMEOUT", "600"))
    verify_timeout_s: int = int(_env("SWARM_VERIFY_TIMEOUT", "300"))

    # --- Persistence ----------------------------------------------------
    checkpoint_db: str = field(default_factory=lambda: _env("SWARM_CHECKPOINTS", ".swarm/checkpoints.sqlite"))


SETTINGS = Settings()
