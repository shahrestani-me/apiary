"""Central configuration.

Two model roles:

  ORCHESTRATOR_MODEL - planning, routing, progress judgement. Short
                       schema-constrained JSON calls, a few hundred tokens a
                       round. Cheap even on a large model, so buy quality here.

  WORKER_MODEL       - the one that writes code. It emits whole files, so this
                       is where the wall-clock time goes. Buy throughput here.

The two are not "big vs small" - see the note on the model fields below.

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
    # Split by ARCHITECTURE, not by size. On Apple Silicon, generation speed is
    # roughly bandwidth / bytes-read-per-token, so a dense model reads its whole
    # file per token while an MoE reads only its active experts. Measured here:
    #
    #   gemma4:31b  dense  19 GB  30.7B active   17.4 tok/s
    #   gemma4:26b  MoE    17 GB   3.8B active   81.7 tok/s   (4.7x)
    #
    # The orchestrator emits a few hundred tokens of schema-constrained JSON per
    # round; quality there is worth more than speed, so it gets the dense model.
    # The worker emits whole files - that is where the wall-clock goes - so it
    # gets the MoE.
    #
    # 19 + 17 GB of weights exceeds the ~27 GB GPU budget, so keep
    # OLLAMA_MAX_LOADED_MODELS=1 and let Ollama swap. A swap measured 6.7 s here,
    # roughly 2 swaps per round, against minutes saved on every worker call.
    # Set both to gemma4:26b to eliminate swapping entirely if you prefer.
    orchestrator_model: str = field(default_factory=lambda: _env("SWARM_ORCHESTRATOR_MODEL", "gemma4:31b"))
    worker_model: str = field(default_factory=lambda: _env("SWARM_WORKER_MODEL", "gemma4:26b"))

    # Low temperature: we want obedient structure, not creativity.
    orchestrator_temperature: float = 0.0
    worker_temperature: float = 0.1

    # Context window requested from Ollama.
    #
    # IMPORTANT: gemma4 advertises 256K context, but the KV cache for a 31B
    # model at 256K would need more memory than the weights themselves. Never
    # request the advertised maximum. 16-32K is plenty for one focused task,
    # and curated context beats large context on every model we tested.
    # Ollama allocates a runner per (model, options) combination, so if you set
    # both roles to the SAME model, keep these two values equal - differing
    # num_ctx would spawn a second runner and double your memory use.
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
