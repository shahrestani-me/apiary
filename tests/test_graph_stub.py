"""End-to-end test with a stubbed model and a real throwaway git repo.

Proves the machinery works - worktree isolation, fan-out, verification,
stall detection, merge - without needing Ollama running. Run this first
after install; if it passes, any remaining problem is model quality, not
plumbing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "demo"
    r.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=r, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=r, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=r, check=True)
    (r / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (r / "test_calc.py").write_text(
        "from calc import add, sub\n\n"
        "def test_add():\n    assert add(1, 2) == 3\n\n"
        "def test_sub():\n    assert sub(3, 1) == 2\n"
    )
    subprocess.run(["git", "add", "-A"], cwd=r, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=r, check=True)
    return r


def test_full_run(repo: Path, monkeypatch):
    os.environ["SWARM_REPO"] = str(repo)
    os.environ["SWARM_VERIFY"] = f"{sys.executable} -m pytest -q"
    os.environ["SWARM_MAX_ROUNDS"] = "4"

    # Reload config so the env vars above take effect.
    import importlib

    from swarm import config as config_mod

    importlib.reload(config_mod)

    from swarm.state import Plan, PlannedTask, ProgressJudgement, WorkerOutput

    # --- stub the two model call sites --------------------------------
    class StubStructured:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, _messages):
            if self.schema is Plan:
                return Plan(
                    tasks=[
                        PlannedTask(
                            id="add-sub",
                            goal="add a sub(a, b) function to calc.py",
                            files=["calc.py"],
                        )
                    ],
                    reasoning="single file, single change",
                )
            if self.schema is WorkerOutput:
                return WorkerOutput(
                    edits=[
                        {
                            "path": "calc.py",
                            "content": "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n",
                        }
                    ],
                    notes="added sub",
                )
            if self.schema is ProgressJudgement:
                return ProgressJudgement(
                    request_satisfied=True,
                    progress_being_made=True,
                    in_loop=False,
                    reason="stub",
                )
            raise AssertionError(f"unexpected schema {self.schema}")

    import swarm.llm as llm_mod
    import swarm.nodes.judge as judge_mod
    import swarm.nodes.planner as planner_mod
    import swarm.nodes.worker as worker_mod

    for mod in (llm_mod, judge_mod, planner_mod, worker_mod):
        monkeypatch.setattr(mod, "structured", lambda _llm, schema: StubStructured(schema), raising=False)
    for mod in (llm_mod, judge_mod, planner_mod):
        monkeypatch.setattr(mod, "orchestrator_llm", lambda: None, raising=False)
    monkeypatch.setattr(worker_mod, "worker_llm", lambda: None, raising=False)

    # config module was reloaded, so re-point SETTINGS in every consumer
    for mod_name in ("swarm.worktree", "swarm.nodes.worker", "swarm.nodes.verifier",
                     "swarm.nodes.integrator", "swarm.graph", "swarm.llm"):
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "SETTINGS"):
            monkeypatch.setattr(mod, "SETTINGS", config_mod.SETTINGS, raising=False)

    from swarm.graph import build_graph

    graph = build_graph()
    final = graph.invoke({"objective": "add sub() to calc.py"},
                         config={"recursion_limit": 60})

    print("\n".join(final.get("events", [])))

    assert final["tasks"]["add-sub"]["status"] == "verified"
    assert final["merged_branches"] == ["swarm/add-sub"]
    assert "def sub" in (repo / "calc.py").read_text()
