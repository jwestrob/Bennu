import importlib.util
import sys
from pathlib import Path
import pytest


def _load_module(rel_path: str, name: str):
    path = Path(__file__).resolve().parents[2] / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


fsm_mod = _load_module("src/llm/rag_system/fsm/action_graph.py", "fsm_action_graph")


def test_fsm_legal_transitions():
    f = fsm_mod.FSM()
    assert f.state == fsm_mod.State.PLAN
    f.transition(fsm_mod.State.DB)
    assert f.state == fsm_mod.State.DB
    f.transition(fsm_mod.State.ACCUM)
    assert f.state == fsm_mod.State.ACCUM
    f.transition(fsm_mod.State.DECIDE)
    assert f.state == fsm_mod.State.DECIDE
    f.transition(fsm_mod.State.PLAN)
    assert f.state == fsm_mod.State.PLAN
    f.transition(fsm_mod.State.SIM)
    assert f.state == fsm_mod.State.SIM


def test_fsm_illegal_transition_raises():
    f = fsm_mod.FSM()
    with pytest.raises(ValueError):
        f.transition(fsm_mod.State.ACCUM)  # PLAN -> ACCUM not allowed

