import json
from pathlib import Path
import importlib
import sys

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

spec = importlib.util.spec_from_file_location("db_template_mapper", root / "src/llm/rag_system/db_template_mapper.py")
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)  # type: ignore[attr-defined]
map_question_to_template = mod.map_question_to_template


def _stage_a_detect(question: str) -> bool:
    q = (question or "").lower()
    spatial_markers = ("genome", "contig", "locus", "neighborhood", "operon", "coordinates")
    return any(tok in q for tok in spatial_markers)


def test_router_regression_scaffold():
    data = json.loads((Path(__file__).parent / "router_regression_set.json").read_text())
    for case in data:
        prompt = case["prompt"]
        expect = case["expect"]
        if expect["stage"] == "A":
            assert _stage_a_detect(prompt) is True
        else:
            tpl = map_question_to_template(prompt)
            assert tpl is not None, f"expected template mapping for: {prompt}"
            name, _ = tpl
            assert name == expect["template"], f"expected template {expect['template']} got {name}"
